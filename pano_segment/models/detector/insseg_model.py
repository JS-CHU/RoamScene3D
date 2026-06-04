#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-11 下午1:49
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : insseg_model.py
# @IDE: PyCharm Community Edition
"""
instance segmentation model with sam and clip
"""
import numpy as np
import cv2
from PIL import Image
import torch

from pano_segment.models.detector import utils
from pano_segment.models.clip import tokenize
from pano_segment.models import build_clip_model
from pano_segment.models import build_sam_mask_generator


class SamClipInsSegmentor(object):
    """

    """
    def __init__(self, sam_cfg, clip_cfg, insseg_cfg):
        """

        :param sam_cfg:
        :param clip_cfg:
        :param insseg_cfg:
        """
        self.mask_generator = build_sam_mask_generator(sam_cfg)
        self.clip_model, self.clip_preprocess = build_clip_model(clip_cfg)
        self.device = torch.device(insseg_cfg.MODEL.DEVICE)
        self.top_k_objs = insseg_cfg.INS_SEG.TOP_K_MASK_COUNT
        self.cls_score_thresh = insseg_cfg.INS_SEG.CLS_SCORE_THRESH
        self.max_input_size = insseg_cfg.INS_SEG.MAX_INPUT_SIZE
        self.obj365_text_prompts = utils.generate_object365_text_prompts()
        self.obj365_text_token = tokenize(self.obj365_text_prompts).to(self.device)
        self.text_token = None

    def _set_text_tokens(self, texts):
        """

        :param texts:
        :return:
        """
        self.text_token = tokenize(texts=texts).to(self.device)
        return

    def _generate_sam_mask(self, input_image: np.ndarray):
        """

        :param input_image:
        :return:
        """
        masks = self.mask_generator.generate(input_image)
        most_stable_mask = sorted(masks, key=lambda d: d['area'], reverse=True)
        if len(most_stable_mask) > self.top_k_objs:
            most_stable_mask = most_stable_mask[:self.top_k_objs]
        sam_masks = {
            'segmentations': [tmp['segmentation'] for tmp in most_stable_mask],
            'bboxes': [tmp['bbox'] for tmp in most_stable_mask],
            'stability_scores': [tmp['stability_score'] for tmp in most_stable_mask],
        }
        return sam_masks

    @staticmethod
    def _crop_rotate_image_roi(input_image, seg_mask):
        """

        :param input_image:
        :param seg_mask:
        :return:
        """
        y, x = np.where(seg_mask == 1)
        fg_pts = np.vstack((x, y)).transpose()
        src_image = cv2.bitwise_or(input_image, input_image, mask=np.asarray(seg_mask, dtype=np.uint8))
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(fg_pts)
        extend_size = 20
        if roi_x - extend_size >= 0:
            roi_x -= extend_size
            roi_w += 2 * extend_size
        if roi_y - extend_size >= 0:
            roi_y -= extend_size
            roi_h += 2 * extend_size
        roi_image = src_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]
        if np.any(np.shape(roi_image) < (3, 3)):
            return None
        return roi_image

    def _classify_image(self, input_image: np.ndarray, text=None):
        """

        :param input_image:
        :return:
        """
        image = Image.fromarray(input_image)
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        if text is None:
            logits_per_image, logits_per_text = self.clip_model(image, self.obj365_text_token)
        else:
            if self.text_token is None:
                text_token = tokenize(texts=text).to(self.device)
                logits_per_image, logits_per_text = self.clip_model(image, text_token)
            else:
                logits_per_image, logits_per_text = self.clip_model(image, self.text_token)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0, :]
        cls_id = np.argmax(probs)
        score = probs[cls_id]
        if text is None:
            if score < 0.15:
                cls_id = probs.shape[0] - 1
            return cls_id
        else:
            if score < self.cls_score_thresh:
                cls_id = probs.shape[0] - 1
            return cls_id

    def _classify_mask(self, input_image, mask, text=None):
        """

        :param input_image:
        :param mask:
        :return:
        """
        bboxes_cls_names = []
        for idx, bbox in enumerate(mask['bboxes']):
            roi_image = self._crop_rotate_image_roi(input_image, mask['segmentations'][idx])
            if roi_image is None:
                cls_name = 'background'
                bboxes_cls_names.append(cls_name)
                continue
            # cv2.imwrite('{:d}_mask.png'.format(idx), roi_image[:, :, (2, 1, 0)])
            cls_id = self._classify_image(roi_image, text=text)
            if text is None:
                cls_name = self.obj365_text_prompts[cls_id].split('a photo of')[1].strip(' ')
                # bboxes_cls_names.append(cls_name)
            else:
                cls_name = text[cls_id]
                if cls_name.startswith('a photo of'):
                    cls_name = cls_name.split(' ')[3]
            
            bboxes_cls_names.append(cls_name)

        mask['bbox_cls_names'] = bboxes_cls_names
        return

    def process_mask(self, masks):
        """
        过滤掉 masks 中 'bbox_cls_names' 列表中为 'background' 的元素，
        并同时保留其他三个列表中对应索引的元素。
        
        Args:
            masks (dict): 包含以下键的字典:
                - 'segmentations': 分割掩码列表
                - 'bboxes': 边界框列表  
                - 'stability_scores': 稳定性分数列表
                - 'bbox_cls_names': 类别名称列表
                
        Returns:
            dict: 过滤后的 masks 字典，保持相同的结构
        """
        if not isinstance(masks, dict):
            return masks
            
        # 检查必要的键是否存在
        required_keys = ['segmentations', 'bboxes', 'stability_scores', 'bbox_cls_names']
        if not all(key in masks for key in required_keys):
            return masks
            
        # 获取各个列表
        segmentations = masks['segmentations']
        bboxes = masks['bboxes']
        stability_scores = masks['stability_scores']
        bbox_cls_names = masks['bbox_cls_names']
        
        # 检查所有列表长度是否一致
        if not all(len(lst) == len(bbox_cls_names) for lst in [segmentations, bboxes, stability_scores]):
            return masks
            
        # 使用单次遍历优化性能，避免多次循环
        filtered_segmentations = []
        filtered_bboxes = []
        filtered_stability_scores = []
        filtered_bbox_cls_names = []
        
        def calculate_bbox_overlap(bbox1, bbox2):
            """计算两个bbox的重叠度 (IoU)
            bbox格式: [x, y, width, height]
            """
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # 计算交集区域
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area# + bbox2_area - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        
        def merge_masks(mask1, mask2):
            """合并两个分割掩码"""
            return mask1 | mask2
        
        def merge_bboxes(bbox1, bbox2):
            """合并两个bbox，返回包含两者的最小边界框"""
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            
            return [x_min, y_min, x_max - x_min, y_max - y_min]

        for i, cls_name in enumerate(bbox_cls_names):
            if cls_name != 'background':
                # 查找已存在的相同类别的索引
                same_class_indices = [j for j, existing_cls in enumerate(filtered_bbox_cls_names) 
                                    if existing_cls == cls_name]
                
                merged = False
                for existing_idx in same_class_indices:
                    overlap = calculate_bbox_overlap(bboxes[i], filtered_bboxes[existing_idx])
                    if overlap > 0.6:  # 重叠度大于60%
                        # 合并到现有元素
                        filtered_segmentations[existing_idx] = merge_masks(
                            filtered_segmentations[existing_idx], segmentations[i])
                        filtered_bboxes[existing_idx] = merge_bboxes(
                            filtered_bboxes[existing_idx], bboxes[i])
                        # 取较高的稳定性分数
                        filtered_stability_scores[existing_idx] = max(
                            filtered_stability_scores[existing_idx], stability_scores[i])
                        merged = True
                        break
                
                # 如果没有合并，则作为新元素添加
                if not merged:
                    # 为每个分割区域分配唯一的标识符
                    filtered_segmentations.append(segmentations[i])
                    filtered_bboxes.append(bboxes[i])
                    filtered_stability_scores.append(stability_scores[i])
                    filtered_bbox_cls_names.append(cls_name)

        # 当全部为 background 导致 filtered_* 为空时，返回全 False 的 map，并提供占位的全 False 掩码以保证下游尺寸正常
        if len(filtered_segmentations) == 0:
            if len(segmentations) > 0:
                base_mask = segmentations[0]
                mask_map = np.zeros_like(base_mask, dtype=np.int32)
                zero_mask = np.zeros_like(base_mask, dtype=base_mask.dtype)
                filtered_masks = {
                    'segmentations': [zero_mask],       # 占位掩码，保证后续可视化拿到尺寸
                    'bboxes': [],
                    'stability_scores': [],
                    'bbox_cls_names': [],
                    'map': mask_map                      # 全 False（全 0）
                }
            else:
                # 极端情况：原始 segmentations 也为空
                filtered_masks = {
                    'segmentations': [],
                    'bboxes': [],
                    'stability_scores': [],
                    'bbox_cls_names': [],
                    'map': np.zeros((0, 0), dtype=np.int32)
                }
            return filtered_masks

        mask_map = np.zeros_like(segmentations[0], dtype=np.int32)
        for i in range(len(filtered_segmentations)):
            unique_mask = filtered_segmentations[i].copy()
            unique_mask = np.where(unique_mask, i + 1, 0)
            mask_map += unique_mask
        # print(mask_map.sum())
        filtered_masks = {
            'segmentations': filtered_segmentations,
            'bboxes': filtered_bboxes,
            'stability_scores': filtered_stability_scores,
            'bbox_cls_names': filtered_bbox_cls_names,
            'map': mask_map
        }
        return filtered_masks
    
    def seg_image(self, input_image_path, unique_label=None, use_text_prefix=False):
        """

        :param input_image_path:
        :param unique_label:
        :param use_text_prefix:
        :return:
        """
        # read input image
        input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        # if input_image.shape[0] > self.max_input_size[0] or input_image.shape[1] > self.max_input_size[1]:
        #     h, w, _ = input_image.shape
        #     hw_ratio = h / w if h > w else w / h
        #     if h > w:
        #         dsize = (int(self.max_input_size[1] / hw_ratio), self.max_input_size[1])
        #     else:
        #         dsize = (self.max_input_size[0], int(self.max_input_size[0] / hw_ratio))
        #     input_image = cv2.resize(input_image, dsize=dsize)
        input_image = cv2.resize(input_image, (2048, 1024))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # extract mask from sam model
            masks = self._generate_sam_mask(input_image)
            # classify each mask's label
            if unique_label is None:
                self._classify_mask(input_image, masks, text=None)
            else:
                texts = utils.generate_text_prompts_for_instance_seg(
                    unique_labels=unique_label,
                    use_text_prefix=use_text_prefix
                )
                self._set_text_tokens(texts)
                self._classify_mask(input_image, masks, text=texts)
        masks = self.process_mask(masks)
        # visualize segmentation result
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        ins_seg_mask = utils.visualize_instance_seg_results(masks, draw_bbox=True)
        ins_seg_add = cv2.addWeighted(input_image, 0.5, ins_seg_mask, 0.5, 0.0)

        ret = {
            'source': input_image,
            'ins_seg_mask': ins_seg_mask,
            'ins_seg_add': ins_seg_add,
            'masks': masks
        }

        return ret
