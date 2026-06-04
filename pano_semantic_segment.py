import os
import os.path as ops
import argparse
import cv2
# from pano_segment.local_utils.log_util import init_logger
from pano_segment.local_utils.config_utils import parse_config_utils
from pano_segment.models import build_sam_clip_text_ins_segmentor

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()


class PanoSemanticSegmentor(object):
    
    def __init__(self, image_path, scene_graph):
        self.image_path = image_path
        self.scene_graph = scene_graph
        self.major_list = None
        self.object_list = self._extract_objects_names()
        self.use_text_prefix = True
        insseg_cfg_path = './pano_segment/config/insseg.yaml'
        self.insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
        self.debug = True

    def _extract_objects_names(self):
            """提取场景图中的对象名称列表"""
            if self.scene_graph is not None:
                objects: dict = self.scene_graph['objects']
                self.major_list = self.scene_graph['major']
                # filtered_objects = {k: v for k, v in objects.items() if 'major' not in k.lower()}
                return list(objects.keys())
            else:
                return None
    
    def segment(self, save_dir):

        input_image_name = ops.split(self.image_path)[1]

        if self.major_list is not None:
            unique_labels = self.object_list
        else:
            unique_labels = None
        # unique_labels = None
        segmentor = build_sam_clip_text_ins_segmentor(cfg=self.insseg_cfg)

        # 让返回的结果中包含类别信息
        ret = segmentor.seg_image(self.image_path, unique_label=unique_labels, use_text_prefix=self.use_text_prefix)
        semantic_map = ret['masks']['map']
        instance_num = len(ret['masks']['bbox_cls_names'])
        # semantic_map = None
        # instance_num = 0
        if self.debug:
            # save cluster result
            # save_dir = './output/debug/pano_seg'
            os.makedirs(save_dir, exist_ok=True)
            ori_image_save_path = ops.join(save_dir, input_image_name)
            cv2.imwrite(ori_image_save_path, ret['source'])
            mask_save_path = ops.join(save_dir, '{:s}_insseg_mask.png'.format(input_image_name.split('.')[0]))
            cv2.imwrite(mask_save_path, ret['ins_seg_mask'])
            mask_add_save_path = ops.join(save_dir, '{:s}_insseg_add.png'.format(input_image_name.split('.')[0]))
            cv2.imwrite(mask_add_save_path, ret['ins_seg_add'])

        return semantic_map, instance_num


if __name__ == "__main__":
    # 创建输出目录
    args = init_args()
    segmentor = PanoSemanticSegmentor()
