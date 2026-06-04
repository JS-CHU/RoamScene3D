import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from kornia.morphology import erosion, dilation
import os
from .inpainter import Inpainter
from .lama_inpainter import LamaInpainter
from .SDFT_inpainter import SDFTInpainter

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import img_coord_to_sample_coord,\
    direction_to_img_coord, img_coord_to_pano_direction, direction_to_pers_img_coord

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils.functions import get_cubemap_views_world_to_cam
from modules.equilib import cube2equi, equi2cube

class _PanoPersFusionInpainter(Inpainter):
    def __init__(self, save_path, subset_name=None):
        super().__init__()

        self.diff_inpainter = SDFTInpainter()
            
        self.lama_inpainter = LamaInpainter()
        # self.lama_inpainter = None
        
        self.save_path = save_path

    @torch.no_grad()
    def inpaint(self, idx, img, mask, poses, label=''):
        img = img.squeeze().permute(2, 0, 1)
        mask = mask.squeeze()[None]
        inpainted_img = img.clone()

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=512, ratio=1.4)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        _, pano_height, pano_width = img.shape
        pano_img_coords = torch.meshgrid(torch.linspace(.5 / pano_height, 1. - .5 / pano_height, pano_height),
                                         torch.linspace(.5 / pano_width,  1. - .5 / pano_width, pano_width),
                                         indexing='ij')
        pano_img_coords = torch.stack(list(pano_img_coords), dim=-1)

        pano_dirs = img_coord_to_pano_direction(pano_img_coords)

        for i in range(n_pers):
            cur_sample_coords = sample_coords[i]
            '''这里筛选一下主视角'''
            pers_image = F.grid_sample(inpainted_img[None], cur_sample_coords[None], padding_mode='border')[0]
            # save
            plt.imsave(f'./output/debug/pano2room/pers_{str(i)}.png', pers_image.permute(1, 2, 0).cpu().numpy())
            pers_mask = F.grid_sample(mask[None, :, :], cur_sample_coords[None], padding_mode='border')[0]
            pers_mask = (pers_mask > 0.5).float() #CHW
            if self.lama_inpainter is not None:
                kernel = torch.from_numpy(cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))).float().to(pers_mask.device)
                smooth_mask = pers_mask
                smooth_mask = erosion(pers_mask[None], kernel=kernel)[0]
                smooth_mask = dilation(smooth_mask[None], kernel=kernel)[0]
                smooth_mask = torch.minimum(smooth_mask, pers_mask)
                plt.imsave(f'./output/debug/pano2room/pers_mask_{str(i)}.png', pers_mask.squeeze(0).cpu().numpy(), cmap="gray")
                plt.imsave(f'./output/debug/pano2room/smooth_mask_{str(i)}.png', smooth_mask.squeeze(0).cpu().numpy(), cmap="gray")
                lama_inpainted = self.lama_inpainter.inpaint(pers_image[None], pers_mask[None])[0]
                record_lama = lama_inpainted * (1 - pers_mask) + lama_inpainted * pers_mask
                # save
                plt.imsave(f'./output/debug/pano2room/lama_{str(i)}.png', record_lama.permute(1, 2, 0).cpu().numpy())

                if smooth_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(lama_inpainted[None], smooth_mask[None], label)[0]
                else:
                    cur_inpainted = lama_inpainted
            else:
                if pers_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(pers_image[None], pers_mask[None], label)[0]
                else:
                    cur_inpainted = pers_image

            cur_inpainted = pers_image * (1 - pers_mask) + cur_inpainted * pers_mask
            # save
            plt.imsave(f'./output/debug/pano2room/cur_{str(i)}.png', cur_inpainted.permute(1, 2, 0).cpu().numpy())

            proj_coord, proj_mask = direction_to_pers_img_coord(pano_dirs, to_vecs[i], down_vecs[i], right_vecs[i])
            proj_coord = img_coord_to_sample_coord(proj_coord)

            cur_inpainted_pano_img = F.grid_sample(cur_inpainted[None], proj_coord[None], padding_mode='border')[0]
            proj_mask = proj_mask.permute(2, 0, 1).float()
            inpainted_img = inpainted_img * (1. - proj_mask) + cur_inpainted_pano_img * proj_mask
            plt.imsave(f'./output/debug/pano2room/pano_{str(i)}.png', inpainted_img.permute(1, 2, 0).cpu().numpy())
            mask = mask * (1. - proj_mask) + 0. * proj_mask

        inpainted_img = img * mask + inpainted_img * (1 - mask)
        # save
        plt.imsave(f'./output/debug/pano2room/pano.png', inpainted_img.permute(1, 2, 0).cpu().numpy())

        return inpainted_img.permute(1, 2, 0)
    
class PanoPersFusionInpainter(Inpainter):
    def __init__(self, save_path, subset_name=None):
        super().__init__()

        self.diff_inpainter = SDFTInpainter(model='withcam', SDFT_path='SDFT_weights/cam6_lora12_distill_fixloss', step=1000)
            
        self.lama_inpainter = LamaInpainter()
        # self.lama_inpainter = None
        
        self.save_path = save_path

    @torch.no_grad()
    def inpaint_s(self, idx, img, mask, inpaint_poses, label=''):
        H, W, _ = img.shape
        # pano = img.cpu().numpy()
        # save_version = 'FT_withcam_4170'
        # os.makedirs(f'./output/debug/{save_version}', exist_ok=True)
        # plt.imsave(f'./output/debug/{save_version}/pano.png', img.cpu().numpy())
        img = img.squeeze().permute(2, 0, 1)
        mask = mask.squeeze()[None]
        inpainted_img = img.clone()

        # cubemap_poses = get_cubemap_views_world_to_cam()
        # poses = [cp[:3].cpu().numpy() for cp in cubemap_poses]
        # fov = 90

        pers_images = equi2cube(img, {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, 512, 'list')
        pers_masks = equi2cube(mask, {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, 512, 'list')
        lama_inpainted_list = []
        diff_inpainted_list = []
        diff_all_inpainted_list = []
        print('加载透视图……')
        for i in range(6):
            pers_image = pers_images[i]
            # pose = pose.view(-1).unsqueeze(0)
            # save
            # plt.imsave(f'./output/debug/{save_version}/pers_{str(i)}.png', pers_image.permute(1, 2, 0).cpu().numpy())
            pers_mask = pers_masks[i]
            pers_mask = (pers_mask > 0.5).float() #CHW
            if self.lama_inpainter is not None:
                kernel = torch.from_numpy(cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))).float().to(pers_mask.device)
                smooth_mask = pers_mask
                smooth_mask = erosion(pers_mask[None], kernel=kernel)[0]
                smooth_mask = dilation(smooth_mask[None], kernel=kernel)[0]
                smooth_mask = torch.minimum(smooth_mask, pers_mask)
                # plt.imsave(f'./output/debug/{save_version}/pers_mask_{str(i)}.png', pers_mask.squeeze().cpu().numpy(), cmap="gray")
                # plt.imsave(f'./output/debug/{save_version}/smooth_mask_{str(i)}.png', smooth_mask.squeeze().cpu().numpy(), cmap="gray")
                lama_inpainted = self.lama_inpainter.inpaint(pers_image[None], pers_mask[None])[0]
                # record_lama = lama_inpainted * (1 - pers_mask) + lama_inpainted * pers_mask
                lama_inpainted_list.append(lama_inpainted)
                # save
                # plt.imsave(f'./output/debug/{save_version}/lama_{str(i)}.png', lama_inpainted.permute(1, 2, 0).cpu().numpy())

                if smooth_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(lama_inpainted[None], smooth_mask[None], inpaint_poses[i].view(1, -1), label)[0]
                else:
                    cur_inpainted = lama_inpainted
            else:
                if pers_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(pers_image[None], pers_mask[None], inpaint_poses[i].view(1, -1), label)[0]
                else:
                    cur_inpainted = pers_image
            diff_cur_inpainted = cur_inpainted.clone()
            cur_inpainted = pers_image * (1 - pers_mask) + cur_inpainted * pers_mask
            # save
            # plt.imsave(f'./output/debug/{save_version}/diff_{str(i)}.png', diff_cur_inpainted.permute(1, 2, 0).cpu().numpy())
            # plt.imsave(f'./output/debug/{save_version}/diff_masked_{str(i)}.png', cur_inpainted.permute(1, 2, 0).cpu().numpy())
            diff_inpainted_list.append(cur_inpainted)
            diff_all_inpainted_list.append(diff_cur_inpainted)

        if self.lama_inpainter is not None:
            lama_inpainted_img = cube2equi(lama_inpainted_list, 'list', H, W)
            lama_inpainted_img = img * (1 - mask) + lama_inpainted_img * mask
            # plt.imsave(f'./output/debug/{save_version}/lama_pano.png', lama_inpainted_img.permute(1, 2, 0).cpu().numpy())
        inpainted_img = cube2equi(diff_inpainted_list, 'list', H, W)
        # diff_all_inpainted = cube2equi(diff_all_inpainted_list, 'list', H, W)
        inpainted_img = img * (1 - mask) + inpainted_img * mask
        # save
        # plt.imsave(f'./output/debug/{save_version}/diff_pano.png', diff_all_inpainted.permute(1, 2, 0).cpu().numpy())
        # plt.imsave(f'./output/debug/{save_version}/diff_masked_pano.png', inpainted_img.permute(1, 2, 0).cpu().numpy())
        # exit()

        return inpainted_img.permute(1, 2, 0)

    @torch.no_grad()
    def inpaint(self, idx, img, mask, inpaint_poses, label=''):
        H, W, _ = img.shape
        img = img.squeeze().permute(2, 0, 1)
        mask = mask.squeeze()[None]
        inpainted_img = img.clone()

        pers_images = equi2cube(img, {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, 512, 'list')
        pers_masks = equi2cube(mask, {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, 512, 'list')
        lama_inpainted_list = []
        diff_inpainted_list = []
        diff_all_inpainted_list = []
        print('Inpainting')
        pers_images_b = torch.stack(pers_images, dim=0)  # [6, 3, 512, 512]
        # 阈值化并保持单通道掩膜
        pers_masks_b = torch.stack([(m > 0.5).float() for m in pers_masks], dim=0)  # [6, 1, 512, 512]

        if self.lama_inpainter is not None:
            kernel = torch.from_numpy(cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))).float().to(pers_masks_b.device)
            # 批量形态学滤波
            smooth_masks_b = erosion(pers_masks_b, kernel=kernel)
            smooth_masks_b = dilation(smooth_masks_b, kernel=kernel)
            smooth_masks_b = torch.minimum(smooth_masks_b, pers_masks_b)

            # 批量 LAMA 修复
            lama_inpainted_b = self.lama_inpainter.inpaint(pers_images_b, pers_masks_b)
            lama_inpainted_list = list(torch.unbind(lama_inpainted_b, dim=0))

            need_diff = (smooth_masks_b.flatten(1).max(dim=1).values > 0.5)
            cur_inpainted_b = lama_inpainted_b.clone()
            idxs = torch.nonzero(need_diff, as_tuple=False).squeeze(-1)
            if idxs.numel() > 0:
                batch_imgs = lama_inpainted_b.index_select(0, idxs)
                batch_masks = smooth_masks_b.index_select(0, idxs)
                if inpaint_poses is not None:
                    batch_poses = torch.stack([inpaint_poses[i].view(-1) for i in idxs.tolist()], dim=0)
                else:
                    batch_poses = None
                batch_out = self.diff_inpainter.inpaint_batch(batch_imgs, batch_masks, batch_poses, label)
                cur_inpainted_b.index_copy_(0, idxs, batch_out)
        else:
            # 不使用 LAMA 时，仅对有掩膜的面调用扩散
            need_diff = (pers_masks_b.flatten(1).max(dim=1).values > 0.5)
            cur_inpainted_b = pers_images_b.clone()
            idxs = torch.nonzero(need_diff, as_tuple=False).squeeze(-1)
            if idxs.numel() > 0:
                batch_imgs = pers_images_b.index_select(0, idxs)
                batch_masks = pers_masks_b.index_select(0, idxs)
                if inpaint_poses is not None:
                    batch_poses = torch.stack([inpaint_poses[i].view(-1) for i in idxs.tolist()], dim=0)
                else:
                    batch_poses = None
                batch_out = self.diff_inpainter.inpaint_batch(batch_imgs, batch_masks, batch_poses, label)
                cur_inpainted_b.index_copy_(0, idxs, batch_out)

        diff_cur_inpainted_b = cur_inpainted_b.clone()
        # 批量掩膜融合
        diff_inpainted_b = pers_images_b * (1 - pers_masks_b) + cur_inpainted_b * pers_masks_b

        # 保持与 cube2equi 接口一致的 list 格式
        diff_inpainted_list = list(torch.unbind(diff_inpainted_b, dim=0))
        # diff_all_inpainted_list = list(torch.unbind(diff_cur_inpainted_b, dim=0))

        if self.lama_inpainter is not None:
            lama_inpainted_img = cube2equi(lama_inpainted_list, 'list', H, W)
            lama_inpainted_img = img * (1 - mask) + lama_inpainted_img * mask
            # plt.imsave(f'./output/debug/{save_version}/lama_pano.png', lama_inpainted_img.permute(1, 2, 0).cpu().numpy())
        inpainted_img = cube2equi(diff_inpainted_list, 'list', H, W)
        # diff_all_inpainted = cube2equi(diff_all_inpainted_list, 'list', H, W)
        inpainted_img = img * (1 - mask) + inpainted_img * mask

        return inpainted_img.permute(1, 2, 0)
