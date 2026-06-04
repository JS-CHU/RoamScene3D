import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from .geo_predictor import GeoPredictor
from .omnidata.omnidata_predictor import OmnidataPredictor
from .omnidata.omnidata_normal_predictor import OmnidataNormalPredictor

from modules.geo_predictors.networks import VanillaMLP
import tinycudann as tcnn

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *

from PIL import Image, ImageDraw

def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


class SphereDistanceField(nn.Module):
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=2048):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)

        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoJointPredictor_new(GeoPredictor):
    def __init__(self, save_path=None):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        self.save_path = save_path

    def grads_to_normal(self, grads):
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals

    def __call__(self, key, img, ref_distance, mask, gen_res=512, 
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        
        height, width, _ = img.shape
        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)                                            
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)                      
        ref_distance = ref_distance.clone().squeeze()[..., None].float().permute(2, 0, 1)     
        ref_distance_mask = torch.cat([ref_distance, mask], 0)

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        for ratio in [1.4]:
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio)
            pers_dirs.append(cur_pers_dirs)
            pers_ratios.append(cur_pers_ratios)
            to_vecs.append(cur_to_vecs)
            down_vecs.append(cur_down_vecs)
            right_vecs.append(cur_right_vecs)

        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)

        # fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        # fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        # cx = torch.ones_like(fx) * gen_res * .5
        # cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border')
        pred_distances_raw = []
        pred_normals_raw = []

        # pers_distances = F.grid_sample(ref_distance[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') 
        # pers_masks = F.grid_sample(mask[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') 

        # ===== 加速：改为批量推理（带微批分块避免 OOM） =====
        with torch.no_grad():
            chunk = min(12, n_pers)  # 微批大小，可按显存调整
            pred_depth_chunks = []
            pred_normals_chunks = []
            for s in range(0, n_pers, chunk):
                imgs_chunk = pers_imgs[s:s+chunk]  # [B, 3, 512, 512]
                pd = self.depth_predictor.predict_depth(imgs_chunk).clip(0., None)  # [B, 1, 512, 512]
                pred_depth_chunks.append(pd)
                pn = self.normal_predictor.predict_normal(imgs_chunk)  # [B, 3, 512, 512]
                pred_normals_chunks.append(pn)

            pred_depth = torch.cat(pred_depth_chunks, dim=0)  # [n_pers, 1, 512, 512]
            pred_normals = torch.cat(pred_normals_chunks, dim=0)  # [n_pers, 3, 512, 512]

            # 每张归一化
            pred_depth = pred_depth / (pred_depth.mean(dim=(2, 3), keepdim=True) + 1e-5)
            # 畸变比例按批处理
            pred_distances_raw = pred_depth * pers_ratios.permute(0, 3, 1, 2)  # [n_pers, 1, 512, 512]

            # 法线归一化并批量旋转到世界系
            pred_normals = pred_normals * 2. - 1.
            pred_normals = pred_normals / torch.linalg.norm(pred_normals, ord=2, dim=1, keepdim=True)
            pred_normals = pred_normals.permute(0, 2, 3, 1)
            pred_normals = torch.einsum('bij,bhwj->bhwi', rot_c2w, pred_normals)
            pred_normals_raw = pred_normals.permute(0, 3, 1, 2)
        # ===== 以上替代原来的逐视角循环 =====

        # 这里不再需要 cat 列表
        # pred_distances_raw = torch.cat(pred_distances_raw, dim=0)  
        # pred_normals_raw = torch.cat(pred_normals_raw, dim=0)     
        pers_dirs = pers_dirs.permute(0, 3, 1, 2)

        sup_infos = torch.cat([pers_dirs, pred_distances_raw, pred_normals_raw], dim=1)

        scale_params = torch.zeros([n_pers], requires_grad=True)
        bias_params_global = torch.zeros([n_pers], requires_grad=True)
        bias_params_local_distance  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True)
        bias_params_local_normal  = torch.zeros([n_pers, 3, 128, 128], requires_grad=True)

        # Optimize global parameters
        sp_dis_field = SphereDistanceField()
        all_iter_steps = 2000 
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        optimizer_sp = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params, bias_params_global], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance, bias_params_local_normal], lr=init_lr_local)

        # 加速：启用 cudnn 基准、对齐内存、AMP 梯度缩放
        torch.backends.cudnn.benchmark = True
        sup_infos = sup_infos.contiguous()
        scaler = torch.cuda.amp.GradScaler()

        for phase in ['global', 'hybrid']:
            for iter_step in range(all_iter_steps):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                # 重要：半精度前向与设备一致的随机采样（移除未使用的 idx）
                with torch.cuda.amp.autocast(enabled=True):
                    sample_coords = torch.rand(n_pers, local_batch_size, 1, 2, device=img.device) * 2. - 1
                    cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')
                    distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')
                    distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                    normal_bias = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border')
                    normal_bias = normal_bias[:, :, :, 0].permute(0, 2, 1)

                    dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)
                    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                    ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)
                    scale_params_sp = F.softplus(scale_params)
                    ref_pred_distances = ref_pred_distances * scale_params_sp[:, None, None]
                    ref_pred_distances = ref_pred_distances + distance_bias

                    ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
                    ref_normals = ref_normals + normal_bias
                    ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                    pred_distances, pred_grads = sp_dis_field(dirs.reshape(-1, 3), requires_grad=True)
                    pred_distances = pred_distances.reshape(n_pers, local_batch_size, 1)
                    pred_grads = pred_grads.reshape(n_pers, local_batch_size, 3)

                    distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                    ortho_a = torch.randn([n_pers, local_batch_size, 3], device=img.device)
                    ortho_b = torch.linalg.cross(dirs, ortho_a)
                    ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
                    ortho_a = torch.linalg.cross(ortho_b, dirs)
                    ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

                    val_a = (pred_grads * ortho_a).sum(-1, True) * dirs + ortho_a
                    val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
                    val_b = (pred_grads * ortho_b).sum(-1, True) * dirs + ortho_b
                    val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)
                    error_a = (val_a * ref_normals).sum(-1, True)
                    error_b = (val_b * ref_normals).sum(-1, True)
                    errors = torch.cat([error_a, error_b], -1)
                    normal_loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=5e-1, reduction='mean')

                    reg_loss = ((scale_params_sp.mean() - 1.)**2).mean()

                    if phase == 'hybrid':
                        distance_bias_local = bias_params_local_distance
                        distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                                F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
                        normal_bias_local = bias_params_local_normal
                        normal_bias_tv_loss = F.smooth_l1_loss(normal_bias_local[:, :, 1:, :], normal_bias_local[:, :, :-1, :], beta=1e-2) + \
                                              F.smooth_l1_loss(normal_bias_local[:, :, :, 1:], normal_bias_local[:, :, :, :-1], beta=1e-2)
                    else:
                        distance_bias_tv_loss = 0.
                        normal_bias_tv_loss = 0.

                    pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
                    pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)
                    # 参考掩码采样不参与反传，减少图构建
                    with torch.no_grad():
                        sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')
                    sampled_ref_distance = sampled_ref_distance_mask[0, 0]
                    sampled_ref_mask = sampled_ref_distance_mask[0, 1]
                    ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                    ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()

                    loss = ref_distance_loss * 20. * progress + \
                           distance_loss + reg_loss * reg_loss_weight + \
                           normal_loss * normal_loss_weight + \
                           distance_bias_tv_loss * 1. + \
                           normal_bias_tv_loss * normal_tv_loss_weight

                # 更快的梯度清零
                optimizer_global.zero_grad(set_to_none=True)
                optimizer_sp.zero_grad(set_to_none=True)
                if phase == 'hybrid':
                    optimizer_local.zero_grad(set_to_none=True)

                # AMP 梯度缩放与多优化器 step
                scaler.scale(loss).backward()
                scaler.step(optimizer_global)
                scaler.step(optimizer_sp)
                if phase == 'hybrid':
                    scaler.step(optimizer_local)
                scaler.update()

        # Get new distance map and normal map
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        new_distances, new_grads = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3))

        return new_distances, new_normals

class PanoJointPredictor_super(GeoPredictor):
    def __init__(self, D=4, save_path=None):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        self.save_path = save_path
        self.D = D
        exps = []
        for p in range(D + 1):
            for q in range(D + 1 - p):
                for r in range(D + 1 - p - q):
                    exps.append((p, q, r))
        self.exps = exps
        self.reg_w = torch.tensor([(p + q + r) * (p + q + r + 1) for p, q, r in exps], dtype=torch.float32)

    def _eval(self, dirs, alpha):
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]
        N = dirs.shape[0]
        v = torch.zeros(N, device=dirs.device, dtype=dirs.dtype)
        gx = torch.zeros_like(v)
        gy = torch.zeros_like(v)
        gz = torch.zeros_like(v)
        for k, (p, q, rz) in enumerate(self.exps):
            t = (x ** p) * (y ** q) * (z ** rz)
            v = v + alpha[k] * t
            if p > 0:
                gx = gx + alpha[k] * p * (x ** (p - 1)) * (y ** q) * (z ** rz)
            if q > 0:
                gy = gy + alpha[k] * q * (x ** p) * (y ** (q - 1)) * (z ** rz)
            if rz > 0:
                gz = gz + alpha[k] * rz * (x ** p) * (y ** q) * (z ** (rz - 1))
        grad = torch.stack([gx, gy, gz], -1)
        g = grad - (grad * dirs).sum(-1, True) * dirs
        m = v[:, None] * dirs - g
        m = m / torch.linalg.norm(m, 2, -1, True)
        is_in = ((m * dirs).sum(-1, True) < 0.).float()
        m = m * is_in + -m * (1. - is_in)
        return v[:, None], m

    def __call__(self, key, img, ref_distance, mask, gen_res=512, reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        h, w, _ = img.shape
        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)
        ref_distance = ref_distance.clone().squeeze()[..., None].float().permute(2, 0, 1)
        ref_distance_mask = torch.cat([ref_distance, mask], 0)
        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        for ratio in [1.4]:
            a, b, c, d, e = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio)
            pers_dirs.append(a)
            pers_ratios.append(b)
            to_vecs.append(c)
            down_vecs.append(d)
            right_vecs.append(e)
        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)
        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)
        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)
        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border')
        with torch.no_grad():
            chunk = min(12, n_pers)
            pd_ch = []
            pn_ch = []
            for s in range(0, n_pers, chunk):
                pd_ch.append(self.depth_predictor.predict_depth(pers_imgs[s:s + chunk]).clip(0., None))
                pn_ch.append(self.normal_predictor.predict_normal(pers_imgs[s:s + chunk]))
            pred_depth = torch.cat(pd_ch, 0)
            pred_normals = torch.cat(pn_ch, 0)
            pred_depth = pred_depth / (pred_depth.mean(dim=(2, 3), keepdim=True) + 1e-5)
            pred_distances_raw = pred_depth * pers_ratios.permute(0, 3, 1, 2)
            pred_normals = pred_normals * 2. - 1.
            pred_normals = pred_normals / torch.linalg.norm(pred_normals, ord=2, dim=1, keepdim=True)
            pred_normals = pred_normals.permute(0, 2, 3, 1)
            pred_normals = torch.einsum('bij,bhwj->bhwi', rot_c2w, pred_normals)
            pred_normals_raw = pred_normals.permute(0, 3, 1, 2)
        pers_dirs = pers_dirs.permute(0, 3, 1, 2)
        sup_infos = torch.cat([pers_dirs, pred_distances_raw, pred_normals_raw], dim=1)
        K = len(self.exps)
        alpha = torch.zeros(K, device=device, requires_grad=True)
        scale_param = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([alpha, scale_param], lr=5e-2)
        all_iter_steps = 1500
        lr_alpha = 1e-2
        local_batch_size = 512
        reg_w = self.reg_w.to(device)
        for iter_step in range(all_iter_steps):
            progress = iter_step / all_iter_steps
            lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
            for g in opt.param_groups:
                g['lr'] = 5e-2 * lr_ratio
            sample_coords = torch.rand(n_pers, local_batch_size, 1, 2, device=device) * 2. - 1
            cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')
            dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)
            dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
            ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)
            s = F.softplus(scale_param)[None, None]
            ref_pred_distances = ref_pred_distances * s
            ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
            ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)
            r_pred, m_pred = self._eval(dirs.reshape(-1, 3), alpha)
            r_pred = r_pred.reshape(n_pers, local_batch_size, 1)
            m_pred = m_pred.reshape(n_pers, local_batch_size, 3)
            distance_loss = F.smooth_l1_loss(r_pred, ref_pred_distances, beta=5e-1, reduction='mean')
            normal_loss = F.smooth_l1_loss(m_pred, ref_normals, beta=5e-1, reduction='mean')
            pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
            pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)
            sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')
            sampled_ref_distance = sampled_ref_distance_mask[0, 0]
            sampled_ref_mask = sampled_ref_distance_mask[0, 1]
            ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), r_pred.reshape(-1), beta=1e-2, reduction='none')
            ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()
            reg_spec = ((alpha ** 2) * (reg_w ** 2)).mean()
            loss = ref_distance_loss * 20. * progress + distance_loss + normal_loss * normal_loss_weight + reg_spec * reg_loss_weight
            opt.zero_grad()
            loss.backward()
            opt.step()
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(h, w))
        r_out, m_out = self._eval(pano_dirs.reshape(-1, 3), alpha)
        new_distances = r_out.detach().reshape(h, w, 1)
        new_normals = m_out.detach().reshape(h, w, 3)
        return new_distances, new_normals

class PanoJointPredictor(GeoPredictor):
    def __init__(self, save_path=None):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        self.save_path = save_path

    def grads_to_normal(self, grads):
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals

    def __call__(self, key, img, ref_distance, mask, gen_res=512, 
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        
        height, width, _ = img.shape
        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)                                            
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)                      
        ref_distance = ref_distance.clone().squeeze()[..., None].float().permute(2, 0, 1)     
        ref_distance_mask = torch.cat([ref_distance, mask], 0)

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        for ratio in [1.4]:
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio)
            pers_dirs.append(cur_pers_dirs)
            pers_ratios.append(cur_pers_ratios)
            to_vecs.append(cur_to_vecs)
            down_vecs.append(cur_down_vecs)
            right_vecs.append(cur_right_vecs)

        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)

        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border')
        pred_distances_raw = []
        pred_normals_raw = []

        pers_distances = F.grid_sample(ref_distance[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') 
        pers_masks = F.grid_sample(mask[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') 

        # ===== 加速：改为批量推理（带微批分块避免 OOM） =====
        with torch.no_grad():
            chunk = min(10, n_pers)  # 微批大小，可按显存调整
            pred_depth_chunks = []
            pred_normals_chunks = []
            for s in range(0, n_pers, chunk):
                imgs_chunk = pers_imgs[s:s+chunk]  # [B, 3, 512, 512]
                pd = self.depth_predictor.predict_depth(imgs_chunk).clip(0., None)  # [B, 1, 512, 512]
                pred_depth_chunks.append(pd)
                pn = self.normal_predictor.predict_normal(imgs_chunk)  # [B, 3, 512, 512]
                pred_normals_chunks.append(pn)

            pred_depth = torch.cat(pred_depth_chunks, dim=0)  # [n_pers, 1, 512, 512]
            pred_normals = torch.cat(pred_normals_chunks, dim=0)  # [n_pers, 3, 512, 512]

            # 每张归一化
            pred_depth = pred_depth / (pred_depth.mean(dim=(2, 3), keepdim=True) + 1e-5)
            # 畸变比例按批处理
            pred_distances_raw = pred_depth * pers_ratios.permute(0, 3, 1, 2)  # [n_pers, 1, 512, 512]

            # 法线归一化并批量旋转到世界系
            pred_normals = pred_normals * 2. - 1.
            pred_normals = pred_normals / torch.linalg.norm(pred_normals, ord=2, dim=1, keepdim=True)
            pred_normals = pred_normals.permute(0, 2, 3, 1)
            pred_normals = torch.einsum('bij,bhwj->bhwi', rot_c2w, pred_normals)
            pred_normals_raw = pred_normals.permute(0, 3, 1, 2)  # [n_pers, 3, 512, 512]
        # ===== 以上替代原来的逐视角循环 =====

        # 这里不再需要 cat 列表
        # pred_distances_raw = torch.cat(pred_distances_raw, dim=0)  
        # pred_normals_raw = torch.cat(pred_normals_raw, dim=0)     
        pers_dirs = pers_dirs.permute(0, 3, 1, 2)

        sup_infos = torch.cat([pers_dirs, pred_distances_raw, pred_normals_raw], dim=1)

        scale_params = torch.zeros([n_pers], requires_grad=True)
        bias_params_global = torch.zeros([n_pers], requires_grad=True)
        bias_params_local_distance  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True)
        bias_params_local_normal  = torch.zeros([n_pers, 3, 128, 128], requires_grad=True)

        # Optimize global parameters
        sp_dis_field = SphereDistanceField()
        if key == 0:
            all_iter_steps = 3500 
        else:
            all_iter_steps = 3500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 512

        optimizer_sp = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params, bias_params_global], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance, bias_params_local_normal], lr=init_lr_local)

        for phase in ['global', 'hybrid']:
            for iter_step in range(all_iter_steps):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                # idx = np.random.randint(low=0, high=n_pers)
                sample_coords = torch.rand(n_pers, local_batch_size, 1, 2) * 2. - 1
                cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')     
                distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                normal_bias   = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border') 
                normal_bias   = normal_bias[:, :, :, 0].permute(0, 2, 1)

                dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                  
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                     
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])             
                ref_pred_distances = ref_pred_distances + distance_bias

                ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
                ref_normals = ref_normals + normal_bias
                ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                pred_distances, pred_grads = sp_dis_field(dirs.reshape(-1, 3), requires_grad=True)
                pred_distances = pred_distances.reshape(n_pers, local_batch_size, 1)
                pred_grads = pred_grads.reshape(n_pers, local_batch_size, 3)

                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                ortho_a = torch.randn([n_pers, local_batch_size, 3])
                ortho_b = torch.linalg.cross(dirs, ortho_a)
                ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
                ortho_a = torch.linalg.cross(ortho_b, dirs)
                ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

                val_a = (pred_grads * ortho_a).sum(-1, True) * dirs + ortho_a
                val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
                val_b = (pred_grads * ortho_b).sum(-1, True) * dirs + ortho_b
                val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)
                error_a = (val_a * ref_normals).sum(-1, True)
                error_b = (val_b * ref_normals).sum(-1, True)
                errors = torch.cat([error_a, error_b], -1)
                normal_loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=5e-1, reduction='mean')

                reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()

                if phase == 'hybrid':
                    distance_bias_local = bias_params_local_distance
                    distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
                    normal_bias_local = bias_params_local_normal
                    normal_bias_tv_loss = F.smooth_l1_loss(normal_bias_local[:, :, 1:, :], normal_bias_local[:, :, :-1, :], beta=1e-2) + \
                                          F.smooth_l1_loss(normal_bias_local[:, :, :, 1:], normal_bias_local[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss = 0.
                    normal_bias_tv_loss = 0.

                pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
                pano_sample_coords = img_coord_to_sample_coord(pano_image_coords) 
                sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')  
                sampled_ref_distance = sampled_ref_distance_mask[0, 0]
                sampled_ref_mask =     sampled_ref_distance_mask[0, 1]
                ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()

                loss = ref_distance_loss * 20. * progress + \
                       distance_loss + reg_loss * reg_loss_weight +\
                       normal_loss * normal_loss_weight +\
                       distance_bias_tv_loss * 1. +\
                       normal_bias_tv_loss * normal_tv_loss_weight
            
                optimizer_global.zero_grad()
                optimizer_sp.zero_grad()
                if phase == 'hybrid':
                    optimizer_local.zero_grad()

                loss.backward()
                optimizer_global.step()
                optimizer_sp.step()
                if phase == 'hybrid':
                    optimizer_local.step()

        # Get new distance map and normal map
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        new_distances, new_grads = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3))

        return new_distances, new_normals