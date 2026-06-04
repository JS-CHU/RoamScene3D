import math
import torch
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm
import json

from modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    # features_to_world_space_mesh_label,
    render_mesh,
    edge_threshold_filter,
    unproject_points,
)
from utils.common_utils import (
    visualize_depth_numpy,
    save_rgbd,
)

from modules.mesh_fusion.util import unproject_points_distance

import time
from utils.camera_utils import gen_pano_rays

import utils.functions as functions
from utils.functions import rot_x_world_to_cam, rot_y_world_to_cam, rot_z_world_to_cam, colorize_single_channel_image, write_video
from modules.equilib import equi2pers, cube2equi, equi2cube

from modules.geo_predictors.PanoFusionDistancePredictor import PanoFusionDistancePredictor
from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
from modules.mesh_fusion.sup_info import SupInfoPool
from kornia.morphology import erosion, dilation
from scene.arguments import GSParams, CameraParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics import focal2fov
from utils.loss import l1_loss, ssim
from utils.gen_pano import generate_pano
# from SceneGraph import SceneGraph
from pano_semantic_segment import PanoSemanticSegmentor
from modules.pose_sampler.circle_pose_sampler import CirclePoseSampler
import warnings
warnings.filterwarnings("ignore")
import random
from scene.dataset_readers import loadCamerasFromData
from utils.projection import *

        
@torch.no_grad()
class GenerationPipeline(torch.nn.Module):
    def __init__(self, scene_name, attempt_idx=""):
        '''initialize models and define shared variables'''

        super().__init__()

        # renderer setting
        self.blur_radius = 0
        self.faces_per_pixel = 8
        self.fov = 90
        self.R, self.T = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]), torch.Tensor([[0., 0., 0.]])
        self.pano_width, self.pano_height = 1024*2, 512*2
        self.H, self.W = 512, 512
        self.device = "cuda:0"

        # initialize
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device) 
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)# gaussian_train_data
        self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)# 前3行表示颜色
        self.pc = None
        self.labels = None
        self.scene_graph = None
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)# gaussian_train_data
        self.pix_to_face = None
        self.object_aware = False

        self.pose_scale = 0.6
        self.pano_center_offset = (-0.2,0.3)
        self.inpaint_frame_stride = 20
        self.poses = []

        self.scene_name = scene_name
        self.input_dir = './input'
        self.save_path = f'./output/{self.scene_name}'
        self.GS_render_dir = os.path.join(self.save_path, 'GS_render')
        self.save_details = True

        os.makedirs(self.GS_render_dir, exist_ok=True)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("makedir:", self.save_path)

        self.world_to_cam = torch.eye(4, dtype=torch.float32).to(self.device)
        self.cubemap_w2c_list = functions.get_cubemap_views_world_to_cam()

        self.scene_depth_max = 4.0228885328450446

        self.prompts = []
        self.pidx = []
        self.size = []
        self.size_factor = 1.0

        self.namer = 0

    def set_save_path(self, save_path):
        self.save_path = save_path
        self.GS_render_dir = os.path.join(self.save_path, 'GS_render')
        os.makedirs(self.GS_render_dir, exist_ok=True)

    def set_sampler(self):
        self.global_pose_sampler_conf_list = [{
                'traverse_ratios': [0.1],
                'n_anchors_per_ratio': [4]
            },
            {
                'traverse_ratios': [0.2],
                'n_anchors_per_ratio': [4]
            },
            {
                'traverse_ratios': [0.3],
                'n_anchors_per_ratio': [4]
            },
        ]
        self.perturbation_pose_sampler_conf = {
            'traverse_ratios': [0.2],
            'n_anchors_per_ratio': [4]
        }
        self.max_pose_sampler_conf = {
            'traverse_ratios': [0.6],
            'n_anchors_per_ratio': [200]
        }
    
    def load_modules(self):
        '''在__init__函数中调用 加载两个模型inpainter, geo_predictor'''
        self.inpainter = PanoPersFusionInpainter(save_path=self.save_path)
        self.geo_predictor = PanoJointPredictor(save_path=self.save_path)

    def project(self, world_to_cam):
        '''
        mesh_to_perspective
        using render_mesh
        INPUT:world_to_perspective_camera_pose OUTPUT:rendered_image_tensor, rendered_image_pil
        '''

        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.mesh = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )
        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask
        
        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        '''以下的三个变量暂时未被使用'''
        self.inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        self.inpaint_mask_restore = self.inpaint_mask
        self.inpaint_mask_pil_restore = self.inpaint_mask_pil

        return rendered_image_tensor[:3, ...], rendered_image_pil

    def render_pano(self, pose):
        '''
        mesh_to_cubemap_to_panorama
        using project(), depth_to_distance(), cube2equi()
        INPUT:world_to_panorama_camera_pose OUTPUT:pano_rgb, pano_depth, pano_mask
        '''

        cubemap_list = [] 
        for cubemap_pose in self.cubemap_w2c_list:# self.cubemap_w2c_list于__init__中定义，本质上是pano_to_cubemap的六个坐标转换矩阵形成的列表
            pose_tmp = pose.clone()
            pose_tmp = cubemap_pose.cuda().float() @ pose_tmp# world_to_pano@pano_to_cubemap_sub_i=world_to_cubemap_sub_i 注意可能被名称误导
            rendered_image_tensor, rendered_image_pil = self.project(pose_tmp.cuda())# 渲染cubemap

            rgb_CHW = rendered_image_tensor.squeeze(0).cuda()
            depth_CHW = self.rendered_depth.unsqueeze(0).cuda()
            distance_CHW = functions.depth_to_distance(depth_CHW)
            mask_CHW = self.inpaint_mask.unsqueeze(0).cuda()
            cubemap_list += [torch.cat([rgb_CHW, distance_CHW, mask_CHW], axis=0)]

        torch.set_default_tensor_type('torch.FloatTensor')
        pano_rgbd = cube2equi(cubemap_list,
                                "list",
                                1024,2048)# CHW
        '''六个cubemap拼接形成pano 随后进行切片'''

        pano_rgb = pano_rgbd[:3,:,:]
        pano_depth =  pano_rgbd[3:4,:,:].squeeze(0)
        pano_mask =  pano_rgbd[4:,:,:].squeeze(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return pano_rgb, pano_depth, pano_mask# CHW, HW, HW

    def rgbd_to_mesh(self, rgb, depth, world_to_cam=None, mask=None, pix_to_face=None, using_distance_map=False, pseudo=False):
        '''
        RGBD_to_mesh
        using features_to_world_space_mesh()
        INPUT:RGBD OUTPUT:None
        mesh iteration
        '''
        
        predicted_depth = depth.cuda()
        rgb = rgb.squeeze(0).cuda()
        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam = world_to_cam.cuda()
        if pix_to_face is not None:
            self.pix_to_face = pix_to_face
        if mask is None:
            self.inpaint_mask = torch.ones_like(predicted_depth)
        else:
            self.inpaint_mask = mask

        if self.inpaint_mask.sum() == 0:
            return

        vertices, faces, colors, pc = features_to_world_space_mesh(
            colors=rgb,
            depth=predicted_depth,
            fov_in_degrees=self.fov,
            world_to_cam=world_to_cam,
            mask=self.inpaint_mask,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices,
            using_distance_map=using_distance_map,
            edge_threshold=0.05
        )
        if self.pc is None:
            self.pc = pc
        faces += self.vertices.shape[1] 
        self.vertices_restore = self.vertices.clone()
        self.colors_restore = self.colors.clone()
        self.faces_restore = self.faces.clone()

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def find_depth_edge(self, depth, dilate_iter=0):
        '''
        depth_to_EdgeMask
        usingcv2.canny()
        INPUT:depth OUTPUT:EdgeMask
        '''

        gray = (depth/depth.max() * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 60, 150)
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        return edges

    def pano_distance_to_mesh(self, pano_rgbl, pano_distance, depth_edge_inpaint_mask, pose=None, pseudo=False):
        '''
        panoramaRGBD_to_mesh
        using rgbd_to_mesh
        INPUT:panoramaRGBD OUTPUT:None
        mesh iteration
        '''
        self.rgbd_to_mesh(pano_rgbl, pano_distance, mask=depth_edge_inpaint_mask, using_distance_map=True, world_to_cam=pose, pseudo=pseudo)

    def stage_inpaint_pano_greedy_search(self, pose_dict, source_depth, add_mesh=False):
        '''
        选取完整度处于2/3分位的pose进行inpainting 完成mesh_iteration并且收集pseudo_view
        using render_pano(), inpaint_new_panorama(), geo_check(), pano_distance_to_mesh()
        INPUT:pose_dict OUTPUT:inpainted_panos_and_poses(list)
        '''

        inpainted_panos_and_poses = []
        while len(pose_dict) > 0:
            print(f"len(pose_dict):{len(pose_dict)}")
            keys = list(pose_dict.keys())
            key = random.choice(keys)
            pose = pose_dict[key]
            # print(f"random_selected_key:{key}")
            del pose_dict[key]
            
            # rendering rgb depth mask
            cam = pose[:3, 3].to('cuda')
            cam = self.cam_scale(source_depth, cam)
            pose = pose.cuda()
            pano_rgb, pano_distance, pano_mask = self.render_pano(pose)

            # inpaint pano
            colors = pano_rgb.permute(1,2,0).clone() # HWC
            distances = pano_distance.unsqueeze(-1).clone()
            pano_inpaint_mask = pano_mask.clone()

            if pano_inpaint_mask.min().item() < .5:
                # inpainting pano
                colors, distances, normals = self.inpaint_new_panorama(idx=1, colors=colors, distances=distances, pano_mask=pano_inpaint_mask, pose=cam) # HWC, HWC, HW
                
                #apply_GeoCheck:
                perf_pose = pose.clone()
                perf_pose[0,3], perf_pose[1,3], perf_pose[2,3] = -pose[0,3], pose[2,3], 0 
                rays = gen_pano_rays(perf_pose, self.pano_height, self.pano_width)
                conflict_mask = self.sup_pool.geo_check(rays, distances.unsqueeze(-1))    # 0 conflict, 1 not conflict
                pano_inpaint_mask = pano_inpaint_mask * conflict_mask # 没有冲突的地方被标记为需要修复的地方，有冲突的地方略过
                    
            # add new mesh
            if add_mesh:
                self.pano_distance_to_mesh(colors.permute(2,0,1), distances, pano_inpaint_mask, pose=pose) #CHW, HW, HW

            # apply_GeoCheck:
            sup_mask = pano_inpaint_mask.clone()
            self.sup_pool.register_sup_info(pose=perf_pose, mask=sup_mask, rgb=colors, distance=distances.unsqueeze(-1), normal=normals)
            
            # save renderred
            self.namer += 1
            panorama_tensor_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/renderred_pano_{self.namer}.png")
            if self.save_details:
                inpaint_mask_pil = Image.fromarray(pano_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/mask_{self.namer}.png")  

            # save inpainted
            panorama_tensor_pil = functions.tensor_to_pil(colors.permute(2,0,1).unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/inpainted_pano_{self.namer}.png")

            # save pose
            torch.save(pose, f"{self.save_path}/pose_{self.namer}.pt") 
            
        return inpainted_panos_and_poses

    def cam_scale(self, source_depth, cam):
        top_rows = source_depth[0:5, :]
        bottom_rows = source_depth[-5:, :]
        valid_top_depths = top_rows[top_rows > 0]
        valid_bottom_depths = bottom_rows[bottom_rows > 0]
        dist_to_ceiling = torch.median(valid_top_depths)
        dist_to_floor = torch.median(valid_bottom_depths)
        scene_scale = dist_to_ceiling + dist_to_floor
        normalized_pos = cam / scene_scale
        directions = torch.tensor([[[1, 0, 0],
                               [0, 0, 1],
                               [-1, 0, 0],
                               [0, 0, -1],
                               [0, 1, 0],
                               [0, -1, 0]]],
                               dtype=torch.float32).to(cam.device)
        cam = torch.cat([normalized_pos.unsqueeze(0).unsqueeze(0).repeat(1, directions.size(1), 1), directions], dim=-1)
        cam = cam.view(-1, 6)
        return cam

    def inpaint_new_panorama(self, idx, colors, distances, pano_mask, pose):
        '''
        inpainting
        using cv2.getStructuringElement(), inpainter.inpaint(), geo_predictor()
        INPUT:idx, RGBD, mask OUTPUT:inpainted_img, inpainted_distances, inpainted_normals
        '''

        print(f"inpaint_new_panorama")

        # must dilate mask first
        mask = pano_mask.unsqueeze(-1)
        s_size = (9, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, s_size)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)
        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask.permute(0, 2, 3, 1).contiguous().squeeze(0).squeeze(-1)
        '''扩大需修复区域 确保边缘覆盖完整'''

        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        inpainted_distances = None
        inpainted_normals = None
        inpainted_img = self.inpainter.inpaint(idx, colors, mask, pose)

        # Keep renderred part
        inpainted_img = colors * (1 - mask) + inpainted_img * mask# 仅改变掩码部分
        inpainted_img = inpainted_img.cuda()

        inpainted_distances, inpainted_normals = self.geo_predictor(idx,
                                                                    inpainted_img,
                                                                    distances,
                                                                    mask=mask,
                                                                    reg_loss_weight=0.,
                                                                    normal_loss_weight=5e-2,
                                                                    normal_tv_loss_weight=5e-2)
        '''深度估计+法线预测'''
        inpainted_distances = inpainted_distances.squeeze()
        return inpainted_img, inpainted_distances, inpainted_normals

    def load_pano(self):
        '''
        加载panorama_init
        using resize_image_with_aspect_ratio(), pano_fusion_distance_predictor.predict()
        INPUT:Null OUTPUT:panorama_tensor, depth 
        '''

        image_path = f"{self.input_dir}/scene_panoramas/{self.scene_name}.png"
        image = Image.open(image_path)
        if image.size[0] < image.size[1]: # size[0]表示图像的宽，size[1]表示图像的高
            image = image.transpose(Image.TRANSPOSE)
        image = functions.resize_image_with_aspect_ratio(image, new_width=self.pano_width)
        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).float()/255

        depth = self.predict_depth(panorama_tensor)
        
        return panorama_tensor, depth
    
    def predict_depth(self, panorama_tensor):
        depth_scale_factor = 3.4092
        pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        depth = pano_fusion_distance_predictor.predict(panorama_tensor.permute(1,2,0))# input:HW3
        depth = depth/depth.max() * depth_scale_factor # 1.8
        return depth

    def load_camera_poses(self, pano_center_offset=[0,0]):# panorama_camera中心偏移量默认为0
        '''
        create panorama_pose, pose
        using nothing
        INPUT:None OUTPUT:panorama_pose(NDArray), pose(list)
        '''

        subset_path = f'{self.input_dir}/Camera_Trajectory'# initial 6 poses are cubemaps poses
        files = os.listdir(subset_path)

        pano_pose_44 = None
        pose_files = [f for f in files if f.startswith('camera_pose')]
        pose_files = sorted(pose_files)
        poses_name = pose_files
        poses = []
        for i, pose_name in enumerate(poses_name):
            with open(f'{subset_path}/{pose_name}', 'r') as f: 
                lines = f.readlines()
            pose_44 = []
            for line in lines:
                pose_44 += line.split()
            pose_44 = np.array(pose_44).reshape(4, 4).astype(float)
            if pano_pose_44 is None:
                pano_pose_44 = pose_44.copy()
                pano_pose_44_cubemaps = pose_44.copy()
                pano_pose_44[0,3] += pano_center_offset[0]
                pano_pose_44[2,3] += pano_center_offset[1]
            
            if i < 6:
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44_cubemaps)  
            else:
                ### convert gt_pose to gt_relative_pose with pano_pose
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44)

            pose_relative_44 = np.vstack((-pose_relative_44[0:1,:], -pose_relative_44[1:2,:], pose_relative_44[2:3,:], pose_relative_44[3:4,:]))
            pose_relative_44 = pose_relative_44 @ rot_z_world_to_cam(180).cpu().numpy()

            pose_relative_44[:3,3] *= self.pose_scale
            poses += [torch.tensor(pose_relative_44).float()]# w2c
            '''relative:以第一个位姿pano_pose_44为基准 计算其他位姿的相对值 相当于形成了w2c'''

        return pano_pose_44, poses

    def pano_to_perpective(self, pano_bchw, pitch, yaw, fov):
        '''
        panorama_to_perspective
        using equi2pers()
        INPUT:panorama, pitch, yaw, fov OUTPUT:Perspective
        '''

        rots = {
            'roll': 0.,
            'pitch': pitch,# rotate vertical
            'yaw': yaw,# rotate horizontal
        }
        '''pitch:俯仰角ψ yaw:偏航角θ'''

        perspective = equi2pers(
            equi=pano_bchw.squeeze(0),
            rots=rots,
            height=self.H,
            width=self.W,
            fov_x=fov,
            mode="bilinear",
        ).unsqueeze(0)# BCHW

        return perspective

    def pano_to_cubemap(self, pano_tensor, pano_depth_tensor=None):# BCHW, HW
        '''
        panorama_to_cubemap
        using pano_to_perspective()
        INPUT:panorama OUTPUT:cubemap, cubelap_depth
        '''

        '''注意这里INPUT:pano_depth_tensor=None && OUTPUT:cubemaps_depth=None'''

        cubemaps_pitch_yaw = [(0, 0), (0, 3/2 * np.pi), (0, 1 * np.pi), (0, 1/2 * np.pi),\
                            (-1/2 * np.pi, 0), (1/2 * np.pi, 0)]
        pitch_yaw_list = cubemaps_pitch_yaw
        '''pitch:俯仰角ψ yaw:偏航角θ'''

        cubemaps = []
        cubemaps_depth = []
        # collect fov 90 cubemaps
        for view_idx, (pitch, yaw) in enumerate(pitch_yaw_list):
            view_rgb = self.pano_to_perpective(pano_tensor, pitch, yaw, 90)
            cubemaps += [view_rgb.cpu().clone()]
            if pano_depth_tensor is not None:
                view_depth = self.pano_to_perpective(pano_depth_tensor.unsqueeze(0).unsqueeze(0), pitch, yaw, 90)
                cubemaps_depth += [view_depth.cpu().clone()]
        return cubemaps, cubemaps_depth# BCHW, BCHW

    def train_GS(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in tqdm(iterable_gauss):
            self.gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam, mesh_pose = viewpoint_stack[iteration%len(viewpoint_stack)]

            # Render GS
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            render_image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
            loss.backward()

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
    def GS2pano(self, viewpoint_cam, gaussians, opt, background, i=0):
        """
        参考 eval_GS 的相机创建流程：
        - 以当前 viewpoint_cam 的位姿为基准，叠加六个立方体朝向构造相机
        - 用 render 渲染六个面并拼接为全景
        - 保存到 f"{self.GS_render_dir}/pano_render_{i}.png"
        """

        # 构造 evaldata（与 eval_GS 相同风格）
        evaldata = {
            'camera_angle_x': focal2fov(self.H / 2, self.W),
            'W': self.W,
            'H': self.H,
            'frames': [],
        }

        # 基准位姿：viewpoint_cam 的 world_view_transform 作为 w2c
        base_w2c = viewpoint_cam.world_view_transform.clone()

        for cubemap_pose in self.cubemap_w2c_list:
            mesh_pose = cubemap_pose.cuda().float() @ base_w2c.clone()

            pose_44 = mesh_pose.clone().float()
            pose_44[0:1, :] *= -1
            pose_44[1:2, :] *= -1

            Rw2c = pose_44[:3, :3].detach().cpu().numpy()
            Tw2c = pose_44[:3, 3:].detach().cpu().numpy()
            yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0, 0, 0, 1]])), axis=0)

            # 构造帧；图像可用空白占位（render 不依赖它）
            dummy_img = Image.new("RGB", (self.W, self.H))
            evaldata['frames'].append({
                'image': dummy_img,
                'transform_matrix': Pc2w.tolist(),
                'fovx': focal2fov(self.H / 2, self.W),
                'mesh_pose': mesh_pose.clone()
            })

        # 生成六个评估相机（与 eval_GS 一致）
        eval_cams = loadCamerasFromData(evaldata, opt.white_background)

        # 渲染六面
        cubemap_list = []
        for face_cam, _ in eval_cams:
            results = render(face_cam, gaussians, opt, background)
            frame = results['render']  # BCHW
            cubemap_list.append(frame.squeeze(0).detach().cpu())  # CHW

        # 立方体转全景
        torch.set_default_tensor_type('torch.FloatTensor')
        pano_rgb = cube2equi(cubemap_list, "list", self.pano_height, self.pano_width)  # CHW
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # 保存全景图
        pano_rgb = torch.flip(pano_rgb, dims=(1, 2))
        pano_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
        pano_pil.save(f"{self.GS_render_dir}/pano_render_{i}.png")

        return pano_rgb

    def GS2pano_roampose(self, inpainted_panos_and_poses, gaussians, opt, background):
        """
        对于inpainted_panos_and_poses中的每一个相机位姿，渲染一个全景图
        保存为./{self.save_path}/render_roampose{i}.png
        """
        for i, (inpainted_pano_images, pano_pose_44) in enumerate(inpainted_panos_and_poses):
            # 构造 evaldata
            evaldata = {
                'camera_angle_x': focal2fov(self.H / 2, self.W),
                'W': self.W,
                'H': self.H,
                'frames': [],
            }

            # 基准位姿
            base_w2c = pano_pose_44.clone()

            for cubemap_pose in self.cubemap_w2c_list:
                # 注意：这里需要确保 base_w2c 在正确的 device 上，且类型匹配
                mesh_pose = cubemap_pose.cuda().float() @ base_w2c.cuda().float()

                pose_44 = mesh_pose.clone().float()
                pose_44[0:1, :] *= -1
                pose_44[1:2, :] *= -1

                Rw2c = pose_44[:3, :3].detach().cpu().numpy()
                Tw2c = pose_44[:3, 3:].detach().cpu().numpy()
                yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                Rc2w = np.matmul(yz_reverse, Rw2c).T
                Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
                Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0, 0, 0, 1]])), axis=0)

                # 构造帧
                dummy_img = Image.new("RGB", (self.W, self.H))
                evaldata['frames'].append({
                    'image': dummy_img,
                    'transform_matrix': Pc2w.tolist(),
                    'fovx': focal2fov(self.H / 2, self.W),
                    'mesh_pose': mesh_pose.clone()
                })

            # 生成六个评估相机
            eval_cams = loadCamerasFromData(evaldata, opt.white_background)

            # 渲染六面
            cubemap_list = []
            for face_cam, _ in eval_cams:
                results = render(face_cam, gaussians, opt, background)
                frame = results['render']  # BCHW
                cubemap_list.append(frame.squeeze(0).detach().cpu())  # CHW

            # 立方体转全景
            torch.set_default_tensor_type('torch.FloatTensor')
            pano_rgb = cube2equi(cubemap_list, "list", self.pano_height, self.pano_width)  # CHW
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            # 保存全景图
            # pano_rgb = torch.flip(pano_rgb, dims=(1, 2))
            pano_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
            
            renderings_dir = os.path.join(self.save_path, 'renderings')
            os.makedirs(renderings_dir, exist_ok=True)
            save_path = os.path.join(renderings_dir, f"render_roampose{i}.png")
            pano_pil.save(save_path)
            print(f"Saved {save_path}")

    def eval_GS(self, eval_GS_cams):
        viewpoint_stack = eval_GS_cams
        # l1_val = 0
        # ssim_val = 0
        # psnr_val = 0
        framelist = []
        depthlist = []
        for i in range(len(viewpoint_stack)):
            viewpoint_cam, mesh_pose = viewpoint_stack[i]
            self.GS2pano(viewpoint_cam, self.gaussians, self.opt, self.background, i=i)
            results = render(viewpoint_cam, self.gaussians, self.opt, self.background) # 用这个方法来得到3DGS的渲染结果
            frame, depth = results['render'], results['depth'].detach().cpu()
            framelist.append(
                np.round(frame.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depthlist.append(colorize_single_channel_image(depth.detach().cpu()/self.scene_depth_max))

        if self.save_details:
            for i, frame in enumerate(framelist):
                image = Image.fromarray(frame, mode="RGB")
                image.save(os.path.join(self.GS_render_dir, f"pers_render_{i}.png"))
                functions.write_image(f"{self.GS_render_dir}/pers_render_depth_{i}.png", depthlist[i])
        
        write_video(f"{self.GS_render_dir}/GS_render_video.mp4", framelist[6:], fps=30)
        write_video(f"{self.GS_render_dir}/GS_depth_video.mp4", depthlist[6:], fps=30)
        print("Result saved at: ", self.GS_render_dir)

    def xyz_to_xz_y(self, xyz_coords):
        x = xyz_coords[..., 0]
        y = xyz_coords[..., 1]
        z = xyz_coords[..., 2]
        return torch.stack([x, -z, y], dim = -1)

    def save_mesh(self, stage):
        def to_uint8_rgb(colors_np: np.ndarray) -> np.ndarray:
            if colors_np.size == 0:
                return colors_np.astype(np.uint8)
            cmax = float(np.nanmax(colors_np))
            if cmax <= 1.0 + 1e-6:
                colors_np = colors_np * 255.0
            colors_np = np.clip(np.rint(colors_np), 0, 255).astype(np.uint8)
            return colors_np

        def bbx_edges_from_corners(bbx8_np: np.ndarray) -> np.ndarray:
            edges = np.array(
                [
                    [0, 1],
                    [1, 5],
                    [5, 4],
                    [4, 0],
                    [2, 3],
                    [3, 7],
                    [7, 6],
                    [6, 2],
                    [0, 2],
                    [1, 3],
                    [4, 6],
                    [5, 7],
                ],
                dtype=np.int64,
            )
            return edges

        def sample_segment_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            return p0[None, :] * (1.0 - t) + p1[None, :] * t

        def tube_mesh_for_segment(p0: np.ndarray, p1: np.ndarray, thickness: float):
            d = p1 - p0
            seg_len = float(np.linalg.norm(d))
            if seg_len < 1e-10:
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
            d = d / seg_len
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(d, up))) > 0.9:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            u = np.cross(d, up)
            u_norm = float(np.linalg.norm(u))
            if u_norm < 1e-10:
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
            u = u / u_norm
            v = np.cross(d, u)
            half = float(thickness) * 0.5

            offsets = np.array(
                [
                    u * half + v * half,
                    u * half - v * half,
                    -u * half - v * half,
                    -u * half + v * half,
                ],
                dtype=np.float32,
            )

            v0 = p0[None, :] + offsets
            v1 = p1[None, :] + offsets
            verts = np.concatenate([v0, v1], axis=0)  # (8, 3)

            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 6, 5],
                    [4, 7, 6],
                    [0, 4, 5],
                    [0, 5, 1],
                    [1, 5, 6],
                    [1, 6, 2],
                    [2, 6, 7],
                    [2, 7, 3],
                    [3, 7, 4],
                    [3, 4, 0],
                ],
                dtype=np.int64,
            )
            return verts, faces

        os.makedirs(self.save_path, exist_ok=True)

        vertices_np = self.vertices.detach().cpu().numpy().T.astype(np.float32)  # (N, 3)
        colors_np = self.colors.detach().cpu().numpy().T  # (N, 3)
        colors_np = to_uint8_rgb(colors_np)

        if vertices_np.shape[0] == 0:
            print(f"警告: 没有顶点数据可保存 {stage}_pc.ply / {stage}_mesh.ply")
            return

        bbx8 = self.instance_bbx8
        if bbx8 is None or (torch.is_tensor(bbx8) and bbx8.numel() == 0):
            bbx8_np = np.zeros((0, 8, 3), dtype=np.float32)
        else:
            bbx8_np = bbx8.detach().cpu().numpy().astype(np.float32).reshape(-1, 8, 3)

        diag = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        thickness = max(diag * 0.003, 1e-3)

        edge_rgb = np.array([0, 0, 139], dtype=np.uint8)
        pc_tube_verts_all = []
        pc_tube_faces_all = []
        if bbx8_np.shape[0] > 0:
            edges = bbx_edges_from_corners(bbx8_np[0])
            for m in range(bbx8_np.shape[0]):
                corners = bbx8_np[m]
                for a, b in edges:
                    p0 = corners[a]
                    p1 = corners[b]
                    tv, tf = tube_mesh_for_segment(p0, p1, thickness=thickness)
                    if tv.shape[0] == 0:
                        continue
                    offset = vertices_np.shape[0] + sum(v.shape[0] for v in pc_tube_verts_all)
                    pc_tube_verts_all.append(tv)
                    pc_tube_faces_all.append(tf + offset)

        if len(pc_tube_verts_all) > 0:
            pc_tube_verts_np = np.concatenate(pc_tube_verts_all, axis=0).astype(np.float32)
            pc_tube_faces_np = np.concatenate(pc_tube_faces_all, axis=0).astype(np.int64)
            pc_tube_cols_np = np.tile(edge_rgb[None, :], (pc_tube_verts_np.shape[0], 1))
            pc_vertices_out = np.concatenate([vertices_np, pc_tube_verts_np], axis=0)
            pc_colors_out = np.concatenate([colors_np, pc_tube_cols_np], axis=0)
            pc_faces_out = pc_tube_faces_np
        else:
            pc_vertices_out = vertices_np
            pc_colors_out = colors_np
            pc_faces_out = np.zeros((0, 3), dtype=np.int64)

        pc_path = os.path.join(self.save_path, f"{stage}_pc.ply")
        with open(pc_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pc_vertices_out.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {pc_faces_out.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(pc_vertices_out.shape[0]):
                x, y, z = pc_vertices_out[i]
                r, g, b = pc_colors_out[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            for i in range(pc_faces_out.shape[0]):
                a, b, c = pc_faces_out[i]
                f.write(f"3 {int(a)} {int(b)} {int(c)}\n")
        print(f"PLY saved to: {pc_path}")

        mesh_vertices_np = vertices_np
        mesh_colors_np = colors_np
        faces_tensor = self.faces.detach().cpu()
        faces_np = faces_tensor.numpy().T.astype(np.int64) if faces_tensor.numel() > 0 else np.zeros((0, 3), dtype=np.int64)
        if faces_np.size > 0 and faces_np.shape[1] != 3:
            if faces_tensor.shape[0] == 3:
                faces_np = faces_tensor.numpy().T.astype(np.int64)
            else:
                faces_np = np.zeros((0, 3), dtype=np.int64)

        tube_verts_all = []
        tube_faces_all = []
        if bbx8_np.shape[0] > 0:
            edges = bbx_edges_from_corners(bbx8_np[0])
            for m in range(bbx8_np.shape[0]):
                corners = bbx8_np[m]
                for a, b in edges:
                    p0 = corners[a]
                    p1 = corners[b]
                    tv, tf = tube_mesh_for_segment(p0, p1, thickness=thickness)
                    if tv.shape[0] == 0:
                        continue
                    offset = mesh_vertices_np.shape[0] + sum(v.shape[0] for v in tube_verts_all)
                    tube_verts_all.append(tv)
                    tube_faces_all.append(tf + offset)

        if len(tube_verts_all) > 0:
            tube_verts_np = np.concatenate(tube_verts_all, axis=0).astype(np.float32)
            tube_faces_np = np.concatenate(tube_faces_all, axis=0).astype(np.int64)
            tube_cols_np = np.tile(edge_rgb[None, :], (tube_verts_np.shape[0], 1))
            mesh_vertices_out = np.concatenate([mesh_vertices_np, tube_verts_np], axis=0)
            mesh_colors_out = np.concatenate([mesh_colors_np, tube_cols_np], axis=0)
            mesh_faces_out = np.concatenate([faces_np, tube_faces_np], axis=0) if faces_np.shape[0] > 0 else tube_faces_np
        else:
            mesh_vertices_out = mesh_vertices_np
            mesh_colors_out = mesh_colors_np
            mesh_faces_out = faces_np

        mesh_path = os.path.join(self.save_path, f"{stage}_mesh.ply")
        with open(mesh_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {mesh_vertices_out.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {mesh_faces_out.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(mesh_vertices_out.shape[0]):
                x, y, z = mesh_vertices_out[i]
                r, g, b = mesh_colors_out[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            for i in range(mesh_faces_out.shape[0]):
                a, b, c = mesh_faces_out[i]
                f.write(f"3 {int(a)} {int(b)} {int(c)}\n")
        print(f"PLY saved to: {mesh_path}")

    def load_vertices_and_colors(self, filename=None):
        """
        从 PLY 文件加载 vertices 和 colors 数据
        
        Args:
            filename (str, optional): PLY 文件路径。如果为 None，则使用默认路径
        """
        if filename is None:
            filename = os.path.join(self.save_path, "initial_scene.ply")
        
        if not os.path.exists(filename):
            print(f"错误: PLY 文件不存在: {filename}")
            return
        
        vertices_list = []
        colors_list = []
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
                # 解析 PLY 文件头
                header_end = 0
                num_vertices = 0
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith("element vertex"):
                        num_vertices = int(line.split()[-1])
                    elif line == "end_header":
                        header_end = i + 1
                        break
                
                if num_vertices == 0:
                    print("警告: PLY 文件中没有顶点数据")
                    return
                
                # 读取顶点数据
                for i in range(header_end, header_end + num_vertices):
                    if i >= len(lines):
                        break
                    
                    parts = lines[i].strip().split()
                    if len(parts) >= 6:
                        # 解析位置 (x, y, z)
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        vertices_list.append([x, y, z])
                        
                        # 解析颜色 (r, g, b)
                        r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                        # 将颜色从 [0, 255] 转换为 [0, 1]
                        colors_list.append([r, g, b])
                
                if len(vertices_list) == 0:
                    print("警告: 没有成功解析到顶点数据")
                    return
                
                # 转换为 PyTorch 张量
                vertices_np = np.array(vertices_list, dtype=np.float32)  # (N, 3)
                colors_np = np.array(colors_list, dtype=np.float32)      # (N, 3)
                
                # 转换为 (3, N) 格式并移动到设备
                self.vertices = torch.from_numpy(vertices_np.T).to(self.device)  # (3, N)
                self.colors = torch.from_numpy(colors_np.T).to(self.device)      # (3, N)
                
                print(f"PLY 文件加载成功: {filename}")
                print(f"加载的顶点数量: {self.vertices.shape[1]}")
                
        except Exception as e:
            print(f"加载 PLY 文件时出错: {e}")
            # 如果加载失败，重置为空张量
            self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)
            self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)
    
    def load_inpainted_panos_and_poses(self):
        inpainted_panos_and_poses = []
        sr_dir = os.path.join(self.save_path, 'tmp')
        for item in os.listdir(sr_dir):
            if item.startswith('sr_'):
                num = item.split('_')[-1].split('.')[0]
                # print(num)
                inpainted_pano = Image.open(f"{sr_dir}/{item}")
                inpainted_pano = functions.pil_to_tensor(inpainted_pano)
                # print(f"{self.save_path}/pose_{num}.pt")
                pose = torch.load(f"{self.save_path}/pose_{num}.pt")
                inpainted_panos_and_poses += [(inpainted_pano, pose)]
        return inpainted_panos_and_poses

    def load_scene_graph(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                scene_graph_json = file.read()
                self.scene_graph: dict = json.loads(scene_graph_json)

    def pano_segment(self):
        self.load_scene_graph(f"{self.input_dir}/scene_graphs/{self.scene_name}.json")
        pano_path = f"{self.input_dir}/scene_panoramas/{self.scene_name}.png"
        segmentor = PanoSemanticSegmentor(pano_path, self.scene_graph)
        seg_map, intance_num = segmentor.segment(f"{self.save_path}/instances")
        self.instance_num = intance_num
        self.object_aware = True
        self.labels = torch.from_numpy(seg_map).flatten()
        
        del segmentor
        torch.cuda.empty_cache()

        np.save(f"{self.save_path}/instances/seg_map.npy", seg_map)
        np.save(f"{self.save_path}/instances/labels.npy", self.labels.detach().cpu().numpy())

    def decouple_instances(self):
        device = self.pc.device
        # device = 'cuda'
        labels = self.labels.to(device).long()
        M = self.instance_num
        
        # 边界与空输入处理：直接写入空结果属性
        if M <= 0 or self.pc is None or labels.numel() == 0 or torch.all(labels == 0):
            instance_centers = torch.zeros(max(M, 0), 3, device=device, dtype=torch.float32)
            instance_bbx8 = torch.zeros(max(M, 0), 8, 3, device=device, dtype=torch.float32)
            return instance_centers, instance_bbx8
        
        valid = (labels >= 1) & (labels <= M)
        labels_f = labels[valid] - 1  # 0-based
        pc_f = self.pc[:, valid]      # [3, N_valid]
        
        # 若无有效点，写入零张量并返回
        if pc_f.numel() == 0:
            instance_centers = torch.zeros(M, 3, device=device, dtype=torch.float32)
            instance_bbx8 = torch.zeros(M, 8, 3, device=device, dtype=torch.float32)
            return instance_centers, instance_bbx8
        
        dtype = pc_f.dtype
        
        # 计数/求和用于中心点
        ones = torch.ones(labels_f.shape[0], dtype=dtype, device=device)
        counts = torch.zeros(M, dtype=dtype, device=device)
        counts.scatter_add_(0, labels_f, ones)  # [M]f
        
        sums = torch.zeros(3, M, dtype=dtype, device=device)
        sums[0].scatter_add_(0, labels_f, pc_f[0])
        sums[1].scatter_add_(0, labels_f, pc_f[1])
        sums[2].scatter_add_(0, labels_f, pc_f[2])
        centers = (sums / counts.clamp(min=1)).T  # [M, 3]
        
        # AABB 的最小/最大轴值
        mins = torch.full((M, 3), float('inf'), dtype=dtype, device=device)
        maxs = torch.full((M, 3), -float('inf'), dtype=dtype, device=device)
        mins[:, 0].scatter_reduce_(0, labels_f, pc_f[0], reduce='amin', include_self=True)
        mins[:, 1].scatter_reduce_(0, labels_f, pc_f[1], reduce='amin', include_self=True)
        mins[:, 2].scatter_reduce_(0, labels_f, pc_f[2], reduce='amin', include_self=True)
        maxs[:, 0].scatter_reduce_(0, labels_f, pc_f[0], reduce='amax', include_self=True)
        maxs[:, 1].scatter_reduce_(0, labels_f, pc_f[1], reduce='amax', include_self=True)
        maxs[:, 2].scatter_reduce_(0, labels_f, pc_f[2], reduce='amax', include_self=True)
        
        # 空实例填零
        empty = counts == 0
        mins[empty] = 0
        maxs[empty] = 0
        
        # 生成 8 顶点 [M, 8, 3]
        xs = torch.stack([mins[:, 0], mins[:, 0], mins[:, 0], mins[:, 0], maxs[:, 0], maxs[:, 0], maxs[:, 0], maxs[:, 0]], dim=1)
        ys = torch.stack([mins[:, 1], mins[:, 1], maxs[:, 1], maxs[:, 1], mins[:, 1], mins[:, 1], maxs[:, 1], maxs[:, 1]], dim=1)
        zs = torch.stack([mins[:, 2], maxs[:, 2], mins[:, 2], maxs[:, 2], mins[:, 2], maxs[:, 2], mins[:, 2], maxs[:, 2]], dim=1)
        bbx_all = torch.stack([xs, ys, zs], dim=2)      # [M, 8, 3]

        return centers, bbx_all

    def traj_gen(self, positions, sampler, max_traj):
        perturbation_pose_dict = {}
        key = 0
        if (
            self.instance_num == 0
            or self.labels is None
            or torch.all(self.labels == 0)
            or self.instance_centers is None
            or torch.all(self.instance_centers == 0)
            or self.instance_bbx8 is None
            or self.instance_bbx8.numel() == 0
        ):
            for pos in positions:
                perturbation_pose_dict[key] = sampler.sample_pose(pos)
                key += 1
            return perturbation_pose_dict

        two_pi = 2.0 * math.pi
        pi = math.pi

        def _angle_wrap(a: torch.Tensor) -> torch.Tensor:
            return torch.remainder(a + pi, two_pi) - pi

        max_traj_x = max_traj[:, 0]
        max_traj_z = max_traj[:, 2]
        max_traj_phi = torch.atan2(max_traj_z, max_traj_x)
        max_traj_phi = torch.remainder(max_traj_phi + two_pi, two_pi)
        max_traj_r = torch.sqrt(max_traj_x * max_traj_x + max_traj_z * max_traj_z)

        sort_idx = torch.argsort(max_traj_phi)
        max_traj_phi = max_traj_phi[sort_idx]
        max_traj_r = max_traj_r[sort_idx]
        max_traj_phi = torch.cat([max_traj_phi, max_traj_phi[:1] + two_pi], dim=0)
        max_traj_r = torch.cat([max_traj_r, max_traj_r[:1]], dim=0)

        up = torch.tensor([0.0, 1.0, 0.0], device=max_traj.device, dtype=max_traj.dtype)

        def _interp_max_r(phi_query: torch.Tensor) -> torch.Tensor:
            phi_query = torch.remainder(phi_query + two_pi, two_pi)
            idx = torch.searchsorted(max_traj_phi, phi_query).clamp(1, max_traj_phi.numel() - 1)
            idx0 = idx - 1
            a0 = max_traj_phi[idx0]
            a1 = max_traj_phi[idx]
            r0 = max_traj_r[idx0]
            r1 = max_traj_r[idx]
            denom = (a1 - a0).clamp_min(1e-8)
            t = (phi_query - a0) / denom
            return r0 * (1.0 - t) + r1 * t

        traj_points = []
        for pos in positions:
            dx = self.instance_centers[:, 0] - pos[0]
            dz = self.instance_centers[:, 2] - pos[2]
            distances = torch.sqrt(dx * dx + dz * dz + 1e-8)
            min_idx = torch.argmin(distances)

            center = self.instance_centers[min_idx]
            bbx = self.instance_bbx8[min_idx]
            bbx_min = bbx.min(dim=0).values
            bbx_max = bbx.max(dim=0).values

            w_x = torch.abs(bbx_max[0] - bbx_min[0])
            w_z = torch.abs(bbx_max[2] - bbx_min[2])
            s_obj = torch.sqrt(w_x * w_x + w_z * w_z + 1e-8)
            sigma_phi = torch.clamp(0.6 * s_obj, min=0.2, max=1.2)
            amplitude = torch.clamp(0.25 * s_obj, min=0.05, max=0.6)

            phi_l = torch.atan2(pos[2], pos[0])
            phi_o = torch.atan2(center[2], center[0])
            dphi = _angle_wrap(phi_l - phi_o)
            weight = dphi * torch.exp(-(dphi * dphi) / (2.0 * sigma_phi * sigma_phi + 1e-8))

            view_vec = torch.stack([center[0] - pos[0], torch.zeros_like(pos[1]), center[2] - pos[2]], dim=0)
            tangent = torch.cross(view_vec, up)
            tangent_norm = torch.linalg.norm(tangent) + 1e-8
            tangent = tangent / tangent_norm

            offset = -amplitude * weight * tangent
            p_new = pos + offset
            p_new = torch.stack([p_new[0], pos[1], p_new[2]], dim=0)

            phi_new = torch.atan2(p_new[2], p_new[0])
            r_new = torch.sqrt(p_new[0] * p_new[0] + p_new[2] * p_new[2] + 1e-8)
            r_max = _interp_max_r(phi_new)
            if r_new > r_max:
                p_new = torch.stack(
                    [torch.cos(phi_new) * r_max, pos[1], torch.sin(phi_new) * r_max],
                    dim=0,
                )

            traj_points.append(p_new)
            perturbation_pose_dict[key] = sampler.sample_pose(p_new)
            key += 1

        os.makedirs(self.save_path, exist_ok=True)
        traj_path = os.path.join(self.save_path, "traj.txt")
        if len(traj_points) > 0:
            traj_arr = torch.stack(traj_points, dim=0).detach().cpu().numpy()
            np.savetxt(traj_path, traj_arr, fmt="%.6f")
        return perturbation_pose_dict

    def scene_inpainting(self, init_depth, pose_conf):
        # Global Inpainting
        print(f"Global Inpainting...")
        globale_sampler = CirclePoseSampler(init_depth, **pose_conf)
        global_cameras = globale_sampler.anchor_pts # [N, 3]
        global_cameras = self.xyz_to_xz_y(global_cameras)
        global_pose_dict = {}
        key = 0
        for pos in global_cameras:
            global_pose_dict[key] = globale_sampler.sample_pose(pos)
            key += 1
        inpainted_panos_and_poses = self.stage_inpaint_pano_greedy_search(global_pose_dict, init_depth)

    def roaming(self):
        self.load_modules()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # load pano
        print(f"Loading Pano and Estimating Depth...")
        pano_rgb, pano_depth = self.load_pano()
        panorama_tensor, init_depth = pano_rgb.squeeze(0).cuda(), pano_depth.cuda()
        depth_edge = self.find_depth_edge(init_depth.cpu().detach().numpy(), dilate_iter=1)
        depth_edge_pil = Image.fromarray(depth_edge)
        depth_pil = Image.fromarray(visualize_depth_numpy(init_depth.cpu().detach().numpy())[0].astype(np.uint8))
        _, _ = save_rgbd(depth_pil, depth_edge_pil, f'depth_edge', 0, self.save_path)  
        depth_edge_inpaint_mask = ~(torch.from_numpy(depth_edge).cuda().bool()) 

        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(pose=torch.eye(4).cuda(),
                                        mask=torch.ones([self.pano_height, self.pano_width]),
                                        rgb=panorama_tensor.permute(1,2,0),
                                        distance=init_depth.unsqueeze(-1))
        self.sup_pool.gen_occ_grid(256)

        # print(f"Instance-aware Scene Modeling...")
        # segment
        self.pano_segment()

        # Pano2Mesh
        self.pano_distance_to_mesh(panorama_tensor, init_depth, depth_edge_inpaint_mask)

        self.instance_centers, self.instance_bbx8 = self.decouple_instances()
        # 保存 instance_bbx8 到 instances 目录下的 bbx.txt
        if self.instance_bbx8 is not None and self.instance_bbx8.numel() > 0:
            bbx_path = os.path.join(self.save_path, "instances", "bbx.txt")
            np.savetxt(bbx_path, self.instance_bbx8.cpu().numpy().reshape(-1, 3), fmt="%.6f")
            print(f"已保存 instance_bbx8 到 {bbx_path}")

        # Global Inpainting
        for i in range(len(self.global_pose_sampler_conf_list)): 
            self.scene_inpainting(pano_depth, self.global_pose_sampler_conf_list[i])

        # Object-aware Inpainting
        print(f"Object-aware Inpainting...")
        perturbation_sampler = CirclePoseSampler(pano_depth, **self.perturbation_pose_sampler_conf)
        perturbation_cameras = perturbation_sampler.anchor_pts # [N, 3]
        perturbation_cameras = self.xyz_to_xz_y(perturbation_cameras)
        max_sampler = CirclePoseSampler(pano_depth, **self.max_pose_sampler_conf)
        max_cameras = max_sampler.anchor_pts # [N, 3]
        max_cameras = self.xyz_to_xz_y(max_cameras)
        perturbation_pose_dict = self.traj_gen(perturbation_cameras, perturbation_sampler, max_cameras)
        inpainted_panos_and_poses = self.stage_inpaint_pano_greedy_search(perturbation_pose_dict, init_depth)

        panorama_pil = functions.tensor_to_pil(panorama_tensor.unsqueeze(0))
        panorama_pil.save(f"{self.save_path}/panorama_tensor.png")
        
        print(f"Saved panorama_tensor.png to {self.save_path}")

    def create_3DGS(self, inpainted_panos_and_poses):
        # Train 3DGS
        self.opt = GSParams()
        self.cam = CameraParams()
        self.gaussians = GaussianModel(self.opt.sh_degree)
        self.opt.white_background = True
        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': self.W,
            'H': self.H,
            'pcd_points': self.vertices.detach().cpu(),
            'pcd_colors': self.colors.permute(1,0).detach().cpu(),
            'frames': [],
        }
        for inpainted_pano_images, pano_pose_44 in inpainted_panos_and_poses:
            cubemaps, cubemaps_depth = self.pano_to_cubemap(inpainted_pano_images) # BCHW
            for i in range(len(cubemaps)):
                inpainted_img = cubemaps[i] 

                mesh_pose = self.cubemap_w2c_list[i].cuda() @ pano_pose_44.clone()

                pose_44 = mesh_pose.clone()
                pose_44 = pose_44.float()
                pose_44[0:1,:] *= -1
                pose_44[1:2,:] *= -1

                Rw2c = pose_44[:3,:3].cpu().numpy()
                Tw2c = pose_44[:3,3:].cpu().numpy()
                yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

                Rc2w = np.matmul(yz_reverse, Rw2c).T
                Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
                Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)  #化为齐次矩阵

                traindata['frames'].append({
                    'image': functions.tensor_to_pil(inpainted_img),
                    'transform_matrix': Pc2w.tolist(), 
                    'fovx': focal2fov(self.H / 2, inpainted_img.shape[-1]),
                    'mesh_pose': mesh_pose
                })

        self.scene = Scene(traindata, self.gaussians, self.opt)   
        self.train_GS()
        outfile = self.gaussians.save_ply(os.path.join(self.GS_render_dir, '3DGS.ply'))

        # Eval GS
        self.pano_pose, self.poses = self.load_camera_poses(self.pano_center_offset)
        evaldata = {
            'camera_angle_x': self.cam.fov[0],
            'W': self.W,
            'H': self.H,
            'frames': [],
        }

        for i in range(len(self.poses)):
            gt_img = inpainted_img

            pose_44 = self.poses[i].clone()
            pose_44 = pose_44.float()
            pose_44[0:1,:] *= -1
            pose_44[1:2,:] *= -1

            Rw2c = pose_44[:3,:3].cpu().numpy()
            Tw2c = pose_44[:3,3:].cpu().numpy()
            yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)                  

            evaldata['frames'].append({
                'image': functions.tensor_to_pil(gt_img),
                'transform_matrix': Pc2w.tolist(), 
                'fovx': focal2fov(self.H / 2, self.W),
                'mesh_pose': self.poses[i].clone()
            })
        eval_GS_cams = loadCamerasFromData(evaldata, self.opt.white_background)
        self.eval_GS(eval_GS_cams)
        self.GS2pano_roampose(inpainted_panos_and_poses, self.gaussians, self.opt, self.background)
