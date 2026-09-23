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
from utils.scene_geometry import plan_trajectory, relative_motion, support_pose, evaluation_poses
from utils.semantic_grounding import validate_graph, guidance_text
# from SceneGraph import SceneGraph
from pano_semantic_segment import PanoSemanticSegmentor
from modules.pose_sampler.circle_pose_sampler import CirclePoseSampler
import warnings
warnings.filterwarnings("ignore")
import random
from scene.dataset_readers import loadCamerasFromData
from utils.projection import *

        
class GenerationPipeline(torch.nn.Module):
    def __init__(self, scene_name, attempt_idx="", seed=42):
        '''initialize models and define shared variables'''

        super().__init__()

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        self.anchor_ids = []
        self.bindings = {}
        self.verified_relations = []
        self.trajectory_focus = {}

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
        self.n_roaming_views = 24
        self.base_trajectory_gamma = 0.4

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

    @torch.no_grad()
    def stage_inpaint_pano_greedy_search(self, pose_dict, source_depth, add_mesh=True):
        """Traverse the closed path in order, filter and fuse every new observation."""
        inpainted_panos_and_poses = []
        source_pose = torch.eye(4, device=self.device)
        for key, pose in pose_dict.items():
            pose = pose.to(self.device)
            motion = self.cam_scale(source_depth, pose, source_pose)
            pano_rgb, pano_distance, pano_mask = self.render_pano(pose)
            colors = pano_rgb.permute(1, 2, 0).clone()
            distances = pano_distance.clone()
            missing = pano_mask > .5
            perf_pose = torch.as_tensor(support_pose(pose.cpu().numpy()), device=pose.device, dtype=pose.dtype)
            if missing.any():
                focus = self.trajectory_focus.get(key)
                prompt = guidance_text(self.scene_graph, focus, self.bindings, self.verified_relations)
                colors, distances, normals = self.inpaint_new_panorama(
                    idx=key+1, colors=colors, distances=distances[..., None],
                    pano_mask=missing.float(), pose=motion, prompt=prompt)
                rays = gen_pano_rays(perf_pose, self.pano_height, self.pano_width)
                keep = missing & (self.sup_pool.geo_check(rays, distances[..., None]) > .5)
                keep &= torch.isfinite(distances) & (distances > 1e-5)
                if keep.any():
                    if add_mesh:
                        self.pano_distance_to_mesh(colors.permute(2, 0, 1), distances, keep, pose=pose)
                    self.sup_pool.register_sup_info(pose=perf_pose, mask=keep, rgb=colors,
                                                    distance=distances[..., None], normal=normals)
            self.namer += 1
            functions.tensor_to_pil(pano_rgb[None]).save(f"{self.save_path}/renderred_pano_{self.namer}.png")
            if self.save_details:
                Image.fromarray((missing.cpu().numpy()*255).astype(np.uint8)).save(f"{self.save_path}/mask_{self.namer}.png")
            functions.tensor_to_pil(colors.permute(2, 0, 1)[None]).save(f"{self.save_path}/inpainted_pano_{self.namer}.png")
            torch.save(pose.cpu(), f"{self.save_path}/pose_{self.namer}.pt")
            inpainted_panos_and_poses.append((colors.permute(2, 0, 1)[None].cpu(), pose.cpu()))
            source_pose = pose
        return inpainted_panos_and_poses

    def cam_scale(self, source_depth, target_pose, source_pose):
        # Express the same physical transition in each source cubemap camera's
        # coordinates: forward motion for one face is lateral for another.
        source = source_pose.cpu().numpy()
        target = target_pose.cpu().numpy()
        distance = source_depth.cpu().numpy()
        motion = np.stack([relative_motion(face.cpu().numpy() @ source,
                                           face.cpu().numpy() @ target, distance)
                           for face in self.cubemap_w2c_list])
        return torch.as_tensor(motion, device=target_pose.device, dtype=torch.float32)

    def inpaint_new_panorama(self, idx, colors, distances, pano_mask, pose, prompt=""):
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
        inpainted_img = self.inpainter.inpaint(idx, colors, mask, pose, label=prompt)

        # Keep renderred part
        inpainted_img = colors * (1 - mask) + inpainted_img * mask# 仅改变掩码部分
        inpainted_img = inpainted_img.cuda()

        inpainted_distances, inpainted_normals = self.geo_predictor(idx,
                                                                    inpainted_img,
                                                                    distances,
                                                                    mask=mask,
                                                                    reg_loss_weight=0.1,
                                                                    normal_loss_weight=0.01,
                                                                    normal_tv_loss_weight=0.01)
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
        image = Image.open(image_path).convert('RGB').resize((self.pano_width, self.pano_height))
        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).float()/255

        depth = self.predict_depth(panorama_tensor)
        
        return panorama_tensor, depth
    
    def predict_depth(self, panorama_tensor):
        image = panorama_tensor.permute(1, 2, 0).to(self.device)
        depth, self.initial_normals = self.geo_predictor(
            0, image, torch.ones_like(image[..., :1]), torch.ones_like(image[..., :1]))
        return depth.squeeze(-1)

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

    @torch.enable_grad()
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
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
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

    @torch.no_grad()
    def eval_GS(self, cameras, mode='geometry'):
        outdir = os.path.join(self.GS_render_dir, mode)
        os.makedirs(outdir, exist_ok=True)
        frames = []
        for i, (camera, _) in enumerate(cameras):
            result = render(camera, self.gaussians, self.opt, self.background)
            rgb = (result['render'].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(outdir, f'{i:03d}.png'))
            if mode == 'geometry':
                # Preserve quantitative depth; a colormap is not an RGBD sequence.
                np.save(os.path.join(outdir, f'{i:03d}_depth.npy'), result['depth'].detach().cpu().numpy().squeeze())
                frames.append(rgb)
        if frames:
            write_video(os.path.join(outdir, 'rgb.mp4'), frames, fps=30)

    def evaluate_views(self):
        from scene.cameras import Camera
        distance = np.load(os.path.join(self.save_path, 'initial_distance.npy'))
        for mode in ('appearance', 'geometry'):
            poses = evaluation_poses(distance, mode, seed=self.seed)
            np.save(os.path.join(self.GS_render_dir, f'{mode}_poses.npy'), poses)
            cameras = []
            for i, pose in enumerate(poses):
                cv_pose = np.diag([-1., -1., 1., 1.]) @ pose
                camera = Camera(colmap_id=i, R=cv_pose[:3, :3].T, T=cv_pose[:3, 3],
                                FoVx=np.pi/2, FoVy=np.pi/2, image=torch.zeros(3, 512, 512),
                                gt_alpha_mask=None, image_name=str(i), uid=i, data_device=self.device)
                cameras.append((camera, pose))
            self.eval_GS(cameras, mode)

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
        from plyfile import PlyData
        filename = filename or os.path.join(self.save_path, 'fused_scene.ply')
        vertices = PlyData.read(filename)['vertex']
        xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
        rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.float32)/255.
        if not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError('Fused mesh must contain finite vertices')
        self.vertices = torch.from_numpy(xyz.T.copy()).to(self.device)
        self.colors = torch.from_numpy(rgb.T.copy()).to(self.device)

    def load_inpainted_panos_and_poses(self):
        inpainted_panos_and_poses = []
        sr_dir = self.save_path
        for item in sorted(os.listdir(sr_dir), key=lambda name: int(name.rsplit('_', 1)[-1].split('.')[0]) if name.startswith('sr_2_inpainted_pano_') else -1):
            if item.startswith('sr_2_inpainted_pano_'):
                num = item.split('_')[-1].split('.')[0]
                # print(num)
                inpainted_pano = Image.open(f"{sr_dir}/{item}")
                inpainted_pano = functions.pil_to_tensor(inpainted_pano)
                # print(f"{self.save_path}/pose_{num}.pt")
                pose = torch.load(f"{self.save_path}/pose_{num}.pt", map_location=self.device).float()
                inpainted_panos_and_poses += [(inpainted_pano, pose)]
        if len(inpainted_panos_and_poses) != 24:
            raise ValueError(f"Expected 24 PASD panoramas, got {len(inpainted_panos_and_poses)}; rerun roaming and SR")
        return inpainted_panos_and_poses

    def load_scene_graph(self, filepath):
        with open(filepath) as file:
            self.scene_graph = validate_graph(json.load(file))

    def pano_segment(self, depth, panorama):
        self.load_scene_graph(f"{self.input_dir}/scene_graphs/{self.scene_name}.json")
        segmentor = PanoSemanticSegmentor(f"{self.input_dir}/scene_panoramas/{self.scene_name}.png", self.scene_graph)
        image = (panorama.permute(1, 2, 0).cpu().numpy()*255).clip(0, 255).astype(np.uint8)
        seg_map, self.instance_num = segmentor.segment(f"{self.save_path}/instances", depth.cpu().numpy(), image)
        self.anchor_ids = segmentor.anchor_ids
        self.bindings = segmentor.bindings
        self.verified_relations = segmentor.verified_relations
        self.labels = torch.from_numpy(seg_map).flatten()
        centers, boxes = [], []
        from itertools import product
        for geometry in getattr(segmentor, 'anchor_geometries', []):
            centers.append(geometry['center'])
            boxes.append(list(product(*zip(geometry['min'], geometry['max']))))
        self.instance_centers = torch.as_tensor(np.asarray(centers).reshape(-1, 3), device=self.device, dtype=torch.float32)
        self.instance_bbx8 = torch.as_tensor(np.asarray(boxes).reshape(-1, 8, 3), device=self.device, dtype=torch.float32)
        np.save(f"{self.save_path}/instances/seg_map.npy", seg_map)
        self.object_aware = bool(self.anchor_ids)
        del segmentor
        torch.cuda.empty_cache()

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

    def traj_gen(self, distance):
        points, focus, knots = plan_trajectory(
            distance.cpu().numpy(), self.instance_centers.cpu().numpy(), self.instance_bbx8.cpu().numpy(),
            n_views=self.n_roaming_views, gamma=self.base_trajectory_gamma)
        poses = {}
        for i, (point, anchor) in enumerate(zip(points, focus)):
            pose = torch.eye(4, device=self.device)
            # The renderer accepts world-to-camera transforms, not camera centers.
            pose[:3, 3] = -torch.as_tensor(point, device=self.device, dtype=pose.dtype)
            poses[i] = pose
            self.trajectory_focus[i] = None if anchor is None else self.anchor_ids[anchor]
        np.savetxt(os.path.join(self.save_path, 'traj.txt'), points)
        np.savetxt(os.path.join(self.save_path, 'trajectory_knots.txt'), knots)
        with open(os.path.join(self.save_path, 'trajectory_focus.json'), 'w') as f:
            json.dump(self.trajectory_focus, f, indent=2)
        return poses

    @torch.no_grad()
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
                                        distance=init_depth.unsqueeze(-1), normal=self.initial_normals)

        # print(f"Instance-aware Scene Modeling...")
        # segment
        self.pano_segment(init_depth, panorama_tensor)

        # Pano2Mesh
        self.pano_distance_to_mesh(panorama_tensor, init_depth, depth_edge_inpaint_mask)

        # 保存 instance_bbx8 到 instances 目录下的 bbx.txt
        if self.instance_bbx8 is not None and self.instance_bbx8.numel() > 0:
            bbx_path = os.path.join(self.save_path, "instances", "bbx.txt")
            np.savetxt(bbx_path, self.instance_bbx8.cpu().numpy().reshape(-1, 3), fmt="%.6f")
            print(f"已保存 instance_bbx8 到 {bbx_path}")

        pose_dict = self.traj_gen(init_depth)
        self.stage_inpaint_pano_greedy_search(pose_dict, init_depth)
        # Persist the fused mesh used to initialize GS, without visualization boxes.
        import trimesh
        trimesh.Trimesh(vertices=self.vertices.T.cpu().numpy(), faces=self.faces.T.cpu().numpy(),
                        vertex_colors=(self.colors.T.cpu().numpy()*255).clip(0, 255).astype(np.uint8),
                        process=False).export(os.path.join(self.save_path, 'fused_scene.ply'))
        np.save(os.path.join(self.save_path, 'initial_distance.npy'), init_depth.cpu().numpy())

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

        self.evaluate_views()
