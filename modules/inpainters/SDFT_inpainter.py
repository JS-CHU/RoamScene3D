import utils.functions as functions
import torch
from .inpainter import Inpainter
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import os
import json
from models.UNet_Camtext import *

class Args:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting/"
    revision = None
    variant = None
    cam_latent_dim = 256
    use_learnable_pose_token = True
    rank = 8
    enable_lora = True
    mixed_precision = None
    enable_xformers_memory_efficient_attention = True


class SDFTInpainter(Inpainter):
    def __init__(self, model, SDFT_path, step):
        super().__init__()
        args = Args()
        if model == 'wocam':
            UNet = UNetWOCam(args)
        elif model == 'withcam':
            UNet = UNetWithCam(args)
        # SDFT_path = f"output/lora1"
        if not model == 'origin':
            if os.path.exists(SDFT_path):
                ckpt_path = os.path.join(SDFT_path, f'checkpoint-step-{step}', 'unet.pth')
                state_dict = torch.load(ckpt_path, map_location='cuda')
                UNet.load_state_dict(state_dict)
            UNet.to('cuda')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(args.pretrained_model_name_or_path, local_files_only=True, variant="fp16").to("cuda") # torch_dtype=torch.float16
        if not model == 'origin':
            pipe.unet = UNet
        # 绕过 __setattr__，直接在 __dict__ 中补齐执行设备属性
        try:
            pipe.__dict__['_execution_device'] = torch.device("cuda")
            pipe.__dict__['device'] = torch.device("cuda")
        except Exception:
            pass
        self.inpaint_pipe = pipe
        

    @torch.no_grad()
    def inpaint(self, img, mask, pose, label=''): 
        '''
        :param img: B C H W?
        :param mask: 
        :return:
        '''
        inpaint_mask_pil = Image.fromarray(mask.detach().cpu().squeeze(0).squeeze(0).float().numpy() * 255).convert("RGB")
  
        rendered_image_pil = functions.tensor_to_pil(img)
        
        # prompt_generator = Prompt(self.SceneGraph_path, label)
        # prompt = prompt_generator.prompt
        prompt = ''
        generator = torch.Generator(device="cuda").manual_seed(0)

        inpainted_image_pil = self.inpaint_pipe(
        prompt=prompt,
        image=rendered_image_pil,
        mask_image=inpaint_mask_pil,
        guidance_scale=7.5,
        num_inference_steps=30,  
        generator=generator,
        pose = pose
        ).images[0]
        result = functions.pil_to_tensor(inpainted_image_pil)

        return result.to(torch.float32)
        
    @torch.no_grad()
    def inpaint_batch(self, imgs_b, masks_b, poses_b, label=''):
        B = imgs_b.shape[0]
        images_pil = []
        masks_pil = []
        for b in range(B):
            mask_pil = Image.fromarray((masks_b[b].detach().cpu().squeeze(0).float().numpy() * 255).astype(np.uint8)).convert("RGB")
            pil_img = functions.tensor_to_pil(imgs_b[b][None])
            images_pil.append(pil_img)
            masks_pil.append(mask_pil)
        generator = torch.Generator(device="cuda").manual_seed(0)
        # prompt 与 batch 对齐，避免 2*B vs B 的维度不一致
        prompt = [''] * B
        result_pil_list = self.inpaint_pipe(
            prompt=prompt,
            image=images_pil,
            mask_image=masks_pil,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
            pose=poses_b
        ).images
        result_tensors = [functions.pil_to_tensor(pil_img) for pil_img in result_pil_list]
        return torch.cat(result_tensors, dim=0).to(torch.float32)
    