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
        ckpt_path = os.environ.get('ROAM_MOTION_CHECKPOINT',
                                  os.path.join(SDFT_path, f'checkpoint-step-{step}', 'unet.pth'))
        if model != 'origin' and not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Missing trained motion-inpainting weights: {ckpt_path}. '
                                    'Set ROAM_MOTION_CHECKPOINT; random motion weights are not a valid fallback.')
        args = Args()
        if model == 'wocam':
            UNet = UNetWOCam(args)
        elif model == 'withcam':
            UNet = UNetWithCam(args)
        # SDFT_path = f"output/lora1"
        if not model == 'origin':
            state_dict = torch.load(ckpt_path, map_location='cpu')
            try:
                UNet.load_state_dict(state_dict)
            except RuntimeError as error:
                raise RuntimeError('Checkpoint must match the six-DoF positional encoder. '
                                   'Legacy position+face-direction weights require retraining.') from error
            UNet.to('cuda').eval()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(args.pretrained_model_name_or_path, local_files_only=True, variant="fp16").to("cuda") # torch_dtype=torch.float16
        if not model == 'origin':
            pipe.unet = UNet
        self.inpaint_pipe = pipe

    def _run_with_motion(self, pose, **kwargs):
        # StableDiffusionInpaintPipeline has no `pose` argument. The wrapper UNet
        # consumes this temporary condition on every denoising step.
        self.inpaint_pipe.unet.motion_condition = pose
        try:
            return self.inpaint_pipe(**kwargs)
        finally:
            self.inpaint_pipe.unet.motion_condition = None
        

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
        prompt = label
        generator = torch.Generator(device="cuda").manual_seed(torch.initial_seed())

        inpainted_image_pil = self._run_with_motion(pose,
        prompt=prompt,
        image=rendered_image_pil,
        mask_image=inpaint_mask_pil,
        guidance_scale=7.5,
        num_inference_steps=30,  
        generator=generator,
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
        generator = torch.Generator(device="cuda").manual_seed(torch.initial_seed())
        # prompt 与 batch 对齐，避免 2*B vs B 的维度不一致
        prompt = [label] * B
        result_pil_list = self._run_with_motion(poses_b,
            prompt=prompt,
            image=images_pil,
            mask_image=masks_pil,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator
        ).images
        result_tensors = [functions.pil_to_tensor(pil_img) for pil_img in result_pil_list]
        return torch.cat(result_tensors, dim=0).to(torch.float32)
