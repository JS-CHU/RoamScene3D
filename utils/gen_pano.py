import os
from PIL import Image

import sys
sys.path.append('stitch_diffusion/kohya_trainer')

from stitch_diffusion.kohya_trainer.StitchDiffusionPipeline import StitchDiffusion, my_args

def generate_pano(name, txtfile, save_dir):
    generator = StitchDiffusion(my_args)
    with open(txtfile) as f:
        prompt = f.read()
    pano_path = os.path.join(save_dir, f"{name}.png")
    generator.inference(prompt, savename=pano_path)

if __name__ == '__main__':
    generate_pano('bedroom', './input/scene_texts/bedroom.txt', './input/scene_panoramas')