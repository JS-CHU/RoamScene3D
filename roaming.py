import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
from pipeline import GenerationPipeline
import time
import shutil


def main():
    parser = argparse.ArgumentParser(description='Run roaming pipeline')
    parser.add_argument('--scene_name', type=str, default='3', 
                       help='Name of the scene to process (default: kitchen)')
    
    args = parser.parse_args()
    scene_name = args.scene_name
    save_path = f'./output/{scene_name}'
    pipeline = GenerationPipeline(scene_name=scene_name)
    pipeline.set_save_path(save_path)
    pipeline.set_sampler()
    pipeline.roaming()
    # pipeline.save_initial_scene()

if __name__ == '__main__':
    main()
    