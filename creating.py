import argparse
from pipeline import GenerationPipeline
import time

def main():
    parser = argparse.ArgumentParser(description='Run creating pipeline')
    parser.add_argument('--scene_name', type=str, default='livingroom', 
                       help='Name of the scene to process (default: kitchen)')
    
    args = parser.parse_args()
    scene_name = args.scene_name
    
    pipeline = GenerationPipeline(scene_name=scene_name)
    pipeline.H, pipeline.W = 1024, 1024
    inpainted_panos_and_poses = pipeline.load_inpainted_panos_and_poses()
    pipeline.load_vertices_and_colors()
    pipeline.create_3DGS(inpainted_panos_and_poses)

if __name__ == '__main__':
    main()