#!/bin/bash

# 支持从命令行参数传入 scene_name；若未传入则使用默认值
scene_name=${1:-'3'}

export HF_ENDPOINT='https://hf-mirror.com'


echo "Activating roamscene3d environment..."
conda deactivate
conda activate roamscene3d

echo "Running roaming.py..."
python roaming.py \
    --scene_name ${scene_name}

echo "Switching to layerpano3d environment..."
conda activate layerpano3d   
echo "Running pano_SR.py..."
python ./pasd/panoSR.py --inputs_dir "./output/${scene_name}" --decoder_tiled_size 448 --encoder_tiled_size 4096 --num_inference_steps 20 --latent_tiled_size 128 --latent_tiled_overlap 4
else
    echo "Warning: LayerPano3D directory not found, skipping super-resolution step..."
fi
conda deactivate

conda activate roamscene3d
echo "Running creating.py..."
python creating.py \
    --scene_name ${scene_name}

echo "Pipeline completed!"
