#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python3 -m torch.distributed.launch \
#             --nproc_per_node=1 \
#             --master_port 20220 \
#             --use_env crop_image.py \
#             --data '/home/AD/rraina/pain/Frustration/generated_faces/AU25_experiments/morpha_cropMat/' \


python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 20220 \
            --use_env change_contrast_image.py \
            --data '/home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5/' \
            --mode '2shoulder' \
            --method 'histmatch' \
            --space 'hsv' \
            --backgroundcolor 255 \