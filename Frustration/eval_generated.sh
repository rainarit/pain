#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# value="DarkSkin"

# declare -a arr=("/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteFalse_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteFalse_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteTrue_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_2shoulder_zscore_hsv_diffexpFalse_excludewhiteTrue_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_2shoulder_zscore_hsv_diffexpFalse_excludewhiteTrue_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean-62.1796std210.06_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean-62.1796std210.06_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean100.0264std40.4236_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean100.0264std40.4236_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean100.8909std45.773_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpFalse_excludewhiteTrue_mean100.8909std45.773_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean-62.1796std210.06_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean-62.1796std210.06_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean100.0264std40.4236_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean100.0264std40.4236_background255"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean100.8909std45.773_background0"
# "/home/AD/rraina/pain/Frustration/generated_faces/${value}_cropped_custom_zscore_hsv_diffexpTrue_excludewhiteTrue_mean100.8909std45.773_background255")

# for comm in "${arr[@]}"
# do
#         python3 -m torch.distributed.launch \
#                 --nproc_per_node=1 \
#                 --master_port 20220 \
#                 --use_env eval_generated.py \
#                 --data $comm
# done

python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteFalse_background0/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteFalse_background255/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteTrue_background0/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_histmatch_hsv_diffexpFalse_excludewhiteTrue_background255/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_zscore_hsv_diffexpFalse_excludewhiteTrue_background0/
python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port 20220 \
                --use_env eval_generated_simple.py \
                --data /home/AD/rraina/pain/Frustration/generated_faces/8.6_darken0.5_2shoulder_zscore_hsv_diffexpFalse_excludewhiteTrue_background255/
