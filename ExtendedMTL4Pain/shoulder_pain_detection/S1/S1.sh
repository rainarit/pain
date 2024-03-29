#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 20220 \
            --use_env detect_pscore_script.py
