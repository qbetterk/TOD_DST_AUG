#!/bin/bash
#
set -xue

if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=6
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-scratch1/data/qkun/tod_aug/

elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=6
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/tod_aug/
fi

model_card_or_path=ckpt_aug/flan-t5-large/MULTIWOZ2_2/utt
# model_card_or_path=google/flan-t5-base

# python dst_aug.py \
#         --data_name MULTIWOZ2_2 \
#         --model_card_or_path ${model_card_or_path} \
#         --act aug \
#         --batch_size 32

python dst_aug.py \
        --data_name MULTIWOZ2_2 \
        --act proc