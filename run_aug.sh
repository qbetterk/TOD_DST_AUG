#!/bin/bash
#
set -xue

# export CUDA_VISIBLE_DEVICES=2
# export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
# export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
# export HF_HOME=/local-storage/data/qkun/huggingface/
# python dst_aug.py

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
export HF_CACHE_HOME=/local/data/shared/huggingface_cache
export HF_HOME=/local/data/shared/huggingface_cache/
python dst_aug.py \
        --data_name MULTIWOZ2_2 \
        --model_card google/flan-t5-small \
        --act proc