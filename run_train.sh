#!/bin/bash
#
set -xue
# # ****************************** script for communication ****************************** 
# export CUDA_VISIBLE_DEVICES=0
# export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
# export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
# export HF_HOME=/local-storage/data/qkun/huggingface/

# python train_dst.py \
#         --model_name_or_path google/flan-t5-base \
#         --do_eval \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --output_dir /local-scratch1/data/qkun/tod_aug/ckpt_aug \
#         --per_device_train_batch_size=16 \
#         --per_device_eval_batch_size=4 \
#         --overwrite_output_dir \
#         --predict_with_generate

# ****************************** script for coffee ****************************** 
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
export HF_CACHE_HOME=/local/data/shared/huggingface_cache
export HF_HOME=/local/data/shared/huggingface_cache/

# # # train with original data
model_name=flan-t5-large
dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
aug_model=flan-t5-xxl
# aug_model=ori
python train_dst.py \
        --model_name_or_path google/${model_name} \
        --dataset_name ${dataset_name} \
        --aug_model ${aug_model} \
        --output_dir /local/data/qkun/tod_aug/ckpt_aug/${model_name}/${dataset_name}/${aug_model} \
        --do_train \
        --do_eval \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=4 \
        --predict_with_generate \
        --auto_find_batch_size \
        --load_best_model_at_end \
        --evaluation_strategy epoch \
        --metric_for_best_model jga \
        --save_total_limit 10 \
        # --overwrite_output_dir \
        # --debug_mode True \
        # --log_level warning


# python train_dst.py \
#         --model_name_or_path /local/data/qkun/tod_aug/ckpt_aug/${model_name}/${dataset_name}/${aug_model} \
#         --dataset_name ${dataset_name} \
#         --aug_model ${aug_model} \
#         --output_dir /local/data/qkun/tod_aug/ckpt_aug/${model_name}/${dataset_name}/${aug_model} \
#         --do_eval \
#         --num_train_epochs 10 \
#         --save_strategy epoch \
#         --per_device_train_batch_size=16 \
#         --per_device_eval_batch_size=16 \
#         --predict_with_generate \
#         --debug_mode True
