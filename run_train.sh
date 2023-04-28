#!/bin/bash
#
set -xue


action=train_dst_model 
action=finetune_aug_model


if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=2
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-scratch1/data/qkun/tod_aug/
elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/tod_aug/
fi
if [ $action == "train_dst_model" ]; then
        # # # train dst model with ori/aug data
        model_name=flan-t5-base
        dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
        ft_method=utt_nodst                   # utt, utt_nodst, utt_instruct
        constraint=no_constraint_decoding  # constraint_decoding, no_constraint_decoding
        # aug_model=flan-t5-large         # "ori", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"
        aug_model=ori                   # "ori", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"
        mode=train
        output_dir=${proj_path}/ckpt_dst/${model_name}/${dataset_name}/${aug_model}
        if [ $aug_model == "ori" ]; then
                data_path=${proj_path}/${dataset_name}/ori_data
        else
                data_path=${proj_path}/${dataset_name}/aug_data/${ft_method}/${constraint}/${aug_model}
        fi
        if [ $mode == "train" ]; then
                python train_dst.py \
                        --model_name_or_path google/${model_name} \
                        --dataset_name ${dataset_name} \
                        --aug_model ${aug_model} \
                        --output_dir ${output_dir} \
                        --proj_path ${proj_path} \
                        --data_path ${data_path} \
                        --do_train \
                        --do_eval \
                        --num_train_epochs 10 \
                        --save_strategy epoch \
                        --per_device_train_batch_size=32 \
                        --per_device_eval_batch_size=4 \
                        --auto_find_batch_size \
                        --load_best_model_at_end \
                        --evaluation_strategy epoch \
                        --metric_for_best_model jga \
                        --save_total_limit 3 \
                        --overwrite_output_dir
        else
                python train_dst.py \
                        --model_name_or_path ${output_dir} \
                        --dataset_name ${dataset_name} \
                        --aug_model ${aug_model} \
                        --output_dir ${output_dir} \
                        --proj_path ${proj_path} \
                        --do_predict \
                        --per_device_train_batch_size=16 \
                        --per_device_eval_batch_size=2 \
                        --predict_with_generate
        fi
elif [ $action == "finetune_aug_model" ]; then
        # # # finetune the aug model with user utterance data
        model_name=flan-t5-base
        dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
        aug_model=utt                   # utt, utt_nodst, utt_instruct
        aug_model=utt_instruct
        output_dir=${proj_path}/ckpt_aug/${model_name}/${dataset_name}/${aug_model}
        python train_dst.py \
                --model_name_or_path google/${model_name} \
                --dataset_name ${dataset_name} \
                --output_dir ${output_dir} \
                --proj_path ${proj_path} \
                --aug_model ${aug_model} \
                --do_train \
                --do_eval \
                --num_train_epochs 16 \
                --save_strategy epoch \
                --auto_find_batch_size \
                --load_best_model_at_end \
                --evaluation_strategy epoch \
                --metric_for_best_model loss \
                --load_best_model_at_end \
                --save_total_limit 3 \
                --per_device_train_batch_size=64 \
                --overwrite_output_dir
fi