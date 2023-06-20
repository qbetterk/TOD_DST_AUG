#!/bin/bash
#
set -xue


action=train_dst_model 
# action=finetune_aug_model
version=2


if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=3
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-scratch1/data/qkun/tod_aug
elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=7
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/tod_aug
fi
if [ $action == "train_dst_model" ]; then
        # # # train dst model with ori/aug data
        model_name=flan-t5-xl
        dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
        ft_method=utt_instruct                   # no_ft utt utt_nodst utt_instruct
        constraint=constraint_decoding  # constraint_decoding no_constraint_decoding constraint_decoding_v3
        aug_model=flan-t5-base         # "ori", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"
        # aug_model=ori                   # "ori", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"
        dst_acc=True
        mode=test
        for aug_model in flan-t5-base flan-t5-large; do
                for constraint in constraint_decoding; do
                        for ft_method in utt utt_nodst utt_instruct ; do
                # for dst_acc in True False ; do
                                data_ori_dir=dataset/${dataset_name}/ori_data/
                                if [ ${dst_acc} == "True" ]; then
                                        if [ $aug_model == "ori" ]; then
                                                data_path=${data_ori_dir}
                                                output_dir=${proj_path}/ckpt_dst_${version}/${model_name}/${dataset_name}/${aug_model}--dst_acc
                                        else
                                                data_path=dataset/${dataset_name}/aug_data/${ft_method}/${constraint}/${aug_model}
                                                output_dir=${proj_path}/ckpt_dst_${version}/${model_name}/${dataset_name}/${ft_method}--${constraint}--${aug_model}--dst_acc
                                        fi
                                else
                                        if [ $aug_model == "ori" ]; then
                                                data_path=${data_ori_dir}
                                                output_dir=${proj_path}/ckpt_dst_${version}/${model_name}/${dataset_name}/${aug_model}
                                        else
                                                data_path=dataset/${dataset_name}/aug_data/${ft_method}/${constraint}/${aug_model}
                                                output_dir=${proj_path}/ckpt_dst_${version}/${model_name}/${dataset_name}/${ft_method}--${constraint}--${aug_model}
                                        fi

                                fi
                                if [ ! -d "$output_dir" ] || [ ! -d "$data_path" ]; then
                                        continue
                                fi
                                if [ $mode == "train" ]; then
                                        python train_dst.py \
                                                --model_name_or_path google/${model_name} \
                                                --dataset_name ${dataset_name} \
                                                --aug_model ${aug_model} \
                                                --output_dir ${output_dir} \
                                                --proj_path ${proj_path} \
                                                --data_path ${data_path} \
                                                --data_version ${version} \
                                                --data_ori_dir ${data_ori_dir} \
                                                --dst_acc ${dst_acc} \
                                                --do_train \
                                                --do_eval \
                                                --num_train_epochs 2 \
                                                --save_strategy epoch \
                                                --per_device_train_batch_size=32 \
                                                --gradient_accumulation_steps 8 \
                                                --per_device_eval_batch_size=32 \
                                                --learning_rate 1e-4 \
                                                --auto_find_batch_size \
                                                --load_best_model_at_end \
                                                --evaluation_strategy epoch \
                                                --metric_for_best_model jga \
                                                --save_total_limit 0 \
                                                --overwrite_output_dir \
                                                --predict_with_generate \
                                                --log_level warning
                                else
                                        python train_dst.py \
                                                --model_name_or_path ${output_dir} \
                                                --dataset_name ${dataset_name} \
                                                --aug_model ${aug_model} \
                                                --output_dir ${output_dir} \
                                                --proj_path ${proj_path} \
                                                --data_path ${data_path} \
                                                --data_version ${version} \
                                                --data_ori_dir ${data_ori_dir} \
                                                --dst_acc ${dst_acc} \
                                                --do_predict \
                                                --per_device_eval_batch_size=64 \
                                                --auto_find_batch_size \
                                                --predict_with_generate \
                                                --log_level warning
                                fi
                        done
                done
        done
elif [ $action == "finetune_aug_model" ]; then
        # # # finetune the aug model with user utterance data
        model_name=flan-t5-xl
        dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
        aug_model=utt_instruct                   # utt, utt_nodst, utt_instruct
        # aug_model=utt_instruct
        for aug_model in utt_instruct utt ; do
                data_ori_dir=dataset/${dataset_name}/ori_data/
                output_dir=${proj_path}/ckpt_aug_${version}/${model_name}/${dataset_name}/${aug_model}
                python train_dst.py \
                        --model_name_or_path google/${model_name} \
                        --dataset_name ${dataset_name} \
                        --output_dir ${output_dir} \
                        --proj_path ${proj_path} \
                        --aug_model ${aug_model} \
                        --data_ori_dir ${data_ori_dir} \
                        --do_train \
                        --do_eval \
                        --num_train_epochs 5 \
                        --save_strategy epoch \
                        --auto_find_batch_size \
                        --load_best_model_at_end \
                        --evaluation_strategy epoch \
                        --metric_for_best_model loss \
                        --load_best_model_at_end \
                        --save_total_limit 1 \
                        --per_device_train_batch_size 8 \
                        --gradient_accumulation_steps 8 \
                        --overwrite_output_dir
        done
fi