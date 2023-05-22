#!/bin/bash
#
set -xue

if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=3
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-scratch1/data/qkun/tod_aug/

elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=4
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/tod_aug/
fi

# python dst_aug.py \
#         --save_dir dataset \
#         --data_name MULTIWOZ2_2 \
#         --act proc

aug_model=flan-t5-xl
dataset_name=MULTIWOZ2_2        # "SGD", "MULTIWOZ2_2"
ft_method=utt_instruct                   # no_ft utt utt_nodst utt_instruct
sample_new_value=True
constraint_folder=constraint_decoding  # constraint_decoding, no_constraint_decoding

for constraint_folder in constraint_decoding no_constraint_decoding; do
        for ft_method in utt utt_instruct; do
                model_card_or_path=${proj_path}/ckpt_aug_2/${aug_model}/MULTIWOZ2_2/${ft_method}
                if [ $sample_new_value == "True" ] && [ $constraint_folder == "constraint_decoding" ]; then
                        save_dir=dataset/${dataset_name}/aug_data/${ft_method}/constraint_decoding_v3/${aug_model}
                else
                        save_dir=dataset/${dataset_name}/aug_data/${ft_method}/${constraint_folder}/${aug_model}
                fi

                python dst_aug.py \
                        --data_name MULTIWOZ2_2 \
                        --model_card_or_path ${model_card_or_path} \
                        --save_dir ${save_dir} \
                        --constraint_folder ${constraint_folder} \
                        --act aug \
                        --version 2 \
                        --batch_size 32 \
                        --sample_new_value ${sample_new_value}
        done
done

