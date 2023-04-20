#!/usr/bin/env python3
#
import sys, os, pdb
import json
import random, argparse
from typing import Iterable, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from zero_shot.init_beam import get_init_candidate
# from zero_shot.generate import generate, BeamHypotheses
# from lexical_constraints import init_batch, ConstrainedHypothesis
# from zero_shot.topK import topk_huggingface
# from zero_shot.utils import tokenize_constraints
# from constraint.constraint_seeker import _generate
DST_SPLIT = " , "

class Augmentation(object):
    def __init__(self, args) -> None:
        self.data_dir = "/local/data/qkun/dataset/processed/Task-Oriented/"
        self.data_name = args.data_name
        self.save_dir = "/local/data/qkun/tod_aug/"
        self.batch_size = 1
        self.beam_size = 2
        self.model_card = args.model_card
        self.model_name = self.model_card.split("/")[0]


    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
            # return None
        with open(path) as df:
            data = json.loads(df.read())
        return data

    
    def _load_txt(self, path=None, split_tok="\n"):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            data = df.read().strip().split(split_tok)
        return data


    def _load_dir_json(self, dir_path=None):
        if dir_path is None or not os.path.exists(dir_path): 
            raise IOError('Folder does not exist: %s ...' % dir_path)
        total_data = [] # assume data is a list of dialogs
        print(f"Loading data from {dir_path} ...")
        for filename in os.listdir(dir_path):
            if filename in ["schema.json", "dialog_v0.json"]: continue
            if not filename.startswith("dialog"): continue
            if not filename.endswith(".json"): continue
            file_path = os.path.join(dir_path, filename)
            data = self._load_json(path=file_path)
            if type(data) == list:
                total_data.extend(data)
            elif type(data) == dict:
                total_data.extend(list(data.values()))
            else:
                total_data.append(data)
        return total_data


    def _save_json(self, data, dir_path, file_name):
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, file_name), "w") as tf:
            json.dump(data, tf, indent=2)
            print(f"saving {file_name} to {dir_path} ......")


    def string_to_dict(self, dst_str):
        dst_dict = {}
        for slot in dst_str.split(DST_SPLIT):
            if not slot: continue
            if len(slot.split()) < 2:
                pdb.set_trace()
            dom, slot_type = slot.split()[:2]
            if dom not in dst_dict:
                dst_dict[dom] = {}
            dst_dict[dom][slot_type] = " ".join(slot.split()[2:])
        return dst_dict


    def load_model(self):
        print(f"Loading model: {self.model_card} ......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card)
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.cuda.empty_cache()
        self.model.eval()
        self.model = self.model.to('cuda')


    def aug_hf(self):
        self.load_model()
        data = self._load_dir_json(os.path.join(self.data_dir, self.data_name, "train"))
        # save_path = os.path.join(self.save_dir, self.data_name, "aug_data", self.model_name)
        save_path = os.path.join(self.save_dir, self.data_name, "aug_data", self.model_name, "no_slot")
        # random.shuffle(data)
        aug_data, file_idx, sample_num, total_turn_num = [], 0, 5, 0
        for dial in tqdm(data):
            for turn in dial["log"][1:]:
                self.model.eval()
                usr_utt, sys_utt, dial_hist = turn["user utterance"], turn["system response"], turn["dialog history"]
                dial_hist = dial_hist.replace("<USER>", "User:").replace("<SYSTEM>", "System:")
                prompt = "Pretend you are user and talk to a system: "
                input_ids = self.tokenizer(dial_hist+" User:", return_tensors="pt").input_ids
                # input_ids = input_ids.view(2,-1)
                input_ids = input_ids.to('cuda')

                dst_dict = self.string_to_dict(turn["dst"])
                const_list = []
                for domain in dst_dict:
                    const_list.extend(list(dst_dict[domain].values()))
                const_list = list(set(const_list))
                if const_list:
                    continue
                    const_list_ids = self.tokenizer(const_list, add_special_tokens=False).input_ids
                    outputs = self.model.generate(
                        input_ids,
                        force_words_ids=const_list_ids,
                        num_beams=5,
                        num_return_sequences=sample_num,
                        no_repeat_ngram_size=1,
                        remove_invalid_values=True,
                    )
                else:
                    # continue
                    outputs = self.model.generate(
                        input_ids,
                        num_beams=5,
                        num_return_sequences=sample_num,
                        no_repeat_ngram_size=1,
                        remove_invalid_values=True,
                    )
                for sample_id in range(sample_num):
                    new_turn = {}
                    new_turn["turn id"] = turn["turn id"]
                    new_turn["dialog history"] = turn["dialog history"]
                    new_turn["dst"] = turn["dst"]
                    # replace with aug utt
                    new_turn["user utterance"] = self.tokenizer.decode(outputs[sample_id], skip_special_tokens=True)
                    # adding dialog id
                    new_turn["dialog_id"] = dial["original dialog id"]
                    new_turn["sample_id"] = sample_id
                    aug_data.append(new_turn)
                    total_turn_num += 1
                    
                    # if len(aug_data) > 100000:
                    #     # random.shuffle(aug_data)
                    #     file_name = f"dialog_{file_idx}.json"
                    #     self._save_json(aug_data, dir_path=save_path, file_name=file_name)
                    #     aug_data = []
                    #     file_idx += 1
                        
        file_name = f"dialog_{file_idx}.json"
        self._save_json(aug_data, dir_path=save_path, file_name=file_name)
        print(f"Saving {total_turn_num} turns in total ... ")


class PostProcessData(Augmentation):
    def __init__(self, args) -> None:
        super().__init__(args)

    def reformat_unaug_data(self):
        for mode in ["train", "val", "test"]:
            data = self._load_dir_json(os.path.join(self.data_dir, self.data_name, mode))
            save_dir = os.path.join(self.save_dir, self.data_name, "ori_data")
            turn_level_data, file_idx, total_turn_num = [], 0, 0
            for dial in tqdm(data):
                for turn in dial["log"][1:]:
                    new_turn = {}
                    new_turn["dialog_id"] = dial["original dialog id"]
                    new_turn["turn id"] = turn["turn id"]
                    new_turn["user utterance"] = turn["user utterance"]
                    new_turn["dialog history"] = turn["dialog history"].replace("<USER>", "User:").replace("<SYSTEM>", "System:")
                    new_turn["dst"] = turn["dst"].replace(",, ", " , ")
                    turn_level_data.append(new_turn)
                    total_turn_num += 1
            file_name = f"dialog_{mode}.json"
            self._save_json(turn_level_data, dir_path=save_dir, file_name=file_name)
            # save a short version for debugging
            file_name = f"dialog_{mode}_debug.json"
            self._save_json(turn_level_data[:100], dir_path=save_dir, file_name=file_name)
            print(f"Saving {total_turn_num} turns in total ... ")


    
    def combine_wiwo_slot(self):
        """
        The augmentation is conducted for turn wi/wo slots separately
        """
        for data_name in ["SGD", "MULTIWOZ2_2"]:
            aug_dir = os.path.join(self.save_dir, data_name, "aug_data")
            for model_name in os.listdir(aug_dir):
                data_path_wi_slot = os.path.join(aug_dir, model_name)
                data_path_wo_slot = os.path.join(data_path_wi_slot, "no_slot")
                if not os.path.exists(data_path_wi_slot) or not os.path.exists(data_path_wo_slot): continue
                data_wi_slot = self._load_dir_json(data_path_wi_slot)
                data_wo_slot = self._load_dir_json(data_path_wo_slot)
                data = data_wi_slot + data_wo_slot
                random.shuffle(data)
                os.makedirs(os.path.join(data_path_wi_slot, "splited_data"), exist_ok=True)
                for file_name in os.listdir(data_path_wi_slot):
                    if file_name in ["dialog_v0.json", "splited_data"]: continue
                    os.rename(os.path.join(data_path_wi_slot, file_name), os.path.join(data_path_wi_slot, "splited_data", file_name))
                self._save_json(data=data, dir_path=data_path_wi_slot, file_name="dialog_v0.json")

    def combine_num_data(self):
        """
        combine dialog_x.json into one file, 
        since combination has been done in combine_wiwo_slot() for aug_data
        this functin only targets at ori_dta
        """
        for data_name in ["SGD", "MULTIWOZ2_2"]:
            data_dir = os.path.join(self.save_dir, data_name, "ori_data")
            for mode in ["train", "val", "test"]:
                data = self._load_dir_json(os.path.join(data_dir, mode))
                random.shuffle(data)
                self._save_json(data=data, dir_path=data_dir, file_name=f"dialog_{mode}.json")


    def remove_lowq_data(self):
        """
        remove those turns that constrained decoding does not actually generate 
        constraint tokens
        only for v0 version
        """
        for data_name in ["SGD", "MULTIWOZ2_2"]:
            aug_dir = os.path.join(self.save_dir, data_name, "aug_data")
            for model_name in os.listdir(aug_dir):
                data = self._load_json(os.path.join(aug_dir, model_name, "dialog_v0.json"))
                clean_data = []
                for turn in data:
                    new_turn = {}
                    new_turn["dialog_id"] = turn["dialog_id"]
                    new_turn["turn id"] = turn["turn id"]
                    new_turn["user utterance"] = turn["user utterance"]
                    new_turn["dialog history"] = turn["dialog history"]
                    new_turn["dst"] = turn["dst"]
                    clean_data.append(new_turn)
                self._save_json(clean_data, os.path.join(aug_dir, model_name), "dialog_v0.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="MULTIWOZ2_2",
        choices=["MULTIWOZ2_2", "SGD"],
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="flan-t5-base",
        help="The model card used to load pre-trained model from huggingface",
    )
    parser.add_argument(
        "--act",
        type=str,
        default="aug",
        choices=["aug", "proc"],
        help="Choose whether to do augmentation or post processing",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.act == "act":
        aug = Augmentation(args)
        aug.aug_hf()
    elif args.act == "proc":
        proc = PostProcessData(args)
        proc.reformat_unaug_data()
        # proc.combine_wiwo_slot()
        # pro.combine_num_data()
        # proc.remove_lowq_data()
    else:
        print("Skip, since none of the pre-defined action is chosen ...")


if __name__ == "__main__":
    main()

