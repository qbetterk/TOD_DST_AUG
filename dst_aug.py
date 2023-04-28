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
        if os.uname()[1] == "communication":
            self.data_dir = "/local-storage/data/qkun/dataset/processed/Task-Oriented/"
            self.save_dir = "/local-scratch1/data/qkun/tod_aug/"
        elif os.uname()[1] == "coffee":
            self.data_dir = "/local/data/qkun/dataset/processed/Task-Oriented/"
            self.save_dir = "/local/data/qkun/tod_aug/"
        self.data_name = args.data_name
        if args.model_card_or_path.startswith("ckpt"):
            self.model_card_or_path = os.path.join(self.save_dir, args.model_card_or_path)
            self.model_name = args.model_card_or_path.split("/")[1]
            self.ft_method = args.model_card_or_path.split("/")[-1]
        else:
            self.model_card_or_path = args.model_card_or_path
            self.model_name = self.model_card_or_path.split("/")[-1]
            self.ft_method = "no_ft"
        self.version = 1
        self.batch_size = args.batch_size
        self.constraint = args.constraint
        self.sample_num = 5


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


    def _save_json(self, data, dir_path=None, file_name=None, file_path=None):
        if not file_path:
            os.makedirs(dir_path, exist_ok=True)
            path = os.path.join(dir_path, file_name)
            print(f"saving {file_name} to {dir_path} ......")
        else:
            path = file_path
            print(f"saving to {file_path} ......")
        with open(path, "w") as tf:
            json.dump(data, tf, indent=2)
    

    def _update_json(self, data, append=False, dir_path=None, file_name=None, file_path=None):
        """
        if append:
            extend data to the existing data in the file (assuming in the format of list)
        else:
            rewrite the original file with the data"""
        if not file_path:
            os.makedirs(dir_path, exist_ok=True)
            path = os.path.join(dir_path, file_name)
            print(f"saving {file_name} to {dir_path} ......")
        else:
            path = file_path
            print(f"saving to {file_path} ......")
        if append:
            with open(path, "r") as df:
                try:
                    old_data = json.load(df)
                except json.JSONDecodeError:
                    old_data = []
            data = old_data.extend(data)
        with open(path, "w") as tf:
            json.dump(data, tf, indent=2)


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
            if dst_dict[dom][slot_type] == "yes":
                dst_dict[dom][slot_type] = slot_type
            elif dst_dict[dom][slot_type] == "no":
                dst_dict[dom][slot_type] = f"no {slot_type}"
        return dst_dict


    def load_model(self):
        print(f"Loading model: {self.model_card_or_path} ......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card_or_path)
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.cuda.empty_cache()
        self.model.eval()
        self.model = self.model.to('cuda')


    def aug_hf(self):
        self.load_model()
        self.model.eval()
        data = self._load_dir_json(os.path.join(self.data_dir, self.data_name, "train"))
        if self.constraint:
            constraint_folder = "constraint_decoding"
        else:
            constraint_folder = "no_constraint_decoding"
        save_path = os.path.join(self.save_dir, 
                                 self.data_name, 
                                 "aug_data",
                                 self.ft_method,
                                 constraint_folder,
                                 self.model_name)
        # random.shuffle(data)
        aug_data, self.sample_num, total_turn_num = [], 5, 0
        data_in_turn = {
            "use_constraint": [],
            "no_constraint": [],
        }
        for dial in data:
            for turn in dial["log"]:
                turn["dialog_id"] = dial["original dialog id"]
                if turn["dst"] and self.constraint:
                    data_in_turn["use_constraint"].append(turn)
                else:
                    data_in_turn["no_constraint"].append(turn)
        len_wi_slot, len_wo_slot = len(data_in_turn["use_constraint"]), len(data_in_turn["no_constraint"])
        print(f"There are totally {len_wi_slot} turns with slots and {len_wo_slot} turns without any slots ... ")

        # # # Processing turns without slots
        print("Processing turns without slots ... ")
        for i in tqdm(range(0, len(data_in_turn["no_constraint"]), self.batch_size)):
            batch_input = []
            for turn in data_in_turn["no_constraint"][i:i + self.batch_size]:
                input_seq = turn["dialog history"].replace("<USER> ", "User: ").replace("<SYSTEM> ", "System: ")
                if self.model_card_or_path.split("/")[-1] == "utt":
                    input_seq += " Dialog states: " + turn["dst"]
                batch_input.append(input_seq)
            input_tensors = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True)
            input_tensors = input_tensors.to('cuda')

            outputs = self.model.generate(
                **input_tensors,
                num_return_sequences=self.sample_num,
                remove_invalid_values=True,
                do_sample=True,
                max_length=128,
            )
            output_seq = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for batch_id in range(self.batch_size):
                # the last batch might be not full
                if i+batch_id >= len(data_in_turn["no_constraint"]): continue
                turn = data_in_turn["no_constraint"][i + batch_id]
                aug_data, total_turn_num = self.add_new_turn(aug_data, total_turn_num, turn, output_seq, batch_id)

        # # # Processing turns with slots
        print("Processing turns with slots ... ")
        for turn in tqdm(data_in_turn["use_constraint"]):
            # usr_utt, sys_utt, dial_hist = turn["user utterance"], turn["system response"], turn["dialog history"]
            dial_hist = turn["dialog history"].replace("<USER>", "User:").replace("<SYSTEM>", "System:")
            if self.model_card_or_path.split("/")[-1] == "utt":
                dial_hist += " Dialog states: " + turn["dst"]
            # prompt = "Pretend you are user and talk to a system: "
            input_ids = self.tokenizer(dial_hist, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')
            # prepare constraints
            dst_dict = self.string_to_dict(turn["dst"])
            const_list = []
            for domain in dst_dict:
                const_list.extend(list(dst_dict[domain].values()))
            const_list = list(set(const_list))
            if not const_list:
                pdb.set_trace()
            const_list_ids = self.tokenizer(const_list, add_special_tokens=False).input_ids
            outputs = self.model.generate(
                input_ids,
                force_words_ids=const_list_ids,
                num_beams=20,
                num_return_sequences=self.sample_num,
                remove_invalid_values=True,
                max_length=64,
                early_stopping=True,
            )
            output_seq = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            aug_data, total_turn_num = self.add_new_turn(aug_data, total_turn_num, turn, output_seq)
 
        file_name = f"dialog_v{self.version}.json"
        self._save_json(aug_data, dir_path=save_path, file_name=file_name)
        print(f"Saving {total_turn_num} turns in total ... ")


    def add_new_turn(self, aug_data, total_turn_num, turn, output_seq, batch_id=None):
        for sample_id in range(self.sample_num):
            new_turn = {}
            new_turn["turn id"] = turn["turn id"]
            new_turn["dialog history"] = turn["dialog history"]
            new_turn["dst"] = turn["dst"]
            new_turn["dialog_id"] = turn["dialog_id"]
            new_turn["ori_user_utt"] = turn["user utterance"]
            new_turn["sample_id"] = sample_id
            # replace with aug utt
            if batch_id:
                new_turn["user utterance"] = output_seq[self.sample_num*batch_id+sample_id]
            else:
                new_turn["user utterance"] = output_seq[sample_id]
            aug_data.append(new_turn)
            total_turn_num += 1
            return aug_data, total_turn_num


class PostProcessData(Augmentation):
    def __init__(self, args) -> None:
        super().__init__(args)

    def reformat_unaug_data(self):
        """
        One-time use, not needed anymore"""
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
        One-time use, no need any more
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
        One-time use, no use anymore
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
                self._save_json(clean_data, dir_path=os.path.join(aug_dir, model_name), file_name="dialog_v0.json")


    def normalize(self):
        """

        Normalize both aug/unaug data
        e.g. <USER> --> User:
             remove redundant "User:" in 
        """
        # # for aug data
        # for aug_model in os.listdir(os.path.join(self.save_dir, self.data_name, "aug_data")):
        #     aug_path = os.path.join(self.save_dir, self.data_name, "aug_data", aug_model)
        #     for data_version in os.listdir(os.path.join(aug_path)):
        #         if not data_version.endswith("json"): continue
        #         self.normalize_update(os.path.join(aug_path, data_version))

        # for ori data
        for mode in ["train", "test", "val"]:
            self.normalize_update(os.path.join(self.save_dir, self.data_name, "ori_data", f"dialog_{mode}.json"))
            self.add_instruct(os.path.join(self.save_dir, self.data_name, "ori_data", f"dialog_{mode}.json"))


    def normalize_update(self, data_path):
        data=self._load_json(data_path)
        for turn in tqdm(data):
            turn["dialog history"] = turn["dialog history"].replace("<USER>", "User:").replace("<SYSTEM>", "System:")
            turn["user utterance"] = turn["user utterance"].replace("User: ", "")
            turn["dst"] = turn["dst"].replace(",, ", " , ")
        self._save_json(data, file_path=data_path)

    
    def add_instruct(self, data_path):
        data_21_path = os.path.join(self.data_dir.split("processed")[0], "raw/multiwoz/data/MultiWOZ_2.1/data.json")
        data_21 = self._load_json(data_21_path)
        data=self._load_json(data_path)
        for turn in tqdm(data):
            dial_id = turn["dialog_id"]
            raw_instruction = data_21[dial_id]["goal"]["message"]
            instruct_str = " ".join(raw_instruction)
            instruct_str = instruct_str.replace("</span>", "").replace("<span class='emphasis'>", "")
            turn["user_goal"] = instruct_str
        self._save_json(data, file_path=data_path)


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
        "--model_card_or_path",
        type=str,
        default="google/flan-t5-base",
        help="The model card used to load pre-trained model from huggingface",
    )
    parser.add_argument(
        "--act",
        type=str,
        default="proc",
        choices=["aug", "proc", "user"],
        help="Choose whether to do augmentation or post processing or have a user_utt_generation format",
    )
    parser.add_argument(
        "--constraint",
        action='store_true',
        help="Choose whether to do constraint decoding or not (for data augmentation)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for no_constraint augmentation",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.act == "aug":
        aug = Augmentation(args)
        aug.aug_hf()
    elif args.act == "proc":
        proc = PostProcessData(args)
        # proc.reformat_unaug_data()
        # proc.combine_wiwo_slot()
        # pro.combine_num_data()
        # proc.remove_lowq_data()
        proc.normalize()
    else:
        print("Skip, since none of the pre-defined action is chosen ...")


if __name__ == "__main__":
    main()

