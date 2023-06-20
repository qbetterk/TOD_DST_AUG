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

DST_SPLIT = " , "

class Augmentation(object):
    def __init__(self, args) -> None:
        if os.uname()[1] == "communication":
            self.data_dir = "/local-storage/data/qkun/dataset/processed/Task-Oriented/"
        elif os.uname()[1] == "coffee":
            self.data_dir = "/local/data/qkun/dataset/processed/Task-Oriented/"
        self.save_dir = args.save_dir
        self.data_name = args.data_name
        self.model_card_or_path = args.model_card_or_path
        self.version = args.version
        self.batch_size = args.batch_size
        self.constraint = args.constraint_folder == "constraint_decoding"
        self.sample_num = args.sample_num
        self.sample_new_value = args.sample_new_value
        if self.sample_new_value:
            self.otgy = self._load_json(os.path.join("dataset", self.data_name, "otgy.json"))
        self._set_seed(args.seed)


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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
            dom, slot_type = slot.split()[:2]
            if dom not in dst_dict:
                dst_dict[dom] = {}
            dst_dict[dom][slot_type] = " ".join(slot.split()[2:])
            if dst_dict[dom][slot_type] == "yes":
                dst_dict[dom][slot_type] = slot_type
            elif dst_dict[dom][slot_type] == "no":
                dst_dict[dom][slot_type] = f"no {slot_type}"
        return dst_dict


    def dict_to_string(self, dst_dict):
        """
        dst_dict : {domain: {slot_type: slot_value, ...}, ...}
        e.g.:
        {'hotel': {'type': 'guesthouse', 'parking': 'parking', 'internet': 'internet', 'stars': '4'}}"""
        dst_list= []
        for domain in dst_dict:
            for slot_type in dst_dict[domain]:
                slot_value = dst_dict[domain][slot_type]
                if slot_value == slot_type: slot_value = "yes"
                elif slot_value == f"no {slot_type}": slot_value = "no"
                dst_list.append(f"{domain} {slot_type} {slot_value}")
        return DST_SPLIT.join(dst_list)


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

        data_in_turn = {
            "use_constraint": [],
            "no_constraint": [],
        }
        if self.model_card_or_path.split("/")[-1] == "utt_instruct":
            # user goal is needed, therefore we load data from ./dataset/data_name/ori_data
            data = self._load_json(os.path.join("dataset", self.data_name, "ori_data/dialog_train.json"))
            for turn in data:
                if turn["dst"] and self.constraint:
                    data_in_turn["use_constraint"].append(turn)
                else:
                    data_in_turn["no_constraint"].append(turn)
        else:
            data = self._load_dir_json(os.path.join(self.data_dir, self.data_name, "train"))
            for dial in data:
                for turn in dial["log"]:
                    turn["dialog_id"] = dial["original dialog id"]
                    if turn["dst"] and self.constraint:
                        data_in_turn["use_constraint"].append(turn)
                    else:
                        data_in_turn["no_constraint"].append(turn)
        len_wi_slot, len_wo_slot = len(data_in_turn["use_constraint"]), len(data_in_turn["no_constraint"])
        print(f"There are totally {len_wi_slot} turns with slots and {len_wo_slot} turns without any slots ... ")

        save_path = self.save_dir
        aug_data, total_turn_num = [], 0
        # # # Processing turns without slots
        print("Processing turns without slots ... ")
        for i in tqdm(range(0, len(data_in_turn["no_constraint"]), self.batch_size)):
            batch_input = []
            for turn in data_in_turn["no_constraint"][i:i + self.batch_size]:
                input_seq = turn["dialog history"].replace("<USER> ", "User: ").replace("<SYSTEM> ", "System: ")
                if self.model_card_or_path.split("/")[-1] == "utt":
                    input_seq += " Dialog states: " + turn["dst"]
                if self.model_card_or_path.split("/")[-1] == "utt_instruct":
                    input_seq = f" User goal: {turn['user_goal']} Dialog context: {input_seq} Dialog states: {turn['dst']}"
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
            # sample new slot values
            if self.sample_new_value:
                turn = self._replace_with_new_value(turn)
            # usr_utt, sys_utt, dial_hist = turn["user utterance"], turn["system response"], turn["dialog history"]
            input_seq = turn["dialog history"].replace("<USER> ", "User: ").replace("<SYSTEM> ", "System: ")
            if self.model_card_or_path.split("/")[-1] == "utt":
                input_seq += " Dialog states: " + turn["dst"]
            if self.model_card_or_path.split("/")[-1] == "utt_instruct":
                input_seq = f" User goal: {turn['user_goal']} Dialog context: {input_seq} Dialog states: {turn['dst']}"
            # prompt = "Pretend you are user and talk to a system: "
            input_ids = self.tokenizer(input_seq, return_tensors="pt").input_ids
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
        # save a short version for debugging
        debug_file_name = f"dialog_v{self.version}_debug.json"
        self._save_json(aug_data[:50]+aug_data[-50:], dir_path=save_path, file_name=debug_file_name)
        print(f"Saving {total_turn_num} turns in total ... ")


    def _replace_with_new_value(self, turn):
        """
        dst_dict : {domain: {slot_type: slot_value, ...}, ...}
        e.g.:
        {'hotel': {'type': 'guesthouse', 'parking': 'parking', 'internet': 'internet', 'stars': '4'}}
        """
        dst_dict = self.string_to_dict(turn["dst"])
        for domain in dst_dict:
            for slot_type in dst_dict[domain]:
                value_old = dst_dict[domain][slot_type]
                value_new = random.choice(self.otgy[domain][slot_type])
                turn["dst"] = turn["dst"].replace(f"{domain} {slot_type} {value_old}", 
                                                    f"{domain} {slot_type} {value_new}")
                turn["dst accumulated"] = turn["dst accumulated"].replace(f"{domain} {slot_type} {value_old}", 
                                                                            f"{domain} {slot_type} {value_new}")    
        return turn


    def add_new_turn(self, aug_data, total_turn_num, turn, output_seq, batch_id=None):
        for sample_id in range(self.sample_num):
            new_turn = {}
            new_turn["turn id"] = turn["turn id"]
            new_turn["dialog history"] = turn["dialog history"]
            new_turn["dst"] = turn["dst"]
            new_turn["dst accumulated"] = turn["dst accumulated"]
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
                for turn in dial["log"]:
                    new_turn = {}
                    new_turn["dialog_id"] = dial["original dialog id"]
                    new_turn["turn id"] = turn["turn id"]
                    new_turn["user utterance"] = turn["user utterance"].replace("User: ", "")
                    new_turn["dialog history"] = turn["dialog history"].replace("<USER>", "User:").replace("<SYSTEM>", "System:")
                    new_turn["dst"] = turn["dst"].replace(",, ", DST_SPLIT)
                    new_turn["dst accumulated"] = turn["dst accumulated"].replace(",, ", DST_SPLIT)
                    turn_level_data.append(new_turn)
                    total_turn_num += 1
            self._save_json(turn_level_data, dir_path=save_dir, file_name=f"dialog_{mode}.json")
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


    def add_acc_dst(self):
        """
        add accumulated dst annotation based on ori_data/dialog_train.json
        """
        aug_path = os.path.join(self.save_dir, self.data_name, "aug_data")
        ori_data = self._load_json(os.path.join(self.save_dir, self.data_name, "ori_data", "dialog_train.json"))
        for ft_method in os.listdir(aug_path):
            if ft_method == "no_ft": continue
            for constraint in os.listdir(os.path.join(aug_path, ft_method)):
                for aug_model in os.listdir(os.path.join(aug_path, ft_method, constraint)):
                    for file_name in os.listdir(os.path.join(aug_path, ft_method, constraint, aug_model)):
                        if file_name.endswith("debug.json"): continue
                        data_path = os.path.join(aug_path, ft_method, constraint, aug_model, file_name)
                        data = self._load_json(data_path)
                        for turn in tqdm(data):
                            for ori_turn in ori_data:
                                if turn["dialog_id"] == ori_turn["dialog_id"] and \
                                    turn["turn id"] == ori_turn["turn id"]:
                                    turn["dst accumulated"] = ori_turn["dst accumulated"]
                        self._save_json(data, file_path=data_path)


    def normalize(self):
        """

        Normalize both aug/unaug data
        e.g. <USER> --> User:
             remove redundant "User:" in 
        """

        # for ori data
        for mode in ["train", "test", "val"]:
            self.normalize_update(os.path.join(self.save_dir, self.data_name, "ori_data", f"dialog_{mode}.json"))
            self.add_instruct(os.path.join(self.save_dir, self.data_name, "ori_data", f"dialog_{mode}.json"))


    def normalize_update(self, data_path):
        data=self._load_json(data_path)
        for turn in tqdm(data):
            turn["dialog history"] = turn["dialog history"].replace("<USER>", "User:").replace("<SYSTEM>", "System:")
            turn["user utterance"] = turn["user utterance"].replace("User: ", "")
            turn["dst"] = turn["dst"].replace(",, ", DST_SPLIT)
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

        
    def sample_data(self):
        """
        sample the first 10 dialog as example
        """
        for mode in ["train", "val", "test"]:
            save_dir = os.path.join(self.save_dir, self.data_name, "ori_data")
            data = self._load_json(os.path.join(save_dir, f"dialog_{mode}.json"))
            self._save_json(data[:10], dir_path=save_dir, file_name=f"dialog_{mode}_debug.json")


    def create_otgy(self):
        data = self._load_dir_json(os.path.join(self.save_dir, self.data_name, "ori_data"))
        save_dir = os.path.join(self.save_dir, self.data_name)
        otgy = {}
        for turn in data:
            for slot in turn["dst"].split(DST_SPLIT):
                if not slot: continue
                domain, slot_type = slot.split()[0], slot.split()[1]
                slot_value = " ".join(slot.split()[2:])
                if slot_value == "dontcare": continue
                if domain not in otgy:
                    otgy[domain] = {}
                if slot_type not in otgy[domain]:
                    otgy[domain][slot_type] = []
                if slot_value not in otgy[domain][slot_type]:
                    otgy[domain][slot_type].append(slot_value)
        self._save_json(otgy, dir_path=save_dir, file_name= "otgy.json")


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
        "--constraint_folder",
        type=str,
        default="constraint_decoding",
        choices=["constraint_decoding", "no_constraint_decoding"],
        help="Choose whether to do constraint decoding or not (for data augmentation)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for no_constraint augmentation",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="version for augmentation data",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=5,
        help="number of augmented utt for each turn",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="path to save augmented data",
    )
    parser.add_argument(
        "--sample_new_value",
        type=bool,
        default=True,
        help="Choose whether to sample new values for augmentation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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
        # # proc.combine_wiwo_slot()
        # # pro.combine_num_data()
        # # proc.remove_lowq_data()
        proc.normalize()
        # proc.add_acc_dst()
        proc.sample_data()
        # proc.create_otgy()
    else:
        print("Skip, since none of the pre-defined action is chosen ...")


if __name__ == "__main__":
    main()