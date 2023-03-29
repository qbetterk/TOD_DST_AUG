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
DST_SPLIT = ",, "
class Augmentation(object):
    def __init__(self) -> None:
        self.data_dir = "./sgd/train"
        self.save_dir = "/local-scratch1/data/qkun/tod_aug/aug_data/"
        self.batch_size = 1
        self.beam_size = 2
        self.model_name = "flan-t5-xl"
        print(f"Using model: {self.model_name} ......")
        self.load_model()


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
        if dir_path is None or not os.path.exists(dir_path): return None
        total_data = [] # assume data is a list of dialogs
        for filename in os.listdir(dir_path):
            if filename in ["schema.json"]: continue
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
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/{self.model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{self.model_name}")
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.cuda.empty_cache()
        self.model.eval()
        self.model = self.model.to('cuda')


    def aug_hf(self):
        data = self._load_dir_json(self.data_dir)
        # random.shuffle(data)
        aug_data = []
        file_idx = 5
        sample_num = 5
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
                    continue
                    # outputs = self.model.generate(
                    #     input_ids,
                    #     num_beams=5,
                    #     num_return_sequences=sample_num,
                    #     no_repeat_ngram_size=1,
                    #     remove_invalid_values=True,
                    # )
                for sample_id in range(sample_num):
                    new_turn = dict(turn)
                    # replace with aug utt
                    new_turn["user utterance"] = self.tokenizer.decode(outputs[sample_id], skip_special_tokens=True)
                    # adding dialog id
                    new_turn["dialog_id"] = dial["original dialog id"]
                    new_turn["sample_id"] = sample_id
                    aug_data.append(new_turn)
                    
                    if len(aug_data) > 100000:
                        # random.shuffle(aug_data)
                        file_name = f"dialog_aug_{file_idx}.json"
                        self._save_json(aug_data, dir_path=os.path.join(self.save_dir, self.model_name), file_name=file_name)
                        aug_data = []
                        file_idx += 1

        file_name = f"dialog_aug_{file_idx}.json"
        self._save_json(aug_data, dir_path=os.path.join(self.save_dir, self.model_name), file_name=file_name)



def main():
    aug = Augmentation()
    aug.aug_hf()


if __name__ == "__main__":
    main()

