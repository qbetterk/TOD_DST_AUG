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

import openai
openai.api_key = "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"# Use the API
openai.organization = "org-vBJ9i7PnvVK5CUVrD7emHHC2"

DST_SPLIT = " , "

class EvalLLMDST(object):
    def __init__(self, args) -> None:
        if os.uname()[1] == "communication":
            self.data_dir = "/local-storage/data/qkun/dataset/processed/Task-Oriented/"
        elif os.uname()[1] == "coffee":
            self.data_dir = "/local/data/qkun/dataset/processed/Task-Oriented/"
        self.save_dir = args.save_dir
        self.data_name = args.data_name
        self.model_card_or_path = args.model_card_or_path
        self.dst_acc = args.dst_acc
        self._set_seed(args.seed)


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
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


    def load_model(self):
        print(f"Loading model: {self.model_card_or_path} ......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card_or_path)
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.cuda.empty_cache()
        self.model.eval()
        self.model = self.model.to('cuda')


    def test_manual(self):
        self.load_model()
        self.model.eval()

        input_seq = self.construct_input()
        print(input_seq)
        input_ids = self.tokenizer(input_seq, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")

        outputs = self.model.generate(
            input_ids,
            remove_invalid_values=True,
            max_length=64,
            early_stopping=True,
        )
        output_seq = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(output_seq)


    def json_to_string(self, data_json):
        data_json = str(data_json).replace("'","")
        data_json = data_json.replace("{","").replace("}","")
        return data_json

    def construct_input(self, test_case=None):
        # otgy_ds = self._load_json(f"dataset/{self.data_name}/ori_data/otgy_ds.json")
        otgy_ds = self._load_json("./dataset/MULTIWOZ2_2/otgy_ds.json")
        otgy_ds = self.json_to_string(otgy_ds)
        if self.dst_acc:
            Instruction = """
            Given the dialog context and the current user utterance, try to understand it and extract dialog states in terms of "domain slot_type slot_value".
            """

            OTGY = f"""
            The possible domain and slot_type are listed below:
            {otgy_ds}
            """

            Example = """
            dialog context:
            User: I need a train going to Cambridge that will depart after 10:15 from broxbourne. System: I have train TR5678 that would suit you. User: Could you just tell me when that one departs? System: Train TR5678 departs Broxbourne at 10:32 and arrives at Cambridge at 11:32. User: Great can you get me a booking for 3 people? System: Booked! Your reference number is: XZU4Z1RR . Can I help with anything else? User: Thank you. Yes. I am also looking for an international restaurant. System: There are three. Two located in the centre that are moderate in price and one in the east that is cheap. User: I want the one in the east, please book me a table for 3 people. System: Certainly. Can you please tell me for what day you would like for me to make this reservation? User: can you book it for sunday afternoon? System: What time exactly would you like to dine?
            user utterance:
            Actually, I'd like a moderately priced international restaurant in the centre. I need the postcode and address please. Sorry, I keep changing my request!
            dialog states:
            restaurant area centre , restaurant bookday sunday , restaurant bookpeople 3 , restaurant food international , restaurant pricerange moderate , train arriveby 11:32 , train bookpeople 3 , train departure broxbourne , train destination cambridge , train leaveat 10:32

            dialog context:
            User: It's Sunday and I am bored. Do you have any trains into Cambridge today? System: There are many trains, when would you like to depart? User: I would like to leave from Kings Lynn after 9:30. System: TR5928 leaves at 10:11, would you like me to book that for you? User: Can you give me the price of a ticket? System: Tickets are 7.84 pounds. Can I book that for you? User: Great I also need to find a place to stay that needs to have free parking and is in the cheap price range. System: There are 33 hotels in the cheap range, any preference on area or anything else?
            user utterance:
            Cheap price range in the centre.
            dialog states:
            hotel area centre , hotel parking yes , hotel pricerange cheap , train day sunday , train departure kings lynn , train destination cambridge , train leaveat 10:11

            """

            Test_case = """
            dialog context:
            User: I'm looking for some hungarian food restaurants near the centre, please. System: I am sorry there are no Hungarian Restaurants near centre. User: What kind of expensive restaurants are in the center of town? System: Unfortunately there aren't any Hungarian places. Do you want to try other cuisine types? User: yes let me see the options System: There is a variety of choices; Chinese, African, British, Italian, American, Gastro Pub, etc... Do you have a preference? User: i want indian food. System: Curry Garden is a nice place would you like that info ? User: Yes can I get a address and phone number please? System: Certainly, the phone number for the Curry Garden restaurant is 01223302330 and the address is 106 Regent Street City Centre. May I help you with anything else today?
            user utterance:
            Yes I am also looking for a place in the same area as the restaurant without free parking.I want it for 3 people for 5 nights starting Tuesday.
            dialog states:
            """ if test_case is None else test_case
        else:
            Instruction = """
            Given the dialog context and the current user utterance, try to understand it and extract dialog states in terms of "domain slot_type slot_value".
            """

            OTGY = f"""
            The possible domain and slot_type are listed below:
            {otgy_ds}
            """

            Example = """
            dialog context:
            User: I need to find a train out of Cambridge this Friday. System: I can book you one that leaves at 05:00 and you will arrive by 05:51 will that be alright?
            user utterance:
            No, I need a train that leaves after 17:15 that goes to broxbourne.
            dialog states:
            train destination broxbourne , train leaveat 17:15

            dialog context:
            User: Could you find a moderate priced place to stay for me? I do need free WiFi. System: Absolutely. We have many guesthouses and hotels with free Wifi. Any other preferences?
            user utterance:
            I'd prefer something with 2 stars, and I need free parking as well.
            dialog states:
            hotel parking yes , hotel stars 2
            
            """

            Test_case = """
            dialog context:
            User: I'm looking for some hungarian food restaurants near the centre, please. System: I am sorry there are no Hungarian Restaurants near centre. User: What kind of expensive restaurants are in the center of town? System: Unfortunately there aren't any Hungarian places. Do you want to try other cuisine types? User: yes let me see the options System: There is a variety of choices; Chinese, African, British, Italian, American, Gastro Pub, etc... Do you have a preference? User: i want indian food. System: Curry Garden is a nice place would you like that info ? User: Yes can I get a address and phone number please? System: Certainly, the phone number for the Curry Garden restaurant is 01223302330 and the address is 106 Regent Street City Centre. May I help you with anything else today?
            user utterance:
            Yes I am also looking for a place in the same area as the restaurant with free parking.I want it for 3 people for 5 nights starting Tuesday.
            dialog states:
            """ if test_case is None else test_case
        Input = f"{Instruction} {OTGY} {Example} {Test_case}"
        # print(Input)
        return Input


    def openai_api(self, test_case):
        input_seq = self.construct_input(test_case=test_case)
        # print(input_seq)
        # pdb.set_trace()
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # assuming the GPT-4 model identifier
        messages=[
                {"role": "user", "content": input_seq},
            ]
        )
        return response['choices'][0]['message']['content']

    def openai_test(self):
        test_data = self._load_json(f"dataset/{self.data_name}/ori_data/dialog_test.json")
        random.shuffle(test_data)
        result, jga = [], 0
        for turn in tqdm(test_data[:100]):
            test_case = f"""
            dialog context:
            {turn["dialog history"]}
            user utterance:
            {turn["user utterance"]}
            dialog states:
            """
            output = self.openai_api(test_case)
            label  = turn["dst accumulated"]
            if set(output.split(DST_SPLIT)) == set(label.split(DST_SPLIT)):
                jga += 1

            result.append({
                "generate_seq": DST_SPLIT.join(sorted(output.split(DST_SPLIT))),
                "ground_truth": DST_SPLIT.join(sorted(label.split(DST_SPLIT))),
            })
        print(jga)
        self._save_json(data=result ,dir_path="tod_aug/openai/", file_name="output.json")


    def create_otgy(self):
        """
        create otgy for only slot type and domain
        """
        otgy ={}
        for mode in ["train", "val", "test"]:
            data = self._load_json(os.path.join("dataset", self.data_name, f"ori_data/dialog_{mode}.json"))
            for turn in data:
                slots = self.string_to_dict(turn["dst"])
                for domain in slots:
                    if domain not in otgy:
                        otgy[domain] = []
                    otgy[domain].extend(list(slots[domain].keys()))
                    otgy[domain] = list(set(otgy[domain]))
        self._save_json(otgy, dir_path=f"dataset/{self.data_name}/ori_data/", file_name="otgy_ds.json")
                

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
        default="google/flan-t5-xl",
        help="The model card used to load pre-trained model from huggingface",
    )
    parser.add_argument(
        "--test_target",
        type=str,
        default="dst",
        choices=["dst", "aug"],
        help="Choose which capability of LLM to evaluate",
    )
    parser.add_argument(
        "--dst_acc",
        type=bool,
        default=False,
        help="Whether to use accumulated dst or just current turn dst",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.test_target == "dst":
        test = EvalLLMDST(args)
        # test.create_otgy()
        # test.test_manual()
        test.openai_test()
    elif args.test_target == "aug":
        pass
    else:
        print("Skip, since none of the pre-defined test target is chosen ...")


if __name__ == "__main__":
    main()