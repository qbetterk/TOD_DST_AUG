#!/usr/bin/env python3
#
import sys, os, pdb
import json
from tqdm import tqdm

class Analysis(object):
    def __init__(self) -> None:
        pass

    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            data = json.loads(df.read())
        return data

    def _load_json_dir(self, dir_path=None):
        if dir_path is None or not os.path.exists(dir_path): return None
        total_data = [] # assume data is a list of dialogs
        for filename in os.listdir(dir_path):
            if filename in ["schema.json"]: continue # exclude non-dialog files
            if not filename.endswith(".json"): continue # exclude non-dialog files
            file_path = os.path.join(dir_path, filename)
            data = self._load_json(path=file_path)
            if type(data) == list:
                total_data.extend(data)
            else:
                total_data.append(data)
        return total_data

    def analysis_raw_multiwoz22(self):
        self.data_path = "./multiwoz22/"
        slot_in_utt, slot_num, slot_in_utt_turn, turn_num_with_slot, turn_num = 0, 0, 0, 0, 0 
        for mode in ["train", "dev", "test"]:
            data = self._load_json_dir(os.path.join(self.data_path, mode))
            for dial_idx, dial in tqdm(enumerate(data)):
                for idx, turn in enumerate(dial["turns"]):
                    flag_slot_exits, flag_slot_not_in_utt = 0, 0
                    for domain in turn["frames"]:
                        for slot in domain["slots"]:
                            # print(slot["value"], turn["utterance"])
                            # pdb.set_trace()
                            slot_value = slot["value"] if type(slot["value"])==str else slot["value"][0]
                            if " " + slot_value in turn["utterance"] or \
                                slot_value + " " in turn["utterance"]:
                                slot_in_utt += 1
                            else:
                                flag_slot_not_in_utt = 1
                            slot_num += 1
                            flag_slot_exits = 1
                    if flag_slot_exits:
                        turn_num_with_slot += 1
                        if not flag_slot_not_in_utt:
                            slot_in_utt_turn += 1
                    turn_num += 1
        print(slot_in_utt, slot_num, slot_in_utt_turn, turn_num_with_slot, turn_num)


    def analysis_pptod(self):
        self.result_path = "./pptod/DST/inference_result/large/full_training/test_joint_accuracy_53.89.json"
        result = self._load_json(self.result_path)
        slot_in_utt_m, slot_num_m, slot_in_utt_e, slot_num_e = 0, 0, 0, 0 
        slot_in_utt_turn, turn_num_with_slot, turn_num = 0, 0, 0
        acc = 0
        accu = 1 # analysis for accumulated slots
        if not accu:
            # count for single turn (non-accumulated slots)
            for dial_id, dial in result.items():
                for turn_id, turn in dial.items():
                    if "predbs" not in turn: # correct prediction
                        missing_set, extra_set = set(), set()
                        acc += 1
                    elif turn_id == "0":
                        missing_set = set(turn["gtbs"]).difference(set(turn["predbs"]))
                        extra_set = set(turn["predbs"]).difference(set(turn["gtbs"]))
                    else:
                        if "predbs" not in dial[str(int(turn_id)-1)]:
                            # extract from bspn_reform
                            dom_slot_vals = dial[str(int(turn_id)-1)]["bspn_reform"].split("[")[1:]
                            predict_set_prev = gtruth_set_prev = set()
                            for dom_slot_val in dom_slot_vals:
                                domain = dom_slot_val.split("]")[0]
                                for slot_val in dom_slot_val.split("]")[-1].split(" , "):
                                    slot_val = slot_val.strip()
                                    slot_type, slot_val = slot_val.split(" is ")
                                    gtruth_set_prev.add(f"[{domain}] {slot_type} {slot_val}")
                        else:
                            predict_set_prev = set(dial[str(int(turn_id)-1)]["predbs"])
                            gtruth_set_prev = set(dial[str(int(turn_id)-1)]["gtbs"])
                        predict_set = set(turn["predbs"]).difference(predict_set_prev)
                        gtruth_set = set(turn["gtbs"]).difference(gtruth_set_prev)
                        missing_set = gtruth_set.difference(predict_set)
                        extra_set = predict_set.difference(gtruth_set)
                    for slot in missing_set:
                        slot_value = " ".join(slot.split()[2:])
                        if slot_value in turn["user"]:
                            slot_in_utt_m += 1
                        slot_num_m += 1
                    for slot in extra_set:
                        slot_value = " ".join(slot.split()[2:])
                        if slot_value in turn["user"]:
                            slot_in_utt_e += 1
                        slot_num_e += 1
                    turn_num += 1
            print(acc/turn_num, slot_in_utt_m, slot_num_m, slot_in_utt_e, slot_num_e)
        else:
            # count for accumulated slots
            # count for single turn (non-accumulated slots)
            for dial_id, dial in result.items():
                for turn_id, turn in dial.items():
                    all_in_utt = 1
                    if "predbs" not in turn: # correct prediction
                        missing_set, extra_set = set(), set()
                        acc += 1
                    else:
                        # accumulate dialog history (no system side utterances)
                        for i in range(int(turn_id)):
                            turn["user"] += " " + dial[str(i)]["user"]
                        missing_set = set(turn["gtbs"]).difference(set(turn["predbs"]))
                        extra_set = set(turn["predbs"]).difference(set(turn["gtbs"]))
                        for slot in missing_set:
                            slot_value = " ".join(slot.split()[2:])
                            if slot_value in turn["user"]:
                                slot_in_utt_m += 1
                            else:
                                all_in_utt = 0
                            slot_num_m += 1
                        for slot in extra_set:
                            slot_value = " ".join(slot.split()[2:])
                            if slot_value in turn["user"]:
                                slot_in_utt_e += 1
                                # all_in_utt = 0
                            slot_num_e += 1
                        if all_in_utt:
                            acc += 1
                    turn_num += 1
            print(acc/turn_num, acc, turn_num, slot_in_utt_m, slot_num_m, slot_in_utt_e, slot_num_e)
                        
                    


                      

def main():
    analysis = Analysis()
    # analysis.analysis_raw_multiwoz22()
    analysis.analysis_pptod()


if __name__ == "__main__":
    main()