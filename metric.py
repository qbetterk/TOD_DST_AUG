#!/usr/bin/env python3
#
""" DST metric. """

import datasets

_CITATION = ""

_DESCRIPTION = """
This metric compute the joint goal accuracy for the dialog state tracking task
"""

_KWARGS_DESCRIPTION = """
Computes JGA score.
Args:
    predictions: A list of predicted slots. Each element corresponds to the slots for a dialog turn.
                 Each slot is predicted in the order of "domain, slot type, slot value", separated 
                 by certain separate token (default by ",, ")
    references: A list of ground truth slots.
Returns:
    'jga': joint goal accuracy (the rate of turns that all "domain', "slot_type" and "slot_value" match the ground truth)
Examples:
    >>> predictions = ["restaurant name ABC,, restaurant parking no",
                        "hotel internet yes,, hotel area north"]
    >>> references = ["restaurant name BCD,, restaurant parking no",
                        "hotel internet yes,, hotel area north"]
    >>> dst_metric = datasets.load_metric("jga")
    >>> results = dst_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'jga': 0.5}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DST(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _compute(self, predictions, references):
        # # # extract ground truth from labels
        slots_truths = [self._extract_slot_from_string(label) for label in references]
        # # # extract generated slots from model_response
        slots_preds = [self._extract_slot_from_string(predict) for predict in predictions]

        if len(slots_truths) != len(slots_truths): raise ValueError("The number of predictions does not match the number of predictions")
        score = 0.0
        for (pred, truth) in zip(slots_preds, slots_truths):
            score += set(pred) == set(truth)
        score /= len(slots_preds)
        return {"jga": score}


    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]

        slot_types = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
                    "area", "leave", "stars", "department", "people", "time", "food", 
                    "post", "phone", "name", 'internet', 'parking',
                    'book stay', 'book people','book time', 'book day',
                    'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']
        slots_list = []

        # # # remove start and ending token
        str_split = slots_string.strip().split()
        if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
            str_split = str_split[1:]
        if "</bs>" in str_split:
            str_split = str_split[:str_split.index("</bs>")]

        # # # split according to ","
        str_split = " ".join(str_split).split(",, ")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in domains:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1]+" "+slot[2]
                    slot_val  = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val  = " ".join(slot[2:])
                if not slot_val == 'dontcare':
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
        return slots_list
