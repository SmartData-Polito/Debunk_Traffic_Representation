import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from typing import Any, Dict, List, Optional, Union
import numpy as np
import random
from transformers.utils import requires_backends, is_torch_device
from utils import get_logger

logger = get_logger(name=__name__)


class DataCollatorWithMeta(DataCollatorForLanguageModeling):
    def __init__(self, values_clip: Optional[int] = None, swap_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_clip = values_clip
        self.swap_rate = swap_rate

    def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        burstsInEachFlow = [example["total_bursts"] for example in examples]
        maxBursts = max(burstsInEachFlow)
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            for key in inputs.keys():
                if key == "labels" or key == "total_bursts" or key == "replacedAfter":
                    continue
                if key not in batch:
                    if key != "replacedAfter":
                        batch[key] = []
                if key == "ports":
                    batch[key].append(inputs[key] + 1)
                elif key in ("protocol", "flow_duration"):
                    batch[key].append(inputs[key])
                else:
                    batch[key].append(
                        inputs[key][: maxBursts * self.tokenizer.max_burst_length]
                    )
        for key in batch.keys():
            batch[key] = torch.Tensor(np.array(batch[key]))
            if (
                    key == "input_ids"
                    or key == "attention_masks"
                    or key == "ports"
                    or key == "protocol"
            ):
                batch[key] = torch.Tensor(batch[key]).to(torch.long)

        if self.mlm:
            batch["input_ids"], batch["labels"], batch["swappedLabels"], batch[
                "burstMetasToBeMasked"] = self.torch_mask_tokens(
                batch["input_ids"], burstsInEachFlow, self.tokenizer.max_burst_length, self.swap_rate,
                batch["protocol"], special_tokens_mask=None
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return BatchEncoding(batch)

    def swap_bursts_adjust_prob_matrix(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate):
        labels = torch.from_numpy(np.array(np.random.rand(len(burstsInEachFlow)) < swap_rate, dtype=int))
        swappedIds = []
        for i in range(input_ids.shape[0]):
            if labels[i] == 1:
                burstToRep = random.randint(0, burstsInEachFlow[i] - 1)
                flowChoice = random.randint(0, input_ids.shape[0] - 1)
                if flowChoice == i:
                    flowChoice = (flowChoice + 1) % input_ids.shape[0]
                burstChoice = random.randint(0, burstsInEachFlow[flowChoice] - 1)
                swappedIds.append([i, burstToRep])
                input_ids[i][burstToRep * max_burst_length:(burstToRep + 1) * max_burst_length] = input_ids[flowChoice][burstChoice * max_burst_length:(burstChoice + 1) * max_burst_length]
        return input_ids, swappedIds, labels

    def maskMetaData(self, input_ids, burstsInEachFlow, swapped_bursts):
        maskedMetaBursts = np.full((input_ids.shape[0], max(burstsInEachFlow)), 0.3)
        for ids in swapped_bursts:
            maskedMetaBursts[ids[0]][ids[1]] = 0
        candidateFlows = np.array(
            [np.array(np.array(burstsInEachFlow) > 3, dtype=int)]).transpose()  # converting to nX1 matrix
        return torch.bernoulli(torch.from_numpy(candidateFlows * maskedMetaBursts)).bool()

    def torch_mask_tokens(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate, protos, **kwargs):
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        new_ip_ids, swappedIds, swappedLabels = self.swap_bursts_adjust_prob_matrix(input_ids, burstsInEachFlow,
                                                                                    max_burst_length, swap_rate)
        maskMetaData = self.maskMetaData(input_ids, burstsInEachFlow, swappedIds)
        for ids in swappedIds:
            probability_matrix[ids[0]][ids[1] * max_burst_length:(ids[1]) * max_burst_length] = 0
        input_ids = new_ip_ids

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels, swappedLabels, maskMetaData


class DataCollatorForFlowClassification:
    label_names: Dict

    def __init__(self, max_burst_length):
        self.max_burst_length = max_burst_length

    def __call__(self, examples):
        first = examples[0]
        maxBursts = max([int(example["total_bursts"]) for example in examples])
        for i in range(len(examples)):
            if "stats" in examples[i]:
                examples[i]["stats"] = [
                    float(t) for t in examples[i]["stats"].split(" ")
                ]
        batch = {}
        if "labels" in first and first["labels"] is not None:
            label = (
                first["labels"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["labels"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["labels"] for f in examples], dtype=dtype)
        if "protocol" in first and first["protocol"] is not None:
            label = (
                first["protocol"].item()
                if isinstance(first["protocol"], torch.Tensor)
                else first["protocol"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["protocol"] = torch.tensor([f["protocol"] for f in examples], dtype=dtype)
        if "flow_duration" in first and first["flow_duration"] is not None:
            label = (
                first["flow_duration"].item()
                if isinstance(first["flow_duration"], torch.Tensor)
                else first["flow_duration"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["flow_duration"] = torch.tensor([f["flow_duration"] for f in examples], dtype=dtype)
        for k, v in first.items():
            if (
                    k not in ("labels", "label_ids", "total_bursts", "protocol", "flow_duration")
                    and v is not None
                    and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
                else:
                    batch[k] = torch.tensor(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
        return batch
