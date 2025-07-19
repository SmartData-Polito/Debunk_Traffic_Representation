import os
import random
from typing import List, Union, Optional, Tuple
import itertools

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding
from datasets.formatting.formatting import LazyBatch

PROTOS_TO_LEN = {6: 18, 1: 13, 17: 12}  # TODO(maybe-hello-world): refactor


class NetFoundTokenizer(PreTrainedTokenizer):
    CLS_TOKEN = 65537
    PAD_TOKEN = 0
    mask_token = 65538
    vocab_size = 65539
    ATTN_PRESENCE_TOKEN = 1
    ATTN_ABSENCE_TOKEN = 0

    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.max_bursts = config.max_bursts
        self.max_burst_length = config.max_burst_length
        self.p = config.p
        self.pretraining = config.pretraining
        self.name_or_path = config.name_or_path
        self.limit_bursts = config.limit_bursts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, max_bursts={self.max_bursts}, max_burst_length={self.max_burst_length}, p={self.p})"
        )

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [self.CLS_TOKEN, self.PAD_TOKEN]

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ) -> Tuple[str]:
        return

    def __len__(self):
        return self.vocab_size

    def pad_bursts(
            self,
            flow: list[list[int]],
            max_burst_length: int,
            pad_token: Optional[int] = None
    ) -> np.ndarray:
        """
        Truncate each burst to `max_burst_length` and pad with token if necessary.
        """
        if pad_token is None:
            pad_token = self.PAD_TOKEN
        return np.array([
            burst[:max_burst_length] + [pad_token] * max((max_burst_length - len(burst)), 0)
            for burst in flow
        ])

    def pad_flow(self, flow, max_bursts: int, token: int = None):
        """
        Truncate the flow to `max_bursts` and pad with token if necessary.
        """
        if token is None:
            token = self.PAD_TOKEN

        pad_bursts = max(max_bursts - len(flow), 0)
        pads = [token] * len(flow[0]) * pad_bursts

        flow = list(itertools.chain.from_iterable(flow[:max_bursts]))  # flatten
        flow += pads
        return flow

    @staticmethod
    def prepend_to_list(flow: list[list[int]], token: Optional[int]) -> list[list[int]]:
        # Sometimes we prepend CLS_TOKEN or similar
        if token is not None:
            return [[token] + burst for burst in flow]
        else:
            return [[burst[0]] + burst for burst in flow]

    @staticmethod
    def convert_to_tokens(flow: list[list[int]], add_one: bool = False) -> list[list[int]]:
        if not add_one:
            return flow  # noop
        return [[tok + add_one for tok in burst] for burst in flow]

    @staticmethod
    def convert_to_attn(bursts):
        return [[1] * len(burst) for burst in bursts]

    def __call__(self, dataset):
        return self.tokenize(dataset)

    def trunc_flow(self, ls, idxs):
        return [
            ".".join(ls[i].split(".")[:idxs[i]]) + "."
            for i in range(len(ls))
        ]

    @staticmethod
    def _expand_bursts(flows: list[list[int]], burst_sizes: list[list[int]]) -> list[list[list[int]]]:
        """
        To save space, some repetitive info is stored as a single value for the entire burst.
        This function expands the burst sizes to match the actual burst lengths.
        """
        return [
            [
                [value] * burst_sizes[idx][i]
                for i, value in enumerate(flow)
            ]
            for idx, flow in enumerate(flows)
        ]

    @staticmethod
    def multiply_burst_values(flows: list[list[float]], multiplier: float, ftype=float) -> list[list[float]]:
        return [
            [ftype(burst_value * multiplier) for burst_value in flow]
            for flow in flows
        ]

    def tokenize(self, text, **kwargs):
        dataset: LazyBatch = text
        dataset['iats'] = self.multiply_burst_values(dataset['iats'], 1e-3, int)
        dataset_burst_sizes = [[len(burst) for burst in flow] for flow in dataset["burst_tokens"]]

        if not self.pretraining and "labels" in dataset:
            labels = np.array(dataset["labels"], dtype=int)
            labels = labels.tolist()

        # restore directions: true/false -> 1/-1
        direction = [[1 if direction else -1 for direction in flow] for flow in dataset["directions"]]
        direction = self.tokenize_fields(self._expand_bursts(direction, dataset_burst_sizes))

        pkt_bytes = self.tokenize_fields(self._expand_bursts(dataset["bytes"], dataset_burst_sizes))
        pkt_count = self.tokenize_fields(self._expand_bursts(dataset["counts"], dataset_burst_sizes))
        iats = self.tokenize_fields(self._expand_bursts(dataset["iats"], dataset_burst_sizes))
        input_ids, attention_mask = self.tokenize_fields_with_attn(
            dataset["burst_tokens"], prepend_token=self.CLS_TOKEN, add_one=True
        )
        total_bursts = [len(flow) for flow in dataset["burst_tokens"]]

        batchDict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "direction": direction,
            "bytes": pkt_bytes,
            "pkt_count": pkt_count,
            "iats": iats,
            "total_bursts": total_bursts,
            "flow_duration": dataset["flow_duration"],
            "protocol": dataset["protocol"],
        }
        if not self.pretraining and "labels" in dataset:
            batchDict.update({"labels": labels})

        return BatchEncoding(batchDict)

    def tokenize_fields(
            self,
            dataset: list[list[list[int]]],
            prepend_token: int = None,
            add_one: bool = False
    ) -> list[list[list[int]]]:
        tokenized_data = [
            self.pad_flow(
                self.pad_bursts(
                    self.prepend_to_list(self.convert_to_tokens(flow, add_one), prepend_token),
                    self.max_burst_length,
                ),
                self.max_bursts,
            )
            for flow in dataset
        ]

        return tokenized_data

    def tokenize_fields_with_attn(
            self,
            dataset: list[list[list[int]]],
            prepend_token: int = None,
            add_one: bool = False
    ) -> Tuple[list[list[list[int]]], list[list[list[int]]]]:
        tokenized_data = self.tokenize_fields(dataset, prepend_token, add_one)
        attn = [
            self.pad_flow(
                self.pad_bursts(
                    self.prepend_to_list(self.convert_to_attn(flow), self.ATTN_PRESENCE_TOKEN),
                    max_burst_length=self.max_burst_length,
                    pad_token=self.ATTN_ABSENCE_TOKEN
                ),
                max_bursts=self.max_bursts,
                token=self.ATTN_ABSENCE_TOKEN
            )
            for flow in dataset
        ]
        return tokenized_data, attn
