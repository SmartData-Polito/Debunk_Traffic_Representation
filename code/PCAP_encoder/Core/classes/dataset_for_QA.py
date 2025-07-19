import torch
import pandas as pd
from torch.utils.data import Dataset, RandomSampler

from sklearn.model_selection import train_test_split


class QA_Dataset(Dataset):
    def __init__(self, opts, tokenizer):
        self.tokenizer = tokenizer
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.seed = opts["seed"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]

        question_tokenized = self.tokenizer.tokenize_question(question, context)
        answer_tokenized = self.tokenizer.tokenize_answer(str(answer))
        decoder_input_ids = [0] + answer_tokenized["input_ids"]
        decoder_input_ids = torch.tensor(
            decoder_input_ids[: self.t_len], dtype=torch.long
        )
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)

        labels[labels == 0] = -100

        return idx, {
            "input_ids": torch.tensor(
                question_tokenized["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(
                answer_tokenized["attention_mask"], dtype=torch.long
            ),
            "decoder_input_ids": decoder_input_ids,
        }


    def load_dataset(self, input_path, test_path, format):
        """
        load_dataset
        ------------
        The method is used to load the data in the class.
        Args:
            - input_path: must be a csv file were a pandas dataframe is stored
        """
        self.data = pd.read_parquet(input_path)
        self.test_set = pd.read_parquet(test_path)
        self.data["context"] = [pkt.replace(" ", "") for pkt in self.data.context]
        self.test_set["context"] = [pkt.replace(" ", "") for pkt in self.test_set.context]
        if format == "every4":
            self.data["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.data.context]
            self.test_set["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.test_set.context]
        elif format == "every2":
            self.data["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.data.context]
            self.test_set["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.test_set.context]
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data["answer"]
        try:
            self.pkt_field = list(self.data["pkt_field"])
        except:
            pass

    def retrieveTypes(self):
        elems = self.data["pkt_field"].unique()
        return elems.tolist()

    def split_train_val_test(self, percentage):
        """
        split_train_val_test
        --------------------
        The method splits the dataset in three: train, validation and test, the
        test set is memorized on a file.
        Args:
            - opts
        """
        num_rows = int(len(self.data) * percentage / 100)
        self.data = self.data.sample(frac=1, random_state=self.seed)
        data_f = self.data.sample(n=num_rows, random_state=self.seed)
        self.train_data, self.val_data = train_test_split(
            data_f, test_size=0.2, random_state=self.seed
        )

        self.size_val = len(self.val_data)
        self.size_train = len(self.train_data)
        self.train_sampler = RandomSampler(self.train_data.index)
        self.val_sampler = RandomSampler(self.val_data.index)
        self.test_sampler = RandomSampler(self.test_set.index)

    def get_train_sampler(self):
        return self.train_sampler

    def get_val_sampler(self):
        return self.val_sampler

    def get_test_sampler(self):
        return self.test_sampler

    def get_class_Byindex(self, index):
        return self.pkt_field[index]
