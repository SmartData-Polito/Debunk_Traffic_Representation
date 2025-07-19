import torch
import pandas as pd
from torch.utils.data import Dataset, RandomSampler
from Core.classes.dataset_for_QA import QA_Dataset
from Core.functions.utils import add_noise
from sklearn.model_selection import train_test_split


class Denoiser_Dataset(QA_Dataset):
    def __init__(self, opts, tokenizer, noise_percentage):
        super().__init__(opts=opts,tokenizer=tokenizer)
        self.noise_percentage = noise_percentage

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        context = self.context[idx]
        question = self.questions[idx]
        if idx not in self.validation_range:
            question_tokenized = self.tokenizer.tokenize_question(question, context)
            if  self.noise_percentage > 0:
                noise_question_tokenized = add_noise(question_tokenized['input_ids'], self.tokenizer.vocab_size, self.noise_percentage)
            else:
                noise_question_tokenized = question_tokenized['input_ids']
        
        # If in validation the noise must be equal for all the epochs
        else:
            noise_question_tokenized = self.val_questions[idx][1]
            question_tokenized = self.val_questions[idx][0]
        
        decoder_input_ids = [0] + noise_question_tokenized
        decoder_input_ids = torch.tensor(
            decoder_input_ids[: self.t_len], dtype=torch.long
        )
        labels = torch.tensor(question_tokenized["input_ids"], dtype=torch.long)

        labels[labels == 0] = -100

        return idx, {
            "input_ids": torch.tensor(
                noise_question_tokenized, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ),
            "decoder_input_ids": decoder_input_ids,
        }


    def load_dataset(self, trainval_path, test_path, format):
        """
        load_dataset
        ------------
        The method is used to load the data in the class.
        Args:
            - input_path: must be a csv file were a pandas dataframe is stored
        """
        self.data = pd.read_parquet(trainval_path)
        self.test_set = pd.read_parquet(test_path)
        self.data["context"] = [pkt.replace(" ", "") for pkt in self.data.context]
        self.test_set["context"] = [pkt.replace(" ", "") for pkt in self.test_set.context]
        if format == "every4":
            self.data["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.data.context]
            self.test_set["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.test_set.context]
        elif format == "every2":
            self.data["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.data.context]
            self.test_set["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.test_set.context]
        self.data = self.data
        self.context = self.data["context"]
        self.questions = self.data["question"]


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
        data_f = self.data.sample(n=num_rows, random_state=self.seed)
        self.train_data, self.val_data = train_test_split(
            data_f, test_size=0.2, random_state=self.seed
        )

        self.validation_range = self.val_data.index
        self.size_val = len(self.val_data)
        self.size_train = len(self.train_data)
        self.train_sampler = RandomSampler(self.train_data.index)
        self.val_sampler = RandomSampler(self.val_data.index)
        self.test_sampler = RandomSampler(self.test_set.index)
        self.val_questions = self.compute_one_time_noise()


    def compute_one_time_noise(self):
        dict_changed_questions = {}
        for i in self.val_data.index:
            question_tokenized = self.tokenizer.tokenize_question(self.data["question"].iloc[i], self.data["context"].iloc[i])
            dict_changed_questions[i] = [question_tokenized, add_noise(question_tokenized['input_ids'] ,self.tokenizer.vocab_size, self.noise_percentage)]

        return dict_changed_questions
        

    def get_train_sampler(self):
        return self.train_sampler

    def get_val_sampler(self):
        return self.val_sampler

    def get_test_sampler(self):
        return self.test_sampler

