import torch
import pandas as pd
from torch.utils.data import Dataset, RandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class Classification_Dataset(Dataset):
    def __init__(self, opts, tokenizer):
        self.tokenizer = tokenizer
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.seed = opts["seed"]
        self.batch = opts["batch_size"]
        # In validation, loss is not needed
        if "loss" in opts.keys():
            self.type_loss = opts["loss"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # question = self.questions[idx]
        question = self.questions
        context = self.context[idx]
        answer = self.answer[idx]
        question_tokenized = self.tokenizer.tokenize_question(question, context)
        return idx, {
            "input_ids": torch.tensor(
                question_tokenized["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ),
            "label_class": torch.tensor(answer)
        }

    def load_dataset(self, type, input_path, format_input, input_validation="", percentage=100, pkts_in_flow=5):
        """
        load_dataset
        ------------
        The method is used to load the data in the class.
        Args:
            - input_path: must be a csv file were a pandas dataframe is stored
        """
        self.train_data = pd.read_parquet(input_path)
        self.train_data = self.train_data[self.train_data.position_in_flow < pkts_in_flow].reset_index(drop=True)
        self.train_data["context"] = [pkt.replace(" ", "") for pkt in self.train_data.context]
        if format_input == "every4":
            self.train_data["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.train_data.context]
        elif format_input == "every2":
            self.train_data["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.train_data.context]
        num_rows_train = int(len(self.train_data) * percentage / 100)
        self.train_data = self.train_data.sample(
            n=num_rows_train, random_state=self.seed
        )
        multiple_batch = int(len(self.train_data) / self.batch)
        self.train_data = self.train_data[: multiple_batch * self.batch]
        self.size_train = len(self.train_data)
        self.train_data = self.train_data.reset_index()

        if input_validation != "":
            self.val_data = pd.read_parquet(input_validation)
            self.val_data = self.val_data[self.val_data.position_in_flow < pkts_in_flow].reset_index(drop=True)
            self.val_data["context"] = [pkt.replace(" ", "") for pkt in self.val_data.context]
            if format_input == "every4":
                self.val_data["context"] = [''.join([str(pkt[i:i+4])+' ' for i in range(0, len(pkt), 4)]).strip() for pkt in self.val_data.context]
            elif format_input == "every2":
                self.val_data["context"] = [''.join([str(pkt[i:i+2])+' ' for i in range(0, len(pkt), 2)]).strip() for pkt in self.val_data.context]
            self.val_data.index += self.size_train
            num_rows_val = int(len(self.val_data) * percentage / 100)
            self.val_data = self.val_data.sample(n=num_rows_val, random_state=self.seed)
            multiple_batch = int(len(self.val_data) / self.batch)
            self.val_data = self.val_data[: multiple_batch * self.batch]
            self.size_val = len(self.val_data)

            self.data = pd.concat([self.train_data, self.val_data], axis=0)
        elif type == "Train":
            self.data = self.train_data
            self.train_data, self.val_data = train_test_split(
                self.data, test_size=0.2, random_state=self.seed
            )
        else:
            self.data = self.train_data
        #self.questions = self.data["question"].iloc[0]
        self.questions = self.data["question"].iloc[0]
        self.context = self.data["context"]
        self.answer = self.data["class"]
        if hasattr(self, 'type_loss'):
            self.retrieveCardinality()

    def create_test_sampler(self):
        self.test_sampler = RandomSampler(self.data.index)

    def retrieveTypes(self):
        elems = self.data["type_q"].unique()
        return elems.tolist()
    
    def retrieveCardinality(self):
        elems = self.data["class"].value_counts()
        if self.type_loss == "weighted":
            total_samples = sum(elems)
            class_weights = [pow(1 - count / total_samples, 2) for count in elems]
            self.weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.weights = torch.tensor([1 for i in range(len(elems))], dtype=torch.float32)
    
    def create_trainVal_sampler(self):
        self.train_sampler = RandomSampler(self.train_data.index)
        self.val_sampler = RandomSampler(self.val_data.index)

    # GETTERS
    def get_weights(self):
        return self.weights

    def get_train_sampler(self):
        return self.train_sampler

    def get_val_sampler(self):
        return self.val_sampler

    def get_test_sampler(self):
        return self.test_sampler

    def get_class_Byindex(self, index):
        return self.type_q.iloc[index]

    def get_if_time(self):
        return False
    
    def get_classification_stats(self):
        tmp = self.val_data.sort_values('class')
        labels = list(tmp["type_q"].drop_duplicates())
        return len(labels), labels

    def get_classification_test_stats(self):
        tmp = self.data.sort_values('class')
        labels = list(tmp["type_q"].drop_duplicates())
        return len(labels), labels
    

class Flow_Classification_Dataset():
    def __init__(self, df):
        self.class_flow = df["class"].values
        self.representation_flow = df["flow_representation"].values
        self.repr_flow_dim = max([len(repr) for repr in self.representation_flow])
        self.num_classes = max(list(set(self.class_flow)))+1
        #self.representation_flow = [inner_list + [0] * (self.repr_flow_dim - len(inner_list)) for inner_list in self.representation_flow]


    def __len__(self):
        return len(self.class_flow)

    def __getitem__(self, idx):
        return idx, {
            "class_label": torch.tensor(self.class_flow[idx], dtype=torch.long),
            "input": torch.tensor(self.representation_flow[idx])
        }

    def get_num_classes(self):
        return self.num_classes
    
    def get_repr_flow_dim(self):
        return self.repr_flow_dim