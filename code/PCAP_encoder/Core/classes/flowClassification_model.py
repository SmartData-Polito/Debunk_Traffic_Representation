import torch
import wandb
from tqdm import tqdm
import json
import pandas as pd
import time
import multiprocessing
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from Core.functions.utils import scale_lr
from Core.classes.dataset_for_flowClassification import Flow_Classification_Dataset
from Core.classes.custom_models import (
    Attention_Luong,
    MultiClassification_head,
    ModelWithBottleneck,
    MultiClassification_head_flow
)

from Core.functions.utils import majority_vote, concatenate_pkt_repr
from sklearn.metrics import accuracy_score, f1_score
from transformers import T5ForConditionalGeneration
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"
# T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]


CHECKPOINT_PATIENCE = 500
CHECKPOINT_X_EPOCH = 1
GENERATION_IN_VALIDATION = True
WANDB = True


class Flow_Classification_Model:
    def __init__(self, opts, tokenizer, dataset_test, dataset_trainval=None):
        self.batch_size = opts["batch_size"]
        self.device = torch.device("cuda" if opts["use_cuda"] else "cpu")
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.tokenizer_obj = tokenizer
        self.class_dataset_test = dataset_test
        self.flow_level = opts["flow_level"]
        if self.flow_level == "representation_concat":
            self.class_dataset_train_val = dataset_trainval
            self.num_epochs = opts["epochs"]
            self.lr = opts["lr"]



    def run(self, logger, opts):
        """
        run
        ---
        Performs the training and testing on the classification head.

        Args
            - logger (Logger) -- to log the results
            - opts (dict) -- contains all the parameters of the training.
        """
        self.current_experiment = opts["experiment"]+opts["identifier"]
        if WANDB:
            wandb.init(
                project=opts["experiment"],
                name=opts["identifier"],
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            )
        self.n_classes, self.labels = self.class_dataset_test.get_classification_test_stats()
        self.defineModel(type_bottleneck=opts["bottleneck"], pkt_dim=opts["pkt_repr_dim"], use_pkt_reduction=opts["use_pkt_reduction"], model_finetuned_path=opts["finetuned_path_model"], bottleneck_finetuned_path=opts["finetuned_path_bottleneck"], classification_finetuned_path=opts["finetuned_path_classification"],model_name="T5-base")
        self.custom_model.remove_decoder()
        inf_time = self.compute_pkt_repr(logger)
        if self.flow_level == "majority_vote":
            self.predict_flow_majorityvote(inf_time)
        elif self.flow_level == "representation_concat":
            self.predict_flow_representationconcat(opts["pkts_in_flow"])
        #self.save_representation_parquet(opts["experiment"], opts["identifier"])
        if WANDB:
            wandb.finish()

    def save_representation_parquet(self, experiment, identifier, dict_pkt):
        """
        save_representation_parquet
        ---------------------------
        Save the input dataframe as a new parquet file with a new column "pkt_repr"
        """
        if not os.path.exists(os.path.join('results/', experiment)):
            os.makedirs(os.path.join('results/', experiment))
        df = pd.DataFrame(dict_pkt)
        df.to_parquet(f"{os.path.join('results/', experiment, identifier)}.parquet")

    def defineModel(self, type_bottleneck, pkt_dim, use_pkt_reduction=False, model_finetuned_path="Empty", bottleneck_finetuned_path="Empty", classification_finetuned_path="Empty", model_name="T5-base"):
        """
        defineModel
        -----------
        Instatiator for the 'ModelWithBottleneck' object depending on the bottleneck 
        selected.

        Args 
            - logger (Logger) -- to log the results
            - model_finetuned_path (str) -- path to a pre-trained 'ModelWithBottleneck'
                                            model, default 'Empty' 
            - model_name (str) -- name of the original model, default 'T5-base'
            
        """
        pretrained_model = T5ForConditionalGeneration.from_pretrained(
            model_name, return_dict=True
        )
        # If the bottleneck is NOT trainable
        if type_bottleneck in ["none", "first", "mean"]:
            self.custom_model = ModelWithBottleneck(
                pretrained_model,
                type_bottleneck,
                pkt_dim,
                use_pkt_reduction,
                pretrained_model.decoder
            )
        # If the bottleneck is trainable
        else:
            if type_bottleneck == "Luong":
                bottleneck_model = Attention_Luong(pretrained_model.config.d_model)
            # Default trainable bottleneck is Luong attention
            else:
                bottleneck_model = Attention_Luong(pretrained_model.config.d_model)
                self.custom_model = ModelWithBottleneck(
                    pretrained_model,
                    type_bottleneck,
                    pkt_dim,
                    use_pkt_reduction,
                    pretrained_model.decoder,
                    bottleneck_model,
                )
            if bottleneck_finetuned_path != "Empty":
                self.custom_model.load_state_dict(
                    torch.load(f"{bottleneck_finetuned_path}/weights.pth"),
                    strict=False
                )
        if model_finetuned_path != "Empty":
            self.custom_model.load_state_dict(
                torch.load(f"{model_finetuned_path}/weights.pth"),
                strict=False
            )
        self.classification_head = MultiClassification_head(
            pkt_dim, self.n_classes
        )
        if classification_finetuned_path != "Empty":
            self.classification_head.load_state_dict(
                torch.load(f"{classification_finetuned_path}/weights.pth"),
                strict=False
            )

    def validation_batch(self, logger, loader):
        # Evaluation
        progress_bar = tqdm(
            range(len(loader)),
            disable=not logger.accelerator.is_local_main_process,
        )
        dict_pkt = []
        total_time = 0
        for step_indexes, batch in loader:
            with torch.no_grad():
                ### ENCODER
                start_time = time.time()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                model_outputs = self.custom_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                packet_representation = model_outputs[:, 0, :]
                label_prob, _ = self.classification_head(packet_representation.to(self.device))
                total_time += time.time() - start_time
                for i in range(self.batch_size):
                    step_ind = int(step_indexes[i])
                    pkt_repr_bef=packet_representation[i, :]
                    row = self.class_dataset_test.data.loc[step_ind]
                    pkt_info = {
                        "class": row["class"],
                        "type_q": row.type_q,
                        "context": row.context,
                        "position_pkt":row.position_in_flow,
                        "prediction": int(torch.argmax(label_prob[i].cpu(), 0)),
                        "pkt_repr_before": pkt_repr_bef.tolist(),
                        "flow":row.flow
                    }
                    dict_pkt.append(pkt_info)
                progress_bar.update(1)
        return dict_pkt, total_time


    def compute_pkt_repr(self, logger):
        logger.accelerator.print(f"Start {self.flow_level} flow classification ...")
        if self.flow_level == "majority_vote":
            self.prepare_for_majority_vote(logger)
            self.custom_model.eval()
            self.classification_head.eval()
            dict_pkt_test, inf_time = self.validation_batch(logger, self.test_loader)
            self.df_test = pd.DataFrame(dict_pkt_test)
            self.free_gpu()
            return inf_time

        elif self.flow_level == "representation_concat":
            self.prepare_for_representation_concat(logger)
            dict_pkt_test, _ = self.validation_batch(logger, self.test_loader )
            dict_pkt_train, _ = self.validation_batch(logger, self.train_loader)
            dict_pkt_val, inf_time = self.validation_batch(logger, self.val_loader)
            self.df_test = pd.DataFrame(dict_pkt_test)
            self.df_train = pd.DataFrame(dict_pkt_train)
            self.df_val = pd.DataFrame(dict_pkt_val)
            self.free_gpu()
        return inf_time

    def predict_flow_majorityvote(self, inf_time):
        self.df_test = self.df_test.groupby('flow').agg({
            'prediction': majority_vote,
            'class': 'first'
        }).reset_index()
        self.report_results(self.df_test["class"], self.df_test["prediction"], inf_time, "test")
    
    def predict_flow_representationconcat(self, pkts_in_flow):
        # In column "pkt_repr_before" we have the concatenation
        self.df_train = self.df_train.groupby(['flow',"class", "type_q"])['pkt_repr_before'].agg(lambda x: concatenate_pkt_repr(x,pkts_in_flow)).reset_index()
        self.df_val = self.df_val.groupby(['flow',"class", "type_q"])['pkt_repr_before'].agg(lambda x: concatenate_pkt_repr(x,pkts_in_flow)).reset_index()
        self.df_test = self.df_test.groupby(['flow',"class", "type_q"])['pkt_repr_before'].agg(lambda x: concatenate_pkt_repr(x,pkts_in_flow)).reset_index()
        self.df_train.rename(columns={"pkt_repr_before":"flow_representation"}, inplace=True)
        self.df_val.rename(columns={"pkt_repr_before":"flow_representation"}, inplace=True)
        self.df_test.rename(columns={"pkt_repr_before":"flow_representation"}, inplace=True)
        class_model = self.train_flow_representationconcat()
        self.test_flow_representationconcat(class_model)
    
    def report_results(self, true_labels, predicted_labels, inf_time=None, epoch=None, losses=None):
        results = {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)), 
            "f1_score": float(f1_score(true_labels, predicted_labels, average='macro')),
            "epoch": epoch,
            "inf_time": inf_time }
        if losses is not None:
            results["loss"] = float(sum(losses) / len(losses))
        os.makedirs("./evaluation", exist_ok=True)
        with open(f"./evaluation/{self.current_experiment}.json", "a") as results_file:
            json.dump(results, results_file)
            results_file.write("\n")
    
    def free_gpu(self):
        self.custom_model = self.custom_model.cpu()
        self.classification_head = self.classification_head.cpu()

    def prepare_for_majority_vote(self, logger):
        self.class_dataset_test.create_test_sampler()
        self.test_loader = DataLoader(
            self.class_dataset_test,
            batch_size=self.batch_size,
            sampler=self.class_dataset_test.get_test_sampler(),
            num_workers=multiprocessing.cpu_count() - 2,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.test_loader, self.custom_model, self.classification_head = logger.accelerator.prepare(self.test_loader, self.custom_model, self.classification_head)
    
    def prepare_for_representation_concat(self, logger):
        self.class_dataset_test.create_test_sampler()
        self.class_dataset_train_val.create_trainVal_sampler()
        self.test_loader = DataLoader(
            self.class_dataset_test,
            batch_size=self.batch_size,
            sampler=self.class_dataset_test.get_test_sampler(),
            num_workers=multiprocessing.cpu_count() - 2,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.train_loader = DataLoader(
            self.class_dataset_train_val,
            batch_size=self.batch_size,
            sampler=self.class_dataset_train_val.get_train_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.val_loader = DataLoader(
            self.class_dataset_train_val,
            batch_size=self.batch_size,
            sampler=self.class_dataset_train_val.get_val_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.train_loader, self.val_loader, self.test_loader, self.custom_model, self.classification_head = logger.accelerator.prepare(self.train_loader, self.val_loader, self.test_loader, self.custom_model, self.classification_head)

    def train_flow_representationconcat(self):
        train_dataset = Flow_Classification_Dataset(self.df_train)
        val_dataset = Flow_Classification_Dataset(self.df_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True, shuffle=False)
        model = MultiClassification_head_flow(train_dataset.get_repr_flow_dim(), train_dataset.get_num_classes()).to(self.device)
        self.lr = scale_lr(self.lr)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        loss_fct = CrossEntropyLoss()
        
        for epoch in range(self.num_epochs):
            # Training step
            progress_bar = tqdm(
            range(len(train_loader))
            )
            model.train()
            for idx, batch in train_loader:
                batch["input"] = batch["input"].to(self.device)
                batch["class_label"] = batch["class_label"].to(self.device)
                optimizer.zero_grad()
                outputs, _ = model(batch["input"])
                loss = loss_fct(outputs, batch["class_label"])
                loss.backward()
                optimizer.step()
                progress_bar.update(1)

            # Validation step
            model.eval()
            val_loss = []
            with torch.no_grad():
                for idx, batch  in val_loader:
                    batch["input"] = batch["input"].to(self.device)
                    batch["class_label"] = batch["class_label"].to(self.device)
                    val_outputs, _ = model(batch["input"])
                    val_loss.append(loss_fct(val_outputs, batch["class_label"]).item())
                    _, predicted = torch.max(val_outputs, 1)
            self.report_results(batch["class_label"].cpu(), predicted.cpu(), epoch=epoch, losses=val_loss)
        return model

    def test_flow_representationconcat(self, model):
        test_dataset = Flow_Classification_Dataset(self.df_val)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
        loss_fct = CrossEntropyLoss()
        model.eval()
        val_loss = []
        with torch.no_grad():
            for idx, batch  in test_loader:
                batch["input"] = batch["input"].to(self.device)
                batch["class_label"] = batch["class_label"].to(self.device)
                val_outputs, _ = model(batch["input"])
                val_loss.append(loss_fct(val_outputs, batch["class_label"]).item())
                _, predicted = torch.max(val_outputs, 1)
        self.report_results(batch["class_label"].cpu(), predicted.cpu(), "test", val_loss)