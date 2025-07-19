import torch
import wandb
from tqdm import tqdm
import time
import pandas as pd
import multiprocessing
from torch.utils.data import DataLoader
from Core.classes.custom_models import (
    Attention_Luong,
    MultiClassification_head,
    ModelWithBottleneck,
)
from transformers import T5ForConditionalGeneration
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"
# T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]


CHECKPOINT_PATIENCE = 500
CHECKPOINT_X_EPOCH = 1
GENERATION_IN_VALIDATION = True
WANDB = True


class Classification_model:
    def __init__(self, opts, tokenizer, dataset_test):
        self.batch_size = opts["batch_size"]
        self.device = torch.device("cuda" if opts["use_cuda"] else "cpu")
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.class_dataset_test = dataset_test
        self.tokenizer_obj = tokenizer
        self.prediction = torch.Tensor()
        self.actual = torch.Tensor()
        self.dict_pkt = []


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
        self.compute_pkt_repr(logger)
        self.save_representation_parquet(opts["experiment"], opts["identifier"])
        if WANDB:
            wandb.finish()

    def save_representation_parquet(self, experiment, identifier):
        """
        save_representation_parquet
        ---------------------------
        Save the input dataframe as a new parquet file with a new column "pkt_repr"
        """
        if not os.path.exists(os.path.join('results/', experiment)):
            os.makedirs(os.path.join('results/', experiment))
        df = pd.DataFrame(self.dict_pkt)
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

    def validation_batch(self, logger, loader, flow_level=None):
        # Evaluation
        progress_bar = tqdm(
            range(len(loader)),
            disable=not logger.accelerator.is_local_main_process,
        )
        for step_indexes, batch in loader:
            with torch.no_grad():
                ### ENCODER
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                model_outputs = self.custom_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                packet_representation = model_outputs[:, 0, :]
                label_prob, pkt_repr_a = self.classification_head(packet_representation.to(self.device))
                for i in range(self.batch_size):
                    step_ind = int(step_indexes[i])
                    pkt_repr_bef=packet_representation[i, :]
                    pkt_repr_after=pkt_repr_a[i]
                    row = self.class_dataset_test.data.loc[step_ind]
                    pkt_info = {
                        "class": row["class"],
                        "type_q": row.type_q,
                        "context": row.context,
                        "prediction": int(torch.argmax(label_prob[i].cpu(), 0)),
                        "pkt_repr_before": pkt_repr_bef.tolist(),
                        "pkt_repr_after": pkt_repr_after.tolist(),
                        "question": row.question
                    }
                    self.dict_pkt.append(pkt_info)
                progress_bar.update(1)


    def compute_pkt_repr(self, logger):
        logger.accelerator.print(f"Start testing...")
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
        self.custom_model.eval()
        self.classification_head.eval()
        self.validation_batch(logger, self.test_loader)



