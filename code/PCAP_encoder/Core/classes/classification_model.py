import torch
import wandb
import json
from contextlib import nullcontext
from tqdm import tqdm
from torch.optim import AdamW
import time
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from Core.functions.utils import scale_lr
from Core.classes.custom_models import (
    Attention_Luong,
    MultiClassification_head,
    ModelWithBottleneck,
)
from Core.functions.ml_functions import (
    create_scheduler,
    determine_n_checkpoints,
    reshape_loss,
    post_process,
    update_best_model,
)
from transformers import T5ForConditionalGeneration
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"
#T5ForConditionalGeneration._keys_to_ignore_on_load_unexpected = ["decoder.*"]


CHECKPOINT_PATIENCE = 500
CHECKPOINT_X_EPOCH = 1
GENERATION_IN_VALIDATION = True
WANDB = False


class Classification_model:
    def __init__(self, opts, tokenizer, dataset_trainval, dataset_test):
        self.lr = opts["lr"]
        self.batch_size = opts["batch_size"]
        self.device = torch.device("cuda" if opts["use_cuda"] else "cpu")
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.num_epochs = opts["epochs"]
        self.current_best_loss = float("inf")
        self.correct = 0
        self.total = 0
        self.fix_encoder = opts["fix_encoder"]
        self.class_dataset_trainval = dataset_trainval
        self.class_dataset_test = dataset_test
        self.tokenizer_obj = tokenizer
        self.best_classification_model = None
        self.best_encoder = None
        self.prediction = torch.Tensor()
        self.actual = torch.Tensor()
        self.dict_pkt_repr = {}
        self.custom_model = None


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
                config={
                    "learning_rate": opts["lr"],
                    "architecture": opts["model_name"],
                    "epochs": opts["epochs"],
                },
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            )
        self.n_classes, self.labels = self.class_dataset_trainval.get_classification_stats()
        self.defineModel(opts["bottleneck"], opts["pkt_repr_dim"], opts["use_pkt_reduction"], opts["finetuned_path_model"], opts["model_name"])
        if self.fix_encoder:
            for param in self.custom_model.parameters():
                param.requires_grad = False
        self.custom_model.remove_decoder()

        self.class_dataset_trainval.create_trainVal_sampler()
        self.train_model(logger)
        self.test_model(logger)
        if WANDB:
            wandb.finish()

    def defineModel(self, type_bottleneck, pkt_dim, use_pkt_reduction=False, model_finetuned_path="Empty", model_name="T5-base"):
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
        if self.fix_encoder:
            for param in pretrained_model.parameters():
                param.requires_grad = False
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
        if model_finetuned_path != "Empty":
            self.custom_model.load_state_dict(torch.load(f"{model_finetuned_path}/weights.pth"), strict=False)
        self.classification_head = MultiClassification_head(
            pkt_dim, self.n_classes
        )

    def train_model(self, logger):
        """
        train_model
        -----------
        Prepares the elements necessary to perform the training steps.
        Args
            - logger (Logger)
        """
        train_loader = DataLoader(
            self.class_dataset_trainval,
            batch_size=self.batch_size,
            sampler=self.class_dataset_trainval.get_train_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        val_loader = DataLoader(
            self.class_dataset_trainval,
            batch_size=self.batch_size,
            sampler=self.class_dataset_trainval.get_val_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.lr = scale_lr(self.lr)
        if not self.fix_encoder:
            self.optimizer = AdamW(
                [
                    {
                        "params": self.classification_head.parameters(),
                    },
                    {"params": self.custom_model.parameters()},
                ],
                lr=self.lr,
            )
        else:
            self.optimizer = AdamW(self.classification_head.parameters(), lr=self.lr)
        lr_scheduler = create_scheduler(
            optimizer=self.optimizer,
            train_dataloader=train_loader,
            epochs=self.num_epochs,
        )
        logger.register_optimizer_scheduler(self.optimizer, lr_scheduler)
        lr_scheduler, train_loader, val_loader = self.accelerator_train_load(
            lr_scheduler, train_loader, val_loader, logger
        )
        self.checkpoint_id = 0
        self.start_training(
            logger=logger,
            classification_model=self.classification_head,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.num_epochs,
        )

    def start_training(
        self,
        logger,
        classification_model,
        lr_scheduler,
        train_loader,
        val_loader,
        num_epochs,
    ):
        """
        start_training
        --------------
        Starts the training loop.

        Args
            - logger (Logger)
            - classification_model (MultiClassification_head)
            - lr_scheduler
            - train_loader, val_loader (DataLoader) -- for train and validation respectively
            - num_epochs (int)
        """
        training_progress_bar = tqdm(
            range(len(train_loader) * (num_epochs - 1)),
            disable=not logger.accelerator.is_local_main_process,
        )
        checkpoint_steps = determine_n_checkpoints(
            train_dataloader=train_loader,
            checkpoints_x_epoch=CHECKPOINT_X_EPOCH,
            checkpoints_patience=CHECKPOINT_PATIENCE,
        )
        self.best_classification_model = update_best_model(
            model=classification_model, accelerator=logger.accelerator
        )
        if not self.fix_encoder:
            self.best_encoder = update_best_model(
                model=self.custom_model, accelerator=logger.accelerator
            )
        training_val_time = time.time()
        for epoch_id in range(num_epochs):
            self.training_batch(
                classification_model=classification_model,
                training_dataloader=train_loader,
                validation_dataloader=val_loader,
                epoch_id=epoch_id,
                lr_scheduler=lr_scheduler,
                logger=logger,
                progress_bar=training_progress_bar,
                checkpoint_steps=checkpoint_steps,
            )
        training_val_time = time.time() - training_val_time
        with open(f"./evaluation/{self.current_experiment}.json", "a") as results_file:
            json.dump({"total_loop":training_val_time}, results_file)
            results_file.write("\n")
        logger.accelerator.wait_for_everyone()

    def training_batch(
        self,
        classification_model,
        training_dataloader,
        validation_dataloader,
        epoch_id,
        lr_scheduler,
        logger,
        progress_bar,
        checkpoint_steps,
    ):

        batch_losses = []
        step = 0
        loss_fct = CrossEntropyLoss()
        train_time = time.time()
        train_model_time = 0
        for step_indexes, batch in training_dataloader:
            current_step = (
                epoch_id * len(training_dataloader) + step + 1
            )
            if current_step % checkpoint_steps == 0:
                train_time = time.time() - train_time
                val_time = time.time()
                classification_model.eval()
                self.custom_model.eval()
                losses, _ = self.validation_batch(
                    classification_model=classification_model,
                    logger=logger,
                    loader=validation_dataloader,
                    epoch_id=epoch_id
                )
                val_time = time.time() - val_time
                self.send_data_wandb(losses, epoch_id, False, total_train_time=train_time, model_train_time=train_model_time, total_val_time=val_time )
                losses = torch.Tensor(losses)
                lr_scheduler.step()
                if logger.accelerator.is_main_process:
                    self.is_best_model(
                        loss=torch.mean(losses),
                        accelerator=logger.accelerator,
                        logger=logger,
                        model=classification_model,
                        checkpoint_id=self.checkpoint_id,
                    )
                if logger.accelerator.is_main_process:
                    self.checkpoint_id += 1
            classification_model.train()
            if not self.fix_encoder:
                self.custom_model.train()
            with logger.accelerator.accumulate(classification_model), (
                logger.accelerator.accumulate(self.custom_model)
                if self.fix_encoder == False
                else nullcontext()
            ):
                
                if epoch_id == 0 or not self.fix_encoder:
                    ### ENCODER
                    model_time = time.time()
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    model_outputs = self.custom_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    packet_representation = model_outputs[:, 0, :]
                    train_model_time += (time.time() - model_time)
                    if self.fix_encoder:
                        for i in range(self.batch_size):
                            self.dict_pkt_repr[int(step_indexes[i])] = (
                                post_process(packet_representation[i, :])
                            )

                else:
                    packet_representation = torch.stack(
                        [self.dict_pkt_repr[int(index)] for index in step_indexes]
                    )
                model_time = time.time()
                label_prob, _ = classification_model(packet_representation.to(self.device))
                ### LOSS COMPUTATION
                loss = loss_fct(label_prob, batch["label_class"])
                gathered_loss = post_process(logger.accelerator.gather(loss))
                gathered_loss = reshape_loss(gathered_loss)
                ### MODEL UPDATE
                logger.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_model_time += (time.time() - model_time)
                progress_bar.update(1)

            if current_step % checkpoint_steps == 0:
                logger.save_checkpoint(
                    epoch_id=epoch_id,
                    batch_id=(step),
                    validation_steps=epoch_id,
                )
            batch_losses.append(gathered_loss)
            step += 1
        if WANDB:
            wandb.log({"loss_train": sum(batch_losses) / len(batch_losses)})
        return batch_losses

    def validation_batch(self, classification_model, logger, loader, epoch_id=0):
        # Evaluation
        val_losses = []
        progress_bar = tqdm(
            range(len(loader)),
            disable=not logger.accelerator.is_local_main_process,
        )
        self.prediction = torch.Tensor()
        self.actual = torch.Tensor()
        loss_fct = CrossEntropyLoss()
        inf_time = 0
        for step_indexes, batch in loader:
            with torch.no_grad():
                ### ENCODER
                if epoch_id == 0 or not self.fix_encoder:
                    time_start = time.time()
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    model_outputs = self.custom_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    packet_representation = model_outputs[:, 0, :]
                    if epoch_id == 0:
                        inf_time += time.time() - time_start
                    if self.fix_encoder:
                        for i in range(self.batch_size):
                            self.dict_pkt_repr[int(step_indexes[i])] = (
                                post_process(packet_representation[i, :])
                            )

                else:
                    packet_representation = torch.stack(
                        [self.dict_pkt_repr[int(index)] for index in step_indexes]
                    )
                time_start = time.time()
                label_prob, _ = classification_model(packet_representation.to(self.device))
                if epoch_id == 0:
                    inf_time += time.time() - time_start
                loss = loss_fct(label_prob, batch["label_class"])
                gathered_loss = post_process(logger.accelerator.gather(loss))
                gathered_loss = reshape_loss(gathered_loss)

                val_losses.append(gathered_loss.item())
                self.prediction = torch.cat(
                    (self.prediction, torch.argmax(label_prob.cpu(), 1))
                )
                self.actual = torch.cat(
                    (self.actual, batch["label_class"].cpu())
                )

                progress_bar.update(1)
        return val_losses, inf_time

    def send_data_wandb(self, losses, epoch, conf_matrix=False, type_res="val", inf_time=None, total_train_time=None, model_train_time=None, total_val_time=None, test_time=None):
        """
        send_data_wandb
        ---------------
        Sends the data of the experiment to WandB

        Args
            - losses (list)
            - type_res (string) -- identified if the metrics are for validation or test
            - confusion_matrix (bool) -- if the confusion matrix is needed or not
        """
        if WANDB:
            wandb.log({f"{type_res}_loss_validation": sum(losses) / len(losses)})
            wandb.log(
                {
                    f"{type_res}_Accuracy": self.compute_accuracy(
                        self.prediction.int(), self.actual.int()
                    )
                }
            )
            wandb.log(
                {
                    f"{type_res}_f1_macro": f1_score(
                        self.actual.int(), self.prediction.int(), average="macro"
                    )
                }
            )
            if conf_matrix:
                wandb.sklearn.plot_confusion_matrix(
                    self.actual.int().tolist(),
                    self.prediction.int().tolist(),
                    labels=self.labels,
                )
        else:
            
            results = {
                "epoch": epoch,
                "loss": sum(losses) / len(losses), 
                "accuracy": float(self.compute_accuracy(
                        self.prediction.int(), self.actual.int()
                    )), 
                "f1_score_macro": float(f1_score(
                        self.actual.int(), self.prediction.int(), average="macro")),
                "f1_score_micro": float(f1_score(
                        self.actual.int(), self.prediction.int(), average="micro")),
                "total_train_time": total_train_time,
                "model_train_time": model_train_time,
                "total_val_time": total_val_time,
                }
            if inf_time is not None:
                results["inference_time"] = inf_time
                results["test_time"] = test_time
            if conf_matrix:
                results["conf_matrix"] = confusion_matrix(self.actual.int(), self.prediction.int()).tolist()
            os.makedirs("./evaluation", exist_ok=True)
            with open(f"./evaluation/{self.current_experiment}.json", "a") as results_file:
                json.dump(results, results_file)
                results_file.write("\n")

    def test_model(self, logger, opts=None):
        logger.accelerator.print(f"End training...")
        
        if self.custom_model == None:
            self.defineModel(opts["bottleneck"], opts["pkt_repr_dim"], opts["use_pkt_reduction"], opts["finetuned_path_model"], opts["model_name"])
        logger.accelerator.print(f"Start testing...")
        self.class_dataset_test.create_test_sampler()
        self.test_loader = DataLoader(
            self.class_dataset_test,
            batch_size=self.batch_size,
            sampler=self.class_dataset_test.get_test_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.test_loader = logger.accelerator.prepare(self.test_loader)
        self.classification_head.eval()
        self.custom_model.eval()
        test_time = time.time()
        losses, inf_time = self.validation_batch(
            self.classification_head, logger, self.test_loader
        )
        self.send_data_wandb(losses, "test", False, "test", inf_time, test_time=time.time()-test_time)

    def compute_accuracy(self, prediction, actual):
        correct = torch.eq(prediction, actual).int().sum()
        return correct / len(actual)

    def accelerator_train_load(self, lr_scheduler, train_loader, val_loader, logger):
        """
        accelerator_train_load
        ----------------------
        Prepares the training elements by calling 'logger.accelerator.prepare' which 
        handles device placement and other configurations for efficient training.
        Args
            - lr_scheduler 
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - logger (Logger)
        """
        (
            self.classification_head,
            self.custom_model,
            self.optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
        ) = logger.accelerator.prepare(
            self.classification_head,
            self.custom_model,
            self.optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
        )

        return lr_scheduler, train_loader, val_loader

    def is_best_model(self, loss, accelerator, logger, model, checkpoint_id):
        """
        is_best_model
        -------------
        Updates the best model if the current avg. batch loss is smaller than the current best.

        Arguments:
            loss (float) -- Average loss inter-batch.
            accelerator (Accelerator) -- accelerator to update best model if it's better than previous
            logger (Logger) -- Logger to log the results.
            model (ModelForMLModelling) -- Current model that produced the avg_loss
            checkpoint_id (int) -- Validation checkpoint. Will act as a step for the logger.
        """
        updated = 0
        if loss <= self.current_best_loss:
            self.best_classification_model = update_best_model(
                model=model, accelerator=accelerator
            )
            self.current_best_loss = loss
            logger.save_best_huggingface_model(model=model, type="custom_model")
            if not self.fix_encoder:
                self.best_encoder = update_best_model(
                    model=self.custom_model, accelerator=logger.accelerator
                )
                logger.save_best_huggingface_model(model=self.custom_model, type="custom_encoder")
            updated = 1
        logger.log_result(key="best_checkpoint", value=updated, step=checkpoint_id)

