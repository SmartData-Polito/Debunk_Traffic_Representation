import torch
import wandb
import os
import json
import torch.nn.functional as F
import numpy as np
import random as rnd
import multiprocessing
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoModel
from torch.nn import CrossEntropyLoss
from Core.functions.utils import scale_lr, generate_experiment_report
from Core.classes.custom_models import Attention_Luong, ModelWithBottleneck
from Core.functions.ml_functions import (
    create_scheduler,
    determine_n_checkpoints,
    reshape_loss,
    post_process,
    compute_perplexity,
    update_best_model,
)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

CHECKPOINT_PATIENCE = 500
CHECKPOINT_X_EPOCH = 1
WANDB = False


class T5_PCAP_translator:
    def __init__(self, opts, tokenizer, dataset):

        self.lr = opts["lr"]
        self.num_epochs = opts["epochs"]
        self.batch_size = opts["batch_size"]
        self.device = torch.device("cuda" if opts["use_cuda"] else "cpu")
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.model = None
        self.current_best_loss = float("inf")
        self.qa_dataset = dataset
        self.type_pretrain = opts["type_pretrain"]
        if self.type_pretrain == "QA":
            distinct_types = self.qa_dataset.retrieveTypes()
            self.total = {label: 0 for label in distinct_types}
            self.correct = {label: 0 for label in distinct_types}
        elif self.type_pretrain == "denoiser":
            self.total = 0
            self.correct = 0
        self.tokenizer_obj = tokenizer
        self.best_model = None

    def run(self, logger, opts):
        """
        run
        ---
        Performs the training and testing on the classification head.

        Args
            - logger (Logger) -- to log the results
            - opts (dict) -- contains all the parameters of the training.
        """
        if WANDB and logger.accelerator.is_local_main_process:
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
        self.current_experiment = opts["experiment"] + opts["identifier"]
        self.split_dataset(opts["percentage"])
        self.defineModel(
            opts["bottleneck"],
            opts["pkt_repr_dim"],
            opts["use_pkt_reduction"],
            opts["finetuned_path_model"],
            opts["model_name"],
        )
        self.train_model(logger)
        self.test_model(logger)
        if self.type_pretrain == "QA":
            self.generate_test_report(opts)
        if WANDB:
            wandb.finish()

    def defineModel(
        self,
        type_bottleneck,
        pkt_dim,
        use_pkt_reduction,
        model_finetuned_path,
        model_name,
    ):
        """
        defineModel
        -----------
        """
        HF_model = T5ForConditionalGeneration.from_pretrained(
            model_name, return_dict=True
        )

        # If the bottleneck is NOT trainable
        if type_bottleneck in ["none", "first", "mean"]:

            self.custom_model = ModelWithBottleneck(
                HF_model, type_bottleneck, pkt_dim, use_pkt_reduction, HF_model.decoder
            )
        # If the bottleneck is trainable
        else:
            if type_bottleneck == "Luong":
                bottleneck_model = Attention_Luong(HF_model.config.d_model)
            # Default trainable bottleneck is Luong attention
            else:
                bottleneck_model = Attention_Luong(HF_model.config.d_model)

            self.custom_model = ModelWithBottleneck(
                HF_model,
                type_bottleneck,
                pkt_dim,
                use_pkt_reduction,
                HF_model.decoder,
                bottleneck_model,
            )

        if model_finetuned_path != "Empty":
            pretrained_model = torch.load(f"{model_finetuned_path}/weights.pth")
            # Extract the encoder weights
            encoder_weights = {
                k: v for k, v in pretrained_model.items() if "encoder" in k
            }

            # Load the encoder weights into your custom model
            self.custom_model.load_state_dict(encoder_weights, strict=False)

    def split_dataset(self, percentage):
        """
        split_dataset
        -------------
        The method creates the train, validation and test set from a single
        dataset
        Args:
            - opts
        """
        self.qa_dataset.split_train_val_test(percentage)

    def train_model(self, logger):
        """
        train_model
        -----------
        It is used to prepare the elements necessary to perform the training
        steps.
        Args:
            - logger
        """
        train_loader = DataLoader(
            self.qa_dataset,
            batch_size=self.batch_size,
            sampler=self.qa_dataset.get_train_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        val_loader = DataLoader(
            self.qa_dataset,
            batch_size=self.batch_size,
            sampler=self.qa_dataset.get_val_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        self.lr = scale_lr(self.lr)
        self.optimizer = Adam(self.custom_model.parameters(), lr=self.lr)

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
            model=self.custom_model,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.num_epochs,
        )

    def start_training(
        self,
        logger,
        model,
        lr_scheduler,
        train_loader,
        val_loader,
        num_epochs,
    ):
        """
        start_training
        --------------
        It is used to start the training loop.

        Args:
            - logger
            - model
            - lr_scheduler
            - train_loader
            - num_epochs
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
        self.best_model = update_best_model(model=model, accelerator=logger.accelerator)

        for epoch_id in range(num_epochs):
            train_losses = self.training_batch(
                model=model,
                training_dataloader=train_loader,
                validation_dataloader=val_loader,
                epoch_id=epoch_id,
                lr_scheduler=lr_scheduler,
                logger=logger,
                progress_bar=training_progress_bar,
                checkpoint_steps=checkpoint_steps,
            )
            train_losses = torch.Tensor(train_losses)[: self.qa_dataset.size_train]
            self.report_results(
                logger=logger,
                partition="best_training",
                step=self.checkpoint_id,
                losses=train_losses,
            )
        logger.accelerator.wait_for_everyone()

    def training_batch(
        self,
        model,
        training_dataloader,
        validation_dataloader,
        epoch_id,
        lr_scheduler,
        logger,
        progress_bar,
        checkpoint_steps,
    ):
        model.train()
        batch_losses = []
        step = 0
        for _, batch in training_dataloader:
            current_step = epoch_id * len(training_dataloader) + step + 1
            if current_step % checkpoint_steps == 0:
                model.eval()
                losses = self.validation_batch(
                    model=model,
                    validation_dataloader=validation_dataloader,
                    logger=logger,
                    checkpoint_id=self.checkpoint_id,
                )
                losses = torch.Tensor(losses)
                lr_scheduler.step()
                self.report_results(
                    logger=logger,
                    partition="best_validation",
                    step=self.checkpoint_id,
                    losses=losses,
                )
                if logger.accelerator.is_main_process:
                    self.is_best_model(
                        loss=torch.mean(losses),
                        accelerator=logger.accelerator,
                        logger=logger,
                        model=model,
                        checkpoint_id=self.checkpoint_id,
                    )
                if logger.accelerator.is_main_process:
                    self.checkpoint_id += 1

            with logger.accelerator.accumulate(model):

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)

                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )

                ### LOSS COMPUTATION

                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits, batch["labels"].view(-1))
                gathered_loss = post_process(logger.accelerator.gather(loss))
                gathered_loss = reshape_loss(gathered_loss)
                ### MODEL UPDATE
                logger.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            if current_step % checkpoint_steps == 0:
                logger.save_checkpoint(
                    epoch_id=epoch_id,
                    batch_id=(step),
                    validation_steps=self.checkpoint_id,
                )
            batch_losses.append(gathered_loss)
            step += 1
        if WANDB and logger.accelerator.is_local_main_process:
            wandb.log({"loss_train": sum(batch_losses) / len(batch_losses)})
        return batch_losses

    def validation_batch(self, model, validation_dataloader, logger, checkpoint_id):
        # Evaluation
        val_losses = []
        progress_bar = tqdm(
            range(len(validation_dataloader)),
            disable=not logger.accelerator.is_local_main_process,
        )
        for step, batch in validation_dataloader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)

                logits, output_ids = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    answer_len=self.t_len,
                )

                ### LOSS COMPUTATION
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits, batch["labels"].view(-1))
                gathered_loss = post_process(logger.accelerator.gather(loss))
                gathered_loss = reshape_loss(gathered_loss)
                val_losses.append(loss.item())
                self.compute_accuracy(
                    output_ids.cpu(), batch["labels"].cpu(), step.cpu()
                )
                progress_bar.update(1)
        if WANDB:
            wandb.log({"loss_valid": sum(val_losses) / len(val_losses)})
        if self.type_pretrain == "QA":
            self.report_accuracy_QA(
                logger=logger,
                partition="accuracy",
                step=self.checkpoint_id,
                loss_valid=sum(val_losses) / len(val_losses),
            )
            self.correct = {key: 0 for key in self.correct.keys()}
            self.total = {key: 0 for key in self.total.keys()}
        elif self.type_pretrain == "denoiser":
            self.report_accuracy_denoiser(
                logger=logger, partition="accuracy", step=self.checkpoint_id
            )
            self.correct = 0
            self.total = 0

        return val_losses

    def test_model(self, logger):
        logger.accelerator.print(f"Start testing...")
        test_loader = DataLoader(
            self.qa_dataset,
            batch_size=self.batch_size,
            sampler=self.qa_dataset.get_test_sampler(),
            num_workers=1,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=5,
        )
        test_loader = logger.accelerator.prepare(test_loader)
        self.custom_model.eval()
        test_loss = []

        progress_bar = tqdm(
            range(len(test_loader)),
            disable=not logger.accelerator.is_local_main_process,
        )
        for step, batch in test_loader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)

                logits, output_ids = self.custom_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    answer_len=self.t_len,
                )
                ### LOSS COMPUTATION
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits, batch["labels"].view(-1))
                gathered_loss = post_process(logger.accelerator.gather(loss))
                gathered_loss = reshape_loss(gathered_loss)
                test_loss.append(loss)
                progress_bar.update(1)
                ### PERFORMANCE EVALUATION
                self.compute_accuracy(
                    output_ids.cpu(), batch["labels"].cpu(), step.cpu()
                )
                progress_bar.update(1)

    def compute_accuracy(self, predictions, labels, index):
        decoded_preds = self.tokenizer_obj.get_tokenizer().batch_decode(
            predictions, skip_special_tokens=True
        )
        # Replace -100 in the labels as we can't decode them
        labels = np.where(
            labels != -100, labels, self.tokenizer_obj.get_tokenizer().pad_token_id
        )
        # Decode reference summaries into text
        decoded_labels = self.tokenizer_obj.get_tokenizer().batch_decode(
            labels, skip_special_tokens=True
        )
        i = 0
        if self.type_pretrain == "QA":
            for prediction, label in zip(decoded_preds, decoded_labels):
                # index[i] because whe are considering the predictions in a batch
                elem_class = self.qa_dataset.get_class_Byindex(int(index[i]))
                self.total[elem_class] += 1
                i += 1
                if prediction == label:
                    self.correct[elem_class] += 1
                    with open(f"{self.current_experiment}.json", "a") as results_file:
                        json.dump(
                            {"prediction": prediction, "label": label}, results_file
                        )
                        results_file.write("\n")
        elif self.type_pretrain == "denoiser":
            for prediction, label in zip(decoded_preds, decoded_labels):
                self.total += 1
                if prediction == label:
                    self.correct += 1

    def generate_test_report(self, opts):
        """
        print_accuracy_Bycategory
        ------------------------
        The method computes and prints in stdout the accuracy category by
        category and the global accuracy (except for IPchk).
        """
        sum_corr = 0
        sum_total = 0
        dict_results = {}
        list_keys = [
            "bottleneck",
            "seed",
            "max_qst_length",
            "max_ans_length",
            "batch_size",
        ]
        dict_params = {key: opts[key] for key in opts.keys() if key in list_keys}
        dict_params["dataset"] = (opts["training_data"].split("/"))[-1]
        for key in self.total.keys():
            dict_results[f"Accuracy_{key}"] = (
                f"{round(self.correct[key]/self.total[key]*100, 2)} %"
            )
            sum_corr += self.correct[key]
            sum_total += self.total[key]
        dict_results[f"Average_accuracy"] = f"{round(sum_corr/sum_total*100, 2)} %"
        generate_experiment_report(opts["identifier"], dict_params, dict_results)

    def report_accuracy_QA(self, logger, partition, step, loss_valid=None):
        results = {}
        results["loss"] = float(loss_valid)
        for key in self.total.keys():
            try:
                logger.log_result(
                    key=f"Micro_accuracy/{key}",
                    value=round(self.correct[key] / self.total[key] * 100, 2),
                    step=step,
                )
                if WANDB and logger.accelerator.is_local_main_process:
                    wandb.log(
                        {
                            f"acc{key}": round(
                                self.correct[key] / self.total[key] * 100, 2
                            )
                        }
                    )
                else:
                    results[f"acc{key}"] = round(
                        self.correct[key] / self.total[key] * 100, 2
                    )

            except:
                continue
        results["Total_accuracy"] = round(
            sum(self.correct.values()) / sum(self.total.values()) * 100, 2
        )
        os.makedirs("./evaluation", exist_ok=True)
        with open(f"./evaluation/{self.current_experiment}.json", "a") as results_file:
            json.dump(results, results_file)
            results_file.write("\n")
        logger.log_result(
            key=f"Macro_accuracy/Total_average",
            value=round(sum(self.correct.values()) / sum(self.total.values()) * 100, 2),
            step=step,
        )
        if WANDB:
            wandb.log(
                {
                    f"Total_accuracy": round(
                        sum(self.correct.values()) / sum(self.total.values()) * 100, 2
                    )
                }
            )

    def report_accuracy_denoiser(self, logger, partition, step):
        logger.log_result(
            key=f"Macro_accuracy/Total_average",
            value=round((self.correct / self.total) * 100, 2),
            step=step,
        )
        if WANDB:
            wandb.log({f"Total_accuracy": round((self.correct / self.total) * 100, 2)})

    def report_results(self, logger, partition, losses, step):
        """
        Function that report the results (losses, perplexities) on training +
        validation.
        Arguments:
            logger (Logger) -- Logger to log results.
            partition (str) -- Name of the partition (e.g., eval, test, ...)
            step (int) -- Step id (e.g., epoch, validation step, etc.)
            losses (Tensor) -- Loss at a given step.
        """
        logger.log_result(
            key=f"loss_epoch/{partition}", value=torch.mean(losses), step=step
        )
        perplexity = compute_perplexity(losses)
        logger.log_result(key=f"perplexity/{partition}", value=perplexity, step=step)

    def is_best_model(self, loss, accelerator, logger, model, checkpoint_id):
        """
        is_best_model
        -------------
        Function that update the best model if the current avg. batch loss is
        smaller than the current best.

        Arguments:
            loss (float) -- Average loss inter-batch.
            accelerator (Accelerator) -- accelerator to update best model if it's
                                         better than previous
            logger (Logger) -- Logger to log the results.
            model (ModelForMLModelling) -- Current model that produced the avg_loss
            checkpoint_id (int) -- Validation checkpoint. Will act as a step for
                                   the logger.
        """
        updated = 0
        if loss <= self.current_best_loss:
            self.best_model = update_best_model(model=model, accelerator=accelerator)
            self.current_best_loss = loss
            logger.save_best_huggingface_model(model, "custom_model")
            updated = 1
        logger.log_result(key="best_checkpoint", value=updated, step=checkpoint_id)

    def accelerator_train_load(self, lr_scheduler, train_loader, val_loader, logger):
        """
        accelerator_load
        ----------------
        """
        (
            self.custom_model,
            self.optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
        ) = logger.accelerator.prepare(
            self.custom_model,
            self.optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
        )
        return lr_scheduler, train_loader, val_loader
