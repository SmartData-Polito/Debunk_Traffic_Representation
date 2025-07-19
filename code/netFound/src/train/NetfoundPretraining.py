import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import math
import random
import os
import torch
import torch.distributed
import utils
from dataclasses import field, dataclass
from typing import Optional
from torchinfo import summary

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.distributed.elastic.multiprocessing.errors import record
from datasets.distributed import split_dataset_by_node
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)

from NetFoundModels import NetFoundLanguageModelling
from NetFoundTrainer import NetfoundTrainer
from NetFoundDataCollator import DataCollatorWithMeta
from NetfoundConfig import NetfoundConfig, NetFoundTCPOptionsConfig
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, get_90_percent_cpu_count, initialize_model_with_deepspeed, get_logger, init_tbwriter, update_deepspeed_config, \
    LearningRateLogCallback


random.seed(42)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class PretrainingDataTrainingArguments(CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    no_mlm: bool = field(
        default=False,
        metadata={"help": "no MLM loss function"},
    )
    no_swapped_bursts: bool = field(
        default=False,
        metadata={"help": "no swapped bursts loss function"},
    )
    no_metadata_loss: bool = field(
        default=False,
        metadata={"help": "no metadata loss function"},
    )
    no_direction_loss: bool = field(
        default=False,
        metadata={"help": "no direction loss function"},
    )
    swap_rate: Optional[float] = field(
        default=0.5,
        metadata={"help": "probability of swapping the burst in the flow during training"},
    )
    subflow_len: Optional[int] = field(
        default=-1,
        metadata={"help": "subflow length, -1 for no subflow"},
    )
    mlm_probability: float = field(
        default=0.30,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )


def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple):
        return tuple(i.argmax(dim=-1) for i in logits)
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    all_preds, all_labels = eval_preds

    labels = all_labels[0] if isinstance(all_labels, tuple) else all_labels
    preds = all_preds[0] if isinstance(all_preds, tuple) else all_preds
    swappedBurstGTs = all_labels[1] if isinstance(all_labels, tuple) else None
    swappedBurstPreds = all_preds[1] if isinstance(all_preds, tuple) else None

    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return_metrics = {
        "macro_mlm_f1": f1_score(labels, preds, average="macro"),
        "macro_mlm_prec": precision_score(labels, preds, average="macro"),
        "macro_mlm_recall": recall_score(labels, preds, average="macro"),
        "weighted_mlm_f1": f1_score(labels, preds, average="weighted"),
        "weighted_mlm_prec": precision_score(labels, preds, average="weighted"),
        "weighted_mlm_recall": recall_score(labels, preds, average="weighted"),
        "mlm_acc": accuracy_score(labels, preds),
    }
    if swappedBurstGTs is not None and swappedBurstPreds is not None:
        return_metrics.update(
            {
                "swapped_macro_pred_f1": f1_score(swappedBurstGTs, swappedBurstPreds, average="macro"),
                "swapped_macro_pred_prec": precision_score(
                    swappedBurstGTs, swappedBurstPreds, average="macro"
                ),
                "swapped_macro_pred_recall": recall_score(
                    swappedBurstGTs, swappedBurstPreds, average="macro"
                ),
                "swapped_weighted_pred_f1": f1_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_weighted_pred_prec": precision_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_weighted_pred_recall": recall_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_pred_acc": accuracy_score(swappedBurstGTs, swappedBurstPreds),
            }
        )
    return return_metrics

@record
def main():
    parser = HfArgumentParser(
        (ModelArguments, PretrainingDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = get_logger(name=__name__)

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    train_dataset, test_dataset = load_train_test_datasets(logger, data_args)
    if "WORLD_SIZE" in os.environ:
        train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

    logger.warning("Tokenizing datasets")
    config = NetFoundTCPOptionsConfig if data_args.tcpoptions else NetfoundConfig
    config = config(
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            hidden_size=model_args.hidden_size,
            no_meta=data_args.no_meta,
            flat=data_args.flat,
        )

    config.roformer = False
    config.limit_bursts = data_args.limit_bursts
    config.no_mlm = data_args.no_mlm
    if config.no_mlm:
        data_args.mlm_probability = 0.00001  # epsilon
    swap_rate = data_args.swap_rate
    config.no_swapped_bursts = data_args.no_swapped_bursts
    config.no_metadata_loss = data_args.no_metadata_loss
    config.no_direction_loss = data_args.no_direction_loss
    if config.no_swapped_bursts:
        swap_rate = 0
    config.name_or_path = model_args.model_name_or_path
    tokenizer = NetFoundTokenizer(config=config)

    data_collator = DataCollatorWithMeta(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        swap_rate=swap_rate
    )

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0 and not data_args.streaming:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "function": tokenizer,
        "batched": True
    }
    if not data_args.streaming:
        params['num_proc'] = data_args.preprocessing_num_workers or get_90_percent_cpu_count()
    train_dataset = train_dataset.map(**params)
    test_dataset = test_dataset.map(**params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0 and not data_args.streaming:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        model = freeze(NetFoundLanguageModelling.from_pretrained(model_args.model_name_or_path, config=config), model_args)
    else:
        model = freeze(NetFoundLanguageModelling(config=config), model_args)
    if training_args.local_rank == 0:
        summary(model)

    trainer = NetfoundTrainer(
        label_names=["swappedLabels"],
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )
    init_tbwriter(training_args.output_dir)
    trainer.add_callback(LearningRateLogCallback(utils.TB_WRITER))
    utils.start_gpu_logging(training_args.output_dir)
    utils.start_cpu_logging(training_args.output_dir)

    verify_checkpoint(logger, training_args)
    
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
