import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
from datetime import datetime, timedelta
import torch
import torch.distributed as dist
import numpy as np
import random

from dataclasses import field, dataclass
from datasets.distributed import split_dataset_by_node
from typing import Optional
from copy import deepcopy
from torchinfo import summary
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report, confusion_matrix
)

from NetFoundDataCollator import DataCollatorForFlowClassification
from NetFoundModels import NetfoundFinetuningModel, NetfoundNoPTM
from NetFoundTrainer import NetfoundTrainer
from NetfoundConfig import NetfoundConfig, NetFoundTCPOptionsConfig, NetFoundLarge
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, get_90_percent_cpu_count, get_logger, init_tbwriter, update_deepspeed_config, \
    LearningRateLogCallback

random.seed(42)
logger = get_logger(name=__name__)

@dataclass
class FineTuningDataTrainingArguments(CommonDataTrainingArguments):
    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(default=None)
    p_val: float = field(default=0)
    netfound_large: bool = field(default=False)
    dataset_name: str = field(default=None)

def regression_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    return {"loss": np.mean(np.absolute((logits - label_ids)))}

def classif_metrics(p: EvalPrediction, num_classes):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    pred = logits.argmax(axis=1)
    results = {
        "weighted_f1": f1_score(label_ids, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(label_ids, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(label_ids, pred),
        "weighted_prec": precision_score(label_ids, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(label_ids, pred, average="weighted", zero_division=0),
    }
    logger.warning(classification_report(label_ids, pred, digits=5))
    logger.warning(confusion_matrix(label_ids, pred))
    if num_classes > 3:
        logger.warning(f"top3: {top_k_accuracy_score(label_ids, logits, k=3)}")
    if num_classes > 5:
        logger.warning(f"top5: {top_k_accuracy_score(label_ids, logits, k=5)}")
    if num_classes > 10:
        logger.warning(f"top10: {top_k_accuracy_score(label_ids, logits, k=10)}")
    return results

def append_metrics_to_file(metrics, split_name, output_dir, filename="custom_train_results.json"):
    filepath = os.path.join(output_dir, filename)
    data = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if split_name not in data:
        data[split_name] = []
    data[split_name].append({"timestamp": timestamp, "metrics": metrics})
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Appended {split_name} metrics to {filepath}")

@record
def main():
    parser = HfArgumentParser((ModelArguments, FineTuningDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(seconds=300)
            )
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[Rank {rank}] initialized on GPU {torch.cuda.current_device()} using backend {dist.get_backend()}")
        logger.warning(f"[Rank {rank}] initialized")

    train_dataset, val_dataset, test_dataset = load_train_test_datasets(logger, data_args)

    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_dataset = split_dataset_by_node(train_dataset, rank=rank, world_size=world_size)
        val_dataset = split_dataset_by_node(val_dataset, rank=rank, world_size=world_size)
        test_dataset = split_dataset_by_node(test_dataset, rank=rank, world_size=world_size)

    config = NetFoundTCPOptionsConfig if data_args.tcpoptions else NetfoundConfig
    config = config(
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        hidden_size=model_args.hidden_size,
        no_meta=data_args.no_meta,
        flat=data_args.flat,
    )
    if data_args.netfound_large:
        nf = NetFoundLarge()
        config.hidden_size = nf.hidden_size
        config.num_hidden_layers = nf.num_hidden_layers
        config.num_attention_heads = nf.num_attention_heads

    config.pretraining = False
    config.num_labels = data_args.num_labels
    config.problem_type = data_args.problem_type

    tokenizer = NetFoundTokenizer(config=config)
    collator = DataCollatorForFlowClassification(config.max_burst_length)

    train_dataset = train_dataset.map(tokenizer, batched=True)
    val_dataset = val_dataset.map(tokenizer, batched=True)
    test_dataset = test_dataset.map(tokenizer, batched=True)

    if training_args.local_rank == 0:
        logger.warning("*** Dataset ready ***")

    if model_args.model_name_or_path and os.path.exists(model_args.model_name_or_path):
        model = freeze(NetfoundFinetuningModel.from_pretrained(model_args.model_name_or_path, config=config), model_args)
    elif model_args.no_ptm:
        model = NetfoundNoPTM(config=config)
    else:
        model = freeze(NetfoundFinetuningModel(config=config), model_args)

    if training_args.local_rank == 0:
        summary(model)

    compute_metrics = regression_metrics if data_args.problem_type == "regression" else \
                      lambda p: classif_metrics(p, data_args.num_labels)
    training_args.output_dir = os.path.join(training_args.output_dir, f"{data_args.train_dir[-1]}_{training_args.learning_rate}_{model_args.freeze_base}")
    trainer = NetfoundTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        data_collator=collator,
    )

    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.save_metrics("eval", metrics)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

