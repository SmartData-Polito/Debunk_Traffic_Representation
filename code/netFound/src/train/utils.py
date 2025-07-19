from dataclasses import dataclass, field
from typing import Optional
import datasets
import transformers
import logging
import os
import json
import threading
import subprocess
import torch
import time
import socket
import psutil

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

LOGGING_LEVEL = logging.WARNING
TB_WRITER: Optional[SummaryWriter] = None


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    metaFeatures: int = field(
        default=4,
        metadata={"help": "number of metadata fields."},
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "Number of hidden layers."},
    )
    num_attention_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads."},
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size."},
    )
    no_ptm: bool = field(
        default=False,
        metadata={"help": "If True, use NoPTM model (only for fine-tuning)."},
    )
    freeze_flow_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze flow encoders"},
    )
    freeze_burst_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze burst encoders"},
    )
    freeze_embeddings: bool = field(
        default=False,
        metadata={"help": "Freeze embeddings"},
    )
    freeze_base: bool = field(
        default=False,
        metadata={"help": "Freeze base model"},
    )


@dataclass
class CommonDataTrainingArguments:
    train_dir: Optional[str] = field(
        metadata={"help": "Directory with training data (Apache Arrow files)"})
    test_dir: Optional[str] = field(default=None, metadata={
        "help": "Directory with testing data (Apache Arrow files)"})
    no_meta: bool = field(
        default=False,
        metadata={"help": "no meta fields"},
    )
    flat: bool = field(
        default=False,
        metadata={"help": "no cross burst encoder"},
    )
    limit_bursts: bool = field(
        default=False,
        metadata={"help": "limit_bursts"},
    )
    validation_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory with optional input evaluation data to evaluate the perplexity on (Apache Arrow files)"},
    )
    validation_split_percentage: Optional[int] = field(
        default=30,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )
    data_cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to store the dataset cache."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_bursts: int = field(
        default=12,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_seq_length: Optional[int] = field(
        default=1296 + 12,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to load dataset in the streaming mode."},
    )
    tcpoptions: bool = field(
        default=False,
        metadata={"help": "Whether the data contains TCP options."},
    )


def freeze(model, model_args):
    for name, param in model.base_transformer.named_parameters():
        if model_args.freeze_flow_encoder and (
                "flow_encoder" in name or ("encoder" in name and "position_embeddings" in name)):
            param.requires_grad = False
        if model_args.freeze_burst_encoder and "burst_encoder" in name:
            param.requires_grad = False
        if model_args.freeze_embeddings and (name.startswith("embed") or name.startswith("seg_embed")):
            param.requires_grad = False
        if model_args.freeze_base:
            param.requires_grad = False
    return model


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(LOGGING_LEVEL)
    datasets.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def verify_checkpoint(logger, training_args):
    if not training_args.resume_from_checkpoint:
        folders = set(os.listdir(training_args.output_dir)) - {"runs"}
        if len(folders) > 0:
            if training_args.local_rank == 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overwrite it."
                )
    else:
        if training_args.local_rank == 0:
            resume_from_checkpoint = training_args.resume_from_checkpoint if isinstance(training_args.resume_from_checkpoint, str) else get_last_checkpoint(training_args.output_dir)
            logger.warning(
                f"Checkpoint detected, resuming training at {resume_from_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def get_90_percent_cpu_count():
    return max(1, int(os.cpu_count() * 0.9))


def load_train_test_datasets(logger, data_args):
    logger.warning("Loading datasets")
    if data_args.test_dir is None:
        data_args.test_dir = data_args.train_dir
        train_split = f"train[{data_args.validation_split_percentage}%:]"
        val_split = f"train[:{data_args.validation_split_percentage}%]"
        test_split = f"train[:{data_args.validation_split_percentage}%]"
    else:
        train_split = "train"
        val_split = "train"
        test_split = "train"

    train_dataset = load_dataset(
        "arrow",
        data_dir= f"{data_args.train_dir}/train",
        split=train_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    val_dataset = load_dataset(
        "arrow",
        data_dir=f"{data_args.train_dir}/val",
        split=val_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    test_dataset = load_dataset(
        "arrow",
        data_dir=data_args.test_dir,
        split=test_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    if data_args.max_eval_samples is not None:
        test_dataset = test_dataset.select(
            range(min(test_dataset.shape[0], data_args.max_eval_samples))
        )
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(train_dataset.shape[0], data_args.max_train_samples))
        )

    if not data_args.streaming:
        total_bursts_train = [0] * len(train_dataset)
        total_bursts_val = [0] * len(val_dataset)
        total_bursts_test = [0] * len(test_dataset)
    else:
        total_bursts_train = defaultdict(lambda: 0)
        total_bursts_val = defaultdict(lambda: 0)
        total_bursts_test = defaultdict(lambda: 0)

    train_dataset = train_dataset.add_column("total_bursts", total_bursts_train)
    val_dataset = val_dataset.add_column("total_bursts", total_bursts_val)
    test_dataset = test_dataset.add_column("total_bursts", total_bursts_test)


    return train_dataset, val_dataset, test_dataset


def initialize_model_with_deepspeed(logger, training_args, get_model):
    '''
    here we do only specific init if stage 3 is used, otherwise huggingface trainer will do the rest
    '''
    import deepspeed
    import base64
    logger.warning("Initializing deepspeed-optimized model")
    # only if stage 3
    if training_args.deepspeed.endswith(".json"):
        with open(training_args.deepspeed, "r") as f:
            deepspeed_config = json.load(f)
    else:
        deepspeed_config = training_args.deepspeed
        # unbase64
        deepspeed_config = json.loads(base64.b64decode(deepspeed_config).decode("utf-8"))

    is_stage_3 = deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3
    with deepspeed.zero.Init(enabled=is_stage_3):
        model = get_model()
    optimizers = (None, None)
    return model, optimizers


def init_tbwriter(output_dir=".") -> None:
    global TB_WRITER
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    if not torch.cuda.is_available():
        TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_pid{os.getpid()}_custom_metrics"))
        return
    TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_gpu{torch.cuda.current_device()}_custom_metrics"))

def get_gpu_utilization(gpu_id):
    """Fetch GPU utilization using nvidia-smi for the given GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        utilization = int(result.stdout.strip())
        return utilization
    except Exception as e:
        get_logger(__name__).error(f"Error fetching GPU utilization: {e}")
        return 0


def log_gpu_stats(gpu_id, output_dir, interval=10):
    """
    Log GPU utilization and memory usage for the assigned GPU to TensorBoard every `interval` seconds.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return
    
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_gpu{gpu_id}"))

    while True:
        # Get GPU stats for the current process's assigned GPU
        device = torch.device(f"cuda:{gpu_id}")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # In GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # In MB
        memory_free = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) - memory_reserved

        # Get GPU utilization using nvidia-smi
        utilization = get_gpu_utilization(gpu_id)

        # Log to TensorBoard
        writer.add_scalar(f"GPU/Memory Allocated (GB)", memory_allocated, time.time())
        writer.add_scalar(f"GPU/Memory Reserved (GB)", memory_reserved, time.time())
        writer.add_scalar(f"GPU/Memory Free (GB)", memory_free, time.time())
        writer.add_scalar(f"GPU/Utilization (%)", utilization, time.time())

        # Sleep before logging the next set of stats
        time.sleep(interval)

def start_gpu_logging(output_dir="."):
    """
    Start logging GPU stats to TensorBoard for the current process's assigned GPU.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return

    gpu_id = torch.cuda.current_device()

    # Start logging GPU stats in a separate thread
    gpu_stats_thread = threading.Thread(target=log_gpu_stats, args=(gpu_id, output_dir))
    gpu_stats_thread.daemon = True
    gpu_stats_thread.start()

def log_cpu_stats(output_dir, interval=10):
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_cpu_metrics"))

    while True:
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            writer.add_scalar(f"CPU/Utilization %", psutil.cpu_percent(interval=None), time.time())
        except Exception as e:
            get_logger(__name__).error(f"Error fetching CPU utilization: {e}")
            return 0
        
        time.sleep(interval)

def start_cpu_logging(output_dir="."):
    """
    Start logging overall CPU stats to TensorBoard.
    """
    # do it only for a single process per node
    if os.environ.get("SLURM_LOCALID", "-1") != "0":
        return

    cpu_stats_thread = threading.Thread(target=log_cpu_stats, args=(output_dir,))
    cpu_stats_thread.daemon = True
    cpu_stats_thread.start()

def update_deepspeed_config(training_args):
    if training_args.deepspeed is not None and training_args.deepspeed.endswith(".json"):
        with open(training_args.deepspeed, "r") as f:
            training_args.deepspeed = json.load(f)
        if "tensorboard" in training_args.deepspeed:
            training_args.deepspeed["tensorboard"]["output_path"] = training_args.output_dir
            training_args.deepspeed["tensorboard"]["job_name"] = os.environ.get("SLURM_JOB_NAME", "local")
    return training_args

class LearningRateLogCallback(TrainerCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def on_step_end(self, args, state, control, **kwargs):
        # The optimizer is passed as a keyword argument
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            # If you have multiple parameter groups, you can log each group’s LR
            for i, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar(f"train/learning_rate/group_{i}", param_group['lr'], state.global_step)
        return control