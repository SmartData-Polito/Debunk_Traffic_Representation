from copy import deepcopy
import json
import math
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import warnings
from torch.nn.functional import cross_entropy


def compute_cross_entropy_loss(logits, labels, loss_type, weights):
    """Function to compute the loss. Loss can be "default" (cross-entropy) or "weighted"
    Arguments:
        logits (Tensor) -- model's logits
        labels (Tensor) -- real labels for the corresponding predictions
    Returns:
        Tensor -- corresponding loss
    """
    if loss_type == "default":
        loss = cross_entropy(logits, labels)
    elif loss_type == "weighted":
        # weights have been precomputed earlier (training weights)
        weights = weights.to(logits.device)
        loss = cross_entropy(logits, labels, weight=weights)
    return loss


def extract_metrics(metrics):
    results = {
        "accuracy": metrics["accuracy"]["precision"],
        "precision": metrics["macro avg"]["precision"],
        "recall": metrics["macro avg"]["recall"],
        "f1-score": metrics["macro avg"]["f1-score"],
    }
    return results


def update_best_model(model, accelerator):
    """Function that initialize the best model at the beginning of the training
    Arguments:
        model (Model) -- model we want to set as current best
        accelerator (Accelerator) -- accelerator to unwrap the model (best model must be on a single GPU)
    Returns:
        (Model) -- best model, unwrapped on a single GPU.
    """
    return deepcopy(accelerator.unwrap_model(model))


def compute_perplexity(losses):
    """Function that compute the perplexity given the losses
    Arguments:
        losses (Tensor) -- losses per batches
    Returns:
        float -- perplexity score
    """
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def determine_n_checkpoints(
    train_dataloader, checkpoints_x_epoch, checkpoints_patience=500
):
    """Function that chooses how many checkpoint steps we'll have.
    Args:
        train_dataloader (Dataloader): Dataloader. Determines the number of training steps per epoch (|data|//batch_size)
        checkpoints_x_epoch (int): Number of checkpoints per epoch we will perform if patience is not met.
        checkpoints_patience (int, optional): Maximum number of training steps before creating intermidiate checkpoints.

    Returns:
        int: Number of training steps after training/validation checkpoints
    """
    train_steps_x_epoc = len(train_dataloader)
    if (
        train_steps_x_epoc > checkpoints_patience
    ):  # if number of steps > threshold (patience), then obtain checkpoint more frequently.
        # e.g. 100_000 training steps and 10 checkpoints_x_epoch > 10_000 steps before performing validation
        return round(train_steps_x_epoc / checkpoints_x_epoch)
    else:  # 1 checkpoint per epoch
        return len(train_dataloader)


def restore_interrupted_training(
    accelerator,
    train_dataloader,
    total_epochs,
    completed_epochs=0,
    completed_steps_current_epoch=0,
):
    """Function that restarts a training that might have stopped. Idea is to:
    1) Since the dataloader is a stream-based object, bring the stream to the exact point in which the training was interrupted
    2) Find the number of total remaining training steps from this point on
    Args:
        accelerator (Accelerator): It advances the Dataloader stream to the desired point, based on the number of `completed_steps_current_epoch`
        train_dataloader (Dataloader): Training Dataloader we want to advance
        total_epochs (int): Number of epochs specified for the training. Does not take into account of completed epochs yet.
        completed_epochs (int, optional): Number of already completed epoch. Can be 0 if training starts from scratch.
        completed_steps_current_epoch (int, optional): Number of completed steps within an epoch.
                                                        Basically, it means that the training was interrupted at some point in between an epoch.
    Returns:
        (Dataloader, int): Updated dataloader for the first round (then we will use the original train dataloader again) and number of remaining training steps before training ends.
    """
    # 1a. Find remaining training steps (number of elements in the training * (missing epochs+1))
    remaining_training_steps = len(train_dataloader) * (
        total_epochs - completed_epochs - 1
    )
    # 1b. If there were training steps we've already completed during the current epoch, skip those batches
    skipped_dataloader = accelerator.skip_first_batches(
        train_dataloader, completed_steps_current_epoch
    )
    # 1c. Find steps remaining in the current epoch
    missing_steps_current_epoch = len(skipped_dataloader)
    # 1d. Eventually, sum them to the number of remaining training steps
    remaining_training_steps += missing_steps_current_epoch
    return skipped_dataloader, remaining_training_steps


def create_scheduler(optimizer, train_dataloader, epochs, num_warmup_steps=0):
    """Function that creates a linear scheduler (most common in Huggingface scripts).

    Arguments:
        optimizer (torch.optim) -- Torch optimizer
        train_dataloader (Dataloader) -- training dataloader
        epochs (int) -- number of training epochs
        num_warmup_steps (int, optional) -- How many steps to wait before activating the scheduler. Defaults to 0.
    """
    type_scheduler = "linear"
    #num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = epochs 
    #num_training_steps = epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        type_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler

def create_dataloaders(
    dataset, data_collator, batch_size, n_proc, shuffle=False, sampler=None
):
    """Create a dataloader for the given partition.

    Arguments:
        dataset (Dataset): partition (e.g., train, eval or test) of the original dataset. It contains the tokenized inputs
        data_collator (DataCollator): data collator object. Can be default_data_collator or DataCollatorForLanguageModeling
        batch_size (int): number of elements per batch in the dataloader instances
        n_proc (int): number of processes used to prepare a batch when the queue is pulled.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        sampler (type, optional): customized sampling, useful for the SupCon model. Default to None (default random sampler).
    Returns:
        Dataloader: dataloader with the desired settings
    """
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=n_proc,
        batch_sampler=sampler,
        pin_memory=True,  # last two are optimmizing parameters
        prefetch_factor=5,
    )
    return dataloader


# def load_tokenizer(tokenizer_name, tokenizer_status, n_tokens, cache_folder, proxies):
#     if tokenizer_status == "standard":
#         cache_folder = (
#             os.path.join(cache_folder, "pretrained") if cache_folder else None
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             tokenizer_name,
#             return_tensors="pt",
#             model_max_length=n_tokens,
#             proxies=proxies,
#             cache_dir=cache_folder,
#         )
#     else:  # can be that we finetuned our own tokenizer (#NEVER TESTED SO FAR)
#         tokenizer_path = os.path.join(cache_folder, "finetuned", f"{tokenizer_status}")
#         tokenizer = AutoTokenizer.from_pretrained(
#             tokenizer_path,
#             tokenizer_file=os.path.join(tokenizer_path, f"config.json"),
#             return_tensors="pt",
#             model_max_length=n_tokens,
#         )
#     return tokenizer


# def _tokenize_dataset(
#     ds,
#     tokenizer,
#     tokenizer_processing,
#     test_name="test",
#     train_name="train",
#     num_proc=30,
#     load_from_cache_file=True,
#     cache_file_names=None,
# ):
#     # 1. Define columns we want to remove in the tokenized dataset
#     #   N.b. Necessary: otherwise later DataCollator complains about non numerical elements
#     #   Get them from test set if present, otherwise from training
#     key_remove_col = test_name if test_name in ds.keys() else train_name
#     remove_columns = ds[key_remove_col].column_names
#     num_proc = is_tokenizer_fast(n_proc=num_proc, tokenizer=tokenizer)
#     tokenized_ds = ds.map(
#         tokenizer_processing,
#         fn_kwargs={"tokenizer": tokenizer},
#         remove_columns=remove_columns,
#         num_proc=num_proc,
#         batched=True,
#         load_from_cache_file=load_from_cache_file,
#         cache_file_names=cache_file_names,
#     )
#     return tokenized_ds


def extract_cti(cti_path, ips):
    """Function to extract CTI info for each IP.
    Arguments:
        cti_path (str) -- Path to CTI info. It leads to a `parquet` object.
        ips (list) -- List of IPs. Can be source or destination.

    Returns:
        list -- CTI info for each IP.
    """
    df_cti = pd.read_parquet(cti_path).set_index("ip")
    cti_info = []
    for ip in ips:
        if ip in df_cti.index:
            cti_info.append(df_cti.loc[ip]["result"])
        else:
            cti_info.append("No Info")
    return cti_info

def compute_perplexity(losses):
    """Function that compute the perplexity given the losses
    Arguments:
        losses (Tensor) -- losses per batches
    Returns:
        float -- perplexity score
    """
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def retrieve_additional_features(sample, current_input, additional_features, cti_path):
    """Function retrieving additional features (e.g., cti info) and append them to the current input to tokenize.

    Args:
        sample (dict): current processed row of the original dataset
        current_input (str): current (main) input feature selected for now (e.g., usually "payload")
        additional_features (list): list of additional features we want to add
    Returns:
        str: concatenation of inputs
    """
    # 1. Elements will be appended to the current input. So, elements at the beginning of this list will be the first appended and "slide back" while more and more elments get appended
    #   Since this is not so intuitive, revert the initial list so that, when elements are appended, original order is respected
    additional_features.reverse()
    for feature in additional_features:
        if feature == "cti":
            src_ips, dst_ips = sample["srcIP"], sample["dstIP"]
            src_ctis = extract_cti(cti_path=cti_path, ips=src_ips)
            dst_ctis = extract_cti(cti_path=cti_path, ips=dst_ips)
            el_2_append = [
                f"source IP diagnosis: {src_cti}\ndestination IP diagnosis: {dst_cti}"
                for src_cti, dst_cti in zip(src_ctis, dst_ctis)
            ]
        else:
            # e.g., "srcIP: 10.09.82.213"
            el_2_append = f"{feature}: {sample[feature]}"
        current_input = [
            f"{add_feature}\n{prev_feature}"
            for add_feature, prev_feature in zip(el_2_append, current_input)
        ]
    return current_input


def is_tokenizer_fast(n_proc, tokenizer):
    """Function that returns how many cpu to use for the tokenization. If the chosen tokenizer is a FastTokenizer, use 1.
    Args:
        n_proc (int): candidate number of processes if tokenizer is not fast
        tokenizer (AutoTokenizer): tokenizer. Output of the function changes if the tokenizer `is_fast`.
    Returns:
        int: number of cpu to use
    """
    cpus2use = 1 if tokenizer.is_fast else n_proc
    return cpus2use


def insert_random_mask(batch, data_collator):
    """Function that insert a random mask to each feature of the input batch.
    Useful for masked language modelling to process the validation dataset (as suggested in https://huggingface.co/learn/nlp-course/chapter7/3?fw=tf)
    Arguments:
        batch (Dictionary) -- Batch of elements all with the same features
        data_collator (DataCollator) -- Datacollator object defined in the self supervised model.
    Returns:
        Dictionary -- Create a new "masked" column for each column in the dataset
    """
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {k: v.numpy() for k, v in masked_inputs.items()}


def chunk_txt(examples, chunk_size):
    """Function to chunk text into chunks. Copied from https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt.
    Used in self-supervision in order to NOT truncate text.
    Arguments:
        examples (Dictionary) -- Original dictionary of elements
        chunk_size (int) -- Number of tokens per chunks. Must be < max_lenght for the given model.
    Returns:
        Dictionary -- New dictionary of chunks
    """
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()
    }  # Concatenate all texts
    total_length = len(
        concatenated_examples[list(examples.keys())[0]]
    )  # Compute length of concatenated texts
    total_length = (
        total_length // chunk_size
    ) * chunk_size  # We drop the last chunk if it's smaller than chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }  # Split by chunks of max_len
    result["labels"] = result["input_ids"].copy()  # Create a new labels column
    return result


def reshape_loss(loss):
    """Function that reshape the loss if trained on 1 gpu/on cpu. We need a Tensor, while loss could be possibly a Scalar.
    Arguments:
        loss (_type_) -- Loss, could potentially be a scalar
    Returns:
        Tensor -- Loss transformed to a tensor.
    """
    if not torch.is_tensor(loss):
        return loss.reshape(1)
    else:
        return loss


def post_process(tensor):
    """
    Post-process the tensor. Gather the tensor if it was distributed moves to cpu and clone it.
    Function is used when saving the predictions, evaluating the metrics, etc.
    Arguments:
        tensor (torch.Tensor) -- The tensor.
    Returns:
        torch.Tensor -- The post-processed tensor.
    """
    tensor = tensor.detach().cpu().clone()
    return tensor


def map2id(example, labels2id, column_label):
    """Map label to corresponding id. Function used in `map` operations.
    Arguments:
        example (Dataset Row) -- Row from the Dataset we're applying the map function with
        labels2id (Dictionary) -- Dictionary labels2id
        column_label (str) -- Name of the label column
    Returns:
        Dataset Row -- Modified dataset row
    """
    example[column_label] = labels2id[example[column_label]]
    return example


def sanity_check_model_choice(config_path, training_stage, task):
    """Function that performs a sanity check on the developer's choices. Some combinations (e.g. inference with a non-trained model) should not be allowed.
    Arguments:
        config_path (str) -- Path to the configuration file, which must exist.
        training_stage (str) -- Training stage, can be "training" or "inference". If "inference", we have to check more details.
        task (str) -- Task to be solved. Triggers some more task-specific checks.
    """
    assert os.path.isfile(
        config_path
    ), "Error: configuration file for the trained model does not exist (are you sure it's not a pre-trained model?)!"
    with open(config_path) as f:
        config = json.load(f)
    architecture = config["architectures"][0]
    if training_stage == "inference":
        if task == "classification":
            assert (
                "ForSequenceClassification" in architecture
            ), "Error: you are performing inference, the trained model must have a classification head already!"
        elif task == "contrastive_learning":
            assert (
                "ForSequenceClassification" not in architecture
                and "ForMaskedLM" not in architecture
            ), "Error: you are performing inference, the trained model must be a simple encoder (AutoModel class)!"


def convert_2_int(id2labels, label2id):
    """Convert the keys and values of the given dictionaries to integers.
    Useful when the dictionaries are loaded and the integers might have been converted to string.
    Arguments:
        id2labels (Dict[str, Any]): A dictionary mapping IDs to labels.
        label2id (Dict[str, Any]): A dictionary mapping labels to IDs.
    Returns:
        Tuple[Dict[int, Any], Dict[Any, int]]: A tuple containing the converted dictionaries.
    """
    id2labels = {int(key): val for key, val in id2labels.items()}
    label2id = {key: int(val) for key, val in label2id.items()}
    return id2labels, label2id


def _load_labels_mapping(labels, labels_mapping_path):
    """
    Creates the label mapping ids2labels and labels2ids for a given set of labels.
    Arguments:
        labels (list) -- Set of labels from the training set.
    Returns:
        Tuple[Dict[int, str], Dict[str, int]] -- A tuple containing two dictionaries.
        The first dictionary maps label IDs to label names, and the second dictionary maps label names to label IDs.
    """
    # 1) produce labels
    ids = [it for it in range(len(labels))]
    label2id = dict(zip(labels, ids))
    id2label = dict(zip(ids, labels))
    # 2) save them for inference
    with open(os.path.join(labels_mapping_path, "label_mapping.json"), "w+") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
    return id2label, label2id


def recover_labels_mapping(labels_mapping_path):
    """Function that recovers id2label and label2id from the training configuration file
    For compatibility reasons, since not all trainings were originally saving it, we also put a default mapping.
    Arguments:
        labels_mapping_path (str) -- Path to the training configuration file.
    Returns:
        Tuple(Dictionary, Dictionary) -- Dictionaries id2label, label2id
    """
    training_config = os.path.join(labels_mapping_path, "label_mapping.json")
    with open(training_config, encoding="utf-8") as f:
        config_file = json.load(f)
    if "id2label" in config_file.keys():
        return config_file["id2label"], config_file["label2id"]
    else:
        warnings.warn("Mapping files were NOT FOUND. Loading default mappings...")
        labels = ["TOP1", "TOP2", "TOP3-4", "TOP5", "TOP6"]
        return {key: el for key, el in enumerate(labels)}, {
            el: str(key) for key, el in enumerate(labels)
        }
