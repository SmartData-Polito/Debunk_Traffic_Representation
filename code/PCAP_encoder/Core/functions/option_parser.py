import time
import argparse
from torch.cuda import is_available
from distutils.util import strtobool


def get_training_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters to train a Language Model for entity classification"
    )

    # Experiment details
    parser.add_argument(
        "--identifier", type=str, required=True, help="Special identifier."
    )

    parser.add_argument(
        "--experiment", type=str, default="training", help="Special identifier."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["supervised", "self_supervision", "inference"],
        help="Task solved by the experiment.",
    )

    parser.add_argument(
        "--type_pretrain",
        type=str,
        required=True,
        choices=["QA", "denoiser"],
        help="pretrain performed in the experiment."
    )

    parser.add_argument(
        "--bottleneck",
        type=str,
        choices=["none", "first", "mean", "Luong"],
        help="Compression method in the model.",
    )

    parser.add_argument(
        "--pkt_repr_dim",
        type=int,
        default=768,
        help="Dimension of the packet representation.",
    )

    parser.add_argument(
        "--use_pkt_reduction",
        action=argparse.BooleanOptionalAction,
        help="The user can decide if to introduce additional internal linear layers to reducce the dimension of the packet representation from 768 to 'pkt_repr_dim'."
    )

    parser.add_argument(
        "--input_format",
        type=str,
        choices=["every4", "every2", "noSpace"],
        default="every4",
        help="Input format of the context in datasets.",
    )

    parser.add_argument(
        "--denoiser_CR",
        type=int,
        default=0,
        help="If the training is on a denoiser, this parameter fixes the corruption rate of it.",
    )

    parser.add_argument(
        "--gpu", type=str, required=True, help="GPU(s) used to run the experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed. Influences the partitions divisions and the model's initialization.",
        default=1,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether to enforce the script not to use GPU even if available.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path where to save the run's output.",
    )
    
    parser.add_argument(
        "--use_date",
        action="store_true",
        help="Flag indicating whether to include the date info in the experiment name.",
    )
    parser.add_argument(
        "--clean_start",
        action="store_true",
        help="Flag indicating whether, given the same experiment ID, we want to clean previous logs, results and cache before starting.",
    )

    # Data
    parser.add_argument(
        "--training_data",
        required=True,
        type=str,
        help="Path to the input data. Supposed to be a parquet or csv file. See more in the README.",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        default="",
        help="Path to the validation data. Alternative to obtaining the validation set from the training.",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        default="",
        help="Path to the test data.",
    )
    parser.add_argument(
        "--eval_size", type=float, default=0.2, help="Size of the evaluation partition."
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Chosen model.")
    parser.add_argument(
        "--finetuned_path_model",
        type=str,
        default="Empty",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_model' (run a simple training for an example).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Chosen tokenizer. \
                        Can be a simple name (e.g., bert-uncased) or the path to the finetuned tokenizer.",
    )
    
    parser.add_argument(
        "--max_qst_length",
        type=int,
        default=256,
        help="Maximum number of tokens before truncation (in question).",
    )
    parser.add_argument(
        "--max_ans_length",
        type=int,
        default=32,
        help="Maximum number of tokens before truncation (in answer).",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=100,
        help="Number of rows taken from the dataset.",
    )

    # Hyper-parameters
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs before stopping training.",
    )
    parser.add_argument(
        "--lr",
        default=5e-6,
        type=float,
        help="Number of epochs before stopping training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="How many epochs to wait before reducing on plateau. Default to 4.",
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=32, help="How many samples per batch."
    )

    opts = parser.parse_args(args)
    # We use GPU if any available
    opts.use_cuda = is_available() and not opts.no_cuda
    # Run identifier
    opts.run_name = (
        f'{opts.identifier}_{time.strftime("%Y%m%dT%H%M%S")}'
        if opts.use_date
        else opts.identifier
    )
    # Set training stage
    opts.training_stage = "training"
    return vars(opts)


def get_inference_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters to perform inference with a Language Model for entity classification"
    )

    # Experiment details
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["supervised", "self_supervision", "inference"],
        help="Task solved by the experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed. Influences the partitions divisions and the model's initialization.",
        default=1,
    )

    parser.add_argument(
        "--gpu", type=str, required=True, help="GPU(s) used to run the experiment"
    )

    parser.add_argument(
        "--bottleneck",
        type=str,
        choices=["none", "first", "mean", "Luong"],
        help="Compression method in the model.",
    )

    parser.add_argument(
        "--input_format",
        type=str,
        choices=["every4", "every2", "noSpace"],
        default="every4",
        help="Input format of the context in datasets.",
    )

    parser.add_argument(
        "--pkt_repr_dim",
        type=int,
        default=768,
        help="Dimension of the packet representation.",
    )

    parser.add_argument(
        "--use_pkt_reduction",
        action=argparse.BooleanOptionalAction,
        help="The user can decide if to introduce additional internal linear layers to reducce the dimension of the packet representation from 768 to 'pkt_repr_dim'."
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )

    parser.add_argument(
        "--flow_level",
        type=str,
        default="none",
        choices=["none", "majority_vote", "representation_concat"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether to enforce the script not to use GPU even if available.",
    )

    parser.add_argument(
        "--clean_start",
        action="store_true",
        help="Flag indicating whether, given the same experiment ID, we want to clean previous logs, results and cache before starting.",
    )

    parser.add_argument(
        "--identifier", type=str, default="classification", help="Special identifier."
    )

    parser.add_argument(
        "--experiment", type=str, default="classification", help="Special identifier."
    )

    # Data
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path where to save the run's output. Used only if using a finetuned model from Huggingface for inference.",
    )

    parser.add_argument(
        "--testing_data",
        type=str,
        default="",
        help="Path to the test data.",
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="",
        help="Path to the train data.",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        default="",
        help="Path to the validation data.",
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Chosen model.")
    parser.add_argument(
        "--finetuned_path_model",
        type=str,
        default="Empty",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_model' (run a simple training for an example).",
    )

    parser.add_argument(
        "--finetuned_path_bottleneck",
        type=str,
        default="",
        help="Path to the domain-adapted finetuned bottleneck model. Remember that the path must point to a folder that contains a subfolder 'best_bottleneck_model' (run a simple training for an example).",
    )

    parser.add_argument(
        "--finetuned_path_classification",
        type=str,
        default="",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_bottleneck_model' (run a simple training for an example).",
    )
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Chosen tokenizer. \
                        Can be a simple name (e.g., bert-uncased) or the path to the finetuned tokenizer.",
    )

    # Hyper-parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="How many samples per batch."
    )

    parser.add_argument(
        "--percentage",
        type=int,
        default=50,
        help="Number of rows taken from the dataset.",
    )
    
    parser.add_argument(
        "--max_qst_length",
        type=int,
        default=256,
        help="Maximum number of tokens before truncation (in question).",
    )
    parser.add_argument(
        "--max_ans_length",
        type=int,
        default=32,
        help="Maximum number of tokens before truncation (in answer).",
    )
    parser.add_argument(
        "--max_chunk_length",
        type=int,
        default=512,
        help="Maximum number of tokens before truncation.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs before stopping training.",
    )
    parser.add_argument(
        "--lr",
        default=5e-6,
        type=float,
        help="Learning rate used during training",
    )
    parser.add_argument(
        "--pkts_in_flow",
        default=5,
        type=float,
        help="Number packets for each flow.",
    )

    opts = parser.parse_args(args)
    # We use GPU if any available
    opts.use_cuda = is_available() and not opts.no_cuda
    # Set training stage
    opts.training_stage = "inference"
    opts.run_name = opts.identifier
    # Option only for training > set as placeholders
    opts.available_percentage = 1
    opts.eval_size = 0
    return vars(opts)


def get_classification_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters to perform inference with a Language Model for entity classification"
    )

    # Experiment details
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["supervised", "self_supervision", "inference"],
        help="Task solved by the experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed. Influences the partitions divisions and the model's initialization.",
        default=1,
    )

    parser.add_argument(
        "--gpu", type=str, required=True, help="GPU(s) used to run the experiment"
    )

    parser.add_argument(
        "--bottleneck",
        type=str,
        choices=["none", "first", "mean", "Luong"],
        help="Compression method in the model.",
    )

    parser.add_argument(
        "--input_format",
        type=str,
        choices=["every4", "every2", "noSpace"],
        default="every4",
        help="Input format of the context in datasets.",
    )

    parser.add_argument(
        "--pkt_repr_dim",
        type=int,
        default=768,
        help="Dimension of the packet representation.",
    )

    parser.add_argument(
        "--use_pkt_reduction",
        action=argparse.BooleanOptionalAction,
        help="The user can decide if to introduce additional internal linear layers to reducce the dimension of the packet representation from 768 to 'pkt_repr_dim'."
    )

    parser.add_argument(
        "--fix_encoder",
        action=argparse.BooleanOptionalAction,
        help="The user can decide if to fine-tune the encoder."
    )
    parser.add_argument(
        "--save_embeddings",
        action=argparse.BooleanOptionalAction,
        help="The user can decide if to save or not the packet representations."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether to enforce the script not to use GPU even if available.",
    )
    parser.add_argument(
        "--store_embeddings",
        action="store_true",
        help="Whether to store the embeddings of the processed sentences.",
    )

    parser.add_argument(
        "--clean_start",
        action="store_true",
        help="Flag indicating whether, given the same experiment ID, we want to clean previous logs, results and cache before starting.",
    )

    parser.add_argument(
        "--identifier", type=str, default="classification", help="Special identifier."
    )

    parser.add_argument(
        "--experiment", type=str, default="classification", help="Special identifier."
    )

    # Data
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path where to save the run's output. Used only if using a finetuned model from Huggingface for inference.",
    )

    parser.add_argument(
        "--training_data",
        required=True,
        type=str,
        help="Path to the input data. Supposed to be a parquet or csv file. See more in the README.",
    )

    parser.add_argument(
        "--validation_data",
        type=str,
        default="",
        help="Path to the validation data. Alternative to obtaining the validation set from the training.",
    )

    parser.add_argument(
        "--testing_data",
        type=str,
        default="",
        help="Path to the test data.",
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Chosen model.")
    parser.add_argument(
        "--finetuned_path_model",
        type=str,
        default="Empty",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_model' (run a simple training for an example).",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="normal",
        choices=["normal", "weighted"],
        help="Selection of the type of loss computed during training: normal (all classes are equal), weighted (each class as a weight of total_samples / class_samples)",
    )

    parser.add_argument(
        "--finetuned_path_bottleneck",
        type=str,
        default="",
        help="Path to the domain-adapted finetuned bottleneck model. Remember that the path must point to a folder that contains a subfolder 'best_bottleneck_model' (run a simple training for an example).",
    )

    parser.add_argument(
        "--finetuned_path_classification",
        type=str,
        default="",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_bottleneck_model' (run a simple training for an example).",
    )
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Chosen tokenizer. \
                        Can be a simple name (e.g., bert-uncased) or the path to the finetuned tokenizer.",
    )

    # Hyper-parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="How many samples per batch."
    )

    parser.add_argument(
        "--percentage",
        type=int,
        default=50,
        help="Number of rows taken from the dataset.",
    )
    
    parser.add_argument(
        "--max_qst_length",
        type=int,
        default=256,
        help="Maximum number of tokens before truncation (in question).",
    )
    parser.add_argument(
        "--max_ans_length",
        type=int,
        default=32,
        help="Maximum number of tokens before truncation (in answer).",
    )
    parser.add_argument(
        "--max_chunk_length",
        type=int,
        default=512,
        help="Maximum number of tokens before truncation.",
    )

    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs before stopping training.",
    )
    
    parser.add_argument(
        "--lr",
        default=5e-6,
        type=float,
        help="Number of epochs before stopping training.",
    )

    opts = parser.parse_args(args)
    # We use GPU if any available
    opts.use_cuda = is_available() and not opts.no_cuda
    # Set training stage
    opts.training_stage = "inference"
    opts.run_name = opts.identifier
    # Option only for training > set as placeholders
    opts.available_percentage = 1
    opts.eval_size = 0
    return vars(opts)
