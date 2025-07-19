import shutil
import os
import json
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from Core.functions.utils import create_dir, clean_folder
from warnings import warn
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path


class ExperimentLogger:
    """Basic class to keep track of the experiment logs. All following Loggers will inherit these classes."""

    def __init__(self, output_folder, gpu, max_number_checkpoints=1):
        # 1.Create log, checkpoints, cache and best model directories starting from the output folder
        #   1a. Inside this folder we store the checkpoints automatically saved by the model (at most `max_number_checkpoints` files)
        self.checkpoint_folder = create_dir(output_folder, "checkpoints")
        #   1b. Inside this folder we store embeddings + tensorboard results + current training status if interrupted
        self.log_dir = create_dir(output_folder, "logs")
        #   1c. Eventually, the cache folder (see `get_cache_folder` for more details)
        self.cache_dir = create_dir(self.get_cache_folder(output_folder), "cache")
        self.accelerator = self.load_accelerator(
            output_folder, max_number_checkpoints, gpu
        )
        # 2. Declare variables which are going to be initialized later in the code
        #   2a. Preparing tensorboard writer
        self.tensorboard_writer = None

    def get_cache_folder(self, output_folder):
        """Function to get the cache folder. By construction, this is simply the grand-parent folder of the output folder.
        In fact, cache is independent 1) from the seed, 2) from the training parameters (e.g., lr, n_epochs, etc.)
        Notice that we will use cache to store the processed data (e.g., tokenized, etc).
        Args:
            output_folder (str): Output path of the experiment (level in which we takes into account the seed and the training parameters)
        Returns:
            str -- Path of the cache folder.
        """
        experiment_folder = Path(output_folder)
        cache_folder = (
            experiment_folder.parent.parent
        )  # by construction, model_folder is 2-levels parent of experiment folder
        return cache_folder

    def load_accelerator(self, output_folder, max_number_checkpoints, gpu):
        project_configuration = ProjectConfiguration(
            project_dir=output_folder,
            logging_dir=output_folder,
            automatic_checkpoint_naming=True,
            total_limit=max_number_checkpoints,
        )
        use_cpu = True if gpu == "cpu" else False
        accelerator = Accelerator(
            log_with="tensorboard", project_config=project_configuration, cpu=use_cpu
        )
        return accelerator

    def log_config(self, config_file, partition="training"):
        #   Notice: We only want to log only with 1 process when we use multi-gpus
        if self.accelerator.is_main_process:
            with open(
                os.path.join(self.log_dir, f"{partition}_config.json"),
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(config_file, f, indent=4)

    def init_trackers(self):
        """Initialize the trackers, given an accelerator.
        Notice that having direct access to the tracker enables us more customized way of logging the results (e.g., logging text).
        """
        # logging folder = accelerator.log_dir + "logs"
        self.accelerator.init_trackers(project_name="logs")
        if self.accelerator.is_main_process:
            self.tensorboard_writer = self.accelerator.get_tracker("tensorboard").writer

    def set_exp_seed(self, seed):
        """Set the seed value for the experiment (https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py).
        Arguments:
            seed (int) -- The seed value.
        """
        set_seed(seed=seed)

    def print_ds_stats(self, ds):
        """Function to print the dataset stats (sizes)
        Arguments:
            ds (DatasetDict) -- Dataset object.
        """
        self.accelerator.print("\tSizes of the partitions:")
        for key in ds.keys():
            self.accelerator.print(f"\t\t{key}: {ds[key].shape[0]:,}")

    def register_optimizer_scheduler(self, optimizer, scheduler):
        """Register the optimizer and the learning rate scheduler for checkpointing.
        Arguments:
            optimizer (torch.optim.Optimizer) -- The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler) -- The learning rate scheduler.
        """
        self.accelerator.register_for_checkpointing(
            scheduler
        )  # Register the LR scheduler
        self.accelerator.register_for_checkpointing(optimizer)  # Register the optimizer

    def log_result(self, key, value, step):
        """Log the result of a metric to TensorBoard.
        Arguments:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int): The current step.
        """
        if self.accelerator.is_main_process:
            self.tensorboard_writer.add_scalar(key, value, step)
            self.tensorboard_writer.flush()

    def log_results(self, results, freq, partition, step):
        """Log the results of multiple metrics to TensorBoard.
        Arguments:
            results (Dict[str, float]): A dictionary containing the results.
            freq (str): The frequency of the metric.
            partition (str): The partition of the metric.
            step (int): The current step.
        """
        for key, value in results.items():
            self.log_result(key=f"{key}_{freq}/{partition}", value=value, step=step)

    def log_image(self, tag, fig, step):
        """Log an image to TensorBoard.
        Arguments:
            tag (str): The name of the image.
            fig (plt.Figure): The figure object to log.
            step (int): The current step.`
        """
        if self.accelerator.is_main_process:
            self.tensorboard_writer.add_figure(tag=tag, figure=fig, global_step=step)

    def end_experiment(self):
        """Close the TensorBoard writer and end the training."""
        if self.accelerator.is_main_process:
            self.tensorboard_writer.close()
        self.accelerator.end_training()

    def save_textual_report(self, results, id_result):
        """Save the textual report of the results in a markdown format on tensorboard.

        Arguments:
            results (Dict[str, float]): A dictionary containing the results.
            id_result (str): The ID of the result.
        """
        if self.accelerator.is_main_process:
            self.tensorboard_writer.add_text(id_result, str(results.to_markdown()), 1)


class TrainingExperimentLogger(ExperimentLogger):
    """Object for generic training (can be self-supervised or supervised). With respect to the base object, it saves a configuration file (with the parameters used for training) and saves the best model once trained."""

    def __init__(self, opts):
        # 0. Save state variables
        self.experiment_id = opts["experiment"] + opts["identifier"]
        self.experiment_taks = opts["task"]
        # 1. Create output folder for experiment
        #   1a. Create a single string concatenating the experiment's parameters
        self.experiment_parameters = self.get_experiment_params(opts)
        #   1b. Create a single string concatenating model name and tokenizer status
        self.model_info = self.get_model_info(opts["model_name"], "standard")
        #   1c. Concatenate all meaningful information about the experiment
        #       Those will behave as a human-interpretable hash for the experiment
        if "type_pretrain" in opts.keys():
            self.experiment_output_folder = self.create_experiment_folder(
                opts["output_path"], self.experiment_taks, opts["seed"], opts["type_pretrain"]
            )
        else:
            self.experiment_output_folder = self.create_experiment_folder(
                opts["output_path"], self.experiment_taks, opts["seed"]
            )

        #   2. Now, create default logger
        super().__init__(output_folder=self.experiment_output_folder, gpu=opts["gpu"])
        #   3. Some logging path of a TrainingExperiment only
        #       3a. Folder to store the best model
        self.path_best_model = create_dir(self.experiment_output_folder, "best_model")
        if not hasattr(opts, "fix_encoder"):
            self.path_best_encoder = create_dir(self.experiment_output_folder, "best_encoder")
        #       3b. Json file to keep track of training status
        #       Notice: I cannot write this file in the checkpoint folder (see https://github.com/huggingface/accelerate/blob/v0.26.1/src/accelerate/accelerator.py#L2649, line 2961)
        self.training_checkpoint_file = os.path.join(
            self.log_dir, "checkpoint_epoch_steps.json"
        )
        #   4. Eventually, set training stage to "training"
        self.training_stage = "training"

    def get_experiment_params(self, opts):
        if opts["task"] == "inference":
            return f"task-{opts['task']}__batch-{opts['batch_size']}"
        else:
            return f"task-{opts['task']}_lr-{opts['lr']}_epochs-{opts['epochs']}_batch-{opts['batch_size']}"

    def get_model_info(self, model_name, tokenizer_status):
        if tokenizer_status != "standard":
            # full tokenizer name would be verbose.
            # Remember to choose a meaningful experiment_id to avoid overwriting
            tokenizer_status = "finetuned"
        return f"{model_name}_{tokenizer_status}-tokenizer"

    def create_experiment_folder(self, output_path, experiment_task, seed, type_task=""):
        if type_task == "":
            experiment_folder = os.path.join(
                output_path,
                experiment_task,
                self.model_info,
                self.experiment_id,
                self.experiment_parameters,
                f"seed_{seed}",
            )
        else:
            experiment_folder = os.path.join(
                output_path,
                type_task,
                experiment_task,
                self.model_info,
                self.experiment_id,
                self.experiment_parameters,
                f"seed_{seed}",
            )
        return experiment_folder

    def get_experiment_name(self):
        experiment_name = (
            f"{self.model_info}_{self.experiment_id}_{self.experiment_parameters}"
        )
        return experiment_name

    def start_experiment(self, opts, start_fresh=False):
        self.accelerator.print(
            "\n"
            + "#" * 15
            + f" Running experiment {self.get_experiment_name()} for {self.experiment_taks} "
            + "#" * 15
            + "\n"
        )
        # 1. Clean previous logs (if any, if start fresh)
        self.clean_old_results(start_fresh)
        # 2. Log experiment config file (with details about training!)
        self.log_config(opts, partition="training")
        # 3. Init tensorboard trackers
        self.init_trackers()
        # 4. Set the experiment seed
        self.set_exp_seed(opts["seed"])

    def clean_old_results(self, start_fresh):
        #   Notice: We only want to delete only with 1 process when we use multi-gpus
        if start_fresh and self.accelerator.is_main_process:
            # clean logs and recreate folder
            clean_folder(self.experiment_output_folder, "logs")
            # clean best_model and recreate folder
            clean_folder(self.experiment_output_folder, "best_model")
            # clean checkpoints and recreate folder
            clean_folder(self.experiment_output_folder, "checkpoints")

    def is_checkpoint_available(self):
        """
        Check if a training checkpoint is available. Returns series of 0 if none is found.
        Returns:
            Tuple[int, int, int] -- A tuple containing the number of completed epochs, the number of training steps to skip, and the number of completed validation steps.
        """
        with self.accelerator.main_process_first():  # wait for all workers to have the weights
            if len(os.listdir(self.checkpoint_folder)) != 0:
                completed_epochs, train_step_2_skip, completed_validation_steps = (
                    self.retrieve_checkpoint_state()
                )
            else:
                completed_epochs, train_step_2_skip, completed_validation_steps = (
                    0,
                    0,
                    0,
                )
        return completed_epochs, train_step_2_skip, completed_validation_steps

    def retrieve_checkpoint_state(self):
        """Retrieve the last saved checkpoint state and return the number of epochs, steps performed, and validation steps if available.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of epochs, steps performed, and validation steps.
        """
        self.accelerator.load_state()  # Restore last available state (automatically take the last)
        with open(self.training_checkpoint_file, encoding="utf-8") as f:
            checkpoint = json.load(f)
        epochs = checkpoint["epochs"]
        steps_performed = checkpoint["steps_performed"]
        validation_steps = checkpoint["validation_steps"]
        return epochs, steps_performed, validation_steps

    def save_checkpoint(self, epoch_id, batch_id, validation_steps):
        """Save the current checkpoint state (current epoch, training batch and validation step) if different from the last previously saved.
        Arguments:
            epoch_id (int): The current epoch ID.
            batch_id (int): The current batch ID.
            validation_steps (int): The number of validation steps.
        """
        if os.path.isfile(self.training_checkpoint_file):
            with open(self.training_checkpoint_file, encoding="utf-8") as f:
                results = json.load(f)
                last_epoch, last_step = results["epochs"], results["steps_performed"]
        else:
            last_epoch, last_step = 0, 0
        if last_epoch != epoch_id or last_step != batch_id:
            self.accelerator.save_state()  # Checkpoint location is chosen automatically
            with open(self.training_checkpoint_file, "w+", encoding="utf-8") as f:
                checkpoint = {
                    "epochs": epoch_id,
                    "steps_performed": batch_id,
                    "validation_steps": validation_steps,
                }
                json.dump(checkpoint, f)

    def save_best_huggingface_model(self, model, type):
        unwrapped_model = self.accelerator.unwrap_model(model)
        if type == "HF_model":
            unwrapped_model.save_pretrained(
                self.path_best_model,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
        elif type == "custom_model":
            torch.save(
                unwrapped_model.state_dict(),
                f"{self.path_best_model}/weights.pth",
            )
        elif type == "custom_encoder":
            torch.save(
                unwrapped_model.state_dict(),
                f"{self.path_best_encoder}/weights.pth",
            )
        
