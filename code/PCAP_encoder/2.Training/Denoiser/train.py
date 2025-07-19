import sys  # nopep8


sys.path.append("../../")  # nopep8
from Core.functions.option_parser import get_training_options
from Core.classes.dataset_for_denoiser import Denoiser_Dataset
from Core.classes.logger import TrainingExperimentLogger
from Core.classes.T5_model import T5_PCAP_translator
from Core.classes.tokenizer import QA_Tokenizer_T5


def run(opts):
    
    ### Experiment Initialization ###
    logger = TrainingExperimentLogger(opts)
    logger.start_experiment(opts)
    logger.accelerator.print(f"Running experiment '{opts['identifier']}' started!")
    ### Tokenizer ###
    tokenizer_obj = QA_Tokenizer_T5(opts)
    ### Dataset ###
    logger.accelerator.print(f"Loading and processing the dataset...")
    dataset_obj = Denoiser_Dataset(opts, tokenizer_obj, opts["denoiser_CR"])
    dataset_obj.load_dataset(opts["training_data"], opts["test_data"], opts['input_format'])
    ### Model ###
    logger.accelerator.print(f"Load the model...")
    model_obj = T5_PCAP_translator(opts, tokenizer_obj, dataset_obj)
    ### Train ###
    logger.accelerator.print(f"Start training...")
    model_obj.run(logger, opts)
   
    ### End ###
    logger.accelerator.print(f"End training...")
    logger.end_experiment()


if __name__ == "__main__":
    run(get_training_options())
