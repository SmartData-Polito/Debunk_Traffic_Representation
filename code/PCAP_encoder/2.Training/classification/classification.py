import sys  # nopep8

sys.path.append("../../")  # nopep8
from Core.functions.option_parser import get_classification_options
from Core.classes.dataset_for_classification import Classification_Dataset
from Core.classes.logger import TrainingExperimentLogger
from Core.classes.classification_model import Classification_model
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
    dataset_obj_trainval = Classification_Dataset(opts, tokenizer_obj)
    dataset_obj_trainval.load_dataset(
        "Train", opts["training_data"], opts['input_format'], opts["validation_data"], opts["percentage"]
    )
    dataset_obj_test = Classification_Dataset(opts, tokenizer_obj)
    dataset_obj_test.load_dataset("Test", opts["testing_data"], opts['input_format'])
    ### Model ###
    logger.accelerator.print(f"Load the model...")
    model_obj = Classification_model(opts, tokenizer_obj, dataset_obj_trainval, dataset_obj_test)
    ### Train ###
    logger.accelerator.print(f"Start classification train...")
    model_obj.run(logger, opts)

    ### End ###
    logger.accelerator.print(f"Experiment ended!")
    logger.end_experiment()


if __name__ == "__main__":
    run(get_classification_options())
