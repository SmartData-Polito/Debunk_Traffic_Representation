import sys  # nopep8

sys.path.append("../../")  # nopep8
from Core.functions.option_parser import get_inference_options
from Core.classes.dataset_for_flowClassification import Classification_Dataset, Flow_Classification_Dataset
from Core.classes.logger import TrainingExperimentLogger
from Core.classes.flowClassification_model import Flow_Classification_Model
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
    test_dataset_obj = Classification_Dataset(opts, tokenizer_obj)
    test_dataset_obj.load_dataset("Test", opts["testing_data"], opts['input_format'], percentage=opts['percentage'], pkts_in_flow=opts["pkts_in_flow"])
    if opts["flow_level"] == "representation_concat":
        trainval_dataset_obj = Classification_Dataset(opts, tokenizer_obj)
        trainval_dataset_obj.load_dataset(
            "Train", opts["training_data"], opts['input_format'], opts["validation_data"], opts["percentage"], opts["pkts_in_flow"]
        )
        ### Model ###
        logger.accelerator.print(f"Load the model...")
        model_obj = Flow_Classification_Model(opts, tokenizer_obj, test_dataset_obj, trainval_dataset_obj)
    
    else:
        ### Model ###
        logger.accelerator.print(f"Load the model...")
        model_obj = Flow_Classification_Model(opts, tokenizer_obj, test_dataset_obj)
    ### Train ###
    logger.accelerator.print(f"Start classification test...")
    model_obj.run(logger, opts)

    ### End ###
    logger.accelerator.print(f"End classification test...")
    logger.end_experiment()


if __name__ == "__main__":
    run(get_inference_options())
