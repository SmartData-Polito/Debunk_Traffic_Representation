import argparse
import subprocess
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_args():
    description = """
    This script preprocesses the raw pcap data into the tokenized format. It takes the input folder as an argument and one of two required flags: --pretrain or --finetune.
    The input folder must contain '/raw' folder with either raw pcap files (for pretraining, no labels) or folders with pcap files (finetuning, folder names must be integers and would be used as labels).
    The input folder would be used for intermediate files and the final tokenized data would be stored in the <input_folder>/final/shards folder as Apache Arrow shards.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_folder", type=str, required=True, help="The input folder")
    parser.add_argument("--action", choices=["pretrain", "finetune"], required=True,
                        help="Preprocess data for pretraining or finetuning.")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="The tokenizer config file.")
    parser.add_argument("--tcp_options", action="store_true", default=False, help="Include TCP options in the tokenized data.")
    parser.add_argument("--combined", action="store_true", default=False,
                        help="Combine all the pcap files in the /final/shards into a single file (suitable for small datasets).")

    return parser


def run(command: list[str]) -> subprocess.CompletedProcess:
    logger.info(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, check=True, capture_output=True)
    if process.stderr:
        logger.error(process.stderr.decode())
    if process.stdout:
        logger.info(process.stdout.decode())
    return process


def get_base_directory(args):
    # one step up of the directory of this file
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def preprocess_pretrain(args):
    base_directory = get_base_directory(args)
    input_folder = args.input_folder
    run([f"{base_directory}/src/pre_process/1_filter.sh", f"{input_folder}/raw", f"{input_folder}/filtered"])
    run([f"{base_directory}/src/pre_process/2_pcap_splitting.sh", f"{input_folder}/filtered", f"{input_folder}/split"])
    run([f"{base_directory}/src/pre_process/3_extract_fields.sh", f"{input_folder}/split", f"{input_folder}/extracted", "1" if args.tcp_options else ""])

    for folder_name in os.listdir(f"{input_folder}/extracted"):
        full_folder_name = os.path.join(f"{input_folder}/extracted", folder_name)
        os.makedirs(os.path.join(f"{input_folder}/final/shards", folder_name), exist_ok=True)
        run(["python3", f"{base_directory}/src/pre_process/Tokenize.py", "--conf_file", args.tokenizer_config,
             "--input_dir", full_folder_name, "--output_dir",
             os.path.join(f"{input_folder}/final/shards", folder_name)])
        if args.combined:
            os.makedirs(os.path.join(f"{input_folder}/final", "combined"), exist_ok=True)
            run(["python3", f"{base_directory}/src/pre_process/CollectTokensInFiles.py",
                 os.path.join(f"{input_folder}/final/shards", folder_name),
                 os.path.join(f"{input_folder}/final/combined", f"{folder_name}.arrow")])


def preprocess_finetune(args, path_list):
    base_directory = get_base_directory(args)
    input_folder = args.input_folder

    for label in os.listdir(os.path.join(input_folder, "raw", *path_list)):
        for stage_name in ["filtered", "split", "extracted", "final/shards"]:
            os.makedirs(os.path.join(input_folder, stage_name, *path_list, label), exist_ok=True)
        run([f"{base_directory}/src/pre_process/1_filter.sh", os.path.join(input_folder, "raw", *path_list, label),
             os.path.join(input_folder, "filtered", *path_list, label)])
        run([f"{base_directory}/src/pre_process/2_pcap_splitting.sh", os.path.join(input_folder, "filtered", *path_list, label),
             os.path.join(input_folder, "split", *path_list, label)])
        run([f"{base_directory}/src/pre_process/3_extract_fields.sh", os.path.join(input_folder, "split", *path_list, label),
             os.path.join(input_folder, "extracted", *path_list, label), "1" if args.tcp_options else ""])

        for folder_name in os.listdir(os.path.join(input_folder, "extracted", *path_list, label)):
            full_folder_name = os.path.join(input_folder, "extracted", *path_list, label, folder_name)
            os.makedirs(os.path.join(input_folder, "final/shards", *path_list, label, folder_name), exist_ok=True)
            run(["python3", f"{base_directory}/src/pre_process/Tokenize.py", "--conf_file", args.tokenizer_config,
                 "--input_dir", full_folder_name, "--output_dir",
                 os.path.join(input_folder, "final/shards", *path_list, label, folder_name), '--label', label])
            if args.combined:
                os.makedirs(os.path.join(input_folder, "final", "combined", *path_list), exist_ok=True)
                run(["python3", f"{base_directory}/src/pre_process/CollectTokensInFiles.py",
                     os.path.join(input_folder, "final/shards", *path_list, label, folder_name),
                     os.path.join(input_folder, "final/combined", *path_list, f"{label}_{folder_name}.arrow")])

def main():
    parser = get_args()
    args = parser.parse_args()
    input_folder = args.input_folder
    action = args.action

    for folder in ["filtered", "split", "extracted", "final", "final/shards"]:
        os.makedirs(os.path.join(input_folder, folder), exist_ok=True)

    raw_data_folder = os.path.join(input_folder, "raw")

    for exp in os.listdir(raw_data_folder): # flow/polishedns/polishednsLen811
        if exp != 'flow':
            continue
        for task in os.listdir(os.path.join(raw_data_folder, exp)): # tls/vpn-app
            if task != 'vpn-app':
                continue
            for split_folder in os.listdir(os.path.join(raw_data_folder, exp, task)): #test/train_val_split_0/1/2
                if split_folder == 'test':
                    preprocess_finetune(args, [exp, task, split_folder])
                else:
                    for type in os.listdir(os.path.join(raw_data_folder, exp, task, split_folder)): # train/val
                        preprocess_finetune(args, [exp, task, split_folder, type])

if __name__ == "__main__":
    main()
