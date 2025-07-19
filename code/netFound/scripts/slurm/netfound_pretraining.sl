#!/bin/bash

### ATTENTION: this script is provided for reference only to give an idea how to make it run on SLURM-based systems
### you need to adjust the script to your needs and test it properly

### START: do not change this
#SBATCH --account=<your_project>
#SBATCH --licenses=<your_licenses>
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=<your_constraints>
#SBATCH --gpus-per-node=<your_gpus_per_node>
### END: do not change this

# --constraint= gpu with 80gb - should fit batch size 20 with the big model
# --constraint= gpu with 40gb - should fit batch size 8 with big model and 16 with medium model

### START: usually you do not need to change this
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=<your_qos>
### END: usually you do not need to change this

### START: feel free to change this
#SBATCH --job-name=<your_job_name>
#SBATCH --time=<your_time>
#SBATCH --nodes=<your_nodes>
### END: feel free to change this

### START: do not change this
set -x -e

module load cudatoolkit
module load cray-mpich
module load gcc
module load conda
conda activate <conda environment>

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WORLD_SIZE=$(($GPUS_PER_NODE*$SLURM_NNODES))
### END: do not change this

## add this to resume from checkpoint
## --resume_from_checkpoint checkpoint_path \
## --ignore_data_skip True \
## ignore data skip until https://github.com/huggingface/transformers/pull/33544 is merged



srun --ntasks-per-node=1 --gpus-per-node=4 --jobid $SLURM_JOBID bash -c '\
    torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $PSCRATCH/network-data-representation/src/train/NetFoundPretraining.py \
    --report_to tensorboard \
    --save_safetensors false \
    --dispatch_batches False \
    --bf16 \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --save_strategy steps \
    --dataloader_num_workers 32 \
    --dataloader_prefetch_factor 16 \
    --logging_steps 20000 \
    --save_steps 20000 \
    --streaming True \
    --gradient_accumulation_steps 1 \
    --hidden_size 1024 \
    --num_hidden_layers 24 \
    --num_attention_heads 16 \
    --tcpoptions False \
    --train_dir /path/to/train_data \
    --test_dir /path/to/eval \
    --output_dir /path/to/modelXX \
    --model_name_or_path /path/to/checkpointYY \
    --per_device_train_batch_size 20 \
    --learning_rate 0.000013 \
    --lr_scheduler_type linear \
    --warmup_steps 5000 \
    --max_steps 300000
    '