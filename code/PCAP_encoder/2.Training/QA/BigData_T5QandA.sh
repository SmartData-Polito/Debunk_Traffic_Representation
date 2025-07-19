#EXPERIMENT MACRO-DATA
TASK="supervised"
LOG_LEVEL="info"
OUTPUT_PATH="./results/"
TYPE_PRETRAIN="QA" # can assume values [QA, denoiser]

#Model info
FINETUNED_PATH_MODEL="Empty" # If needed, insert the path for the finetuned model
#FINETUNED_PATH_MODEL="results/denoiser/supervised/T5-base_standard-tokenizer/TrashTrain_100K_512_0005_mean_Linked/task-supervised_lr-0.0005_epochs-2_batch-2/seed_43/best_model"
#MODEL_NAME="results/supervised/T5-base_standard-tokenizer/Denoiser_trainingTrain_512_512_Perc30_600K_mean_batch8_0005_epochs10_015/task-supervised_lr-0.00$
MODEL_NAME="T5-base"
TOKENIZER_NAME="T5-base" #if you use a finetuned tokenizer, specify the path
GPU=(0) #CHOOSE GPUs HERE. Elements in bash lists shall look like `(GPU1 GPU2 ...)`
GPU_STRING="$(IFS=, ; echo "${GPU[*]}"),"
PORT=2950"${GPU[0]}"
BOTTLENECK="Luong"

#Training details
BATCH_SIZE=8
EPOCHS=8 #Training epochs
LR=0.0005
MAX_QST_LENGTH=512
MAX_ANS_LENGTH=32
PERC=10 # value between [1,100]
SEED=43

#Input info
TRAINING_DATA="../../1.Datasets/QA/My_TCPIPv4.parquet"
TEST_DATA="../../1.Datasets/QA/My_TCPIPv4.parquet"

EXPERIMENT="Exp_refactoring"
IDENTIFIER="QandA_Luong_scheduler0005_constant_not0"

export GPUS_PER_NODE=1

export SCRIPT=train.py

export SCRIPT_ARGS=" \
    --identifier $IDENTIFIER --experiment $EXPERIMENT --task $TASK --clean_start\
    --tokenizer_name $TOKENIZER_NAME --type_pretrain $TYPE_PRETRAIN --lr $LR\
    --model_name $MODEL_NAME --log_level $LOG_LEVEL --output_path $OUTPUT_PATH\
    --training_data $TRAINING_DATA --epochs $EPOCHS --batch_size $BATCH_SIZE \
    --seed $SEED --bottleneck $BOTTLENECK --max_qst_length $MAX_QST_LENGTH\
    --max_ans_length $MAX_ANS_LENGTH --percentage $PERC --gpu $GPU_STRING\
    --finetuned_path_model $FINETUNED_PATH_MODEL --test_data $TEST_DATA\
    "

#accelerate launch --num_processes=$GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS

accelerate launch --num_processes=$GPUS_PER_NODE --num_processes=1 -q \
        --main_process_port=$PORT --mixed_precision="no" $SCRIPT $SCRIPT_ARGS