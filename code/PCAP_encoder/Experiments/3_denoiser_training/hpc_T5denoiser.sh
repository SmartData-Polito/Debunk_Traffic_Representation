#EXPERIMENT MACRO-DATA
TASK="supervised"
LOG_LEVEL="info"
OUTPUT_PATH="./results/"
TYPE_PRETRAIN="denoiser" # can assume values [QA, denoiser]

#Model info
#FINETUNED_PATH_MODEL="../4_QA_model_training/results/QA/supervised/T5-base_standard-tokenizer/PayloadvsNoPayloadPayload_mean_seed43_512/task-supervised_lr-0.0005_epochs-20_batch-24/seed_43/best_model" # If needed, insert the path for the finetuned model
FINETUNED_PATH_MODEL="Empty"
MODEL_NAME="T5-base"
TOKENIZER_NAME="T5-base" #if you use a finetuned tokenizer, specify the path
GPU=(0 1 2) #CHOOSE GPUs HERE. Elements in bash lists shall look like `(GPU1 GPU2 ...)`
GPU_STRING="$(IFS=, ; echo "${GPU[*]}"),"
PORT=2950"${GPU[0]}"

BOTTLENECK="mean"
CR=0

#Training details
BATCH_SIZE=8
EPOCHS=15 #Training epochs
LR=0.0005
MAX_QST_LENGTH=512
MAX_ANS_LENGTH=512
PERC=100 # value between [1,100]
SEED=(43)


TRAINING_DATA="../../1.Datasets/Denoiser/Train_for_denoiser_450K.parquet"
TEST_DATA="../../1.Datasets/Denoiser/Test_denoiser.parquet"

export GPUS_PER_NODE=3

export SCRIPT=../../2.Training/Denoiser/train.py

for i in 0
do
    EXPERIMENT="Denoiser_training"
    IDENTIFIER="Denoiser450K_0_${BOTTLENECK}_CR0_SEED${SEED[i]}"
    export SCRIPT_ARGS=" \
        --identifier $IDENTIFIER --experiment $EXPERIMENT --task $TASK --clean_start\
        --tokenizer_name $TOKENIZER_NAME --type_pretrain $TYPE_PRETRAIN --lr $LR\
        --model_name $MODEL_NAME --log_level $LOG_LEVEL --output_path $OUTPUT_PATH\
        --training_data $TRAINING_DATA --epochs $EPOCHS --batch_size $BATCH_SIZE \
        --seed ${SEED[i]} --bottleneck $BOTTLENECK --max_qst_length $MAX_QST_LENGTH\
        --max_ans_length $MAX_ANS_LENGTH --percentage $PERC --gpu $GPU_STRING\
        --finetuned_path_model $FINETUNED_PATH_MODEL --test_data $TEST_DATA --denoiser_CR $CR\
        "

    accelerate launch --num_processes=$GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS
done
