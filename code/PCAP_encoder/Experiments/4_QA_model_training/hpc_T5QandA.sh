#EXPERIMENT MACRO-DATA
TASK="supervised"
LOG_LEVEL="info"
OUTPUT_PATH="./results/"
TYPE_PRETRAIN="QA" # can assume values [QA, denoiser]

#Model info
FINETUNED_PATH_MODEL="Empty" # If needed, insert the path for the finetuned model
#FINETUNED_PATH_MODEL="../3_denoiser_training/results/denoiser/supervised/T5-base_standard-tokenizer/Denoiser_trainingDenoiser450K_0_mean_CR0_SEED43/task-supervised_lr-0.0005_epochs-15_batch-8/seed_43/best_model"
#FINETUNED_PATH_MODEL="../4_QA_model_training/results/QA/supervised/T5-base_standard-tokenizer/TrainingQAdenoiser0450K_QA_hard_mean_seed43_512/task-supervised_lr-0.0005_epochs-20_batch-24/seed_43/best_model"
MODEL_NAME="T5-base"
TOKENIZER_NAME="T5-base" #if you use a finetuned tokenizer, specify the path
GPU=(0) #CHOOSE GPUs HERE. Elements in bash lists shall look like `(GPU1 GPU2 ...)`
GPU_STRING="$(IFS=, ; echo "${GPU[*]}"),"
PORT=2950"${GPU[0]}"

BOTTLENECK="mean"

#Training details
BATCH_SIZE=24
EPOCHS=20 #Training epochs
LR=0.0005
MAX_QST_LENGTH=512
MAX_ANS_LENGTH=32
PERC=100 # value between [1,100]
SEED=(43)

#Input info
TRAINING_DATA="../../1.Datasets/QA/Train_Hard.parquet"
TEST_DATA="../../1.Datasets/QA/Test_Hard.parquet"
PKT_REPR_DIM=768
export GPUS_PER_NODE=1

export SCRIPT=../../2.Training/QA/train.py

EXPERIMENT="TrainingQA"
for i in 0
do
    IDENTIFIER="_Hard_mean_seed${SEED[i]}_512_lr${LR}_noDen"
    export SCRIPT_ARGS=" \
    --identifier $IDENTIFIER --experiment $EXPERIMENT --task $TASK --clean_start\
    --tokenizer_name $TOKENIZER_NAME --type_pretrain $TYPE_PRETRAIN --lr $LR\
    --model_name $MODEL_NAME --log_level $LOG_LEVEL --output_path $OUTPUT_PATH\
    --training_data $TRAINING_DATA --epochs $EPOCHS --batch_size $BATCH_SIZE \
    --seed ${SEED[i]} --bottleneck $BOTTLENECK --max_qst_length $MAX_QST_LENGTH\
    --max_ans_length $MAX_ANS_LENGTH --percentage $PERC --gpu $GPU_STRING\
    --finetuned_path_model $FINETUNED_PATH_MODEL --test_data $TEST_DATA\
    "

    accelerate launch --num_processes=$GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS
done
#accelerate launch --num_processes=$GPUS_PER_NODE --num_processes=1 -q \
 #       --main_process_port=$PORT --mixed_precision="no" $SCRIPT $SCRIPT_ARGS
