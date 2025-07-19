TASK="inference"
LOG_LEVEL="info"
OUTPUT_PATH="./results/"

#Model info
MODEL_NAME="T5-base" #Chosen model
FINETUNED_PATH_MODEL="../4_QA_model_training/results/QA/supervised/T5-base_standard-tokenizer/TrainingQAdenoiser0450K_QA_hard_mean_seed43_512/task-supervised_lr-0.0005_epochs-20_batch-24/seed_43/best_model"

FINETUNED_PATH_BOTTLENECK="Empty"
#FINETUNED_PATH_CLASS="../5_classification_training/results/supervised/T5-base_standard-tokenizer/Task6_den450_QA_filter_ver0_mean_lr0.0001_seed43_lossnormal_batch24_UnFROZEN/task-supervised_lr-0.0001_epochs-20_batch-24/seed_43/best_model"
FINETUNED_PATH_CLASS=""
TOKENIZER_NAME="T5-base" #if you use a finetuned tokenizer, specify the path 
GPU=(0) #CHOOSE GPUs HERE. Elements in bash lists shall look like `(GPU1 GPU2 ...)` 
GPU_STRING="$(IFS=, ; echo "${GPU[*]}"),"
PORT=2950"${GPU[0]}"
BOTTLENECK="mean"

#Training details
BATCH_SIZE=24
MAX_QST_LENGTH=512
MAX_ANS_LENGTH=32
PERCENTAGE=100
SEED=43
EPOCHS=500
LR=0.0001
PKT_IN_FLOW=5
PKT_REPR_DIM=256
FLOW_LEVEL="majority_vote"

for i in 0 1 2 
do
    FINETUNED_PATH_MODEL="../5_classification_training/results/supervised/T5-base_standard-tokenizer/Task3_den450_QA_filter_ver${i}_mean_lr0.0001_seed43_lossnormal_batch24_UnFROZEN/task-supervised_lr-0.0001_epochs-20_batch-24/seed_43/best_encoder"
    FINETUNED_PATH_CLASS="../5_classification_training/results/supervised/T5-base_standard-tokenizer/Task3_den450_QA_filter_ver${i}_mean_lr0.0001_seed43_lossnormal_batch24_UnFROZEN/task-supervised_lr-0.0001_epochs-20_batch-24/seed_43/best_model"
    EXPERIMENT="flow_class_majorityvoting"
    IDENTIFIER="Task3_flowMV_den450KQA0005_Hard_batch${BATCH_SIZE}_pktFlow${PKT_IN_FLOW}_ver${i}_unFrozen"
    TRAIN_DATA="../../1.Datasets/Classification/Task3_flow/train_${i}.parquet"
    VAL_DATA="../../1.Datasets/Classification/Task3_flow/val_${i}.parquet"
    TEST_DATA="../../1.Datasets/Classification/Task3_flow/test.parquet"
    export GPUS_PER_NODE=1
    export SCRIPT=../../2.Training/classification/flow_classification.py
    export SCRIPT_ARGS=" \
    --identifier $IDENTIFIER --experiment $EXPERIMENT --task $TASK --clean_start\
    --tokenizer_name $TOKENIZER_NAME --finetuned_path_model $FINETUNED_PATH_MODEL\
    --model_name $MODEL_NAME --finetuned_path_classification $FINETUNED_PATH_CLASS\
    --log_level $LOG_LEVEL --output_path $OUTPUT_PATH --testing_data $TEST_DATA\
    --training_data $TRAIN_DATA --validation_data $VAL_DATA --epochs $EPOCHS --lr $LR\
    --batch_size $BATCH_SIZE --seed $SEED --bottleneck $BOTTLENECK --gpu $GPU_STRING\
    --max_ans_length $MAX_ANS_LENGTH --percentage $PERCENTAGE --max_qst_length $MAX_QST_LENGTH\
    --finetuned_path_bottleneck $FINETUNED_PATH_BOTTLENECK --flow_level $FLOW_LEVEL\
    --pkts_in_flow $PKT_IN_FLOW\
    "    
#--pkt_repr_dim $PKT_REPR_DIM --use_pkt_reduction\
    
    accelerate launch --num_processes=$GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS
done
#accelerate launch --num_processes=$GPUS_PER_NODE --num_processes=1 -q \
 #       --main_process_port=$PORT --mixed_precision="no" $SCRIPT $SCRIPT_ARGS
        
