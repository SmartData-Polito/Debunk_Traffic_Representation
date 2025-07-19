# this script finetunes a model on a test dataset

python \
    src/train/NetfoundFinetuning.py \
    --train_dir data/test/finetuning/final/combined \
    --model_name_or_path models/test/pretraining/pretrained_model \
    --output_dir models/test/finetuning/finetuned_model \
    --report_to tensorboard \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_train \
    --do_eval \
    --eval_strategy epoch \
    --save_strategy epoch \
    --learning_rate 0.01 \
    --num_train_epochs 4 \
    --problem_type single_label_classification \
    --num_labels 2 \
    --load_best_model_at_end


