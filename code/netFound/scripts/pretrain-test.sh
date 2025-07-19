# this script pretrains a model on a test dataset

# TODO(maybe-hello-world): understand why safetensors are not usable and resolve

python \
    src/train/NetfoundPretraining.py \
    --train_dir data/test/pretraining/final/combined/ \
    --output_dir models/test/pretraining/pretrained_model \
    --report_to tensorboard \
    --do_train \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --save_safetensors false \
    --mlm_probability 0.10 \
    --learning_rate 2e-5 \
    --do_eval \
    --validation_split_percentage 30
