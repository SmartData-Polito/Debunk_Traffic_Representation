# netFound

**The modified repository of netFound.**

Note: this code is based on [netFound](https://github.com/SNL-UCSB/netFound). Many thanks to the authors. We changed the Python file under fine-tuning: we added validation dataset configuration, modified the evaluation metrics, etc. We also provided our multi-GPU training code based on netFound original version.

## Checkpoint
https://huggingface.co/snlucsb/netFound-640M-base
The checkpoint is pretrained on ~450mln flows of the real-world network traffic of the University of California, Santa Barbara.  
As the checkpoint is built on the Large version of the netFound, use `--netfound_large True` as a fine-tuning flag.  

# Bring Your Own Data (BYOD) 
To train or fine-tune **netFound** on your own dataset, follow the steps below to **preprocess and tokenize your PCAP files**. 
## Preprocessing Your Dataset 
The easiest way to preprocess your dataset is to use the **`scripts/preprocess_data.py`** script. 
### Folder Structure for Pretraining 
Organize your dataset as follows: 
```
folder_name/
 ├── raw/
 │   ├── file1.pcap
 │   ├── file2.pcap
 │   ├── ...
```
Then, run the following command: 
```bash
python3 scripts/preprocess_data.py --input_folder folder_name --action pretrain --tokenizer_config configs/TestPretrainingConfig.json --combined
```
:small_blue_diamond: **What happens next?** 
- The script will generate **intermediate folders** (`extracted`, `split`, etc.). 
- The resulting **tokenized data** will be stored in the `"tokens"` folder. 
- The **`--combined`** flag merges all tokenized files into a single **Arrow** file (useful for training). 
- If you **remove `--combined`**, multiple **Arrow** files (one per PCAP) will be created—this is beneficial for parallel processing across multiple nodes. 
- You can **modify the tokenizer configuration** (`configs/TestPretrainingConfig.json`) to control how internal and external IPs are handled. 
### Folder Structure for Fine-Tuning 
To fine-tune netFound, structure your dataset into **class-separated folders**, where **folder names should be integers** (used as class labels). 
```
raw/
 ├── 0/
 │   ├── class1_sample1.pcap
 │   ├── class1_sample2.pcap
 │   ├── ...
 ├── 1/
 │   ├── class2_sample1.pcap
 │   ├── class2_sample2.pcap
 │   ├── ...
```
Run the preprocessing script again, changing the `--action` to `finetune`: 
```bash
python3 scripts/preprocess_data.py --input_folder folder_name --action finetune --tokenizer_config configs/TestPretrainingConfig.json --combined
```
**Fine-Tuning Notes:** 
- **Class labels must be integers** (e.g., `1, 2, 3, ...`). 
- The resulting **Arrow files** will include a `"labels"` column. 
- You can **manually edit the `"labels"` column** for **custom class adjustments** (including regression tasks).
- As default validation data split does not shuffle the data file before the split, if your data is not shuffled, please use `scripts/shuffler.py` to shuffle the train file to ensure that the resulting test file contains instances of different classes.

## Finetuning netFound
```
python src/train/NetfoundFinetuning.py \
  --train_dir data/final/combined/flow/vpn-app/train_val_split_0 \
  --test_dir data/final/combined/flow/vpn-app/test \
  --model_name_or_path models/pytorch_model.bin \
  --overwrite_output_dir \
  --output_dir outputs \
  --do_train \
  --do_eval \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model accuracy \
  --greater_is_better True \
  --learning_rate 2.5e-6 \
  --num_train_epochs 1 \
  --problem_type single_label_classification \
  --num_labels 16 \
  --load_best_model_at_end \
  --save_safetensors false \
  --netfound_large True \
  --dataset_name vpn-app \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --freeze_base True
```
<br/>


## Citation
```
@misc{guthula2024netfoundfoundationmodelnetwork,
      title={netFound: Foundation Model for Network Security},
      author={Satyandra Guthula and Roman Beltiukov and Navya Battula and Wenbo Guo and Arpit Gupta and Inder Monga},
      year={2024},
      eprint={2310.17025},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2310.17025},
}
```
