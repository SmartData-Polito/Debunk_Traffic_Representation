# ET-BERT

**The changed repository of ET-BERT, a network traffic classification model on encrypted traffic.**

Note: this code is based on [ET-BERT](https://github.com/linwhitehat/ET-BERT). Many thanks to the authors. We changed the python file under fine-tuning

![The framework of ET-BERT](images/etbert.png)

The work is introduced in the *[31st The Web Conference](https://www2022.thewebconf.org/)*:
> Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi and Jing Yu. 2022. ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification. In Proceedings of The Web Conference (WWW) 2022, Lyon, France. Association for Computing Machinery. 
<br/>

Table of Contents
=================
  * [Requirements](#requirements)
  <!-- * [Datasets](#datasets) -->
  * [Using ET-BERT](#using-et-bert)
<br/>

## Requirements
* Python >= 3.6
* CUDA: 11.4
* GPU: Tesla V100S
* torch >= 1.1
* six >= 1.12.0
* scapy == 2.4.4
* numpy == 1.19.2
* shutil, random, json, pickle, binascii, flowcontainer
* argparse
* packaging
* tshark
* [SplitCap](https://www.netresec.com/?page=SplitCap)
* [scikit-learn](https://scikit-learn.org/stable/)
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with wordpiece model you will need [WordPiece](https://github.com/huggingface/tokenizers)
* For the use of CRF in sequence labeling downstream task you will need [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
<br/>

## Using ET-BERT
You can now use ET-BERT directly through the pre-trained [model](https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view?usp=sharing) or download via:
```
wget -O pretrained_model.bin https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view?usp=sharing
```

After obtaining the pre-trained model, ET-BERT could be applied to the spetic task by fine-tuning at packet-level with labeled network traffic:
```
python3 fine-tuning/run_classifier_ori.py --pretrained_model_path models/pre-trained_model.bin \
                                          --vocab_path models/encryptd_vocab.txt \
                                          --train_path data/vpn-app/train_val_split_0 \
                                          --dev_path data/vpn-app/train_val_split_0 \
                                          --test_path data/vpn-app \
                                          --epochs_num 10 --batch_size 32 --embedding word pos seg \
                                          --encoder transformer --mask fully_visible \
                                          --seq_length 128 --learning_rate 2e-5 -dataset vpn-app \
                                          --frozen --output_model_path outputs/train_val_split_0 \
```
<br/>

## Citation

```
@inproceedings{lin2022etbert,
  author    = {Xinjie Lin and
               Gang Xiong and
               Gaopeng Gou and
               Zhen Li and
               Junzheng Shi and
               Jing Yu},
  title     = {{ET-BERT:} {A} Contextualized Datagram Representation with Pre-training
               Transformers for Encrypted Traffic Classification},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022, Virtual Event, Lyon, France,
               April 25 - 29, 2022},
  pages     = {633--642},
  publisher = {{ACM}},
  year      = {2022}
}
```

<br/>