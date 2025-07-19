<div align="center">
<h1>NetMamba </h1>
<h3>Efficient Network Traffic Classification via Pre-training Unidirectional Mamba</h3>

Note: this code is based on [NetMamba](https://github.com/wangtz19/NetMamba). Many thanks to the authors. We changed the python fine-tuning file and evaluate function.

[Tongze Wang](https://github.com/wangtz19), [Xiaohui Xie](https://thuxiexiaohui.github.io/), [Wenduo Wang](https://github.com/Viz7), [Chuyi Wang](https://github.com/Judy456abc), [Youjian Zhao](https://www.cs.tsinghua.edu.cn/info/1126/3576.htm), [Yong Cui](https://www.cuiyong.net/index.html)

ICNP 2024 ([arXiv paper](https://arxiv.org/abs/2405.11449))
</div>

## Overview
<div align="center">
<img src="assets/NetMamba.png" />
</div>

## Environment Setup
- Create python environment
    - `conda create -n NetMamba python=3.10.13`
    - `conda activate NetMamba`
- Install PyTorch 2.1.1+cu121 (we conduct experiments on this version)
    - `pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121`
- Install Mamba 1.1.1
    - `cd mamba-1p1p1`
    - `pip install -e .`
- Install other dependent libraries
    - `pip install -r requirements.txt`

- Note: if above steps don't work, one possible solution:
    - `conda install nvidia/label/cuda-11.6.0::cuda-nvcc -y`
    - `Download and install right version of cuda-cudart, cuda-cudart-dev, libcusparse-dev, libcublas-dev, libcusolver-dev, cuda-cccl using conda`

    - `git clone https://github.com/Dao-AILab/causal-conv1d.git`
    - `git checkout v1.1.1`
    - `cd causal-conv1d && pip install .`

    - `cd ../mamba-1p1p1 && pip install -e .`
    - `pip install tensorboard timm scikit-learn`

## Data Preparation
### Download our processed datasets
For simplicity, you are welcome to download our processed datasets on which our experiments are conducted from [google drive](https://drive.google.com/drive/folders/1C1urXBhk09V7Z80Kk5JYuP7QeXiedUIl?usp=sharing). 

Each dataset is organized into the following structure:
```text
.
|-- train
|   |-- Category 1
|   |   |-- Sample 1
|   |   |-- Sample 2
|   |   |-- ...
|   |   `-- Sample M
|   |-- Category 2
|   |-- ...
|   `-- Catergory N
|-- test
`-- valid
```
### Process your own datasets
If you'd like to generate customized datasets, please refer to preprocessing scripts provided in [dataset](https://github.com/wangtz19/NetMamba/tree/main/dataset). Note that you need to change several file paths accordingly.

## Run NetMamba

```
- Run fine-tuning (including evaluation)
```shell
python src/fine-tune.py --blr 2e-3 \
                        --epochs 120 \
                        --nb_classes 16 \
                        --finetune models/pretrained_model.pth \
                        --data_path ./data_flow/vpn-app/train_val_split_0 \
                        --test_path ./data_flow/vpn-app/test \
                        --output_dir ./outputs/train_val_split_0 \
                        --log_dir ./logs/train_val_split_0 \
                        --model net_mamba_classifier \
                        --no_amp \
                        --frozen \
                        --dataset vpn-app
```

## Checkpoint
The pre-trained checkpoint of NetMamba is available for download on our [huggingface repo](https://huggingface.co/wangtz/NetMamba). Feel free to access it at your convenience. If you require any other type of checkpoints, please contact us via email (wangtz23@mails.tsinghua.edu.cn).

## Citation
```
@inproceedings{wang2024netmamba,
  title={Netmamba: Efficient network traffic classification via pre-training unidirectional mamba},
  author={Wang, Tongze and Xie, Xiaohui and Wang, Wenduo and Wang, Chuyi and Zhao, Youjian and Cui, Yong},
  booktitle={2024 IEEE 32nd International Conference on Network Protocols (ICNP)},
  pages={1--11},
  year={2024},
  organization={IEEE}
}
```