# Debunking Represenation Learning for Encrypted Traffic Classification

This repository contains the implementation of various baseline algorithms and data preprocessing methods.

## Folder Structure

- **code**: This folder contains the implementation of various baseline algorithms.
- **process_finetune_data**: This folder contains the methods for finetuned data preprocessing.

## Baseline Algorithms

The `code` folder includes the following baseline algorithms:
- Pcap Encoder
- [ET-BERT](https://github.com/linwhitehat/ET-BERT)
- [YaTC](https://github.com/NSSL-SJTU/YaTC)
- [NetMamba](https://github.com/wangtz19/NetMamba)
- [TrafficFormer](https://github.com/IDP-code/TrafficFormer)
- [netFound](https://github.com/SNL-UCSB/netFound)
- [AutoGluon](https://auto.gluon.ai/stable/index.html) based Feature Engineering

**Many thanks to authors for contributions.**

## Data Preprocessing

The `process_finetune_data` folder includes the following data preprocessing methods:
- **Split**: Per-packet/Per-flow split
- **Filter**: How to filter the unrelevant protocal packets
- **Data processing** (pkt/flow) of different models

Data is available on [HuggingFace](https://huggingface.co/datasets/rigcor7/Debunk_Traffic_Representation)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all previous works.
