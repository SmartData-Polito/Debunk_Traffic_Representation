import os
os.chdir('debunk_representation/code/TrafficFormer/data_generation')
from finetuning_data_gen_flow import generation_multiP, dataset_extract, enhance_based_tsv
""
data_path = "debunk_representation/code/netFound/data/raw/flow/vpn-app"
# data_path = "/share/smartdata/external_pcaps/ISCX-VPN-2016/Filtered/App/flow"
meta_path = "debunk_representation/code/TrafficFormer/data_flow/json/vpn-app"
save_path = "debunk_representation/code/TrafficFormer/data_flow/dataset/vpn-app"
class_num = 16

subdirs = ["test/", 
           "train_val_split_0/train/", "train_val_split_0/val/",]

for subdir in subdirs:
    pcap_path = os.path.join(data_path, subdir)
    json_path = os.path.join(meta_path, subdir)
    tsv_path = os.path.join(save_path, subdir)
    os.makedirs(json_path, exist_ok=True)
    print("Process", pcap_path)
    print("Save to", json_path)

    # pcap_path: the path of splited pcap. In the pacp path, each dir is one class. In each dir, each pcap is one flow.
    # samples: samples * _category, samples is the maximum samples of each class, _category is the number of class.
    # dataset_save_path: generated dataset path
    # payload_length: the used bytes of data
    # payload_packet: the used packets 
    # start_index: the index of start byte (the number of byte * 2, i.e., If start from IP header, start_index = 28)
    # 生成json文件
    generation_multiP(pcap_path, [10000]*class_num, json_path, 64, 5, 28)
    
    print("Process", json_path)
    print("Save to", tsv_path)

    dataset_extract(json_path,
                     tsv_path,
                     features=['datagram'],
                     category=class_num,)

for type in os.listdir(save_path):
    if type == 'test.tsv':
        continue
    print("Process", type)
    enhance_based_tsv(f"{save_path}/{type}/", 'train.tsv', "train_enhance.tsv", 5)