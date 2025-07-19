import os
import binascii
from PIL import Image
import scapy.all as scapy
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import argparse

def clean_packet(packet):
    if packet.haslayer(scapy.Ether):
        packet = packet[scapy.Ether].payload

    if packet.haslayer(scapy.IP):
        packet[scapy.IP].src = "0.0.0.0"
        packet[scapy.IP].dst = "0.0.0.0"
    elif packet.haslayer('IPv6'):
        packet['IPv6'].src = "::"
        packet['IPv6'].dst = "::"
    else:
        return 0

    if packet.haslayer(scapy.UDP):
        packet[scapy.UDP].sport = 0  # 设置源端口为0
        packet[scapy.UDP].dport = 0  # 设置目的端口为0
    elif packet.haslayer(scapy.TCP):
        packet[scapy.TCP].sport = 0  # 设置源端口为0
        packet[scapy.TCP].dport = 0  # 设置目的端口为0
    
    return packet

def get_header_payload(packet):
    packet = clean_packet(packet)
    if packet.haslayer(scapy.IP):
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
    elif packet.haslayer('IPv6'):
        header = (binascii.hexlify(bytes(packet['IPv6']))).decode()

    if packet.haslayer('Raw'):
        payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
        header = header.replace(payload, '')
    else:
        payload = ''
    
    return header, payload

def read_MFR_bytes(file_name):
    data = []

    with scapy.PcapReader(file_name) as packets:
        for packet in packets:
            header, payload = get_header_payload(packet)

            if len(header) > 160:
                header = header[:160]
            elif len(header) < 160:
                header += '0' * (160 - len(header))
            if len(payload) > 480:
                payload = payload[:480]
            elif len(payload) < 480:
                payload += '0' * (480 - len(payload))
            data.append((header, payload))

            if len(data) >= 5:
                break

    if len(data) < 5:
        for i in range(5-len(data)):
            data.append(('0'*160, '0'*480))

    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
        
    return final_data

def save_image(file_name, path):
    content = read_MFR_bytes(file_name)
    content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
    fh = np.reshape(content, (40, 40))
    fh = np.uint8(fh)
    im = Image.fromarray(fh)
    im.save(path)

def save_flow(id, file_name, class_name, output_path):
    os.makedirs(f"{output_path}/{class_name}", exist_ok=True)
    save_image(file_name, f"{output_path}/{class_name}/{id}.png")

def main(dataset_path, output_path):
# file/test/class/flow.pcap or file/train_val_split_0/{train/val}/class/flow.pcap
    for type in os.listdir(dataset_path):
        print(f'Processing {type}')

        if type == "test":
            for class_id, class_name in enumerate(os.listdir(f'{dataset_path}/{type}')):
                print(f'Processing {type} {class_name}')

                with Pool(cpu_count()) as pool:
                    tasks = [(id, f"{dataset_path}/{type}/{class_name}/{file_name}", class_name, f"{output_path}/{type}") 
                             for id, file_name in enumerate(os.listdir(f'{dataset_path}/{type}/{class_name}'))]
                    pool.starmap(save_flow, tasks)
        else:
            for _, folder in enumerate(os.listdir(f'{dataset_path}/{type}')):
                print(f'Processing {type} {folder}')

                for class_name in os.listdir(f'{dataset_path}/{type}/{folder}'):
                    print(f'Processing {type} {folder} {class_name}')

                    with Pool(cpu_count()) as pool:
                        tasks = [(id, f"{dataset_path}/{type}/{folder}/{class_name}/{file_name}", class_name, f"{output_path}/{type}/{folder}") 
                                for id, file_name in enumerate(os.listdir(f'{dataset_path}/{type}/{folder}/{class_name}'))]
                        pool.starmap(save_flow, tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pcap files and save as images.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
   
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path

    main(dataset_path, output_path)