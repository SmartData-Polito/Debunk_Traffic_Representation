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

def read_MFR_bytes(packet, repeat):
    data = []
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
        
    if len(header) > 160:
        header = header[:160]
    elif len(header) < 160:
        header += '0' * (160 - len(header))
    if len(payload) > 480:
        payload = payload[:480]
    elif len(payload) < 480:
        payload += '0' * (480 - len(payload))
    data.append((header, payload))

    if not repeat:
        if len(data) < 5:
            for i in range(5-len(data)):
                data.append(('0'*160, '0'*480))
    elif repeat:
        while len(data) < 5:
            data.append((header, payload))

    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
        
    return final_data

def save_image(packet, path, repeat):
    content = read_MFR_bytes(packet, repeat)
    content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
    fh = np.reshape(content, (40, 40))
    fh = np.uint8(fh)
    im = Image.fromarray(fh)
    im.save(path)

def save_packet(id, packet, class_str, output_path, repeat):
    os.makedirs(f"{output_path}/{class_str}", exist_ok=True)
    save_image(packet, f"{output_path}/{class_str}/{id}.png", repeat)


def main(dataset_path, output_path, repeat):
# file/test/{test.pcap} or file/train_val_split_0/{train/val}/.pcap

    for type in os.listdir(dataset_path):
        print(f'Processing {type}')

        if 'test' == type:
            for class_id, file_name in enumerate(os.listdir(f'{dataset_path}/{type}')):
                if 'pcap' in file_name:
                    print(f'Processing {type} {file_name}')

                    packets =  scapy.PcapReader(f'{dataset_path}/{type}/{file_name}')
                    with Pool(cpu_count()) as pool:
                        tasks = [(id, packet, file_name[:-5], f"{output_path}/{type}", repeat) for id, packet in enumerate(packets)]
                        pool.starmap(save_packet, tasks)
        else:
            for class_id, folder in enumerate(os.listdir(f'{dataset_path}/{type}')):
                print(f'Processing {type} {folder}')

                for file_name in os.listdir(f'{dataset_path}/{type}/{folder}'):
                    if 'pcap' in file_name:
                        packets =  scapy.PcapReader(f'{dataset_path}/{type}/{folder}/{file_name}')

                        with Pool(cpu_count()) as pool:
                            tasks = [(id, packet, file_name[:-5], f"{output_path}/{type}/{folder}", repeat) for id, packet in enumerate(packets)]
                            pool.starmap(save_packet, tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pcap files and save as images.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--repeat', action="store_true",  help='If repeat the pkt process')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    repeat = args.repeat

    main(dataset_path, output_path, repeat)