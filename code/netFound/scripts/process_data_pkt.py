import os
import json
import scapy.all as scapy
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import argparse

config_json_file = "./configs/TestFinetuningConfig.json"

def write2arrow(data, output_filename):
    with pa.OSFile(output_filename, "wb") as sink:
        with pa.ipc.new_stream(sink, schema=table_schema) as writer:
            batch = pa.record_batch(
                [
                    pa.array(data[0], type=flow_duration_type),
                    pa.array(data[1], type=burst_tokens_type),
                    pa.array(data[2], type=directions_type),
                    pa.array(data[3], type=bytes_type),
                    pa.array(data[4], type=iats_type),
                    pa.array(data[5], type=counts_type),
                    pa.array(data[6], type=protocol_type),
                    pa.array(data[7], type=label_type),
                ],
                schema=table_schema,
            )
            writer.write_batch(batch)

# header length = 5 (real value) * 4 = 20
# seq and ack is relative, so set to 0
def tokenize_fields_df(_config, type_of_field, packet):
    result = {}

    for conf_dict in _config[type_of_field]:
        field = conf_dict["field"]
        numOfTokens = conf_dict["numberTokens"]

        if field == "IP_hl":
            val = packet[scapy.IP].ihl * 4
        elif field == "IP_tos":
            val = packet[scapy.IP].tos
        elif field == "IP_tl":
            val = packet[scapy.IP].len
        elif field == "IP_Flags":
            val = packet[scapy.IP].flags
        elif field == "IP_ttl":
            val = packet[scapy.IP].ttl
        elif field == "TCP_Flags":
            val = int(packet[scapy.TCP].flags)
        elif field == "TCP_wsize":
            val = packet[scapy.TCP].window
        elif field == "TCP_seq":
            val = packet[scapy.TCP].seq
            val = 0
        elif field == "TCP_ackn":
            val = packet[scapy.TCP].ack
            val = 0
        elif field == "TCP_urp":
            val = packet[scapy.TCP].urgptr
        elif field == "UDP_len":
            val = packet[scapy.UDP].len
        elif field == "Payload":
            if packet.haslayer(scapy.Raw):
                val = packet[scapy.Raw].load
                result[field] = val[:numOfTokens * 2].ljust(numOfTokens * 2, b'\x00')
            else:
                result[field] = b'\x00' * (numOfTokens * 2)
            continue
        
        result[field] = int(val).to_bytes(numOfTokens * 2, byteorder="big") 

    return result

def process_pcap_file(config, input_base, output_base, split_folder, class_folder, pcap_file):
    num_of_bursts_per_flow = 12
    num_of_pkts_per_burst = 6
    payloadTokenNum = config["Payload"][0]["numberTokens"]

    directions = [[True] * num_of_bursts_per_flow]
    iat = [[0] * num_of_bursts_per_flow]
    counts = [[num_of_pkts_per_burst] * num_of_bursts_per_flow]
    flow_duration = [0] 

    tokensPerPacket = sum([field["numberTokens"] for field in config["IPFields"]])

    full_path = os.path.join(input_base, split_folder, class_folder, pcap_file)
    try:
        with scapy.PcapReader(full_path) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                packet_token = {}

                if pkt.haslayer(scapy.Ether):
                    pkt = pkt[scapy.Ether].payload

                if pkt.haslayer(scapy.TCP):
                    field_name = 'TCPFields'
                    protocol = [6]
                elif pkt.haslayer(scapy.UDP):
                    field_name = 'UDPFields'
                    protocol = [17]
                else:
                    continue  # skip non-TCP/UDP packets

                if pkt.haslayer(scapy.Raw):
                    pkt[scapy.Raw].load = pkt[scapy.Raw].load[:12]

                bytes_of_each_burst = [[len(pkt) * num_of_pkts_per_burst] * num_of_bursts_per_flow]

                tokensPerPacket += (
                    sum([field["numberTokens"] for field in config[field_name]]) +
                    payloadTokenNum
                )

                packet_token.update(tokenize_fields_df(config, "IPFields", pkt))
                packet_token.update(tokenize_fields_df(config, field_name, pkt))
                packet_token.update(tokenize_fields_df(config, "Payload", pkt))

                all_bytes = b''.join(list(packet_token.values()))
                tokens = [int.from_bytes(all_bytes[j:j+2], byteorder="big") for j in range(0, len(all_bytes), 2)]
                repeated_tokens = tokens * num_of_pkts_per_burst
                burst_tokens = [[repeated_tokens] * num_of_bursts_per_flow]

                label = [class_folder]
                arrow_data = [flow_duration, burst_tokens, directions, bytes_of_each_burst, iat, counts, protocol, label]

                os.makedirs(os.path.join(output_base, split_folder), exist_ok=True)
                output_path = os.path.join(output_base, split_folder, f"{pcap_file[:-5]}_{i}.arrow")
                write2arrow(arrow_data, output_path)
    except Exception as e:
        print(f"Failed to process {full_path}: {e}")

def create_arrow(input_folder, split_folder, output_folder):
    with open(config_json_file, "r") as config_file:
        config = json.load(config_file)

    tasks = []
    for class_folder in os.listdir(os.path.join(input_folder, split_folder)):
        for pcap_file in os.listdir(os.path.join(input_folder, split_folder, class_folder)):
            tasks.append((config, input_folder, output_folder, split_folder, class_folder, pcap_file))

    print(f"Total PCAP files to process: {len(tasks)}")

    # 使用 multiprocessing Pool 并行处理
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_pcap_file, tasks)

if __name__ == "__main__":
    # task = 'tls'
    # exp = 'polishedns'
    # input_folder = f'debunk_representation/code/netFound/data/raw/{exp}/{task}'
    # output_folder = f'debunk_representation/code/netFound/data/final/combined/pkt/{exp}/{task}'

    parser = argparse.ArgumentParser(description="Process PCAP files and convert to Arrow format.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing PCAP files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for Arrow files.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    flow_duration_type = pa.uint64()
    burst_tokens_type = pa.list_(pa.list_(pa.uint16()))
    directions_type = pa.list_(pa.bool_())
    bytes_type = pa.list_(pa.uint32())
    iats_type = pa.list_(pa.uint64())
    counts_type = pa.list_(pa.uint32())
    protocol_type = pa.uint16()
    label_type = pa.string()

    table_schema = pa.schema(
            [
                pa.field("flow_duration", flow_duration_type),
                pa.field("burst_tokens", burst_tokens_type),
                pa.field("directions", directions_type),
                pa.field("bytes", bytes_type),
                pa.field("iats", iats_type),
                pa.field("counts", counts_type),
                pa.field("protocol", protocol_type),
                pa.field("labels", label_type),
            ]
        )
    
    for exp in os.listdir(input_folder): 
        if exp == 'flow':
            continue
        for task in os.listdir(os.path.join(input_folder, exp)): # tls, vpn-app
            for split_folder in os.listdir(os.path.join(input_folder, exp, task)): #test, train_val_split_0
                if split_folder == 'test':
                    create_arrow(os.path.join(input_folder, exp, task), split_folder, os.path.join(output_folder, exp, task))
                else:
                    for type_folder in os.listdir(os.path.join(input_folder, exp, task, split_folder)):
                        create_arrow(os.path.join(input_folder, exp, task), os.path.join(split_folder, type_folder), os.path.join(output_folder, exp, task))


