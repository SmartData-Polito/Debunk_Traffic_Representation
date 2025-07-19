from scapy.all import rdpcap, wrpcap
import os
from multiprocessing import Pool, cpu_count

# packet-level split
def process_packet(args):
    """
    多进程处理单个数据包
    """
    packet, output_file = args
    try:
        wrpcap(output_file, [packet])  # 将数据包写入单独的文件
        print(f"Saved packet to {output_file}")
    except OSError as e:
        print(f"Error saving packet to {output_file}: {e}")

def split_pcap_to_packets_parallel(input_pcap, output_dir):
    """
    使用多进程将 pcap 文件按数据包拆分，并保存到指定目录
    """
    # 检查输入文件和输出目录
    if not os.path.exists(input_pcap):
        print(f"Input file {input_pcap} does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 pcap 文件
    packets = rdpcap(input_pcap)

    # 准备参数列表 [(packet, output_file), ...]
    args = [
        (packet, os.path.join(output_dir, f"packet_{i+1}.pcap"))
        for i, packet in enumerate(packets)
    ]

    # 使用多进程处理数据包
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_packet, args)

    print(f"Split complete. Total {len(packets)} packets saved to {output_dir}.")

# flow-level split
def process_session(args):
    """
    多进程处理单个会话
    """
    session_packets, output_file = args
    wrpcap(output_file, session_packets)  # 将会话数据包写入单独的文件
    print(f"Saved session to {output_file}")

def split_pcap_to_sessions_parallel(input_pcap, output_dir):
    """
    使用多进程将 pcap 文件按会话拆分，并保存到指定目录
    """
    # 检查输入文件和输出目录
    if not os.path.exists(input_pcap):
        print(f"Input file {input_pcap} does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 pcap 文件
    packets = rdpcap(input_pcap)

    # 获取会话字典 {会话标识: 数据包列表}
    sessions = packets.sessions()

    # 准备参数列表 [(session_packets, output_file), ...]
    args = [
        (session_packets, os.path.join(output_dir, f"session_{i+1}.pcap"))
        for i, session_packets in enumerate(sessions.values())
    ]

    # 使用多进程处理会话
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_session, args)

    print(f"Split complete. Total {len(sessions)} sessions saved to {output_dir}.")

# delete
def size_format(size):
    """
    格式化文件大小，返回以 KB 为单位的大小
    """
    return size / 1024.0

def process_file(args):
    """
    处理单个文件，判断是否需要删除
    """
    file_path, protocol_size_limits = args
    try:
        # 读取 pcap 文件
        current_packet = rdpcap(file_path)
        file_size = size_format(os.path.getsize(file_path))  # 文件大小（KB）

        # 检查协议并判断是否删除
        if 'TCP' in str(current_packet.res):
            if file_size < protocol_size_limits['TCP']:
                os.remove(file_path)
                print(f"Removed TCP sample: {file_path} (size: {file_size:.2f} KB).")
        elif 'UDP' in str(current_packet.res):
            if file_size < protocol_size_limits['UDP']:
                os.remove(file_path)
                print(f"Removed UDP sample: {file_path} (size: {file_size:.2f} KB).")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # 删除读取失败的文件
        os.remove(file_path)
        print(f"Removed unreadable file: {file_path}.")

def process_directory_parallel(directory, protocol_size_limits):
    """
    使用多进程处理目录下的所有文件
    """
    # 获取所有文件路径
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
    ]

    # 准备多进程参数 [(file_path, protocol_size_limits), ...]
    args = [(file_path, protocol_size_limits) for file_path in all_files]

    # 使用多进程处理
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_file, args)

    print(f"Processing complete for directory: {directory}")

def split_pcap_to_packets_with_limited(input_pcap, output_dir, number):
    """
    将 pcap 文件按数据包拆分，并保存到指定目录
    """
    # 检查输入文件和输出目录
    if not os.path.exists(input_pcap):
        print(f"Input file {input_pcap} does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 pcap 文件
    packets = rdpcap(input_pcap)

    # 遍历每个数据包并保存为独立文件
    for i, packet in enumerate(packets):
        if i > number:
            break
        output_file = os.path.join(output_dir, f"packet_{i+1}.pcap")
        wrpcap(output_file, [packet])  # 将每个数据包单独写入文件
        print(f"Saved packet {i+1} to {output_file}")
        
    print(f"Split complete. Total {len(packets)} packets saved to {output_dir}.")

def split_pcap_to_sessions(input_pcap, output_dir, number):
    """
    将 pcap 文件按会话拆分，并保存到指定目录
    """
    # 检查输入文件和输出目录
    if not os.path.exists(input_pcap):
        print(f"Input file {input_pcap} does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 pcap 文件
    packets = rdpcap(input_pcap)

    # 获取会话字典 {会话标识: 数据包列表}
    sessions = packets.sessions()

    # 遍历会话并保存
    for i, (session, session_packets) in enumerate(sessions.items()):
        if i > number:
            break
        output_file = os.path.join(output_dir, f"session_{i+1}.pcap")
        wrpcap(output_file, session_packets)  # 将每个会话的数据包保存为一个文件
        print(f"Saved session {i+1} to {output_file}")

    print(f"Split complete. Total {len(sessions)} sessions saved to {output_dir}.")


