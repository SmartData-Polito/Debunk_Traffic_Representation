from scapy.all import *
import pandas as pd
import random as rnd

# ------------------------------------------------------------------------------
# GLOBAL VARIABLE DEFINITION
# ------------------------------------------------------------------------------
NAME_FILE_IN = "mix_protocols_19K.pcap"
NAME_FILE_OUT = "Test_RetrmoreProt_noPayload"

MAX_NUM_QUESTIONS = 20000
RND_NUMBER_Q4PKT = 3  # [0, 10] --> Defines the mean number of questions
# generated for each packet
PKT_FORMAT = "every4"  # [every4, every2, noSpace]
PAYLOAD = False
RETRIEVAL_ONLY = True
# ------------------------------------------------------------------------------


def read_pcap_header(input_path):
    """
    read_pcap_header_hex
    --------------------
    This function takes a PCAP file and saves:
    - The bytes of the packet two by two on a json file
    - The values of the different fields on another json file
    Args:
        - input_path: .pcap file
        - output_path1: .json file
        - output_path2: .json file
    """
    try:
        pr = PcapReader(input_path)
    except:
        print("file ", input_path, "  error")
        exit(-1)
    list_dict_values = []
    list_dict_hex = []
    j = 0
    while j < MAX_NUM_QUESTIONS:
        try:
            print("Processed packet N°:", j, end="\r")

            pkt = pr.read_packet()
            if not PAYLOAD:
                pkt = remove_payload(pkt)
            if IPv6 in pkt:
                pkt = pkt["IPv6"]
                pkt = modify_IPv6packets(pkt)
                raw_bytes = bytes(pkt["IPv6"])
            elif IP in pkt:
                pkt = pkt["IP"]
                pkt = modify_IPv4packets(pkt)
                raw_bytes = bytes(pkt["IP"])
            dict_pkt = pkt2dict(pkt)
            try:
                if not RETRIEVAL_ONLY:
                    dict_pkt, pkt = check_checksum(dict_pkt, pkt)
                dict_pkt = convert_hexadecimal(dict_pkt, pkt)
            except:
                continue

            header_hex = raw_bytes.hex()
            string = ""
            for i in range(0, len(header_hex), 2):
                string = string + " " + header_hex[i : i + 2]
            final_hex = string.strip()
            if PAYLOAD and not RETRIEVAL_ONLY:
                success, dict_pkt["last_header3L_byte"], dict_pkt["len_payload"] = (
                    compute_byte_payload(dict_pkt, final_hex, pkt)
                )
            if not PAYLOAD or RETRIEVAL_ONLY or success:
                list_dict_values.append(dict_normalize(dict_pkt))
                list_dict_hex.append(final_hex)
                j += 1
        except EOFError:
            break
    return list_dict_values, list_dict_hex


def dict_normalize(dictionary):
    keys = dictionary.keys()
    if "IPv6" in keys:
        dictionary["IP"] = dictionary.pop("IPv6")
    if "ICMP" in keys:
        dictionary["3L"] = dictionary.pop("ICMP")
    if "TCP" in keys:
        dictionary["3L"] = dictionary.pop("TCP")
    if "UDP" in keys:
        dictionary["3L"] = dictionary.pop("UDP")
    return dictionary


def remove_payload(pkt):
    """
    remove_payload
    --------------
    If needed, the function removes the payload of protocols TCP and UDP. For
    ICMP is not done becasue not crypted.
    """
    if TCP in pkt:
        pkt[TCP].remove_payload()
    elif UDP in pkt:
        pkt[UDP].remove_payload()
    elif ICMP in pkt:
        pkt = pkt
    return pkt


def compute_byte_payload(dict_pkt, hex_string, pkt):
    """
    compute_byte_payload
    --------------------
    The function computes the length and the last header byte of the packet.
    Args
        - dict_pkt: a dictionary with the fields of the packet
        - hex_string: the hexadecimal string of the packet
        - pkt: packet (scapy object)
    Output
    Length and the last header byte of the packet, and if the computation
    succeded.
    """
    hex_list = hex_string.split(" ")
    if TCP in pkt:
        if IPv6 in pkt:
            header_len = 40 + int(dict_pkt["TCP"]["dataofs"]) * 4
        else:
            header_len = (
                int(dict_pkt["IP"]["ihl"]) * 4 + int(dict_pkt["TCP"]["dataofs"]) * 4
            )
        return 1, hex_list[header_len - 1], len(hex_list[header_len:])

    elif UDP in pkt:
        payload_len = int(dict_pkt["UDP"]["len"]) - 8
        return 1, hex_list[-(payload_len + 1)], payload_len

    elif ICMP in pkt:
        if IPv6 in pkt:
            header_len = 40 + int(dict_pkt["TCP"]["dataofs"]) * 4 + 4
        else:
            header_len = int(dict_pkt["IP"]["ihl"]) * 4 + 4
        return 1, hex_list[header_len - 1], len(hex_list[header_len:])
    else:
        breakpoint()
        return 0


def modify_IPv4packets(pkt):
    pkt["IP"].src = generate_rnd_IP()
    pkt["IP"].dst = generate_rnd_IP()
    pkt["IP"].ttl = random.randint(0, 255)
    return pkt


def modify_IPv6packets(pkt):
    pkt["IPv6"].src = generate_rnd_IPv6()
    pkt["IPv6"].dst = generate_rnd_IPv6()
    pkt["IPv6"].ttl = random.randint(0, 255)
    return pkt


def generate_rnd_IPv6():
    return ":".join(
        map(
            str,
            (
                ("{:02x}".format(random.randint(0, 255)))
                + ("{:02x}".format(random.randint(0, 255)))
                for _ in range(8)
            ),
        )
    )


def generate_rnd_IP():
    return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))


def check_checksum(dict_pkt, pkt):
    """
    check_checksum
    --------------
    It control if the IP checksum is correct in the packet. For IPv6 protocol
    the field is not present, so it is not computed.
    Args:
        - dict_pkt: a dictionary with the fields of the packet
        - pkt: packet (scapy object)
    Output
    a dictionary with two fields (IP and TCP) with the results of the comparison
    """
    if IPv6 in pkt:
        pkt["IPv6"]
        dict_pkt["checksum_check"] = "IPv6"
        return dict_pkt, pkt
    elif IP in pkt and pkt["IP"].version == 4:
        ip_pkt = pkt["IP"]
        checksum_real = ip_pkt.chksum
        ip_pkt.chksum = 0x00  # The checksum must to be not considered
        list_header_ip = [int(byte) for byte in raw(ip_pkt)][:20]
        calculated_checksum = checksum(list_header_ip)
        dict_pkt["IP"]["chksum"] = pkt["IP"].chksum = (
            calculated_checksum if rnd.randint(0, 10) >= 5 else checksum_real
        )
        dict_pkt["checksum_check"] = (
            "Correct" if pkt["IP"].chksum == calculated_checksum else "Wrong"
        )
    else:
        dict_pkt["checksum_check"] = "Unknown"
    return dict_pkt, pkt


def convert_hexadecimal(dict_pkt, pkt):
    """
    convert_hexadecimal
    -------------------
    It converts the field translated of the dictionary in input in hexadecimal.
    Args:
        - dict_pkt: a dictionary with the fields of the packet
        - pkt: packet (scapy object)
    Output:
    The dictionary with some fields changed
    """
    # Network level
    if IPv6 in pkt:
        dict_pkt["IPv6"]["ttl"] = "{:02x}".format(pkt[IPv6].hlim)
        dict_pkt["IPv6"]["id"] = "{:02x}".format(pkt[IPv6].fl)
    else:
        dict_pkt["IP"]["src"] = (
            "".join(
                [
                    (
                        str(hex(int(el)))[2:] + "."
                        if len(str(hex(int(el)))[2:]) == 2
                        else "0" + str(hex(int(el)))[2:] + "."
                    )
                    for el in dict_pkt["IP"]["src"].split(".")
                ]
            )
        )[:-1]
        dict_pkt["IP"]["dst"] = (
            "".join(
                [
                    (
                        str(hex(int(el)))[2:] + "."
                        if len(str(hex(int(el)))[2:]) == 2
                        else "0" + str(hex(int(el)))[2:] + "."
                    )
                    for el in dict_pkt["IP"]["dst"].split(".")
                ]
            )
        )[:-1]
        dict_pkt["IP"]["ttl"] = "{:02x}".format(pkt[IP].ttl)
        dict_pkt["IP"]["id"] = "{:02x}".format(pkt[IP].id)

    # Transport level
    if TCP in pkt:
        dict_pkt["TCP"]["ack"] = "{:02x}".format(int(dict_pkt["TCP"]["ack"]))
        dict_pkt["TCP"]["seq"] = "{:02x}".format(int(pkt[TCP].seq))
        dict_pkt["TCP"]["sport"] = "{:02x}".format(int(pkt[TCP].sport))
        dict_pkt["TCP"]["window"] = "{:02x}".format(int(pkt[TCP].window))
        dict_pkt["TCP"]["dport"] = "{:02x}".format(int(pkt[TCP].dport))
        dict_pkt["TCP"]["chksum"] = "{:02x}".format(int(pkt[TCP].chksum))
    elif UDP in pkt:
        dict_pkt["UDP"]["chksum"] = "{:02x}".format(pkt[UDP].chksum)
        dict_pkt["UDP"]["sport"] = "{:02x}".format(int(pkt[UDP].sport))
    elif ICMP in pkt:
        dict_pkt["ICMP"]["chksum"] = "{:02x}".format(pkt[ICMP].chksum)
    return dict_pkt


def pkt2dict(pkt):
    """
    pkt2dict
    -------
    This function converts a packet object into a dictionary.
    Args:
        - pkt: packet (scapy object)
    """
    packet_dict = {}
    for line in pkt.show2(dump=True).split("\n"):
        if "###" in line:
            if "|###" in line:
                sublayer = line.strip("|#[] ")
                packet_dict[layer][sublayer] = {}
            else:
                layer = line.strip("#[] ")
                packet_dict[layer] = {}
        elif "=" in line:
            if "|" in line and "sublayer" in locals():
                key, val = line.strip("| ").split("=", 1)
                packet_dict[layer][sublayer][key.strip()] = val.strip("' ")
            else:
                key, val = line.split("=", 1)
                val = val.strip("' ")
                if val:
                    try:
                        packet_dict[layer][key.strip()] = str(val)
                    except:
                        packet_dict[layer][key.strip()] = val
        else:
            continue
    return packet_dict


def clean_df(df):
    all_elems = [
        "IP.src",
        "IP.dst",
        "IP.ttl",
        "IP.id",
        "3L.ack",
        "3L.window",
        "3L.sport",
        "3L.seq",
        "3L.chksum",
        "len_payload",
        "last_header3L_byte",
        "checksum_check",
    ]
    fields_to_maintain = [field for field in all_elems if field in df.columns]
    df = df[fields_to_maintain]
    df = df.rename(
        columns={
            "IP.src": "srcIP",
            "IP.dst": "dstIP",
            "3L.chksum": "chk3L",
            "IP.id": "IPid",
            "3L.sport": "src3L",
            "3L.ack": "3Lack",
            "3L.seq": "3Lseq",
            "IP.ttl": "IPttl",
            "3L.window": "3Lwnd",
        }
    )
    return df


def create_dictionary_questions():
    q_dictionary = {}
    with open("./sub/questions_txt/questionsQA.txt", "r") as f:
        for line in f:
            line = line.split(",")
            q_dictionary[line[0]] = line[1]
    return q_dictionary


def main():
    rnd.seed(43)
    list_val, list_hex = read_pcap_header(f"./sub/pcap_files/{NAME_FILE_IN}")
    df_values = pd.json_normalize(list_val)
    df_values = clean_df(df_values)
    quest_dict = create_dictionary_questions()
    question = []
    context = []
    answers = []
    type_q = []
    final_df = pd.DataFrame()
    fields = [
        "srcIP",
        "dstIP",
        "chk3L",
        "src3L",
        "IPid",
        "IPttl",
        "3Lwnd",
        "3Lseq",
        "3Lack",
        "last_header3L_byte",
        "len_payload",
        "checksum_check",
    ]
    z = 0
    for i in range(len(list_hex)):
        used_index = []
        count = 0
        while 1:
            index = random.randint(0, len(fields) - 1)
            if (
                fields[index] not in df_values.columns
                or index in used_index
                or pd.isna(df_values[fields[index]].iloc[i])
            ):
                if len(used_index) == len(df_values.columns) or count == 100:
                    break
                count+=1
                continue
            used_index.append(index)
            question.append(quest_dict[fields[index]])
            pkt = list_hex[i].replace(" ", "")
            if PKT_FORMAT == "every4":
                pkt = "".join(
                    [str(pkt[i : i + 4]) + " " for i in range(0, len(pkt), 4)]
                ).strip()
            elif PKT_FORMAT == "every2":
                pkt = "".join(
                    [str(pkt[i : i + 2]) + " " for i in range(0, len(pkt), 2)]
                ).strip()
            context.append(pkt)
            answers.append(f"{df_values[fields[index]].iloc[i]}")
            type_q.append(fields[index])
            print("Created row N°:", z, end="\r")
            z += 1
            # The following to give randomicity
            if random.randint(0, 10) > RND_NUMBER_Q4PKT:
                break
        if z > MAX_NUM_QUESTIONS:
            break
    final_df["question"] = question
    final_df["context"] = context
    final_df["answer"] = answers
    final_df["pkt_field"] = type_q

    final_df.to_parquet(f"../1.Datasets/QA/{NAME_FILE_OUT}.parquet")
    final_df.to_csv(f"../1.Datasets/QA/{NAME_FILE_OUT}.csv")


main()
