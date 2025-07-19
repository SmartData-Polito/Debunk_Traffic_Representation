from scapy.all import *
import pandas as pd

# ------------------------------------------------------------------------------
# GLOBAL VARIABLE DEFINITION
# ------------------------------------------------------------------------------
NAME_FILE_IN = "Train_for_denoiser_450K.pcap"
NAME_FILE_OUT = "Train_for_denoiser_450K"

MAX_NUM_QUESTIONS = 450000

PKT_FORMAT = "every4"  # [every4, every2, noSpace]
PAYLOAD = False
# ------------------------------------------------------------------------------


def read_pcap_header(input_path):
    """
    read_pcap_header
    --------------------
    This function read a PCAP file and process packet by packet.
    Args:
        - input_path: .pcap file

    Output:
        - list of hexadecimal strings of the packets
    """
    try:
        pr = PcapReader(input_path)
    except:
        print("file ", input_path, "  error")
        exit(-1)
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
            header_hex = raw_bytes.hex()
            string = ""
            for i in range(0, len(header_hex), 2):
                string = string + " " + header_hex[i : i + 2]
            final_hex = string.strip()
            list_dict_hex.append(final_hex)
            j += 1
        except EOFError:
            break
    return list_dict_hex


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


def create_list_questions():
    list_questions = []
    with open("./sub/questions_txt/questionsDenoiser.txt", "r") as f:
        for line in f:
            list_questions.append(line)
    return list_questions


def main():
    list_hex = read_pcap_header(f"./sub/pcap_files/{NAME_FILE_IN}")
    quest_list = create_list_questions()
    question = []
    context = []
    final_df = pd.DataFrame()
    z = 0
    for i in range(len(list_hex)):
        index = random.randint(0, len(quest_list) - 1)
        pkt = list_hex[i].replace(" ", "")
        if PKT_FORMAT == "every4":
            pkt = "".join(
                [str(pkt[i : i + 4]) + " " for i in range(0, len(pkt), 4)]
            ).strip()
        elif PKT_FORMAT == "every2":
            pkt = "".join(
                [str(pkt[i : i + 2]) + " " for i in range(0, len(pkt), 2)]
            ).strip()
        question.append(quest_list[index])
        context.append(pkt)
        print("Created row N°:", z, end="\r")
        z += 1
        if z > MAX_NUM_QUESTIONS:
            break
    final_df["question"] = question
    final_df["context"] = context

    final_df.to_parquet(f"../1.Datasets/Denoiser/{NAME_FILE_OUT}.parquet")


main()
