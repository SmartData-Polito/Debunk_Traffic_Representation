import os
import hashlib
import numpy as np
import pandas as pd
from typing import Any
from scapy.all import*

def export_k_fold_dataset(df, output_dir, fold_idx, type_df):
    """
    Export a DataFrame to parquet format and save its characteristics in a specific fold directory structure.

    Args:
        df (pandas.DataFrame): The DataFrame to export
        output_dir (str): Base directory where the fold directories will be created
        fold_idx (int): Index of the current fold
        partition (str): Name of the partition (e.g., 'train', 'test', 'val')

    Returns:
        None: Files are saved to disk:
            - A parquet file at {output_dir}/fold_{fold_idx}/{partition}.parquet
            - A pcap file at {output_dir}/fold_{fold_idx}/{partition}.parquet
    """
    if fold_idx is not None:
        folder = os.path.join(output_dir, f"train_val_split_{fold_idx}", type_df)
    else:
        folder = os.path.join(output_dir, type_df)
    os.makedirs(folder, exist_ok=True)
    for partition in df['class_str'].unique():
        partition_df = df[df['class_str'] == partition]
        wrpcap(os.path.join(folder, f"{partition}.pcap"), list(partition_df['pkt_raw']), linktype=101)
    df = df.drop(["pkt_raw"], axis=1)
    df.to_parquet(
        os.path.join(folder, f"{type_df}.parquet"),
        index=False,
    )


def get_flow_key(packet, label):
    """
    Compute the 6-tuple that identifies a packet (IP.src, IP.dst, port.src, port.dst, proto, label).
    The flow is considered as bidirectional, so the exchange of src and destination (IP,port) doesn't have effect.
    
    Args:
        packet (scapy): Packet we are analysing
        label (str): Name of the classification task we are solving
    
    Returns:
        six-dimensions tuple: the identifier of the flow in which the packet belongs
    """
    if 'IP' in packet:
        ip_layer = packet['IP']
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        if 'TCP' in packet:
            src_port = packet['TCP'].sport
            dst_port = packet['TCP'].dport
        elif 'UDP' in packet:
            src_port = packet['UDP'].sport
            dst_port = packet['UDP'].dport
        else:
            return False
        
        # Create a sorted tuple to handle inverted IPs and ports
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        port_pair = tuple(sorted([src_port, dst_port]))
        tuple_of_tuples = (ip_pair, port_pair)
        tmp = tuple(item for subtuple in tuple_of_tuples for item in subtuple)
        return (*tmp, proto, label)
    if 'IPv6' in packet:
        ip_layer = packet['IPv6']
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.nh
        if 'TCP' in packet:
            src_port = packet['TCP'].sport
            dst_port = packet['TCP'].dport
        elif 'UDP' in packet:
            src_port = packet['UDP'].sport
            dst_port = packet['UDP'].dport
        else:
            return False
        
        # Create a sorted tuple to handle inverted IPs and ports
        ip_pair = tuple(sorted([src_ip, dst_ip]))
        port_pair = tuple(sorted([src_port, dst_port]))
        tuple_of_tuples = (ip_pair, port_pair)
        tmp = tuple(item for subtuple in tuple_of_tuples for item in subtuple)
        return (*tmp, proto, label)
    return False
        
        
def create_dataset_from_pcap(input_dir):
    """
    Generates a single pandas DataFrame with the pcap or pcapng files contained in 
    the 'input_dir' directory.
    
    Args:
        input_dir (str): Base directory where the pcaps are loaded
        
    Returns:
        (pandas.DataFrame): The new DataFrame created
    """
    list_pcaps = os.listdir(input_dir)
    num_label=0
    list_hex = []
    pkt_raw = []
    label_app = []
    hash_key = []
    numeric_class = []
    for pcap_file in list_pcaps:
        if ".pcap" in pcap_file or ".pcapng" in pcap_file:
            pr = PcapReader(f"{input_dir}/{pcap_file}")
            #label = pcap_file.replace(".pcapng", "").replace(".pcap","")
            label = pcap_file.split(".")[0]
            while 1:
                try:
                    pkt = pr.read_packet()
                    flow_key = get_flow_key(pkt, label)
                    hashed_key = deterministic_hash(flow_key)
                    try:
                        pkt = pkt[IP]
                    except:
                        pkt = pkt[IPv6]
                    header_hex = bytes(pkt).hex()
                    pkt_string = " ".join(header_hex[j:j+4] for j in range(0, len(header_hex), 4))
                    list_hex.append(pkt_string)
                    pkt_raw.append(pkt)
                    label_app.append(label)
                    hash_key.append(hashed_key)
                    numeric_class.append(num_label)
                except EOFError:
                    break
            print(pcap_file, "Done")
            num_label+=1
    return pd.DataFrame({"context":list_hex, "flow":hash_key,"class_str": label_app, "class_num": numeric_class, "pkt_raw": pkt_raw})

def weighted_train_val_split(
    df, stratify_col, test_size, weight_col=None, random_state=None, max_sample_per_type=1000
):
    np.random.seed(random_state)

    # Get unique groups
    unique_groups = df[stratify_col].unique()

    if weight_col is not None:
        # Compute group weights more efficiently
        group_weights = df.groupby(stratify_col)[weight_col].count().reset_index(name="weight")
        group_weights['weight'] = [weight if weight <= max_sample_per_type else max_sample_per_type for weight in group_weights['weight']]

        # Use more efficient quartile calculation
        try:
            group_weights["weight_quartile"] = pd.qcut(
                group_weights["weight"],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
            )
        except ValueError:
            quartiles = group_weights["weight"].quantile([0.25, 0.5, 0.75])

            def assign_quartile(x):
                if x <= quartiles[0.25]:
                    return "Q1"
                elif x <= quartiles[0.5]:
                    return "Q2"
                elif x <= quartiles[0.75]:
                    return "Q3"
                else:
                    return "Q4"

            group_weights["weight_quartile"] = group_weights["weight"].apply(
                assign_quartile
            )

        # More efficient group selection
        test_groups = []
        print("Quartiles defined ...")
        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            quartile_groups = group_weights[group_weights["weight_quartile"] == quartile]
            quartile_groups = quartile_groups.sample(frac=1, random_state=random_state)
            
            quartile_total_weight = quartile_groups["weight"].sum()
            quartile_target_test_weight = quartile_total_weight * test_size
            
            quartile_test_weight = 0
            for _, row in quartile_groups.iterrows():
                if quartile_test_weight < quartile_target_test_weight:
                    test_groups.append(row[stratify_col])
                    quartile_test_weight += row["weight"]
                else:
                    break

        # Use set for faster lookups
        test_groups_set = set(test_groups)
        train_groups = [g for g in unique_groups if g not in test_groups_set]
    
    else:
        # Simple random split
        n_test = int(len(unique_groups) * test_size)
        np.random.shuffle(unique_groups)
        test_groups = unique_groups[:n_test]
        train_groups = unique_groups[n_test:]

    # More efficient DataFrame creation
    def sample_group(group_name, is_train):
        groups = train_groups if is_train else test_groups
        if group_name in groups:
            group_df = df[df[stratify_col] == group_name].copy()
            return group_df.sample(min(len(group_df), max_sample_per_type))
        return pd.DataFrame()

    # Create train and test DataFrames in one pass
    train_df = pd.concat([sample_group(group, is_train=True) for group in unique_groups], ignore_index=True)
    test_df = pd.concat([sample_group(group, is_train=False) for group in unique_groups], ignore_index=True)

    return train_df, test_df


def kfold_split_dataframe(
    df, stratify_col, n_splits, weight_col=None, random_state=None
):
    """
    Splits a DataFrame into K folds while ensuring samples with the same stratify_col value
    stay in the same fold. If weight_col is specified, distributes groups based on weights.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to split
    stratify_col : str
        Column name to use for grouping samples that should stay together
    n_splits : int
        Number of folds for K-fold split
    weight_col : str, optional
        Column name to use for calculating group weights. If None, just splits groups
        randomly while keeping them together
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    list of tuples
        Each tuple contains (train_indices, test_indices) for one fold
    """
    # Reset index to ensure we're working with consecutive integers
    df = df.reset_index(drop=True)
    # Get unique groups
    unique_groups = df[stratify_col].unique()
    if weight_col is not None:
        # Weight-based strategy
        group_weights = (
            df.groupby(stratify_col)[weight_col].count().reset_index(name="weight")
        )
        group_weights = group_weights.sort_values("weight", ascending=False)
        # Initialize random state
        np.random.seed(random_state)
        # Initialize lists to store groups for each fold
        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        # Distribute groups across folds based on weights
        for _, row in group_weights.iterrows():
            group = row[stratify_col]
            weight = row["weight"]
            # Find the fold with the smallest current weight
            min_fold_idx = np.argmin(fold_sizes)
            fold_groups[min_fold_idx].append(group)
            fold_sizes[min_fold_idx] += weight
    else:
        # Simple random split strategy
        np.random.seed(random_state)
        # Shuffle the groups
        np.random.shuffle(unique_groups)
        # Split groups into roughly equal-sized folds
        fold_groups = np.array_split(unique_groups, n_splits)
        fold_groups = [group.tolist() for group in fold_groups]

    # Initialize list to store final fold indices
    folds = []
    # Convert fold groups to indices
    for i in range(n_splits):
        # Current fold becomes test set
        test_groups = fold_groups[i]
        # All other folds become training set
        train_groups = [
            group for j, groups in enumerate(fold_groups) for group in groups if j != i
        ]
        # Get indices for samples belonging to these groups
        train_indices = df[df[stratify_col].isin(train_groups)].index
        test_indices = df[df[stratify_col].isin(test_groups)].index

        folds.append((train_indices, test_indices))
    return folds

def deterministic_hash(value: Any) -> str:
    """
    Creates a deterministic hash of the input value that remains consistent across different Python sessions.
    Args:
        value (Any): The value to hash. Will be converted to string before hashing.
    Returns:
        str: A hexadecimal string representation of the MD5 hash.
    Examples:
        >>> deterministic_hash("Hello World")
        'b10a8db164e0754105b7a99be72e3fe5'
        >>> deterministic_hash({"a": 1, "b": 2})
        '608de49a4600dbb5b173492759792e4a'
    Notes:
        - Unlike Python's built-in hash() function, this will give the same result across different Python sessions
        - Uses MD5 for speed and consistency, not for cryptographic security
        - If you need cryptographic security, use SHA-256 or another secure hash function
        - The function converts input to string representation before hashing
    """
    # Convert input to string to handle different types
    str_value = str(value)
    # Create MD5 hash of the string value
    hash_object = hashlib.md5(str_value.encode())
    # Return hexadecimal representation of hash
    return hash_object.hexdigest()

def undersample_class(df, max_elem, class_column, random_state):
    df_undersampled = df.groupby(class_column).apply(
        lambda x: x.sample(
            min(len(x), max_elem), 
            random_state=random_state,
            # Additional sampling parameters if needed
            # replace=False,  # Ensure no replacement
            # weights=None    # Optional weighting
        )
    ).reset_index(drop=True)
    return df_undersampled

def undersample_to_min_class(df, max_elem, class_column, random_state):
    # Find the minimum class count
    min_class_count = df[class_column].value_counts().min()
    if max_elem < min_class_count:
        min_class_count = max_elem
    # Group by the class column and sample each group to the minimum class count
    df_undersampled = df.groupby(class_column).apply(lambda x: x.sample(min_class_count,random_state=random_state)).reset_index(drop=True)
    return df_undersampled
