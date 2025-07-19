import sys

sys.path.append("../")

from utils_1 import (
    export_k_fold_dataset,
    kfold_split_dataframe,
    weighted_train_val_split,
    create_dataset_from_pcap,
    undersample_to_min_class,
    undersample_class
)


def pre_process_data(
    dataset_path="./ISCXVPN2016/Application",
    output_dir="./ISCXVPN2016/Application/train_val_test_split",
    training_partition="train",
    validation_partition="val",
    testing_partition="test",
    seed=43,
    n_splits=3,
    test_size=0.25,
):
    # 1. From pcaps files to pandas dataframe
    print("Creating single pandas dataset from pcap")
    combined_dataset = create_dataset_from_pcap(dataset_path)
   
    # 2. Extract training and test sets
    #   N.b. We want that samples of the same rule don't go in different partitions
    #   At the same time, let's spreading common and rare rules across all folds
    print(
        f"Get train and test sets with % ratio of {(1-test_size)*100:.0f}vs{(test_size)*100:.0f}"
    )
    print("\tNotice: train and test will contain disjoint flows")
    train_df, test_df = weighted_train_val_split(
        combined_dataset,
        test_size=test_size,
        stratify_col="flow",
        random_state=seed,
        weight_col="class_num",
    )
    train_df = train_df.reset_index(drop=True)
    
    # 3. Train dataFrame undersampled to the minority class
    print("Undersampling with maximum number of samples per class...")
    train_df = undersample_class(train_df, max_elem=700000, class_column="class_num", random_state=seed)
    test_df = test_df.sample(frac=0.5, random_state=seed)
    # 3. Now, obtain train and validation following a K-Fold strategy
    #   N.b. We want that samples of the same rule don't go in different partitions
    #   At the same time, let's spreading common and rare rules across all folds
    print(f"Splitting the training dataset into {n_splits} folds...")
    folds = kfold_split_dataframe(
        df=train_df,
        stratify_col="flow",
        weight_col=None,
        n_splits=n_splits,
        random_state=seed,
    )
    
    # 4. Saving partitions
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"Saving fold {fold_idx}...", end="\r")
        final_train_df = train_df.iloc[train_indices].copy()
        final_valid_df = train_df.iloc[val_indices].copy()
        # 4a. Save the output dataframes
        final_train_df = undersample_to_min_class(final_train_df, max_elem=5000, class_column="class_num", random_state=seed)
        export_k_fold_dataset(
            df=final_train_df,
            output_dir=output_dir,
            fold_idx=fold_idx,
            type_df=training_partition,
        )
        final_valid_df = undersample_to_min_class(final_valid_df, max_elem=1000, class_column="class_num", random_state=seed)
        export_k_fold_dataset(
            df=final_valid_df,
            output_dir=output_dir,
            fold_idx=fold_idx,
            type_df=validation_partition,
        )
    export_k_fold_dataset(
        df=test_df,
        output_dir=output_dir,
        fold_idx=None,
        type_df=testing_partition,
    )


if __name__ == "__main__":
    pre_process_data()
