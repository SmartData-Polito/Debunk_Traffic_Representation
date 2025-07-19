#!/bin/bash

set -e
set +x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

input_folder="$1"
output_folder="$2"

mkdir -p "$output_folder"

output_folder="${output_folder%/}"  # 去掉尾部的 `/` 防止双斜杠
find "$input_folder" -type f | parallel "mkdir -p '$output_folder/{/.}' && /home/dauin_user/yzhao/debunk_representation/code/netFound/src/pre_process/runPcap/bin/PcapSplitter -f {} -o '$output_folder/{/.}/' -m connection"
