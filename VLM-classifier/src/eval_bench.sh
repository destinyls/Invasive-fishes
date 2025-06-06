#!/bin/bash
# run_models.sh

model_paths=(
    "/home/yanglei/QWen/Video-R1-7B/"
)

file_names=(
    "pred"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
