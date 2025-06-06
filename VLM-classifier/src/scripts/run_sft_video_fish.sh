cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

CUDA_VISIBLE_DEVICES=0 python src/open_r1/sft_video.py \
    --output_dir "outputs/log/Qwen2.5-VL-2B-Video-2B-cot-sft" \
    --model_name_or_path "/home/yanglei/QWen/Qwen2-VL-2B-Instruct" \
    --dataset_name "fish-mllms-dataset/fish-cot-100k.json" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    # --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-2B-Video-fish-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
