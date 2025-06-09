#!/bin/bash

# Fish Classification Multi-GPU Training Script
# Enhanced for better distributed training support

set -e  # Exit on any error

# --- Configuration ----
N_GPUS=8             # Number of GPUs to use (0 for CPU, 1 for single GPU, >1 for multi-GPU)
TRAIN_SCRIPT="train.py"

# Data and Model Paths
TRAIN_DIR="data/fish-cl-dataset/train/"
VAL_DIR="data/fish-cl-dataset/val/"
SAVE_PATH="models/fish_classifier_ddp.pth"

# Training Hyperparameters
BATCH_SIZE=64          # Per GPU batch size (reduced from 32 to help with memory)
EPOCHS=20
LR=1e-4                 # Base learning rate (will be scaled by world_size in distributed training)
WEIGHT_DECAY=1e-6
WARMUP_EPOCHS=5         # Number of warmup epochs
GRADIENT_CLIP=1.0       # Gradient clipping value
BACKBONE="resnet50"     # Backbone architecture

# Data Loading & Augmentation
IMAGE_SIZE=224
NUM_WORKERS=4           # Number of data loader workers per process
USE_CLASS_PAIRS=True    # Set to False to use standard BYOL with augmentations

# Profiling and Debugging
PROFILE_BATCHES=0       # Number of batches to profile (0 to disable)
                        # Note: Profiling only works on rank 0 and first epoch

# Advanced Training Options
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7,8,9"  # Use GPUs with available memory instead of 0,1,2,3
export NCCL_DEBUG=WARN                  # NCCL debug level (INFO, WARN, ERROR)
export TORCH_DISTRIBUTED_DEBUG=INFO    # PyTorch distributed debug level

#----------------------

# Validate configuration
if [ ! -d "$TRAIN_DIR" ]; then
    echo "Error: Training directory does not exist: $TRAIN_DIR"
    exit 1
fi

if [ ! -z "$VAL_DIR" ] && [ ! -d "$VAL_DIR" ]; then
    echo "Warning: Validation directory does not exist: $VAL_DIR"
    echo "Training will proceed without validation"
    VAL_DIR=""
fi

# Create output directory
mkdir -p "$(dirname "$SAVE_PATH")"
mkdir -p "profiler_traces"

# Convert USE_CLASS_PAIRS to lowercase for python script argument
USE_CLASS_PAIRS_ARG=$(echo "$USE_CLASS_PAIRS" | tr '[:upper:]' '[:lower:]')

# Prepare common arguments
COMMON_ARGS=(
    "--train_dir" "$TRAIN_DIR"
    "--batch_size" "$BATCH_SIZE"
    "--epochs" "$EPOCHS"
    "--lr" "$LR"
    "--weight_decay" "$WEIGHT_DECAY"
    "--warmup_epochs" "$WARMUP_EPOCHS"
    "--gradient_clip" "$GRADIENT_CLIP"
    "--backbone" "$BACKBONE"
    "--image_size" "$IMAGE_SIZE"
    "--save_path" "$SAVE_PATH"
    "--num_workers" "$NUM_WORKERS"
    "--use_class_pairs" "$USE_CLASS_PAIRS_ARG"
    "--profile_batches" "$PROFILE_BATCHES"
)

# Add validation directory if provided
if [ ! -z "$VAL_DIR" ]; then
    COMMON_ARGS+=("--val_dir" "$VAL_DIR")
fi

# Display configuration
echo "========================================"
echo "Fish Classification Training Configuration"
echo "========================================"
echo "GPUs: $N_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total effective batch size: $((BATCH_SIZE * N_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Backbone: $BACKBONE"
echo "Use class pairs: $USE_CLASS_PAIRS"
echo "Training data: $TRAIN_DIR"
echo "Validation data: ${VAL_DIR:-"None"}"
echo "Model save path: $SAVE_PATH"
echo "========================================"

# Training execution based on number of GPUs
if [ $N_GPUS -le 0 ]; then
    echo "Running on CPU..."
    python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}"
    
elif [ $N_GPUS -eq 1 ]; then
    echo "Running on single GPU..."
    echo "Expected batches per epoch: Depends on dataset size / $BATCH_SIZE"
    CUDA_VISIBLE_DEVICES=0 python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}"
    
else
    echo "Running distributed training on $N_GPUS GPUs using torchrun..."
    echo "Expected batches per GPU per epoch: (dataset_size / $N_GPUS) / $BATCH_SIZE"
    echo "Total batches per epoch across all GPUs: dataset_size / $BATCH_SIZE"
    
    # Check if torchrun is available
    if ! command -v torchrun &> /dev/null; then
        echo "Error: torchrun not found. Please upgrade to PyTorch 1.10+ or use:"
        echo "pip install torch>=1.10.0"
        exit 1
    fi
    
    # Use torchrun for distributed training (recommended for PyTorch 1.10+)
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$N_GPUS \
        "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}"
    
    # Alternative: Use the older torch.distributed.launch (deprecated but still works)
    # python -m torch.distributed.launch \
    #     --nproc_per_node=$N_GPUS \
    #     --use_env \
    #     "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}"
fi

echo "========================================"
echo "Training script finished."
echo "Model saved to: $SAVE_PATH"
echo "========================================"
