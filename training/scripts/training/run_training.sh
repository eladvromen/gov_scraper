#!/bin/bash

# Set up CUDA memory management for A100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Clear any existing CUDA cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Set up logging
LOG_DIR="models/llama3_8b_pre_brexit"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training.log"

# Function to handle errors
handle_error() {
    echo "Error occurred at $(date)" >> $LOG_FILE
    echo "Error: $1" >> $LOG_FILE
    echo "Attempting to save checkpoint..." >> $LOG_FILE
    # The training script should handle this automatically
    exit 1
}

# Set up error handling
trap 'handle_error "Training interrupted"' SIGINT SIGTERM

# Function to check memory
check_memory() {
    free -h >> $LOG_FILE
    nvidia-smi >> $LOG_FILE
}

# Start training with safety measures
echo "Starting training at $(date)" | tee -a $LOG_FILE
echo "System info:" | tee -a $LOG_FILE
check_memory | tee -a $LOG_FILE

# Run the training script with error handling
python training/scripts/training/train.py training/configs/pre_brexit_2013_2016_config.yaml 2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a $LOG_FILE
else
    echo "Training failed at $(date)" | tee -a $LOG_FILE
    handle_error "Training script exited with error"
fi 