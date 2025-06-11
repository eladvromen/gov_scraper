#!/bin/bash

# Default values
NUM_GPUS=2  # Default to 2 A100s
CONFIG="configs/base_config.yaml"
DATASET_PATTERN=""
OUTPUT_DIR=""
RUN_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset-pattern)
      DATASET_PATTERN="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Build command
CMD="deepspeed --num_gpus $NUM_GPUS"
CMD="$CMD scripts/train.py $CONFIG"

# Add optional arguments if provided
if [ ! -z "$DATASET_PATTERN" ]; then
  CMD="$CMD --dataset-patterns $DATASET_PATTERN"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ ! -z "$RUN_NAME" ]; then
  CMD="$CMD --run-name $RUN_NAME"
fi

# Print command
echo "Running command: $CMD"

# Execute
$CMD 