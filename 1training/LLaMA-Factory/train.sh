#!/usr/bin/env bash
# Environment setup
ulimit -n 65535

export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=keyboard-string-detection32-2
export WANDB_RUN_NAME=qwen2.5-vl-3b-lora32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Path to the config file
CONFIG_FILE="train_keyboard32.yml"
# Directory where checkpoints are saved
CKPT_DIR="saves/keyboard32-2"

# Infinite loop to resume training automatically
while true; do
  # Find the latest checkpoint (highest number)
  if ls ${CKPT_DIR}/checkpoint-* 1> /dev/null 2>&1; then
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* | sort -V | tail -n 1)
    echo "Updating ${CONFIG_FILE} to resume from: ${LATEST_CKPT}"
    # Inject into YAML in-place:
    yq eval --inplace ".resume_from_checkpoint = \"${LATEST_CKPT}\"" ${CONFIG_FILE}

    echo "Launching training (resuming)..."
    llamafactory-cli train ${CONFIG_FILE}
  else
    echo "No checkpoint found. Starting fresh training run."
    # If you want to clear any prior resume setting:
    yq eval --inplace "del(.resume_from_checkpoint)" ${CONFIG_FILE}

    llamafactory-cli train ${CONFIG_FILE}
  fi

  # kill any stray Python3 from previous run
  killall -9 python3

  echo "Training exited. Sleeping for 30 seconds before retry..."
  sleep 30
done
