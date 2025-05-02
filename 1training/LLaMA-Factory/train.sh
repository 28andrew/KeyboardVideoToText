#!/usr/bin/env bash
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=keyboard-string-detection
export WANDB_RUN_NAME=qwen2.5-vl-3b-lora

CONFIG_FILE="train_keyboard.yml"
CKPT_DIR="saves/keyboard"

while true; do
  if ls ${CKPT_DIR}/checkpoint-* 1> /dev/null 2>&1; then
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* | sort -V | tail -n 1)
    echo "Updating ${CONFIG_FILE} to resume from: ${LATEST_CKPT}"
    yq eval --inplace ".resume_from_checkpoint = \"${LATEST_CKPT}\"" ${CONFIG_FILE}

    echo "Launching training (resuming)..."
    llamafactory-cli train ${CONFIG_FILE}
  else
    echo "No checkpoint found. Starting fresh training run."
    yq eval --inplace "del(.resume_from_checkpoint)" ${CONFIG_FILE}

    llamafactory-cli train ${CONFIG_FILE}
  fi

  echo "Training exited. Sleeping for 30 seconds before retry..."
  sleep 30
done
