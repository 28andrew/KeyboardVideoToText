#!/bin/sh
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=keyboard-string-detection
export WANDB_RUN_NAME=qwen2.5-vl-3b-lora
#export PYTHONWARNINGS="ignore::FutureWarning:librosa.core.audio"
llamafactory-cli train train_keyboard.yml