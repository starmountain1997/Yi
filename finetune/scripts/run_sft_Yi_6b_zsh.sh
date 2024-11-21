#!/usr/bin/env bash

if [[ -n "$ZSH_VERSION" ]]; then
	cd "$(dirname "${0:A}")/.."
elif [[ -n "$BASH_VERSION" ]]; then
    cd "$(dirname "$BASH_SOURCE")/.."
else
    exit 1
fi

export PYTHONPATH="$(dirname "${0:A}")/..":$PYTHONPATH

PROFILING_DATA_SAVE_PATH="/tmp/pretrainmodel/profiling"
TRAINING_DEBUG_STEPS=5
PROFILING_DATA_STEPS=1
NUM_EPOCHS=1
MODEL_NAME="/tmp/pretrainmodel/Yi-1.5-6B"

deepspeed sft/main.py \
	--data_path /tmp/code/01-ai/finetune/yi_example_dataset \
	--model_name_or_path $MODEL_NAME \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs $NUM_EPOCHS \
	--training_debug_steps $TRAINING_DEBUG_STEPS \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--output_dir /tmp/pretrainmodel/output \
	--profiling_data_save_path $PROFILING_DATA_SAVE_PATH
