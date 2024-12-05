#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../sft/"

PROFILING_DATA_SAVE_PATH="/tmp/code/LoRA_100"
TRAINING_DEBUG_STEPS=100
PROFILING_DATA_STEPS=10
NUM_EPOCHS=1
# MODEL_NAME="openMind-ecosystem/Yi-6B"
# MODEL_NAME="/tmp/pretrainmodel/Yi-6b"
MODEL_NAME="/tmp/pretrainmodel/Yi-6B"

deepspeed main.py \
	--data_path /tmp/code/Yi/finetune/yi_example_dataset \
	--model_name_or_path $MODEL_NAME \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs $NUM_EPOCHS \
	--training_debug_steps $TRAINING_DEBUG_STEPS \
	--profiling_data_steps $PROFILING_DATA_STEPS \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--lora_dim 128 \
	--lora_module_name "layers." \
	--output_dir /tmp/pretrainmodel/output \
	--profiling_data_save_path $PROFILING_DATA_SAVE_PATH
