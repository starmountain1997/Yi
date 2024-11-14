#!/usr/bin/env zsh

cd "$(dirname "${0:A}")/../sft/"

export PYTHONPATH="$(dirname "${0:A}")/..":$PYTHONPATH


deepspeed main.py \
	--data_path ../yi_example_dataset/ \
	--model_name_or_path /root/.cache/openmind/hub/models--HangZhou_Ascend--Yi-1.5-9B-Chat/snapshots/f86e5e88ffa308d4630539e991ff16b166907b84 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs 4 \
	--training_debug_steps 20 \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--output_dir ./finetuned_model