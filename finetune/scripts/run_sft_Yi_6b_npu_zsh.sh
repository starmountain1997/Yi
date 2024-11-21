#!/usr/bin/env zsh

if [[ -n "$ZSH_VERSION" ]]; then
	cd "$(dirname "${0:A}")/.."
elif [[ -n "$BASH_VERSION" ]]; then
    cd "$(dirname "$BASH_SOURCE")/.."
else
    exit 1
fi

export PYTHONPATH="$(dirname "${0:A}")/..":$PYTHONPATH

PROFILING_DATA_SAVE_PATH="/mnt/data00/guozr/Yi/prof/fused_OmNpuRMSNorm_AMP"
TRAINING_DEBUG_STEPS=5
PROFILING_DATA_STEPS=1
NUM_EPOCHS=1
# MODEL_NAME="HangZhou_Ascend/Yi-6B"
MODEL_NAME="HangZhou_Ascend/Yi-1.5-9B-chat"
# MODEL_NAME="HangZhou_Ascend/Yi-1.5-9B"

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 deepspeed sft/main_npu.py \
	--data_path ../yi_example_dataset/ \
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
	--output_dir /mnt/data00/guozr/Yi/finetuned_model \
	--profiling false \
	--profiling_data_save_path $PROFILING_DATA_SAVE_PATH

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 deepspeed sft/main_npu.py \
	--data_path ../yi_example_dataset/ \
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
	--output_dir /mnt/data00/guozr/Yi/finetuned_model \
	--profiling true \
	--profiling_data_save_path $PROFILING_DATA_SAVE_PATH 


LATEST_FOLDER=$(ls -1td "$PROFILING_DATA_SAVE_PATH"/*(/) | head -1)
EARLIEST_FOLDER=$(ls -1td "$PROFILING_DATA_SAVE_PATH"/*(/) | tail -1)

COMMAND="msprof-analyze advisor all -d ${LATEST_FOLDER}/ASCEND_PROFILER_OUTPUT"
echo "Executing: $COMMAND"
eval $COMMAND


COMMAND="msprof-analyze compare -d ${LATEST_FOLDER}/ASCEND_PROFILER_OUTPUT -bp ${EARLIEST_FOLDER}/ASCEND_PROFILER_OUTPUT --output_path ${PROFILING_DATA_SAVE_PATH}/compare_result"
echo "Executing: $COMMAND"
eval $COMMAND
