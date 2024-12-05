import argparse

from openmind import AutoTokenizer, AutoModelForCausalLM
from openmind_hub import snapshot_download
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default=None,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name_or_path:
        model_path = args.model_name_or_path
    else:
        model_path = snapshot_download("openMind-ecosystem/Yi-1.5-9b-chat", revision="main", resume_download=True,
                                       ignore_patterns=["*.h5", "*.ot", "*.msgpack"])

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    messages = [
        {"role": "user", "content": "hi"}
    ]
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False)
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            "./prof"
            ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config)

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    prof.start()
    output_ids = model.generate(input_ids.to(model.device))
    prof.stop()
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Model response: "Hello! How can I assist you today?"
    print(response)


if __name__ == "__main__":
    main()
 