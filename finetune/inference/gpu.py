import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


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
        # model_path = snapshot_download("openMind-ecosystem/Yi-1.5-9b-chat", revision="main", resume_download=True,
                                    #    ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
        model_path="/tmp/pretrainmodel/Yi-1.5-9B-Chat"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    messages = [
        {"role": "user", "content": "hi"}
    ]
    prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA  # If GPU is being used
    ],
    profile_memory=True,  # Enable memory profiling
    record_shapes=True,   # Enable input shape recording
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof")  # Save to TensorBoard
)


    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    prof.start()
    output_ids = model.generate(input_ids.to(model.device))
    prof.stop()
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Model response: "Hello! How can I assist you today?"
    print(response)


if __name__ == "__main__":
    main()
 