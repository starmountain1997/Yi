import argparse
import torch
import time
from loguru import logger

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
        model_path = snapshot_download("openMind-ecosystem/codegeex4-all-9b", revision="main", resume_download=True,
                                       ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    )
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": "write a quick sort"}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True ).to(model.device)
    start_time=time.time()
    for i in range(10):
        outputs = model.generate(**inputs, max_length=256)
        logger.info(i)
    end_time=time.time()
    logger.info(f"{end_time-start_time}")

    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
 