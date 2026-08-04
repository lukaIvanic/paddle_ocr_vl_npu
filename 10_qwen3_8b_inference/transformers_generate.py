#!/usr/bin/env python3

import argparse

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Transformers Qwen3-8B generation reference.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--prompt", default="Write a tiny Python function that adds two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name_or_path = args.model_dir or args.model_id
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    if args.device == "auto":
        device = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
    model = model.to(device)

    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = args.prompt

    inputs = tokenizer(text, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, prompt_len:]
    print(tokenizer.decode(new_tokens[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
