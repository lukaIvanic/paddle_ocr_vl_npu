#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate direct token-ID reference output with vLLM-Ascend."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt_token_ids = tokenizer(
        args.prompt, return_tensors="pt", add_special_tokens=True
    )["input_ids"][0].tolist()
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )
    request = {"prompt_token_ids": prompt_token_ids}
    outputs = llm.generate([request], sampling, use_tqdm=False)
    generated_token_ids = list(outputs[0].outputs[0].token_ids)
    payload = {
        "model": "Qwen/Qwen3-30B-A3B",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("QWEN3_MOE_VLLM_REFERENCE " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
