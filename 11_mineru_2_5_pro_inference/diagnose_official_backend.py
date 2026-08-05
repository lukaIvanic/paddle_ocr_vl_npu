#!/usr/bin/env python3
"""Compare the stock Transformers and vLLM MinerU execution contracts.

Run this script once with each backend-specific environment.  It deliberately
stops after a short generation and records prompt IDs, generated IDs, and the
first-step top log probabilities.  This separates language-model loading from
the longer two-step page parser.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from run_transformers_recognition_smoke import configure_npu, synchronize


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transformers", "vllm"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="\nLayout Detection:")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def messages(prompt: str, has_image: bool) -> list[dict[str, Any]]:
    from mineru_vl_utils.vlm_client.base_client import (
        DEFAULT_SYSTEM_PROMPT,
    )

    result: list[dict[str, Any]] = []
    if DEFAULT_SYSTEM_PROMPT:
        result.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    content: list[dict[str, str]] = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    result.append({"role": "user", "content": content})
    return result


def top_transformers(logits, tokenizer, count: int = 20) -> list[dict[str, Any]]:
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logprobs, k=count)
    return [
        {
            "token_id": int(token_id),
            "logprob": float(value),
            "text": tokenizer.decode([int(token_id)]),
        }
        for value, token_id in zip(values.cpu(), indices.cpu())
    ]


def run_transformers(args: argparse.Namespace, image: Image.Image | None) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    setup_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model = model.to("npu:0").eval()
    setup_s = time.perf_counter() - setup_started

    chat_prompt = processor.apply_chat_template(
        messages(args.prompt, image is not None),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_prompt],
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt",
    ).to(device=model.device, dtype=model.dtype)

    infer_started = time.perf_counter()
    with torch.inference_mode():
        first = model(**inputs, use_cache=False, return_dict=True)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    synchronize()
    infer_s = time.perf_counter() - infer_started
    prompt_ids = inputs.input_ids[0].cpu().tolist()
    output_ids = generated[0, len(prompt_ids) :].cpu().tolist()
    return {
        "backend": "transformers",
        "setup_s": setup_s,
        "inference_s": infer_s,
        "chat_prompt": chat_prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": output_ids,
        "generated_text": processor.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "first_token_top_logprobs": top_transformers(
            first.logits[0, -1], processor.tokenizer
        ),
    }


def serialize_vllm_logprobs(logprobs, tokenizer) -> list[dict[str, Any]]:
    if not logprobs:
        return []
    return [
        {
            "token_id": int(token_id),
            "logprob": float(entry.logprob),
            "text": tokenizer.decode([int(token_id)]),
        }
        for token_id, entry in sorted(
            logprobs.items(), key=lambda item: item[1].logprob, reverse=True
        )
    ]


def run_vllm(args: argparse.Namespace, image: Image.Image | None) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    setup_started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
    )
    setup_s = time.perf_counter() - setup_started
    tokenizer = llm.get_tokenizer()
    chat_prompt = tokenizer.apply_chat_template(
        messages(args.prompt, image is not None),
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_prompt: dict[str, Any] = {"prompt": chat_prompt}
    if image is not None:
        raw_prompt["multi_modal_data"] = {"image": [image]}

    infer_started = time.perf_counter()
    output = llm.generate(
        [raw_prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            logprobs=20,
            skip_special_tokens=False,
        ),
        use_tqdm=False,
    )[0]
    synchronize()
    infer_s = time.perf_counter() - infer_started
    choice = output.outputs[0]
    return {
        "backend": "vllm",
        "setup_s": setup_s,
        "inference_s": infer_s,
        "chat_prompt": chat_prompt,
        "prompt_token_ids": list(output.prompt_token_ids),
        "generated_token_ids": list(choice.token_ids),
        "generated_text": choice.text,
        "finish_reason": choice.finish_reason,
        "first_token_top_logprobs": serialize_vllm_logprobs(
            choice.logprobs[0] if choice.logprobs else None,
            tokenizer,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    configure_npu()
    image = None
    if args.image is not None:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    result = (
        run_transformers(args, image)
        if args.backend == "transformers"
        else run_vllm(args, image)
    )
    result.update(
        {
            "model": str(args.model),
            "image": str(args.image) if args.image is not None else None,
            "prompt": args.prompt,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
