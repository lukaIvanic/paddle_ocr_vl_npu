#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message=r"The following torchair config or properties may not take effect.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated.*",
    category=UserWarning,
)

import torch

from run_local_qwen3_reranker import LocalQwen3RerankerRunner
from transformers_rerank import DEFAULT_TASK, build_inputs, format_instruction


class StageLogger:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.last = self.started
        self.count = 0
        if self.enabled:
            print(f"[start] {datetime.now().isoformat(timespec='seconds')}", flush=True)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self.count += 1
        print(
            f"[stage {self.count:02d} +{now - self.last:.3f}s total={now - self.started:.3f}s]\t{message}",
            flush=True,
        )
        self.last = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM-shaped Qwen3 reranker benchmark.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--random-input-len", type=int, default=200)
    parser.add_argument("--random-batch-size", type=int, default=1)
    parser.add_argument("--forward-batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-forward", action="store_true")
    parser.add_argument("--attention-impl", choices=("eager", "prompt_flash_attention"), default="eager")
    parser.add_argument("--ffn-weight-mode", choices=("dense", "w8a8", "all_w8a8"), default="dense")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--json-out")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def time_call(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start


def synthetic_text(tokenizer, rng: random.Random, *, token_count: int) -> str:
    high = min(int(tokenizer.vocab_size) - 16, 32000)
    token_ids = [rng.randint(1000, high) for _ in range(token_count)]
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def generate_pairs(tokenizer, *, count: int, random_input_len: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    query_len = max(8, random_input_len // 2)
    doc_len = max(8, random_input_len)
    return [
        (
            synthetic_text(tokenizer, rng, token_count=query_len),
            synthetic_text(tokenizer, rng, token_count=doc_len),
        )
        for _ in range(count)
    ]


def build_batch(
    runner: LocalQwen3RerankerRunner,
    pairs: list[tuple[str, str]],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    formatted = [format_instruction(DEFAULT_TASK, query, document) for query, document in pairs]
    encoded = build_inputs(runner.tokenizer, formatted, max_length=max_length, device=device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return input_ids, attention_mask, int(attention_mask.sum().item())


def iter_batches(
    pairs: list[tuple[str, str]],
    *,
    forward_batch_size: int,
) -> list[tuple[list[tuple[str, str]], int]]:
    batches = []
    for start in range(0, len(pairs), forward_batch_size):
        batch = pairs[start : start + forward_batch_size]
        real_count = len(batch)
        if real_count < forward_batch_size:
            batch = batch + [batch[-1]] * (forward_batch_size - real_count)
        batches.append((batch, real_count))
    return batches


def run_forward_batches(
    runner: LocalQwen3RerankerRunner,
    pairs: list[tuple[str, str]],
    *,
    forward_batch_size: int,
    max_length: int,
    warmup_batches: int,
    device: torch.device,
) -> dict:
    batches = iter_batches(pairs, forward_batch_size=forward_batch_size)
    encoded_batches = [
        (*build_batch(runner, batch, max_length=max_length, device=device), real_count)
        for batch, real_count in batches
    ]
    if runner.ffn_weight_mode != "dense":
        input_ids, attention_mask, _real_tokens, _real_count = encoded_batches[0]
        runner.calibrate_ffn_input_scales(input_ids, attention_mask)

    warmup_timings = []
    for index in range(min(warmup_batches, len(encoded_batches))):
        input_ids, attention_mask, _real_tokens, _real_count = encoded_batches[index]
        warmup_timings.append(time_call(device, lambda: runner.score_ids(input_ids, attention_mask)))

    total_real_pairs = 0
    total_real_tokens = 0
    total_padded_tokens = 0
    measured = []
    for input_ids, attention_mask, real_tokens, real_count in encoded_batches:
        total_real_pairs += real_count
        total_real_tokens += real_tokens
        total_padded_tokens += int(input_ids.numel())
        measured.append(time_call(device, lambda: runner.score_ids(input_ids, attention_mask)))

    total_sec = sum(measured)
    request_count = (len(pairs) + runner.random_batch_size - 1) // runner.random_batch_size
    return {
        "forward_batch_size": forward_batch_size,
        "num_batches": len(encoded_batches),
        "num_pairs": len(pairs),
        "num_requests": request_count,
        "random_batch_size": runner.random_batch_size,
        "random_input_len": max_length,
        "compile_first_batch_sec": warmup_timings[0] if runner.compile_forward and warmup_timings else None,
        "warmup_sec": warmup_timings,
        "mean_batch_sec": mean(measured),
        "total_sec": total_sec,
        "pairs_s": total_real_pairs / total_sec,
        "requests_s": request_count / total_sec,
        "real_input_tok_s": total_real_tokens / total_sec,
        "padded_tok_s": total_padded_tokens / total_sec,
        "total_real_tokens": total_real_tokens,
        "total_padded_tokens": total_padded_tokens,
    }


def print_summary(summary: dict) -> None:
    compile_time = summary.get("compile_first_batch_sec")
    compile_text = "" if compile_time is None else f" compile_first_batch={compile_time:.3f}s"
    print(
        f"forward_batch={summary['forward_batch_size']} "
        f"batches={summary['num_batches']} "
        f"pairs/s={summary['pairs_s']:.2f} "
        f"requests/s={summary['requests_s']:.2f} "
        f"real_tok/s={summary['real_input_tok_s']:.2f} "
        f"padded_tok/s={summary['padded_tok_s']:.2f} "
        f"total_sec={summary['total_sec']:.3f}s"
        f"{compile_text}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    log = StageLogger(enabled=args.verbose)
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    log.log("loading runner and model")
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=dtype,
        max_length=args.random_input_len,
        batch_size=max(args.forward_batch_sizes),
        compile_forward=args.compile_forward,
        attention_impl=args.attention_impl,
        ffn_weight_mode=args.ffn_weight_mode,
    )
    runner.random_batch_size = int(args.random_batch_size)

    log.log("generating synthetic rerank pairs")
    pairs = generate_pairs(
        runner.tokenizer,
        count=args.num_prompts,
        random_input_len=args.random_input_len,
        seed=args.seed,
    )

    summaries = []
    for forward_batch_size in args.forward_batch_sizes:
        log.log(f"benchmarking forward_batch_size={forward_batch_size}")
        summary = run_forward_batches(
            runner,
            pairs,
            forward_batch_size=forward_batch_size,
            max_length=args.random_input_len,
            warmup_batches=args.warmup_batches,
            device=device,
        )
        summaries.append(summary)
        print_summary(summary)

    result = {
        "model_dir": args.model_dir,
        "num_prompts": args.num_prompts,
        "random_input_len": args.random_input_len,
        "random_batch_size": args.random_batch_size,
        "forward_batch_sizes": args.forward_batch_sizes,
        "dtype": args.dtype,
        "device": args.device,
        "compile_forward": args.compile_forward,
        "attention_impl": args.attention_impl,
        "ffn_weight_mode": args.ffn_weight_mode,
        "summaries": summaries,
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=2)
    log.log("benchmark complete")


if __name__ == "__main__":
    main()
