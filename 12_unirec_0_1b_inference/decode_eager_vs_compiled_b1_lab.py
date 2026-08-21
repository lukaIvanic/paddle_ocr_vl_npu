#!/usr/bin/env python3
"""Compare raw eager and TorchAir UniRec lane-A decode throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import modeling_optimized_unirec as unirec
from decode_lm_head_padding_b1_lab import (
    ALIGNED_VOCAB_SIZE,
    SEMANTIC_VOCAB_SIZE,
    SemanticVocabDecodeStepModule,
    pad_lm_head,
)
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import make_state, run_steps


SELF_CACHE_LENGTH = 256
CROSS_CACHE_LENGTH = 256
INITIAL_CACHE_POSITION = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-length", type=int, default=56)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--validation-steps", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 1 <= args.source_length <= CROSS_CACHE_LENGTH:
        parser.error("--source-length must be within C256")
    for name in ("warmup_steps", "measure_steps", "repeats", "validation_steps"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    longest = max(args.warmup_steps, args.measure_steps, args.validation_steps)
    if INITIAL_CACHE_POSITION + longest > SELF_CACHE_LENGTH:
        parser.error("a timing phase would advance beyond S256")
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_EAGER_COMPILED_B1_PROGRESS "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def configure_cross_mask_(state: dict[str, Any], source_length: int) -> None:
    mask = state["cross_mask"]
    mask.fill_(torch.finfo(mask.dtype).min)
    mask[..., :source_length].zero_()
    valid = (~mask.bool()).reshape(mask.shape[0], -1).any(dim=1)
    if not bool(valid.all().item()):
        raise RuntimeError("constructed a fully masked IncreFA row")


def summarize(
    round_s: list[float], steps: int, batch_size: int
) -> dict[str, Any]:
    median_s = statistics.median(round_s)
    token_slots = steps * batch_size
    return {
        "round_s": round_s,
        "median_s": median_s,
        "median_step_ms": median_s * 1000.0 / steps,
        "median_raw_tok_s": token_slots / median_s,
        "round_raw_tok_s": [token_slots / value for value in round_s],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    progress("model_load_begin", model=str(args.model))
    load_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    model_load_s = time.perf_counter() - load_started
    progress("model_load_end", seconds=model_load_s)
    if int(runner.config.vocab_size) != SEMANTIC_VOCAB_SIZE:
        raise ValueError("unexpected UniRec vocabulary size")

    padded_rows = pad_lm_head(runner)
    unirec.LocalUniRecCachedDecodeStepModule = SemanticVocabDecodeStepModule
    nz_started = time.perf_counter()
    nz_tensor_count = runner.cast_decoder_weights_nz()
    nz_format_s = time.perf_counter() - nz_started
    progress(
        "optimizations_ready",
        lm_head_rows=ALIGNED_VOCAB_SIZE,
        padded_rows=padded_rows,
        nz_tensor_count=nz_tensor_count,
        nz_format_s=nz_format_s,
    )

    def new_state(seed: int) -> dict[str, Any]:
        state = make_state(
            runner,
            batch_size=args.batch_size,
            self_cache_length=SELF_CACHE_LENGTH,
            cross_cache_length=CROSS_CACHE_LENGTH,
            cache_position=INITIAL_CACHE_POSITION,
            seed=seed,
        )
        configure_cross_mask_(state, args.source_length)
        return state

    eager_module = SemanticVocabDecodeStepModule(
        runner.model,
        self_attention_backend="increfa_all",
    )
    compiled_module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=CROSS_CACHE_LENGTH,
        batch_size=args.batch_size,
        self_cache_len=SELF_CACHE_LENGTH,
    )

    first_state = new_state(7)
    progress("compiled_first_call_begin")
    first_started = time.perf_counter()
    run_steps(compiled_module, first_state, 1)
    synchronize_device(args.device)
    compiled_first_call_s = time.perf_counter() - first_started
    progress("compiled_first_call_end", seconds=compiled_first_call_s)
    run_steps(compiled_module, first_state, args.warmup_steps - 1)
    synchronize_device(args.device)

    eager_warm_state = new_state(7)
    progress("eager_warmup_begin", steps=args.warmup_steps)
    run_steps(eager_module, eager_warm_state, args.warmup_steps)
    synchronize_device(args.device)
    progress("eager_warmup_end")

    round_s: dict[str, list[float]] = {"raw_eager": [], "compiled_torchair": []}
    functions = {
        "raw_eager": eager_module,
        "compiled_torchair": compiled_module,
    }
    for repeat in range(args.repeats):
        order = (
            ("raw_eager", "compiled_torchair")
            if repeat % 2 == 0
            else ("compiled_torchair", "raw_eager")
        )
        for lane in order:
            state = new_state(100 + repeat)
            synchronize_device(args.device)
            started = time.perf_counter()
            run_steps(functions[lane], state, args.measure_steps)
            synchronize_device(args.device)
            elapsed = time.perf_counter() - started
            round_s[lane].append(elapsed)
            progress(
                "measure_round",
                lane=lane,
                repeat=repeat,
                seconds=elapsed,
                raw_tok_s=args.batch_size * args.measure_steps / elapsed,
            )

    validation: dict[str, dict[str, Any]] = {}
    for lane, function in functions.items():
        state = new_state(911)
        logits, tokens = run_steps(
            function,
            state,
            args.validation_steps,
            collect=True,
        )
        synchronize_device(args.device)
        validation[lane] = {
            "tokens": tokens,
            "logits": logits.detach().float().cpu(),
        }
    eager_validation = validation["raw_eager"]
    compiled_validation = validation["compiled_torchair"]
    delta = (eager_validation["logits"] - compiled_validation["logits"]).abs()
    token_exact = eager_validation["tokens"] == compiled_validation["tokens"]
    if not token_exact:
        raise RuntimeError("raw eager and compiled validation tokens differ")

    lanes = {
        name: summarize(values, args.measure_steps, args.batch_size)
        for name, values in round_s.items()
    }
    speedup = (
        lanes["compiled_torchair"]["median_raw_tok_s"]
        / lanes["raw_eager"]["median_raw_tok_s"]
    )
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_eager_vs_compiled_b1_lab",
        "status": "ok",
        "scope": "full six-layer decode step plus argmax and state advance; prefill excluded",
        "shape": {
            "batch_size": args.batch_size,
            "self_cache_length": SELF_CACHE_LENGTH,
            "cross_cache_length": CROSS_CACHE_LENGTH,
            "cache_position": INITIAL_CACHE_POSITION,
            "source_length": args.source_length,
            "attention_backend": "increfa_all",
            "semantic_heads": 6,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "lm_head_rows": ALIGNED_VOCAB_SIZE,
        },
        "optimizations": {
            "dtype": "float16",
            "npu_jit_compile": False,
            "decode_weight_format": "nz",
            "nz_tensor_count": nz_tensor_count,
            "lm_head_rows": ALIGNED_VOCAB_SIZE,
            "qkv_projection": "separate",
        },
        "model_load_s": model_load_s,
        "nz_format_s": nz_format_s,
        "compiled_first_call_s_excluded": compiled_first_call_s,
        "warmup_steps": args.warmup_steps,
        "measure_steps_per_round": args.measure_steps,
        "repeats": args.repeats,
        "lanes": lanes,
        "compiled_speedup": speedup,
        "validation": {
            "steps": args.validation_steps,
            "token_exact": token_exact,
            "final_logits_max_abs": float(delta.max()),
            "final_logits_mean_abs": float(delta.mean()),
            "final_logits_cosine": float(
                F.cosine_similarity(
                    eager_validation["logits"].flatten(),
                    compiled_validation["logits"].flatten(),
                    dim=0,
                )
            ),
        },
        "compile": compile_meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_DECODE_EAGER_VS_COMPILED_B1: PASS")
    print("UNIREC_DECODE_EAGER_VS_COMPILED_B1_RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
