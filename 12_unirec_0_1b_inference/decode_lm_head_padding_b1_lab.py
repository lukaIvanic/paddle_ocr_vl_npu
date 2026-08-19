#!/usr/bin/env python3
"""Measure an aligned UniRec LM head in the lane-A B1 decode graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import modeling_optimized_unirec as unirec
from decode_head_padding_b1_lab import make_state
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import (
    profile_compiled_lane,
    profile_compiled_timing,
    run_steps,
    step,
)


SEMANTIC_VOCAB_SIZE = 56_371
ALIGNED_VOCAB_SIZE = 57_344
SELF_CACHE_LENGTH = 256
CROSS_CACHE_LENGTH = 256
INITIAL_CACHE_POSITION = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lm-head-rows",
        type=int,
        choices=(SEMANTIC_VOCAB_SIZE, ALIGNED_VOCAB_SIZE),
        required=True,
    )
    parser.add_argument("--source-length", type=int, default=56)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.source_length <= CROSS_CACHE_LENGTH:
        parser.error("--source-length must be within C256")
    phase_calls = {
        "warmup": args.warmup_steps,
        "measure": args.measure_steps,
        "timing": 2 * args.timing_steps,
        "validation": args.validation_steps,
    }
    for phase, calls in phase_calls.items():
        if calls < 1:
            parser.error(f"--{phase}-steps must be positive")
        if INITIAL_CACHE_POSITION + calls > SELF_CACHE_LENGTH:
            parser.error(
                f"{phase} would advance cache position beyond S256: "
                f"start={INITIAL_CACHE_POSITION} calls={calls}"
            )
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_LM_HEAD_PADDING_B1_PROGRESS "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def pad_lm_head(runner: OptimizedUniRecRunner) -> int:
    head = runner.model.lm_head
    rows = int(head.weight.shape[0])
    if rows != SEMANTIC_VOCAB_SIZE:
        raise ValueError(f"expected {SEMANTIC_VOCAB_SIZE} LM-head rows, got {rows}")
    padded_rows = ALIGNED_VOCAB_SIZE - SEMANTIC_VOCAB_SIZE
    weight = torch.nn.functional.pad(head.weight.detach(), (0, 0, 0, padded_rows))
    head.weight = nn.Parameter(weight, requires_grad=False)
    head.out_features = ALIGNED_VOCAB_SIZE
    synchronize_device(runner.device)
    return padded_rows


class SemanticVocabDecodeStepModule(unirec.LocalUniRecCachedDecodeStepModule):
    """Run the aligned head, then hide the zero-padding rows from argmax."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        logits = super().forward(*args, **kwargs)
        return logits[..., :SEMANTIC_VOCAB_SIZE]


def eager_logits(
    runner: OptimizedUniRecRunner,
    state: dict[str, Any],
) -> torch.Tensor:
    logits = runner.model.forward_cached_logits(
        decoder_input_ids=state["next_token"],
        cache_position=state["cache_position"],
        active_length=0,
        key_cache=state["self_keys"],
        value_cache=state["self_values"],
        cross_key_cache=state["cross_keys"],
        cross_value_cache=state["cross_values"],
        cross_attention_mask=state["cross_mask"],
        self_attention_backend="increfa_all",
    )[..., :SEMANTIC_VOCAB_SIZE]
    synchronize_device(runner.device)
    return logits.detach().float().cpu()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    progress("lane_begin", lm_head_rows=args.lm_head_rows, weights_nz=True)
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    if int(runner.config.vocab_size) != SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"expected config vocab {SEMANTIC_VOCAB_SIZE}, "
            f"got {runner.config.vocab_size}"
        )

    padded_rows = 0
    if args.lm_head_rows == ALIGNED_VOCAB_SIZE:
        padded_rows = pad_lm_head(runner)
        unirec.LocalUniRecCachedDecodeStepModule = SemanticVocabDecodeStepModule

    nz_started = time.perf_counter()
    nz_tensor_count = runner.cast_decoder_weights_nz()
    nz_format_s = time.perf_counter() - nz_started

    def new_state(seed: int) -> dict[str, Any]:
        return make_state(
            runner,
            physical_heads=6,
            seed=seed,
            source_length=args.source_length,
        )

    eager_state = new_state(11)
    reference_logits = eager_logits(runner, eager_state)

    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=CROSS_CACHE_LENGTH,
        batch_size=1,
        self_cache_len=SELF_CACHE_LENGTH,
    )

    warm_state = new_state(7)
    progress("first_call_begin", lm_head_rows=args.lm_head_rows)
    first_started = time.perf_counter()
    step(module, warm_state)
    synchronize_device(args.device)
    first_call_s = time.perf_counter() - first_started
    progress(
        "first_call_end",
        lm_head_rows=args.lm_head_rows,
        first_call_s=first_call_s,
    )
    run_steps(module, warm_state, args.warmup_steps - 1)
    synchronize_device(args.device)

    measure_state = new_state(7)
    synchronize_device(args.device)
    measured_started = time.perf_counter()
    run_steps(module, measure_state, args.measure_steps)
    synchronize_device(args.device)
    measured_s = time.perf_counter() - measured_started

    timing_state = new_state(17)
    compiled_timing = profile_compiled_timing(
        fn=module,
        state=timing_state,
        device=args.device,
        steps=args.timing_steps,
        stepper=step,
    )

    validation_state = new_state(11)
    compiled_logits, validation_tokens = run_steps(
        module, validation_state, args.validation_steps, collect=True
    )
    synchronize_device(args.device)
    compiled_logits_cpu = compiled_logits.detach().float().cpu()
    local_delta = (reference_logits - compiled_logits_cpu).abs()

    lane_root = args.output.parent / f"vocab{args.lm_head_rows}"
    compiled_profile = profile_compiled_lane(
        backend="increfa_all",
        fn=module,
        state=new_state(19),
        device=args.device,
        output_root=lane_root,
        steps=1,
        metric="pipe",
        stepper=step,
    )

    logits_path = args.output.with_suffix(".validation_logits.npy")
    eager_path = args.output.with_suffix(".eager_logits.npy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, compiled_logits_cpu.numpy())
    np.save(eager_path, reference_logits.numpy())
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_lm_head_padding_b1_lab",
        "status": "ok",
        "shape": {
            "batch_size": 1,
            "self_cache_length": SELF_CACHE_LENGTH,
            "cross_cache_length": CROSS_CACHE_LENGTH,
            "cache_position": INITIAL_CACHE_POSITION,
            "source_length": args.source_length,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "lm_head_rows": args.lm_head_rows,
        },
        "padded_rows": padded_rows,
        "logits_rows_returned": int(compiled_logits_cpu.shape[-1]),
        "weights_nz": bool(runner.weights_nz),
        "nz_tensor_count": nz_tensor_count,
        "nz_format_s": nz_format_s,
        "first_call_s": first_call_s,
        "measure_steps": args.measure_steps,
        "validation_steps": args.validation_steps,
        "measure_s": measured_s,
        "step_ms": measured_s * 1000.0 / args.measure_steps,
        "raw_tok_s": args.measure_steps / measured_s,
        "compiled_timing": compiled_timing,
        "compiled_profile": compiled_profile,
        "compile": compile_meta,
        "compiled_vs_same_lane_eager": {
            "max_abs": float(local_delta.max().item()),
            "mean_abs": float(local_delta.mean().item()),
            "argmax_exact": bool(
                compiled_logits_cpu.argmax().item() == reference_logits.argmax().item()
            ),
        },
        "validation_tokens": validation_tokens,
        "validation_logits_npy": str(logits_path),
        "eager_logits_npy": str(eager_path),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress(
        "lane_end",
        lm_head_rows=args.lm_head_rows,
        step_ms=payload["step_ms"],
        raw_tok_s=payload["raw_tok_s"],
    )
    print("UNIREC_DECODE_LM_HEAD_PADDING_B1: PASS")
    print(
        "UNIREC_DECODE_LM_HEAD_PADDING_B1_RESULT "
        + json.dumps(payload, sort_keys=True)
    )


if __name__ == "__main__":
    main()
