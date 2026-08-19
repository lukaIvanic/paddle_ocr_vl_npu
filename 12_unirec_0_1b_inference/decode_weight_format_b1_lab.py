#!/usr/bin/env python3
"""Compare ND and preformatted FRACTAL_NZ weights for UniRec lane-A B1 decode."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import (
    make_state,
    profile_compiled_lane,
    profile_compiled_timing,
    run_steps,
    step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-length", type=int, default=56)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--timing-steps", type=int, default=30)
    parser.add_argument(
        "--weight-formats",
        nargs="+",
        choices=("nd", "nz"),
        default=("nd",),
        help=(
            "Run one format per process for valid TorchDynamo code-object "
            "caching. Passing both is rejected."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.source_length <= 256:
        parser.error("--source-length must be within C256")
    if len(args.weight_formats) != 1:
        parser.error("run exactly one --weight-formats value per process")
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_WEIGHT_B1_PROGRESS "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def configure_realistic_mask_(state: dict[str, Any], source_length: int) -> None:
    mask = state["cross_mask"]
    mask.fill_(torch.finfo(mask.dtype).min)
    mask[..., :source_length].zero_()
    if not bool((~mask.bool()).reshape(mask.shape[0], -1).any(dim=1).all().item()):
        raise RuntimeError("B1 weight lab constructed a fully masked IncreFA row")


@torch.inference_mode()
def run_lane(args: argparse.Namespace, weight_format: str) -> tuple[dict[str, Any], torch.Tensor, list[list[int]]]:
    lane_root = args.output.parent / weight_format
    lane_root.mkdir(parents=True, exist_ok=True)
    progress("lane_begin", weight_format=weight_format)
    model_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    model_load_s = time.perf_counter() - model_started
    nz_tensor_count = 0
    if weight_format == "nz":
        nz_started = time.perf_counter()
        nz_tensor_count = runner.cast_decoder_weights_nz()
        nz_format_s = time.perf_counter() - nz_started
    else:
        nz_format_s = 0.0

    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=256,
        batch_size=1,
        self_cache_len=256,
    )

    def new_state(seed: int) -> dict[str, Any]:
        state = make_state(
            runner,
            batch_size=1,
            self_cache_length=256,
            cross_cache_length=256,
            cache_position=32,
            seed=seed,
        )
        configure_realistic_mask_(state, args.source_length)
        tensors = (
            state["next_token"],
            state["cache_position"],
            *state["self_keys"],
            *state["self_values"],
            *state["cross_keys"],
            *state["cross_values"],
            state["cross_mask"],
        )
        if not all(tensor.is_inference() for tensor in tensors):
            raise RuntimeError("B1 weight lab state must use inference tensors")
        return state

    warm_state = new_state(7)
    progress("first_call_begin", weight_format=weight_format)
    first_started = time.perf_counter()
    step(module, warm_state)
    synchronize_device(args.device)
    first_call_s = time.perf_counter() - first_started
    progress(
        "first_call_end",
        weight_format=weight_format,
        first_call_s=first_call_s,
    )
    run_steps(module, warm_state, max(0, args.warmup_steps - 1))
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
    validation_logits, validation_tokens = run_steps(
        module,
        validation_state,
        1,
        collect=True,
    )
    synchronize_device(args.device)
    validation_logits_cpu = validation_logits.detach().float().cpu()

    profile_state = new_state(19)
    compiled_profile = profile_compiled_lane(
        backend="increfa_all",
        fn=module,
        state=profile_state,
        device=args.device,
        output_root=lane_root,
        steps=1,
        metric="pipe",
        stepper=step,
    )

    lane = {
        "weight_format": weight_format,
        "weights_nz": bool(runner.weights_nz),
        "nz_tensor_count": nz_tensor_count,
        "model_load_s": model_load_s,
        "nz_format_s": nz_format_s,
        "first_call_s": first_call_s,
        "measure_steps": args.measure_steps,
        "measure_s": measured_s,
        "step_ms": measured_s * 1000.0 / args.measure_steps,
        "raw_tok_s": args.measure_steps / measured_s,
        "compiled_timing": compiled_timing,
        "compiled_profile": compiled_profile,
        "compile": compile_meta,
        "validation_tokens": validation_tokens,
    }
    progress(
        "lane_end",
        weight_format=weight_format,
        step_ms=lane["step_ms"],
        raw_tok_s=lane["raw_tok_s"],
    )
    del profile_state, validation_state, timing_state, measure_state, warm_state
    del module, runner
    gc.collect()
    torch.npu.empty_cache()
    return lane, validation_logits_cpu, validation_tokens


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lanes = {}
    logits = {}
    tokens = {}
    for weight_format in args.weight_formats:
        lanes[weight_format], logits[weight_format], tokens[weight_format] = run_lane(
            args,
            weight_format,
        )
    weight_format = args.weight_formats[0]
    logits_path = args.output.with_suffix(".validation_logits.npy")
    np.save(logits_path, logits[weight_format].numpy())
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_weight_format_b1_lab",
        "status": "ok",
        "shape": {
            "batch_size": 1,
            "self_cache_length": 256,
            "cross_cache_length": 256,
            "cache_position": 32,
            "source_length": args.source_length,
        },
        "lanes": lanes,
        "validation_logits_npy": str(logits_path),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_DECODE_WEIGHT_B1: PASS")
    print("UNIREC_DECODE_WEIGHT_B1_RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
