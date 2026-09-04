#!/usr/bin/env python3
"""Synchronously test every position through the production MinerU decode graph.

This is not a single-operator imitation. It runs the same packed-projection,
24-layer, NZ-weight, NPU-RoPE, IncreFA, static-KV and LM-head graph used by the
continuous page pipeline. Every graph submission has flushed start/finish
markers, so the final unmatched start identifies a device stall exactly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from local_modeling_mineru import (
    DECODE_ATTENTION_INCREFA,
    DECODE_ROTARY_IMPL_NPU_APPLY,
    DECODE_WEIGHT_FORMAT_NZ,
    INCREFA_LENGTH_MODE_CHOICES,
    INCREFA_LENGTH_MODE_PSE_SENTINEL_310P,
    LocalMinerU2_5ForConditionalGeneration,
    LocalMinerUStaticCache,
    configure_decode_attention_impl,
    configure_decode_increfa_length_mode,
    configure_decode_packed_projections,
    configure_decode_rotary_impl,
    configure_decode_weight_format,
)
from run_local_model_two_step_extract import (
    compile_static_decode,
    configure_npu_conv3d_mode,
    configure_npu_jit_compile,
    maybe_sync_device,
    parse_torch_dtype,
)


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_CACHE = Path(".runtime_cache/11_mineru_2_5_pro_inference/production_decode")
DEFAULT_OUTPUT = Path(
    "tmp/11_mineru_2_5_pro_inference/production_decode_position_sweep/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--end-position", type=int, default=4095)
    parser.add_argument(
        "--increfa-length-mode",
        choices=INCREFA_LENGTH_MODE_CHOICES,
        default=INCREFA_LENGTH_MODE_PSE_SENTINEL_310P,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 1:
        parser.error("--cache-length must be greater than one")
    if not 0 <= args.start_position <= args.end_position < args.cache_length:
        parser.error("position range must be ordered and inside the KV cache")
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "MINERU_POSITION_SWEEP "
        + json.dumps({"event": event, **fields}, sort_keys=True, default=str),
        flush=True,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    configure_npu_jit_compile("off", device=args.device, verbose=True)
    configure_npu_conv3d_mode("auto", device=args.device, verbose=True)
    device = torch.device(args.device)
    dtype = parse_torch_dtype(args.dtype)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    progress("model_load", phase="start", model=str(args.model))
    model_started = time.perf_counter()
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        args.model.expanduser().resolve(), dtype=dtype, device=device
    ).eval()
    maybe_sync_device(device)
    progress("model_load", phase="finish", elapsed_s=time.perf_counter() - model_started)

    packed = configure_decode_packed_projections(model)
    weight_format = configure_decode_weight_format(model, DECODE_WEIGHT_FORMAT_NZ)
    rotary = configure_decode_rotary_impl(model, DECODE_ROTARY_IMPL_NPU_APPLY)
    attention = configure_decode_attention_impl(model, DECODE_ATTENTION_INCREFA)
    length_mode = configure_decode_increfa_length_mode(model, args.increfa_length_mode)
    tile_size = int(length_mode["ascend_310p_inner_tile_size"])
    boundaries = [
        value
        for value in range(tile_size, args.cache_length + 1, tile_size)
    ]
    progress(
        "production_configuration",
        phase="finish",
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        position_range=[args.start_position, args.end_position],
        weight_format=weight_format,
        rotary=rotary,
        attention=attention,
        length_mode=length_mode,
        exact_tile_effective_lengths=boundaries,
    )

    flat_decode = model.make_flat_static_decode_module(
        cache_length=args.cache_length
    ).eval()
    progress("compile_wrapper", phase="start", cache_dir=str(args.cache_dir))
    compiled_decode, compile_meta = compile_static_decode(
        flat_decode,
        device=device,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        decode_weight_format=str(weight_format["effective_mode"]),
        decode_rotary_impl=str(rotary["effective_mode"]),
        decode_attention_impl=str(attention["effective_mode"]),
        decode_increfa_length_mode=str(length_mode["effective_mode"]),
    )
    progress("compile_wrapper", phase="finish", compile=compile_meta)

    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    next_token = torch.randint(
        0,
        int(model.config.text_config.vocab_size),
        (args.batch_size, 1),
        generator=generator,
        dtype=torch.int64,
    ).to(device)
    cache_position = torch.full(
        (args.batch_size,),
        args.start_position,
        device=device,
        dtype=torch.int64,
    )
    rope_deltas = torch.zeros(
        (args.batch_size, 1), device=device, dtype=torch.int64
    )
    cache = LocalMinerUStaticCache.allocate(
        model.config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )

    elapsed_ms: list[float] = []
    token_samples: dict[str, list[int]] = {}
    sweep_started = time.perf_counter()
    for position in range(args.start_position, args.end_position + 1):
        effective_length = position + 1
        boundary = effective_length % tile_size == 0
        progress(
            "decode_step_graph",
            phase="start",
            position=position,
            effective_length=effective_length,
            effective_length_mod_tile=effective_length % tile_size,
            exact_tile_boundary=boundary,
        )
        step_started = time.perf_counter()
        logits = compiled_decode(
            next_token,
            cache_position,
            rope_deltas,
            *cache.flat_tensors(),
        )
        maybe_sync_device(device)
        step_ms = (time.perf_counter() - step_started) * 1000.0
        elapsed_ms.append(step_ms)
        next_token = torch.argmax(
            logits[:, -1, :].float(), dim=-1, keepdim=True
        )
        if position in (
            args.start_position,
            args.end_position,
            *(value - 1 for value in boundaries),
        ):
            token_samples[str(position)] = [
                int(value) for value in next_token[:4, 0].detach().cpu().tolist()
            ]
        progress(
            "decode_step_graph",
            phase="finish",
            position=position,
            effective_length=effective_length,
            effective_length_mod_tile=effective_length % tile_size,
            exact_tile_boundary=boundary,
            elapsed_ms=step_ms,
        )
        if position < args.end_position:
            cache_position.add_(1)

    sweep_s = time.perf_counter() - sweep_started
    payload = {
        "schema_version": 1,
        "kind": "mineru_production_decode_position_sweep",
        "scope": (
            "full production 24-layer compiled decode graph; every position "
            "synchronized; prefill and page scheduling excluded"
        ),
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": int(args.batch_size),
        "cache_length": int(args.cache_length),
        "start_position": int(args.start_position),
        "end_position": int(args.end_position),
        "positions_tested": int(len(elapsed_ms)),
        "all_positions_completed": len(elapsed_ms)
        == args.end_position - args.start_position + 1,
        "ascend_310p_inner_tile_size": tile_size,
        "exact_tile_effective_lengths": boundaries,
        "compile": compile_meta,
        "configuration": {
            "packed_projections": packed,
            "weight_format": weight_format,
            "rotary": rotary,
            "attention": attention,
            "increfa_length_mode": length_mode,
        },
        "timing": {
            "sweep_s": float(sweep_s),
            "step_ms_mean": float(statistics.fmean(elapsed_ms)),
            "step_ms_p50": percentile(elapsed_ms, 0.50),
            "step_ms_p95": percentile(elapsed_ms, 0.95),
            "step_ms_max": float(max(elapsed_ms)),
        },
        "token_samples_first_four_rows": token_samples,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress("position_sweep", phase="finish", output=str(output), **payload["timing"])
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
