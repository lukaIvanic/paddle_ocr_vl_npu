#!/usr/bin/env python3
"""Production-shaped MinerU text-decode benchmark and NPU profiler lab.

The lab excludes image processing and prefill.  It executes the real 24-layer
decoder, static KV-cache update, rotary encoding, LM head, and token feedback.
It can compare the existing manual GQA implementation with Ascend IncreFA.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from local_modeling_mineru import (
    DECODE_ATTENTION_CHOICES,
    DECODE_ATTENTION_INCREFA,
    DECODE_ATTENTION_MANUAL,
    DECODE_OPTIMIZATION_CHOICES,
    DECODE_OPTIMIZATION_CURRENT,
    DECODE_ROTARY_IMPL_CHOICES,
    DECODE_ROTARY_IMPL_MANUAL,
    DECODE_WEIGHT_FORMAT_CHOICES,
    DECODE_WEIGHT_FORMAT_NONE,
    LocalMinerU2_5ForConditionalGeneration,
    LocalMinerUStaticCache,
    configure_decode_attention_impl,
    configure_decode_packed_projections,
    configure_decode_rotary_impl,
    configure_decode_weight_format,
)
from run_local_model_two_step_extract import (
    compile_static_decode,
    configure_npu_conv3d_mode,
    configure_npu_jit_compile,
    maybe_sync_device,
    npu_profiler_config,
    parse_torch_dtype,
)


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_CACHE = Path(".runtime_cache/11_mineru_2_5_pro_inference/text_decode_lab")
DEFAULT_OUTPUT = Path("tmp/11_mineru_2_5_pro_inference/text_decode_lab/result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--profile-position", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measure-steps", type=int, default=64)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument(
        "--attention",
        action="append",
        choices=DECODE_ATTENTION_CHOICES,
        default=None,
        help="Attention lane. Repeat to select lanes; default: manual and increfa.",
    )
    parser.add_argument(
        "--decode-weight-format",
        choices=DECODE_WEIGHT_FORMAT_CHOICES,
        default=DECODE_WEIGHT_FORMAT_NONE,
    )
    parser.add_argument(
        "--decode-rotary-impl",
        choices=DECODE_ROTARY_IMPL_CHOICES,
        default=DECODE_ROTARY_IMPL_MANUAL,
    )
    parser.add_argument(
        "--decode-optimization",
        choices=DECODE_OPTIMIZATION_CHOICES,
        default=DECODE_OPTIMIZATION_CURRENT,
        help=(
            "Static-decode graph optimization preset. Paddle transfer presets "
            "retain the current model math while hoisting mask/RoPE work."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 1:
        parser.error("--cache-length must be greater than one")
    if not 0 <= args.profile_position < args.cache_length:
        parser.error("--profile-position must be inside the KV cache")
    needed = max(args.warmup_steps, args.measure_steps, args.validation_steps, args.profile_steps)
    if args.profile_position + needed >= args.cache_length:
        parser.error("profile position plus requested steps reaches KV capacity")
    if min(args.warmup_steps, args.measure_steps, args.validation_steps, args.profile_steps) < 0:
        parser.error("step counts must be non-negative")
    if args.measure_steps == 0 or args.validation_steps == 0:
        parser.error("measure and validation steps must be positive")
    if args.attention is None:
        args.attention = [DECODE_ATTENTION_MANUAL, DECODE_ATTENTION_INCREFA]
    return args


def progress(event: str, **fields: Any) -> None:
    print("MINERU_DECODE_LAB " + json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def make_state(
    model: LocalMinerU2_5ForConditionalGeneration,
    *,
    batch_size: int,
    cache_length: int,
    cache_position: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    input_ids = torch.randint(
        0,
        int(model.config.text_config.vocab_size),
        (batch_size, 1),
        generator=generator,
        dtype=torch.int64,
    ).to(model.device)
    cache = LocalMinerUStaticCache.allocate(
        model.config.text_config,
        batch_size=batch_size,
        cache_length=cache_length,
        device=model.device,
        dtype=model.dtype,
        init_mode="zeros",
    )
    return {
        "next_token": input_ids,
        "cache_position": torch.full(
            (batch_size,), cache_position, device=model.device, dtype=torch.int64
        ),
        "rope_deltas": torch.zeros(
            (batch_size, 1), device=model.device, dtype=torch.int64
        ),
        "flat_cache": cache.flat_tensors(),
    }


def run_steps(fn: Any, state: dict[str, Any], steps: int, *, collect: bool = False) -> dict[str, Any]:
    token_history: list[torch.Tensor] = []
    logits = None
    for _ in range(int(steps)):
        logits = fn(
            state["next_token"],
            state["cache_position"],
            state["rope_deltas"],
            *state["flat_cache"],
        )
        state["next_token"] = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        state["cache_position"].add_(1)
        if collect:
            token_history.append(state["next_token"].detach().cpu())
    return {
        "logits": logits,
        "token_ids": None
        if not collect
        else torch.cat(token_history, dim=1).tolist(),
    }


def profile_lane(
    *,
    fn: Any,
    state: dict[str, Any],
    output_root: Path,
    attention: str,
    steps: int,
    metric: str,
    device: torch.device,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profile_dir = output_root / f"profile_{attention}_{metric}"
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    maybe_sync_device(device)
    started = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(f"mineru.text_decode.{attention}"):
            run_steps(fn, state, steps)
        maybe_sync_device(device)
        profiler.step()
    maybe_sync_device(device)
    return {
        "profile_dir": str(profile_dir),
        "profile_steps": int(steps),
        "profile_wall_s": float(time.perf_counter() - started),
        "profile_wall_is_throughput_measurement": False,
        "metric": str(metric),
    }


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

    progress("model_load_begin", model=str(args.model))
    load_started = time.perf_counter()
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        args.model.expanduser().resolve(), dtype=dtype, device=device
    ).eval()
    maybe_sync_device(device)
    model_load_s = time.perf_counter() - load_started
    progress("model_load_end", seconds=model_load_s)
    packed_projections = configure_decode_packed_projections(model)
    weight_format = configure_decode_weight_format(model, args.decode_weight_format)
    rotary = configure_decode_rotary_impl(model, args.decode_rotary_impl)

    lanes: dict[str, Any] = {}
    validation: dict[str, dict[str, Any]] = {}
    for attention in args.attention:
        progress("lane_begin", attention=attention)
        attention_meta = configure_decode_attention_impl(model, attention)
        flat_decode = model.make_flat_static_decode_module(
            cache_length=args.cache_length,
            decode_optimization=args.decode_optimization,
        ).eval()
        compile_started = time.perf_counter()
        fn, compile_meta = compile_static_decode(
            flat_decode,
            device=device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            decode_weight_format=str(weight_format["effective_mode"]),
            decode_rotary_impl=str(rotary["effective_mode"]),
            decode_attention_impl=attention,
            decode_optimization=args.decode_optimization,
        )
        compile_wrapper_s = time.perf_counter() - compile_started

        warm_state = make_state(
            model,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            cache_position=args.profile_position,
            seed=args.seed,
        )
        progress("first_call_begin", attention=attention)
        first_started = time.perf_counter()
        run_steps(fn, warm_state, 1)
        maybe_sync_device(device)
        first_call_s = time.perf_counter() - first_started
        progress("first_call_end", attention=attention, seconds=first_call_s)
        if args.warmup_steps > 1:
            run_steps(fn, warm_state, args.warmup_steps - 1)
            maybe_sync_device(device)

        measure_state = make_state(
            model,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            cache_position=args.profile_position,
            seed=args.seed,
        )
        maybe_sync_device(device)
        measured_started = time.perf_counter()
        run_steps(fn, measure_state, args.measure_steps)
        maybe_sync_device(device)
        measured_s = time.perf_counter() - measured_started

        validation_state = make_state(
            model,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            cache_position=args.profile_position,
            seed=args.seed,
        )
        validation_result = run_steps(
            fn, validation_state, args.validation_steps, collect=True
        )
        maybe_sync_device(device)
        validation_logits = validation_result["logits"].float().cpu()
        validation[attention] = {
            "token_ids": validation_result["token_ids"],
            "logits": validation_logits,
        }

        profile = None
        if args.profile and args.profile_steps > 0:
            progress("profile_begin", attention=attention)
            profile_state = make_state(
                model,
                batch_size=args.batch_size,
                cache_length=args.cache_length,
                cache_position=args.profile_position,
                seed=args.seed,
            )
            profile = profile_lane(
                fn=fn,
                state=profile_state,
                output_root=output.parent,
                attention=attention,
                steps=args.profile_steps,
                metric=args.profile_metric,
                device=device,
            )
            progress("profile_end", attention=attention, profile_dir=profile["profile_dir"])

        raw_tokens = args.batch_size * args.measure_steps
        lanes[attention] = {
            "attention": attention_meta,
            "compile": {
                **compile_meta,
                "compile_wrapper_s": float(compile_wrapper_s),
                "first_call_s": float(first_call_s),
            },
            "measure": {
                "steps": int(args.measure_steps),
                "raw_tokens": int(raw_tokens),
                "decode_s": float(measured_s),
                "step_ms": float(measured_s * 1000.0 / args.measure_steps),
                "raw_tok_s": float(raw_tokens / measured_s),
            },
            "validation_token_ids": validation_result["token_ids"],
            "profile": profile,
        }
        progress("lane_end", attention=attention, raw_tok_s=raw_tokens / measured_s)

    comparisons: dict[str, Any] = {}
    if DECODE_ATTENTION_MANUAL in validation and DECODE_ATTENTION_INCREFA in validation:
        manual = validation[DECODE_ATTENTION_MANUAL]
        increfa = validation[DECODE_ATTENTION_INCREFA]
        left = manual["logits"]
        right = increfa["logits"]
        delta = (left - right).abs()
        comparisons["manual_vs_increfa"] = {
            "validation_steps": int(args.validation_steps),
            "token_exact": manual["token_ids"] == increfa["token_ids"],
            "manual_token_ids": manual["token_ids"],
            "increfa_token_ids": increfa["token_ids"],
            "final_logits_max_abs": float(delta.max()),
            "final_logits_mean_abs": float(delta.mean()),
            "final_logits_cosine": float(
                F.cosine_similarity(left.flatten(), right.flatten(), dim=0)
            ),
            "speedup": float(
                lanes[DECODE_ATTENTION_INCREFA]["measure"]["raw_tok_s"]
                / lanes[DECODE_ATTENTION_MANUAL]["measure"]["raw_tok_s"]
            ),
        }

    payload = {
        "schema_version": 1,
        "kind": "mineru_text_decode_lab",
        "scope": "warmed full 24-layer one-token static decode; prefill excluded",
        "model": str(args.model.expanduser().resolve()),
        "device": str(device),
        "dtype": str(dtype),
        "shape": {
            "batch_size": int(args.batch_size),
            "cache_length": int(args.cache_length),
            "initial_cache_position": int(args.profile_position),
            "decode_optimization": str(args.decode_optimization),
        },
        "model_load_s": float(model_load_s),
        "packed_projections": packed_projections,
        "weight_format": weight_format,
        "rotary": rotary,
        "lanes": lanes,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
