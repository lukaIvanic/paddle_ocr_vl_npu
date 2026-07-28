#!/usr/bin/env python3
"""Benchmark experiment-3 static-compiled decode as a tok/s ladder.

One run measures the same crop through four decode configurations and prints:

  eager              growing KV cache, uncompiled (the exp02 reference loop)
  eager_fixed_cache  preallocated KV cache + flat decode module, uncompiled
  compiled_native    TorchAir static-compiled decode graph, native weights
  compiled_nz        the same graph with FRACTAL_NZ decode linear weights;
                     reported unavailable (with the runtime's reason) when
                     the weight-format cast falls back

Correctness rides along: all lanes must produce identical token ids, and the
compiled graph is walked in lockstep with the uncompiled flat module to bound
per-step logit drift. The fixed-cache lanes stop at EOS (checked without a
blocking host sync); the eager lane always runs the full step count, so when
EOS fires inside the token budget only the EOS-trimmed matches are expected
to hold.

Reading map, top to bottom: decode lanes, TorchAir compile, correctness
helpers, profiler capture, then main() runs the lanes in ladder order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu  # noqa: F401  (must precede torchair: it registers the module)
import torchair
import torchair.inference
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_LINEAR_WEIGHT_FORMAT,
    LocalPaddleOCRVLForConditionalGeneration,
    cast_decode_linear_weights_to_nz,
)
from run_local_recognition import DEFAULT_CROP, DTYPES, build_inputs, preprocess_image

DEFAULT_TORCHAIR_CACHE_DIR = Path("outputs") / "torchair_cache"


def timed(fn: Callable):
    torch.npu.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.npu.synchronize()
    return result, time.perf_counter() - start


def tok_per_s(tokens: int, seconds: float) -> float:
    return float(tokens) / float(seconds) if seconds > 0 else float("inf")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


# ---------------------------------------------------------------------------
# Decode lanes
# ---------------------------------------------------------------------------


@dataclass
class DecodeLoopResult:
    ids: torch.Tensor
    decode_calls: int
    eos_detected: bool
    eos_step: int | None


@torch.inference_mode()
def eager_decode_loop(
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    """Reference decode: growing KV cache, one model.forward per token,
    fixed step count (no EOS stop) so the measured rate is per-step cost."""
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=True,
        logits_to_keep=1,
    )
    past = outputs.past_key_values
    rope_deltas = outputs.rope_deltas
    next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
    generated = [next_token]
    current_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    for _ in range(max(0, int(max_new_tokens) - 1)):
        outputs = model.forward(
            input_ids=next_token,
            attention_mask=current_attention_mask,
            pixel_values=None,
            image_grid_thw=None,
            past_key_values=past,
            use_cache=True,
            rope_deltas=rope_deltas,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)
    return torch.cat(generated, dim=1)


@torch.inference_mode()
def make_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    cache_length: int,
):
    """Run prefill into a fresh fixed-size KV cache; return it with the
    first generated token."""
    prefill = model.forward_static_prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        cache_length=cache_length,
        logits_to_keep=1,
    )
    next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
    return prefill, next_token


@torch.inference_mode()
def fixed_cache_decode_loop(
    decode_fn: Callable,
    prefill,
    next_token: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int,
) -> DecodeLoopResult:
    """Decode against the fixed cache with `decode_fn` (the flat module, or
    its compiled graph).

    Stops at EOS without a blocking host sync: each step's EOS flag is copied
    to pinned CPU memory on a second stream and checked one step later, so the
    check overlaps the next decode step (queue-depth-1)."""
    generated = [next_token]
    cache_position = prefill.next_cache_position
    flat_cache = prefill.cache.flat_tensors()
    max_decode_calls = max(0, int(max_new_tokens) - 1)
    decode_calls = 0
    eos_detected = False
    eos_step = None
    cpu_flags = torch.zeros((max(1, max_decode_calls),), dtype=torch.bool, pin_memory=True)
    copy_stream = torch.npu.Stream(device=next_token.device)
    pending_event = None
    pending_step = None

    for step in range(max_decode_calls):
        logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        cache_position = cache_position + 1
        decode_calls += 1

        hit_eos = (next_token.reshape(-1) == int(eos_token_id)).all()
        eos_ready = torch.npu.current_stream().record_event()
        copy_done = torch.npu.Event()
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(eos_ready)
            cpu_flags[step : step + 1].copy_(hit_eos.reshape(1), non_blocking=True)
            copy_done.record(copy_stream)
        if pending_event is not None:
            pending_event.synchronize()
            if bool(cpu_flags[pending_step].item()):
                eos_detected = True
                eos_step = int(pending_step)
                break
        pending_event = copy_done
        pending_step = int(step)

    if not eos_detected and pending_event is not None:
        pending_event.synchronize()
        if bool(cpu_flags[pending_step].item()):
            eos_detected = True
            eos_step = int(pending_step)

    return DecodeLoopResult(
        ids=torch.cat(generated, dim=1),
        decode_calls=decode_calls,
        eos_detected=eos_detected,
        eos_step=eos_step,
    )


# ---------------------------------------------------------------------------
# TorchAir compile
# ---------------------------------------------------------------------------


def compile_decode_module(
    flat_decode: torch.nn.Module,
    *,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    linear_weight_format: str,
) -> tuple[Any, dict[str, Any]]:
    """Compile the flat decode module into one cached TorchAir GE graph.

    The cache dir is keyed by everything that changes the graph: weight
    format, attention kind, batch size, and cache length."""
    shape_key = f"{linear_weight_format}_{DECODE_ATTENTION}_bs{int(batch_size)}_cache{int(cache_length)}"
    shape_cache_dir = cache_root.expanduser().resolve() / shape_key
    shape_cache_dir.mkdir(parents=True, exist_ok=True)
    compiled_decode = torchair.inference.cache_compile(
        flat_decode.forward,
        config=torchair.CompilerConfig(),
        dynamic=False,
        cache_dir=str(shape_cache_dir),
        ge_cache=True,
    )
    return compiled_decode, {
        "backend": "torchair",
        "torchair_cache_dir": str(shape_cache_dir),
        "torchair_ge_cache": True,
        "compile_api": "torchair.inference.cache_compile",
        "linear_weight_format": linear_weight_format,
        "decode_attention": DECODE_ATTENTION,
    }


@torch.inference_mode()
def run_compiled_lane(
    label: str,
    *,
    flat_decode: torch.nn.Module,
    new_prefill: Callable,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    max_new_tokens: int,
    eos_token_id: int,
) -> tuple[Callable, dict[str, Any]]:
    """Compile the decode graph under `label`, warm it with one call, then
    run and time one full decode loop."""
    compile_start = time.perf_counter()
    compiled_decode, compile_meta = compile_decode_module(
        flat_decode,
        cache_root=cache_root,
        batch_size=batch_size,
        cache_length=cache_length,
        linear_weight_format=label,
    )
    compile_wrapper_s = time.perf_counter() - compile_start

    warm_prefill, warm_next_token = new_prefill()
    _, compile_first_s = timed(
        lambda: compiled_decode(warm_next_token, warm_prefill.next_cache_position, warm_prefill.rope_deltas, *warm_prefill.cache.flat_tensors()),
    )

    prefill, next_token = new_prefill()
    result, decode_s = timed(
        lambda: fixed_cache_decode_loop(
            compiled_decode,
            prefill,
            next_token,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ),
    )
    loop_summary = decode_loop_summary(result, eos_token_id=eos_token_id)
    lane = {
        "compile": compile_meta,
        "compile_wrapper_s": float(compile_wrapper_s),
        "compile_first_call_s": float(compile_first_s),
        "decode_s": float(decode_s),
        "loop": loop_summary,
        "ids": result.ids,
        "tok_per_s_raw": tok_per_s(result.decode_calls, decode_s),
        "tok_per_s_effective": tok_per_s(loop_summary["effective_decode_calls"], decode_s),
    }
    return compiled_decode, lane


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def trim_after_first_eos(ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    values = [int(value) for value in ids[0].detach().cpu().tolist()]
    try:
        first_eos = values.index(int(eos_token_id))
    except ValueError:
        return ids
    return ids[:, : first_eos + 1]


def decode_loop_summary(result: DecodeLoopResult, *, eos_token_id: int) -> dict[str, Any]:
    trimmed = trim_after_first_eos(result.ids, eos_token_id)
    return {
        "generated_new_tokens": int(result.ids.shape[1]),
        "trimmed_new_tokens": int(trimmed.shape[1]),
        "decode_calls": int(result.decode_calls),
        "effective_decode_calls": max(0, int(trimmed.shape[1]) - 1),
        "eos_detected": bool(result.eos_detected),
        "eos_step": None if result.eos_step is None else int(result.eos_step),
    }


@torch.inference_mode()
def compare_decode_logits(
    eager_decode: Callable,
    compiled_decode: Callable,
    eager_prefill,
    compiled_prefill,
    eager_next_token: torch.Tensor,
    compiled_next_token: torch.Tensor,
    *,
    max_new_tokens: int,
):
    """Walk the uncompiled and compiled decode side by side, step for step,
    and track the absolute logit difference."""
    max_abs = 0.0
    mean_abs_sum = 0.0
    compared_steps = 0
    eager_generated = [eager_next_token]
    compiled_generated = [compiled_next_token]
    eager_cache_position = eager_prefill.next_cache_position
    compiled_cache_position = compiled_prefill.next_cache_position
    eager_flat_cache = eager_prefill.cache.flat_tensors()
    compiled_flat_cache = compiled_prefill.cache.flat_tensors()
    for _ in range(max(0, int(max_new_tokens) - 1)):
        eager_logits = eager_decode(eager_next_token, eager_cache_position, eager_prefill.rope_deltas, *eager_flat_cache)
        compiled_logits = compiled_decode(
            compiled_next_token,
            compiled_cache_position,
            compiled_prefill.rope_deltas,
            *compiled_flat_cache,
        )
        diff = (eager_logits.float() - compiled_logits.float()).abs()
        max_abs = max(max_abs, float(diff.max()))
        mean_abs_sum += float(diff.mean())
        compared_steps += 1
        eager_next_token = torch.argmax(eager_logits[:, -1, :].float(), dim=-1, keepdim=True)
        compiled_next_token = torch.argmax(compiled_logits[:, -1, :].float(), dim=-1, keepdim=True)
        eager_generated.append(eager_next_token)
        compiled_generated.append(compiled_next_token)
        eager_cache_position = eager_cache_position + 1
        compiled_cache_position = compiled_cache_position + 1
    mean_abs = mean_abs_sum / compared_steps if compared_steps else 0.0
    return torch.cat(eager_generated, dim=1), torch.cat(compiled_generated, dim=1), max_abs, mean_abs, compared_steps


# ---------------------------------------------------------------------------
# Profiler capture
# ---------------------------------------------------------------------------


@torch.inference_mode()
def profile_compiled_decode(
    *,
    profile_root: Path,
    compiled_decode: Callable,
    new_prefill: Callable,
    max_new_tokens: int,
    eos_token_id: int,
    tokenizer: Tokenizer,
    linear_weight_format: str,
) -> dict[str, Any]:
    """Capture one post-warmup torch_npu profiler trace of compiled decode
    (pipe-utilization metrics), for parse_npu_profile.py."""
    if int(max_new_tokens) >= 16:
        raise ValueError("--profile-dir requires --max-new-tokens < 16 to keep profiler JSON bounded.")

    import torch_npu.profiler as npu_prof

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    profile_dir = profile_root.expanduser().resolve() / f"bench_static_compile_{timestamp}_torchair_{DECODE_ATTENTION}_{max_new_tokens}tok"
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    warm_prefill, warm_next_token = new_prefill()
    _warmup_result, profile_warmup_s = timed(
        lambda: fixed_cache_decode_loop(
            compiled_decode,
            warm_prefill,
            warm_next_token,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ),
    )

    prof_prefill, prof_next_token = new_prefill()
    experimental_config = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )
    torch.npu.synchronize()
    start = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=experimental_config,
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("paddle_ocr_vl.compiled_decode_profile"):
            profile_result = fixed_cache_decode_loop(
                compiled_decode,
                prof_prefill,
                prof_next_token,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        torch.npu.synchronize()
        profiler.step()
    torch.npu.synchronize()
    profile_wall_s = time.perf_counter() - start

    profile_summary = {
        "profile_dir": str(profile_dir),
        "linear_weight_format": linear_weight_format,
        "decode_attention": DECODE_ATTENTION,
        "profile_warmup_s": float(profile_warmup_s),
        "profile_wall_s": float(profile_wall_s),
        "profiled_decode_steps": int(profile_result.decode_calls),
        "loop": decode_loop_summary(profile_result, eos_token_id=eos_token_id),
        "generated_text": tokenizer.decode(profile_result.ids[0].detach().cpu().tolist(), skip_special_tokens=True),
    }
    (profile_dir / "bench_profile_summary.json").write_text(json.dumps(profile_summary, indent=2, default=json_default), encoding="utf-8")
    return profile_summary


# ---------------------------------------------------------------------------
# main: run the lanes in ladder order
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a local model directory.")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--profile-dir", type=Path, default=None, help="Write one post-warmup torch_npu profiler capture of compiled decode.")
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    dtype = DTYPES[args.dtype]

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(args.crop)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt)
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cache_length = int(args.cache_length or (input_ids.shape[1] + args.max_new_tokens))
    decode_steps = max(0, int(args.max_new_tokens) - 1)

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(args.model, dtype=dtype, device=device)
    eos_token_id = int(model.config.eos_token_id)
    flat_decode = model.make_flat_static_decode_module().eval()

    def new_prefill():
        return make_prefill(model, input_ids, attention_mask, pixel_values, image_grid_thw, cache_length=cache_length)

    # Lane 1: eager, growing KV cache.
    eager_ids, eager_decode_s = timed(
        lambda: eager_decode_loop(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    # Lane 2: eager, preallocated fixed KV cache.
    fixed_prefill, fixed_next_token = new_prefill()
    fixed_result, fixed_decode_s = timed(
        lambda: fixed_cache_decode_loop(
            flat_decode,
            fixed_prefill,
            fixed_next_token,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
        ),
    )
    fixed_loop_summary = decode_loop_summary(fixed_result, eos_token_id=eos_token_id)

    # Lane 3: TorchAir-compiled decode, native weight format.
    compiled_native, native_lane = run_compiled_lane(
        "native",
        flat_decode=flat_decode,
        new_prefill=new_prefill,
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
    )

    # Lockstep logit comparison: uncompiled flat module vs compiled graph.
    compare_eager_prefill, compare_eager_next = new_prefill()
    compare_compiled_prefill, compare_compiled_next = new_prefill()
    compare_fixed_ids, compare_compiled_ids, max_abs, mean_abs, compared_steps = compare_decode_logits(
        flat_decode,
        compiled_native,
        compare_eager_prefill,
        compare_compiled_prefill,
        compare_eager_next,
        compare_compiled_next,
        max_new_tokens=args.max_new_tokens,
    )

    # Lane 4: cast decode linear weights to FRACTAL_NZ and recompile. On
    # runtimes that refuse internal formats the cast falls back and the lane
    # is reported unavailable instead of silently re-measuring native.
    torch.npu.synchronize()
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    torch.npu.synchronize()
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start
    nz_available = str(weight_format_meta.get("effective_mode")) == DECODE_LINEAR_WEIGHT_FORMAT
    nz_lane = None
    compiled_for_profile = compiled_native
    profile_weight_format = "native"
    if nz_available:
        compiled_nz, nz_lane = run_compiled_lane(
            DECODE_LINEAR_WEIGHT_FORMAT,
            flat_decode=flat_decode,
            new_prefill=new_prefill,
            cache_root=args.torchair_cache_dir,
            batch_size=int(input_ids.shape[0]),
            cache_length=cache_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        compiled_for_profile = compiled_nz
        profile_weight_format = DECODE_LINEAR_WEIGHT_FORMAT

    profile_summary = None
    if args.profile_dir is not None:
        profile_summary = profile_compiled_decode(
            profile_root=args.profile_dir,
            compiled_decode=compiled_for_profile,
            new_prefill=new_prefill,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            tokenizer=tokenizer,
            linear_weight_format=profile_weight_format,
        )

    native_ids = native_lane["ids"]
    fixed_ids = fixed_result.ids
    eager_trimmed_ids = trim_after_first_eos(eager_ids, eos_token_id)
    fixed_trimmed_ids = trim_after_first_eos(fixed_ids, eos_token_id)
    native_trimmed_ids = trim_after_first_eos(native_ids, eos_token_id)
    matches = {
        "eager_vs_fixed_cache": bool(torch.equal(eager_ids, fixed_ids)),
        "eager_vs_compiled_native": bool(torch.equal(eager_ids, native_ids)),
        "fixed_cache_vs_compiled_native": bool(torch.equal(fixed_ids, native_ids)),
        "compare_loop_fixed_cache_vs_compiled_native": bool(torch.equal(compare_fixed_ids, compare_compiled_ids)),
        "eager_vs_fixed_cache_trimmed": bool(torch.equal(eager_trimmed_ids, fixed_trimmed_ids)),
        "eager_vs_compiled_native_trimmed": bool(torch.equal(eager_trimmed_ids, native_trimmed_ids)),
        "fixed_cache_vs_compiled_native_trimmed": bool(torch.equal(fixed_trimmed_ids, native_trimmed_ids)),
    }
    if nz_lane is not None:
        nz_ids = nz_lane["ids"]
        matches["compiled_native_vs_compiled_nz"] = bool(torch.equal(native_ids, nz_ids))
        matches["compiled_native_vs_compiled_nz_trimmed"] = bool(
            torch.equal(native_trimmed_ids, trim_after_first_eos(nz_ids, eos_token_id))
        )

    ladder = [
        ("eager", tok_per_s(decode_steps, eager_decode_s)),
        ("eager_fixed_cache", tok_per_s(fixed_result.decode_calls, fixed_decode_s)),
        ("compiled_native", native_lane["tok_per_s_raw"]),
    ]
    if nz_lane is not None:
        ladder.append(("compiled_nz", nz_lane["tok_per_s_raw"]))

    def lane_summary(lane: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in lane.items() if key != "ids"}

    summary = {
        "backend": "torchair",
        "device": str(device),
        "dtype": str(dtype),
        "decode_attention": DECODE_ATTENTION,
        "eos_token_id": eos_token_id,
        "linear_weight_format": weight_format_meta,
        "nz_available": bool(nz_available),
        "cache_update": "prefill_slice_decode_npu_scatter",
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(args.max_new_tokens),
        "requested_decode_steps": int(decode_steps),
        "cache_length": int(cache_length),
        "lanes": {
            "eager": {
                "decode_s": float(eager_decode_s),
                "tok_per_s_raw": tok_per_s(decode_steps, eager_decode_s),
            },
            "eager_fixed_cache": {
                "decode_s": float(fixed_decode_s),
                "loop": fixed_loop_summary,
                "tok_per_s_raw": tok_per_s(fixed_result.decode_calls, fixed_decode_s),
                "tok_per_s_effective": tok_per_s(fixed_loop_summary["effective_decode_calls"], fixed_decode_s),
            },
            "compiled_native": lane_summary(native_lane),
        },
        "matches": matches,
        "logit_diff_fixed_cache_vs_compiled_native_decode": {
            "steps": int(compared_steps),
            "max_abs": float(max_abs),
            "mean_abs": float(mean_abs),
        },
        "tok_per_s_ladder": {name: value for name, value in ladder},
        "texts": {
            "eager": tokenizer.decode(eager_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
            "eager_fixed_cache": tokenizer.decode(fixed_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
            "compiled_native": tokenizer.decode(native_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
        },
    }
    if nz_lane is not None:
        summary["lanes"]["compiled_nz"] = lane_summary(nz_lane)
        summary["texts"]["compiled_nz"] = tokenizer.decode(nz_lane["ids"][0].detach().cpu().tolist(), skip_special_tokens=True)
    if profile_summary is not None:
        summary["profile"] = profile_summary

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return

    print(f"backend={summary['backend']} device={summary['device']} dtype={summary['dtype']} decode_attention={summary['decode_attention']}")
    print(f"prompt_tokens={summary['prompt_tokens']} generated_tokens={summary['generated_tokens']} requested_decode_steps={summary['requested_decode_steps']} cache_length={summary['cache_length']}")
    print("matches=" + json.dumps(matches, sort_keys=True))
    print("logit_diff_fixed_cache_vs_compiled_native_decode=" + json.dumps(summary["logit_diff_fixed_cache_vs_compiled_native_decode"], sort_keys=True))
    print("compile_s=" + json.dumps({"wrapper": native_lane["compile_wrapper_s"], "first_call": native_lane["compile_first_call_s"]}, sort_keys=True))
    print("tok_per_s ladder:")
    for name, value in ladder:
        print(f"  {name:<17} {value:8.2f}")
    if not nz_available:
        reason = weight_format_meta.get("fallback_reason") or weight_format_meta.get("effective_mode")
        print(f"  compiled_nz       unavailable ({reason})")
    if profile_summary is not None:
        print("profile=" + json.dumps(profile_summary, sort_keys=True, default=json_default))
    print(f"compiled_text={summary['texts']['compiled_native']!r}")


if __name__ == "__main__":
    main()
