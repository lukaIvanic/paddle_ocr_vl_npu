#!/usr/bin/env python3
"""Benchmark experiment-3 static-compiled decode as a tok/s ladder.

Every rung after the first is the same measurement — prefill into a fixed
KV cache, then a fixed number of decode steps — applied to a different
decode_fn:

  eager              the modeling file's own generate loop, growing KV cache
                     (the exp02 reference; the one rung with different code)
  eager_fixed_cache  decode_fn = the flat decode module, uncompiled
  compiled_native    decode_fn = the same module TorchAir-compiled
  compiled_nz        decode_fn = the same graph after casting decode linear
                     weights to FRACTAL_NZ; reported unavailable (with the
                     runtime's reason) when the cast falls back

Every lane runs once untimed before its timed run, so one-time costs (GE
graph build/load for compiled lanes, kernel warm-start for eager) stay out
of the rates. Correctness rides along: all lanes must produce identical
token ids, and the compiled graph is walked in lockstep with the uncompiled
module to bound per-step logit drift.

Reading map, top to bottom: decode loop, TorchAir compile, measurement,
correctness, profiler capture, then main() runs the rungs in ladder order.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu  # noqa: F401  (must precede torchair: it registers the module)

try:
    import torchair
    import torchair.inference
except ModuleNotFoundError:  # some torch_npu builds only vendor torchair internally
    from torch_npu.dynamo import torchair

    importlib.import_module(f"{torchair.__name__}.inference")

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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------


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
) -> torch.Tensor:
    """Decode a fixed number of steps against the fixed cache with
    `decode_fn` (the flat module, or its compiled graph). No EOS stop:
    the fixed step count keeps the measured rate a pure per-step cost."""
    generated = [next_token]
    cache_position = prefill.next_cache_position
    flat_cache = prefill.cache.flat_tensors()
    for _ in range(max(0, int(max_new_tokens) - 1)):
        logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        cache_position = cache_position + 1
    return torch.cat(generated, dim=1)


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


# ---------------------------------------------------------------------------
# Measurement: identical for every fixed-cache lane, compiled or not
# ---------------------------------------------------------------------------


@torch.inference_mode()
def measure_decode(decode_fn: Callable, *, new_prefill: Callable, max_new_tokens: int) -> dict[str, Any]:
    """One untimed warmup run, then one timed run, decode steps only.

    For a compiled decode_fn the warmup absorbs the GE graph build/load;
    for the uncompiled module it is just a warm pass. warmup_s is reported
    so that one-time cost stays visible."""
    prefill, next_token = new_prefill()
    _, warmup_s = timed(lambda: fixed_cache_decode_loop(decode_fn, prefill, next_token, max_new_tokens=max_new_tokens))
    prefill, next_token = new_prefill()
    ids, decode_s = timed(lambda: fixed_cache_decode_loop(decode_fn, prefill, next_token, max_new_tokens=max_new_tokens))
    decode_calls = max(0, int(ids.shape[1]) - 1)
    return {
        "ids": ids,
        "warmup_s": float(warmup_s),
        "decode_s": float(decode_s),
        "decode_calls": decode_calls,
        "tok_per_s": decode_calls / decode_s if decode_s > 0 else float("inf"),
    }


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


def all_pairs_matches(lane_ids: dict[str, torch.Tensor], eos_token_id: int) -> dict[str, bool]:
    """Exact and EOS-trimmed token-id equality for every lane pair. The
    eager lane stops at EOS while fixed-step lanes run on, so when EOS
    fires inside the token budget only the trimmed comparisons must hold."""
    matches: dict[str, bool] = {}
    for a, b in itertools.combinations(lane_ids, 2):
        matches[f"{a}_vs_{b}"] = bool(torch.equal(lane_ids[a], lane_ids[b]))
        matches[f"{a}_vs_{b}_trimmed"] = bool(
            torch.equal(trim_after_first_eos(lane_ids[a], eos_token_id), trim_after_first_eos(lane_ids[b], eos_token_id))
        )
    return matches


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
    _, profile_warmup_s = timed(
        lambda: fixed_cache_decode_loop(compiled_decode, warm_prefill, warm_next_token, max_new_tokens=max_new_tokens),
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
            profile_ids = fixed_cache_decode_loop(compiled_decode, prof_prefill, prof_next_token, max_new_tokens=max_new_tokens)
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
        "profiled_decode_steps": max(0, int(profile_ids.shape[1]) - 1),
        "generated_text": tokenizer.decode(profile_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
    }
    (profile_dir / "bench_profile_summary.json").write_text(json.dumps(profile_summary, indent=2, default=json_default), encoding="utf-8")
    return profile_summary


# ---------------------------------------------------------------------------
# main: run the rungs in ladder order
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

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(args.model, dtype=dtype, device=device)
    eos_token_id = int(model.config.eos_token_id)
    flat_decode = model.make_flat_static_decode_module().eval()

    def new_prefill():
        return make_prefill(model, input_ids, attention_mask, pixel_values, image_grid_thw, cache_length=cache_length)

    def run_eager():
        return model.generate_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=args.max_new_tokens,
        )

    lanes: dict[str, dict[str, Any]] = {}

    # Rung 1: the exp02 eager path, exactly as the modeling file implements
    # it (growing KV cache; its generate loop stops at EOS on the host).
    _, eager_warmup_s = timed(run_eager)
    eager_ids, eager_decode_s = timed(run_eager)
    eager_decode_calls = max(0, int(eager_ids.shape[1]) - 1)
    lanes["eager"] = {
        "ids": eager_ids,
        "warmup_s": float(eager_warmup_s),
        "decode_s": float(eager_decode_s),
        "decode_calls": eager_decode_calls,
        "tok_per_s": eager_decode_calls / eager_decode_s if eager_decode_s > 0 else float("inf"),
    }

    # Rung 2: same measurement, decode_fn = the uncompiled flat module.
    lanes["eager_fixed_cache"] = measure_decode(flat_decode, new_prefill=new_prefill, max_new_tokens=args.max_new_tokens)

    # Rung 3: same measurement, decode_fn = the TorchAir-compiled module.
    compiled_native, native_compile_meta = compile_decode_module(
        flat_decode,
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
        linear_weight_format="native",
    )
    lanes["compiled_native"] = measure_decode(compiled_native, new_prefill=new_prefill, max_new_tokens=args.max_new_tokens)
    lanes["compiled_native"]["compile"] = native_compile_meta

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

    # Rung 4: cast decode linear weights to FRACTAL_NZ and recompile. On
    # runtimes that refuse internal formats the cast falls back and the rung
    # is reported unavailable instead of silently re-measuring native.
    torch.npu.synchronize()
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    torch.npu.synchronize()
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start
    nz_available = str(weight_format_meta.get("effective_mode")) == DECODE_LINEAR_WEIGHT_FORMAT
    compiled_for_profile = compiled_native
    profile_weight_format = "native"
    if nz_available:
        compiled_nz, nz_compile_meta = compile_decode_module(
            flat_decode,
            cache_root=args.torchair_cache_dir,
            batch_size=int(input_ids.shape[0]),
            cache_length=cache_length,
            linear_weight_format=DECODE_LINEAR_WEIGHT_FORMAT,
        )
        lanes["compiled_nz"] = measure_decode(compiled_nz, new_prefill=new_prefill, max_new_tokens=args.max_new_tokens)
        lanes["compiled_nz"]["compile"] = nz_compile_meta
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

    lane_ids = {name: lane["ids"] for name, lane in lanes.items()}
    matches = all_pairs_matches(lane_ids, eos_token_id)
    matches["compare_loop_fixed_cache_vs_compiled_native"] = bool(torch.equal(compare_fixed_ids, compare_compiled_ids))

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
        "cache_length": int(cache_length),
        "lanes": {name: {key: value for key, value in lane.items() if key != "ids"} for name, lane in lanes.items()},
        "matches": matches,
        "logit_diff_fixed_cache_vs_compiled_native_decode": {
            "steps": int(compared_steps),
            "max_abs": float(max_abs),
            "mean_abs": float(mean_abs),
        },
        "tok_per_s_ladder": {name: lane["tok_per_s"] for name, lane in lanes.items()},
        "texts": {
            name: tokenizer.decode(ids[0].detach().cpu().tolist(), skip_special_tokens=True) for name, ids in lane_ids.items()
        },
    }
    if profile_summary is not None:
        summary["profile"] = profile_summary

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return

    print(f"backend={summary['backend']} device={summary['device']} dtype={summary['dtype']} decode_attention={summary['decode_attention']}")
    print(f"prompt_tokens={summary['prompt_tokens']} generated_tokens={summary['generated_tokens']} cache_length={summary['cache_length']}")
    print("matches=" + json.dumps(matches, sort_keys=True))
    print("logit_diff_fixed_cache_vs_compiled_native_decode=" + json.dumps(summary["logit_diff_fixed_cache_vs_compiled_native_decode"], sort_keys=True))
    print("tok_per_s ladder (warmup_s in parentheses):")
    for name, lane in lanes.items():
        print(f"  {name:<17} {lane['tok_per_s']:8.2f}  ({lane['warmup_s']:.2f})")
    if not nz_available:
        reason = weight_format_meta.get("fallback_reason") or weight_format_meta.get("effective_mode")
        print(f"  compiled_nz       unavailable ({reason})")
    if profile_summary is not None:
        print("profile=" + json.dumps(profile_summary, sort_keys=True, default=json_default))
    print(f"compiled_text={summary['texts']['compiled_native']!r}")


if __name__ == "__main__":
    main()
