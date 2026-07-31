#!/usr/bin/env python3
"""Benchmark experiment-3 static-compiled decode as a tok/s ladder.

Every rung is the same measurement — prefill into a fixed KV cache, then a
fixed number of decode steps — applied to a different decode_fn:

  eager            decode_fn = the flat decode module, uncompiled
  compiled_native  decode_fn = the same module TorchAir-compiled
  compiled_nz      decode_fn = the same graph after casting decode linear
                   weights to FRACTAL_NZ; reported unavailable (with the
                   runtime's reason) when the cast falls back

Every lane runs once untimed before its timed run, so one-time costs (GE
graph build/load for compiled lanes, kernel warm-start for eager) stay out
of the rates. Correctness rides along: all lanes must produce identical
token ids, and the warmup runs record per-step logits so the compiled
graph's drift from the uncompiled module is bounded without extra inference.

Reading map, top to bottom: TorchAir compile, profiler context, the greedy
decode loop (run_once) and its measurement (measure_inference), correctness,
report, then main() runs the rungs in ladder order.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import itertools
import json
import shutil
import time
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
# Profiler context: a no-op unless --profile-dir is set
# ---------------------------------------------------------------------------


def profiler_context(profile_dir: Path | None, *, steps: int):
    """Context for the timed decode run: a do-nothing context normally, the
    torch_npu profiler (pipe-utilization metrics, for parse_npu_profile.py)
    when a directory is given. One profiler step is one decode step — the
    decode loop calls profiler.step() every iteration and `steps` sizes the
    schedule — so the trace is bounded per step. Profiling the timed run
    itself means no extra inference, but the profiled lane's tok/s carries
    the profiler overhead."""
    if profile_dir is None:
        return contextlib.nullcontext()

    import torch_npu.profiler as npu_prof

    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    return npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=npu_prof.schedule(wait=0, warmup=0, active=int(steps), repeat=1),
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=npu_prof.ProfilerLevel.Level1,
            aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
            export_type=npu_prof.ExportType.Text,
        ),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    )


# ---------------------------------------------------------------------------
# Measurement: identical for every rung of the ladder
# ---------------------------------------------------------------------------


def run_once(decode_fn: Callable, prefill, *, max_new_tokens: int, logits_log: list | None = None, profiler=None) -> torch.Tensor:
    # Uniform greedy loop: pick a token from the current logits, then one
    # decode step for the next logits. The prefill's logits seed the loop.
    logits = prefill.logits
    cache_position = prefill.next_cache_position
    flat_cache = prefill.cache.flat_tensors()
    generated = []
    for _ in range(max(0, int(max_new_tokens))):
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
        if logits_log is not None:
            logits_log.append(logits.float().cpu())
        if profiler is not None:
            profiler.step()
        cache_position = cache_position + 1
    return torch.cat(generated, dim=1)


@torch.inference_mode()
def measure_inference(
    decode_fn: Callable, *, new_prefill: Callable, max_new_tokens: int, profile_dir: Path | None = None
) -> dict[str, Any]:
    """Every rung shares this: one untimed warmup run, then one timed run,
    decode steps only. The only difference between rungs is which decode_fn
    is passed in.

    For a compiled decode_fn the warmup absorbs the GE graph build/load;
    for the uncompiled module it is just a warm pass. warmup_s is reported
    so that one-time cost stays visible."""
    warmup_logits: list[torch.Tensor] = []
    warm_prefill = new_prefill()
    _, warmup_s = timed(lambda: run_once(decode_fn, warm_prefill, max_new_tokens=max_new_tokens, logits_log=warmup_logits))
    prefill = new_prefill()
    with profiler_context(profile_dir, steps=max_new_tokens) as profiler:
        ids, decode_s = timed(lambda: run_once(decode_fn, prefill, max_new_tokens=max_new_tokens, profiler=profiler))
    decode_calls = int(ids.shape[1])
    result = {
        "ids": ids,
        "warmup_logits": torch.cat(warmup_logits, dim=1),
        "warmup_s": float(warmup_s),
        "decode_s": float(decode_s),
        "decode_calls": decode_calls,
        "tok_per_s": decode_calls / decode_s if decode_s > 0 else float("inf"),
    }
    if profile_dir is not None:
        result["profile_dir"] = str(profile_dir)
    return result


def run_static_compile(
    flat_decode: torch.nn.Module,
    *,
    linear_weight_format: str,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    new_prefill: Callable,
    max_new_tokens: int,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Rungs 2 and 3: TorchAir-compile the flat decode module, then run the
    exact same measurement as the uncompiled lane."""
    compiled_decode, cache_dir = compile_decode_module(
        flat_decode,
        cache_root=cache_root,
        batch_size=batch_size,
        cache_length=cache_length,
        linear_weight_format=linear_weight_format,
    )
    lane = measure_inference(compiled_decode, new_prefill=new_prefill, max_new_tokens=max_new_tokens, profile_dir=profile_dir)
    lane["torchair_cache_dir"] = str(cache_dir)
    return lane

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
) -> tuple[Any, Path]:
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
    return compiled_decode, shape_cache_dir

def apply_nz_weight_format(model) -> dict[str, Any]:
    """Rung 3 setup: cast the decode linear weights to FRACTAL_NZ, in place —
    only run after the native lanes are measured. On runtimes that refuse
    internal formats the cast falls back; the returned metadata says why."""
    torch.npu.synchronize()
    start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    torch.npu.synchronize()
    weight_format_meta["setup_s"] = time.perf_counter() - start
    return weight_format_meta


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def all_pairs_matches(lane_ids: dict[str, torch.Tensor]) -> dict[str, bool]:
    """Every lane runs the same fixed number of decode steps, so the token
    ids must match exactly, pairwise."""
    return {
        f"{a}_vs_{b}": bool(torch.equal(lane_ids[a], lane_ids[b]))
        for a, b in itertools.combinations(lane_ids, 2)
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return
    print(f"backend={summary['backend']} device={summary['device']} dtype={summary['dtype']} decode_attention={summary['decode_attention']}")
    print(f"prompt_tokens={summary['prompt_tokens']} generated_tokens={summary['generated_tokens']} cache_length={summary['cache_length']}")
    print("matches=" + json.dumps(summary["matches"], sort_keys=True))
    print("logit_diff_eager_vs_compiled_native=" + json.dumps(summary["logit_diff_eager_vs_compiled_native"], sort_keys=True))
    print("tok_per_s ladder (warmup_s in parentheses):")
    for name, lane in summary["lanes"].items():
        print(f"  {name:<17} {lane['tok_per_s']:8.2f}  ({lane['warmup_s']:.2f})")
    if not summary["nz_available"]:
        reason = summary["linear_weight_format"].get("fallback_reason") or summary["linear_weight_format"].get("effective_mode")
        print(f"  compiled_nz       unavailable ({reason})")
    profile_dirs = [lane["profile_dir"] for lane in summary["lanes"].values() if "profile_dir" in lane]
    if profile_dirs:
        print("profile_dirs=" + json.dumps(profile_dirs))
    print(f"compiled_text={summary['texts']['compiled_native']!r}")


# ---------------------------------------------------------------------------
# main: run the rungs in ladder order
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a local model directory.")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Also capture a torch_npu profiler trace of each fixed-cache lane's timed run (one subdirectory per lane; profiled tok/s carries profiler overhead).",
    )
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()
    if args.profile_dir is not None and int(args.max_new_tokens) >= 16:
        parser.error("--profile-dir requires --max-new-tokens < 16 to keep profiler output bounded.")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()

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
    flat_decode = model.make_flat_static_decode_module().eval()

    def new_prefill():
        return model.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )

    def profile_dir_for(lane_name: str) -> Path | None:
        if args.profile_dir is None:
            return None
        return args.profile_dir.expanduser().resolve() / lane_name

    lanes: dict[str, dict[str, Any]] = {}

    # Rung 1: the flat decode module, uncompiled (plain eager execution).
    lanes["eager"] = measure_inference(
        flat_decode,
        new_prefill=new_prefill,
        max_new_tokens=args.max_new_tokens,
        profile_dir=profile_dir_for("eager"),
    )

    # Rung 2: the same measurement on the TorchAir-compiled module.
    lanes["compiled_native"] = run_static_compile(
        flat_decode,
        linear_weight_format="native",
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
        new_prefill=new_prefill,
        max_new_tokens=args.max_new_tokens,
        profile_dir=profile_dir_for("compiled_native"),
    )

    # Logit drift, uncompiled module vs compiled graph: the warmup runs above
    # already recorded per-step logits, so this is just a tensor diff.
    logit_diff = (lanes["eager"]["warmup_logits"] - lanes["compiled_native"]["warmup_logits"]).abs()

    # Rung 3: FRACTAL_NZ decode weights, then the same compile + measurement.
    # When the cast falls back the rung is reported unavailable instead of
    # silently re-measuring native.
    weight_format_meta = apply_nz_weight_format(model)
    nz_available = str(weight_format_meta.get("effective_mode")) == DECODE_LINEAR_WEIGHT_FORMAT
    if nz_available:
        lanes["compiled_nz"] = run_static_compile(
            flat_decode,
            linear_weight_format=DECODE_LINEAR_WEIGHT_FORMAT,
            cache_root=args.torchair_cache_dir,
            batch_size=int(input_ids.shape[0]),
            cache_length=cache_length,
            new_prefill=new_prefill,
            max_new_tokens=args.max_new_tokens,
            profile_dir=profile_dir_for("compiled_nz"),
        )

    lane_ids = {name: lane["ids"] for name, lane in lanes.items()}
    matches = all_pairs_matches(lane_ids)

    summary = {
        "backend": "torchair",
        "device": str(device),
        "dtype": str(dtype),
        "decode_attention": DECODE_ATTENTION,
        "linear_weight_format": weight_format_meta,
        "nz_available": bool(nz_available),
        "cache_update": "prefill_slice_decode_npu_scatter",
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(args.max_new_tokens),
        "cache_length": int(cache_length),
        "lanes": {name: {key: value for key, value in lane.items() if key not in ("ids", "warmup_logits")} for name, lane in lanes.items()},
        "matches": matches,
        "logit_diff_eager_vs_compiled_native": {
            "steps": int(logit_diff.shape[1]),
            "max_abs": float(logit_diff.max()),
            "mean_abs": float(logit_diff.mean()),
        },
        "tok_per_s_ladder": {name: lane["tok_per_s"] for name, lane in lanes.items()},
        "texts": {
            name: tokenizer.decode(ids[0].detach().cpu().tolist(), skip_special_tokens=True) for name, ids in lane_ids.items()
        },
    }
    report(summary, as_json=args.json)


if __name__ == "__main__":
    main()
