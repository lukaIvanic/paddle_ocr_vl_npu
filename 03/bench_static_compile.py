#!/usr/bin/env python3
"""Benchmark experiment-3 static-cache compiled decode.

Reports output matching, decode-logit diffs, and steady decode tok/s for:
- dynamic eager decode with growing KV cache
- static eager decode with fixed KV cache
- static compiled decode with torch.compile(fullgraph=True, dynamic=False)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import LocalPaddleOCRVLForConditionalGeneration, _resolve_model_dir
from probe_static_compile import DEFAULT_TORCHAIR_CACHE_DIR, compile_decode_module, maybe_sync
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    preprocess_image,
    resolve_device,
)


PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def timed(device: torch.device, fn: Callable):
    maybe_sync(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync(device)
    return result, time.perf_counter() - start


@torch.inference_mode()
def dynamic_decode_loop(
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    max_new_tokens: int,
):
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
def static_flat_decode_loop(
    decode_fn: Callable,
    prefill,
    next_token: torch.Tensor,
    *,
    max_new_tokens: int,
):
    generated = [next_token]
    cache_position = prefill.next_cache_position
    flat_cache = prefill.cache.flat_tensors()
    last_logits = None
    for _ in range(max(0, int(max_new_tokens) - 1)):
        last_logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
        next_token = torch.argmax(last_logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        cache_position = cache_position + 1
    return torch.cat(generated, dim=1), last_logits


@torch.inference_mode()
def make_static_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    cache_length: int,
):
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
def compare_static_logits(
    eager_decode: Callable,
    compiled_decode: Callable,
    eager_prefill,
    compiled_prefill,
    eager_next_token: torch.Tensor,
    compiled_next_token: torch.Tensor,
    *,
    max_new_tokens: int,
):
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


def tok_per_s(tokens: int, seconds: float) -> float:
    return float(tokens) / float(seconds) if seconds > 0 else float("inf")


def npu_profiler_config(metric: str):
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
    )


def make_profile_run_dir(root: Path, *, backend: str, max_new_tokens: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"bench_static_compile_{timestamp}_{backend}_{max_new_tokens}tok"


@torch.inference_mode()
def profile_compiled_decode(
    *,
    args: argparse.Namespace,
    model: LocalPaddleOCRVLForConditionalGeneration,
    compiled_decode: Callable,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    cache_length: int,
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("--profile-dir requires --device npu:0; torch_npu profiler is NPU-only.")
    if args.backend != "torchair":
        raise ValueError("--profile-dir requires --backend torchair so the profile captures compiled NPU decode.")
    if int(args.max_new_tokens) >= 16:
        raise ValueError("--profile-dir requires --max-new-tokens < 16 to keep profiler JSON bounded.")

    import torch_npu.profiler as npu_prof

    profile_dir = make_profile_run_dir(args.profile_dir.expanduser().resolve(), backend=args.backend, max_new_tokens=int(args.max_new_tokens))
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    warm_prefill, warm_next_token = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    (_, _), profile_warmup_s = timed(
        device,
        lambda: static_flat_decode_loop(compiled_decode, warm_prefill, warm_next_token, max_new_tokens=args.max_new_tokens),
    )

    prof_prefill, prof_next_token = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    maybe_sync(device)
    start = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(args.profile_metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("paddle_ocr_vl.compiled_decode_profile"):
            profile_ids, _last_logits = static_flat_decode_loop(
                compiled_decode,
                prof_prefill,
                prof_next_token,
                max_new_tokens=args.max_new_tokens,
            )
        maybe_sync(device)
        profiler.step()
    maybe_sync(device)
    profile_wall_s = time.perf_counter() - start

    profile_summary = {
        "profile_dir": str(profile_dir),
        "metric": args.profile_metric,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": False,
        "profile_warmup_s": float(profile_warmup_s),
        "profile_wall_s": float(profile_wall_s),
        "profiled_generated_tokens": int(args.max_new_tokens),
        "profiled_decode_steps": max(0, int(args.max_new_tokens) - 1),
        "generated_ids": [int(v) for v in profile_ids[0].detach().cpu().tolist()],
        "generated_text": tokenizer.decode(profile_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
    }
    (profile_dir / "bench_profile_summary.json").write_text(json.dumps(profile_summary, indent=2, default=json_default), encoding="utf-8")
    return profile_summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png")
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--backend", default="eager", choices=["eager", "aot_eager", "inductor", "default", "torchair"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--profile-dir", type=Path, default=None, help="Write one post-warmup torch_npu profiler capture for compiled decode.")
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model)
    crop = Path(args.crop)
    if not crop.exists():
        crop = Path(__file__).resolve().parents[1] / args.crop
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(crop, pre_cfg)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt, merge_size=int(pre_cfg["merge_size"]))
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cache_length = int(args.cache_length or (input_ids.shape[1] + args.max_new_tokens))
    decode_steps = max(0, int(args.max_new_tokens) - 1)

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    flat_decode = model.make_flat_static_decode_module().eval()

    maybe_sync(device)
    compile_start = time.perf_counter()
    compiled_decode, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=args.backend,
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
    )
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_start

    warm_prefill, warm_next_token = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    _, compile_first_s = timed(
        device,
        lambda: compiled_decode(warm_next_token, warm_prefill.next_cache_position, warm_prefill.rope_deltas, *warm_prefill.cache.flat_tensors()),
    )

    dynamic_ids, dynamic_decode_s = timed(
        device,
        lambda: dynamic_decode_loop(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    static_prefill, static_next_token = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    (static_ids, _), static_decode_s = timed(
        device,
        lambda: static_flat_decode_loop(flat_decode, static_prefill, static_next_token, max_new_tokens=args.max_new_tokens),
    )

    compiled_prefill, compiled_next_token = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    (compiled_ids, _), compiled_decode_s = timed(
        device,
        lambda: static_flat_decode_loop(compiled_decode, compiled_prefill, compiled_next_token, max_new_tokens=args.max_new_tokens),
    )

    compare_eager_prefill, compare_eager_next = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    compare_compiled_prefill, compare_compiled_next = make_static_prefill(
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        cache_length=cache_length,
    )
    compare_static_ids, compare_compiled_ids, max_abs, mean_abs, compared_steps = compare_static_logits(
        flat_decode,
        compiled_decode,
        compare_eager_prefill,
        compare_compiled_prefill,
        compare_eager_next,
        compare_compiled_next,
        max_new_tokens=args.max_new_tokens,
    )

    profile_summary = None
    if args.profile_dir is not None:
        profile_summary = profile_compiled_decode(
            args=args,
            model=model,
            compiled_decode=compiled_decode,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            tokenizer=tokenizer,
            device=device,
        )

    summary = {
        "backend": args.backend,
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": args.npu_jit_compile,
        "compile": compile_meta,
        "cache_update": "prefill_slice_decode_npu_scatter",
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(args.max_new_tokens),
        "decode_steps": int(decode_steps),
        "cache_length": int(cache_length),
        "matches": {
            "dynamic_vs_static_eager": bool(torch.equal(dynamic_ids, static_ids)),
            "dynamic_vs_compiled": bool(torch.equal(dynamic_ids, compiled_ids)),
            "static_eager_vs_compiled": bool(torch.equal(static_ids, compiled_ids)),
            "compare_loop_static_vs_compiled": bool(torch.equal(compare_static_ids, compare_compiled_ids)),
        },
        "logit_diff_static_eager_vs_compiled_decode": {
            "steps": int(compared_steps),
            "max_abs": float(max_abs),
            "mean_abs": float(mean_abs),
        },
        "timing_s": {
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_s),
            "dynamic_decode": float(dynamic_decode_s),
            "static_eager_decode": float(static_decode_s),
            "compiled_decode": float(compiled_decode_s),
        },
        "tok_per_s": {
            "dynamic_decode": tok_per_s(decode_steps, dynamic_decode_s),
            "static_eager_decode": tok_per_s(decode_steps, static_decode_s),
            "compiled_decode": tok_per_s(decode_steps, compiled_decode_s),
        },
        "texts": {
            "dynamic": tokenizer.decode(dynamic_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
            "static_eager": tokenizer.decode(static_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
            "compiled": tokenizer.decode(compiled_ids[0].detach().cpu().tolist(), skip_special_tokens=True),
        },
    }
    if profile_summary is not None:
        summary["profile"] = profile_summary

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return

    print(f"backend={summary['backend']} device={summary['device']} dtype={summary['dtype']} npu_jit_compile={summary['npu_jit_compile']}")
    print(f"cache_update={summary['cache_update']}")
    print(f"prompt_tokens={summary['prompt_tokens']} generated_tokens={summary['generated_tokens']} decode_steps={summary['decode_steps']} cache_length={summary['cache_length']}")
    print("matches=" + json.dumps(summary["matches"], sort_keys=True))
    print("logit_diff_static_eager_vs_compiled_decode=" + json.dumps(summary["logit_diff_static_eager_vs_compiled_decode"], sort_keys=True))
    print("timing_s=" + json.dumps(summary["timing_s"], sort_keys=True))
    print("tok_per_s=" + json.dumps(summary["tok_per_s"], sort_keys=True))
    if profile_summary is not None:
        print("profile=" + json.dumps(profile_summary, sort_keys=True, default=json_default))
    print(f"compiled_text={summary['texts']['compiled']!r}")


if __name__ == "__main__":
    main()
