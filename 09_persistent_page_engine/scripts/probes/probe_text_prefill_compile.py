#!/usr/bin/env python3
"""Validate compiled text prefill KV mutation, generation parity, and speed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    _resolve_model_dir,
)
from paddleocr_vl.model.preprocessing import (
    build_inputs,
    load_preprocessor_config,
    preprocess_pil_image,
)
from paddleocr_vl.model.text_prefill import TextPrefillRuntime
from utils.timing import synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", type=Path, required=True)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--model", default="/workspace/models/PaddleOCR-VL-1.6")
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=("fp16", "float16", "bf16", "bfloat16"),
    )
    parser.add_argument("--bucket", type=int, default=192)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--torchair-cache-dir",
        type=Path,
        default=Path(".runtime_cache/09_persistent_page_engine_text_torchair_probe"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def allocate_cache(
    model: LocalPaddleOCRVLForConditionalGeneration,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> LocalPaddleOCRVLStaticCache:
    return model.allocate_static_cache(
        batch_size=1,
        cache_length=cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )


def compare_valid_cache_prefixes(
    eager: LocalPaddleOCRVLStaticCache,
    compiled: LocalPaddleOCRVLStaticCache,
    *,
    real_seq_len: int,
) -> dict[str, object]:
    tensors = []
    all_finite = True
    global_max_abs = 0.0
    weighted_abs_sum = 0.0
    total_values = 0
    for index, (eager_tensor, compiled_tensor) in enumerate(
        zip(eager.flat_tensors(), compiled.flat_tensors())
    ):
        eager_prefix = eager_tensor[:, :, :real_seq_len]
        compiled_prefix = compiled_tensor[:, :, :real_seq_len]
        difference = (compiled_prefix.float() - eager_prefix.float()).abs()
        max_abs = float(difference.max().item())
        abs_sum = float(difference.sum().item())
        values = int(difference.numel())
        finite = bool(torch.isfinite(compiled_prefix).all().item())
        all_finite &= finite
        global_max_abs = max(global_max_abs, max_abs)
        weighted_abs_sum += abs_sum
        total_values += values
        tensors.append(
            {
                "flat_index": index,
                "kind": "key" if index < len(eager.key_caches) else "value",
                "layer": index % len(eager.key_caches),
                "max_abs": max_abs,
                "mean_abs": abs_sum / values,
                "all_finite": finite,
            }
        )
    return {
        "all_finite": all_finite,
        "global_max_abs": global_max_abs,
        "global_mean_abs": weighted_abs_sum / total_values,
        "tensors": tensors,
    }


@torch.inference_mode()
def generate_from_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    *,
    cache: LocalPaddleOCRVLStaticCache,
    first_hidden_state: torch.Tensor,
    cache_position: int,
    rope_deltas: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    logits = model.lm_head(first_hidden_state)
    next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    token_ids = [int(next_token.item())]
    eos_token_id = int(model.config.eos_token_id)
    position = torch.tensor([cache_position], device=next_token.device, dtype=torch.int64)
    for _ in range(max(0, max_new_tokens - 1)):
        if token_ids[-1] == eos_token_id:
            break
        outputs = model.forward_static_decode(
            input_ids=next_token,
            cache=cache,
            cache_position=position,
            rope_deltas=rope_deltas,
            logits_to_keep=1,
        )
        next_token = torch.argmax(
            outputs.logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        token_ids.append(int(next_token.item()))
        position = position + 1
    return token_ids


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    model_dir = _resolve_model_dir(args.model)
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("The text-prefill probe requires an available NPU")
    if args.dtype in {"fp16", "float16"}:
        dtype = torch.float16
    elif args.dtype in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported dtype: {args.dtype}")
    torch.npu.set_compile_mode(jit_compile=False)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    preprocessor_config = load_preprocessor_config(model_dir)
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    ).eval()

    with Image.open(args.crop) as opened:
        crop = opened.convert("RGB")
    pixel_values, image_grid_thw = preprocess_pil_image(crop, preprocessor_config)
    input_ids, attention_mask = build_inputs(
        tokenizer,
        image_grid_thw,
        args.prompt,
        merge_size=int(preprocessor_config["merge_size"]),
    )
    position_ids_cpu, rope_deltas_cpu = model.get_rope_index(
        input_ids,
        image_grid_thw,
        attention_mask,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids_cpu.to(device)
    rope_deltas = rope_deltas_cpu.to(device)
    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
    inputs_embeds = model.build_inputs_embeds(
        input_ids,
        pixel_values,
        image_grid_thw,
    )
    real_seq_len = int(inputs_embeds.shape[1])
    if real_seq_len > args.bucket:
        raise ValueError(
            f"real prompt length {real_seq_len} exceeds probe bucket {args.bucket}"
        )

    eager_runtime = TextPrefillRuntime(
        model,
        backend="raw_eager",
        buckets=(args.bucket,),
        cache_root=args.torchair_cache_dir,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        linear_weight_format="native_probe",
        padding="none",
    )
    compiled_runtime = TextPrefillRuntime(
        model,
        backend="torchair",
        buckets=(args.bucket,),
        cache_root=args.torchair_cache_dir,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        linear_weight_format="native_probe",
        padding="bucket",
    )
    eager_route = eager_runtime.route(real_seq_len)
    eager_prepared = eager_runtime.prepare(
        inputs_embeds,
        attention_mask,
        position_ids,
        route=eager_route,
    )
    compiled_route = compiled_runtime.route(real_seq_len)
    compiled_prepared = compiled_runtime.prepare(
        inputs_embeds,
        attention_mask,
        position_ids,
        route=compiled_route,
    )

    eager_cache = allocate_cache(
        model,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
    )
    eager_hidden = eager_runtime.run_prepared(eager_prepared, eager_cache)
    compiled_cache = allocate_cache(
        model,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
    )
    compiled_hidden = compiled_runtime.run_prepared(compiled_prepared, compiled_cache)
    synchronize(device)

    hidden_difference = (compiled_hidden.float() - eager_hidden.float()).abs()
    cache_comparison = compare_valid_cache_prefixes(
        eager_cache,
        compiled_cache,
        real_seq_len=real_seq_len,
    )
    eager_token_ids = generate_from_prefill(
        model,
        cache=eager_cache,
        first_hidden_state=eager_hidden,
        cache_position=real_seq_len,
        rope_deltas=rope_deltas,
        max_new_tokens=args.max_new_tokens,
    )
    compiled_token_ids = generate_from_prefill(
        model,
        cache=compiled_cache,
        first_hidden_state=compiled_hidden,
        cache_position=real_seq_len,
        rope_deltas=rope_deltas,
        max_new_tokens=args.max_new_tokens,
    )

    eager_times = []
    compiled_times = []
    for _ in range(args.repeats):
        cache = allocate_cache(
            model,
            cache_length=args.cache_length,
            device=device,
            dtype=dtype,
        )
        synchronize(device)
        started = time.perf_counter()
        eager_runtime.run_prepared(eager_prepared, cache)
        synchronize(device)
        eager_times.append(time.perf_counter() - started)

        cache = allocate_cache(
            model,
            cache_length=args.cache_length,
            device=device,
            dtype=dtype,
        )
        synchronize(device)
        started = time.perf_counter()
        compiled_runtime.run_prepared(compiled_prepared, cache)
        synchronize(device)
        compiled_times.append(time.perf_counter() - started)

    eager_mean = sum(eager_times) / len(eager_times)
    compiled_mean = sum(compiled_times) / len(compiled_times)
    all_required_checks_passed = bool(
        cache_comparison["all_finite"]
        and eager_token_ids == compiled_token_ids
        and torch.isfinite(compiled_hidden).all().item()
    )
    result = {
        "configuration": {
            "crop": str(args.crop.expanduser().resolve()),
            "prompt": args.prompt,
            "model": str(model_dir),
            "device": str(device),
            "dtype": str(dtype),
            "real_seq_len": real_seq_len,
            "physical_seq_len": args.bucket,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
        },
        "compile": compiled_runtime.metadata,
        "correctness": {
            "all_required_checks_passed": all_required_checks_passed,
            "first_token_match": eager_token_ids[:1] == compiled_token_ids[:1],
            "full_token_ids_match": eager_token_ids == compiled_token_ids,
            "eager_token_ids": eager_token_ids,
            "compiled_token_ids": compiled_token_ids,
            "hidden_max_abs": float(hidden_difference.max().item()),
            "hidden_mean_abs": float(hidden_difference.mean().item()),
            "hidden_all_finite": bool(torch.isfinite(compiled_hidden).all().item()),
            "valid_cache_prefixes": cache_comparison,
        },
        "timing_s": {
            "eager_samples": eager_times,
            "compiled_samples": compiled_times,
            "eager_mean": eager_mean,
            "compiled_mean": compiled_mean,
            "speedup": eager_mean / compiled_mean,
            "eager_real_tok_per_s": real_seq_len / eager_mean,
            "compiled_real_tok_per_s": real_seq_len / compiled_mean,
            "compiled_physical_tok_per_s": args.bucket / compiled_mean,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all_required_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
