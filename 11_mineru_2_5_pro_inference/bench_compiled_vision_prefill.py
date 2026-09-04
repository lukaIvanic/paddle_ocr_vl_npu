#!/usr/bin/env python3
"""Compare eager and padded TorchAir B=1 MinerU vision prefill."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from run_transformers_recognition_smoke import configure_npu, synchronize
from vision_prefill_compile import (
    VISION_ATTENTION_IMPL_CHOICES,
    VISION_LAYER_NORM_IMPL_CHOICES,
    VISION_PROJECTION_IMPL_CHOICES,
    MinerUVisionPrefillRuntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="[layout]")
    parser.add_argument("--layout-size", type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument(
        "--eager-reference-attention",
        choices=("manual", "prompt_flash_attention"),
        default="prompt_flash_attention",
    )
    parser.add_argument(
        "--compiled-attention",
        choices=VISION_ATTENTION_IMPL_CHOICES,
        default="prompt_flash_attention",
    )
    parser.add_argument(
        "--layer-norm-impl",
        choices=VISION_LAYER_NORM_IMPL_CHOICES,
        default="module",
    )
    parser.add_argument(
        "--projection-impl",
        choices=VISION_PROJECTION_IMPL_CHOICES,
        default="linear",
    )
    parser.add_argument("--promptfa-pad-head-dim-to", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def chat_prompt(processor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def timed_vision(model, pixel_values, grid_thw, *, warmup: int, repeats: int):
    for _ in range(warmup):
        model.get_image_features(pixel_values, grid_thw)
    synchronize()
    samples = []
    last = None
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        last = model.get_image_features(pixel_values, grid_thw)
        synchronize()
        samples.append(time.perf_counter() - started)
    return last, samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_s": sum(samples) / len(samples),
        "min_s": ordered[0],
        "median_s": ordered[len(ordered) // 2],
        "max_s": ordered[-1],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.bucket <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("bucket and repeats must be positive; warmup must be non-negative")
    print("[setup] configure NPU and load processor/model", flush=True)
    configure_npu()
    import torch_npu  # noqa: F401
    from transformers import AutoProcessor

    torch.npu.set_compile_mode(jit_compile=False)
    model_dir = args.model.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device="npu:0",
    )
    model.set_vision_attention_impl(args.eager_reference_attention)
    print("[setup] model loaded", flush=True)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    original_size = image.size
    if args.layout_size is not None:
        image = image.resize(tuple(args.layout_size), Image.Resampling.BICUBIC)
    processed_size = image.size
    inputs = processor(
        text=[chat_prompt(processor, args.prompt)],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device=model.device, dtype=model.dtype)
    pixel_values = inputs.pixel_values
    grid_thw = inputs.image_grid_thw
    real_tokens = int(pixel_values.shape[0])
    if real_tokens > args.bucket:
        raise ValueError(
            f"real vision tokens {real_tokens} exceed requested bucket {args.bucket}"
        )
    print(
        f"[input] size={processed_size} real_tokens={real_tokens} bucket={args.bucket}",
        flush=True,
    )

    model.set_vision_prefill_runtime(None)
    print("[eager] warmup and measurement", flush=True)
    eager_features, eager_samples = timed_vision(
        model,
        pixel_values,
        grid_thw,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    runtime = MinerUVisionPrefillRuntime(
        model.visual,
        buckets=(args.bucket,),
        cache_root=args.cache_dir,
        model_dir=model_dir,
        device=model.device,
        dtype=model.dtype,
        attention_impl=args.compiled_attention,
        layer_norm_impl=args.layer_norm_impl,
        projection_impl=args.projection_impl,
        promptfa_pad_head_dim_to=args.promptfa_pad_head_dim_to,
    )
    model.set_vision_prefill_runtime(runtime)
    print("[compiled] first call may compile or restore the static graph", flush=True)
    compiled_features, compiled_samples = timed_vision(
        model,
        pixel_values,
        grid_thw,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    delta = (compiled_features.float() - eager_features.float()).abs()
    eager_flat = eager_features.float().flatten()
    compiled_flat = compiled_features.float().flatten()
    cosine = float(F.cosine_similarity(eager_flat, compiled_flat, dim=0).item())

    def generate(runtime_value):
        model.set_vision_prefill_runtime(runtime_value)
        synchronize()
        started = time.perf_counter()
        tokens = model.generate_ids(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=args.generation_tokens,
            eos_token_id=int(model.config.eos_token_id),
            pad_token_id=int(model.config.pad_token_id),
        )
        synchronize()
        return tokens.detach().cpu(), time.perf_counter() - started

    generation_comparison = None
    if not args.skip_generation:
        print("[accuracy] eager generation", flush=True)
        eager_tokens, eager_generation_s = generate(None)
        print("[accuracy] compiled-vision generation", flush=True)
        compiled_tokens, compiled_generation_s = generate(runtime)
        generation_comparison = {
            "requested_tokens": int(args.generation_tokens),
            "eager_tokens": eager_tokens.tolist(),
            "compiled_tokens": compiled_tokens.tolist(),
            "exact": bool(torch.equal(eager_tokens, compiled_tokens)),
            "first_difference": next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(eager_tokens[0].tolist(), compiled_tokens[0].tolist())
                    )
                    if left != right
                ),
                None,
            ),
            "eager_generation_s": eager_generation_s,
            "compiled_generation_s": compiled_generation_s,
        }
    eager_timing = summarize(eager_samples)
    compiled_timing = summarize(compiled_samples)
    result = {
        "model": str(model_dir),
        "image": str(args.image.expanduser().resolve()),
        "prompt": args.prompt,
        "original_size_wh": list(original_size),
        "processed_size_wh": list(processed_size),
        "grid_thw": grid_thw.detach().cpu().tolist(),
        "real_vision_tokens": real_tokens,
        "physical_vision_tokens": int(args.bucket),
        "useful_token_fraction": real_tokens / int(args.bucket),
        "diagnostic_contract": {
            "eager_reference_attention": args.eager_reference_attention,
            "compiled_attention": args.compiled_attention,
            "layer_norm_impl": args.layer_norm_impl,
            "projection_impl": args.projection_impl,
            "promptfa_pad_head_dim_to": int(args.promptfa_pad_head_dim_to),
        },
        "eager_full_vision": {
            **eager_timing,
            "effective_tok_s": real_tokens / eager_timing["mean_s"],
            "physical_tok_s": real_tokens / eager_timing["mean_s"],
        },
        "compiled_full_vision": {
            **compiled_timing,
            "effective_tok_s": real_tokens / compiled_timing["mean_s"],
            "physical_tok_s": int(args.bucket) / compiled_timing["mean_s"],
        },
        "speedup": eager_timing["mean_s"] / compiled_timing["mean_s"],
        "feature_comparison": {
            "shape": list(eager_features.shape),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "relative_l2": float(
                torch.linalg.vector_norm(compiled_flat - eager_flat).item()
                / max(torch.linalg.vector_norm(eager_flat).item(), 1e-12)
            ),
            "cosine": cosine,
            "nonfinite_eager": int((~torch.isfinite(eager_features)).sum().item()),
            "nonfinite_compiled": int((~torch.isfinite(compiled_features)).sum().item()),
        },
        "generation_comparison": generation_comparison,
        "runtime": runtime.metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
