#!/usr/bin/env python3
"""Run the exact Transformers implementation bundled with UniRec-0.1B-1217."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoTokenizer


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_IMAGES = [
    PROJECT_ROOT / "crops/crop_01_text_block_en.png",
    PROJECT_ROOT / "crops/crop_02_equation_matrix.png",
    PROJECT_ROOT / "crops/crop_03_code_block.png",
    PROJECT_ROOT / "crops/crop_04_handwritten_title_zh.png",
    PROJECT_ROOT / "crops/crop_05_table_rwkv_dims.png",
    PROJECT_ROOT / "crops/crop_06_chart_cubic_spline.png",
]
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def synchronize_device(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()


def configure_npu(device: str) -> None:
    if not device.startswith("npu"):
        return
    import torch_npu  # noqa: F401

    torch.npu.set_compile_mode(jit_compile=False)
    print("[npu] jit_compile=False", flush=True)


def import_checkpoint_implementation(model_path: Path) -> tuple[type, type, Any]:
    package_name = model_path.name
    if not package_name.isidentifier():
        raise ValueError(
            "The official checkpoint directory name must be a valid Python package name; "
            f"got {package_name!r}"
        )
    sys.path.insert(0, str(model_path.parent))
    modeling = importlib.import_module(f"{package_name}.modeling_unirec")
    processing = importlib.import_module(f"{package_name}.processing_unirec")
    return (
        modeling.UniRecForConditionalGeneration,
        processing.UniRecImageProcessor,
        processing.clean_special_tokens,
    )


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(inner) for inner in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "tmp/12_unirec_0_1b_inference/original_transformers_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    images = [path.expanduser().resolve() for path in (args.image or DEFAULT_IMAGES)]
    dtype = DTYPE_MAP[args.dtype]
    configure_npu(args.device)

    model_class, processor_class, clean_special_tokens = import_checkpoint_implementation(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = processor_class.from_pretrained(model_path)

    load_started = time.perf_counter()
    model = model_class.from_pretrained(model_path, torch_dtype=dtype).to(args.device)
    model.eval()
    synchronize_device(args.device)
    model_load_s = time.perf_counter() - load_started
    print(f"Loaded bundled UniRec-1217 implementation in {model_load_s:.3f}s", flush=True)

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] original transformers: {image_path.name}", flush=True)
        image = Image.open(image_path).convert("RGB")
        preprocess_started = time.perf_counter()
        processed = processor(image, return_tensors="pt")
        pixel_values = processed["pixel_values"][0][:3].unsqueeze(0).to(args.device, dtype=dtype)
        synchronize_device(args.device)
        preprocess_s = time.perf_counter() - preprocess_started

        inference_started = time.perf_counter()
        with torch.inference_mode():
            token_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=None,
                attention_mask=None,
                max_length=args.max_length,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        synchronize_device(args.device)
        inference_s = time.perf_counter() - inference_started
        decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=False)[0]
        text = clean_special_tokens(decoded)
        returned_tokens = int(token_ids.shape[-1])
        generated_tokens = max(returned_tokens - 1, 0)
        result = {
            "image": str(image_path),
            "pixel_values_shape": list(pixel_values.shape),
            "preprocess_s": preprocess_s,
            "inference_s": inference_s,
            "returned_tokens": returned_tokens,
            "generated_tokens_excluding_start": generated_tokens,
            "generated_tokens_per_s": generated_tokens / inference_s if inference_s > 0 else None,
            "token_ids": token_ids.cpu().tolist()[0],
            "text": text,
        }
        print(f"inference_s={inference_s:.4f}", flush=True)
        print(f"generated_tokens_per_s={result['generated_tokens_per_s']}", flush=True)
        print("generation:")
        print(text, flush=True)
        results.append(result)

    payload = {
        "experiment": "12_unirec_1217_original_transformers",
        "status": "ok",
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "model_load_s": model_load_s,
        "images": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
