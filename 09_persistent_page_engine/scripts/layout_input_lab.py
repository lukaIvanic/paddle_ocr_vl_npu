#!/usr/bin/env python3
"""Benchmark faithful JPEG-to-layout-model tensor preparation paths."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from transformers import AutoImageProcessor


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be non-negative and --limit positive")
    if args.rounds <= 0 or args.warmup_rounds < 0:
        parser.error("--rounds must be positive and --warmup-rounds non-negative")
    return args


def load_paths(
    dataset_json: Path,
    images_dir: Path,
    offset: int,
    limit: int,
) -> list[Path]:
    annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
    pages = annotations[offset : offset + limit]
    if len(pages) != limit:
        raise ValueError(f"requested {limit} pages, got {len(pages)}")
    paths = [
        images_dir / Path(page["page_info"]["image_path"]).name for page in pages
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")
    return paths


def opencv_bgr(encoded: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV failed to decode image")
    return image


def torchvision_rgb(encoded: bytes) -> torch.Tensor:
    encoded_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    return decode_jpeg(encoded_tensor, mode=ImageReadMode.RGB)


def current_paddlex_tensor(processor: Any, encoded: bytes) -> torch.Tensor:
    # PaddleX reads BGR and constructs a PIL image without swapping channels.
    # PIL therefore carries the same B, G, R values as if they were RGB.
    image = Image.fromarray(opencv_bgr(encoded))
    return processor(images=[image], return_tensors="pt")["pixel_values"]


def opencv_tensor_view(processor: Any, encoded: bytes) -> torch.Tensor:
    # Preserve the current BGR NumPy page for PaddleX's downstream crop path,
    # but avoid the full-resolution NumPy -> PIL -> Torch conversion. The CHW
    # tensor is a non-contiguous view; TorchVision resize accepts that directly.
    bgr = opencv_bgr(encoded)
    bgr_chw = torch.from_numpy(bgr).permute(2, 0, 1)
    return processor(images=[bgr_chw], return_tensors="pt")["pixel_values"]


def native_torch_tensor(processor: Any, encoded: bytes) -> torch.Tensor:
    # The processor accepts native CHW tensors. Its resize and rescale are
    # channel-independent, so preserve PaddleX's effective BGR model input by
    # swapping channels only after the image has been reduced to 800x800.
    rgb = torchvision_rgb(encoded)
    pixel_values = processor(images=[rgb], return_tensors="pt")["pixel_values"]
    return pixel_values.flip(1)


PATHS: dict[str, Callable[[Any, bytes], torch.Tensor]] = {
    "current_paddlex": current_paddlex_tensor,
    "opencv_tensor_view": opencv_tensor_view,
    "native_torch": native_torch_tensor,
}


def validate(
    processor: Any,
    encoded_pages: list[bytes],
) -> dict[str, Any]:
    page_results: list[dict[str, Any]] = []
    for index, encoded in enumerate(encoded_pages):
        reference = current_paddlex_tensor(processor, encoded)
        candidates = {
            name: path(processor, encoded)
            for name, path in PATHS.items()
            if name != "current_paddlex"
        }
        page_results.append(
            {
                "page_index": index,
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
                "candidates": {
                    name: {
                        "exact": bool(torch.equal(candidate, reference)),
                        "max_abs": float((candidate - reference).abs().max().item()),
                        "mean_abs": float(
                            (candidate - reference).abs().mean().item()
                        ),
                    }
                    for name, candidate in candidates.items()
                },
            }
        )
    names = [name for name in PATHS if name != "current_paddlex"]
    return {
        name: {
            "all_exact": all(
                row["candidates"][name]["exact"] for row in page_results
            ),
            "max_abs": max(
                (
                    row["candidates"][name]["max_abs"]
                    for row in page_results
                ),
                default=0.0,
            ),
            "mean_abs_max": max(
                (
                    row["candidates"][name]["mean_abs"]
                    for row in page_results
                ),
                default=0.0,
            ),
        }
        for name in names
    } | {
        "pages": page_results,
    }


def run_once(
    path: Callable[[Any, bytes], torch.Tensor],
    processor: Any,
    encoded_pages: list[bytes],
) -> float:
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        for encoded in encoded_pages:
            path(processor, encoded)
        elapsed_ns = time.perf_counter_ns() - started
    finally:
        gc.enable()
    return elapsed_ns / 1_000_000_000


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    output = args.output.expanduser().resolve()

    paths = load_paths(dataset_json, images_dir, args.offset, args.limit)
    encoded_pages = [path.read_bytes() for path in paths]
    non_jpeg = [
        path.name
        for path, encoded in zip(paths, encoded_pages)
        if not encoded.startswith(b"\xff\xd8\xff")
    ]
    if non_jpeg:
        raise ValueError(
            "this experiment uses the JPEG-specific TorchVision decoder; "
            f"non-JPEG inputs: {non_jpeg[:5]}"
        )

    processor = AutoImageProcessor.from_pretrained(layout_model)
    processor_contract = {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "size": dict(processor.size),
        "resample": int(processor.resample),
        "do_resize": bool(processor.do_resize),
        "do_rescale": bool(processor.do_rescale),
        "rescale_factor": float(processor.rescale_factor),
        "do_normalize": bool(processor.do_normalize),
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "do_convert_rgb": processor.do_convert_rgb,
    }

    validation = validate(processor, encoded_pages)
    for _ in range(args.warmup_rounds):
        for path in PATHS.values():
            run_once(path, processor, encoded_pages)

    samples = {name: [] for name in PATHS}
    for _ in range(args.rounds):
        for name, path in PATHS.items():
            samples[name].append(run_once(path, processor, encoded_pages))

    results = {
        name: {
            "rounds_s": values,
            "median_s": float(statistics.median(values)),
            "pages_per_s": len(encoded_pages) / statistics.median(values),
        }
        for name, values in samples.items()
    }
    baseline_s = results["current_paddlex"]["median_s"]
    candidate_s = results["native_torch"]["median_s"]
    report = {
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "layout_model": str(layout_model),
        "offset": args.offset,
        "pages": len(paths),
        "compressed_bytes": sum(len(encoded) for encoded in encoded_pages),
        "processor_contract": processor_contract,
        "paths": {
            "current_paddlex": (
                "JPEG -> OpenCV BGR HWC NumPy -> PIL -> processor CHW "
                "resize/rescale"
            ),
            "opencv_tensor_view": (
                "JPEG -> OpenCV BGR HWC NumPy -> zero-copy Torch CHW view "
                "-> processor resize/rescale"
            ),
            "native_torch": (
                "JPEG -> TorchVision RGB CHW torch -> processor resize/rescale "
                "-> channel flip at 800x800"
            ),
        },
        "validation": validation,
        "results": results,
        "native_speedup": baseline_s / candidate_s,
        "native_saved_s": baseline_s - candidate_s,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
