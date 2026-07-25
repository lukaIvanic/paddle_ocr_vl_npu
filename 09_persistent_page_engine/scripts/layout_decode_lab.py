#!/usr/bin/env python3
"""Isolated sequential PNG-decoder benchmark for layout-page inputs."""

from __future__ import annotations

import argparse
import gc
import io
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")


@dataclass(frozen=True)
class Decoder:
    name: str
    version: str | None
    native_format: str
    decode_native: Callable[[bytes], Any]
    adapt_to_current_bgr: Callable[[Any], np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["opencv", "pillow", "torchvision"],
        choices=["opencv", "pillow", "torchvision", "pyspng"],
    )
    parser.add_argument(
        "--format",
        choices=["all", "png", "jpeg"],
        default="all",
        help="Filter the selected pages by their encoded-byte format.",
    )
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


def rgb_to_bgr(value: np.ndarray) -> np.ndarray:
    if value.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {value.dtype}")
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    elif value.ndim == 3 and value.shape[2] == 4:
        value = value[:, :, :3]
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxW, HxWx3, or HxWx4 image, got {value.shape}")
    return np.ascontiguousarray(value[:, :, ::-1])


def encoded_format(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def make_opencv() -> Decoder:
    def decode(data: bytes) -> np.ndarray:
        value = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if value is None:
            raise RuntimeError("OpenCV failed to decode PNG")
        return value

    return Decoder(
        "opencv",
        cv2.__version__,
        "NumPy uint8 HWC BGR",
        decode,
        np.ascontiguousarray,
    )


def make_pillow() -> Decoder:
    from PIL import Image, __version__

    def decode(data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))

    return Decoder(
        "pillow",
        __version__,
        "NumPy uint8 HWC RGB",
        decode,
        rgb_to_bgr,
    )


def make_torchvision() -> Decoder:
    import torch
    import torchvision
    from torchvision.io import ImageReadMode, decode_image

    def decode(data: bytes) -> Any:
        encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        return decode_image(encoded, mode=ImageReadMode.RGB)

    def adapt(value: Any) -> np.ndarray:
        rgb = value.permute(1, 2, 0).contiguous().numpy()
        return rgb_to_bgr(rgb)

    return Decoder(
        "torchvision",
        torchvision.__version__,
        "Torch uint8 CHW RGB",
        decode,
        adapt,
    )


def make_pyspng() -> Decoder:
    import pyspng

    return Decoder(
        "pyspng",
        getattr(pyspng, "__version__", None),
        "NumPy uint8 HWC native RGB/RGBA/gray",
        pyspng.load,
        rgb_to_bgr,
    )


FACTORIES: dict[str, Callable[[], Decoder]] = {
    "opencv": make_opencv,
    "pillow": make_pillow,
    "torchvision": make_torchvision,
    "pyspng": make_pyspng,
}


def validate(
    decoder: Decoder,
    encoded_pages: list[bytes],
    references: list[np.ndarray],
) -> dict[str, Any]:
    mismatches: list[int] = []
    for index, (encoded, reference) in enumerate(zip(encoded_pages, references)):
        candidate = decoder.adapt_to_current_bgr(decoder.decode_native(encoded))
        if (
            candidate.dtype != reference.dtype
            or candidate.shape != reference.shape
            or not candidate.flags.c_contiguous
            or not np.array_equal(candidate, reference)
        ):
            mismatches.append(index)
    return {
        "current_bgr_exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_indices": mismatches,
    }


def run_once(decoder: Decoder, encoded_pages: list[bytes]) -> dict[str, float]:
    decode_ns = 0
    adapter_ns = 0
    gc.collect()
    gc.disable()
    try:
        wall_started = time.perf_counter_ns()
        for encoded in encoded_pages:
            decode_started = time.perf_counter_ns()
            native = decoder.decode_native(encoded)
            decode_finished = time.perf_counter_ns()
            decoder.adapt_to_current_bgr(native)
            adapted = time.perf_counter_ns()
            decode_ns += decode_finished - decode_started
            adapter_ns += adapted - decode_finished
        wall_ns = time.perf_counter_ns() - wall_started
    finally:
        gc.enable()
    return {
        "native_decode_s": decode_ns / 1_000_000_000,
        "current_bgr_adapter_s": adapter_ns / 1_000_000_000,
        "current_bgr_contract_s": wall_ns / 1_000_000_000,
    }


def median(samples: list[dict[str, float]], key: str) -> float:
    return float(statistics.median(sample[key] for sample in samples))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    paths = load_paths(dataset_json, images_dir, args.offset, args.limit)

    read_started = time.perf_counter_ns()
    encoded_pages = [path.read_bytes() for path in paths]
    read_s = (time.perf_counter_ns() - read_started) / 1_000_000_000
    formats = [encoded_format(data) for data in encoded_pages]
    format_counts = {
        name: formats.count(name) for name in sorted(set(formats))
    }
    if args.format != "all":
        selected = [
            (path, data, name)
            for path, data, name in zip(paths, encoded_pages, formats)
            if name == args.format
        ]
        paths = [path for path, _, _ in selected]
        encoded_pages = [data for _, data, _ in selected]
        formats = [name for _, _, name in selected]
        if not paths:
            raise ValueError(f"selected corpus has no {args.format} images")

    decoders: list[Decoder] = []
    unavailable: dict[str, str] = {}
    for name in args.decoders:
        try:
            decoders.append(FACTORIES[name]())
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    if not any(decoder.name == "opencv" for decoder in decoders):
        raise RuntimeError("OpenCV is required as the current-pipeline reference")

    opencv = next(decoder for decoder in decoders if decoder.name == "opencv")
    references = [
        opencv.adapt_to_current_bgr(opencv.decode_native(encoded))
        for encoded in encoded_pages
    ]
    validation = {
        decoder.name: validate(decoder, encoded_pages, references)
        for decoder in decoders
    }
    total_pixels = sum(int(image.shape[0]) * int(image.shape[1]) for image in references)
    del references

    for _ in range(args.warmup_rounds):
        for decoder in decoders:
            run_once(decoder, encoded_pages)

    rounds: dict[str, list[dict[str, float]]] = {
        decoder.name: [] for decoder in decoders
    }
    for round_index in range(args.rounds):
        start = round_index % len(decoders)
        for decoder in decoders[start:] + decoders[:start]:
            rounds[decoder.name].append(run_once(decoder, encoded_pages))

    results: dict[str, Any] = {}
    for decoder in decoders:
        samples = rounds[decoder.name]
        native_s = median(samples, "native_decode_s")
        contract_s = median(samples, "current_bgr_contract_s")
        results[decoder.name] = {
            "version": decoder.version,
            "native_format": decoder.native_format,
            "validation": validation[decoder.name],
            "rounds": samples,
            "median_native_decode_s": native_s,
            "median_current_bgr_adapter_s": median(
                samples, "current_bgr_adapter_s"
            ),
            "median_current_bgr_contract_s": contract_s,
            "native_megapixels_per_s": total_pixels / 1_000_000 / native_s,
            "current_bgr_megapixels_per_s": total_pixels / 1_000_000 / contract_s,
        }
    opencv_s = results["opencv"]["median_current_bgr_contract_s"]
    for result in results.values():
        result["current_bgr_speedup_vs_opencv"] = (
            opencv_s / result["median_current_bgr_contract_s"]
        )

    report = {
        "scope": {
            "native_decode": "decoder's natural CPU output",
            "current_bgr_contract": (
                "diagnostic compatibility adapter only; not a permanent "
                "architecture requirement"
            ),
            "execution": "sequential pages, encoded bytes preloaded",
        },
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "offset": args.offset,
        "pages": len(paths),
        "format_filter": args.format,
        "input_format_counts_before_filter": format_counts,
        "selected_format_counts": {
            name: formats.count(name) for name in sorted(set(formats))
        },
        "compressed_bytes": sum(len(data) for data in encoded_pages),
        "read_all_compressed_bytes_s": read_s,
        "total_pixels": total_pixels,
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "python": sys.version,
        "unavailable": unavailable,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    main()
