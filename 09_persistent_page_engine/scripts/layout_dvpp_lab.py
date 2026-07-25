#!/usr/bin/env python3
"""Benchmark an isolated Ascend DVPP page-decode and layout-input path."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from utils.timing import DeviceTimeline, synchronize


DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_EXTENSION = (
    REPO_ROOT
    / ".runtime_cache"
    / "dvpp_probe_src3"
    / "paddle_layout_dvpp_probe_ops.so"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
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


def jpeg_shape(encoded: bytes) -> tuple[int, int, int]:
    """Return HWC from JPEG SOF metadata without decoding pixel data."""

    stream = io.BytesIO(encoded)
    if stream.read(2) != b"\xff\xd8":
        raise ValueError("input is not a JPEG byte stream")
    standalone = {0x01, *range(0xD0, 0xDA)}
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        prefix = stream.read(1)
        if not prefix:
            break
        if prefix != b"\xff":
            continue
        marker = stream.read(1)
        while marker == b"\xff":
            marker = stream.read(1)
        if not marker:
            break
        code = marker[0]
        if code in standalone:
            continue
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            break
        length = int.from_bytes(length_bytes, "big")
        if length < 2:
            raise ValueError(f"invalid JPEG segment length: {length}")
        if code in sof_markers:
            payload = stream.read(length - 2)
            if len(payload) < 6:
                break
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            channels = int(payload[5])
            return height, width, channels
        stream.seek(length - 2, io.SEEK_CUR)
    raise ValueError("JPEG SOF marker not found")


def make_encoded_tensor(encoded: bytes) -> tuple[torch.Tensor, bool]:
    tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    try:
        return tensor.pin_memory(), True
    except Exception:
        return tensor, False


def opencv_bgr(encoded: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV failed to decode JPEG")
    return image


def current_layout_tensor(processor: Any, encoded: bytes) -> torch.Tensor:
    image = Image.fromarray(opencv_bgr(encoded))
    return processor(images=[image], return_tensors="pt")["pixel_values"]


def load_dvpp_extension(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"DVPP probe extension not found: {path}")
    torch.ops.load_library(str(path))
    required = (
        "_dvpp_init",
        "_decode_jpeg_aclnn",
        "_resize_aclnn",
        "_img_to_tensor_aclnn",
    )
    missing = [name for name in required if not hasattr(torch.ops.torchvision, name)]
    if missing:
        raise RuntimeError(f"DVPP extension did not register operations: {missing}")
    torch.ops.torchvision._dvpp_init()


def run_dvpp_page(
    encoded_cpu: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    timeline = DeviceTimeline(device)
    encoded_npu = timeline.measure(
        "compressed_h2d",
        lambda: encoded_cpu.to(device, non_blocking=True),
    )
    decoded_rgb = timeline.measure(
        "jpeg_decode",
        lambda: torch.ops.torchvision._decode_jpeg_aclnn(
            encoded_npu,
            image_shape=list(shape),
            channels=3,
            try_recover_truncated=True,
        ),
    )
    resized_rgb = timeline.measure(
        "layout_resize_bicubic",
        lambda: torch.ops.torchvision._resize_aclnn(
            decoded_rgb,
            size=[800, 800],
            interpolation_mode=2,
        ),
    )
    resized_bgr = timeline.measure(
        "layout_channel_reorder",
        lambda: decoded_rgb.new_empty((1, 3, 800, 800)).copy_(
            resized_rgb.flip(1)
        ),
    )
    pixel_values = timeline.measure(
        "layout_img_to_tensor",
        lambda: torch.ops.torchvision._img_to_tensor_aclnn(resized_bgr),
    )
    full_bgr_hwc = timeline.measure(
        "full_page_channel_reorder",
        lambda: decoded_rgb.flip(1).permute(0, 2, 3, 1).contiguous(),
    )
    full_bgr_cpu = timeline.measure(
        "full_page_d2h",
        lambda: full_bgr_hwc.cpu(),
    )
    return pixel_values, full_bgr_cpu.squeeze(0), timeline.resolve()


def run_dvpp_round(
    encoded_tensors: list[torch.Tensor],
    shapes: list[tuple[int, int, int]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float], float]:
    pixel_values: list[torch.Tensor] = []
    full_pages: list[torch.Tensor] = []
    stage_totals: defaultdict[str, float] = defaultdict(float)
    synchronize(device)
    started = time.perf_counter()
    for encoded, shape in zip(encoded_tensors, shapes):
        pixels, full_page, timing = run_dvpp_page(encoded, shape, device)
        pixel_values.append(pixels)
        full_pages.append(full_page)
        for name, seconds in timing.items():
            stage_totals[name] += seconds
    synchronize(device)
    wall_s = time.perf_counter() - started
    return pixel_values, full_pages, dict(stage_totals), wall_s


def run_current_round(
    processor: Any,
    encoded_pages: list[bytes],
) -> tuple[list[torch.Tensor], list[np.ndarray], float]:
    started = time.perf_counter()
    pages = [opencv_bgr(encoded) for encoded in encoded_pages]
    tensors = [
        processor(
            images=[Image.fromarray(page)],
            return_tensors="pt",
        )["pixel_values"]
        for page in pages
    ]
    return tensors, pages, time.perf_counter() - started


def delta_summary(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        return {
            "shape_equal": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        }
    delta = candidate.to(torch.float32) - reference.to(torch.float32)
    return {
        "shape_equal": True,
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "equal_fraction": float((delta == 0).to(torch.float32).mean().item()),
    }


def median_rounds(rounds: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rounds for key in row})
    return {
        key: float(statistics.median(row[key] for row in rounds))
        for key in keys
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    extension = args.extension.expanduser().resolve()
    output = args.output.expanduser().resolve()
    paths = load_paths(dataset_json, images_dir, args.offset, args.limit)
    encoded_pages = [path.read_bytes() for path in paths]
    shapes = [jpeg_shape(encoded) for encoded in encoded_pages]
    encoded_tensors_and_pin = [make_encoded_tensor(data) for data in encoded_pages]
    encoded_tensors = [item[0] for item in encoded_tensors_and_pin]
    pinned = [item[1] for item in encoded_tensors_and_pin]

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("DVPP layout lab requires an available NPU")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    load_dvpp_extension(extension)
    processor = AutoImageProcessor.from_pretrained(layout_model)

    for _ in range(args.warmup_rounds):
        run_current_round(processor, encoded_pages)
        run_dvpp_round(encoded_tensors, shapes, device)

    current_rounds: list[float] = []
    dvpp_rounds: list[dict[str, float]] = []
    current_tensors: list[torch.Tensor] = []
    current_pages: list[np.ndarray] = []
    dvpp_tensors: list[torch.Tensor] = []
    dvpp_pages: list[torch.Tensor] = []
    for _ in range(args.rounds):
        current_tensors, current_pages, current_wall_s = run_current_round(
            processor,
            encoded_pages,
        )
        current_rounds.append(current_wall_s)
        dvpp_tensors, dvpp_pages, stages, dvpp_wall_s = run_dvpp_round(
            encoded_tensors,
            shapes,
            device,
        )
        dvpp_rounds.append({"wall": dvpp_wall_s, **stages})

    validation = []
    for index, (
        current_tensor,
        current_page,
        dvpp_tensor,
        dvpp_page,
    ) in enumerate(
        zip(current_tensors, current_pages, dvpp_tensors, dvpp_pages)
    ):
        validation.append(
            {
                "page_index": index,
                "image": paths[index].name,
                "model_input": delta_summary(
                    dvpp_tensor.cpu(),
                    current_tensor,
                ),
                "full_page_bgr": delta_summary(
                    dvpp_page,
                    torch.from_numpy(current_page),
                ),
            }
        )

    current_median_s = float(statistics.median(current_rounds))
    dvpp_median = median_rounds(dvpp_rounds)
    report = {
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "layout_model": str(layout_model),
        "extension": str(extension),
        "offset": args.offset,
        "pages": len(paths),
        "compressed_bytes": sum(len(encoded) for encoded in encoded_pages),
        "decoded_pixels": sum(height * width for height, width, _ in shapes),
        "all_compressed_inputs_pinned": all(pinned),
        "current": {
            "path": "OpenCV BGR -> PIL -> TorchVision processor",
            "rounds_s": current_rounds,
            "median_s": current_median_s,
            "pages_per_s": len(paths) / current_median_s,
        },
        "dvpp": {
            "path": (
                "compressed H2D -> DVPP JPEG RGB -> DVPP bicubic 800x800 -> "
                "BGR reorder -> DVPP /255; full BGR page D2H"
            ),
            "rounds": dvpp_rounds,
            "median": dvpp_median,
            "pages_per_s": len(paths) / dvpp_median["wall"],
        },
        "speedup": current_median_s / dvpp_median["wall"],
        "saved_s": current_median_s - dvpp_median["wall"],
        "validation": validation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
