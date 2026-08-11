#!/usr/bin/env python3
"""Benchmark sequential portable UniRec crop preprocessing implementations.

The benchmark reconstructs the exact accepted crop corpus from a prior prefill
artifact's saved page/layout metadata.  Corpus construction is outside every
timed lane.  OpenCV and Torch are explicitly limited to one CPU thread.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional

try:
    import numba
except ImportError:  # Optional portable JIT lane.
    numba = None

from layout_process_pool import _decode_rgb, _prepare_frontend_payload
from modeling_optimized_unirec import UniRecImageProcessor


ArrayLane = Callable[[np.ndarray], np.ndarray]


if numba is not None:

    @numba.njit(nogil=True)
    def _numba_u8_hwc_to_normalized_chw(source: np.ndarray) -> np.ndarray:
        height, width, _channels = source.shape
        output = np.empty((1, 3, height, width), dtype=np.float32)
        scale = np.float32(2.0 / 255.0)
        offset = np.float32(1.0)
        for channel in range(3):
            for row in range(height):
                for column in range(width):
                    value = np.float32(source[row, column, channel])
                    output[0, channel, row, column] = value * scale - offset
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-crops", type=int, default=32)
    parser.add_argument("--limit-crops", type=int)
    parser.add_argument(
        "--lanes",
        help="Comma-separated subset of lanes. pillow_reference is added automatically.",
    )
    args = parser.parse_args()
    args.lanes = (
        [value.strip() for value in args.lanes.split(",") if value.strip()]
        if args.lanes
        else None
    )
    if args.rounds < 1 or args.warmup_crops < 0:
        parser.error("--rounds must be positive and --warmup-crops non-negative")
    if args.limit_crops is not None and args.limit_crops < 1:
        parser.error("--limit-crops must be positive")
    return args


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_exact_crops(
    artifact_dir: Path,
    openocr_root: Path,
    processor: UniRecImageProcessor,
) -> list[np.ndarray]:
    sys.path.insert(0, str(openocr_root))
    from tools.utils.opendoc_onnx_utils.utils import (
        crop_margin,
        tokenize_figure_of_table,
    )

    pages = sorted(
        read_jsonl(artifact_dir / "pages.jsonl"),
        key=lambda value: int(value["page_index"]),
    )
    crops: list[np.ndarray] = []
    for page in pages:
        path = Path(page["image_path"])
        rgb, _timing = _decode_rgb(path)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        rebuilt, _frontend_timing = _prepare_frontend_payload(
            page_index=int(page["page_index"]),
            path=path,
            bgr=bgr,
            layout_result=page["layout_results"],
            use_chart_recognition=True,
            crop_margin=crop_margin,
            tokenize_figure_of_table=tokenize_figure_of_table,
        )
        accepted = []
        for crop in rebuilt["crops"]:
            height, width = crop["image_rgb"].shape[:2]
            tokens = processor.estimate_encoder_token_count_for_image_size(
                width,
                height,
            )
            if tokens <= 512:
                accepted.append(crop["image_rgb"])
        if len(accepted) != len(page["crops"]):
            raise RuntimeError(
                f"page {page['page_index']} reconstructed {len(accepted)} "
                f"accepted crops, artifact has {len(page['crops'])}"
            )
        crops.extend(accepted)
    return crops


def _pillow_resize(
    crop: np.ndarray,
    processor: UniRecImageProcessor,
    *,
    reducing_gap: float | None = None,
) -> Image.Image:
    image = Image.fromarray(crop).convert("RGB")
    target_size = processor.get_processed_size(*image.size)
    return image.resize(
        target_size,
        resample=processor.resample,
        reducing_gap=reducing_gap,
    )


def _pillow_resize_without_convert(
    crop: np.ndarray,
    processor: UniRecImageProcessor,
) -> Image.Image:
    image = Image.fromarray(crop)
    if image.mode != "RGB":
        raise ValueError(f"expected uint8 RGB crop, got Pillow mode {image.mode}")
    target_size = processor.get_processed_size(*image.size)
    return image.resize(target_size, resample=processor.resample)


def build_lanes(processor: UniRecImageProcessor) -> dict[str, ArrayLane]:
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    scale = np.float32(processor.rescale_factor)
    fp16_lut = np.arange(256, dtype=np.float32)
    fp16_lut *= np.float32(2.0 / 255.0)
    fp16_lut -= np.float32(1.0)
    fp16_lut = fp16_lut.astype(np.float16)

    def pillow_reference(crop: np.ndarray) -> np.ndarray:
        inputs = processor(Image.fromarray(crop))
        return np.ascontiguousarray(inputs["pixel_values"].numpy(), dtype=np.float32)

    def pillow_inplace_hwc(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize(crop, processor)
        array = np.asarray(image, dtype=np.float32)
        np.multiply(array, scale, out=array)
        np.subtract(array, mean, out=array)
        np.divide(array, std, out=array)
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))[None]

    def pillow_chw_exact_steps(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize(crop, processor)
        chw = np.ascontiguousarray(
            np.transpose(np.asarray(image), (2, 0, 1)),
            dtype=np.float32,
        )
        np.multiply(chw, scale, out=chw)
        for channel in range(chw.shape[0]):
            np.subtract(chw[channel], mean[channel], out=chw[channel])
            np.divide(chw[channel], std[channel], out=chw[channel])
        return chw[None]

    def pillow_chw_fused_formula(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize(crop, processor)
        chw = np.ascontiguousarray(
            np.transpose(np.asarray(image), (2, 0, 1)),
            dtype=np.float32,
        )
        np.multiply(chw, np.float32(2.0 / 255.0), out=chw)
        np.subtract(chw, np.float32(1.0), out=chw)
        return chw[None]

    def pillow_no_convert_chw_fused_formula(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize_without_convert(crop, processor)
        chw = np.ascontiguousarray(
            np.transpose(np.asarray(image), (2, 0, 1)),
            dtype=np.float32,
        )
        np.multiply(chw, np.float32(2.0 / 255.0), out=chw)
        np.subtract(chw, np.float32(1.0), out=chw)
        return chw[None]

    def pillow_no_convert_chw_fp16_lut(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize_without_convert(crop, processor)
        chw_u8 = np.transpose(np.asarray(image), (2, 0, 1))
        return np.ascontiguousarray(fp16_lut[chw_u8])[None]

    def pillow_no_convert_numba_fused(crop: np.ndarray) -> np.ndarray:
        if numba is None:
            raise RuntimeError("Numba is not installed")
        image = _pillow_resize_without_convert(crop, processor)
        return _numba_u8_hwc_to_normalized_chw(np.asarray(image))

    def pillow_reducing_gap_2(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize(crop, processor, reducing_gap=2.0)
        chw = np.ascontiguousarray(
            np.transpose(np.asarray(image), (2, 0, 1)),
            dtype=np.float32,
        )
        np.multiply(chw, np.float32(2.0 / 255.0), out=chw)
        np.subtract(chw, np.float32(1.0), out=chw)
        return chw[None]

    def pillow_opencv_blob(crop: np.ndarray) -> np.ndarray:
        image = _pillow_resize(crop, processor)
        return cv2.dnn.blobFromImage(
            np.asarray(image),
            scalefactor=2.0 / 255.0,
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
            ddepth=cv2.CV_32F,
        )

    def torchvision_uint8_bicubic(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        target_width, target_height = processor.get_processed_size(width, height)
        tensor = torch.from_numpy(crop).permute(2, 0, 1)
        resized = tv_functional.resize(
            tensor,
            size=[target_height, target_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        output = resized.to(torch.float32)
        output.mul_(2.0 / 255.0).sub_(1.0)
        return output.unsqueeze(0).contiguous().numpy()

    def opencv_cubic_chw(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        target_size = processor.get_processed_size(width, height)
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
        chw = np.ascontiguousarray(
            np.transpose(resized, (2, 0, 1)),
            dtype=np.float32,
        )
        np.multiply(chw, np.float32(2.0 / 255.0), out=chw)
        np.subtract(chw, np.float32(1.0), out=chw)
        return chw[None]

    lanes = {
        "pillow_reference": pillow_reference,
        "pillow_inplace_hwc": pillow_inplace_hwc,
        "pillow_chw_exact_steps": pillow_chw_exact_steps,
        "pillow_chw_fused_formula": pillow_chw_fused_formula,
        "pillow_no_convert_chw_fused_formula": (
            pillow_no_convert_chw_fused_formula
        ),
        "pillow_no_convert_chw_fp16_lut": pillow_no_convert_chw_fp16_lut,
        "pillow_reducing_gap_2": pillow_reducing_gap_2,
        "pillow_opencv_blob": pillow_opencv_blob,
        "torchvision_uint8_bicubic": torchvision_uint8_bicubic,
        "opencv_cubic_chw": opencv_cubic_chw,
    }
    if numba is not None:
        lanes["pillow_no_convert_numba_fused"] = pillow_no_convert_numba_fused
    return lanes


def compare_lane(
    crops: list[np.ndarray],
    reference: ArrayLane,
    candidate: ArrayLane,
) -> dict[str, float | int | bool]:
    total_values = 0
    different_values = 0
    exact_crops = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    max_absolute = 0.0
    for crop in crops:
        expected = reference(crop)
        actual = candidate(crop)
        if actual.shape != expected.shape:
            raise RuntimeError(f"shape mismatch: {actual.shape} != {expected.shape}")
        difference = actual - expected
        absolute = np.abs(difference)
        crop_different = int(np.count_nonzero(difference))
        if crop_different == 0:
            exact_crops += 1
        different_values += crop_different
        total_values += int(difference.size)
        absolute_sum += float(np.sum(absolute, dtype=np.float64))
        squared_sum += float(
            np.sum(difference.astype(np.float64) ** 2, dtype=np.float64)
        )
        max_absolute = max(max_absolute, float(np.max(absolute, initial=0.0)))
    return {
        "all_exact": different_values == 0,
        "exact_crops": exact_crops,
        "different_values": different_values,
        "different_fraction": different_values / total_values,
        "mean_absolute": absolute_sum / total_values,
        "root_mean_square": (squared_sum / total_values) ** 0.5,
        "max_absolute": max_absolute,
        "total_values": total_values,
    }


def benchmark_lane(
    crops: list[np.ndarray],
    lane: ArrayLane,
    *,
    rounds: int,
    warmup_crops: int,
) -> dict[str, object]:
    checksum = 0.0
    for crop in crops[:warmup_crops]:
        checksum += float(lane(crop).reshape(-1)[0])
    wall_times = []
    for _round_index in range(rounds):
        gc.collect()
        started = time.perf_counter()
        for crop in crops:
            output = lane(crop)
            checksum += float(output.reshape(-1)[0])
        wall_times.append(time.perf_counter() - started)
    median_s = statistics.median(wall_times)
    return {
        "round_wall_s": wall_times,
        "median_s": median_s,
        "crops_per_s": len(crops) / median_s,
        "checksum": checksum,
    }


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    processor = UniRecImageProcessor()
    load_started = time.perf_counter()
    crops = load_exact_crops(
        args.artifact_dir.expanduser().resolve(),
        args.openocr_root.expanduser().resolve(),
        processor,
    )
    if args.limit_crops is not None:
        crops = crops[: args.limit_crops]
    load_s = time.perf_counter() - load_started
    processed_pixels = sum(
        processor.get_processed_size(crop.shape[1], crop.shape[0])[0]
        * processor.get_processed_size(crop.shape[1], crop.shape[0])[1]
        for crop in crops
    )
    lanes = build_lanes(processor)
    if args.lanes is not None:
        unknown = sorted(set(args.lanes) - set(lanes))
        if unknown:
            raise ValueError(f"unknown lanes: {unknown}; available={sorted(lanes)}")
        selected_names = list(dict.fromkeys(["pillow_reference", *args.lanes]))
    else:
        selected_names = list(lanes)
    results: dict[str, dict[str, object]] = {}
    reference = lanes["pillow_reference"]
    for name in selected_names:
        lane = lanes[name]
        print(f"UNIREC_CPU_PREPROCESS_LANE_BEGIN name={name}", flush=True)
        timing = benchmark_lane(
            crops,
            lane,
            rounds=args.rounds,
            warmup_crops=args.warmup_crops,
        )
        parity = (
            {
                "all_exact": True,
                "exact_crops": len(crops),
                "different_values": 0,
                "different_fraction": 0.0,
                "mean_absolute": 0.0,
                "root_mean_square": 0.0,
                "max_absolute": 0.0,
                "total_values": processed_pixels * 3,
            }
            if name == "pillow_reference"
            else compare_lane(crops, reference, lane)
        )
        timing["processed_megapixels_per_s"] = (
            processed_pixels / 1e6 / float(timing["median_s"])
        )
        timing["speedup_vs_reference"] = None
        timing["parity"] = parity
        if name == "pillow_no_convert_chw_fp16_lut":
            timing["model_input_fp16_parity"] = compare_lane(
                crops,
                lambda crop: reference(crop).astype(np.float16),
                lane,
            )
        results[name] = timing
        print(
            "UNIREC_CPU_PREPROCESS_LANE_END "
            + json.dumps({"name": name, **timing}, sort_keys=True),
            flush=True,
        )
    reference_s = float(results["pillow_reference"]["median_s"])
    for result in results.values():
        result["speedup_vs_reference"] = reference_s / float(result["median_s"])
    summary = {
        "status": "ok",
        "artifact_dir": str(args.artifact_dir),
        "crop_count": len(crops),
        "processed_megapixels": processed_pixels / 1e6,
        "corpus_load_s": load_s,
        "rounds": args.rounds,
        "warmup_crops": args.warmup_crops,
        "threading": {
            "outer_workers": 1,
            "opencv_threads": cv2.getNumThreads(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "environment": {
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
        },
        "lanes": results,
    }
    print(
        "UNIREC_CPU_PREPROCESS_SUMMARY "
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
