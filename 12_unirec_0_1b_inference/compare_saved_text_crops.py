#!/usr/bin/env python3
"""Compare saved UniRec text crops with official OpenDoc ONNX and local eager.

The crop pixels come from an existing full-page run: the script reloads the
source page, applies the saved integer layout box, and requires the resulting
size to match the recognition trace.  This avoids rerunning layout and makes
the official-versus-local recognizer comparison use identical RGB pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modeling_optimized_unirec import OptimizedUniRecRunner


DEFAULT_REQUEST_IDS = (
    "page_000287_crop_0007",  # blank-space loop
    "page_000446_crop_0001",  # answer-list loop
    "page_000450_crop_0006",  # bilingual sentence loop
    "page_000527_crop_0007",  # underscore loop
    "page_000654_crop_0020",  # package-name loop
    "page_000683_crop_0005",  # Chinese phrase loop
    "page_001293_crop_0013",  # math-markup scoring case
    "page_001293_crop_0004",  # normal control
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stock-encoder", type=Path, required=True)
    parser.add_argument("--stock-decoder", type=Path, required=True)
    parser.add_argument("--stock-tokenizer-mapping", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def read_trace(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["request_id"])] = record
    return records


def page_result(source_output: Path, page_name: str) -> dict[str, Any]:
    stem = Path(page_name).stem
    path = source_output / stem / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"saved page JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def recreate_crop(
    *,
    source_output: Path,
    dataset_root: Path,
    trace_record: dict[str, Any],
) -> tuple[Image.Image, list[int]]:
    page_name = str(trace_record["page"])
    page_path = dataset_root / page_name
    if not page_path.is_file():
        raise FileNotFoundError(f"source page is missing: {page_path}")
    page = page_result(source_output, page_name)
    label = str(trace_record["label"])
    matches = [
        row for row in page["recognition_results"] if str(row["label"]) == label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one saved block for {page_name} label={label}, got {len(matches)}"
        )
    bbox = [int(value) for value in matches[0]["bbox"]]
    with Image.open(page_path) as image:
        crop = image.convert("RGB").crop(tuple(bbox))
    expected_size = tuple(int(value) for value in trace_record["crop_size"])
    if crop.size != expected_size:
        raise RuntimeError(
            f"crop-size mismatch for {trace_record['request_id']}: "
            f"recreated={crop.size} trace={expected_size} bbox={bbox}"
        )
    return crop, bbox


def main() -> None:
    args = parse_args()
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    paths = (
        args.openocr_root,
        args.model_path / "model.pth",
        args.stock_encoder,
        args.stock_decoder,
        args.stock_tokenizer_mapping,
        args.source_output / "recognition_trace.jsonl",
        args.dataset_root,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs are missing: {missing}")

    request_ids = tuple(args.request_id) or DEFAULT_REQUEST_IDS
    trace = read_trace(args.source_output / "recognition_trace.jsonl")
    missing_ids = [request_id for request_id in request_ids if request_id not in trace]
    if missing_ids:
        raise KeyError(f"request IDs are missing from source trace: {missing_ids}")

    output_dir = args.output_dir.expanduser().resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "comparison.jsonl"
    results_path.write_text("", encoding="utf-8")

    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.infer_unirec_onnx import UniRecONNX

    stock_setup_started = time.perf_counter()
    stock = UniRecONNX(
        encoder_path=str(args.stock_encoder.expanduser().resolve()),
        decoder_path=str(args.stock_decoder.expanduser().resolve()),
        mapping_path=str(args.stock_tokenizer_mapping.expanduser().resolve()),
        use_gpu=False,
        auto_download=False,
    )
    stock_setup_s = time.perf_counter() - stock_setup_started

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
    custom = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype=args.dtype,
    )

    records: list[dict[str, Any]] = []
    for index, request_id in enumerate(request_ids, start=1):
        source = trace[request_id]
        crop, bbox = recreate_crop(
            source_output=args.source_output.expanduser().resolve(),
            dataset_root=args.dataset_root.expanduser().resolve(),
            trace_record=source,
        )
        crop_path = crops_dir / f"{request_id}.png"
        crop.save(crop_path)
        crop_rgb = np.asarray(crop, dtype=np.uint8)

        stock_pixels = stock.processor(crop.copy())["pixel_values"]
        custom_pixels = custom.processor(crop.copy())["pixel_values"].numpy()
        pixel_diff = np.abs(stock_pixels - custom_pixels)

        print(
            f"COMPARE_CROP_BEGIN index={index}/{len(request_ids)} "
            f"request_id={request_id} size={crop.size}",
            flush=True,
        )
        stock_started = time.perf_counter()
        stock_text, stock_ids_raw = stock(
            image=crop.copy(), max_length=args.max_length
        )
        stock_wall_s = time.perf_counter() - stock_started
        stock_ids = [int(token) for token in stock_ids_raw]

        custom_started = time.perf_counter()
        custom_result = custom.generate_image(
            crop.copy(),
            max_length=args.max_length,
            decode_mode="eager",
            compile_backend="torchair",
            image_source=request_id,
        )
        custom_wall_s = time.perf_counter() - custom_started
        custom_ids = [int(token) for token in custom_result["generated_ids"]]
        saved_ids = [int(token) for token in source["token_ids"]][
            : args.max_length
        ]

        official_custom_diff = first_difference(stock_ids, custom_ids)
        saved_custom_diff = first_difference(saved_ids, custom_ids)
        record = {
            "request_id": request_id,
            "page": source["page"],
            "label": source["label"],
            "bbox": bbox,
            "crop_size": list(crop.size),
            "crop_path": str(crop_path),
            "crop_rgb_sha256": sha256_array(crop_rgb),
            "source_tokens": int(source["text_prefill_real_source_tokens"]),
            "max_length": int(args.max_length),
            "preprocess": {
                "stock_shape": list(stock_pixels.shape),
                "custom_shape": list(custom_pixels.shape),
                "exact": bool(np.array_equal(stock_pixels, custom_pixels)),
                "max_abs": float(pixel_diff.max()) if pixel_diff.size else 0.0,
                "mean_abs": float(pixel_diff.mean()) if pixel_diff.size else 0.0,
                "stock_sha256": sha256_array(stock_pixels),
                "custom_sha256": sha256_array(custom_pixels),
            },
            "official": {
                "wall_s": stock_wall_s,
                "token_ids": stock_ids,
                "token_count": len(stock_ids),
                "eos": bool(stock_ids and stock_ids[-1] == 2),
                "text": stock_text,
            },
            "custom_eager": {
                "wall_s": custom_wall_s,
                "token_ids": custom_ids,
                "token_count": len(custom_ids),
                "eos": bool(custom_ids and custom_ids[-1] == 2),
                "text": custom_result["text"],
                "timing_s": {
                    "prepare": custom_result["prep"]["prepare_total_s"],
                    "prefill": custom_result["ttft_s"],
                    "decode": custom_result["decode_s"],
                },
            },
            "saved_run": {
                "token_ids_prefix": saved_ids,
                "full_token_count": int(source["token_count"]),
                "full_eos": bool(source["token_ids"][-1] == 2),
                "text_prefix": source["text"][:1000],
            },
            "comparison": {
                "official_vs_custom_token_exact": official_custom_diff is None,
                "official_vs_custom_first_difference": official_custom_diff,
                "official_vs_custom_text_exact": stock_text
                == custom_result["text"],
                "saved_prefix_vs_custom_token_exact": saved_custom_diff is None,
                "saved_prefix_vs_custom_first_difference": saved_custom_diff,
            },
        }
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(
            f"COMPARE_CROP_END request_id={request_id} "
            f"preprocess_exact={record['preprocess']['exact']} "
            f"official_tokens={len(stock_ids)} official_eos={record['official']['eos']} "
            f"custom_tokens={len(custom_ids)} custom_eos={record['custom_eager']['eos']} "
            f"token_exact={record['comparison']['official_vs_custom_token_exact']} "
            f"first_diff={official_custom_diff} stock_s={stock_wall_s:.3f} "
            f"custom_s={custom_wall_s:.3f}",
            flush=True,
        )

    summary = {
        "status": "ok",
        "request_ids": list(request_ids),
        "max_length": int(args.max_length),
        "stock_setup_s": stock_setup_s,
        "custom_model_load_s": custom.model_load_s,
        "count": len(records),
        "preprocess_exact_count": sum(
            bool(record["preprocess"]["exact"]) for record in records
        ),
        "official_vs_custom_token_exact_count": sum(
            bool(record["comparison"]["official_vs_custom_token_exact"])
            for record in records
        ),
        "saved_prefix_vs_custom_token_exact_count": sum(
            bool(record["comparison"]["saved_prefix_vs_custom_token_exact"])
            for record in records
        ),
        "official_eos_count": sum(
            bool(record["official"]["eos"]) for record in records
        ),
        "custom_eos_count": sum(
            bool(record["custom_eager"]["eos"]) for record in records
        ),
        "comparison_jsonl": str(results_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("UNIREC_SAVED_TEXT_CROP_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
