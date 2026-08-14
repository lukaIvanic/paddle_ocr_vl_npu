#!/usr/bin/env python3
"""Stream a shared crop list through official ONNX and custom NPU lanes.

The two recognizers run independently so CPU ONNX decode overlaps NPU decode.
Each lane appends one JSON object per completed crop.  The coordinator appends
a third record as soon as both lanes have completed the same crop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from compare_saved_text_crops import (
    first_difference,
    read_trace,
    recreate_crop,
    sha256_array,
)
from modeling_optimized_unirec import OptimizedUniRecRunner


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
    parser.add_argument("--random-count", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=20260814)
    parser.add_argument(
        "--label-family",
        default="text",
        help="Layout label before its numeric suffix; use 'all' for every crop.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def label_family(label: str) -> str:
    return re.sub(r"_[0-9]+$", "", label)


def select_request_ids(
    trace: dict[str, dict[str, Any]], args: argparse.Namespace
) -> list[str]:
    if args.request_id:
        missing = [item for item in args.request_id if item not in trace]
        if missing:
            raise KeyError(f"request IDs are missing from source trace: {missing}")
        return list(dict.fromkeys(args.request_id))
    candidates = sorted(
        request_id
        for request_id, record in trace.items()
        if args.label_family == "all"
        or label_family(str(record["label"])) == args.label_family
    )
    if args.random_count < 1:
        raise ValueError("--random-count must be positive")
    if args.random_count > len(candidates):
        raise ValueError(
            f"requested {args.random_count} crops from {len(candidates)} candidates"
        )
    return random.Random(args.random_seed).sample(candidates, args.random_count)


def append_json(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def official_lane(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    output_path: Path,
    events: queue.Queue[tuple[str, str, Any]],
) -> dict[str, Any]:
    try:
        sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
        from tools.infer_unirec_onnx import UniRecONNX

        setup_started = time.perf_counter()
        runner = UniRecONNX(
            encoder_path=str(args.stock_encoder.expanduser().resolve()),
            decoder_path=str(args.stock_decoder.expanduser().resolve()),
            mapping_path=str(args.stock_tokenizer_mapping.expanduser().resolve()),
            use_gpu=False,
            auto_download=False,
        )
        setup_s = time.perf_counter() - setup_started
        wall_started = time.perf_counter()
        with output_path.open("a", encoding="utf-8") as handle:
            for index, item in enumerate(items, start=1):
                with Image.open(item["crop_path"]) as image:
                    crop = image.convert("RGB")
                pixels = runner.processor(crop.copy())["pixel_values"]
                started = time.perf_counter()
                text, token_ids_raw = runner(
                    image=crop.copy(), max_length=args.max_length
                )
                wall_s = time.perf_counter() - started
                token_ids = [int(token) for token in token_ids_raw]
                record = {
                    "request_id": item["request_id"],
                    "index": index,
                    "wall_s": wall_s,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "eos": bool(token_ids and token_ids[-1] == 2),
                    "text": text,
                    "preprocess_shape": list(pixels.shape),
                    "preprocess_sha256": sha256_array(pixels),
                }
                append_json(handle, record)
                print(
                    "OFFICIAL_DONE "
                    f"index={index}/{len(items)} request_id={item['request_id']} "
                    f"tokens={len(token_ids)} eos={record['eos']} wall_s={wall_s:.3f}",
                    flush=True,
                )
                events.put(("result", "official", record))
        result = {
            "setup_s": setup_s,
            "lane_wall_s": time.perf_counter() - wall_started,
            "count": len(items),
        }
        events.put(("done", "official", result))
        return result
    except BaseException as error:
        events.put(("error", "official", repr(error)))
        raise


def custom_lane(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    output_path: Path,
    events: queue.Queue[tuple[str, str, Any]],
) -> dict[str, Any]:
    try:
        if args.device.startswith("npu"):
            import torch_npu

            torch_npu.npu.set_compile_mode(jit_compile=False)
        runner = OptimizedUniRecRunner(
            model_path=args.model_path.expanduser().resolve(),
            device=args.device,
            dtype=args.dtype,
        )
        wall_started = time.perf_counter()
        with output_path.open("a", encoding="utf-8") as handle:
            for index, item in enumerate(items, start=1):
                with Image.open(item["crop_path"]) as image:
                    crop = image.convert("RGB")
                pixels = runner.processor(crop.copy())["pixel_values"].numpy()
                started = time.perf_counter()
                result = runner.generate_image(
                    crop.copy(),
                    max_length=args.max_length,
                    decode_mode="eager",
                    compile_backend="torchair",
                    image_source=item["request_id"],
                )
                wall_s = time.perf_counter() - started
                token_ids = [int(token) for token in result["generated_ids"]]
                record = {
                    "request_id": item["request_id"],
                    "index": index,
                    "wall_s": wall_s,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "eos": bool(token_ids and token_ids[-1] == 2),
                    "text": result["text"],
                    "preprocess_shape": list(pixels.shape),
                    "preprocess_sha256": sha256_array(pixels),
                    "timing_s": {
                        "prepare": result["prep"]["prepare_total_s"],
                        "prefill": result["ttft_s"],
                        "decode": result["decode_s"],
                    },
                }
                append_json(handle, record)
                print(
                    "CUSTOM_DONE "
                    f"index={index}/{len(items)} request_id={item['request_id']} "
                    f"tokens={len(token_ids)} eos={record['eos']} wall_s={wall_s:.3f}",
                    flush=True,
                )
                events.put(("result", "custom", record))
        result = {
            "setup_s": runner.model_load_s,
            "lane_wall_s": time.perf_counter() - wall_started,
            "count": len(items),
        }
        events.put(("done", "custom", result))
        return result
    except BaseException as error:
        events.put(("error", "custom", repr(error)))
        raise


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

    output_dir = args.output_dir.expanduser().resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    official_path = output_dir / "official.jsonl"
    custom_path = output_dir / "custom.jsonl"
    comparison_path = output_dir / "comparison.jsonl"
    for path in (official_path, custom_path, comparison_path):
        path.write_text("", encoding="utf-8")

    trace = read_trace(args.source_output / "recognition_trace.jsonl")
    request_ids = select_request_ids(trace, args)
    items: list[dict[str, Any]] = []
    for index, request_id in enumerate(request_ids, start=1):
        source = trace[request_id]
        crop, bbox = recreate_crop(
            source_output=args.source_output.expanduser().resolve(),
            dataset_root=args.dataset_root.expanduser().resolve(),
            trace_record=source,
        )
        crop_path = crops_dir / f"{request_id}.png"
        crop.save(crop_path)
        items.append(
            {
                "index": index,
                "request_id": request_id,
                "page": source["page"],
                "label": source["label"],
                "bbox": bbox,
                "crop_size": list(crop.size),
                "crop_path": str(crop_path),
                "saved_token_ids": [int(token) for token in source["token_ids"]][
                    : args.max_length
                ],
            }
        )
    manifest = {
        "random_seed": args.random_seed,
        "random_count": len(items),
        "label_family": args.label_family,
        "max_length": args.max_length,
        "request_ids": request_ids,
        "items": items,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "PARALLEL_COMPARE_BEGIN "
        f"count={len(items)} seed={args.random_seed} label_family={args.label_family} "
        f"max_length={args.max_length} output_dir={output_dir}",
        flush=True,
    )

    event_queue: queue.Queue[tuple[str, str, Any]] = queue.Queue()
    pending: dict[str, dict[str, dict[str, Any]]] = {}
    done: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    item_by_id = {item["request_id"]: item for item in items}
    run_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="unirec_compare") as pool:
        futures = {
            "official": pool.submit(
                official_lane,
                args=args,
                items=items,
                output_path=official_path,
                events=event_queue,
            ),
            "custom": pool.submit(
                custom_lane,
                args=args,
                items=items,
                output_path=custom_path,
                events=event_queue,
            ),
        }
        with comparison_path.open("a", encoding="utf-8") as comparison_handle:
            while len(done) < 2:
                event_type, lane, payload = event_queue.get()
                if event_type == "error":
                    raise RuntimeError(f"{lane} lane failed: {payload}")
                if event_type == "done":
                    done[lane] = payload
                    print(
                        f"LANE_DONE lane={lane} count={payload['count']} "
                        f"wall_s={payload['lane_wall_s']:.3f}",
                        flush=True,
                    )
                    continue
                request_id = str(payload["request_id"])
                slots = pending.setdefault(request_id, {})
                slots[lane] = payload
                if "official" not in slots or "custom" not in slots:
                    continue
                official = slots["official"]
                custom = slots["custom"]
                source = item_by_id[request_id]
                first_diff = first_difference(
                    official["token_ids"], custom["token_ids"]
                )
                saved_diff = first_difference(
                    source["saved_token_ids"], custom["token_ids"]
                )
                comparison = {
                    "request_id": request_id,
                    "index": source["index"],
                    "page": source["page"],
                    "label": source["label"],
                    "bbox": source["bbox"],
                    "crop_size": source["crop_size"],
                    "crop_path": source["crop_path"],
                    "official": official,
                    "custom": custom,
                    "comparison": {
                        "token_exact": first_diff is None,
                        "first_difference": first_diff,
                        "text_exact": official["text"] == custom["text"],
                        "preprocess_exact": official["preprocess_sha256"]
                        == custom["preprocess_sha256"],
                        "saved_prefix_vs_custom_token_exact": saved_diff is None,
                        "saved_prefix_vs_custom_first_difference": saved_diff,
                    },
                }
                append_json(comparison_handle, comparison)
                comparisons.append(comparison)
                del pending[request_id]
                marker = "COMPARE_MATCH" if first_diff is None else "COMPARE_DIFFERENCE"
                print(
                    f"{marker} paired={len(comparisons)}/{len(items)} "
                    f"index={source['index']} request_id={request_id} "
                    f"first_diff={first_diff} official_tokens={official['token_count']} "
                    f"custom_tokens={custom['token_count']}",
                    flush=True,
                )
        for future in futures.values():
            future.result()

    summary = {
        "status": "ok",
        "count": len(comparisons),
        "random_seed": args.random_seed,
        "label_family": args.label_family,
        "max_length": args.max_length,
        "wall_s": time.perf_counter() - run_started,
        "lanes": done,
        "token_exact_count": sum(
            bool(row["comparison"]["token_exact"]) for row in comparisons
        ),
        "text_exact_count": sum(
            bool(row["comparison"]["text_exact"]) for row in comparisons
        ),
        "difference_count": sum(
            not bool(row["comparison"]["token_exact"]) for row in comparisons
        ),
        "official_jsonl": str(official_path),
        "custom_jsonl": str(custom_path),
        "comparison_jsonl": str(comparison_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PARALLEL_COMPARE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
