#!/usr/bin/env python3
"""Run official OpenDoc page parsing with the local eager UniRec recognizer.

The OpenDoc layout detector, crop construction, label routing, postprocessing,
reading order, and output writers remain unchanged.  Only the crop recognizer
object is replaced.  ``--mode compare`` feeds every exact in-memory PIL crop to
both the stock ONNX recognizer and the local NPU implementation before returning
the local result to the official page assembler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from modeling_optimized_unirec import OptimizedUniRecRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--stock-encoder", type=Path, required=True)
    parser.add_argument("--stock-decoder", type=Path, required=True)
    parser.add_argument("--stock-tokenizer-mapping", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("custom", "compare"), default="compare")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--layout-threshold", type=float, default=0.5)
    return parser.parse_args()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _first_token_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _postprocess_text(markdown_converter: Any, text: str, block_label: str) -> str:
    if "table" in block_label:
        return markdown_converter._handle_table(text)
    if "formula" in block_label and block_label != "formula_number":
        return markdown_converter._handle_formula(text)
    return markdown_converter._handle_text(text)


class OpenDocUniRecAdapter:
    """OpenDoc's callable recognizer contract backed by OptimizedUniRecRunner."""

    def __init__(
        self,
        *,
        custom_runner: OptimizedUniRecRunner,
        stock_recognizer: Any | None,
        trace_path: Path,
        markdown_converter: Any,
    ) -> None:
        self.custom_runner = custom_runner
        self.stock_recognizer = stock_recognizer
        self.trace_path = trace_path
        self.markdown_converter = markdown_converter
        self.page = "<unset>"
        self.crop_index = 0
        self.records: list[dict[str, Any]] = []
        self._pending_record: dict[str, Any] | None = None
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")

    def begin_page(self, image_path: Path) -> None:
        self.page = image_path.name
        self.crop_index = 0

    def __call__(self, *, image: Any, max_length: int, **_: Any) -> tuple[str, list[int]]:
        self.crop_index += 1
        crop_name = f"{self.page}:crop_{self.crop_index:04d}"
        rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        record: dict[str, Any] = {
            "page": self.page,
            "crop_index": self.crop_index,
            "crop_name": crop_name,
            "crop_size": [int(image.width), int(image.height)],
            "crop_rgb_sha256": _sha256_array(rgb_array),
            "max_length": int(max_length),
        }

        if self.stock_recognizer is not None:
            stock_pixels = self.stock_recognizer.processor(image.copy())["pixel_values"]
            custom_pixels = self.custom_runner.processor(image.copy())["pixel_values"].numpy()
            preprocess_diff = np.abs(stock_pixels - custom_pixels)
            record["preprocess"] = {
                "stock_shape": list(stock_pixels.shape),
                "custom_shape": list(custom_pixels.shape),
                "stock_sha256": _sha256_array(stock_pixels),
                "custom_sha256": _sha256_array(custom_pixels),
                "exact": bool(np.array_equal(stock_pixels, custom_pixels)),
                "max_abs": float(preprocess_diff.max()) if preprocess_diff.size else 0.0,
                "mean_abs": float(preprocess_diff.mean()) if preprocess_diff.size else 0.0,
            }
            stock_started = time.perf_counter()
            stock_text, stock_ids = self.stock_recognizer(
                image=image.copy(), max_length=max_length
            )
            record["stock"] = {
                "wall_s": time.perf_counter() - stock_started,
                "text": stock_text,
                "token_ids": [int(token) for token in stock_ids],
                "token_count": len(stock_ids),
            }

        custom_result = self.custom_runner.generate_image(
            image.copy(),
            max_length=max_length,
            decode_mode="eager",
            compile_backend="torchair",
            image_source=crop_name,
        )
        custom_ids = [int(token) for token in custom_result["generated_ids"]]
        record["custom"] = {
            "text": custom_result["text"],
            "token_ids": custom_ids,
            "token_count": len(custom_ids),
            "timing_s": {
                "prepare": custom_result["prep"]["prepare_total_s"],
                "prefill": custom_result["ttft_s"],
                "decode": custom_result["decode_s"],
                "total": custom_result["total_latency_s"],
            },
            "decode_tokens_per_s": custom_result["decode_tokens_per_s"],
            "processed_image_size": custom_result["prep"]["processed_image_size"],
            "encoder_seq_len_hint": custom_result["prep"]["encoder_seq_len_hint"],
        }

        if self.stock_recognizer is not None:
            stock_ids = record["stock"]["token_ids"]
            first_diff = _first_token_difference(stock_ids, custom_ids)
            record["comparison"] = {
                "token_exact": first_diff is None,
                "first_token_difference": first_diff,
                "raw_text_exact": record["stock"]["text"] == custom_result["text"],
            }

        self._pending_record = record
        return custom_result["text"], custom_ids

    def finish_block(self, block_label: str, custom_postprocessed_text: str) -> None:
        if self._pending_record is None:
            raise RuntimeError("No pending recognition record to finalize")
        record = self._pending_record
        record["block_label"] = block_label
        record["custom"]["postprocessed_text"] = custom_postprocessed_text
        if self.stock_recognizer is not None:
            stock_postprocessed = _postprocess_text(
                self.markdown_converter,
                record["stock"]["text"],
                block_label,
            )
            record["stock"]["postprocessed_text"] = stock_postprocessed
            record["comparison"]["postprocessed_text_exact"] = (
                stock_postprocessed == custom_postprocessed_text
            )
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self._pending_record = None
        comparison = record.get("comparison")
        comparison_text = (
            "custom_only"
            if comparison is None
            else f"token_exact={comparison['token_exact']} first_diff={comparison['first_token_difference']}"
        )
        print(
            f"UNIREC_CROP_END page={self.page} crop={self.crop_index} "
            f"label={block_label} custom_tokens={record['custom']['token_count']} "
            f"custom_s={record['custom']['timing_s']['total']:.3f} {comparison_text}",
            flush=True,
        )


def _install_block_trace(pipeline: Any, adapter: OpenDocUniRecAdapter) -> None:
    original = pipeline._recognize_single_block

    def traced_recognize_single_block(
        self: Any,
        block_img: np.ndarray,
        block_label: str,
        block_index: int,
        max_length: int,
    ) -> tuple[int, str]:
        result_index, text = original(block_img, block_label, block_index, max_length)
        adapter.finish_block(block_label, text)
        return result_index, text

    pipeline._recognize_single_block = types.MethodType(
        traced_recognize_single_block, pipeline
    )


def main() -> None:
    args = parse_args()
    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required_paths = [
        openocr_root / "tools/infer_doc_onnx.py",
        model_path / "model.pth",
        args.layout_model.expanduser().resolve(),
        args.stock_encoder.expanduser().resolve(),
        args.stock_decoder.expanduser().resolve(),
        args.stock_tokenizer_mapping.expanduser().resolve(),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)

    sys.path.insert(0, str(openocr_root))
    from tools import infer_doc_onnx
    from tools.utils.utility import get_image_file_list

    image_paths = [Path(path).resolve() for path in sorted(get_image_file_list(str(input_path)))]
    image_paths = image_paths[args.offset :]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No input images found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "recognition_comparison.jsonl"
    print(
        f"OPENDOC_CUSTOM_SETUP_BEGIN mode={args.mode} pages={len(image_paths)} "
        f"max_parallel_blocks=1",
        flush=True,
    )
    setup_started = time.perf_counter()
    pipeline = infer_doc_onnx.OpenDocONNX(
        layout_model_path=str(args.layout_model.expanduser().resolve()),
        unirec_encoder_path=str(args.stock_encoder.expanduser().resolve()),
        unirec_decoder_path=str(args.stock_decoder.expanduser().resolve()),
        tokenizer_mapping_path=str(args.stock_tokenizer_mapping.expanduser().resolve()),
        use_gpu=False,
        layout_threshold=args.layout_threshold,
        use_layout_detection=True,
        auto_download=False,
        max_parallel_blocks=1,
    )
    stock_recognizer = pipeline.vlm_recognizer if args.mode == "compare" else None
    custom_runner = OptimizedUniRecRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=None,
    )
    adapter = OpenDocUniRecAdapter(
        custom_runner=custom_runner,
        stock_recognizer=stock_recognizer,
        trace_path=trace_path,
        markdown_converter=infer_doc_onnx.markdown_converter,
    )
    pipeline.vlm_recognizer = adapter
    _install_block_trace(pipeline, adapter)
    setup_s = time.perf_counter() - setup_started
    print(f"OPENDOC_CUSTOM_SETUP_END setup_s={setup_s:.3f}", flush=True)

    page_records: list[dict[str, Any]] = []
    for page_index, image_path in enumerate(image_paths, start=1):
        adapter.begin_page(image_path)
        page_started = time.perf_counter()
        print(
            f"OPENDOC_CUSTOM_PAGE_BEGIN index={page_index}/{len(image_paths)} "
            f"image={image_path.name}",
            flush=True,
        )
        result = pipeline(
            img_path=str(image_path),
            max_length=args.max_length,
            merge_layout_blocks=True,
        )
        pipeline.save_to_json(result, str(output_dir))
        pipeline.save_to_markdown(result, str(output_dir))
        page_s = time.perf_counter() - page_started
        page_records.append(
            {
                "image": str(image_path),
                "wall_s": page_s,
                "recognized_crops": adapter.crop_index,
            }
        )
        print(
            f"OPENDOC_CUSTOM_PAGE_END index={page_index}/{len(image_paths)} "
            f"wall_s={page_s:.3f} recognized_crops={adapter.crop_index}",
            flush=True,
        )

    comparisons = [record["comparison"] for record in adapter.records if "comparison" in record]
    summary = {
        "status": "ok",
        "mode": args.mode,
        "openocr_root": str(openocr_root),
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "max_parallel_blocks": 1,
        "max_length": args.max_length,
        "setup_s": setup_s,
        "pipeline_wall_s": sum(record["wall_s"] for record in page_records),
        "page_count": len(page_records),
        "crop_count": len(adapter.records),
        "preprocess_exact_count": sum(
            int(record.get("preprocess", {}).get("exact", False))
            for record in adapter.records
        ),
        "token_exact_count": sum(int(item["token_exact"]) for item in comparisons),
        "raw_text_exact_count": sum(int(item["raw_text_exact"]) for item in comparisons),
        "postprocessed_text_exact_count": sum(
            int(item["postprocessed_text_exact"]) for item in comparisons
        ),
        "pages": page_records,
        "trace_path": str(trace_path),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("OPENDOC_CUSTOM_RUN_END " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
