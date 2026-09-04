#!/usr/bin/env python3
"""Run live U8 drafts with cross-table batched adaptive verification."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

import table_row_ocr_lab as row_lab  # noqa: E402
import table_spec_decode_lab as fixed_lab  # noqa: E402
import table_spec_live_u8_adaptive_lab as live_lab  # noqa: E402
from paddleocr_vl.serving.table_speculative import TableDraftMatcher  # noqa: E402
from paddleocr_vl.serving.table_speculative_batch import (  # noqa: E402
    BatchedAdaptiveKTableSpeculativeDecodeRuntime,
)
from pipeline.layout_output import normalize_recognition_text  # noqa: E402
from utils.timing import synchronize  # noqa: E402


DEFAULT_COMPACT_VOCAB = (
    EXPERIMENT_ROOT
    / "presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
)
DEFAULT_REQUEST_IDS = (
    "page_000287_table_box_id_8",
    "page_001367_table_51",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=fixed_lab.DEFAULT_TARGETS)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--draft-cache-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--k-values", default="7,15,31,63")
    parser.add_argument("--initial-k", type=int, default=15)
    parser.add_argument("--draft-row-count", type=int, default=8)
    parser.add_argument("--row-overlap-px", type=int, default=3)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument(
        "--decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument(
        "--verifier-optimization",
        default="combined_apply_spec_prefetch_mrope",
    )
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair",
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=(
            REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
        ),
    )
    args = parser.parse_args()
    if args.passes <= 0:
        parser.error("--passes must be positive")
    selected = args.request_id or list(DEFAULT_REQUEST_IDS)
    if len(selected) != args.batch_size:
        parser.error("the request-id count must equal --batch-size")
    args.request_id = selected
    return args


def _live_compatible_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        b1_decode_optimization=args.decode_optimization,
        b1_decode_vocab_token_ids=args.decode_vocab_token_ids,
        b1_cache_length=args.cache_length,
        b1_max_new_tokens=args.max_new_tokens,
        b1_vision_buckets="4096",
        b1_text_buckets="128,256,512,1024,1152,1312",
        draft_decode_optimization=args.decode_optimization,
        draft_decode_vocab_token_ids=args.decode_vocab_token_ids,
        draft_cache_length=args.draft_cache_length,
        draft_row_count=args.draft_row_count,
        draft_batch_size=args.batch_size * args.draft_row_count,
        draft_vision_packing="greedy",
        draft_vision_pack_target=2304,
        draft_prefill_layout="packed_b1",
        draft_batched_vision_shapes="8x640,8x768",
        draft_batched_text_shape="8x256",
        row_overlap_px=args.row_overlap_px,
        compact_uint8_preprocess=False,
        image_resize_backend="pillow",
        target_cpu_delay_ms=0.0,
        overlap_target_cpu_preparation=False,
        k_values=args.k_values,
        initial_k=args.initial_k,
        verifier_optimization=args.verifier_optimization,
        per_call_device_timing=False,
        allow_compile=True,
        token_selection="greedy",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        vision_buckets="256,384,512,640,768,1408,1920,2048,2304,2944,4096",
        text_buckets="128,256,384,512,768",
        decode_cache_dir=args.decode_cache_dir,
        vision_cache_dir=args.vision_cache_dir,
        text_cache_dir=args.text_cache_dir,
        text_packed_cache_dir=args.text_packed_cache_dir,
        vision_batched_cache_dir=None,
        text_batched_cache_dir=None,
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_batched_drafts(
    recognizer: Any,
    prepared: list[dict[str, Any]],
    *,
    draft_row_count: int,
    pass_index: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], float]:
    request_map: dict[str, tuple[int, int]] = {}
    all_requests = []
    rows_by_table: list[list[dict[str, Any]]] = [[] for _ in prepared]
    for table_index, row in enumerate(prepared):
        for row_index, request in enumerate(row["row_requests"]):
            request_map[str(request.request_id)] = (table_index, row_index)
            all_requests.append(request)

    def emit(result: Any) -> None:
        table_index, row_index = request_map[str(result.request_id)]
        source_row = prepared[table_index]
        payload = asdict(result)
        payload["raw_text"] = payload["text"]
        payload["row_index"] = row_index
        payload["row_y"] = list(source_row["row_crops"][row_index][:2])
        rows_by_table[table_index].append(payload)

    started = time.perf_counter()
    schedule = recognizer.run(
        all_requests,
        schedule_id=f"live-u{draft_row_count}:batched:pass{pass_index}",
        emit_result=emit,
    )
    wall_s = time.perf_counter() - started
    schedule_payload = asdict(schedule)
    drafts: dict[str, dict[str, Any]] = {}
    strategy_name = f"uniform_{draft_row_count}_snapped"
    for table_index, source_row in enumerate(prepared):
        source = source_row["source"]
        rows = rows_by_table[table_index]
        rows.sort(key=lambda item: int(item["row_index"]))
        drafts[str(source["request_id"])] = {
            "request_id": str(source["request_id"]),
            "strategy": strategy_name,
            "page_name": source["page_name"],
            "rows": rows,
            "schedule": schedule_payload,
        }
    return drafts, schedule_payload, wall_s


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    compat = _live_compatible_args(args)
    records = fixed_lab.read_jsonl(args.targets)
    by_id = {str(record["request_id"]): record for record in records}
    selected = [by_id[request_id] for request_id in args.request_id]

    print("TABLE_SPEC_BATCH setup=draft_recognizer", flush=True)
    draft_recognizer = row_lab.build_recognizer(live_lab._draft_args(compat))
    print("TABLE_SPEC_BATCH setup=target_recognizer", flush=True)
    target_recognizer = fixed_lab.build_recognizer(live_lab._b1_args(compat))
    k_values = tuple(
        sorted({int(value.strip()) for value in args.k_values.split(",") if value.strip()})
    )
    cache_roots = {value: args.decode_cache_dir.resolve() for value in k_values}
    print(
        f"TABLE_SPEC_BATCH setup=verifiers batch=B{args.batch_size} "
        f"k={','.join(str(value) for value in k_values)}",
        flush=True,
    )
    runtime = BatchedAdaptiveKTableSpeculativeDecodeRuntime(
        target_recognizer,
        batch_size=args.batch_size,
        k_values=k_values,
        initial_k=args.initial_k,
        cache_roots=cache_roots,
        verifier_optimization=args.verifier_optimization,
        decode_cache_root=args.decode_cache_dir.resolve(),
    )

    report: dict[str, Any] = {
        "format": "paddleocr_table_spec_batched_live_lab_v1",
        "status": "running",
        "configuration": {
            "request_ids": list(args.request_id),
            "batch_size": args.batch_size,
            "passes": args.passes,
            "k_values": list(k_values),
            "initial_k": args.initial_k,
            "decode_optimization": args.decode_optimization,
            "verifier_optimization": args.verifier_optimization,
        },
        "passes": [],
    }
    _write(args.output, report)

    for pass_index in range(args.passes):
        pass_started = time.perf_counter()
        prepared_batch: list[tuple[Any, TableDraftMatcher]] = []
        preparation_rows: list[dict[str, Any]] = []
        source_preparation: list[dict[str, Any]] = []
        for source in selected:
            request_id = str(source["request_id"])
            raw_image = live_lab.load_crop(source, args.images_dir)
            target_crop = live_lab._exact_target_crop_from_raw(source, raw_image)
            row_requests, row_crops, row_timing = live_lab._prepare_rows(
                source, raw_image, compat
            )
            target_request = fixed_lab.request_for(source, target_crop, compat)
            source_preparation.append(
                {
                    "source": source,
                    "request_id": request_id,
                    "row_requests": row_requests,
                    "row_crops": row_crops,
                    "row_timing_s": row_timing,
                    "target_request": target_request,
                }
            )

        drafts, draft_schedule, draft_batch_wall_s = _run_batched_drafts(
            draft_recognizer,
            source_preparation,
            draft_row_count=args.draft_row_count,
            pass_index=pass_index,
        )
        for source_row in source_preparation:
            source = source_row["source"]
            request_id = str(source["request_id"])
            draft = drafts[request_id]
            target_prefill_started = time.perf_counter()
            prefilled = target_recognizer.prefill_one(source_row["target_request"])
            target_prefill_wall_s = time.perf_counter() - target_prefill_started
            matcher = TableDraftMatcher(
                draft,
                target_recognizer.tokenizer,
                eos_token_id=int(target_recognizer.model.config.eos_token_id),
                block_size=args.initial_k,
            )
            prepared_batch.append((prefilled, matcher))
            preparation_rows.append(
                {
                    "request_id": request_id,
                    "row_timing_s": source_row["row_timing_s"],
                    "target_prefill_wall_s": target_prefill_wall_s,
                    "draft": draft,
                    "target_prefill": {
                        "input_tokens": int(prefilled.input_tokens),
                        "timing_s": dict(prefilled.timing_s),
                        "device_stage_s": dict(prefilled.device_stage_s),
                    },
                }
            )

        verify_started = time.perf_counter()
        results = runtime.decode_many(
            prepared_batch,
            max_new_tokens=args.max_new_tokens,
        )
        verify_wall_s = time.perf_counter() - verify_started
        synchronize(target_recognizer.device)
        result_rows = []
        for source, result in zip(selected, results, strict=True):
            reference = fixed_lab.target_tokens(source)
            result_rows.append(
                {
                    "request_id": str(source["request_id"]),
                    "exact_saved_reference": result.token_ids == reference,
                    "first_saved_difference": fixed_lab.first_difference(
                        result.token_ids, reference
                    ),
                    "pred_html": normalize_recognition_text("table", result.text),
                    "result": result.to_dict(),
                }
            )
        pass_row = {
            "pass_index": pass_index,
            "measured": pass_index == args.passes - 1,
            "wall_s": time.perf_counter() - pass_started,
            "draft_batch_wall_s": draft_batch_wall_s,
            "draft_schedule": draft_schedule,
            "verify_wall_s": verify_wall_s,
            "preparation": preparation_rows,
            "results": result_rows,
            "runtime_summary": dict(runtime.last_summary),
        }
        report["passes"].append(pass_row)
        _write(args.output, report)
        print(
            "TABLE_SPEC_BATCH_RESULT "
            f"pass={pass_index + 1}/{args.passes} "
            f"wall_s={pass_row['wall_s']:.3f} "
            f"verify_wall_s={verify_wall_s:.3f} "
            f"exact={sum(row['exact_saved_reference'] for row in result_rows)}"
            f"/{len(result_rows)}",
            flush=True,
        )

    report["status"] = "complete"
    report["measured"] = report["passes"][-1]
    _write(args.output, report)
    print(f"TABLE_SPEC_BATCH_COMPLETE output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
