#!/usr/bin/env python3
"""Run real whole-table B1 decoding with precomputed row-draft verification."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.model.text_prefill import parse_text_buckets
from paddleocr_vl.model.text_spec_verify import torchair_cache_dir_for_spec_shape
from paddleocr_vl.model.token_selection import TOKEN_SELECTION_CHOICES
from paddleocr_vl.model.vision_prefill import parse_vision_buckets
from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.table_speculative import (
    TableDraftMatcher,
    TableSpeculativeDecodeRuntime,
)
from paddleocr_vl.serving.types import RecognitionRequest
from pipeline.layout_output import normalize_recognition_text
from table_row_split_lab import load_crop


DEFAULT_TARGETS = Path(
    "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/whole/tables.jsonl"
)
DEFAULT_DRAFTS = Path(
    "tmp/09_persistent_page_engine/"
    "table_row_per_table_kv768_pack2304_355db8b/"
    "uniform_8_snapped/tables.jsonl"
)
DEFAULT_VISION_BUCKETS = "256,384,512,640,768,1408,1920,2048,2944,4096"
DEFAULT_TEXT_BUCKETS = "128,256,512,1024,1312"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all-tables", action="store_true")
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument(
        "--wrapper-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Once per table, allow the actual draft cell-opening token when "
            "the previous aligned cell matches and that token is in live top-k."
        ),
    )
    parser.add_argument("--wrapper-rescue-top-k", type=int, default=3)
    parser.add_argument(
        "--wrapper-rescue-formula-previous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the previous draft cell to be formula-like with matching content.",
    )
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--token-selection",
        choices=TOKEN_SELECTION_CHOICES,
        default="greedy",
        help=(
            "Experimental live-logit token-ID selection. "
            "Speculative verification supports only policies explicitly "
            "accepted by TableSpeculativeDecodeRuntime."
        ),
    )
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--vision-buckets", default=DEFAULT_VISION_BUCKETS)
    parser.add_argument("--text-buckets", default=DEFAULT_TEXT_BUCKETS)
    parser.add_argument(
        "--compare-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-compile", action="store_true")
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
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def target_tokens(record: dict[str, Any]) -> list[int]:
    rows = record.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"{record.get('request_id')}: target must have one row")
    tokens = [int(value) for value in rows[0].get("token_ids") or ()]
    if not tokens:
        raise ValueError(f"{record.get('request_id')}: target tokens are empty")
    return tokens


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        index = min(
            len(ordered) - 1,
            max(0, int(fraction * len(ordered) - 1e-12)),
        )
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def build_recognizer(args: argparse.Namespace) -> ContinuousRecognizer:
    return ContinuousRecognizer(
        model=str(args.model),
        dtype="fp16",
        decode_backend="torchair",
        decode_optimization="combined_apply_pse_sentinel",
        batch_size=1,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        token_selection=getattr(args, "token_selection", "greedy"),
        torchair_cache_dir=args.decode_cache_dir.resolve(),
        vision_backend="torchair",
        vision_attention="prompt_flash_attention",
        vision_promptfa_align_128=True,
        vision_mlp_intermediate_size=4352,
        vision_linear_weight_format="fractal_nz",
        vision_buckets=parse_vision_buckets(args.vision_buckets),
        vision_torchair_cache_dir=args.vision_cache_dir.resolve(),
        vision_padding="bucket",
        vision_packing="off",
        text_backend="torchair",
        text_buckets=parse_text_buckets(args.text_buckets),
        text_torchair_cache_dir=args.text_cache_dir.resolve(),
        text_padding="bucket",
        text_packing="off",
        text_pack_max_members=1,
        preprocessor_min_pixels=args.min_pixels,
        preprocessor_max_pixels=args.max_pixels,
    )


def exact_target_crop(record: dict[str, Any], images_dir: Path) -> Image.Image:
    crop = load_crop(record, images_dir)
    trim_box = record.get("trim_box_in_raw_crop")
    if trim_box is not None:
        crop = crop.crop(tuple(int(value) for value in trim_box))
    expected = tuple(int(value) for value in record["crop_size"])
    if crop.size != expected:
        raise ValueError(
            f"{record['request_id']}: reconstructed crop {crop.size} != {expected}"
        )
    return crop


def request_for(record: dict[str, Any], crop: Image.Image, args: argparse.Namespace) -> RecognitionRequest:
    return RecognitionRequest(
        request_id=str(record["request_id"]),
        crop=crop,
        prompt="Table Recognition:",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        source_crop_size=crop.size,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    targets = read_jsonl(args.targets)
    drafts = {record["request_id"]: record for record in read_jsonl(args.drafts)}
    if args.request_id:
        wanted = set(args.request_id)
        selected = [record for record in targets if record["request_id"] in wanted]
        missing = wanted - {record["request_id"] for record in selected}
        if missing:
            raise KeyError(f"unknown request IDs: {sorted(missing)}")
    else:
        selected = targets[args.offset :]
        if not args.all_tables:
            selected = selected[: args.limit]
    missing_drafts = [record["request_id"] for record in selected if record["request_id"] not in drafts]
    if missing_drafts:
        raise KeyError(f"missing draft records: {missing_drafts[:5]}")

    setup_started = time.perf_counter()
    recognizer = build_recognizer(args)
    spec_cache = torchair_cache_dir_for_spec_shape(
        args.decode_cache_dir,
        draft_length=args.draft_length,
        cache_length=args.cache_length,
        dtype=recognizer.dtype,
        device=recognizer.device,
        model_dir=recognizer.model_dir,
        linear_weight_format=str(recognizer.weight_format["effective_mode"]),
        optimization="combined_apply",
        token_selection=args.token_selection,
        preferred_token_id=recognizer.math_open_token_id,
        alternate_preferred_token_id=recognizer.math_slash_token_id,
        cell_start_token_ids=recognizer.table_cell_token_ids,
    )
    cache_hit = spec_cache.is_dir() and any(spec_cache.iterdir())
    if not cache_hit and not args.allow_compile:
        raise RuntimeError(
            f"missing D{args.draft_length}/KV{args.cache_length} verifier cache; "
            "rerun with --allow-compile"
        )
    print(
        f"TABLE_SPEC_PROGRESS setup=verifier cache={'hit' if cache_hit else 'compile'} "
        f"draft=D{args.draft_length} kv={args.cache_length}",
        flush=True,
    )
    spec_runtime = TableSpeculativeDecodeRuntime(
        recognizer,
        draft_length=args.draft_length,
        cache_root=args.decode_cache_dir.resolve(),
        wrapper_rescue=args.wrapper_rescue,
        wrapper_rescue_top_k=args.wrapper_rescue_top_k,
        wrapper_rescue_formula_previous=args.wrapper_rescue_formula_previous,
    )
    setup_s = time.perf_counter() - setup_started
    print(
        f"TABLE_SPEC_PROGRESS setup=complete wall_s={setup_s:.3f} "
        f"tables={len(selected)}",
        flush=True,
    )

    output_path = args.output_dir / "tables.jsonl"
    output_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for table_index, target in enumerate(selected, start=1):
        request_id = str(target["request_id"])
        crop = exact_target_crop(target, args.images_dir)
        reference_tokens = target_tokens(target)
        baseline_payload: dict[str, Any] | None = None
        if args.compare_baseline:
            baseline_results = []
            baseline_started = time.perf_counter()
            baseline_schedule = recognizer.run(
                [request_for(target, crop, args)],
                schedule_id=f"table-spec-baseline:{request_id}",
                emit_result=baseline_results.append,
            )
            if len(baseline_results) != 1:
                raise RuntimeError(
                    f"{request_id}: baseline emitted {len(baseline_results)} results"
                )
            baseline = baseline_results[0]
            baseline_payload = {
                "wall_s": time.perf_counter() - baseline_started,
                "token_ids": list(baseline.token_ids),
                "text": baseline.text,
                "pred_html": normalize_recognition_text("table", baseline.text),
                "stop_reason": baseline.stop_reason,
                "exact_saved_reference": list(baseline.token_ids) == reference_tokens,
                "first_saved_difference": first_difference(
                    list(baseline.token_ids), reference_tokens
                ),
                "schedule": asdict(baseline_schedule),
            }

        prefill_started = time.perf_counter()
        prefilled = recognizer.prefill_one(request_for(target, crop, args))
        prefill_wall_s = time.perf_counter() - prefill_started
        matcher = TableDraftMatcher(
            drafts[request_id],
            recognizer.tokenizer,
            eos_token_id=int(recognizer.model.config.eos_token_id),
            block_size=args.draft_length,
        )
        spec_result = spec_runtime.decode(
            prefilled,
            matcher,
            max_new_tokens=args.max_new_tokens,
        )
        draft_generation_wall_s = float(
            (drafts[request_id].get("timing_s") or {}).get(
                "table_row_ocr_e2e",
                0.0,
            )
            or 0.0
        )
        saved_baseline_wall_s = float(
            (target.get("timing_s") or {}).get("table_row_ocr_e2e", 0.0)
            or 0.0
        )
        target_spec_wall_s = prefill_wall_s + spec_result.wall_s
        composed_pipeline_wall_s = draft_generation_wall_s + target_spec_wall_s
        payload = {
            "request_id": request_id,
            "page_name": target["page_name"],
            "crop_size": list(crop.size),
            "input_tokens": int(prefilled.input_tokens),
            "projected_image_tokens": int(prefilled.projected_image_tokens),
            "saved_reference_tokens": len(reference_tokens),
            "prefill_wall_s": prefill_wall_s,
            "draft_generation_wall_s": draft_generation_wall_s,
            "target_spec_wall_s": target_spec_wall_s,
            "composed_pipeline_wall_s": composed_pipeline_wall_s,
            "saved_baseline_wall_s": saved_baseline_wall_s,
            "composed_speedup_vs_saved_baseline": (
                saved_baseline_wall_s / composed_pipeline_wall_s
                if composed_pipeline_wall_s > 0.0 and saved_baseline_wall_s > 0.0
                else None
            ),
            "speculative": spec_result.to_dict(),
            "gt_html": target.get("gt_html"),
            "pred_html": normalize_recognition_text("table", spec_result.text),
            "baseline": baseline_payload,
            "exact_saved_reference": spec_result.token_ids == reference_tokens,
            "first_saved_difference": first_difference(
                spec_result.token_ids, reference_tokens
            ),
            "exact_live_baseline": (
                None
                if baseline_payload is None
                else spec_result.token_ids == baseline_payload["token_ids"]
            ),
            "first_live_difference": (
                None
                if baseline_payload is None
                else first_difference(spec_result.token_ids, baseline_payload["token_ids"])
            ),
        }
        records.append(payload)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(
            f"TABLE_SPEC_RESULT table={table_index}/{len(selected)} id={request_id} "
            f"tokens={len(spec_result.token_ids)} calls={spec_result.target_calls} "
            f"spec={spec_result.speculative_calls} fallback={spec_result.fallback_calls} "
            f"accepted={spec_result.accepted_draft_tokens}/{spec_result.proposed_draft_tokens} "
            f"decode_s={spec_result.wall_s:.3f} "
            f"exact_live={payload['exact_live_baseline']} "
            f"exact_saved={payload['exact_saved_reference']}",
            flush=True,
        )

    run_wall_s = time.perf_counter() - run_started
    generated_target_tokens = sum(
        max(0, len(record["speculative"]["token_ids"]) - 1)
        for record in records
    )
    composed_pipeline = [record["composed_pipeline_wall_s"] for record in records]
    saved_baseline = [record["saved_baseline_wall_s"] for record in records]
    per_table_speedup = [
        record["composed_speedup_vs_saved_baseline"]
        for record in records
        if record["composed_speedup_vs_saved_baseline"] is not None
    ]
    summary = {
        "status": "complete",
        "configuration": {
            "draft_length": args.draft_length,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "compare_baseline": args.compare_baseline,
            "wrapper_rescue": args.wrapper_rescue,
            "wrapper_rescue_top_k": args.wrapper_rescue_top_k,
            "verifier_cache_was_warm": cache_hit,
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "latency_composition": (
                "saved measured draft generation wall plus live target "
                "prefill and speculative decode wall"
            ),
            "recognizer": recognizer.configuration(),
            "verifier": spec_runtime.verify.metadata,
        },
        "setup_s": setup_s,
        "run_wall_s": run_wall_s,
        "tables": len(records),
        "exact_saved_reference": sum(record["exact_saved_reference"] for record in records),
        "exact_live_baseline": sum(
            record["exact_live_baseline"] is True for record in records
        ),
        "target_calls": sum(
            record["speculative"]["target_calls"] for record in records
        ),
        "speculative_calls": sum(
            record["speculative"]["speculative_calls"] for record in records
        ),
        "fully_accepted_speculative_calls": sum(
            record["speculative"]["fully_accepted_speculative_calls"]
            for record in records
        ),
        "rejected_speculative_calls": sum(
            record["speculative"]["rejected_speculative_calls"]
            for record in records
        ),
        "fallback_calls": sum(
            record["speculative"]["fallback_calls"] for record in records
        ),
        "accepted_draft_tokens": sum(
            record["speculative"]["accepted_draft_tokens"] for record in records
        ),
        "proposed_draft_tokens": sum(
            record["speculative"]["proposed_draft_tokens"] for record in records
        ),
        "generated_target_tokens": generated_target_tokens,
        "spec_decode_wall_s": sum(record["speculative"]["wall_s"] for record in records),
        "prefill_wall_s": sum(record["prefill_wall_s"] for record in records),
        "draft_generation_wall_s": sum(record["draft_generation_wall_s"] for record in records),
        "composed_pipeline_wall_s": sum(composed_pipeline),
        "saved_baseline_wall_s": sum(saved_baseline),
        "target_call_reduction": (
            generated_target_tokens
            / sum(record["speculative"]["target_calls"] for record in records)
        ),
        "accepted_fraction_of_proposed": (
            sum(record["speculative"]["accepted_draft_tokens"] for record in records)
            / sum(record["speculative"]["proposed_draft_tokens"] for record in records)
        ),
        "composed_speedup_vs_saved_baseline": (
            sum(saved_baseline) / sum(composed_pipeline)
            if sum(composed_pipeline) > 0.0
            else None
        ),
        "per_table_composed_wall_s": distribution(composed_pipeline),
        "per_table_saved_baseline_wall_s": distribution(saved_baseline),
        "per_table_composed_speedup": distribution(per_table_speedup),
        "records": str(output_path),
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(
        f"TABLE_SPEC_COMPLETE tables={len(records)} run_wall_s={run_wall_s:.3f} "
        f"exact_live={summary['exact_live_baseline']}/{len(records)} "
        f"exact_saved={summary['exact_saved_reference']}/{len(records)} "
        f"call_reduction={summary['target_call_reduction']:.3f}x "
        f"composed_speedup={summary['composed_speedup_vs_saved_baseline']:.3f}x "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
