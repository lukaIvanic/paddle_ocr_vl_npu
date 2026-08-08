#!/usr/bin/env python3
"""Test fixed-block Jacobi table decode with the compiled target verifier.

The row OCR output initializes each block.  The target verifier repeatedly
updates every token in that block at one fixed cache position.  A block is
committed only when it is a causal fixed point; otherwise the lab commits the
first target token after the bounded sweep count.  Every committed token is
therefore target-generated or target-verified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.model.text_spec_verify import torchair_cache_dir_for_spec_shape
from paddleocr_vl.serving.repetition import ExactCycleTracker
from paddleocr_vl.serving.table_speculative import (
    TableDraftMatcher,
    TableSpeculativeDecodeRuntime,
)
from pipeline.layout_output import normalize_recognition_text
from table_spec_decode_lab import (
    DEFAULT_DRAFTS,
    DEFAULT_TARGETS,
    build_recognizer,
    exact_target_crop,
    first_difference,
    read_jsonl,
    request_for,
    target_tokens,
)


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
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument(
        "--maximum-sweeps",
        type=int,
        help=(
            "Default: draft length + 1.  The extra call confirms the fixed "
            "point and computes the bonus token from the exact block."
        ),
    )
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
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
        "--vision-buckets",
        default="256,384,512,640,768,1408,1920,2048,2944,4096",
    )
    parser.add_argument("--text-buckets", default="128,256,512,1024,1312")
    parser.add_argument("--allow-compile", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@torch.inference_mode()
def decode_jacobi_blocks(
    runtime: TableSpeculativeDecodeRuntime,
    prefilled: Any,
    matcher: TableDraftMatcher,
    *,
    maximum_sweeps: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    cache, rope_deltas, cache_position, _first, release = prefilled.take_device_state()
    eos = int(runtime.eos_token_id)
    draft_length = int(runtime.draft_length)
    token_ids = [int(prefilled.first_token)]
    matcher.start(token_ids[0])
    tracker = ExactCycleTracker()
    tracker.update(token_ids[0])
    position = int(cache_position.detach().cpu().item())
    flat_cache = cache.flat_tensors()
    target_calls = 0
    verifier_device_s = 0.0
    stable_blocks = 0
    unstable_blocks = 0
    stable_tokens = 0
    sweeps_per_block: list[int] = []
    stop_reason: str | None = "eos" if token_ids[0] == eos else None
    repetition: dict[str, Any] = {}
    started = time.perf_counter()

    def append(values: list[int]) -> bool:
        nonlocal stop_reason, repetition
        for value in values:
            token = int(value)
            token_ids.append(token)
            if token == eos:
                stop_reason = "eos"
                return True
            evidence = tracker.update(token)
            if evidence is not None:
                repetition = evidence.to_dict()
                del token_ids[evidence.trim_length :]
                stop_reason = "repetition"
                return True
            if prefilled.input_tokens + len(token_ids) - 1 >= runtime.cache_length:
                stop_reason = "kv_cache_full"
                return True
            if len(token_ids) >= max_new_tokens:
                stop_reason = "length"
                return True
        return False

    try:
        while stop_reason is None:
            cache_position.fill_(position)
            if position + runtime.query_length > runtime.cache_length:
                next_token, device_s = runtime._decode_call(
                    token_ids[-1], cache_position, rope_deltas, flat_cache
                )
                target_calls += 1
                verifier_device_s += device_s
                matcher.commit(None, accepted_draft_tokens=0, emitted_tokens=(next_token,))
                append([next_token])
                position += 1
                continue

            proposal = matcher.propose(token_ids)
            seed = [] if proposal is None else list(proposal.tokens)
            guess = tuple((seed + [eos] * draft_length)[:draft_length])
            stable_targets: list[int] | None = None
            sweeps = 0
            for sweeps in range(1, maximum_sweeps + 1):
                targets, device_s = runtime._verify_call(
                    token_ids[-1], guess, cache_position, rope_deltas, flat_cache
                )
                target_calls += 1
                verifier_device_s += device_s
                updated = tuple(int(token) for token in targets[:draft_length])
                if updated == guess:
                    stable_targets = [*updated, int(targets[draft_length])]
                    break
                guess = updated
            sweeps_per_block.append(sweeps)

            if stable_targets is None:
                # The first prediction is always conditioned only on the exact
                # committed prefix and is therefore safe to commit.
                unstable_blocks += 1
                emitted = [int(guess[0])]
            else:
                stable_blocks += 1
                emitted = stable_targets
                stable_tokens += len(emitted)

            matcher.commit(None, accepted_draft_tokens=0, emitted_tokens=emitted)
            append(emitted)
            position += len(emitted)
    finally:
        if release is not None:
            release()

    text = runtime.recognizer.tokenizer.decode(
        token_ids,
        skip_special_tokens=prefilled.skip_special_tokens,
    )
    return {
        "token_ids": token_ids,
        "text": text,
        "pred_html": normalize_recognition_text("table", text),
        "stop_reason": str(stop_reason),
        "target_calls": target_calls,
        "stable_blocks": stable_blocks,
        "unstable_blocks": unstable_blocks,
        "stable_tokens": stable_tokens,
        "mean_sweeps_per_block": (
            sum(sweeps_per_block) / len(sweeps_per_block)
            if sweeps_per_block
            else None
        ),
        "sweeps_per_block": sweeps_per_block,
        "maximum_sweeps_observed": max(sweeps_per_block, default=0),
        "verifier_device_s": verifier_device_s,
        "wall_s": time.perf_counter() - started,
        "repetition": repetition,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    if not args.request_id:
        raise ValueError("provide at least one --request-id")
    args.maximum_sweeps = int(args.maximum_sweeps or (args.draft_length + 1))
    if args.maximum_sweeps <= 0:
        raise ValueError("maximum sweeps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)

    targets = {record["request_id"]: record for record in read_jsonl(args.targets)}
    drafts = {record["request_id"]: record for record in read_jsonl(args.drafts)}
    selected = [targets[request_id] for request_id in args.request_id]
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
    )
    cache_hit = spec_cache.is_dir() and any(spec_cache.iterdir())
    if not cache_hit and not args.allow_compile:
        raise RuntimeError(f"missing verifier cache: {spec_cache}")
    runtime = TableSpeculativeDecodeRuntime(
        recognizer,
        draft_length=args.draft_length,
        cache_root=args.decode_cache_dir.resolve(),
    )
    print(
        f"TABLE_JACOBI_SETUP cache={'hit' if cache_hit else 'compile'} "
        f"D={args.draft_length} sweeps={args.maximum_sweeps}",
        flush=True,
    )

    output = args.output_dir / "tables.jsonl"
    output.write_text("")
    records: list[dict[str, Any]] = []
    for index, target in enumerate(selected, start=1):
        request_id = str(target["request_id"])
        if request_id not in drafts:
            raise KeyError(f"missing draft record for {request_id}")
        crop = exact_target_crop(target, args.images_dir)
        baseline_results: list[Any] = []
        baseline_started = time.perf_counter()
        recognizer.run(
            [request_for(target, crop, args)],
            schedule_id=f"table-jacobi-baseline:{request_id}",
            emit_result=baseline_results.append,
        )
        baseline_wall_s = time.perf_counter() - baseline_started
        if len(baseline_results) != 1:
            raise RuntimeError(
                f"expected one baseline result for {request_id}, got {len(baseline_results)}"
            )
        baseline = baseline_results[0]
        prefill_started = time.perf_counter()
        prefilled = recognizer.prefill_one(request_for(target, crop, args))
        prefill_wall_s = time.perf_counter() - prefill_started
        matcher = TableDraftMatcher(
            drafts[request_id],
            recognizer.tokenizer,
            eos_token_id=int(recognizer.model.config.eos_token_id),
            block_size=args.draft_length,
        )
        result = decode_jacobi_blocks(
            runtime,
            prefilled,
            matcher,
            maximum_sweeps=args.maximum_sweeps,
            max_new_tokens=args.max_new_tokens,
        )
        draft_s = float((drafts[request_id].get("timing_s") or {}).get("table_row_ocr_e2e", 0.0))
        record = {
            "request_id": request_id,
            "draft_generation_wall_s": draft_s,
            "prefill_wall_s": prefill_wall_s,
            "composed_pipeline_wall_s": draft_s + prefill_wall_s + result["wall_s"],
            "baseline_wall_s": baseline_wall_s,
            "baseline_decode_iterations": max(0, len(baseline.token_ids) - 1),
            "jacobi_generated_tokens": len(result["token_ids"]),
            "target_call_reduction": (
                max(0, len(baseline.token_ids) - 1) / result["target_calls"]
                if result["target_calls"]
                else None
            ),
            "exact_live_baseline": result["token_ids"] == list(baseline.token_ids),
            "first_live_difference": first_difference(result["token_ids"], list(baseline.token_ids)),
            "exact_saved_reference": result["token_ids"] == target_tokens(target),
            "baseline": {
                "token_ids": list(baseline.token_ids),
                "stop_reason": baseline.stop_reason,
            },
            "jacobi": result,
        }
        records.append(record)
        with output.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        mean_sweeps = result["mean_sweeps_per_block"]
        mean_sweeps_text = "n/a" if mean_sweeps is None else f"{mean_sweeps:.2f}"
        print(
            f"TABLE_JACOBI_RESULT table={index}/{len(selected)} id={request_id} "
            f"calls={result['target_calls']} blocks={result['stable_blocks']} "
            f"unstable={result['unstable_blocks']} sweeps={mean_sweeps_text} "
            f"decode_s={result['wall_s']:.3f} composed_s={record['composed_pipeline_wall_s']:.3f} "
            f"exact={record['exact_live_baseline']}",
            flush=True,
        )

    summary = {
        "configuration": {
            "draft_length": args.draft_length,
            "maximum_sweeps": args.maximum_sweeps,
            "cache_length": args.cache_length,
            "verifier_cache_was_warm": cache_hit,
            "model": str(args.model),
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "vision_buckets": args.vision_buckets,
            "text_buckets": args.text_buckets,
        },
        "tables": len(records),
        "exact_live_baseline": sum(record["exact_live_baseline"] for record in records),
        "target_calls": sum(record["jacobi"]["target_calls"] for record in records),
        "baseline_decode_iterations": sum(
            record["baseline_decode_iterations"] for record in records
        ),
        "stable_blocks": sum(record["jacobi"]["stable_blocks"] for record in records),
        "unstable_blocks": sum(record["jacobi"]["unstable_blocks"] for record in records),
        "stable_tokens": sum(record["jacobi"]["stable_tokens"] for record in records),
        "baseline_wall_s": sum(record["baseline_wall_s"] for record in records),
        "composed_pipeline_wall_s": sum(record["composed_pipeline_wall_s"] for record in records),
        "records": str(output),
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(f"TABLE_JACOBI_COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
