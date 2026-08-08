#!/usr/bin/env python3
"""Test global teacher-forced correction of row-draft table sequences.

This is a bounded research lab, not a production decode path.  One static
Q-token target graph repeatedly rewrites a complete candidate sequence at the
same logical cache position.  The resulting sequence is approximate unless it
reaches the ordinary greedy fixed point, so every saved sweep is scored against
the ordinary B1 output and OmniDocBench ground truth.
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

from paddleocr_vl.model.text_spec_verify import (
    TextSpecVerifyRuntime,
    torchair_cache_dir_for_spec_shape,
)
from paddleocr_vl.serving.table_speculative import TableDraftMatcher
from pipeline.layout_output import normalize_recognition_text
from table_spec_decode_lab import (
    DEFAULT_DRAFTS,
    DEFAULT_TARGETS,
    build_recognizer,
    exact_target_crop,
    first_difference,
    read_jsonl,
    request_for,
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
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--query-length", type=int, default=3072)
    parser.add_argument("--maximum-sweeps", type=int, default=4)
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
        default=REPO_ROOT
        / ".runtime_cache/09_persistent_page_engine_vision_torchair",
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
    parser.add_argument(
        "--defer-teds",
        action="store_true",
        help="Save HTML for scoring in the separate OmniDocBench environment.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def strip_lane_eos(tokens: list[int], eos_token_id: int) -> list[int]:
    result = [int(token) for token in tokens]
    if result and result[-1] == int(eos_token_id):
        result.pop()
    return result


def waterfill_lengths(capacities: list[int], budget: int) -> list[int]:
    """Share a fixed token budget fairly while retaining every short lane."""
    allocations = [0] * len(capacities)
    active = {index for index, size in enumerate(capacities) if size > 0}
    remaining = int(budget)
    while active and remaining > 0:
        share = max(1, remaining // len(active))
        progressed = False
        for index in tuple(sorted(active)):
            room = capacities[index] - allocations[index]
            take = min(room, share, remaining)
            if take > 0:
                allocations[index] += take
                remaining -= take
                progressed = True
            if allocations[index] >= capacities[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not progressed:
            break
    return allocations


def row_lanes(
    record: dict[str, Any],
    *,
    eos_token_id: int,
    newline_token_id: int,
) -> list[list[int]]:
    lanes: list[list[int]] = []
    for row in sorted(record.get("rows") or [], key=lambda item: item["row_index"]):
        tokens = strip_lane_eos(row.get("token_ids") or [], eos_token_id)
        if tokens and tokens[-1] != int(newline_token_id):
            tokens.append(int(newline_token_id))
        lanes.append(tokens)
    return lanes


def pad_seed(
    tokens: list[int],
    *,
    draft_length: int,
    eos_token_id: int,
) -> tuple[int, ...]:
    aligned = [int(token) for token in tokens]
    aligned = aligned[: int(draft_length)]
    aligned.extend([int(eos_token_id)] * (int(draft_length) - len(aligned)))
    return tuple(aligned)


def align_raw_seed_after_first(
    tokens: list[int],
    *,
    first_token: int,
    draft_length: int,
    eos_token_id: int,
) -> tuple[int, ...]:
    aligned = [int(token) for token in tokens]
    if aligned and aligned[0] == int(first_token):
        aligned.pop(0)
    return pad_seed(
        aligned,
        draft_length=draft_length,
        eos_token_id=eos_token_id,
    )


def flat_seed(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    first_token: int,
    eos_token_id: int,
    draft_length: int,
) -> tuple[int, ...]:
    matcher = TableDraftMatcher(
        record,
        tokenizer,
        eos_token_id=eos_token_id,
        block_size=draft_length,
    )
    matcher.start(first_token)
    proposal = matcher.propose([first_token])
    tokens = [] if proposal is None else list(proposal.tokens)
    # matcher.propose() already returns the continuation after first_token.
    return pad_seed(
        tokens,
        draft_length=draft_length,
        eos_token_id=eos_token_id,
    )


def balanced_seed(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    first_token: int,
    eos_token_id: int,
    draft_length: int,
) -> tuple[int, ...]:
    newline = tokenizer.convert_tokens_to_ids("<nl>")
    if newline is None:
        raise ValueError("tokenizer does not contain <nl>")
    lanes = row_lanes(
        record,
        eos_token_id=eos_token_id,
        newline_token_id=int(newline),
    )
    # Reserve one extra source token because alignment can remove the leading
    # target token after the water-fill operation.
    allocations = waterfill_lengths(
        [len(lane) for lane in lanes],
        int(draft_length) + 1,
    )
    tokens = [
        token
        for lane, allocation in zip(lanes, allocations)
        for token in lane[:allocation]
    ]
    return align_raw_seed_after_first(
        tokens,
        first_token=first_token,
        draft_length=draft_length,
        eos_token_id=eos_token_id,
    )


def truncate_at_eos(tokens: list[int], eos_token_id: int) -> list[int]:
    try:
        end = tokens.index(int(eos_token_id)) + 1
    except ValueError:
        return tokens
    return tokens[:end]


def eos_pad(tokens: list[int], width: int, eos_token_id: int) -> tuple[int, ...]:
    result = [int(token) for token in tokens[: int(width)]]
    try:
        eos_index = result.index(int(eos_token_id))
    except ValueError:
        eos_index = len(result)
    if eos_index < len(result):
        result[eos_index + 1 :] = [int(eos_token_id)] * (
            len(result) - eos_index - 1
        )
    result.extend([int(eos_token_id)] * (int(width) - len(result)))
    return tuple(result)


def changed_fraction(left: list[int], right: list[int]) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    changed = sum(
        index >= len(left)
        or index >= len(right)
        or left[index] != right[index]
        for index in range(width)
    )
    return changed / width


def html_document(value: str) -> str:
    text = str(value)
    if "<html" in text.lower():
        return text
    return f"<html><body>{text}</body></html>"


class TedsScorer:
    def __init__(self, evaluator_root: Path):
        sys.path.insert(0, str(evaluator_root.resolve()))
        from src.metrics.table_metric import TEDS

        self.content = TEDS(structure_only=False)
        self.structure = TEDS(structure_only=True)

    def score(self, prediction: str, target: str) -> dict[str, float]:
        prediction_doc = html_document(prediction)
        target_doc = html_document(target)
        started = time.perf_counter()
        content = float(self.content.evaluate(prediction_doc, target_doc))
        structure = float(self.structure.evaluate(prediction_doc, target_doc))
        return {
            "teds": content,
            "structure_teds": structure,
            "scoring_wall_s": time.perf_counter() - started,
        }


class FullSequenceVerifier:
    def __init__(self, recognizer: Any, args: argparse.Namespace):
        self.query_length = int(args.query_length)
        self.draft_length = self.query_length - 1
        self.runtime = TextSpecVerifyRuntime(
            recognizer.model,
            device=recognizer.device,
            cache_root=args.decode_cache_dir.resolve(),
            draft_length=self.draft_length,
            cache_length=args.cache_length,
            dtype=recognizer.dtype,
            model_dir=recognizer.model_dir,
            linear_weight_format=str(recognizer.weight_format["effective_mode"]),
            optimization="combined_apply",
        )
        self.host_input = torch.empty(
            (1, self.query_length), dtype=torch.int64, pin_memory=True
        )
        self.device_input = torch.empty(
            (1, self.query_length), device=recognizer.device, dtype=torch.int64
        )
        self.host_targets = torch.empty(
            (1, self.query_length), dtype=torch.int64, pin_memory=True
        )

    def call(
        self,
        input_tokens: tuple[int, ...],
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        flat_cache: tuple[torch.Tensor, ...],
    ) -> tuple[list[int], float, float]:
        if len(input_tokens) != self.query_length:
            raise ValueError(
                f"expected {self.query_length} input tokens, got {len(input_tokens)}"
            )
        self.host_input[0].copy_(torch.tensor(input_tokens, dtype=torch.int64))
        started = time.perf_counter()
        self.device_input.copy_(self.host_input, non_blocking=True)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        targets = self.runtime.fn(
            self.device_input,
            cache_position,
            rope_deltas,
            *flat_cache,
        )
        end.record()
        self.host_targets.copy_(targets, non_blocking=True)
        done = torch.npu.Event()
        done.record()
        done.synchronize()
        return (
            [int(token) for token in self.host_targets[0].tolist()],
            float(start.elapsed_time(end)) / 1000.0,
            time.perf_counter() - started,
        )


def candidate_record(
    recognizer: Any,
    scorer: TedsScorer | None,
    *,
    tokens: list[int],
    previous_tokens: list[int],
    gt_html: str,
    baseline_html: str,
    baseline_teds: float | None,
    device_s: float,
    call_wall_s: float,
    cumulative_device_s: float,
    cumulative_wall_s: float,
    fixed_s: float,
) -> dict[str, Any]:
    text = recognizer.tokenizer.decode(tokens, skip_special_tokens=True)
    pred_html = normalize_recognition_text("table", text)
    result = {
        "token_ids": tokens,
        "generated_tokens": len(tokens),
        "text": text,
        "pred_html": pred_html,
        "valid_table_markup": "<table" in pred_html and "</table>" in pred_html,
        "changed_token_fraction": changed_fraction(tokens, previous_tokens),
        "device_s": device_s,
        "call_wall_s": call_wall_s,
        "cumulative_device_s": cumulative_device_s,
        "cumulative_wall_s": cumulative_wall_s,
        "composed_pipeline_wall_s": fixed_s + cumulative_wall_s,
    }
    if scorer is None:
        result["teds_pending"] = True
        return result
    assert baseline_teds is not None
    versus_gt = scorer.score(pred_html, gt_html)
    versus_baseline = scorer.score(pred_html, baseline_html)
    result.update(
        {
            "versus_gt": versus_gt,
            "teds_delta_from_baseline": versus_gt["teds"] - baseline_teds,
            "versus_live_baseline": versus_baseline,
        }
    )
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    if not args.request_id:
        raise ValueError("provide at least one --request-id")
    if args.query_length <= 1 or args.query_length > args.cache_length:
        raise ValueError("query length must be in [2, cache length]")
    if args.maximum_sweeps <= 0:
        raise ValueError("maximum sweeps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)

    targets = {record["request_id"]: record for record in read_jsonl(args.targets)}
    drafts = {record["request_id"]: record for record in read_jsonl(args.drafts)}
    for request_id in args.request_id:
        if request_id not in targets or request_id not in drafts:
            raise KeyError(f"missing target or draft record for {request_id}")

    recognizer = build_recognizer(args)
    draft_length = int(args.query_length) - 1
    cache_path = torchair_cache_dir_for_spec_shape(
        args.decode_cache_dir,
        draft_length=draft_length,
        cache_length=args.cache_length,
        dtype=recognizer.dtype,
        device=recognizer.device,
        model_dir=recognizer.model_dir,
        linear_weight_format=str(recognizer.weight_format["effective_mode"]),
        optimization="combined_apply",
    )
    cache_was_warm = cache_path.is_dir() and any(cache_path.iterdir())
    if not cache_was_warm and not args.allow_compile:
        raise RuntimeError(f"missing Q{args.query_length} verifier cache: {cache_path}")
    print(
        f"FULL_JACOBI_SETUP graph={'hit' if cache_was_warm else 'compile'} "
        f"Q={args.query_length} KV={args.cache_length}",
        flush=True,
    )
    verifier = FullSequenceVerifier(recognizer, args)
    print(
        "FULL_JACOBI_GRAPH_READY "
        f"wrapper_s={verifier.runtime.metadata['compile_wrapper_s']:.3f} "
        f"first_call_s={verifier.runtime.metadata['compile_first_call_s']:.3f}",
        flush=True,
    )
    scorer = None if args.defer_teds else TedsScorer(args.evaluator_root)

    output_path = args.output_dir / "tables.jsonl"
    output_path.write_text("")
    records: list[dict[str, Any]] = []
    for table_index, request_id in enumerate(args.request_id, start=1):
        target = targets[request_id]
        draft = drafts[request_id]
        crop = exact_target_crop(target, args.images_dir)
        request = request_for(target, crop, args)
        baseline_results: list[Any] = []
        baseline_started = time.perf_counter()
        recognizer.run(
            [request],
            schedule_id=f"full-jacobi-baseline:{request_id}",
            emit_result=baseline_results.append,
        )
        baseline_wall_s = time.perf_counter() - baseline_started
        if len(baseline_results) != 1:
            raise RuntimeError(f"expected one baseline result for {request_id}")
        baseline = baseline_results[0]
        baseline_tokens = [int(token) for token in baseline.token_ids]
        if baseline.stop_reason != "eos":
            raise ValueError(
                f"{request_id}: live B1 stopped by {baseline.stop_reason}, not EOS"
            )
        if len(baseline_tokens) > args.query_length + 1:
            raise ValueError(
                f"{request_id}: {len(baseline_tokens)} B1 tokens exceed the "
                f"Q{args.query_length} represented output window"
            )
        baseline_html = normalize_recognition_text("table", baseline.text)
        baseline_scores = (
            None
            if scorer is None
            else scorer.score(baseline_html, str(target["gt_html"]))
        )
        baseline_teds = (
            None if baseline_scores is None else float(baseline_scores["teds"])
        )

        prefill_started = time.perf_counter()
        prefilled = recognizer.prefill_one(request)
        prefill_wall_s = time.perf_counter() - prefill_started
        cache, rope_deltas, cache_position, _first_tensor, release = (
            prefilled.take_device_state()
        )
        first_token = int(prefilled.first_token)
        if baseline_tokens[0] != first_token:
            raise RuntimeError(f"prefill first token mismatch for {request_id}")
        position = int(cache_position.detach().cpu().item())
        if position + args.query_length > args.cache_length:
            raise ValueError(
                f"{request_id}: P={position} + Q={args.query_length} exceeds "
                f"KV={args.cache_length}"
            )
        flat_cache = cache.flat_tensors()
        draft_wall_s = float(
            (draft.get("timing_s") or {}).get("table_row_ocr_e2e", 0.0)
        )
        fixed_s = draft_wall_s + prefill_wall_s

        table_record: dict[str, Any] = {
            "request_id": request_id,
            "gt_html": str(target["gt_html"]),
            "prompt_length": position,
            "draft_generation_wall_s": draft_wall_s,
            "prefill_wall_s": prefill_wall_s,
            "baseline_wall_s": baseline_wall_s,
            "baseline": {
                "token_ids": baseline_tokens,
                "pred_html": baseline_html,
                "stop_reason": baseline.stop_reason,
                "scores": baseline_scores,
            },
            "seeds": {},
        }
        try:
            # baseline_tokens[1:] is already the continuation after first_token.
            control_seed = pad_seed(
                baseline_tokens[1:],
                draft_length=draft_length,
                eos_token_id=int(recognizer.model.config.eos_token_id),
            )
            control_targets, control_device_s, control_wall_s = verifier.call(
                (first_token, *control_seed),
                cache_position,
                rope_deltas,
                flat_cache,
            )
            control_tokens = truncate_at_eos(
                [first_token, *control_targets],
                int(recognizer.model.config.eos_token_id),
            )
            control_record = candidate_record(
                recognizer,
                scorer,
                tokens=control_tokens,
                previous_tokens=baseline_tokens,
                gt_html=str(target["gt_html"]),
                baseline_html=baseline_html,
                baseline_teds=baseline_teds,
                device_s=control_device_s,
                call_wall_s=control_wall_s,
                cumulative_device_s=control_device_s,
                cumulative_wall_s=control_wall_s,
                fixed_s=prefill_wall_s,
            )
            control_record["exact_live_baseline"] = (
                control_tokens == baseline_tokens
            )
            control_record["first_live_difference"] = first_difference(
                control_tokens,
                baseline_tokens,
            )
            control_record["quality_gate_pass"] = (
                None
                if scorer is None
                else control_record["teds_delta_from_baseline"] >= -0.005
            )
            table_record["self_projection_control"] = control_record
            if control_record["quality_gate_pass"] is False:
                write_json(
                    args.output_dir / f"invalid_control_{request_id}.json",
                    table_record,
                )
                print(
                    f"FULL_JACOBI_CONTROL_FAILED id={request_id} "
                    f"exact={control_record['exact_live_baseline']} "
                    f"teds_delta={control_record['teds_delta_from_baseline']:+.6f}",
                    flush=True,
                )
                raise RuntimeError(
                    f"{request_id}: self-projection exceeded the -0.005 TEDS gate"
                )

            seeds = {
                "flat_prefix": flat_seed(
                    draft,
                    recognizer.tokenizer,
                    first_token=first_token,
                    eos_token_id=int(recognizer.model.config.eos_token_id),
                    draft_length=draft_length,
                ),
                "balanced_lane": balanced_seed(
                    draft,
                    recognizer.tokenizer,
                    first_token=first_token,
                    eos_token_id=int(recognizer.model.config.eos_token_id),
                    draft_length=draft_length,
                ),
            }
            for seed_name, seed in seeds.items():
                input_tokens = (first_token, *seed)
                previous = truncate_at_eos(
                    list(input_tokens),
                    int(recognizer.model.config.eos_token_id),
                )
                cumulative_device_s = 0.0
                cumulative_wall_s = 0.0
                sweep_records: list[dict[str, Any]] = []
                for sweep in range(1, args.maximum_sweeps + 1):
                    targets_out, device_s, call_wall_s = verifier.call(
                        input_tokens,
                        cache_position,
                        rope_deltas,
                        flat_cache,
                    )
                    cumulative_device_s += device_s
                    cumulative_wall_s += call_wall_s
                    candidate_tokens = truncate_at_eos(
                        [first_token, *targets_out],
                        int(recognizer.model.config.eos_token_id),
                    )
                    record = candidate_record(
                        recognizer,
                        scorer,
                        tokens=candidate_tokens,
                        previous_tokens=previous,
                        gt_html=str(target["gt_html"]),
                        baseline_html=baseline_html,
                        baseline_teds=baseline_teds,
                        device_s=device_s,
                        call_wall_s=call_wall_s,
                        cumulative_device_s=cumulative_device_s,
                        cumulative_wall_s=cumulative_wall_s,
                        fixed_s=fixed_s,
                    )
                    record["sweep"] = sweep
                    sweep_records.append(record)
                    previous = candidate_tokens
                    next_guess = eos_pad(
                        targets_out[:-1],
                        draft_length,
                        int(recognizer.model.config.eos_token_id),
                    )
                    input_tokens = (first_token, *next_guess)
                table_record["seeds"][seed_name] = sweep_records
        finally:
            if release is not None:
                release()

        records.append(table_record)
        with output_path.open("a") as stream:
            stream.write(json.dumps(table_record, ensure_ascii=False) + "\n")
        flat_k4 = table_record["seeds"]["flat_prefix"][-1]
        balanced_k4 = table_record["seeds"]["balanced_lane"][-1]
        if scorer is None:
            print(
                f"FULL_JACOBI_RESULT table={table_index}/{len(args.request_id)} "
                f"id={request_id} scoring=deferred "
                f"control_exact={table_record['self_projection_control']['exact_live_baseline']} "
                f"flat_K{args.maximum_sweeps}={flat_k4['composed_pipeline_wall_s']:.3f}s "
                f"balanced_K{args.maximum_sweeps}={balanced_k4['composed_pipeline_wall_s']:.3f}s",
                flush=True,
            )
        else:
            assert baseline_scores is not None
            print(
                f"FULL_JACOBI_RESULT table={table_index}/{len(args.request_id)} "
                f"id={request_id} baseline_teds={baseline_scores['teds']:.6f} "
                f"control_delta={table_record['self_projection_control']['teds_delta_from_baseline']:+.6f} "
                f"flat_K{args.maximum_sweeps}={flat_k4['versus_gt']['teds']:.6f}/"
                f"{flat_k4['composed_pipeline_wall_s']:.3f}s "
                f"balanced_K{args.maximum_sweeps}={balanced_k4['versus_gt']['teds']:.6f}/"
                f"{balanced_k4['composed_pipeline_wall_s']:.3f}s",
                flush=True,
            )

    summary = {
        "configuration": {
            "query_length": args.query_length,
            "draft_length": draft_length,
            "maximum_sweeps": args.maximum_sweeps,
            "cache_length": args.cache_length,
            "graph_cache_was_warm": cache_was_warm,
            "graph_metadata": verifier.runtime.metadata,
            "teds_deferred": bool(args.defer_teds),
        },
        "tables": len(records),
        "control_within_teds_gate": (
            None
            if scorer is None
            else sum(
                record["self_projection_control"]["teds_delta_from_baseline"]
                >= -0.005
                for record in records
            )
        ),
        "records": str(output_path),
    }
    write_json(args.output_dir / "run_summary.json", summary)
    print(f"FULL_JACOBI_COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
