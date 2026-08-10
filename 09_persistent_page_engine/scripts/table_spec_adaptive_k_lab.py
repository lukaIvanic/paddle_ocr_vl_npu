#!/usr/bin/env python3
"""Run table speculative decoding with a success-driven adaptive verifier K."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch


SCRIPT_LOCATION = Path(__file__).resolve().parent
if (SCRIPT_LOCATION.parent / "paddleocr_vl").is_dir():
    HERE = SCRIPT_LOCATION
    EXPERIMENT_ROOT = HERE.parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
else:
    # The Blue-zone checkout is pull-only.  Lab revisions can therefore run
    # from /tmp while importing the tracked checkout selected as the cwd.
    REPO_ROOT = Path.cwd().resolve()
    EXPERIMENT_ROOT = REPO_ROOT / "09_persistent_page_engine"
    HERE = EXPERIMENT_ROOT / "scripts"
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

import table_spec_decode_lab as fixed_lab  # noqa: E402
from paddleocr_vl.model.text_spec_verify import (  # noqa: E402
    torchair_cache_dir_for_spec_shape,
)
from paddleocr_vl.model.token_selection import TOKEN_SELECTION_CHOICES  # noqa: E402
from paddleocr_vl.serving.repetition import ExactCycleTracker  # noqa: E402
from paddleocr_vl.serving.table_speculative import (  # noqa: E402
    TableDraftMatcher,
    TableSpecDecodeResult,
    TableSpeculativeDecodeRuntime,
)
from pipeline.layout_output import normalize_recognition_text  # noqa: E402


DEFAULT_K_VALUES = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
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
    parser.add_argument("--k-values", default="8,16,32,64")
    parser.add_argument("--initial-k", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--token-selection",
        default="greedy",
        choices=TOKEN_SELECTION_CHOICES,
    )
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--vision-buckets", default=fixed_lab.DEFAULT_VISION_BUCKETS)
    parser.add_argument("--text-buckets", default=fixed_lab.DEFAULT_TEXT_BUCKETS)
    parser.add_argument("--allow-compile", action="store_true")
    parser.add_argument(
        "--cell-boundary-math-open-draft-trust",
        action="store_true",
        help=(
            "At a verified cell boundary with a sufficiently long exact match, "
            "follow one draft token when either greedy next token is exact \\("
        ),
    )
    parser.add_argument(
        "--cell-boundary-slash-draft-trust",
        action="store_true",
        help=(
            "Apply the same one-token cell-boundary draft trust rule when "
            "either greedy next token is the exact standalone backslash token."
        ),
    )
    parser.add_argument(
        "--cell-boundary-min-match",
        type=int,
        default=5,
        help="Require current exact draft/target match length to be greater than this.",
    )
    parser.add_argument(
        "--in-cell-draft-script-open-trust",
        action="store_true",
        help=(
            "Inside a cell with a sufficiently long exact match, follow one "
            "draft token when the draft next token is the exact ^{ token."
        ),
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
        "--k-cache-root",
        action="append",
        default=[],
        metavar="K=PATH",
        help="Optional per-K verifier cache root; unspecified K uses decode-cache-dir.",
    )
    return parser.parse_args()


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("k-values must contain positive integers")
    return values


def parse_k_cache_roots(
    values: list[str],
    *,
    k_values: tuple[int, ...],
    default: Path,
) -> dict[int, Path]:
    result = {value: default for value in k_values}
    for raw in values:
        key_raw, separator, path_raw = raw.partition("=")
        if not separator:
            raise ValueError(f"invalid --k-cache-root value: {raw!r}")
        key = int(key_raw)
        if key not in result:
            raise ValueError(f"cache root supplied for unconfigured K{key}")
        result[key] = Path(path_raw)
    return result


def next_k(
    current: int,
    *,
    fully_accepted: bool,
    k_values: tuple[int, ...],
) -> int:
    index = k_values.index(int(current))
    if fully_accepted:
        return k_values[min(len(k_values) - 1, index + 1)]
    return k_values[max(0, index - 1)]


def cell_boundary_math_open_draft_token(
    token_ids: list[int],
    proposal: Any,
    *,
    accepted_before_rejection: int,
    base_next_token: int,
    cell_token_ids: set[int],
    math_open_token_id: int,
    additional_trigger_token_ids: set[int] | None = None,
    minimum_match: int,
) -> int | None:
    """Select one draft token for a narrow cell-opening style disagreement."""

    accepted = int(accepted_before_rejection)
    if accepted < 0 or accepted >= len(proposal.tokens):
        return None
    previous_token = (
        int(proposal.tokens[accepted - 1]) if accepted else int(token_ids[-1])
    )
    if previous_token not in cell_token_ids:
        return None
    exact_match = int(proposal.anchor_tokens) + accepted
    if exact_match <= int(minimum_match):
        return None
    draft_next_token = int(proposal.tokens[accepted])
    trigger_token_ids = {int(math_open_token_id)}
    trigger_token_ids.update(int(value) for value in (additional_trigger_token_ids or ()))
    if not trigger_token_ids.intersection((int(base_next_token), draft_next_token)):
        return None
    return draft_next_token


def in_cell_draft_script_open_token(
    token_ids: list[int],
    proposal: Any,
    *,
    accepted_before_rejection: int,
    cell_token_ids: set[int],
    newline_token_id: int,
    script_open_token_id: int,
    minimum_match: int,
) -> int | None:
    """Follow one exact draft ``^{`` token after a well-aligned cell prefix."""

    accepted = int(accepted_before_rejection)
    if accepted < 0 or accepted >= len(proposal.tokens):
        return None
    draft_next_token = int(proposal.tokens[accepted])
    if draft_next_token != int(script_open_token_id):
        return None
    if int(proposal.anchor_tokens) + accepted <= int(minimum_match):
        return None
    prospective_prefix = [
        *[int(value) for value in token_ids],
        *[int(value) for value in proposal.tokens[:accepted]],
    ]
    tokens_inside_cell = 0
    for token in reversed(prospective_prefix):
        if token in cell_token_ids:
            return draft_next_token if tokens_inside_cell > 0 else None
        if token == int(newline_token_id):
            return None
        tokens_inside_cell += 1
    return None


@dataclass
class AdaptiveKTableSpecDecodeResult(TableSpecDecodeResult):
    adaptive_k: dict[str, Any] = field(default_factory=dict)


class AdaptiveKTableSpeculativeDecodeRuntime:
    """Select among independently compiled fixed-K verifier graphs per call."""

    def __init__(
        self,
        recognizer: Any,
        *,
        k_values: tuple[int, ...],
        initial_k: int,
        cache_roots: dict[int, Path],
        cell_boundary_math_open_draft_trust: bool = False,
        cell_boundary_slash_draft_trust: bool = False,
        in_cell_draft_script_open_trust: bool = False,
        cell_boundary_min_match: int = 5,
    ) -> None:
        if initial_k not in k_values:
            raise ValueError("initial-k must be one of k-values")
        self.recognizer = recognizer
        self.k_values = k_values
        self.initial_k = int(initial_k)
        self.cache_length = int(recognizer.cache_length)
        self.eos_token_id = int(recognizer.model.config.eos_token_id)
        self.cell_boundary_math_open_draft_trust = bool(
            cell_boundary_math_open_draft_trust
        )
        self.cell_boundary_slash_draft_trust = bool(
            cell_boundary_slash_draft_trust
        )
        self.in_cell_draft_script_open_trust = bool(
            in_cell_draft_script_open_trust
        )
        tokenizer = recognizer.tokenizer
        script_open_token_id = (
            tokenizer.token_to_id("^{")
            if hasattr(tokenizer, "token_to_id")
            else tokenizer.convert_tokens_to_ids("^{")
        )
        if script_open_token_id is None:
            raise ValueError("tokenizer does not contain exact ^{ token")
        self.script_open_token_id = int(script_open_token_id)
        self.cell_boundary_min_match = int(cell_boundary_min_match)
        if self.cell_boundary_min_match < 0:
            raise ValueError("cell-boundary-min-match must be non-negative")
        self.runtimes = {
            value: TableSpeculativeDecodeRuntime(
                recognizer,
                draft_length=value,
                cache_root=cache_roots[value].resolve(),
                wrapper_rescue=False,
            )
            for value in k_values
        }

    @torch.inference_mode()
    def decode(
        self,
        prefilled: Any,
        matcher: TableDraftMatcher,
        *,
        max_new_tokens: int | None = None,
    ) -> AdaptiveKTableSpecDecodeResult:
        cache, rope_deltas, cache_position, _first_token_tensor, cache_release = (
            prefilled.take_device_state()
        )
        started = time.perf_counter()
        token_ids = [int(prefilled.first_token)]
        matcher.start(token_ids[0])
        tracker = ExactCycleTracker()
        tracker.update(token_ids[0])
        position = int(cache_position.detach().cpu().item())
        policy_k = self.initial_k
        target_calls = 0
        speculative_calls = 0
        fully_accepted_speculative_calls = 0
        rejected_speculative_calls = 0
        fallback_calls = 0
        proposed = 0
        accepted = 0
        verifier_device_s = 0.0
        fallback_device_s = 0.0
        stop_reason: str | None = "eos" if token_ids[0] == self.eos_token_id else None
        repetition: dict[str, Any] = {}
        limit = int(max_new_tokens or self.cache_length)
        flat_cache = cache.flat_tensors()
        trace: list[dict[str, Any]] = []
        transition_counts: Counter[str] = Counter()
        cell_boundary_events: list[dict[str, Any]] = []
        per_k = {
            value: {
                "calls": 0,
                "fully_accepted_calls": 0,
                "rejected_calls": 0,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "verifier_device_s": 0.0,
            }
            for value in self.k_values
        }

        def append_tokens(values: Iterable[int]) -> bool:
            nonlocal stop_reason, repetition
            for value in values:
                token = int(value)
                token_ids.append(token)
                if token == self.eos_token_id:
                    stop_reason = "eos"
                    return True
                evidence = tracker.update(token)
                if evidence is not None:
                    repetition = evidence.to_dict()
                    del token_ids[evidence.trim_length :]
                    stop_reason = "repetition"
                    return True
                if prefilled.input_tokens + len(token_ids) - 1 >= self.cache_length:
                    stop_reason = "kv_cache_full"
                    return True
                if len(token_ids) >= limit:
                    stop_reason = "length"
                    return True
            return False

        try:
            while stop_reason is None:
                usable = [
                    value
                    for value in self.k_values
                    if value <= policy_k and position + value + 1 <= self.cache_length
                ]
                effective_k = max(usable) if usable else None
                if effective_k is not None:
                    matcher.block_size = effective_k
                proposal = matcher.propose(token_ids)
                if effective_k is None or proposal is None or not proposal.tokens:
                    runtime = self.runtimes[min(self.k_values)]
                    cache_position.fill_(position)
                    next_token, device_s = runtime._decode_call(
                        token_ids[-1], cache_position, rope_deltas, flat_cache
                    )
                    target_calls += 1
                    fallback_calls += 1
                    fallback_device_s += device_s
                    matcher.commit(
                        None,
                        accepted_draft_tokens=0,
                        emitted_tokens=(next_token,),
                    )
                    append_tokens((next_token,))
                    position += 1
                    continue

                runtime = self.runtimes[effective_k]
                proposal_tokens = proposal.tokens
                cache_position.fill_(position)
                targets, device_s = runtime._verify_call(
                    token_ids[-1],
                    proposal_tokens,
                    cache_position,
                    rope_deltas,
                    flat_cache,
                )
                target_calls += 1
                speculative_calls += 1
                verifier_device_s += device_s
                proposed += len(proposal_tokens)
                accepted_here = 0
                for draft_token, target_token in zip(proposal_tokens, targets):
                    if draft_token != target_token:
                        break
                    accepted_here += 1
                forced_draft_token = None
                forced_rule = None
                if (
                    (
                        self.cell_boundary_math_open_draft_trust
                        or self.cell_boundary_slash_draft_trust
                    )
                    and accepted_here < len(proposal_tokens)
                ):
                    forced_draft_token = cell_boundary_math_open_draft_token(
                        token_ids,
                        proposal,
                        accepted_before_rejection=accepted_here,
                        base_next_token=int(targets[accepted_here]),
                        cell_token_ids=set(matcher.cell_tokens),
                        math_open_token_id=int(self.recognizer.math_open_token_id),
                        additional_trigger_token_ids=(
                            {int(self.recognizer.math_slash_token_id)}
                            if self.cell_boundary_slash_draft_trust
                            else set()
                        ),
                        minimum_match=self.cell_boundary_min_match,
                    )
                    if forced_draft_token is not None:
                        forced_rule = "cell_boundary_math_open_or_slash"
                if (
                    forced_draft_token is None
                    and self.in_cell_draft_script_open_trust
                    and accepted_here < len(proposal_tokens)
                ):
                    forced_draft_token = in_cell_draft_script_open_token(
                        token_ids,
                        proposal,
                        accepted_before_rejection=accepted_here,
                        cell_token_ids=set(matcher.cell_tokens),
                        newline_token_id=int(matcher.newline_token),
                        script_open_token_id=self.script_open_token_id,
                        minimum_match=self.cell_boundary_min_match,
                    )
                    if forced_draft_token is not None:
                        forced_rule = "in_cell_draft_script_open"
                fully_accepted = accepted_here == len(proposal_tokens)
                accepted += accepted_here
                if fully_accepted:
                    fully_accepted_speculative_calls += 1
                else:
                    rejected_speculative_calls += 1

                stats = per_k[effective_k]
                stats["calls"] += 1
                stats["fully_accepted_calls"] += int(fully_accepted)
                stats["rejected_calls"] += int(not fully_accepted)
                stats["proposed_tokens"] += len(proposal_tokens)
                stats["accepted_tokens"] += accepted_here
                stats["verifier_device_s"] += device_s
                updated_k = next_k(
                    effective_k,
                    fully_accepted=fully_accepted,
                    k_values=self.k_values,
                )
                transition_counts[f"{effective_k}->{updated_k}"] += 1
                trace.append(
                    {
                        "position": position,
                        "k": effective_k,
                        "proposed": len(proposal_tokens),
                        "accepted": accepted_here,
                        "fully_accepted": fully_accepted,
                        "next_k": updated_k,
                    }
                )
                policy_k = updated_k

                if forced_draft_token is not None:
                    cell_boundary_events.append(
                        {
                            "position": position + accepted_here,
                            "proposal_start": int(proposal.start),
                            "accepted_before_override": accepted_here,
                            "matching_prefix_tokens": (
                                int(proposal.anchor_tokens) + accepted_here
                            ),
                            "base_next_token": int(targets[accepted_here]),
                            "draft_next_token": int(forced_draft_token),
                            "rule": str(forced_rule),
                        }
                    )
                    emitted = list(proposal_tokens[: accepted_here + 1])
                    matcher.commit(
                        proposal,
                        accepted_draft_tokens=accepted_here + 1,
                        emitted_tokens=emitted,
                    )
                    append_tokens(emitted)
                    position += accepted_here + 1
                    continue

                emitted = list(proposal_tokens[:accepted_here])
                emitted.append(int(targets[accepted_here]))
                matcher.commit(
                    proposal,
                    accepted_draft_tokens=accepted_here,
                    emitted_tokens=emitted,
                )
                append_tokens(emitted)
                position += accepted_here + 1
        finally:
            if cache_release is not None:
                cache_release()

        text = self.recognizer.tokenizer.decode(
            token_ids,
            skip_special_tokens=prefilled.skip_special_tokens,
        )
        return AdaptiveKTableSpecDecodeResult(
            token_ids=token_ids,
            text=text,
            stop_reason=str(stop_reason),
            target_calls=target_calls,
            speculative_calls=speculative_calls,
            fully_accepted_speculative_calls=fully_accepted_speculative_calls,
            rejected_speculative_calls=rejected_speculative_calls,
            fallback_calls=fallback_calls,
            proposed_draft_tokens=proposed,
            accepted_draft_tokens=accepted,
            verifier_device_s=verifier_device_s,
            fallback_device_s=fallback_device_s,
            wrapper_rescue_probe_device_s=0.0,
            wall_s=time.perf_counter() - started,
            wrapper_rescue={
                "enabled": False,
                "used": False,
                "probe_calls": 0,
                "forced_tokens": 0,
                "events": [],
            },
            repetition=repetition,
            adaptive_k={
                "values": list(self.k_values),
                "initial_k": self.initial_k,
                "per_k": {str(key): value for key, value in per_k.items()},
                "transitions": dict(sorted(transition_counts.items())),
                "trace": trace,
                "cell_boundary_math_open_draft_trust": {
                    "enabled": self.cell_boundary_math_open_draft_trust,
                    "standalone_slash_enabled": self.cell_boundary_slash_draft_trust,
                    "in_cell_script_open_enabled": (
                        self.in_cell_draft_script_open_trust
                    ),
                    "script_open_token_id": self.script_open_token_id,
                    "minimum_match_exclusive": self.cell_boundary_min_match,
                    "forced_tokens": len(cell_boundary_events),
                    "events": cell_boundary_events,
                },
            },
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    k_values = parse_k_values(args.k_values)
    cache_roots = parse_k_cache_roots(
        args.k_cache_root,
        k_values=k_values,
        default=args.decode_cache_dir,
    )
    targets = fixed_lab.read_jsonl(args.targets)
    drafts = {record["request_id"]: record for record in fixed_lab.read_jsonl(args.drafts)}
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
    missing_drafts = [row["request_id"] for row in selected if row["request_id"] not in drafts]
    if missing_drafts:
        raise KeyError(f"missing draft records: {missing_drafts[:5]}")

    setup_started = time.perf_counter()
    recognizer = fixed_lab.build_recognizer(args)
    cache_hits: dict[int, bool] = {}
    for value in k_values:
        spec_cache = torchair_cache_dir_for_spec_shape(
            cache_roots[value],
            draft_length=value,
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
        cache_hits[value] = spec_cache.is_dir() and any(spec_cache.iterdir())
        if not cache_hits[value] and not args.allow_compile:
            raise RuntimeError(f"missing K{value}/KV{args.cache_length} verifier cache")
    print(
        "TABLE_SPEC_ADAPTIVE_K_PROGRESS setup=verifiers "
        + " ".join(
            f"K{value}={'hit' if cache_hits[value] else 'compile'}" for value in k_values
        ),
        flush=True,
    )
    runtime = AdaptiveKTableSpeculativeDecodeRuntime(
        recognizer,
        k_values=k_values,
        initial_k=args.initial_k,
        cache_roots=cache_roots,
        cell_boundary_math_open_draft_trust=(
            args.cell_boundary_math_open_draft_trust
        ),
        cell_boundary_slash_draft_trust=args.cell_boundary_slash_draft_trust,
        in_cell_draft_script_open_trust=args.in_cell_draft_script_open_trust,
        cell_boundary_min_match=args.cell_boundary_min_match,
    )
    setup_s = time.perf_counter() - setup_started
    print(
        f"TABLE_SPEC_ADAPTIVE_K_PROGRESS setup=complete wall_s={setup_s:.3f} "
        f"tables={len(selected)}",
        flush=True,
    )

    output_path = args.output_dir / "tables.jsonl"
    output_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for table_index, target in enumerate(selected, start=1):
        request_id = str(target["request_id"])
        crop = fixed_lab.exact_target_crop(target, args.images_dir)
        reference_tokens = fixed_lab.target_tokens(target)
        prefill_started = time.perf_counter()
        prefilled = recognizer.prefill_one(fixed_lab.request_for(target, crop, args))
        prefill_wall_s = time.perf_counter() - prefill_started
        matcher = TableDraftMatcher(
            drafts[request_id],
            recognizer.tokenizer,
            eos_token_id=int(recognizer.model.config.eos_token_id),
            block_size=args.initial_k,
        )
        result = runtime.decode(
            prefilled,
            matcher,
            max_new_tokens=args.max_new_tokens,
        )
        draft_generation_wall_s = float(
            (drafts[request_id].get("timing_s") or {}).get("table_row_ocr_e2e", 0.0)
            or 0.0
        )
        saved_baseline_wall_s = float(
            (target.get("timing_s") or {}).get("table_row_ocr_e2e", 0.0) or 0.0
        )
        target_spec_wall_s = prefill_wall_s + result.wall_s
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
            "speculative": result.to_dict(),
            "gt_html": target.get("gt_html"),
            "pred_html": normalize_recognition_text("table", result.text),
            "exact_saved_reference": result.token_ids == reference_tokens,
            "first_saved_difference": fixed_lab.first_difference(
                result.token_ids, reference_tokens
            ),
            "exact_live_baseline": None,
            "first_live_difference": None,
        }
        records.append(payload)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(
            f"TABLE_SPEC_ADAPTIVE_K_RESULT table={table_index}/{len(selected)} "
            f"id={request_id} tokens={len(result.token_ids)} calls={result.target_calls} "
            f"accepted={result.accepted_draft_tokens}/{result.proposed_draft_tokens} "
            f"decode_s={result.wall_s:.3f} exact_saved={payload['exact_saved_reference']}",
            flush=True,
        )

    run_wall_s = time.perf_counter() - run_started
    generated_target_tokens = sum(
        max(0, len(row["speculative"]["token_ids"]) - 1) for row in records
    )
    total_calls = sum(row["speculative"]["target_calls"] for row in records)
    accepted_tokens = sum(row["speculative"]["accepted_draft_tokens"] for row in records)
    proposed_tokens = sum(row["speculative"]["proposed_draft_tokens"] for row in records)
    composed = [row["composed_pipeline_wall_s"] for row in records]
    saved = [row["saved_baseline_wall_s"] for row in records]
    per_table_speedup = [row["composed_speedup_vs_saved_baseline"] for row in records]
    aggregate_per_k = {
        str(value): {
            key: sum(
                row["speculative"]["adaptive_k"]["per_k"][str(value)][key]
                for row in records
            )
            for key in (
                "calls",
                "fully_accepted_calls",
                "rejected_calls",
                "proposed_tokens",
                "accepted_tokens",
                "verifier_device_s",
            )
        }
        for value in k_values
    }
    transitions: Counter[str] = Counter()
    for row in records:
        transitions.update(row["speculative"]["adaptive_k"]["transitions"])
    summary = {
        "status": "complete",
        "kind": "adaptive_verifier_k",
        "configuration": {
            "k_values": list(k_values),
            "initial_k": args.initial_k,
            "policy": "fully accepted call doubles K; rejected call halves K",
            "cache_length": args.cache_length,
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "cache_hits": {str(key): value for key, value in cache_hits.items()},
            "recognizer": recognizer.configuration(),
        },
        "setup_s": setup_s,
        "run_wall_s": run_wall_s,
        "tables": len(records),
        "exact_saved_reference": sum(row["exact_saved_reference"] for row in records),
        "target_calls": total_calls,
        "accepted_draft_tokens": accepted_tokens,
        "proposed_draft_tokens": proposed_tokens,
        "generated_target_tokens": generated_target_tokens,
        "target_call_reduction": generated_target_tokens / total_calls,
        "accepted_fraction_of_proposed": accepted_tokens / proposed_tokens,
        "draft_generation_wall_s": sum(row["draft_generation_wall_s"] for row in records),
        "prefill_wall_s": sum(row["prefill_wall_s"] for row in records),
        "spec_decode_wall_s": sum(row["speculative"]["wall_s"] for row in records),
        "composed_pipeline_wall_s": sum(composed),
        "saved_baseline_wall_s": sum(saved),
        "composed_speedup_vs_saved_baseline": sum(saved) / sum(composed),
        "per_table_composed_wall_s": fixed_lab.distribution(composed),
        "per_table_saved_baseline_wall_s": fixed_lab.distribution(saved),
        "per_table_composed_speedup": fixed_lab.distribution(per_table_speedup),
        "per_k": aggregate_per_k,
        "transitions": dict(sorted(transitions.items())),
        "records": str(output_path),
    }
    fixed_lab.write_json(args.output_dir / "run_summary.json", summary)
    print(
        f"TABLE_SPEC_ADAPTIVE_K_COMPLETE tables={len(records)} "
        f"run_wall_s={run_wall_s:.3f} exact_saved={summary['exact_saved_reference']}/{len(records)} "
        f"call_reduction={summary['target_call_reduction']:.3f}x "
        f"composed_speedup={summary['composed_speedup_vs_saved_baseline']:.3f}x "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
