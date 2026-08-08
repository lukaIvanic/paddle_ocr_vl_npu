#!/usr/bin/env python3
"""Simulate legal cell-lattice table drafting from saved row OCR outputs.

Candidate construction sees only row-draft text/token IDs, the tokenizer, and
the target tokens which have already been generated.  Future target tokens are
read only by :func:`verify_candidates`, which models one target-model verifier
call over K candidates.  The script therefore measures an implementable draft
policy rather than an oracle candidate generator.

The lattice is deliberately small and general.  It contains raw row streams,
formula-wrapper/bracing variants, compact de-LaTeX variants for numeric/stat
cells, and joined row/cell streams with repaired ``<nl>`` boundaries.  The
matcher can move between these streams after every verified block, so cells do
not need to select one normalization policy for the whole table.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
import unicodedata


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.table_speculative import (  # noqa: E402
    DraftProposal,
    TableDraftMatcher,
)
from table_multicandidate_simulator import (  # noqa: E402
    lcp,
    ranked_candidates,
    read_jsonl,
    target_tokens,
)


STRUCTURAL_MARKERS = (
    "<fcel>",
    "<ecel>",
    "<lcel>",
    "<ucel>",
    "<xcel>",
    "<nl>",
)
STRUCTURAL_PATTERN = re.compile(
    "(" + "|".join(re.escape(marker) for marker in STRUCTURAL_MARKERS) + ")"
)
MATH_WRAPPER_PATTERN = re.compile(
    r"\\\((.*?)\\\)|\\\[(.*?)\\\]|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$",
    re.DOTALL,
)
SIMPLE_LATEX_COMMAND_PATTERN = re.compile(
    r"\\(?:mathrm|mathbf|mathit|textrm|text|operatorname)\s*\{([^{}]*)\}"
)
SINGLE_ATOM_BRACE_PATTERN = re.compile(r"([_^])\s*\{\s*([A-Za-z0-9])\s*\}")
UNBRACED_ATOM_PATTERN = re.compile(r"([_^])\s*([A-Za-z0-9])")
NUMERIC_STAT_ALLOWED_PATTERN = re.compile(
    r"[0-9A-Za-z+\-−.,:;/%‰±×xX·()\[\]<>≤≥=~^_*'\"\\\s]+"
)


@dataclass(frozen=True)
class VariantRecord:
    name: str
    family: str
    record: dict[str, Any]
    priority: int


@dataclass
class MatcherSource:
    name: str
    family: str
    matcher: TableDraftMatcher
    priority: int


@dataclass(frozen=True)
class LatticeCandidate:
    source_index: int
    source_name: str
    family: str
    proposal: DraftProposal
    local_rank: int
    score: tuple[float, float, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-counts",
        "--k",
        default="1,2,4,8,16",
        help="Comma-separated verifier batch sizes K.",
    )
    parser.add_argument(
        "--draft-lengths",
        "--d",
        default="16",
        help="Comma-separated speculative draft lengths D.",
    )
    parser.add_argument("--maximum-anchor", type=int, default=64)
    parser.add_argument("--column-weight", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_positive_ints(value: str, label: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{label} must contain positive integers")
    return list(dict.fromkeys(parsed))


def tokenizer_decode(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))


def tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned more than one encoded row")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def row_text(row: dict[str, Any], tokenizer: Any, eos_token_id: int) -> str:
    tokens = [int(token) for token in row.get("token_ids") or ()]
    if tokens and tokens[-1] == eos_token_id:
        tokens.pop()
    if tokens:
        return tokenizer_decode(tokenizer, tokens)
    return str(row.get("raw_text") or row.get("text") or "")


def split_structural(text: str) -> list[str]:
    return [piece for piece in STRUCTURAL_PATTERN.split(text) if piece]


def normalize_math_body(body: str, *, braced_atoms: bool) -> str:
    value = unicodedata.normalize("NFKC", body)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\{\s+", "{", value)
    value = re.sub(r"\s+\}", "}", value)
    value = re.sub(r"\s*([_^])\s*", r"\1", value)
    value = re.sub(r"\s*([=+,:;])\s*", r"\1", value)
    if braced_atoms:
        value = SINGLE_ATOM_BRACE_PATTERN.sub(r"\1{\2}", value)
        value = UNBRACED_ATOM_PATTERN.sub(r"\1{\2}", value)
    else:
        value = SINGLE_ATOM_BRACE_PATTERN.sub(r"\1\2", value)
    return value


def normalize_formula_wrappers(text: str, *, braced_atoms: bool) -> str:
    """Canonicalize only explicit inline/display math spans in one cell."""

    def replace(match: re.Match[str]) -> str:
        body = next(group for group in match.groups() if group is not None)
        return rf"\({normalize_math_body(body, braced_atoms=braced_atoms)}\)"

    return MATH_WRAPPER_PATTERN.sub(replace, text)


def de_latex(text: str) -> str:
    """Produce a compact plain-text view for a numeric/stat cell candidate."""

    value = unicodedata.normalize("NFKC", text)
    value = MATH_WRAPPER_PATTERN.sub(
        lambda match: next(group for group in match.groups() if group is not None),
        value,
    )
    previous = None
    while previous != value:
        previous = value
        value = SIMPLE_LATEX_COMMAND_PATTERN.sub(r"\1", value)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\%": "%",
        r"\#": "#",
        r"\&": "&",
        r"\_": "_",
        r"\pm": "±",
        r"\times": "×",
        r"\cdot": "·",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\approx": "≈",
        r"\sim": "~",
        r"\,": "",
        r"\;": "",
        r"\!": "",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    value = re.sub(r"[{}]", "", value)
    value = re.sub(r"\s+", "", value)
    return value


def is_numeric_stat_cell(text: str) -> bool:
    plain = de_latex(text)
    if not plain or not any(character.isdigit() for character in plain):
        return False
    if NUMERIC_STAT_ALLOWED_PATTERN.fullmatch(plain) is None:
        return False
    digits = sum(character.isdigit() for character in plain)
    letters = sum(character.isalpha() for character in plain)
    # Allow units and short statistic labels, but not prose cells which merely
    # contain a number.
    return letters <= max(8, digits)


def transform_cell(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    if mode == "formula_braced":
        return normalize_formula_wrappers(text, braced_atoms=True)
    if mode == "formula_unbraced":
        return normalize_formula_wrappers(text, braced_atoms=False)
    if mode == "compact_numeric":
        return de_latex(text) if is_numeric_stat_cell(text) else text
    if mode == "formula_compact":
        normalized = normalize_formula_wrappers(text, braced_atoms=True)
        return de_latex(normalized) if is_numeric_stat_cell(normalized) else normalized
    raise ValueError(f"unknown cell transform mode: {mode}")


def transform_row(text: str, mode: str) -> str:
    pieces = split_structural(text)
    return "".join(
        piece if piece in STRUCTURAL_MARKERS else transform_cell(piece, mode)
        for piece in pieces
    )


def ensure_row_boundary(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    return value if value.endswith("<nl>") else value + "<nl>"


def make_rows_record(
    source: dict[str, Any],
    tokenizer: Any,
    eos_token_id: int,
    *,
    mode: str,
    joined: bool,
) -> dict[str, Any]:
    rows = sorted(source.get("rows") or (), key=lambda item: int(item["row_index"]))
    texts = [transform_row(row_text(row, tokenizer, eos_token_id), mode) for row in rows]
    if joined:
        joined_text = "".join(ensure_row_boundary(text) for text in texts if text.strip())
        transformed_rows = [
            {
                "row_index": 0,
                "text": joined_text,
                "token_ids": tokenizer_encode(tokenizer, joined_text),
            }
        ]
    else:
        transformed_rows = [
            {
                "row_index": index,
                "text": text,
                "token_ids": tokenizer_encode(tokenizer, text),
            }
            for index, text in enumerate(texts)
            if text
        ]
    return {"request_id": source.get("request_id"), "rows": transformed_rows}


def variant_records(
    source: dict[str, Any],
    tokenizer: Any,
    eos_token_id: int,
) -> list[VariantRecord]:
    # The exact token-ID record is retained as the highest-priority path.  All
    # other records are derived mechanically from decoded row cells.
    records = [VariantRecord("raw_rows", "raw", source, 5)]
    specifications = (
        ("formula_braced_rows", "formula_wrapper_bracing", "formula_braced", False, 3),
        ("formula_unbraced_rows", "formula_wrapper_bracing", "formula_unbraced", False, 2),
        ("compact_numeric_rows", "delatex_compact_numeric", "compact_numeric", False, 3),
        ("formula_compact_rows", "formula_and_compact", "formula_compact", False, 2),
        ("joined_raw_cells", "structurally_joined", "raw", True, 4),
        ("joined_formula_cells", "structurally_joined", "formula_braced", True, 2),
        ("joined_compact_cells", "structurally_joined", "formula_compact", True, 2),
    )
    for name, family, mode, joined, priority in specifications:
        records.append(
            VariantRecord(
                name,
                family,
                make_rows_record(
                    source,
                    tokenizer,
                    eos_token_id,
                    mode=mode,
                    joined=joined,
                ),
                priority,
            )
        )
    return records


def build_source_prototypes(
    variants: list[VariantRecord],
    tokenizer: Any,
    *,
    eos_token_id: int,
    draft_length: int,
    maximum_anchor: int,
    column_weight: float,
) -> tuple[list[MatcherSource], dict[str, Any]]:
    result: list[MatcherSource] = []
    seen: dict[tuple[int, ...], str] = {}
    deduplicated: dict[str, str] = {}
    for variant in variants:
        matcher = TableDraftMatcher(
            variant.record,
            tokenizer,
            eos_token_id=eos_token_id,
            block_size=draft_length,
            maximum_anchor=maximum_anchor,
            column_weight=column_weight,
        )
        signature = tuple(matcher.draft)
        if not signature:
            continue
        if signature in seen:
            deduplicated[variant.name] = seen[signature]
            continue
        seen[signature] = variant.name
        result.append(
            MatcherSource(
                variant.name,
                variant.family,
                matcher,
                variant.priority,
            )
        )
    return result, {
        "requested_sources": len(variants),
        "unique_sources": len(result),
        "deduplicated_sources": deduplicated,
        "source_names": [item.name for item in result],
        "families": sorted({item.family for item in result}),
    }


def clone_sources(prototypes: list[MatcherSource]) -> list[MatcherSource]:
    """Share immutable indexes while resetting per-simulation matcher state."""

    result: list[MatcherSource] = []
    for prototype in prototypes:
        matcher = copy.copy(prototype.matcher)
        matcher.cursor = 0
        matcher.structure = type(prototype.matcher.structure)()
        matcher._started = False
        result.append(
            MatcherSource(
                prototype.name,
                prototype.family,
                matcher,
                prototype.priority,
            )
        )
    return result


def candidate_lattice(
    sources: list[MatcherSource],
    prefix: list[int],
    candidate_count: int,
) -> list[LatticeCandidate]:
    """Construct candidates without access to ungenerated target tokens."""

    unique: dict[tuple[int, ...], LatticeCandidate] = {}
    for source_index, source in enumerate(sources):
        local = ranked_candidates(source.matcher, prefix, candidate_count)
        for item in local:
            candidate = LatticeCandidate(
                source_index=source_index,
                source_name=source.name,
                family=source.family,
                proposal=item.proposal,
                local_rank=item.rank,
                score=(*item.score, source.priority),
            )
            previous = unique.get(candidate.proposal.tokens)
            if previous is None or candidate.score > previous.score:
                unique[candidate.proposal.tokens] = candidate
    return sorted(unique.values(), key=lambda item: item.score, reverse=True)[
        :candidate_count
    ]


def verify_candidates(
    candidates: list[LatticeCandidate],
    future_target: list[int],
) -> tuple[LatticeCandidate, int]:
    """Model one K-way target call; this is the only future-target reader."""

    scored = [
        (lcp(candidate.proposal.tokens, future_target), -rank, candidate)
        for rank, candidate in enumerate(candidates)
    ]
    accepted, _negative_rank, winner = max(scored, key=lambda item: item[:2])
    return winner, accepted


def simulate_one(
    target: list[int],
    source_prototypes: list[MatcherSource],
    source_manifest: dict[str, Any],
    *,
    candidate_count: int,
) -> dict[str, Any]:
    sources = clone_sources(source_prototypes)
    if not sources:
        raise ValueError("no usable draft source")

    for source in sources:
        source.matcher.start(target[0])
    position = 1
    calls = 0
    speculative_calls = 0
    fallback_calls = 0
    accepted = 0
    candidate_branches = 0
    proposed = 0
    accept_lengths: list[int] = []
    winners: Counter[str] = Counter()
    winner_families: Counter[str] = Counter()
    proposed_families: Counter[str] = Counter()

    while position < len(target):
        prefix = target[:position]
        candidates = candidate_lattice(sources, prefix, candidate_count)
        calls += 1
        if not candidates:
            fallback_calls += 1
            emitted = [target[position]]
            for source in sources:
                source.matcher.commit(
                    None,
                    accepted_draft_tokens=0,
                    emitted_tokens=emitted,
                )
            position += 1
            continue

        speculative_calls += 1
        candidate_branches += len(candidates)
        proposed += sum(len(item.proposal.tokens) for item in candidates)
        proposed_families.update(item.family for item in candidates)
        # Candidate construction above receives prefix only.  The simulated
        # target verifier is intentionally isolated at this boundary.
        winner, accepted_here = verify_candidates(candidates, target[position:])
        accepted += accepted_here
        accept_lengths.append(accepted_here)
        winners[winner.source_name] += 1
        winner_families[winner.family] += 1
        emitted = list(winner.proposal.tokens[:accepted_here])
        if position + accepted_here < len(target):
            emitted.append(target[position + accepted_here])
        for source_index, source in enumerate(sources):
            source.matcher.commit(
                winner.proposal if source_index == winner.source_index else None,
                accepted_draft_tokens=(
                    accepted_here if source_index == winner.source_index else 0
                ),
                emitted_tokens=emitted,
            )
        position += len(emitted)

    baseline = len(target) - 1
    return {
        "baseline_decode_iterations": baseline,
        "target_calls": calls,
        "target_call_reduction": baseline / calls if calls else None,
        "speculative_calls": speculative_calls,
        "fallback_calls": fallback_calls,
        "accepted_draft_tokens": accepted,
        "accepted_tokens_per_speculative_call": (
            accepted / speculative_calls if speculative_calls else 0.0
        ),
        "mean_accept_length": statistics.mean(accept_lengths) if accept_lengths else 0.0,
        "candidate_branches": candidate_branches,
        "mean_candidates_per_speculative_call": (
            candidate_branches / speculative_calls if speculative_calls else 0.0
        ),
        "mean_proposed_tokens_per_target_call": proposed / calls if calls else 0.0,
        "winner_sources": dict(winners.most_common()),
        "winner_families": dict(winner_families.most_common()),
        "proposed_families": dict(proposed_families.most_common()),
        "source_manifest": source_manifest,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = sum(row["simulation"]["baseline_decode_iterations"] for row in rows)
    calls = sum(row["simulation"]["target_calls"] for row in rows)
    speculative = sum(row["simulation"]["speculative_calls"] for row in rows)
    accepted = sum(row["simulation"]["accepted_draft_tokens"] for row in rows)
    branches = sum(row["simulation"]["candidate_branches"] for row in rows)
    winners: Counter[str] = Counter()
    proposed: Counter[str] = Counter()
    for row in rows:
        winners.update(row["simulation"]["winner_families"])
        proposed.update(row["simulation"]["proposed_families"])
    return {
        "tables": len(rows),
        "baseline_decode_iterations": baseline,
        "target_calls": calls,
        "target_call_reduction": baseline / calls if calls else None,
        "speculative_calls": speculative,
        "fallback_calls": sum(row["simulation"]["fallback_calls"] for row in rows),
        "accepted_draft_tokens": accepted,
        "accepted_tokens_per_speculative_call": (
            accepted / speculative if speculative else 0.0
        ),
        "candidate_branches": branches,
        "mean_candidates_per_speculative_call": (
            branches / speculative if speculative else 0.0
        ),
        "winner_families": dict(winners.most_common()),
        "proposed_families": dict(proposed.most_common()),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    candidate_counts = parse_positive_ints(args.candidate_counts, "K")
    draft_lengths = parse_positive_ints(args.draft_lengths, "D")
    targets = {record["request_id"]: record for record in read_jsonl(args.targets)}
    drafts = {record["request_id"]: record for record in read_jsonl(args.drafts)}
    request_ids = sorted(set(targets) & set(drafts))
    if args.limit is not None:
        request_ids = request_ids[: args.limit]
    if not request_ids:
        raise ValueError("targets and drafts have no matching request IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    detailed: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_work = len(request_ids) * len(draft_lengths) * len(candidate_counts)
    completed = 0
    for request_id in request_ids:
        target = target_tokens(targets[request_id])
        variants = variant_records(drafts[request_id], tokenizer, target[-1])
        for draft_length in draft_lengths:
            source_prototypes, source_manifest = build_source_prototypes(
                variants,
                tokenizer,
                eos_token_id=target[-1],
                draft_length=draft_length,
                maximum_anchor=args.maximum_anchor,
                column_weight=args.column_weight,
            )
            for candidate_count in candidate_counts:
                detailed.append(
                    {
                        "request_id": request_id,
                        "candidate_count": candidate_count,
                        "draft_length": draft_length,
                        "simulation": simulate_one(
                            target,
                            source_prototypes,
                            source_manifest,
                            candidate_count=candidate_count,
                        ),
                    }
                )
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == total_work:
                    elapsed = time.perf_counter() - started
                    print(
                        f"progress={completed}/{total_work} "
                        f"elapsed_s={elapsed:.1f} simulations_per_s={completed / elapsed:.2f}",
                        flush=True,
                    )

    aggregate_by_shape: dict[str, Any] = {}
    for draft_length in draft_lengths:
        for candidate_count in candidate_counts:
            key = f"K{candidate_count}_D{draft_length}"
            aggregate_by_shape[key] = aggregate(
                [
                    row
                    for row in detailed
                    if row["candidate_count"] == candidate_count
                    and row["draft_length"] == draft_length
                ]
            )

    result = {
        "configuration": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "tokenizer": str(args.tokenizer),
            "candidate_counts": candidate_counts,
            "draft_lengths": draft_lengths,
            "maximum_anchor": args.maximum_anchor,
            "column_weight": args.column_weight,
            "limit": args.limit,
        },
        "legality_contract": {
            "candidate_inputs": [
                "row-draft token IDs and decoded text",
                "tokenizer",
                "generated target prefix and observed OTSL structure",
            ],
            "future_target_access": "verify_candidates LCP scoring only",
            "data_specific_rules": False,
        },
        "candidate_families": [
            "raw",
            "formula_wrapper_bracing",
            "delatex_compact_numeric",
            "formula_and_compact",
            "structurally_joined",
        ],
        "aggregate": aggregate_by_shape,
        "detailed": detailed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, metrics in aggregate_by_shape.items():
        print(
            f"{key} calls={metrics['target_calls']} "
            f"reduction={metrics['target_call_reduction']:.3f}x "
            f"accepted/spec={metrics['accepted_tokens_per_speculative_call']:.3f} "
            f"mean_candidates={metrics['mean_candidates_per_speculative_call']:.2f}",
            flush=True,
        )
    print(f"wrote={output}", flush=True)


if __name__ == "__main__":
    main()
