#!/usr/bin/env python3
"""Audit token-only repetition-stop candidates over a frozen OCR trace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Optional
import zipfile

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.repetition import (
    RepetitionEvidence,
    dominant_token_window,
    duplicate_excess_window,
    exact_repeating_suffix,
    periodic_matches_window,
    run_self_checks,
)


Rule = Callable[[list[int]], Optional[RepetitionEvidence]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "recognition_trace.jsonl, an Experiment-09 output directory, or "
            "a .gdatlas.zip containing recognition_trace.jsonl"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=200)
    args = parser.parse_args()
    if args.case_limit <= 0:
        parser.error("--case-limit must be positive")
    return args


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "recognition_trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            with archive.open("recognition_trace.jsonl") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def visible_tokens(row: dict[str, Any]) -> list[int]:
    tokens = [int(value) for value in row.get("token_ids") or ()]
    if row.get("stop_reason") == "eos" and tokens:
        tokens.pop()
    return tokens


def rules() -> dict[str, Rule]:
    return {
        "dominant_token_20_of_30": lambda tokens: dominant_token_window(tokens),
        "duplicate_excess_20_of_30": lambda tokens: duplicate_excess_window(tokens),
        "periodic_matches_20_of_30": lambda tokens: periodic_matches_window(tokens),
        "exact_cycle_4copies_60tokens_p32": lambda tokens: exact_repeating_suffix(
            tokens,
            min_repeat_copies=4,
            min_repeated_span=60,
            max_period=32,
        ),
        "exact_cycle_6copies_90tokens_p32": lambda tokens: exact_repeating_suffix(
            tokens,
            min_repeat_copies=6,
            min_repeated_span=90,
            max_period=32,
        ),
        "exact_cycle_6copies_128tokens_p32": lambda tokens: exact_repeating_suffix(
            tokens,
            min_repeat_copies=6,
            min_repeated_span=128,
            max_period=32,
        ),
    }


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _case(
    row: dict[str, Any],
    tokens: list[int],
    evidence: RepetitionEvidence,
) -> dict[str, Any]:
    trigger = evidence.trigger_length
    return {
        "request_id": str(row.get("request_id")),
        "source_image_name": str(row.get("source_image_name")),
        "block_index": row.get("block_index"),
        "label": str(row.get("label")),
        "original_stop_reason": row.get("stop_reason"),
        "generated_tokens_without_eos": len(tokens),
        "estimated_tokens_prevented": max(0, len(tokens) - trigger),
        "evidence": evidence.to_dict(),
        "tokens_before_trigger": tokens[max(0, trigger - 64) : trigger],
        "tokens_after_trigger": tokens[trigger : trigger + 64],
        "decoded_text": str(row.get("text") or ""),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Token-only repetition stop audit",
        "",
        f"Input generations: **{report['generations']:,}**  ",
        f"Pages: **{report['pages']:,}**  ",
        f"Visible generated tokens: **{report['visible_generated_tokens']:,}**",
        "",
        "No rule uses request IDs, labels, decoded text, language, reference "
        "outputs, or ground truth. Labels and text appear only in this report "
        "for human review.",
        "",
        "| Rule | Hits | Pages | Estimated tokens prevented |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in report["rules"].items():
        lines.append(
            f"| `{name}` | {summary['hits']:,} | {summary['pages']:,} | "
            f"{summary['estimated_tokens_prevented']:,} |"
        )
    lines.extend(["", "## Review cases", ""])
    for name, cases in report["cases"].items():
        lines.append(f"### `{name}`")
        lines.append("")
        if not cases:
            lines.append("No hits.")
            lines.append("")
            continue
        for case in cases:
            evidence = case["evidence"]
            text = case["decoded_text"].replace("\n", "\\n")
            if len(text) > 500:
                text = text[:500] + "…"
            lines.extend(
                [
                    f"- `{case['request_id']}` — label `{case['label']}`, "
                    f"tokens {case['generated_tokens_without_eos']}, trigger "
                    f"{evidence['trigger_length']}, period {evidence['period']}, "
                    f"prevented {case['estimated_tokens_prevented']}",
                    f"  - text: `{text}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_self_checks()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = list(iter_rows(args.input))
    if not rows:
        raise RuntimeError("input contains no generations")

    configured_rules = rules()
    hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages = {str(row.get("source_image_name")) for row in rows}
    total_tokens = 0
    for row_index, row in enumerate(rows, 1):
        tokens = visible_tokens(row)
        total_tokens += len(tokens)
        for name, rule in configured_rules.items():
            evidence = rule(tokens)
            if evidence is not None:
                if evidence.rule != name:
                    raise AssertionError((evidence.rule, name))
                hits[name].append(_case(row, tokens, evidence))
        if row_index % 5000 == 0 or row_index == len(rows):
            print(
                f"repetition-audit progress={row_index}/{len(rows)}",
                flush=True,
            )

    summaries: dict[str, Any] = {}
    review_cases: dict[str, Any] = {}
    for name in configured_rules:
        selected = sorted(
            hits[name],
            key=lambda case: (
                -int(case["estimated_tokens_prevented"]),
                -int(case["generated_tokens_without_eos"]),
                str(case["request_id"]),
            ),
        )
        summaries[name] = {
            "hits": len(selected),
            "pages": len({case["source_image_name"] for case in selected}),
            "estimated_tokens_prevented": sum(
                int(case["estimated_tokens_prevented"]) for case in selected
            ),
            "by_label": _counter(case["label"] for case in selected),
            "by_original_stop_reason": _counter(
                case["original_stop_reason"] for case in selected
            ),
            "trigger_length_min": (
                min(case["evidence"]["trigger_length"] for case in selected)
                if selected
                else None
            ),
            "trigger_length_max": (
                max(case["evidence"]["trigger_length"] for case in selected)
                if selected
                else None
            ),
        }
        review_cases[name] = selected[: args.case_limit]

    report = {
        "schema_version": 1,
        "input": str(args.input.expanduser().resolve()),
        "generations": len(rows),
        "pages": len(pages),
        "visible_generated_tokens": total_tokens,
        "runtime_inputs": ["generated_token_ids_only"],
        "forbidden_runtime_inputs": [
            "request_id",
            "crop_label",
            "decoded_text",
            "language_or_script",
            "reference_generation",
            "ground_truth",
        ],
        "rules": summaries,
        "cases": review_cases,
    }

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "all_hits.jsonl").open("w", encoding="utf-8") as handle:
        for name in configured_rules:
            for case in hits[name]:
                handle.write(
                    json.dumps(
                        {"rule": name, **case},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    write_markdown(report, output / "report.md")
    print(json.dumps({
        "generations": report["generations"],
        "pages": report["pages"],
        "visible_generated_tokens": report["visible_generated_tokens"],
        "rules": report["rules"],
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
