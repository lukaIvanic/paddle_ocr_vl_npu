#!/usr/bin/env python3
"""Materialize flat OmniDocBench evaluator inputs from completed UniRec runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--b64-output", type=Path, required=True)
    parser.add_argument("--b128-output", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    return parser.parse_args()


def prediction_path(output: Path, stem: str) -> Path:
    candidates = (
        output / stem / f"{stem}.md",
        output / "predictions" / f"{stem}.md",
    )
    found = [path for path in candidates if path.is_file()]
    if not found:
        found = list(output.rglob(f"{stem}.md"))
    unique = sorted({path.resolve() for path in found})
    if len(unique) != 1:
        raise RuntimeError(
            f"expected one Markdown prediction for {stem} under {output}, "
            f"found {len(unique)}: {unique[:5]}"
        )
    return unique[0]


def validate_summary(output: Path, expected_pages: int) -> dict[str, Any] | None:
    path = output / "run_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise RuntimeError(f"run is not complete: {path}")
    page_count = summary.get("page_count", summary.get("count"))
    if page_count != expected_pages:
        raise RuntimeError(
            f"expected {expected_pages} pages in {path}, got {page_count}"
        )
    return summary


def materialize_lane(
    *,
    name: str,
    output: Path,
    evaluation_root: Path,
    dataset: list[dict[str, Any]],
) -> tuple[Path, dict[str, bytes]]:
    lane = evaluation_root / name
    predictions = lane / "predictions"
    work = lane / "work"
    predictions.mkdir(parents=True)
    work.mkdir()
    (lane / "OmniDocBench_subset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    content: dict[str, bytes] = {}
    manifest = []
    for sample in dataset:
        image_name = Path(sample["page_info"]["image_path"]).name
        stem = Path(image_name).stem
        source = prediction_path(output, stem)
        payload = source.read_bytes()
        target = predictions / f"{stem}.md"
        os.symlink(source, target)
        digest = hashlib.sha256(payload).hexdigest()
        content[f"{stem}.md"] = payload
        manifest.append(
            {
                "image": image_name,
                "prediction": str(source),
                "bytes": len(payload),
                "sha256": digest,
            }
        )
    (lane / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    config = f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist]
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 12
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {json.dumps(str((lane / 'OmniDocBench_subset.json').resolve()))}
    prediction:
      data_path: {json.dumps(str(predictions.resolve()))}
    match_method: quick_match
    match_workers: 12
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
    (work / "config.yaml").write_text(config, encoding="utf-8")
    return lane, content


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_json.expanduser().resolve()
    b64_output = args.b64_output.expanduser().resolve()
    b128_output = args.b128_output.expanduser().resolve()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    if evaluation_root.exists():
        raise FileExistsError(evaluation_root)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if len(dataset) != 1651:
        raise RuntimeError(f"expected 1651 OmniDocBench pages, got {len(dataset)}")
    stems = [Path(item["page_info"]["image_path"]).stem for item in dataset]
    if len(set(stems)) != len(stems):
        raise RuntimeError("dataset contains duplicate image stems")
    validate_summary(b64_output, len(dataset))
    validate_summary(b128_output, len(dataset))

    evaluation_root.mkdir(parents=True)
    b64_lane, b64 = materialize_lane(
        name="b64",
        output=b64_output,
        evaluation_root=evaluation_root,
        dataset=dataset,
    )
    b128_lane, b128 = materialize_lane(
        name="b128",
        output=b128_output,
        evaluation_root=evaluation_root,
        dataset=dataset,
    )
    differing = [name for name in sorted(b64) if b64[name] != b128[name]]
    comparison = {
        "page_count": len(dataset),
        "identical_count": len(dataset) - len(differing),
        "differing_count": len(differing),
        "differing_files": differing,
        "b64_lane": str(b64_lane),
        "b128_lane": str(b128_lane),
    }
    (evaluation_root / "prediction_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print("UNIREC_EVAL_PREP " + json.dumps(comparison, separators=(",", ":")))


if __name__ == "__main__":
    main()
