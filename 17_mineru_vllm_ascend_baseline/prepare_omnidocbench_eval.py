#!/usr/bin/env python3
"""Prepare one completed experiment-17 run for OmniDocBench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--match-workers", type=int, default=24)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--cdm-workers", type=int, default=12)
    parser.add_argument("--evaluator-root", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_name(sample: dict[str, Any]) -> str:
    return Path(sample["page_info"]["image_path"]).name


def git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_evaluation(
    *,
    dataset_json: Path,
    run_output: Path,
    evaluation_root: Path,
    expected_pages: int,
    match_workers: int,
    teds_workers: int,
    cdm_workers: int,
    evaluator_root: Path | None,
) -> dict[str, Any]:
    if expected_pages <= 0:
        raise ValueError("expected_pages must be positive")
    for name, value in (
        ("match_workers", match_workers),
        ("teds_workers", teds_workers),
        ("cdm_workers", cdm_workers),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if evaluation_root.exists():
        raise FileExistsError(evaluation_root)

    summary_path = run_output / "run_summary.json"
    input_manifest_path = run_output / "input_manifest.json"
    predictions_source = run_output / "predictions"
    summary = load_json(summary_path)
    input_manifest = load_json(input_manifest_path)
    if summary.get("completed") != expected_pages or summary.get("failed") != 0:
        raise RuntimeError(
            f"incomplete run: completed={summary.get('completed')} "
            f"failed={summary.get('failed')}"
        )
    if summary.get("selected_pages") != expected_pages:
        raise RuntimeError("run summary selected_pages does not match expected_pages")
    if input_manifest.get("count") != expected_pages:
        raise RuntimeError("input manifest count does not match expected_pages")

    selected_names = [str(row["image"]) for row in input_manifest["pages"]]
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError("input manifest contains duplicate image names")
    selected_stems = [Path(name).stem for name in selected_names]
    if len(selected_stems) != len(set(selected_stems)):
        raise RuntimeError("input manifest contains duplicate output stems")

    dataset = load_json(dataset_json)
    dataset_by_name: dict[str, dict[str, Any]] = {}
    for sample in dataset:
        name = image_name(sample)
        if name in dataset_by_name:
            raise RuntimeError(f"dataset contains duplicate image name: {name}")
        dataset_by_name[name] = sample
    missing_ground_truth = [name for name in selected_names if name not in dataset_by_name]
    if missing_ground_truth:
        raise RuntimeError(
            f"selected pages missing from ground truth: {missing_ground_truth[:5]}"
        )
    selected_dataset = [dataset_by_name[name] for name in selected_names]

    prediction_paths = sorted(predictions_source.glob("*.md"))
    prediction_by_stem = {path.stem: path for path in prediction_paths}
    if len(prediction_paths) != expected_pages or len(prediction_by_stem) != expected_pages:
        raise RuntimeError(
            f"expected {expected_pages} unique Markdown predictions, "
            f"found {len(prediction_paths)} files and {len(prediction_by_stem)} stems"
        )
    missing_predictions = [stem for stem in selected_stems if stem not in prediction_by_stem]
    extra_predictions = sorted(set(prediction_by_stem) - set(selected_stems))
    if missing_predictions or extra_predictions:
        raise RuntimeError(
            f"prediction membership mismatch: missing={missing_predictions[:5]} "
            f"extra={extra_predictions[:5]}"
        )

    predictions = evaluation_root / "predictions"
    work = evaluation_root / "work"
    predictions.mkdir(parents=True)
    work.mkdir()
    subset_path = evaluation_root / "OmniDocBench_subset.json"
    subset_path.write_text(
        json.dumps(selected_dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prediction_manifest = []
    for name, stem in zip(selected_names, selected_stems):
        source = prediction_by_stem[stem].resolve()
        if source.stat().st_size == 0:
            raise RuntimeError(f"empty prediction: {source}")
        target = predictions / f"{stem}.md"
        os.symlink(source, target)
        prediction_manifest.append(
            {
                "image": name,
                "source": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    prediction_manifest_path = evaluation_root / "prediction_manifest.json"
    prediction_manifest_path.write_text(
        json.dumps(prediction_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config = f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: {cdm_workers}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {teds_workers}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {json.dumps(str(subset_path.resolve()))}
    prediction:
      data_path: {json.dumps(str(predictions.resolve()))}
    match_method: quick_match
    match_workers: {match_workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
    config_path = work / "config.yaml"
    config_path.write_text(config, encoding="utf-8")

    result = {
        "status": "ok",
        "page_count": expected_pages,
        "source_run_output": str(run_output),
        "source_run_commit": summary.get("git_commit"),
        "dataset_json": str(dataset_json),
        "dataset_pages": len(dataset),
        "dataset_sha256": sha256_file(dataset_json),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "run_summary_sha256": sha256_file(summary_path),
        "prediction_manifest": str(prediction_manifest_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "config": str(config_path),
        "evaluator_root": None if evaluator_root is None else str(evaluator_root),
        "evaluator_commit": git_commit(evaluator_root),
        "match_workers": match_workers,
        "teds_workers": teds_workers,
        "cdm_workers": cdm_workers,
    }
    (evaluation_root / "evaluation_prep_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    result = prepare_evaluation(
        dataset_json=args.dataset_json.expanduser().resolve(),
        run_output=args.run_output.expanduser().resolve(),
        evaluation_root=args.evaluation_root.expanduser().resolve(),
        expected_pages=args.expected_pages,
        match_workers=args.match_workers,
        teds_workers=args.teds_workers,
        cdm_workers=args.cdm_workers,
        evaluator_root=(
            None
            if args.evaluator_root is None
            else args.evaluator_root.expanduser().resolve()
        ),
    )
    print("EXPERIMENT17_EVAL_PREP " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
