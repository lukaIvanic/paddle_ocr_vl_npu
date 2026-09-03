#!/usr/bin/env python3
"""Validate a completed single-shard custom run and prepare unchanged predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stock_preparer():
    path = Path(__file__).resolve().parents[1] / "17_mineru_vllm_ascend_baseline/prepare_omnidocbench_eval.py"
    spec = importlib.util.spec_from_file_location("mineru_stock_eval_prep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(run_output, dataset_json, evaluation_root, expected_pages, evaluator_root,
            match_workers=24, teds_workers=12, cdm_workers=12):
    stock = stock_preparer()
    summary_path = run_output / "run_summary_shard_00.json"
    summary = load(summary_path)
    if (summary.get("completed") != expected_pages or summary.get("failed") != 0
            or summary.get("skipped") != 0 or summary.get("shard_count") != 1
            or summary.get("shard_index") != 0
            or summary.get("selected_pages") != expected_pages
            or summary.get("shard_pages") != expected_pages):
        raise ValueError("expected a complete, unresumed single-shard run")
    dataset = load(dataset_json)
    offset = int(summary["offset"])
    selected = dataset[offset:offset + expected_pages]
    if expected_pages <= 0 or offset < 0 or len(selected) != expected_pages:
        raise ValueError("invalid dataset selection")
    dataset_hash = stock.sha256_file(dataset_json)
    if summary.get("model_hashes", {}).get("dataset_json") != dataset_hash:
        raise ValueError("dataset hash differs from inference manifest")
    names = [Path(row["page_info"]["image_path"]).name for row in selected]
    stems = [Path(name).stem for name in names]
    if len(set(names)) != expected_pages or len(set(stems)) != expected_pages:
        raise ValueError("duplicate dataset page names or output stems")
    progress_path = run_output / "progress_shard_00.jsonl"
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    if len(progress) != expected_pages or len({r["image"] for r in progress}) != expected_pages:
        raise ValueError("incomplete or duplicate progress records")
    by_name = {row["image"]: row for row in progress}
    if set(by_name) != set(names):
        raise ValueError("progress membership differs from selected pages")
    for directory, suffix in (("predictions", ".md"), ("content_lists", ".json"), ("progress", ".json")):
        actual = {p.stem for p in (run_output / directory).glob(f"*{suffix}")}
        if actual != set(stems):
            raise ValueError(f"{directory} membership differs from selected pages")
    if any((run_output / "failures").iterdir()):
        raise ValueError("run contains failure artifacts")
    pages = []
    for index, (name, stem) in enumerate(zip(names, stems), start=offset):
        row = by_name[name]
        if row.get("status") != "completed" or row.get("dataset_index") != index:
            raise ValueError(f"invalid progress record for {name}")
        if load(run_output / "progress" / f"{stem}.json") != row:
            raise ValueError(f"progress file differs from journal for {name}")
        blocks = load(run_output / "content_lists" / f"{stem}.json")
        markdown = (run_output / "predictions" / f"{stem}.md").read_text(encoding="utf-8")
        if not isinstance(blocks, list) or len(blocks) != row["block_count"]:
            raise ValueError(f"content list differs from progress record for {name}")
        if len(markdown) != row["markdown_chars"]:
            raise ValueError(f"Markdown differs from progress record for {name}")
        pages.append({"image": name, "dataset_index": index})
    if evaluation_root.exists():
        raise FileExistsError(evaluation_root)
    adapter = evaluation_root.with_name(evaluation_root.name + "_input_adapter")
    adapter.mkdir(parents=True, exist_ok=False)
    (adapter / "predictions").symlink_to((run_output / "predictions").resolve(), target_is_directory=True)
    (adapter / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (adapter / "input_manifest.json").write_text(json.dumps({"count": expected_pages, "pages": pages}, indent=2) + "\n")
    result = stock.prepare_evaluation(
        dataset_json=dataset_json, run_output=adapter, evaluation_root=evaluation_root,
        expected_pages=expected_pages, match_workers=match_workers,
        teds_workers=teds_workers, cdm_workers=cdm_workers, evaluator_root=evaluator_root)
    result.update({"custom_run_output": str(run_output), "custom_summary_sha256": stock.sha256_file(summary_path),
                   "progress_journal_sha256": stock.sha256_file(progress_path),
                   "prediction_transform": "none", "input_adapter": str(adapter)})
    (evaluation_root / "evaluation_prep_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--match-workers", type=int, default=24)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--cdm-workers", type=int, default=12)
    args = vars(parser.parse_args())
    for key in ("run_output", "dataset_json", "evaluation_root", "evaluator_root"):
        args[key] = args[key].expanduser().resolve()
    print(json.dumps(prepare(**args), indent=2))


if __name__ == "__main__":
    main()
