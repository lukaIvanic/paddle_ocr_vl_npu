#!/usr/bin/env python3
"""Prepare one completed UniRec page subset for OmniDocBench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--page-manifest",
        type=Path,
        help=(
            "Optional unirec_representative_pages_v1 manifest. When supplied, "
            "select these page stems from either a subset or full-run output."
        ),
    )
    parser.add_argument("--strip-image-tags", action="store_true")
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


def index_prediction_paths(
    output: Path, expected_count: int | None
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in output.rglob("*.md"):
        stem = path.stem
        if stem in indexed:
            raise RuntimeError(
                f"duplicate Markdown prediction stem {stem}: {indexed[stem]}, {path}"
            )
        indexed[stem] = path.resolve()
    if expected_count is not None and len(indexed) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} Markdown predictions under {output}, "
            f"found {len(indexed)}"
        )
    return indexed


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_summary(
    output: Path,
    offset: int,
    limit: int,
    *,
    allow_superset: bool,
) -> dict[str, Any]:
    path = output / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise RuntimeError(f"run is not complete: {path}")
    page_count = summary.get("page_count", summary.get("count"))
    if allow_superset:
        if page_count < limit:
            raise RuntimeError(
                f"expected at least {limit} pages in {path}, got {page_count}"
            )
    elif page_count != limit:
        raise RuntimeError(f"expected {limit} pages in {path}, got {page_count}")
    run_offset = summary.get("offset")
    if not allow_superset and run_offset is not None and run_offset != offset:
        raise RuntimeError(f"expected offset {offset} in {path}, got {run_offset}")
    return summary


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")

    dataset_path = args.dataset_json.expanduser().resolve()
    output = args.output.expanduser().resolve()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    if evaluation_root.exists():
        raise FileExistsError(evaluation_root)

    full_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected_stems: set[str] | None = None
    page_manifest_path: Path | None = None
    selection_sha256: str | None = None
    if args.page_manifest is not None:
        page_manifest_path = args.page_manifest.expanduser().resolve()
        page_manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))
        if page_manifest.get("schema") != "unirec_representative_pages_v1":
            raise ValueError("unsupported page-manifest schema")
        filenames = [str(row["filename"]) for row in page_manifest["pages"]]
        if len(filenames) != args.limit:
            raise ValueError(
                f"page-manifest has {len(filenames)} pages, expected {args.limit}"
            )
        selected_stems = {Path(filename).stem for filename in filenames}
        if len(selected_stems) != args.limit:
            raise ValueError("page-manifest contains duplicate image stems")
        selection_sha256 = str(page_manifest["selection"]["selection_sha256"])

    summary = validate_summary(
        output,
        args.offset,
        args.limit,
        allow_superset=selected_stems is not None,
    )
    all_prediction_paths = index_prediction_paths(
        output,
        None if selected_stems is not None else args.limit,
    )
    if selected_stems is None:
        prediction_paths = all_prediction_paths
    else:
        missing_predictions = sorted(selected_stems - set(all_prediction_paths))
        if missing_predictions:
            raise RuntimeError(
                "page-manifest predictions are missing from output: "
                f"{missing_predictions[:10]}"
            )
        prediction_paths = {
            stem: all_prediction_paths[stem]
            for stem in selected_stems
        }
    dataset_by_stem: dict[str, dict[str, Any]] = {}
    for item in full_dataset:
        stem = Path(item["page_info"]["image_path"]).stem
        if stem in dataset_by_stem:
            raise RuntimeError(f"full dataset contains duplicate image stem {stem}")
        dataset_by_stem[stem] = item
    missing_ground_truth = sorted(set(prediction_paths) - set(dataset_by_stem))
    if missing_ground_truth:
        raise RuntimeError(
            "predictions are missing from OmniDocBench ground truth: "
            f"{missing_ground_truth[:10]}"
        )
    # The production runner defines offset/limit over the sorted image-file list,
    # not over OmniDocBench.json order. The completed output names are therefore
    # the authoritative membership list for this subset.
    dataset = [dataset_by_stem[stem] for stem in sorted(prediction_paths)]

    predictions = evaluation_root / "predictions"
    work = evaluation_root / "work"
    predictions.mkdir(parents=True)
    work.mkdir()
    subset_json = evaluation_root / "OmniDocBench_subset.json"
    subset_json.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest: list[dict[str, Any]] = []
    total_removed = 0
    for sample in dataset:
        image_name = Path(sample["page_info"]["image_path"]).name
        stem = Path(image_name).stem
        source = prediction_paths[stem]
        original = source.read_text(encoding="utf-8")
        transformed, removed = (
            IMAGE_TAG_RE.subn("", original) if args.strip_image_tags else (original, 0)
        )
        total_removed += removed
        target = predictions / f"{stem}.md"
        target.write_text(transformed, encoding="utf-8")
        original_bytes = original.encode("utf-8")
        transformed_bytes = transformed.encode("utf-8")
        manifest.append(
            {
                "image": image_name,
                "source_prediction": str(source),
                "evaluation_prediction": str(target),
                "removed_image_tags": removed,
                "original_bytes": len(original_bytes),
                "evaluation_bytes": len(transformed_bytes),
                "original_sha256": sha256(original_bytes),
                "evaluation_sha256": sha256(transformed_bytes),
            }
        )

    manifest_path = evaluation_root / "prediction_transform_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    transform_summary = {
        "status": "ok",
        "offset": args.offset,
        "limit": args.limit,
        "page_count": len(dataset),
        "strip_image_tags": args.strip_image_tags,
        "removed_image_tags": total_removed,
        "source_output": str(output),
        "source_run_summary": str(output / "run_summary.json"),
        "source_run_commit": summary.get("git_commit"),
        "page_manifest": (
            None if page_manifest_path is None else str(page_manifest_path)
        ),
        "selection_sha256": selection_sha256,
        "prediction_transform_manifest": str(manifest_path),
    }
    (evaluation_root / "transform_summary.json").write_text(
        json.dumps(transform_summary, indent=2) + "\n",
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
      data_path: {json.dumps(str(subset_json.resolve()))}
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
    print("UNIREC_SUBSET_EVAL_PREP " + json.dumps(transform_summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
