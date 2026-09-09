"""Copy a completed hybrid run into the established OmniDocBench eval contract."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def prepare(output, dataset_json, evaluation_root, expected_pages=1651):
    output, dataset_json, evaluation_root = (
        Path(p).resolve() for p in (output, dataset_json, evaluation_root))
    if evaluation_root.exists():
        raise FileExistsError(evaluation_root)
    if evaluation_root.is_relative_to(output):
        raise ValueError("Keep evaluation outside the immutable inference output")
    summary_path = output / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    dataset = json.loads(dataset_json.read_text())
    stems = [Path(row["page_info"]["image_path"]).stem for row in dataset]
    if summary["pages"] != expected_pages or len(stems) != expected_pages:
        raise ValueError("Expected a complete full-corpus run and ground truth")
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate ground-truth stems")
    indexed = {}
    for path in output.rglob("*.md"):
        if path.stem in indexed:
            raise ValueError(f"Duplicate prediction: {path.stem}")
        indexed[path.stem] = path
    if set(indexed) != set(stems):
        raise ValueError("Prediction membership differs from ground truth")
    for engine in summary["engines"].values():
        stats = engine["summary"]
        if "completed" in stats and stats["completed"] != stats["submitted"]:
            raise ValueError("Incomplete recognizer")
        pool = engine.get("ready_kv_pool")
        if pool and pool["active_slots"]:
            raise ValueError("Ready pool has not drained")

    predictions = evaluation_root / "predictions"
    work = evaluation_root / "work"
    predictions.mkdir(parents=True)
    work.mkdir()
    # Only HTML image tags are removed, matching the established UniRec eval
    # copy convention. Preserve all text, tables, formulas and original files.
    image_tags = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
    manifest = []
    for stem in stems:
        source = indexed[stem]
        original = source.read_bytes()
        transformed, count = image_tags.subn("", original.decode("utf-8"))
        payload = transformed.encode("utf-8")
        (predictions / f"{stem}.md").write_bytes(payload)
        manifest.append(dict(stem=stem, source=str(source), removed_image_tags=count,
                             original_sha256=hashlib.sha256(original).hexdigest(),
                             evaluation_sha256=hashlib.sha256(payload).hexdigest()))
    (evaluation_root / "prediction_transform_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (work / "config.yaml").write_text(f"""end2end_eval:
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
      data_path: {json.dumps(str(dataset_json))}
    prediction:
      data_path: {json.dumps(str(predictions))}
    match_method: quick_match
    match_workers: 12
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
""")
    report = dict(pages=len(stems), source_output=str(output),
                  source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(dataset_json.read_bytes()).hexdigest(),
                  removed_image_tags=sum(row["removed_image_tags"] for row in manifest),
                  config=str(work / "config.yaml"))
    (evaluation_root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.dataset_json, args.evaluation_root), indent=2))
