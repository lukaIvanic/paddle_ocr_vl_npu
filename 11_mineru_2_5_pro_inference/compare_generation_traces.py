"""Compare stable request identities, exact IDs, frontend inputs and page outputs.

Differences are classified, not automatically forgiven as numerical variation.
Run on the Mac or validation host; no model packages are imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load_trace(path):
    records = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        key = record["request_id"]
        if key in records:
            raise ValueError(f"duplicate trace request: {key}")
        if not record["generated_token_ids"]:
            raise ValueError(f"empty trace generation: {key}")
        records[key] = record
    return records


def compare(reference: Path, candidate: Path, first_pages: int | None = None):
    left = load_trace(reference / "generation_trace.jsonl")
    right = load_trace(candidate / "generation_trace.jsonl")
    selected_pages = None
    if first_pages is not None:
        if first_pages < 1:
            raise ValueError("first-pages must be positive")
        progress = [json.loads(line) for path in reference.glob("progress_shard_*.jsonl")
                    for line in path.read_text().splitlines()]
        selected_pages = {record["image"] for record in sorted(progress, key=lambda x: x["dataset_index"])[:first_pages]}
        if len(selected_pages) != first_pages:
            raise ValueError("reference does not contain requested page prefix")
        left = {key: record for key, record in left.items() if record["page"] in selected_pages}
    totals = Counter()
    differences = []
    unexpected_input_changes = []
    unchanged_layout_pages = {
        record["page"] for key, record in left.items()
        if record["phase"] == "layout" and key in right
        and record["raw_text"] == right[key]["raw_text"]
    }
    fields = ("page", "phase", "block_index", "block_type", "bbox", "angle",
              "image_sha256", "chat_prompt", "prompt_token_ids", "max_new_tokens")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        phase = a["phase"]
        changed_inputs = [field for field in fields if a.get(field) != b.get(field)]
        if changed_inputs and (phase == "layout" or a["page"] in unchanged_layout_pages):
            unexpected_input_changes.append(key)
        same_tokens = a["generated_token_ids"] == b["generated_token_ids"]
        totals[f"{phase}_matched_requests"] += 1
        totals[f"{phase}_input_exact"] += not changed_inputs
        totals[f"{phase}_token_exact"] += same_tokens
        totals[f"{phase}_text_exact"] += a["raw_text"] == b["raw_text"]
        totals[f"{phase}_length_stops"] += b["stop_reason"] == "length"
        if changed_inputs or not same_tokens:
            first = next((i for i, (x, y) in enumerate(zip(a["generated_token_ids"], b["generated_token_ids"])) if x != y),
                         min(len(a["generated_token_ids"]), len(b["generated_token_ids"])))
            differences.append({"request_id": key, "changed_input_fields": changed_inputs,
                                "first_token_difference": first if not same_tokens else None,
                                "reference_tokens": a["generated_token_ids"][max(0, first-3):first+5],
                                "candidate_tokens": b["generated_token_ids"][max(0, first-3):first+5],
                                "reference_length": len(a["generated_token_ids"]),
                                "candidate_length": len(b["generated_token_ids"]),
                                "reference_stop": a["stop_reason"], "candidate_stop": b["stop_reason"]})
    a_pages = {path.name: path for path in (reference / "predictions").glob("*.md")}
    if selected_pages is not None:
        selected_markdown = {Path(page).stem + ".md" for page in selected_pages}
        a_pages = {name: path for name, path in a_pages.items() if name in selected_markdown}
    b_pages = {path.name: path for path in (candidate / "predictions").glob("*.md")}
    pages_changed = [name for name in sorted(a_pages.keys() & b_pages.keys()) if a_pages[name].read_bytes() != b_pages[name].read_bytes()]
    return {"reference": str(reference), "candidate": str(candidate),
            "reference_trace_sha256": hashlib.sha256((reference / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "candidate_trace_sha256": hashlib.sha256((candidate / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "reference_requests": len(left), "candidate_requests": len(right),
            "missing_requests": sorted(left.keys() - right.keys()),
            "extra_requests": sorted(right.keys() - left.keys()),
            "missing_requests_with_unchanged_layout": sorted(key for key in left.keys() - right.keys()
                if left[key]["page"] in unchanged_layout_pages),
            "extra_requests_with_unchanged_layout": sorted(key for key in right.keys() - left.keys()
                if right[key]["page"] in unchanged_layout_pages),
            "unexpected_input_changes": unexpected_input_changes,
            "counts": dict(totals), "differences": differences,
            "missing_pages": sorted(a_pages.keys() - b_pages.keys()),
            "extra_pages": sorted(b_pages.keys() - a_pages.keys()),
            "empty_pages": [name for name, path in b_pages.items() if not path.read_text().strip()],
            "byte_identical_pages": len(a_pages.keys() & b_pages.keys()) - len(pages_changed),
            "changed_pages": pages_changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-pages", type=int, help="Explicitly compare only this reference prefix for a smaller smoke run.")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.first_pages)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("differences", "changed_pages")}, indent=2))
    print(f"changed_pages={len(result['changed_pages'])} differing_requests={len(result['differences'])}")
    if any(result[key] for key in ("missing_pages", "extra_pages", "empty_pages",
           "missing_requests_with_unchanged_layout", "extra_requests_with_unchanged_layout",
           "unexpected_input_changes")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
