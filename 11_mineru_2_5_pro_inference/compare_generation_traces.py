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


def compare(reference: Path, candidate: Path):
    left = load_trace(reference / "generation_trace.jsonl")
    right = load_trace(candidate / "generation_trace.jsonl")
    totals = Counter()
    differences = []
    fields = ("page", "phase", "block_index", "block_type", "bbox", "angle",
              "image_sha256", "chat_prompt", "prompt_token_ids", "max_new_tokens")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        phase = a["phase"]
        changed_inputs = [field for field in fields if a.get(field) != b.get(field)]
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
    b_pages = {path.name: path for path in (candidate / "predictions").glob("*.md")}
    pages_changed = [name for name in sorted(a_pages.keys() & b_pages.keys()) if a_pages[name].read_bytes() != b_pages[name].read_bytes()]
    return {"reference": str(reference), "candidate": str(candidate),
            "reference_trace_sha256": hashlib.sha256((reference / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "candidate_trace_sha256": hashlib.sha256((candidate / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "reference_requests": len(left), "candidate_requests": len(right),
            "missing_requests": sorted(left.keys() - right.keys()),
            "extra_requests": sorted(right.keys() - left.keys()),
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
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("differences", "changed_pages")}, indent=2))
    print(f"changed_pages={len(result['changed_pages'])} differing_requests={len(result['differences'])}")
    if result["missing_pages"] or result["extra_pages"] or result["empty_pages"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
