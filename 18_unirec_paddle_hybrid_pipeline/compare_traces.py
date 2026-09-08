"""Compare saved crop outputs, restricting parity to the same routed model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_trace(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["request_id"]
            if key in rows:
                raise ValueError(f"Duplicate request {key} in {path}")
            rows[key] = row
    return rows


def compare(reference, candidate):
    counts = defaultdict(Counter)
    mismatches = []
    for key, row in candidate.items():
        if key not in reference:
            raise ValueError(f"Candidate request missing from reference: {key}")
        previous = reference[key]
        for field in ("page", "block_index", "prompt"):
            if previous[field] != row[field]:
                raise ValueError(f"Crop metadata changed: {key}, {field}")
        if previous["model"] != row["model"]:
            counts["different_routing"]["skipped"] += 1
            continue
        result = counts[row["model"]]
        result["compared"] += 1
        for field in ("token_ids", "text", "stop_reason"):
            result[f"{field}_equal"] += previous[field] == row[field]
        if previous["token_ids"] != row["token_ids"] and len(mismatches) < 20:
            mismatches.append({"request_id": key, "model": row["model"],
                               "reference_tokens": len(previous["token_ids"]),
                               "candidate_tokens": len(row["token_ids"]),
                               "text_equal": previous["text"] == row["text"]})
    return {"reference_requests": len(reference), "candidate_requests": len(candidate),
            "by_model": dict(counts), "first_token_mismatches": mismatches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    print(json.dumps(compare(read_trace(args.reference), read_trace(args.candidate)), indent=2))


if __name__ == "__main__":
    main()
