"""Compare stable request identities, exact IDs, frontend inputs and page outputs.

Differences are classified, not automatically forgiven as numerical variation.
Run on the Mac or validation host; no model packages are imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import re
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


def canonical_table_placeholders(text):
    names = {}
    def replace(match):
        token = match.group(0)
        if token not in names:
            names[token] = len(names)
        return f"[TABLE_IMAGE_{names[token]}]"
    normalized = re.sub(r"\[[ACDGHKTWXYZ2345678]{4}\]", replace, text)
    return normalized, len(names)


def table_placeholder_equivalent(a, b, reference, candidate):
    if a.get("block_type") != "table" or a["stop_reason"] != "eos" or b["stop_reason"] != "eos":
        return False
    left, left_count = canonical_table_placeholders(a["raw_text"])
    right, right_count = canonical_table_placeholders(b["raw_text"])
    if not left_count or left_count != right_count or left != right:
        return False
    stem = Path(a["page"]).stem
    for directory, suffix in (("predictions", ".md"), ("content_lists", ".json")):
        first, second = reference / directory / (stem + suffix), candidate / directory / (stem + suffix)
        if not first.is_file() or not second.is_file() or first.read_bytes() != second.read_bytes():
            return False
    return True


def compare(reference: Path, candidate: Path, first_pages: int | None = None,
            allow_table_image_placeholders: bool = False):
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
    placeholder_equivalences = []
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
        placeholder_change = (allow_table_image_placeholders and changed_inputs == ["image_sha256"]
                              and table_placeholder_equivalent(a, b, reference, candidate))
        if placeholder_change:
            placeholder_equivalences.append(key)
        if changed_inputs and not placeholder_change and (phase == "layout" or a["page"] in unchanged_layout_pages):
            unexpected_input_changes.append(key)
        same_tokens = a["generated_token_ids"] == b["generated_token_ids"]
        totals[f"{phase}_matched_requests"] += 1
        totals[f"{phase}_input_exact"] += not changed_inputs
        totals[f"{phase}_token_exact"] += same_tokens
        totals[f"{phase}_text_exact"] += a["raw_text"] == b["raw_text"]
        totals[f"{phase}_length_stops"] += b["stop_reason"] == "length"
        if changed_inputs or not same_tokens:
            totals[f"changed_{a.get('block_type', phase)}_requests"] += 1
            first = next((i for i, (x, y) in enumerate(zip(a["generated_token_ids"], b["generated_token_ids"])) if x != y),
                         min(len(a["generated_token_ids"]), len(b["generated_token_ids"])))
            differences.append({"request_id": key, "changed_input_fields": changed_inputs,
                                "classification": "upstream_table_image_placeholders" if placeholder_change else "unclassified_difference",
                                "first_token_difference": first if not same_tokens else None,
                                "reference_tokens": a["generated_token_ids"][max(0, first-3):first+5],
                                "candidate_tokens": b["generated_token_ids"][max(0, first-3):first+5],
                                "reference_length": len(a["generated_token_ids"]),
                                "candidate_length": len(b["generated_token_ids"]),
                                "block_type": a.get("block_type", phase),
                                "raw_text_similarity": SequenceMatcher(None, a["raw_text"], b["raw_text"], autojunk=False).ratio(),
                                "reference_stop": a["stop_reason"], "candidate_stop": b["stop_reason"]})
    a_pages = {path.name: path for path in (reference / "predictions").glob("*.md")}
    if selected_pages is not None:
        selected_markdown = {Path(page).stem + ".md" for page in selected_pages}
        a_pages = {name: path for name, path in a_pages.items() if name in selected_markdown}
    b_pages = {path.name: path for path in (candidate / "predictions").glob("*.md")}
    pages_changed = [name for name in sorted(a_pages.keys() & b_pages.keys()) if a_pages[name].read_bytes() != b_pages[name].read_bytes()]
    accounting = None
    summaries = list(candidate.glob("run_summary_shard_*.json"))
    if len(summaries) == 1:
        summary = json.loads(summaries[0].read_text())
        metrics = summary["local_compiled_generation"]
        accounting = {
            "request_count_matches": summary["generation_trace"]["requests"] == len(right),
            "effective_token_count_matches": metrics["decode_calls"] == sum(len(x["generated_token_ids"]) - 1 for x in right.values()),
            "completed_page_count_matches": summary["completed"] == len(b_pages),
            "no_failed_pages": summary["failed"] == 0,
        }
    return {"reference": str(reference), "candidate": str(candidate),
            "reference_trace_sha256": hashlib.sha256((reference / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "candidate_trace_sha256": hashlib.sha256((candidate / "generation_trace.jsonl").read_bytes()).hexdigest(),
            "reference_requests": len(left), "candidate_requests": len(right),
            "reference_generated_tokens": sum(len(x["generated_token_ids"]) for x in left.values()),
            "candidate_generated_tokens": sum(len(x["generated_token_ids"]) for x in right.values()),
            "missing_requests": sorted(left.keys() - right.keys()),
            "extra_requests": sorted(right.keys() - left.keys()),
            "missing_requests_with_unchanged_layout": sorted(key for key in left.keys() - right.keys()
                if left[key]["page"] in unchanged_layout_pages),
            "extra_requests_with_unchanged_layout": sorted(key for key in right.keys() - left.keys()
                if right[key]["page"] in unchanged_layout_pages),
            "unexpected_input_changes": unexpected_input_changes,
            "table_placeholder_equivalences": placeholder_equivalences,
            "reference_length_stops": sum(x["stop_reason"] == "length" for x in left.values()),
            "candidate_length_stops": sum(x["stop_reason"] == "length" for x in right.values()),
            "new_length_stops": sorted(key for key, row in right.items() if row["stop_reason"] == "length"
                and (key not in left or left[key]["stop_reason"] != "length")),
            "candidate_trace_accounting": accounting,
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
    parser.add_argument("--allow-table-image-placeholders", action="store_true",
                        help="Allow only random table-label renaming with byte-identical final Markdown AND block JSON.")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.first_pages, args.allow_table_image_placeholders)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("differences", "changed_pages")}, indent=2))
    print(f"changed_pages={len(result['changed_pages'])} differing_requests={len(result['differences'])}")
    if any(result[key] for key in ("missing_pages", "extra_pages", "empty_pages",
           "missing_requests_with_unchanged_layout", "extra_requests_with_unchanged_layout",
           "unexpected_input_changes", "new_length_stops")) or (
            result["candidate_trace_accounting"] is not None
            and not all(result["candidate_trace_accounting"].values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
