"""Report full live-layout accuracy, E2E speed, and saved-input/output parity."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from summarize_saved_layout_accuracy import score


def load(path):
    return json.loads(path.read_text())


def trace_pages(run):
    pages = defaultdict(dict)
    with (run / "output/generation_trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["phase"] != "recognition":
                raise ValueError("live/saved comparison must contain recognition only")
            page, index = row["page"], row["block_index"]
            if index in pages[page]:
                raise ValueError("duplicate block trace")
            pages[page][index] = row
    return {p: [rows[i] for i in sorted(rows)] for p, rows in pages.items()}


def compare_saved(live, saved):
    a, b = trace_pages(saved), trace_pages(live)
    signature = lambda row: [row.get(k) for k in (
        "block_type", "bbox", "image_sha256", "chat_prompt", "prompt_token_ids")]
    mismatches = []
    fields = ("block_type", "bbox", "image_sha256", "chat_prompt", "prompt_token_ids")
    changed_fields = Counter()
    changed_generation = Counter()
    comparable_requests = 0
    input_exact = generated_exact = shared_requests = 0
    names = sorted(p.stem for p in (live / "output/predictions").glob("*.md"))
    saved_names = sorted(p.stem for p in (saved / "output/predictions").glob("*.md"))
    if names != saved_names:
        raise ValueError("prediction page membership differs")
    markdown_exact = sum((live / "output/predictions" / f"{name}.md").read_bytes() ==
                         (saved / "output/predictions" / f"{name}.md").read_bytes() for name in names)
    for name in sorted(a.keys() | b.keys()):
        old, new = a.get(name, []), b.get(name, [])
        if len(old) == len(new):
            comparable_requests += len(old)
            for x, y in zip(old, new):
                changed_fields.update(k for k in fields if x.get(k) != y.get(k))
                if x["generated_token_ids"] != y["generated_token_ids"]:
                    changed_generation["same_inputs" if signature(x) == signature(y) else "different_inputs"] += 1
        same_inputs = [signature(r) for r in old] == [signature(r) for r in new]
        if not same_inputs:
            mismatches.append({"page": name, "saved_crops": len(old), "live_crops": len(new)})
            continue
        input_exact += 1
        shared_requests += len(old)
        generated_exact += sum(x["generated_token_ids"] == y["generated_token_ids"] for x, y in zip(old, new))
    return {"prediction_pages": len(names), "markdown_exact_pages": markdown_exact,
            "recognition_pages": len(a.keys() | b.keys()), "input_exact_recognition_pages": input_exact,
            "input_mismatch_pages": mismatches, "requests_on_input_exact_pages": shared_requests,
            "generated_ids_exact_requests_on_input_exact_pages": generated_exact,
            "ordered_comparable_requests": comparable_requests,
            "changed_input_fields_by_request": dict(changed_fields),
            "changed_generation_requests": dict(changed_generation)}


def report(live, saved, native):
    runs = {name: score(path) for name, path in [("live", live), ("saved_layout", saved), ("native_historical", native)]}
    summaries = {name: load(path / "output/run_summary_shard_00.json") for name, path in
                 [("live", live), ("saved_layout", saved), ("native_historical", native)]}
    current = summaries["live"]
    n = current["completed"]
    if n != 1651 or current["failed"] or current["streaming"]["layout_calls"] != n:
        raise ValueError("expected all 1651 live layout pages")
    if any(row["completed"] != n or row["model_hashes"]["dataset_json"] != current["model_hashes"]["dataset_json"]
           for row in summaries.values()):
        raise ValueError("dataset mismatch between comparisons")
    progress = [json.loads(line) for line in (live / "output/progress_shard_00.jsonl").read_text().splitlines()]
    if len(progress) != n or len({r["image"] for r in progress}) != n:
        raise ValueError("invalid completion journal")
    times = [r["pipeline_elapsed_s"] for r in progress]
    performance = {}
    for name, row in summaries.items():
        performance[name] = {"pipeline_wall_s": row["pipeline_wall_s"],
                             "pages_per_second": n / row["pipeline_wall_s"],
                             "setup_s": row.get("setup_s"),
                             "physical_npu": row.get("ascend_rt_visible_devices"),
                             "warmup_pages": row["warmup"]["executed_pages"]}
    ratio = performance["live"]["pages_per_second"] / performance["native_historical"]["pages_per_second"]
    result = {"accuracy": {k: v["scores"] for k, v in runs.items()}, "performance": performance,
              "live_vs_historical_throughput_ratio": ratio,
              "live_vs_historical_throughput_gain_percent": (ratio - 1) * 100,
              "live_completion_10_to_last_pg_s": (n - 10) / (times[-1] - times[9]),
              "parity_with_saved_layout": compare_saved(live, saved),
              "layout": {k: v for k, v in current["streaming"].items() if k.startswith("layout_")},
              "request_phases": runs["live"]["request_phases"], "stop_reasons": runs["live"]["stop_reasons"],
              "evaluation_wall_s": runs["live"]["evaluation_wall_s"],
              "stage_execution": load(live / "evaluation/work/result/predictions_quick_match_stage_execution.json"),
              "caveat": "Native historical run predates the pixel cap and runtime changes; not a layout-only ablation. Saved-layout timing excludes detection/crop export."}
    result["accuracy_delta_live_minus_saved"] = {k: v - runs["saved_layout"]["scores"][k] for k, v in runs["live"]["scores"].items()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.live, args.saved, args.native)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
