"""Summarize unchanged full-corpus predictions and optional native-layout control."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def score(run):
    if (run / "exit_code.txt").read_text().strip() != "0" or (run / "evaluation/exit_code.txt").read_text().strip() != "0":
        raise ValueError(f"incomplete inference/evaluation: {run}")
    summary = load(run / "output/run_summary_shard_00.json")
    result = run / "evaluation/work/result"
    prefix = "predictions_quick_match_"
    raw = load(result / f"{prefix}metric_result.json")
    scores = {
        "text_accuracy": 100 * (1 - raw["text_block"]["page"]["Edit_dist"]["ALL"]),
        "table_teds": 100 * raw["table"]["page"]["TEDS"]["ALL"],
        "formula_cdm": 100 * raw["display_formula"]["page"]["CDM"]["ALL"],
    }
    scores["overall"] = mean(scores.values())
    scores["table_structure_teds"] = 100 * raw["table"]["page"]["TEDS_structure_only"]["ALL"]
    scores["reading_order_edit_distance"] = raw["reading_order"]["page"]["Edit_dist"]["ALL"]
    pages = {"text_accuracy": {p: 100 * (1 - v) for p, v in load(result / f"{prefix}text_block_per_page_edit.json").items()}}
    for metric, filename, field in [("table_teds", "table_per_table_TEDS.json", "TEDS"),
                                     ("formula_cdm", "display_formula_per_sample_CDM.json", None)]:
        grouped = defaultdict(list)
        for key, value in load(result / f"{prefix}{filename}").items():
            grouped[key.rsplit("_[", 1)[0]].append(value if field is None else value[field])
        pages[metric] = {p: 100 * mean(values) for p, values in grouped.items()}
    trace_path = run / "output/generation_trace.jsonl"
    phases, stops, limited = Counter(), Counter(), []
    if trace_path.exists():
        with trace_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                phases[row["phase"]] += 1
                stops[row["stop_reason"]] += 1
                if row["stop_reason"] == "length":
                    limited.append({k: row.get(k) for k in ("request_id", "page", "block_type")})
    if summary.get("saved_layout_manifest") and phases.get("layout", 0):
        raise ValueError("hybrid run unexpectedly generated layout")
    return {"run": str(run), "scores": scores,
            "completed": summary["completed"], "failed": summary["failed"],
            "settings": {k: summary.get(k) for k in (
                "git_commit", "model_hashes", "processor_max_pixels", "processor_min_pixels",
                "local_decode_increfa_length_mode", "local_compiled_cache_length", "image_analysis",
                "saved_layout_manifest_sha256", "warmup")},
            "request_phases": dict(phases), "stop_reasons": dict(stops),
            "length_limited_requests": limited,
            "pipeline_wall_s": summary.get("pipeline_wall_s"),
            "evaluation_wall_s": float((run / "evaluation/wall_s.txt").read_text()),
            "evaluation_diagnostics": {k: raw.get(k) for k in ("match_debug",)},
            "per_page": pages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"hybrid": score(args.run)}
    if args.baseline:
        report["baseline"] = score(args.baseline)
        a, b = report["baseline"], report["hybrid"]
        report["score_delta_hybrid_minus_baseline"] = {k: b["scores"][k] - v for k, v in a["scores"].items()}
        report["setting_differences"] = {k: {"baseline": a["settings"].get(k), "hybrid": v}
                                          for k, v in b["settings"].items() if a["settings"].get(k) != v}
        report["page_changes"] = {}
        for metric, new in b["per_page"].items():
            old = a["per_page"][metric]
            changes = sorted([{"page": p, "baseline": old[p], "hybrid": new[p], "delta": new[p] - old[p]}
                              for p in old.keys() & new.keys()], key=lambda r: r["delta"])
            report["page_changes"][metric] = {
                "improved": sum(r["delta"] > 1e-6 for r in changes),
                "regressed": sum(r["delta"] < -1e-6 for r in changes),
                "unchanged": sum(abs(r["delta"]) <= 1e-6 for r in changes),
                "baseline_only": sorted(old.keys() - new.keys()), "hybrid_only": sorted(new.keys() - old.keys()),
                "worst": changes[:10], "best": changes[-10:][::-1], "all": changes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for label in ("hybrid", "baseline"):
        if label in report:
            row = report[label]
            print(label, json.dumps({k: row[k] for k in ("completed", "failed", "scores", "request_phases", "stop_reasons")}))
    print("DELTA", json.dumps(report.get("score_delta_hybrid_minus_baseline")))
    print("REPORT", args.output)


if __name__ == "__main__":
    main()
