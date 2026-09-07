"""Separate measured selected-crop performance from estimated whole-pipeline speed."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from run_crop_cap_replay import traces
from summarize_saved_layout_accuracy import score
from prepare_crop_cap_replay import dump


def load(path):
    return json.loads(path.read_text())


def report(root, reference_run):
    if (root / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("paired experiment incomplete")
    selection = load(root / "frozen/selection.json")
    runs = {"reference": reference_run, "control": root / "control", "capped": root / "capped"}
    results = {name: score(path) for name, path in runs.items()}
    summaries = {name: load(path / "output/run_summary_shard_00.json") for name, path in runs.items()}
    raw = {name: traces(path / "output") for name, path in runs.items()}
    selected_ids = {r["request_id"] for r in selection["crops"]}
    if set(raw["control"]) != selected_ids or set(raw["capped"]) != selected_ids:
        raise ValueError("selected crop membership mismatch")
    by_type = defaultdict(lambda: Counter())
    image_id = load(Path(summaries["reference"]["model"]) / "config.json")["image_token_id"]
    for key in selected_ids:
        a, b = raw["control"][key], raw["capped"][key]
        c = by_type[a["block_type"]]
        c["crops"] += 1
        c["changed_generated_ids"] += a["generated_token_ids"] != b["generated_token_ids"]
        for label, row in [("control", a), ("capped", b)]:
            c[label + "_vision_tokens"] += row["prompt_token_ids"].count(image_id)*4
            c[label + "_generated_tokens_including_eos"] += len(row["generated_token_ids"])
            c[label + "_length_capped_crops"] += row["stop_reason"] == "length"
    performance = {}
    for label in ("control", "capped"):
        s = summaries[label]
        d = s["streaming"]["decode"]
        p = d["prefill_metrics"]
        performance[label] = {
            "timing_scope": "selected crop recognition plus full-page reconstruction, no layout inference",
            "selected_crops": len(selected_ids), "wall_s": s["pipeline_wall_s"],
            "selected_crops_per_s": len(selected_ids)/s["pipeline_wall_s"], "setup_s": s["setup_s"],
            "physical_npu": s["ascend_rt_visible_devices"],
            "processor_max_pixels": s["processor_max_pixels"],
            "vision_transformer_device_s": p["vision_transformer_blocks"],
            "text_prefill_device_s": p["text_transformer_prefill"], "decode_device_s": d["decode_s"],
            "vision_tokens": p["raw_vision_tokens"], "text_prefill_tokens": p["text_prefill_tokens"],
            "effective_decode_tokens": d["decode_calls"], "active_decode_slot_fraction": d["active_slot_fraction"],
            "decode_first_call_s": d["compiled_first_call_s"],
        }
    if performance["control"]["physical_npu"] != performance["capped"]["physical_npu"]:
        raise ValueError("paired runs must use same physical NPU")
    saved_s = performance["control"]["wall_s"] - performance["capped"]["wall_s"]
    estimate_wall = summaries["reference"]["pipeline_wall_s"] - saved_s
    page_changes = {}
    for metric in ("text_accuracy", "table_teds", "formula_cdm"):
        a, b = results["control"]["per_page"][metric], results["capped"]["per_page"][metric]
        differences = sorted([{"page": p, "delta": b[p]-a[p], "control": a[p], "capped": b[p]} for p in a.keys() & b.keys()], key=lambda r:r["delta"])
        page_changes[metric] = {"common_scored_pages": len(differences),
            "control_only_pages": sorted(a.keys()-b.keys()), "capped_only_pages": sorted(b.keys()-a.keys()),
            "improved": sum(r["delta"]>1e-6 for r in differences),
            "regressed": sum(r["delta"] < -1e-6 for r in differences),
            "unchanged": sum(abs(r["delta"])<=1e-6 for r in differences),
            "worst": differences[:10], "best": differences[-10:][::-1]}
    parity = {}
    for label in ("control", "capped"):
        original_files = list((reference_run / "output/predictions").glob("*.md"))
        parity[label] = {"markdown_exact_vs_reference": sum(
            file.read_bytes() == (runs[label]/"output/predictions"/file.name).read_bytes() for file in original_files),
            "selected_generated_ids_exact_vs_reference": sum(
                row["generated_token_ids"] == raw["reference"][key]["generated_token_ids"] for key,row in raw[label].items())}
    return {"selection": {k:v for k,v in selection.items() if k != "crops"},
        "accuracy": {k:v["scores"] for k,v in results.items()},
        "accuracy_delta_capped_minus_control": {k: v-results["control"]["scores"][k] for k,v in results["capped"]["scores"].items()},
        "accuracy_delta_control_minus_reference": {k: v-results["reference"]["scores"][k] for k,v in results["control"]["scores"].items()},
        "performance": performance, "measured_selected_replay_speedup": performance["control"]["wall_s"]/performance["capped"]["wall_s"],
        "measured_selected_replay_saved_s": saved_s,
        "whole_pipeline_estimate_NOT_MEASURED": {"reference_wall_s": summaries["reference"]["pipeline_wall_s"],
            "estimated_wall_s": estimate_wall, "estimated_pg_s": 1651/estimate_wall,
            "method": "reference full live wall minus paired selected-replay wall savings",
            "limitations": "Different batching, overlap and device/run variability; full E2E rerun needed to validate."},
        "by_crop_type": dict(by_type), "page_changes": page_changes, "parity": parity,
        "stage_execution": {k:load(path/"evaluation/work/result/predictions_quick_match_stage_execution.json") for k,path in runs.items()},
        "evaluation_wall_s": {k:v["evaluation_wall_s"] for k,v in results.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.reference_run)
    dump(args.root / "comparison.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("page_changes", "stage_execution")}, indent=2), flush=True)
