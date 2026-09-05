"""Read actual HTTP results; no simulated duration or generated-text encoding."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
HISTORICAL = REPO / "tmp/09_persistent_page_engine/table_spec_closed_loop_random100_c1_manual_postrope_90b3b3c0_20260904/measured"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(folder):
    summary = json.loads((folder / "summary.json").read_text())
    return {"completion_qps": summary["completion_qps"], **summary["latency_s"]}


def response(row):
    return row["service_result"]["response"]


def profile(row):
    accounting = response(row)["runtime_metrics"]["phase_accounting"]
    own = accounting["own_action_wall_s"]
    other = accounting["other_action_wait_s"]
    values = {
        "request_id": row["request_id"], "latency_s": row["latency_s"],
        "route": response(row)["route_lane"],
        "own_draft_decode_s": sum(v for k, v in own.items() if k.startswith("draft_b")),
        "own_verifier_s": sum(v for k, v in own.items() if k.startswith("verify_b")),
        "own_ordinary_decode_s": sum(v for k, v in own.items() if k.startswith("ordinary_b")),
        "own_preparation_prefill_s": sum(v for k, v in own.items() if k.endswith("prefill") or k == "cpu_prepare"),
        "own_matcher_control_postprocess_s": sum(v for k, v in own.items() if k.startswith("matcher") or k == "output_postprocess"),
        "foreign_preparation_prefill_s": sum(v for k, v in other.items() if k.endswith("prefill") or k == "cpu_prepare"),
        "foreign_other_work_s": sum(v for k, v in other.items() if not (k.endswith("prefill") or k == "cpu_prepare")),
        "unaccounted_in_client_s": row["latency_s"] - sum(own.values()) - sum(other.values()),
        "phase_exposure_s": accounting["decode_phase_combination_wall_s"],
    }
    return values


output = {"hardware": "one Ascend 910B2 physical NPU 6", "contract": "100 distinct all-corpus tables, seed 1; client HTTP latency; closed-loop completion QPS"}
folders = {name: ROOT / name for name in ("c1", "control", "fixed_c1", "fixed_c2", "original_fixed") if (ROOT / name / "summary.json").exists()}
folders["historical"] = HISTORICAL
output["distributions"] = {name: summarize(path) for name, path in folders.items()}
records = {name: read(path / "results.jsonl") for name, path in folders.items()}
historical = {row["request_id"]: row for row in records["historical"]}
output["native_id_parity_vs_historical"] = {
    name: {"identical": sum(response(row)["token_ids"] == response(historical[row["request_id"]])["token_ids"] for row in rows), "total": len(rows)}
    for name, rows in records.items() if name != "historical"
}
if "fixed_c2" in records:
    rows = records["fixed_c2"]
    threshold = output["distributions"]["fixed_c2"]["p95"]
    output["c2_tail"] = [profile(row) for row in sorted(rows, key=lambda r: r["latency_s"], reverse=True) if row["latency_s"] >= threshold]
    phases = Counter()
    for row in rows:
        for name, value in profile(row)["phase_exposure_s"].items():
            # A shared interval appears in each live request's exposure.
            phases[name] += value / len(name.split("+"))
    total = sum(phases.values())
    output["c2_global_decode_phase_wall_s"] = dict(phases)
    output["c2_global_decode_phase_fraction"] = {key: value / total for key, value in phases.items()}
    if "fixed_c1" in records:
        base = {row["request_id"]: row for row in records["fixed_c1"]}
        output["c2_vs_c1"] = {
            "identical_native_outputs": sum(response(row)["token_ids"] == response(base[row["request_id"]])["token_ids"] for row in rows),
            "faster": sum(row["latency_s"] < base[row["request_id"]]["latency_s"] for row in rows),
            "slower": sum(row["latency_s"] > base[row["request_id"]]["latency_s"] for row in rows),
        }
        for row in output["c2_tail"]:
            row["matched_c1_latency_s"] = base[row["request_id"]]["latency_s"]
(ROOT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
