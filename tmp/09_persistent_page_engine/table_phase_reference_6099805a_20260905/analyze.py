"""Read actual HTTP results; no simulated duration or generated-text encoding."""
from collections import Counter
import hashlib
from datetime import datetime
import json
from pathlib import Path
import re

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
        "own_preparation_prefill_s": sum(v for k, v in own.items() if k.endswith("prefill") or k in ("cpu_prepare", "cpu_prepare_wait")),
        "own_matcher_control_postprocess_s": sum(v for k, v in own.items() if k.startswith("matcher") or k == "output_postprocess"),
        "foreign_preparation_prefill_s": sum(v for k, v in other.items() if k.endswith("prefill") or k in ("cpu_prepare", "cpu_prepare_wait")),
        "foreign_other_work_s": sum(v for k, v in other.items() if not (k.endswith("prefill") or k in ("cpu_prepare", "cpu_prepare_wait"))),
        "unaccounted_in_client_s": row["latency_s"] - sum(own.values()) - sum(other.values()),
        "phase_exposure_s": accounting["decode_phase_combination_wall_s"],
    }
    cpu = response(row).get("runtime_metrics", {}).get("cpu_preparation")
    if cpu:
        values["cpu_background_wall_s_nonadditive"] = cpu["finished"] - cpu["started"]
        values["cpu_background_thread_s_nonadditive"] = cpu["thread_s"]
    return values


output = {"hardware": "one Ascend 910B2 physical NPU 6", "contract": "100 distinct all-corpus tables, seed 1; client HTTP latency; closed-loop completion QPS"}
folders = {name: ROOT / name for name in (
    "c1", "control", "fixed_c1", "fixed_c2", "original_fixed", "cached_c1",
    "overlap_c1", "pipeline_c1", "pinned_c1", "pinned_c1_repeat", "identity_c1",
    "original_recheck", "identity_c2", "retained_c1", "retained_c2",
    "async_c1", "async_c2",
) if (ROOT / name / "summary.json").exists()}
folders["historical"] = HISTORICAL
output["distributions"] = {name: summarize(path) for name, path in folders.items()}
records = {name: read(path / "results.jsonl") for name, path in folders.items()}
historical = {row["request_id"]: row for row in records["historical"]}
output["selection_sha256"] = {
    name: hashlib.sha256((path / "tables.jsonl").read_bytes()).hexdigest()
    for name, path in folders.items()
}
output["request_outcomes"] = {
    name: {
        "rows": len(rows), "over_2s": sum(row["latency_s"] > 2 for row in rows),
        "stop_reasons": dict(Counter(response(row).get("stop_reason") for row in rows)),
        "dispatch_order_matches": [row["request_id"] for row in sorted(rows, key=lambda r: r["sequence"])]
        == [row["request_id"] for row in sorted(records["historical"], key=lambda r: r["sequence"])],
    } for name, rows in records.items()
}
output["interval_peak_outstanding"] = {}
for name, rows in records.items():
    events = [event for row in rows for event in (
        (row["dispatch_offset_s"], 1), (row["completion_offset_s"], -1),
    )]
    active = peak = 0
    for _when, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    output["interval_peak_outstanding"][name] = {"peak": peak, "remaining": active}
output["reference_phase_totals_s"] = {}
for name, rows in records.items():
    totals = Counter()
    for row in rows:
        accounting = response(row).get("runtime_metrics", {}).get("phase_accounting")
        if accounting:
            totals.update(accounting["own_action_wall_s"])
    if totals:
        output["reference_phase_totals_s"][name] = dict(totals)
output["native_id_parity_vs_historical"] = {
    name: {"identical": sum(response(row)["token_ids"] == response(historical[row["request_id"]])["token_ids"] for row in rows), "total": len(rows)}
    for name, rows in records.items() if name != "historical"
}
output["c2_comparisons"] = {}
for name, control in (("identity_c2", "identity_c1"), ("retained_c2", "retained_c1"), ("async_c2", "async_c1")):
    if name not in records or control not in records:
        continue
    base = {row["request_id"]: row for row in records[control]}
    rows = records[name]
    global_actions, global_calls = Counter(), Counter()
    phases = Counter()
    pairs = []
    for row in rows:
        accounting = response(row)["runtime_metrics"]["phase_accounting"]
        for action, duration in accounting["own_action_wall_s"].items():
            if action == "matcher_propose_control":
                # Shared host control may have one or two owners in a run.
                # Its exact global total belongs to the service ledger, not
                # a reconstruction from per-request aggregated counters.
                continue
            # B16 drafts and B2 targets have two request owners; count once.
            owners = 2 if action.startswith("draft_b16") or action.startswith(("verify_b2", "ordinary_b2")) else 1
            global_actions[action] += duration / owners
            global_calls[action] += accounting["own_calls"][action] / owners
        for phase, duration in accounting["decode_phase_combination_wall_s"].items():
            phases[phase] += duration / len(phase.split("+"))
        pairs.append({
            **profile(row), "c1_latency_s": base[row["request_id"]]["latency_s"],
            "delta_s": row["latency_s"] - base[row["request_id"]]["latency_s"],
            "native_ids_match_c1": response(row)["token_ids"] == response(base[row["request_id"]])["token_ids"],
        })
    output["c2_comparisons"][name] = {
        "control": control,
        "native_ids_match_c1": sum(pair["native_ids_match_c1"] for pair in pairs),
        "faster": sum(pair["delta_s"] < 0 for pair in pairs),
        "slower": sum(pair["delta_s"] > 0 for pair in pairs),
        "global_actions_s": dict(global_actions), "global_calls": dict(global_calls),
        "global_decode_phase_exposure_s": dict(phases),
        "p95_tail": sorted([pair for pair in pairs if pair["latency_s"] >= output["distributions"][name]["p95"]], key=lambda p: -p["latency_s"]),
        "matched_tables": pairs,
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
expected_hash = "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
audit = {"frozen_selection_hash": expected_hash, "checks": {}, "host_checks": {}}
for name in ("identity_c1", "identity_c2", "retained_c1", "retained_c2", "async_c1", "async_c2", "original_recheck"):
    if name not in records:
        continue
    rows = records[name]
    summary = json.loads((folders[name] / "summary.json").read_text())
    audit["checks"][name] = {
        "identical_selection": output["selection_sha256"][name] == expected_hash,
        "complete_100_no_errors": len(rows) == 100 and all(row["status"] == "ok" for row in rows),
        "all_eos": all(response(row)["stop_reason"] == "eos" for row in rows),
        "native_parity_100": output["native_id_parity_vs_historical"][name]["identical"] == 100,
        "dispatch_order_matches": output["request_outcomes"][name]["dispatch_order_matches"],
        "interval_cap_matches_client": output["interval_peak_outstanding"][name]["peak"] == summary["max_in_flight"],
    }
    if name != "original_recheck":
        original = {row["request_id"]: response(row) for row in records["original_fixed"]}
        audit["checks"][name]["original_splits_and_rotations"] = all(
            response(row)["route_lane"] == "b1" or (
                response(row)["runtime_metrics"]["row_preparation"]["boundaries"]
                == original[row["request_id"]]["stage_timing_s"]["boundaries"]
                and response(row)["runtime_metrics"]["row_preparation"]["row_draft_rotation_cw"]
                == original[row["request_id"]]["stage_timing_s"]["row_draft_rotation_cw"]
            ) for row in rows
        )
for server, pid, names in (
    ("identity", "1553489", ("identity_c1", "identity_c2")),
    ("retained", "1583178", ("retained_c1", "retained_c2")),
    ("async", "1613266", ("async_c1", "async_c2")),
    ("original_recheck", "1643932", ("original_recheck",)),
):
    host_log = ROOT / "host_logs" / f"table_phase_{server}_npu6_20260905.log"
    if not host_log.exists():
        continue
    text = host_log.read_text()
    timestamps = [datetime.fromisoformat(line).timestamp() for line in text.splitlines() if line.startswith("2026-")]
    checks = {"only_expected_pid_in_samples": set(re.findall(r"Process id:(\d+)", text)) == {pid}}
    for name in names:
        summary = json.loads((ROOT / name / "summary.json").read_text())
        checks[name + "_window_bracketed"] = min(timestamps) <= summary["actual_start_epoch_s"] and max(timestamps) >= summary["actual_start_epoch_s"] + summary["run_wall_s"]
    server_log = (ROOT / f"{server}_server.log").read_text()
    checks["no_cache_rejection_or_traceback_log"] = not re.search(r"Skip.*cache|Recompil|Traceback", server_log)
    audit["host_checks"][server] = {"expected_host_worker_pid": pid, "samples": len(timestamps), "checks": checks}
server_log = (ROOT / "async_server.log").read_text()
audit["whole_pipeline_admission"] = {}
for label, cap in (("async-c1", 1), ("async-c2", 2)):
    counts = [int(n) for n in re.findall(r"TABLE_PHASE preparing id=" + label + r"-\d+.*?admitted=(\d+)", server_log)]
    audit["whole_pipeline_admission"][label] = {
        "records": len(counts), "peak": max(counts), "passes": len(counts) == 100 and max(counts) == cap,
    }
audit["all_data_checks_pass"] = (
    all(all(checks.values()) for checks in audit["checks"].values())
    and all(all(item["checks"].values()) for item in audit["host_checks"].values())
    and all(item["passes"] for item in audit["whole_pipeline_admission"].values())
)
audit["limitations"] = [
    "Host sampling is approximately every five seconds, not continuous process tracing.",
    "Host action attribution is not an isolated kernel profile; pending work and background CPU overlap.",
    "Final NPU release is separately documented from the direct-host terminal check.",
]
(ROOT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
(ROOT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
