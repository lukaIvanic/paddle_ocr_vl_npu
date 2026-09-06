"""Read-only serving evidence audit; generated report, never a routing input."""
from collections import Counter
import argparse
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--max-in-flight", type=int, default=2)
parser.add_argument("--worker-pid", type=int, default=1994482)
parser.add_argument("--target-qps", type=float, default=3.0)
parser.add_argument("--target-p95", type=float, default=2.0)
args = parser.parse_args()
ROOT = args.root.resolve()
SOURCE = Path(__file__).resolve().parent.parent / "table_b1_latency_full_04fbc8e/client/tables.jsonl"
source = list(map(json.loads, SOURCE.read_text().splitlines()))
assert len(source) == 665 and len({r["request_id"] for r in source}) == 665
monitor_path = ROOT / "host_npu6_monitor.log"
samples = []
if monitor_path.exists():
    for block in re.split(r"(?=^2026-\d\d-\d\dT)", monitor_path.read_text(), flags=re.M):
        if "Chip Count" in block:
            samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                            set(map(int, re.findall(r"Process id:(\d+)", block)))))


def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)


report = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "hardware": "one 910B2, physical NPU6", "runs": {},
          "required_batch_size":args.batch_size, "required_client_cap":args.max_in_flight,
          "owned_host_worker_pid":args.worker_pid,
          "targets":{"eos_completion_qps_at_least":args.target_qps,"p95_s_below":args.target_p95}}
development_config = None
for name, count, seed in (("client",100,1), ("seed2",100,2),
                          ("validation1000_a",1000,3), ("validation1000_b",1000,3)):
    directory = ROOT / name
    if not (directory / "summary.json").exists():
        report["runs"][name] = {"completed": False}
        continue
    summary = json.loads((directory / "summary.json").read_text())
    rows = list(map(json.loads, (directory / "results.jsonl").read_text().splitlines()))
    manifest = list(map(json.loads, (directory / "tables.jsonl").read_text().splitlines()))
    rng, expected = random.Random(seed), []
    while len(expected) < count:
        expected.extend(rng.sample(source, min(665, count-len(expected))))
    assert [r["request_id"] for r in manifest] == [r["request_id"] for r in expected]
    if seed == 1:
        assert hashlib.sha256((directory / "tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    if count == 1000:
        assert len({r["request_id"] for r in manifest[:665]}) == 665
        assert len({r["request_id"] for r in manifest[665:]}) == 335
    assert len(rows) == count and len({r["sequence"] for r in rows}) == count
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    for row in rows:
        assert row["status"] == "ok" and row["error"] is None
        assert row["service_result"]["http_status"] == 200
        assert row["latency_s"] > 0
        assert abs(row["latency_s"] - (row["completion_offset_s"]-row["dispatch_offset_s"])) < 1e-9
        response = row["service_result"]["response"]
        ids = response["token_ids"]
        assert len(ids) == response["generated_tokens_including_eos"]
        assert response["stop_reason"] in ("eos", "kv_cache_full")
        if response["stop_reason"] == "eos":
            assert ids[-1] == 2
        else:
            assert ids[-1] != 2 and response["input_tokens"] + len(ids) - 1 == 4096
    ordered = sorted(rows, key=lambda r:r["sequence"])
    assert [r["request_id"] for r in ordered] == [r["request_id"] for r in manifest]
    active = peak = 0
    for _, change in sorted([(r["dispatch_offset_s"],1) for r in rows]
                            + [(r["completion_offset_s"],-1) for r in rows]):
        active += change
        peak = max(peak, active)
        assert 0 <= active <= args.max_in_flight
    assert active == 0 and peak == args.max_in_flight
    config = summary["api_configuration"]
    if development_config is None:
        development_config = config
    assert config == development_config
    assert config["batch_size"] == args.batch_size and config["cache_length"] == config["max_new_tokens"] == 4096
    assert config["decode_optimization"] == "combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp"
    assert config["token_selection"]["mode"] == "greedy"
    assert config["token_selection"]["rule"] == "ordinary_argmax" and config["setup_gc"]["gc_remains_enabled"]
    assert config["setup_gc"]["enabled"] and not config["decode_device_timing"]
    responses = [r["service_result"]["response"] for r in rows]
    stops = Counter(r["stop_reason"] for r in responses)
    p95 = percentile([r["latency_s"] for r in rows], .95)
    assert abs(p95-summary["latency_s"]["p95"]) < 1e-9
    qps = count/summary["run_wall_s"]
    assert abs(qps-summary["successful_completion_qps"]) < 1e-9
    begin, end = summary["actual_start_epoch_s"], summary["actual_start_epoch_s"]+summary["run_wall_s"]
    during = [pids for stamp,pids in samples if begin <= stamp <= end]
    ownership_ok = bool(during and samples[0][0] < begin and samples[-1][0] > end
                        and all(pids == {args.worker_pid} for pids in during))
    unexpected = [{"epoch_s":stamp,"pids":sorted(pids)} for stamp,pids in samples
                  if begin <= stamp <= end and pids != {args.worker_pid}]
    targets_met = stops["eos"]/summary["run_wall_s"] >= args.target_qps and p95 < args.target_p95
    # Report EOS throughput separately: capped outputs do not become hidden
    # fast successes. Their actual latency remains in the full distribution.
    report["runs"][name] = {
        "completed": True, "count": count, "seed": seed, "peak_outstanding": peak,
        "manifest_sha256": hashlib.sha256((directory/"tables.jsonl").read_bytes()).hexdigest(),
        "request_completion_qps": qps, "eos_completion_qps": stops["eos"]/summary["run_wall_s"],
        "p95_s": p95, "stop_reasons": dict(stops), "errors": 0,
        "ownership_samples": len(during), "sampled_ownership_ok": ownership_ok,
        "numerical_targets_met": targets_met,
        "qualifying_timing": ownership_ok and targets_met,
        "unexpected_ownership_snapshots": unexpected,
        "non_eos_requests": [{"sequence":r["sequence"], "request_id":r["request_id"],
                              "stop_reason":r["service_result"]["response"]["stop_reason"],
                              "latency_s":r["latency_s"]} for r in rows
                             if r["service_result"]["response"]["stop_reason"] != "eos"],
    }
def compare_outputs(left, right):
    if not (left / "results.jsonl").exists() or not (right / "summary.json").exists():
        return {"available": False}
    a = sorted(map(json.loads, (left / "results.jsonl").read_text().splitlines()), key=lambda r:r["sequence"])
    b = sorted(map(json.loads, (right / "results.jsonl").read_text().splitlines()), key=lambda r:r["sequence"])
    assert [r["request_id"] for r in a] == [r["request_id"] for r in b]
    differences, input_differences, stop_differences = [], [], []
    for x, y in zip(a, b):
        old, new = x["service_result"]["response"], y["service_result"]["response"]
        for key in ("input_tokens", "projected_image_tokens", "crop_size"):
            if old[key] != new[key]:
                input_differences.append([x["sequence"], x["request_id"], key, old[key], new[key]])
        if old["vision"]["real_vision_tokens"] != new["vision"]["real_vision_tokens"]:
            input_differences.append([x["sequence"], x["request_id"], "real_vision_tokens"])
        if old["stop_reason"] != new["stop_reason"]:
            stop_differences.append([x["sequence"], x["request_id"], old["stop_reason"], new["stop_reason"]])
        if old["token_ids"] != new["token_ids"]:
            edits = []
            before, after = old.get("raw_text",old["text"]), new.get("raw_text",new["text"])
            for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
                if tag != "equal":
                    edits.append({"before":before[max(0,i-45):min(len(before),j+45)],
                                  "after":after[max(0,k-45):min(len(after),l+45)],
                                  "removed":before[i:j], "added":after[k:l]})
            differences.append({"sequence":x["sequence"], "request_id":x["request_id"],
                                "old_tokens":len(old["token_ids"]), "new_tokens":len(new["token_ids"]),
                                "rendered_html_equal":old["text"] == new["text"], "edits":edits})
    return {"available":True, "count":len(a), "native_ids_identical":len(a)-len(differences),
            "input_differences":input_differences, "stop_differences":stop_differences,
            "differences":differences}

comparisons = {
    "historical_b2_to_first_validation": compare_outputs(
        ROOT.parent / "table_1000_matrix_02fe5645_20260905/b2/measured", ROOT / "validation1000_a"),
    "first_to_second_validation": compare_outputs(ROOT / "validation1000_a", ROOT / "validation1000_b"),
}
if ROOT != Path(__file__).resolve().parent:
    comparisons["optimized_b2_to_first_validation"] = compare_outputs(
        Path(__file__).resolve().parent / "validation1000_a", ROOT / "validation1000_a")
(ROOT / "output_comparison.json").write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+"\n")
report["output_comparison_summary"] = {name:{k:v for k,v in comparison.items() if k != "differences"}
                                       for name,comparison in comparisons.items()}
(ROOT / "audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
