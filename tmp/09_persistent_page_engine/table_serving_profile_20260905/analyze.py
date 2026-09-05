"""Separate complete real decode graphs from the async capture boundary."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "capture_1e32b233"
inputs = ROOT / "analysis_input"
capture = json.loads((ROOT / "capture.json").read_text())
packed_mlp = capture["configuration"]["decode_optimization"].endswith("_packed_mlp")
assert capture["warmups"] == 5 and capture["repeats"] == 20
assert len(capture["observations"]) == 25
assert all(x["active_slots"] == 2 for x in capture["observations"])
rows = sorted(csv.DictReader((inputs / "kernel_details.csv").open()),
              key=lambda row: float(row["Start Time(us)"]))
# Profiling starts while a preceding asynchronously submitted graph is still
# finishing. Counting every row / 20 would overstate cost (375 vs 360 IncreFA).
# UpdateModelParam_static_bin starts each of the 20 COMPLETE graph executions.
starts = [i for i, row in enumerate(rows) if row["Name"] == "UpdateModelParam_static_bin"]
assert len(starts) == 20
blocks = [rows[a:b] for a, b in zip(starts, starts[1:] + [len(rows)])]
types, counts = Counter(), Counter()
model_spans, model_sums, controls = [], [], []
for block in blocks:
    model_id = block[0]["Model ID"]
    model = [row for row in block if row["Model ID"] == model_id]
    count = Counter(row["Type"] for row in model)
    assert count["IncreFlashAttention"] == 18
    assert count["MatMul"] == (73 if packed_mlp else 91) and count["ArgMaxV2"] == 1
    start = float(model[0]["Start Time(us)"])
    end = max(float(row["Start Time(us)"]) + float(row["Duration(us)"]) for row in model)
    model_spans.append(end - start)
    model_sums.append(sum(float(row["Duration(us)"]) for row in model))
    controls.append(sum(float(row["Duration(us)"]) for row in block if row["Model ID"] != model_id))
    counts.update(count)
    for row in model:
        types[row["Type"]] += float(row["Duration(us)"])
trace_path = inputs / "trace_view.json"
trace = json.loads(trace_path.read_text() if trace_path.exists()
                   else gzip.decompress((inputs / "trace_view.json.gz").read_bytes()))
host, host_count = Counter(), Counter()
for event in trace:
    if event.get("cat") == "cpu_op" and event.get("ph") == "X":
        host[event["name"]] += float(event["dur"])
        host_count[event["name"]] += 1
assert host_count["serving.decode_step"] == 20
report = {
    "packed_mlp": packed_mlp,
    "warmups": 5, "captured_real_b2_iterations": 20,
    "discarded_prior_partial_graph_rows": starts[0],
    "actual_positions_start": capture["observations"][5]["positions"],
    "actual_positions_end": capture["observations"][-1]["positions"],
    "mean_model_kernel_sum_us": statistics.mean(model_sums),
    "mean_model_device_envelope_us": statistics.mean(model_spans),
    "mean_control_kernel_sum_us": statistics.mean(controls),
    "mean_profiled_device_start_cadence_us": statistics.mean(
        float(rows[b]["Start Time(us)"]) - float(rows[a]["Start Time(us)"])
        for a, b in zip(starts, starts[1:])),
    "model_kernel_types": {k: {"count": counts[k], "mean_us_per_iteration": v / 20}
                           for k, v in types.most_common()},
    "host_nested_scopes_not_additive": {k: {"count": host_count[k], "mean_us_per_iteration": host[k] / 20}
                                       for k in ("serving.decode_step", "cache_compiler inference", "TorchNpuGraphBase::Run", "Event::record")},
    "frequency_observations_mhz": [e["args"]["MHz"] for e in trace
                                   if e.get("name") == "AI Core Freq" and e.get("ph") == "C"],
    "raw_trace_sha256": hashlib.sha256(trace_path.read_bytes() if trace_path.exists()
                                       else gzip.decompress((inputs / "trace_view.json.gz").read_bytes())).hexdigest(),
    "profiled_latencies_are_not_goal_measurements": True,
}
(ROOT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
