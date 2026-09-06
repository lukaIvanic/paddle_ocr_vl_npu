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
if (ROOT / "cadence.csv").exists():
    capture = json.loads((ROOT / "capture.json").read_text())
    samples = list(csv.DictReader((ROOT / "cadence.csv").open()))
    assert len(samples) == capture["calls"]

    def describe(values):
        values = sorted(values)
        def quantile(q):
            pos = (len(values) - 1) * q
            lo = int(pos)
            return values[lo] + (values[min(lo+1, len(values)-1)] - values[lo]) * (pos-lo)
        return {"count": len(values), "mean": statistics.mean(values),
                "p50": quantile(.5), "p95": quantile(.95), "max": values[-1]}

    groups = {}
    for active in (None, 1, 2):
        chosen = [r for r in samples if active is None or int(r["active_slots"]) == active]
        if not chosen:
            continue
        values = {}
        for name, start, end in (
            ("host_call_wall_us", "call_start_ns", "call_end_ns"),
            ("host_call_thread_cpu_us", "call_cpu_start_ns", "call_cpu_end_ns"),
            ("host_step_wall_us", "step_start_ns", "step_end_ns"),
            ("event_enqueue_to_host_call_start_us", "event_start_enqueue_ns", "call_start_ns"),
        ):
            values[name] = [(int(r[end]) - int(r[start])) / 1000 for r in chosen]
        values["host_call_wall_minus_thread_cpu_us"] = [
            wall-cpu for wall, cpu in zip(values["host_call_wall_us"], values["host_call_thread_cpu_us"])]
        values["device_event_interval_us"] = [float(r["device_interval_ms"]) * 1000 for r in chosen]
        groups["all" if active is None else str(active)] = {k: describe(v) for k, v in values.items()}
    report = {"diagnostic_only": True, "configuration": capture["configuration"],
              "by_active_slots": groups, "cpu_affinity": capture["cpu_affinity"],
              "torch_num_threads": capture["torch_num_threads"],
              "torch_num_interop_threads": capture["torch_num_interop_threads"],
              "note": "Host scopes overlap device execution; never add or subtract these from request latency. All includes warmup. Two-active-slot rows exclude the C1 warmup, but are still diagnostic."}
    if (ROOT / "gc_events.json").exists():
        gc_events = json.loads((ROOT / "gc_events.json").read_text())
        report["gc_by_generation"] = {
            str(g): describe([(e["end_ns"]-e["start_ns"])/1e6 for e in gc_events if e["generation"] == g])
            for g in sorted({e["generation"] for e in gc_events})}
        longest = sorted(samples, key=lambda r: int(r["call_end_ns"])-int(r["call_start_ns"]), reverse=True)[:10]
        report["longest_calls_and_gc_overlap"] = []
        for row in longest:
            start, end = int(row["call_start_ns"]), int(row["call_end_ns"])
            overlap = [e for e in gc_events if e["start_ns"] < end and e["end_ns"] > start]
            report["longest_calls_and_gc_overlap"].append({
                "iteration": int(row["iteration"]), "host_call_ms": (end-start)/1e6,
                "thread_cpu_ms": (int(row["call_cpu_end_ns"])-int(row["call_cpu_start_ns"]))/1e6,
                "device_interval_ms": float(row["device_interval_ms"]),
                "gc_overlap_ms": sum(max(0,min(end,e["end_ns"])-max(start,e["start_ns"])) for e in overlap)/1e6,
                "gc_generations": [e["generation"] for e in overlap],
            })
    (ROOT / "cadence_analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("configuration", "cpu_affinity")}, indent=2))
    raise SystemExit(0)
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
