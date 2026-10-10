"""Audit profiler-step coverage and compare device work in baseline/V2 runs."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = json.loads((args.run_root / "result.json").read_text())
    report = {"source_commit": result["git_commit"], "device": result["device"], "profiles": []}
    for summary in sorted((args.run_root / "profiles").glob("*/summary.json")):
        metadata = json.loads(summary.read_text())
        directory, = summary.parent.glob("*/ASCEND_PROFILER_OUTPUT")
        with (directory / "kernel_details.csv").open() as f:
            kernels = list(csv.DictReader(f))
        raw = json.loads((directory / "trace_view.json").read_text())
        events = raw["traceEvents"] if isinstance(raw, dict) else raw
        steps = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")]
        assert len(steps) == metadata["active_forwards"]
        origin = min(Decimal(str(s["ts"])) for s in steps)
        timed = [(float(Decimal(k["Start Time(us)"].strip())-origin), float(k["Duration(us)"]), k) for k in kernels]
        per_step, grouped = [], defaultdict(list)
        for step in steps:
            start = float(Decimal(str(step["ts"]))-origin)
            selected = sorted((s, d, k) for s, d, k in timed if start <= s and s+d <= start+float(step["dur"]))
            assert selected
            union, end = 0., selected[0][0]
            types = defaultdict(float)
            for s, d, k in selected:
                union += max(0., s+d-max(s, end))
                end = max(end, s+d)
                types[k["Type"]] += d
            for key, duration in types.items():
                grouped[key].append(duration)
            per_step.append({"kernel_count": len(selected), "kernel_sum_us": sum(d for _, d, _ in selected),
                             "kernel_span_us": end-selected[0][0], "kernel_union_us": union,
                             "internal_gap_us": end-selected[0][0]-union})
        assert sum(s["kernel_count"] for s in per_step) == len(kernels)
        counts = defaultdict(int)
        hardware = defaultdict(list)
        shapes = defaultdict(lambda: [0, 0.])
        for k in kernels:
            counts[k["Type"]] += 1
            if k["Type"] in {"AddLayerNorm", "AddLayerNormQuantV2", "Quantize"}:
                hardware[(k["Type"], k["Input Shapes"])].append(k)
            key = (k["Type"], k["Input Shapes"], k["Input Formats"], k["Input Data Types"])
            shapes[key][0] += 1
            shapes[key][1] += float(k["Duration(us)"])
        row = {"label": summary.parent.name, "steps": per_step,
               "kernel_count_per_forward": len(kernels)/len(steps),
               "mean_kernel_sum_us": statistics.mean(s["kernel_sum_us"] for s in per_step),
               "median_kernel_span_us": statistics.median(s["kernel_span_us"] for s in per_step),
               "median_gap_us": statistics.median(s["internal_gap_us"] for s in per_step),
               "types": {k: {"count": counts[k]/len(steps), "mean_us": statistics.mean(v), "median_us": statistics.median(v)}
                         for k, v in grouped.items()},
               "shapes": [{"type": k[0], "shapes": k[1], "formats": k[2], "dtypes": k[3],
                           "count": v[0]/len(steps), "mean_us": v[1]/len(steps)} for k, v in shapes.items()]}
        fields = ("aiv_time(us)", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio")
        row["norm_quant_hardware"] = [{"type": key[0], "input_shapes": key[1],
            "block_counts": sorted({int(k["Block Num"]) for k in values}),
            "accelerator_cores": sorted({k["Accelerator Core"] for k in values}),
            "mean_counters": {field: statistics.mean(float(k[field]) for k in values) for field in fields}}
            for key, values in hardware.items()]
        report["profiles"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"profiles": len(report["profiles"]), "coverage_audit": "passed"}))


if __name__ == "__main__":
    main()
