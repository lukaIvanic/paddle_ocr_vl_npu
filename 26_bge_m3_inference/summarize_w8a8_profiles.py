"""Normalize raw profiler kernels by forward; audit coverage with CPU step ranges."""
import argparse
import csv
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import statistics


def summarize(run_root: Path):
    benchmark = json.loads((run_root / "result.json").read_text())
    report = {"source_commit": benchmark["git_commit"], "device": benchmark["device"],
              "note": "Device durations are profiled; wall medians are measured without profiling.", "profiles": []}
    for case in benchmark["cases"]:
        for mode, capture in case["profiles"].items():
            label = case["name"] + "_" + mode
            directory, = (run_root / "profiles" / label).glob("*/ASCEND_PROFILER_OUTPUT")
            with (directory / "kernel_details.csv").open(newline="") as handle:
                kernels = list(csv.DictReader(handle))
            raw = json.loads((directory / "trace_view.json").read_text())
            events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
            steps = [event for event in events if event.get("ph") == "X"
                     and event.get("name", "").startswith("ProfilerStep#")]
            assert len(steps) == capture["active_forwards"]
            origin = min(Decimal(str(step["ts"])) for step in steps)
            intervals = sorted((float(Decimal(row["Start Time(us)"].strip()) - origin),
                                float(row["Duration(us)"])) for row in kernels)
            timings = []
            for step in steps:
                start = float(Decimal(str(step["ts"])) - origin)
                end = start + float(step["dur"])
                selected = [(s, s+d) for s, d in intervals if start <= s and s+d <= end]
                assert selected, label
                union, covered_end = 0.0, selected[0][0]
                for s, e in selected:
                    union += max(0.0, e - max(s, covered_end))
                    covered_end = max(covered_end, e)
                span = max(e for _, e in selected) - selected[0][0]
                timings.append({"step": step["name"], "kernel_count": len(selected),
                                "kernel_span_us": span, "kernel_union_us": union,
                                "internal_gap_us": span - union, "cpu_step_us": float(step["dur"])})
            assert sum(row["kernel_count"] for row in timings) == len(kernels), label
            groups = defaultdict(lambda: [0, 0.0])
            for row in kernels:
                for key in ((row["Type"], "", "", ""),
                            (row["Type"], row["Input Shapes"], row["Input Formats"], row["Input Data Types"])):
                    groups[key][0] += 1
                    groups[key][1] += float(row["Duration(us)"])
            normalized = [{"type": key[0], "input_shapes": key[1], "input_formats": key[2],
                           "input_dtypes": key[3], "count_per_forward": count / len(steps),
                           "us_per_forward": duration / len(steps)}
                          for key, (count, duration) in groups.items()]
            types = sorted((row for row in normalized if not row["input_shapes"]),
                           key=lambda row: row["us_per_forward"], reverse=True)
            shapes = sorted((row for row in normalized if row["input_shapes"]),
                            key=lambda row: row["us_per_forward"], reverse=True)
            report["profiles"].append({"label": label, "mode": mode, "case": case["name"],
                "active_forwards": len(steps), "wall_median_ms": case["modes"][mode]["median_ms"],
                "kernel_csv": str((directory / "kernel_details.csv").relative_to(run_root)),
                "kernel_count_per_forward": len(kernels) / len(steps), "steps": timings,
                "kernel_sum_us_per_forward": sum(float(row["Duration(us)"]) for row in kernels) / len(steps),
                "median_kernel_span_us": statistics.median(row["kernel_span_us"] for row in timings),
                "median_internal_gap_us": statistics.median(row["internal_gap_us"] for row in timings),
                "types": types, "shapes": shapes})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kernel_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for key in ("types", "shapes"):
        rows = [{"profile": profile["label"], **row} for profile in report["profiles"] for row in profile[key]]
        with (args.output_dir / f"kernel_{key}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"profiles": len(report["profiles"]), "coverage_audit": "passed"}))


if __name__ == "__main__":
    main()
