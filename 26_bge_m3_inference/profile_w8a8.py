"""Capture warmed compiled forwards using the Ascend PyTorch profiler."""
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu.profiler as npu_prof


def capture(fn, directory: Path, label: str, steps: int = 3):
    directory.mkdir(parents=True, exist_ok=False)
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=npu_prof.schedule(wait=0, warmup=1, active=steps, repeat=1),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(directory), analyse_flag=True),
        record_shapes=True, profile_memory=False, with_stack=False,
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=npu_prof.ProfilerLevel.Level1,
            aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
            export_type=npu_prof.ExportType.Text,
        ),
    ) as profiler:
        for _ in range(steps + 1):
            with torch.profiler.record_function("bge_m3." + label):
                fn()
            torch.npu.synchronize()
            profiler.step()

    # Reuse the repository's model-agnostic CANN CSV/trace parser.
    parser_path = Path(__file__).resolve().parents[1] / "05_full_recognizer_optimizations/parse_npu_profile.py"
    spec = importlib.util.spec_from_file_location("bge_profile_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    runs = [parser.parse_run(root, topn=50, skip_trace=False)
            for root in parser.find_run_roots(directory)]
    if len(runs) != 1 or not runs[0].get("kernel_details", {}).get("row_count"):
        raise RuntimeError(f"Missing device kernel evidence: {directory}")
    report = {"label": label, "active_forwards": steps, "runs": runs,
              "note": "Kernel duration sums may overlap; profiled timings include instrumentation overhead."}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    kernels = runs[0]["kernel_details"]
    return {"directory": str(directory), "active_forwards": steps,
            "kernel_rows": kernels["row_count"],
            "kernel_duration_us_per_forward": kernels["total_duration_us"] / steps,
            "top_types": [{"name": row["name"], "count_per_forward": row["count"] / steps,
                           "us_per_forward": row["duration_us"] / steps}
                          for row in kernels["top_kernel_types"]]}
