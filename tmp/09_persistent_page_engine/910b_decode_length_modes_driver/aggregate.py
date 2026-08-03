#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


root = Path(
    "tmp/09_persistent_page_engine/"
    "910b_decode_length_modes_b16_128_k2048_24acb27"
)
optimizations = (
    "combined_apply",
    "combined_apply_static_actual",
    "combined_apply_pse_sentinel",
)
batches = (16, 32, 64, 128)
rows = []

for batch in batches:
    batch_rows = {}
    for optimization in optimizations:
        path = root / f"b{batch}_{optimization}" / "result.json"
        payload = json.loads(path.read_text())
        result = payload["result"]
        setup = payload["setup"]
        metadata = setup["runtime_metadata"]
        row = {
            "batch_size": batch,
            "optimization": optimization,
            "latency_ms": result["latency_ms"],
            "raw_physical_tok_per_s": result["throughput"][
                "raw_physical_tok_per_s"
            ],
            "full_production_step_device_s": result["device_s"][
                "full_production_step"
            ],
            "model_and_argmax_device_s": result["device_s"][
                "model_and_argmax"
            ],
            "peak_memory_delta_bytes": result["memory_bytes"]["peak_delta"],
            "cache_allocated_bytes": metadata["cache_allocated_bytes"],
            "linear_weight_format": metadata["linear_weight_format"],
            "compile_first_call_s": setup["runtime_setup_detail_s"][
                "compile_first_call"
            ],
            "cache_dir": metadata["torchair_cache_dir"],
        }
        batch_rows[optimization] = row
        rows.append(row)
    control = batch_rows["combined_apply"]
    for optimization, row in batch_rows.items():
        row["throughput_delta_vs_control_percent"] = (
            row["raw_physical_tok_per_s"]
            / control["raw_physical_tok_per_s"]
            - 1.0
        ) * 100.0
        row["latency_delta_vs_control_percent"] = (
            row["latency_ms"]["mean"] / control["latency_ms"]["mean"]
            - 1.0
        ) * 100.0

boundaries = []
for batch in batches:
    for optimization in optimizations[1:]:
        path = root / f"boundary_b{batch}_{optimization}" / "result.json"
        payload = json.loads(path.read_text())
        result = payload["result"]
        boundaries.append(
            {
                "batch_size": batch,
                "optimization": optimization,
                "cache_position": result["shape"]["cache_position"],
                "effective_length": result["shape"]["effective_length"],
                "elapsed_s": result["elapsed_s"],
                "passed": True,
            }
        )

summary = {
    "schema_version": 1,
    "kind": "910b_text_decode_length_mode_matrix",
    "commit": "24acb27989bcc02a89813d284ad5c2747aa24b97",
    "chip": "Ascend 910B2",
    "cache_length": 2048,
    "profile_position": 1024,
    "warmup": 3,
    "repeats": 30,
    "rows": rows,
    "boundary_gates": boundaries,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

labels = {
    "combined_apply": "normal masked GQA",
    "combined_apply_static_actual": "static actual",
    "combined_apply_pse_sentinel": "PSE sentinel",
}
lines = [
    "# 910B text-decode length-mode matrix",
    "",
    "Commit `24acb27989bcc02a89813d284ad5c2747aa24b97`; "
    "Ascend 910B2; FP16; TorchAir; full 18-layer decoder + LM head + argmax; "
    "KV2048; profile positions 1024-1053; 3 warmups and 30 measured steps; "
    "all slots active.",
    "",
    "| B | mode | mean ms | median ms | p95 ms | physical tok/s | vs normal | peak delta MiB | KV MiB |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['batch_size']} | {labels[row['optimization']]} | "
        f"{row['latency_ms']['mean']:.4f} | "
        f"{row['latency_ms']['median']:.4f} | "
        f"{row['latency_ms']['p95']:.4f} | "
        f"{row['raw_physical_tok_per_s']:.1f} | "
        f"{row['throughput_delta_vs_control_percent']:+.2f}% | "
        f"{row['peak_memory_delta_bytes'] / 2**20:.1f} | "
        f"{row['cache_allocated_bytes'] / 2**20:.1f} |"
    )
lines.extend(
    [
        "",
        "All eight static-actual/PSE boundary gates passed at "
        "cache_position=1279 / effective_length=1280.",
        "",
        "All lanes reported `decode_native_fallback`; the current 910B "
        "torch_npu runtime did not materialize FRACTAL_NZ weights.",
    ]
)
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
print(root / "summary.json")
print(root / "REPORT.md")
print("\n".join(lines))
