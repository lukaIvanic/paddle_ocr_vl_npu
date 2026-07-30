# Vision msprof-op analysis

**Status:** `passed`

This report validates the direct replay target against the normalized compiled full-graph dispatch. Replay duration is diagnostic only and is never used as a throughput gate.

## Inputs

- Capture metric: `MemoryDetail`
- Capture directory: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/vision_msprof_op/910b_square_memorydetail_20260730`
- Raw directory: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_msprof_op/910b_square_memorydetail_20260730`
- Reference directory: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_b1s2048_i4352_nz_4lane_bacc8f6_analysis`

## Dispatch and reference validation

| direct | production roles | captured op | core type | Block Dim | input formats | output formats | FLOPs | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| square | q_proj, k_proj, v_proj, out_proj | MatMulV2_NDNZ_ND_ND_FP16_FP16_FP16_false_true_all_197328 | cube | 24 | ND;FRACTAL_NZ;ND | ND | 5,435,817,984 | passed |

## Duration diagnostics

These numbers come from different execution contexts. They are retained for investigation, not compared as a correctness or representativeness threshold.

| role | OpBasic us | visualize us | direct event us | compiled graph reference mean us |
| --- | --- | --- | --- | --- |
| square | 29.360 | 29.360 | 892.600 | q_proj=23.504, k_proj=23.122, v_proj=23.266, out_proj=29.153 |

## Active MTE bandwidth

These are active-cycle bandwidths, not whole-kernel or whole-card averages. Missing values and numeric zero are counted separately; a missing counter is never silently converted to zero.

| metric | records | missing | zeros | mean | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| Pipe.MTE1.active bw(GB/s) | 24 | 0 | 0 | 246.934 | 240.316 | 253.556 |
| Pipe.MTE2.active bw(GB/s) | 24 | 0 | 16 | 41.623 | 0.000 | 131.334 |
| Pipe.MTE3.active bw(GB/s) | 24 | 0 | 0 | 164,794.922 | 164,794.922 | 164,794.922 |
| PipeUtilization.aic_mte1_active_bw(GB/s) | 24 | 0 | 0 | 246.934 | 240.316 | 253.556 |
| PipeUtilization.aic_mte2_active_bw(GB/s) | 24 | 0 | 16 | 41.623 | 0.000 | 131.334 |
| PipeUtilization.aic_mte3_active_bw(GB/s) | 24 | 0 | 0 | 164,794.922 | 164,794.922 | 164,794.922 |
| PipeUtilization.aiv_mte2_active_bw(GB/s) | 24 | 24 | 0 | missing | missing | missing |
| PipeUtilization.aiv_mte3_active_bw(GB/s) | 24 | 24 | 0 | missing | missing | missing |

## Warnings

- MemoryDetail is product-limited in Huawei documentation; absence on an unsupported product is reported as unavailable, never interpreted as zero.

## Machine-readable outputs

- `analysis.json` — validation, statistics, and diagnostics
- `core_metrics.csv` — flattened per-core Occupancy values and flags
- `metric_records.csv` — normalized long-form metrics
- `schema_manifest.json` — recursive CSV and binary schema inventory
