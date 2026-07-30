# Vision MatMul profiler analysis

Generated: `2026-07-30T14:05:48.784115+00:00`

This report treats `kernel_details.csv` as the canonical execution-to-PMU association. Separate metric lanes are separate captures and are never interpreted as simultaneous samples.

## Capture provenance

- Commit: `7c64b7e93a0224039a36a8c123987b9b290b58ca`
- Host/device: `liteserver-c001-4` / `Ascend910B2` / physical `5`
- Torch / torch_npu / CANN: `2.10.0+cpu` / `2.10.0` / `/usr/local/Ascend/cann-9.0.0`
- Shape: `B1 x S2048`, H`1152`, I`4352`, 27 layers
- Execution: `torchair`, attention padding `weights`, RoPE `separate_manual`
- Weight format request/status: `fractal_nz` / `ready`
- Compile API/cache: `torchair.inference.cache_compile` / `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_lab/vision_pfa_b1_s2048_i4352_fractal_nz_hpweights_57f5149e44aa4fe4`
- Unprofiled device-event baseline: `26.013 ms`, `78730.585 physical tok/s`

## Lane summary

| metric lane | kernels | MatMuls | mapping | span / replay | MatMul duration / replay | MatMul share | kernel-local TFLOP/s | stage-effective TFLOP/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `pipe` | 2598 | 486 | validated | 26164.167 us | 8073.913 us | 30.859% | 218.135 | 67.314 |

## Validated vision linear mapping

Ordinal layer/role names are emitted only when every replay contains exactly 162 MatMul kernels and its repeating six-shape motif matches the supplied hidden/intermediate/attention-width contract.

- `pipe`: **validated**, method `step_id`

## Full-layer and cross-lane validation

- `pipe` full graph: **failed**, method `unavailable` — full replay does not match the validated 27x39 fixture
- Cross-lane complete-kernel signature: **single_lane** — cross-lane comparison requires at least two lanes

## Per-role summary

### `pipe`

| role | count | duration (us mean) | duration / replay (us) | TFLOP/s | AICore (us mean) | MAC (us mean) | MTE1 | MTE2 | FixPipe | exported AI-core-time ratio (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 81 | 29.600 | 799.213 | 204.044 | 28.293 | 20.808 | 11.318 | 23.146 | 8.061 | 79.663 |
| k_proj | 81 | 26.695 | 720.773 | 226.249 | 25.759 | 20.790 | 11.302 | 21.371 | 7.920 | 80.419 |
| v_proj | 81 | 27.038 | 730.013 | 223.386 | 25.999 | 20.807 | 11.310 | 21.737 | 7.938 | 80.140 |
| out_proj | 81 | 39.292 | 1060.887 | 153.715 | 37.508 | 17.390 | 9.018 | 31.883 | 12.138 | 95.561 |
| fc1 | 81 | 84.537 | 2282.507 | 242.914 | 78.338 | 58.971 | 39.589 | 63.951 | 30.435 | 92.679 |
| fc2 | 81 | 91.871 | 2480.520 | 223.523 | 88.342 | 59.184 | 42.867 | 79.718 | 13.442 | 96.207 |

## Whole-stage kernel-time composition

Timing composition uses the `pipe` lane. Counts and durations below are normalized to one full-stack replay.

| kernel type | count / replay | duration / replay (us) | share of replay span |
|---|---:|---:|---:|
| MatMulV2 | 162.000 | 8073.913 | 30.859% |
| PromptFlashAttention | 27.000 | 7590.247 | 29.010% |
| StridedSliceD | 108.000 | 2830.667 | 10.819% |
| AddLayerNorm | 54.000 | 1355.993 | 5.183% |
| Transpose | 108.000 | 1185.787 | 4.532% |
| Mul | 108.000 | 1057.000 | 4.040% |
| Gelu | 27.000 | 1011.920 | 3.868% |
| ConcatV2D | 81.000 | 870.667 | 3.328% |
| Add | 54.000 | 611.400 | 2.337% |
| Cast | 54.000 | 566.300 | 2.164% |
| Neg | 54.000 | 462.593 | 1.768% |
| SplitVD | 27.000 | 413.547 | 1.581% |
| LayerNormV3 | 1.000 | 29.607 | 0.113% |
| Data | 1.000 | 5.020 | 0.019% |

## Interpretation warnings

- Pipe-utilization ratios overlap in time. Do not add MAC, MTE, Scalar, FixPipe, or Cube ratios as if they partition kernel duration.
- Block Num/Block Dim is a configured logical block count, not proof that the same number of physical AI Cores were active. Per-core records or hardware counters are required for that claim.
- On the observed CANN 9 whole-graph export, cube_utilization(%) is 100 * aicore_time / kernel Duration (within export rounding). It is an exported AI-core-time ratio that can exceed 100%, not physical occupancy, achieved MAC utilization, or a percentage of peak FLOP/s.
- Profiler README semantics define aicore_time as average task time on AI Core derived from total cycles / Block Num. Generated CANN documentation warns that this value is inaccurate on Atlas 300V and Atlas 300I Pro; keep it unavailable or diagnostic on those products.
- Profiler captures perturb execution. Use unprofiled device-event timings for throughput claims and profiler data for diagnosis.
- Dense MatMul FLOPs are inferred as 2*M*K*N from activation and output shapes. This is a physical-work estimate, not an algorithmic FLOP claim for fused or sparse kernels.
- The full-layer fixture is intentionally fail-closed and applies only to the exact validated 2-prefix + 27x39-kernel PaddleOCR-VL graph. A changed graph emits no non-linear layer labels.
- The first kernel's Wait Time in a replay can include the gap since the prior replay; replay wait totals are not internal graph stall time without timestamp decomposition.
- Profiler databases are schema-inventoried only. TASK, CANN_API, PYTORCH_API, and compiled-node relationships are not silently joined.
- Missing PMU fields are represented as null/unavailable, never zero.
- pipe: full 27-layer mapping is failed; non-linear kernels remain unlabeled.

## Durable outputs

- `profile_manifest.json`
- `profile_analysis.json`
- `kernel_executions.csv`
- `vision_linear_executions.csv`
- `vision_layer_summary.csv`
- `profile.sqlite`
- `profile_report.md`
