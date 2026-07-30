# Vision MatMul profiler analysis

Generated: `2026-07-30T13:57:37.656748+00:00`

This report treats `kernel_details.csv` as the canonical execution-to-PMU association. Separate metric lanes are separate captures and are never interpreted as simultaneous samples.

## Capture provenance

- Commit: `7c64b7e93a0224039a36a8c123987b9b290b58ca`
- Host/device: `liteserver-c001-4` / `Ascend910B2` / physical `5`
- Torch / torch_npu / CANN: `2.10.0+cpu` / `2.10.0` / `/usr/local/Ascend/cann-9.0.0`
- Shape: `B1 x S512`, H`1152`, I`4352`, 27 layers
- Execution: `torchair`, attention padding `weights`, RoPE `separate_manual`
- Weight format request/status: `fractal_nz` / `ready`
- Compile API/cache: `torchair.inference.cache_compile` / `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_lab/vision_pfa_b1_s512_i4352_fractal_nz_hpweights_17431b74d4869cb3`
- Unprofiled device-event baseline: `11.137 ms`, `45970.827 physical tok/s`

## Lane summary

| metric lane | kernels | MatMuls | mapping | span / replay | MatMul duration / replay | MatMul share | kernel-local TFLOP/s | stage-effective TFLOP/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `pipe` | 2598 | 486 | validated | 11339.500 us | 3337.680 us | 29.434% | 131.918 | 38.829 |

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
| q_proj | 81 | 23.496 | 634.393 | 64.264 | 21.506 | 4.609 | 4.251 | 17.733 | 6.933 | 91.557 |
| k_proj | 81 | 12.087 | 326.347 | 124.924 | 10.722 | 4.585 | 4.392 | 8.207 | 4.538 | 88.716 |
| v_proj | 81 | 12.176 | 328.753 | 124.010 | 10.887 | 4.586 | 4.427 | 8.359 | 4.385 | 89.443 |
| out_proj | 81 | 21.339 | 576.147 | 70.761 | 19.122 | 4.613 | 4.745 | 15.630 | 6.306 | 89.905 |
| fc1 | 81 | 27.125 | 732.380 | 189.264 | 24.519 | 14.799 | 10.548 | 19.919 | 8.388 | 90.471 |
| fc2 | 81 | 27.395 | 739.660 | 187.401 | 25.474 | 14.853 | 10.585 | 21.666 | 2.797 | 93.067 |

## Whole-stage kernel-time composition

Timing composition uses the `pipe` lane. Counts and durations below are normalized to one full-stack replay.

| kernel type | count / replay | duration / replay (us) | share of replay span |
|---|---:|---:|---:|
| MatMulV2 | 162.000 | 3337.680 | 29.434% |
| PromptFlashAttention | 27.000 | 1677.093 | 14.790% |
| StridedSliceD | 108.000 | 1479.580 | 13.048% |
| Transpose | 108.000 | 974.213 | 8.591% |
| AddLayerNorm | 54.000 | 745.640 | 6.576% |
| Mul | 108.000 | 613.193 | 5.408% |
| ConcatV2D | 81.000 | 521.987 | 4.603% |
| Cast | 54.000 | 461.873 | 4.073% |
| Add | 54.000 | 422.347 | 3.725% |
| Gelu | 27.000 | 410.040 | 3.616% |
| Neg | 54.000 | 403.933 | 3.562% |
| SplitVD | 27.000 | 169.487 | 1.495% |
| LayerNormV3 | 1.000 | 13.747 | 0.121% |
| Data | 1.000 | 4.800 | 0.042% |

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
