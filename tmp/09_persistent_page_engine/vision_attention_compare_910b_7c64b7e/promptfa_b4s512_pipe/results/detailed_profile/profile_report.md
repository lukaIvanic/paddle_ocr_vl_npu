# Vision MatMul profiler analysis

Generated: `2026-07-30T14:02:06.397195+00:00`

This report treats `kernel_details.csv` as the canonical execution-to-PMU association. Separate metric lanes are separate captures and are never interpreted as simultaneous samples.

## Capture provenance

- Commit: `7c64b7e93a0224039a36a8c123987b9b290b58ca`
- Host/device: `liteserver-c001-4` / `Ascend910B2` / physical `5`
- Torch / torch_npu / CANN: `2.10.0+cpu` / `2.10.0` / `/usr/local/Ascend/cann-9.0.0`
- Shape: `B4 x S512`, H`1152`, I`4352`, 27 layers
- Execution: `torchair`, attention padding `weights`, RoPE `separate_manual`
- Weight format request/status: `fractal_nz` / `ready`
- Compile API/cache: `torchair.inference.cache_compile` / `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_lab/vision_pfa_b4_s512_i4352_fractal_nz_hpweights_8147c443b95ab6d6`
- Unprofiled device-event baseline: `22.280 ms`, `91920.469 physical tok/s`

## Lane summary

| metric lane | kernels | MatMuls | mapping | span / replay | MatMul duration / replay | MatMul share | kernel-local TFLOP/s | stage-effective TFLOP/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `pipe` | 2598 | 486 | validated | 22450.250 us | 8035.960 us | 35.795% | 219.165 | 78.449 |

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
| q_proj | 81 | 30.365 | 819.860 | 198.905 | 28.964 | 20.808 | 11.291 | 23.489 | 8.081 | 79.526 |
| k_proj | 81 | 26.651 | 719.573 | 226.627 | 25.713 | 20.791 | 11.303 | 21.376 | 7.921 | 80.411 |
| v_proj | 81 | 26.903 | 726.380 | 224.503 | 25.929 | 20.808 | 11.307 | 21.698 | 7.925 | 80.318 |
| out_proj | 81 | 37.647 | 1016.460 | 160.434 | 35.978 | 17.407 | 8.965 | 30.186 | 11.767 | 95.594 |
| fc1 | 81 | 85.199 | 2300.373 | 241.028 | 78.887 | 58.975 | 39.575 | 63.958 | 29.867 | 92.622 |
| fc2 | 81 | 90.863 | 2453.313 | 226.002 | 86.932 | 59.172 | 42.896 | 78.141 | 12.806 | 95.692 |

## Whole-stage kernel-time composition

Timing composition uses the `pipe` lane. Counts and durations below are normalized to one full-stack replay.

| kernel type | count / replay | duration / replay (us) | share of replay span |
|---|---:|---:|---:|
| MatMulV2 | 162.000 | 8035.960 | 35.795% |
| StridedSliceD | 108.000 | 2951.027 | 13.145% |
| PromptFlashAttention | 27.000 | 2871.380 | 12.790% |
| Transpose | 108.000 | 1810.613 | 8.065% |
| AddLayerNorm | 54.000 | 1327.487 | 5.913% |
| Mul | 108.000 | 1110.620 | 4.947% |
| Gelu | 27.000 | 1035.640 | 4.613% |
| ConcatV2D | 81.000 | 915.113 | 4.076% |
| Add | 54.000 | 663.513 | 2.955% |
| Cast | 54.000 | 626.333 | 2.790% |
| Neg | 54.000 | 545.647 | 2.430% |
| SplitVD | 27.000 | 422.453 | 1.882% |
| LayerNormV3 | 1.000 | 32.587 | 0.145% |
| Data | 1.000 | 4.800 | 0.021% |

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
