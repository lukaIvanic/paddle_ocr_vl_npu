# 910B validation of the revised vision diagnostics — 2026-10-07

This validates the new handoff controls and measurement tooling. It does not replace the [original twelve-lane 910B results](../vision_crop_contracts_910b_20261007/RESULTS.md), establish a 310P ranking, or turn standalone matmul calibration into model performance.

All inference here is FP16 on Ascend910B2. Full vision ran on physical card 1 at source `af0dd464`; the 48-case internal-format-on calibration ran on physical card 2 (initial two eager cases at `af0dd464`, the remainder at `9d153d90`). Two additional ND internal-format-off checks ran on card 1 at `9d153d90`. The environment reports torch 2.10.0+cpu and torch-npu 2.10.0; the selected device is explicitly NPU, with no CPU inference fallback. Source commits and selected environment variables are recorded per child before launch.

## Configuration and scope

The validation capture deliberately sets internal formats **off**, to exercise recovery of a configuration different from the earlier control. The `internal_format_opposite` lane switches them on. This is a controlled 910B validation choice, not a claim about the historical 310P production setting. The receiving agent must recover that setting from the actual command receipt and source at its recorded commit.

| Setting | Original controlled 910B capture | New validation capture/baseline | New opposite-format lane | Historical 310P production |
|---|---|---|---|---|
| Attention / precision | PromptFA D80 / default | same | same | Must recover |
| Projection | linear | linear; exploratory overrides below | linear | Must recover |
| LayerNorm / dtype | manual FP32 / FP16 tensors | same | same | Must recover |
| Buckets | 384,768,3072 | same | same | Must recover |
| Processor pixel caps | 25088–602112 | same | same | Must recover |
| allow_internal_format | true | false | true | Must recover |
| Execution and measurement | TorchAir fullgraph; 32 vision blocks | same | same | Must recover |

Both inputs are real repository crops: 720 useful raw tokens in bucket768 and 3036 in bucket3072. Capture executes preprocessing, patch embeddings, positions, all 32 blocks and merger. Replay times the complete 32-block stack; CPU/H2D, initial embeddings/positions and merger remain outside that timer. No generated text tok/s or page/s claim is made.

## Full vision validation

Five warm unprofiled forwards per lane, followed by three separately profiled forwards. These short runs validate configurations, graph reuse and diagnostics; they are not a new production performance ranking. Compile/cache loading is outside warm timing. All eight lanes have zero new graphs/recompile warnings inside timing, repeat-exact finite outputs, and both baselines are bit-exact with their own capture. Numerical candidate drift remains visible alongside timing.

| Chip | Lane | Real tokens | Internal format | Wall ms | Event ms | Useful vision tok/s | Relative L2 drift | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
| 910B2 | baseline | 720 | False | 18.733 | 18.403 | 38435 | 0.000000 | completed |
| 910B2 | baseline | 3036 | False | 50.177 | 49.699 | 60505 | 0.000000 | completed |
| 910B2 | grouped_qkv | 720 | False | 18.948 | 18.658 | 38000 | 0.007829 | completed |
| 910B2 | grouped_qkv | 3036 | False | 50.040 | 49.749 | 60672 | 0.000000 | completed |
| 910B2 | grouped_qkv_mlp_fc1 | 720 | False | 20.040 | 19.759 | 35928 | 0.009428 | completed |
| 910B2 | grouped_qkv_mlp_fc1 | 3036 | False | 51.040 | 50.756 | 59483 | 0.000000 | completed |
| 910B2 | internal_format_opposite | 720 | True | 18.243 | 17.950 | 39467 | 0.000000 | completed |
| 910B2 | internal_format_opposite | 3036 | True | 50.006 | 49.604 | 60713 | 0.000000 | completed |

Grouped modes remain exploratory compile-workaround lanes, **not known speedups**. They actually execute GroupedMatmul: 32 calls/forward for grouped_qkv and 64 for grouped_qkv_mlp_fc1. On the small crop, relative L2 drift is 0.78% and 0.94%; on the large crop both are exact in this run. These feature measurements do not establish OCR-quality equivalence. The internal-format toggle is exact on both crops.

## Kernel accounting

Each profile is divided by three full forwards. Full per-type tables, separate total wait and remaining buckets are in [ACCOUNTING.md](ACCOUNTING.md); per-call Block Num/Mix/Accelerator Core/HF32/shapes/formats, conversion directions and stream gaps are in [diagnostic_analysis.json.gz](diagnostic_analysis.json.gz). The generic analyzer classifies GroupedMatmul without a fixed-count assertion.

For the large baseline, MatMulV2 uses Block Num 21 and MatMulV3 uses 24; PromptFA uses Block Num 24/Mix Block Num 48. The grouped kernels report Block Num 24 and MIX_AIC. These are launch-grid fields, not measured utilization percentages. Wait counters and observed gaps are separate, non-additive diagnostics; they are not automatically host delay.

## Synthetic matmul calibration only

All **48 requested cases** completed: M768/3072/5632 × qkv/proj/fc1/fc2 × eager/compiled × ND/NZ, with internal formats on. Two additional M768/qkv ND cases passed with internal formats off (eager and compiled). Every successful case uses 30 warm samples, three separately profiled calls, zero timed compilation, repeat-exact outputs and a reported small FP32 reference-slice error. This does not resolve the full-vision unpad drift.

TFLOPS below is `2*M*K*N / summed_matmul_kernel_duration_per_forward`, excluding bias FLOPs. It is device-kernel calibration, not the tiny-forward wall throughput and not model throughput. Actual kernel formats, block counts and per-call duration are retained. No cross-chip ratio is inferred from these 910B-only numbers. For example, M3072 qkv with NZ weights uses MatMulV2 when compiled and MatMulV3 in eager mode; both profiles show FRACTAL_NZ input weights. Kernel selection must therefore be inspected alongside the storage format.

| Chip | M | Projection K→N | Mode | Internal formats | Requested/actual weight format | Kernel type / Block Num | Kernel us/forward | TFLOPS | Status |
|---|---:|---|---|---|---|---|---:|---:|---|
| 910B2 | 768 | 1280→3840 | compiled | True | nd/None | {} | NA | NA | failed |
| 910B2 | 768 | 1280→3840 | eager | True | nd/2 | {'MatMulV2': {'24': 3}} | 31.574 | 239.115 | completed |
| 910B2 | 768 | 1280→3840 | eager | True | nz/29 | {'MatMulV2': {'24': 3}} | 34.628 | 218.026 | completed |
| 910B2 | 3072 | 1280→5120 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 128.536 | 313.261 | completed |
| 910B2 | 3072 | 1280→5120 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 144.529 | 278.596 | completed |
| 910B2 | 3072 | 5120→1280 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 125.662 | 320.425 | completed |
| 910B2 | 3072 | 5120→1280 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 144.836 | 278.006 | completed |
| 910B2 | 3072 | 1280→1280 | compiled | True | nd/2 | {'MatMulV2': {'21': 3}} | 67.229 | 149.733 | completed |
| 910B2 | 3072 | 1280→1280 | compiled | True | nz/29 | {'MatMulV2': {'21': 3}} | 71.981 | 139.846 | completed |
| 910B2 | 3072 | 1280→3840 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 99.055 | 304.870 | completed |
| 910B2 | 3072 | 1280→3840 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 113.016 | 267.211 | completed |
| 910B2 | 5632 | 1280→5120 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 226.058 | 326.553 | completed |
| 910B2 | 5632 | 1280→5120 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 258.579 | 285.483 | completed |
| 910B2 | 5632 | 5120→1280 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 240.625 | 306.784 | completed |
| 910B2 | 5632 | 5120→1280 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 264.452 | 279.142 | completed |
| 910B2 | 5632 | 1280→1280 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 71.514 | 258.059 | completed |
| 910B2 | 5632 | 1280→1280 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 71.421 | 258.397 | completed |
| 910B2 | 5632 | 1280→3840 | compiled | True | nd/2 | {'MatMulV3': {'24': 3}} | 175.810 | 314.913 | completed |
| 910B2 | 5632 | 1280→3840 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 193.657 | 285.891 | completed |
| 910B2 | 768 | 1280→5120 | compiled | True | nd/2 | {'MatMulV2': {'24': 3}} | 38.854 | 259.081 | completed |
| 910B2 | 768 | 1280→5120 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 43.794 | 229.856 | completed |
| 910B2 | 768 | 5120→1280 | compiled | True | nd/2 | {'MatMulV2': {'24': 3}} | 44.534 | 226.035 | completed |
| 910B2 | 768 | 5120→1280 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 45.668 | 220.426 | completed |
| 910B2 | 768 | 1280→1280 | compiled | True | nd/2 | {'MatMulV2': {'24': 3}} | 21.134 | 119.079 | completed |
| 910B2 | 768 | 1280→1280 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 29.854 | 84.296 | completed |
| 910B2 | 768 | 1280→3840 | compiled | True | nd/2 | {'MatMulV2': {'24': 3}} | 33.201 | 227.397 | completed |
| 910B2 | 768 | 1280→3840 | compiled | True | nz/29 | {'MatMulV2': {'24': 3}} | 35.907 | 210.256 | completed |
| 910B2 | 768 | 1280→5120 | eager | True | nd/2 | {'MatMulV2': {'24': 3}} | 37.714 | 266.910 | completed |
| 910B2 | 768 | 1280→5120 | eager | True | nz/29 | {'MatMulV2': {'24': 3}} | 38.614 | 260.693 | completed |
| 910B2 | 768 | 5120→1280 | eager | True | nd/2 | {'MatMulV2': {'24': 3}} | 45.301 | 222.212 | completed |
| 910B2 | 768 | 5120→1280 | eager | True | nz/29 | {'MatMulV2': {'24': 3}} | 45.367 | 221.885 | completed |
| 910B2 | 768 | 1280→1280 | eager | True | nd/2 | {'MatMulV2': {'24': 3}} | 18.014 | 139.701 | completed |
| 910B2 | 768 | 1280→1280 | eager | True | nz/29 | {'MatMulV2': {'24': 3}} | 18.814 | 133.764 | completed |
| 910B2 | 3072 | 1280→5120 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 124.389 | 323.705 | completed |
| 910B2 | 3072 | 1280→5120 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 127.089 | 316.827 | completed |
| 910B2 | 3072 | 5120→1280 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 126.303 | 318.799 | completed |
| 910B2 | 3072 | 5120→1280 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 127.095 | 316.812 | completed |
| 910B2 | 3072 | 1280→1280 | eager | True | nd/2 | {'MatMulV2': {'21': 3}} | 66.334 | 151.751 | completed |
| 910B2 | 3072 | 1280→1280 | eager | True | nz/29 | {'MatMulV2': {'21': 3}} | 68.341 | 147.296 | completed |
| 910B2 | 3072 | 1280→3840 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 98.348 | 307.062 | completed |
| 910B2 | 3072 | 1280→3840 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 97.809 | 308.756 | completed |
| 910B2 | 5632 | 1280→5120 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 224.978 | 328.120 | completed |
| 910B2 | 5632 | 1280→5120 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 226.464 | 325.966 | completed |
| 910B2 | 5632 | 5120→1280 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 237.112 | 311.329 | completed |
| 910B2 | 5632 | 5120→1280 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 238.372 | 309.683 | completed |
| 910B2 | 5632 | 1280→1280 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 67.795 | 272.218 | completed |
| 910B2 | 5632 | 1280→1280 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 67.208 | 274.596 | completed |
| 910B2 | 5632 | 1280→3840 | eager | True | nd/2 | {'MatMulV3': {'24': 3}} | 172.324 | 321.283 | completed |
| 910B2 | 5632 | 1280→3840 | eager | True | nz/29 | {'MatMulV3': {'24': 3}} | 173.511 | 319.086 | completed |
| 910B2 | 768 | 1280→3840 | compiled | False | nd/2 | {'MatMulV2': {'24': 3}} | 32.481 | 232.438 | completed |
| 910B2 | 768 | 1280→3840 | compiled | False | nz/None | {} | NA | NA | unsupported_configuration |
| 910B2 | 768 | 1280→3840 | eager | False | nd/2 | {'MatMulV2': {'24': 3}} | 31.707 | 238.107 | completed |
| 910B2 | 768 | 1280→3840 | eager | False | nz/None | {} | NA | NA | unsupported_configuration |

## Host/device context and limitations

[HOST_CONTEXT.md](HOST_CONTEXT.md) reports before/after snapshots for all 62 launches, including capture and the retained failed attempt. The host reports 192 logical CPUs and an affinity count of 192. Other NPU jobs were present; calibration and full-vision validation also overlapped on different cards. These facts are retained rather than assuming an idle host. Raw process-name/PID snapshots are in every receipt, but `ps` sees the launcher container namespace rather than all host CPU jobs. System load and the NPU occupancy table are host/device-wide; other-container CPU job identities were not captured. NPU process names are sometimes blank and are labelled unavailable rather than guessed.

Chosen cards report no health errors in the snapshots. Power and temperatures are recorded. AI-core clock was not exposed by the queried interfaces; work-mode reports unsupported, and the common query rejects this device/argument combination. Those are missing telemetry, not healthy clock/mode measurements. Pre/post snapshots cannot rule out throttling during the timed interval. Server timestamps differ from the authoring host; timestamps are preserved as recorded, not repaired.

## Failures, skips and omissions

- The initial compiled calibration attempt at af0dd464 failed before timing: `ValueError: Only method can be cached now`. The wrapper was corrected to pass a bound method in 9d153d90. The failed receipt/log/result remain under `matmul/M768_qkv_compiled_nd_internal_on`; its complete replacement is under `matmul_compiled_retry/`. The initial two successful eager cases were retained without rerunning them.
- Both NZ requests with internal formats off are explicitly `unsupported_configuration`. No implicit ND fallback is used.
- No 310P inference was performed here; the production command/config and 310P clocks/kernel behavior still require the receiving server.
- No full-vision FP32 reference or OCR-quality test was run. The original unpad layer-0/full-encoder drift remains explicitly unexplained.
- Historical per-lane host load/jobs for the original twelve-lane run were not recoverable and were not fabricated or replaced by these later measurements. The original conclusions remain unchanged, with the requested host-bound/scope/drift qualifications.
- No additional large internal-format-off calibration matrix was needed to validate the flag: eager/compiled qkv cases and both full-vision crop routes exercised it. The handoff requires the complete off/ND matrix if that matches 310P production.

## Reproduction and evidence

`raw_evidence.tar.gz` contains 648 files: immutable per-launch receipts (command/before/after/exit), logs, results, capture manifest/config, shell drivers, and processed kernel/operator/profile-step CSVs. Binary capture tensors, model weights, raw profiler databases/traces and compile caches are excluded from Git and remain on the server. Its SHA-256 was verified against the server copy:

```text
bd77188bd187e0b90b3554d8d49f60f92a4de015909debc6dd0d8d4683ca46fe
```

Recreate the generic tables after extracting into a new directory:

```bash
python3 11_mineru_2_5_pro_inference/analyze_vision_diagnostics.py \
  --run-dir /path/to/extracted/evidence --profile-forwards 3 \
  --output /path/to/extracted/evidence/diagnostic_analysis.json
```

The analyzer emits 62 records: 50 completed calibration cases, 8 completed vision lanes, one completed capture launch, two explicit unsupported calibration cases and one failed original attempt. Local portable tests passed (3); the original vision bookkeeping tests passed on 910B (3). The same analyzer reproduces the older 12-lane archive within 1e-6 ms for total, attention and matmul durations.

Full server run root:

```text
/workspace/repos/paddle_ocr_vl_npu_mineru_kv_probe/tmp/11_mineru_2_5_pro_inference/vision_diagnostics_validation_910B_20261007T125040Z_af0dd464
```
