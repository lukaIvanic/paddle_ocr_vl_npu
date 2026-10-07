# MinerU real-crop vision ND/NZ pairs — Ascend910B2, 2026-10-07

All 24 lanes completed at source `a55d24df` on physical NPU 2. Each lane measures all 32 production vision blocks on the same saved real OmniDocBench crop inputs: 720 useful tokens in bucket 768, or 3,036 in bucket 3,072. This excludes preprocessing, transfers, patch embedding, merger, text prefill/decode and page parsing. These are vision-block throughput results, not end-to-end page throughput.

FP16; internal formats enabled in BOTH ND and NZ controls before NPU configuration; manual FP32 LayerNorm; approximate precision off. Each lane performs 30 warm timed forwards, with 3 additional profiled forwards outside throughput timing. All pairs use matching capture/model hashes, configurations, execution modes and input tags. Both baselines match their original production capture bit-exactly. Every lane is finite, repeat-exact and reports zero new graphs/recompile warnings inside timing. Conversion and cold compilation are outside the timer.

NZ conversion uses the locally validated `torch_npu.npu_format_cast(weight, 29)` pattern before compilation. All 128 linear parameters report format 29; each conversion passes a bit-exact logical round trip. Crucially, profiler input formats are checked separately. Ordinary linear variants actually consume NZ. The unchanged grouped helper converts its transposed weight back to ND; those rows are explicitly mixed-format and do not establish all-NZ grouped performance.

Method references are the working precompile conversion helpers in experiment 09
(`paddleocr_vl/model/vision_prefill.py`), experiment 13
(`local_modeling_qwen3_reranker.py`), and experiment 12
(`modeling_optimized_unirec.py`), together with their retained run evidence.
The local GLM quantized grouped-NZ implementation was inspected but was not
substituted for MinerU's FP16 grouped contract.

The two original crop files are `hotswap_001_code_txt_p0001_box_id_3.png`
and `hotswap_002_code_txt_p1474_11.png`; their image and capture hashes are in
`source_capture_manifest.json`. Capture was originally made on 910B physical
card 3; this rerun on card 2 reproduced both baseline captures exactly.

## Paired full-vision timings

Values are mean synchronized wall latency. Positive change means NZ is slower. These are sequential paired runs on a shared host, not randomized trials; differences near 1% do not establish a reliable improvement.

| Path | Useful tokens | ND ms | NZ-parameter ms | Change | ND useful tok/s | NZ useful tok/s | Actual NZ weight inputs |
|---|---:|---:|---:|---:|---:|---:|---|
| PromptFA D80 compiled | 720 | 17.527 | 18.727 | +6.85% | 41,080 | 38,446 | All 128 projections |
| PromptFA D80 compiled | 3036 | 49.984 | 50.487 | +1.01% | 60,740 | 60,135 | All 128 projections |
| PromptFA D128 compiled | 720 | 19.222 | 19.874 | +3.39% | 37,458 | 36,228 | All 128 projections |
| PromptFA D128 compiled | 3036 | 53.073 | 53.153 | +0.15% | 57,204 | 57,118 | All 128 projections |
| Grouped QKV compiled | 720 | 17.800 | 18.933 | +6.37% | 40,450 | 38,029 | 96/128; QKV uses ND |
| Grouped QKV compiled | 3036 | 50.172 | 51.969 | +3.58% | 60,512 | 58,419 | 96/128; QKV uses ND |
| Grouped QKV+FC1 compiled | 720 | 18.961 | 20.723 | +9.29% | 37,972 | 34,743 | 64/128; QKV and FC1 use ND |
| Grouped QKV+FC1 compiled | 3036 | 51.079 | 53.019 | +3.80% | 59,437 | 57,262 | 64/128; QKV and FC1 use ND |
| PromptFA D80 eager | 720 | 56.657 | 55.335 | -2.33% | 12,708 | 13,012 | All 128 projections |
| PromptFA D80 eager | 3036 | 58.832 | 59.750 | +1.56% | 51,605 | 50,812 | All 128 projections |
| Unpad D128 eager | 720 | 53.878 | 54.809 | +1.73% | 13,364 | 13,137 | All 128 projections |
| Unpad D128 eager | 3036 | 70.242 | 70.699 | +0.65% | 43,222 | 42,943 | All 128 projections |

Small-crop eager measurements are host-bound: PromptFA ND has 25.261 ms of summed kernels versus 56.657 ms wall; NZ has 25.590 ms versus 55.335 ms wall. They are retained as within-path ND/NZ observations; do not use them to compare unpad against PromptFA kernel performance. “Stock vLLM-Ascend” here means its stock unpad op inside our eager 32-block stack, not the complete vLLM-Ascend vision path. Grouped modes remain exploratory compile workarounds, not established optimizations.

## Numerical drift from the original capture

Timing is reported even when features differ. These checks are not downstream OCR quality. The pre-existing unpad full-encoder drift remains unexplained; no full FP32 reference was added.

| Path | Useful tokens | ND relative L2 | NZ relative L2 |
|---|---:|---:|---:|
| PromptFA D80 compiled | 720 | 0 | 0.0075260377 |
| PromptFA D80 compiled | 3036 | 0 | 0.012792135 |
| PromptFA D128 compiled | 720 | 0 | 0.0075260377 |
| PromptFA D128 compiled | 3036 | 0 | 0.012792135 |
| Grouped QKV compiled | 720 | 0.0078286314 | 0.0075260377 |
| Grouped QKV compiled | 3036 | 0 | 0.012792135 |
| Grouped QKV+FC1 compiled | 720 | 0.0094278334 | 0.0075260377 |
| Grouped QKV+FC1 compiled | 3036 | 0 | 0.012792135 |
| PromptFA D80 eager | 720 | 0 | 0 |
| PromptFA D80 eager | 3036 | 0 | 0 |
| Unpad D128 eager | 720 | 0.0071989368 | 0.0071989368 |
| Unpad D128 eager | 3036 | 0.021528931 | 0.021528931 |

## Kernel accounting and event timing

Each kernel bucket is the sum over 3 profiled forwards divided by 3. Wait Time is a separate profiler counter, not additive elapsed/host time. Full per-type Block Num, Mix Block Num, Accelerator Core, formats, waits, unclassified kernels and individual attention/matmul rows are retained in `diagnostic_analysis.json.gz`.

| Lane | Event ms | Attention ms | Matmul ms | Format conversion ms | Remaining kernels ms | Total kernels ms | Wait counter ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_crop_0_bucket_768 | 17.280 | 2.673 | 4.519 | 0.000 | 10.119 | 17.312 | 0.496 |
| baseline_crop_1_bucket_3072 | 49.728 | 16.807 | 13.649 | 0.000 | 18.982 | 49.437 | 0.862 |
| eager_pfa_crop_0_bucket_768 | 56.501 | 2.839 | 4.726 | 0.000 | 17.696 | 25.261 | 50.725 |
| eager_pfa_crop_1_bucket_3072 | 58.683 | 16.718 | 14.436 | 0.000 | 27.120 | 58.275 | 15.811 |
| eager_pfa_nz_weights_crop_0_bucket_768 | 55.174 | 2.651 | 4.572 | 0.000 | 18.368 | 25.590 | 49.179 |
| eager_pfa_nz_weights_crop_1_bucket_3072 | 59.598 | 16.787 | 14.198 | 0.000 | 27.666 | 58.652 | 21.268 |
| grouped_qkv_crop_0_bucket_768 | 17.643 | 2.664 | 4.400 | 0.000 | 10.486 | 17.551 | 0.562 |
| grouped_qkv_crop_1_bucket_3072 | 50.011 | 16.771 | 13.593 | 0.000 | 19.521 | 49.886 | 0.581 |
| grouped_qkv_mlp_fc1_crop_0_bucket_768 | 18.814 | 2.703 | 4.418 | 0.000 | 11.515 | 18.636 | 0.636 |
| grouped_qkv_mlp_fc1_crop_1_bucket_3072 | 50.879 | 16.709 | 13.304 | 0.000 | 20.589 | 50.602 | 0.686 |
| grouped_qkv_mlp_fc1_nz_weights_crop_0_bucket_768 | 20.580 | 2.712 | 5.297 | 1.353 | 11.093 | 20.455 | 0.632 |
| grouped_qkv_mlp_fc1_nz_weights_crop_1_bucket_3072 | 52.844 | 16.720 | 14.534 | 1.388 | 20.038 | 52.681 | 0.881 |
| grouped_qkv_nz_weights_crop_0_bucket_768 | 18.790 | 2.730 | 5.204 | 0.549 | 10.186 | 18.669 | 0.561 |
| grouped_qkv_nz_weights_crop_1_bucket_3072 | 51.811 | 16.822 | 14.974 | 0.608 | 19.318 | 51.722 | 0.590 |
| pfa_d128_crop_0_bucket_768 | 19.066 | 2.763 | 4.504 | 0.000 | 11.594 | 18.861 | 0.733 |
| pfa_d128_crop_1_bucket_3072 | 52.906 | 16.835 | 13.649 | 0.000 | 22.161 | 52.646 | 0.736 |
| pfa_d128_nz_weights_crop_0_bucket_768 | 19.706 | 2.815 | 5.238 | 0.000 | 11.394 | 19.447 | 0.739 |
| pfa_d128_nz_weights_crop_1_bucket_3072 | 52.987 | 16.817 | 15.236 | 0.000 | 20.696 | 52.750 | 0.780 |
| pfa_nz_weights_crop_0_bucket_768 | 18.566 | 2.687 | 5.269 | 0.000 | 10.493 | 18.449 | 0.601 |
| pfa_nz_weights_crop_1_bucket_3072 | 50.303 | 16.701 | 15.287 | 0.000 | 18.168 | 50.156 | 0.545 |
| unpad_d128_crop_0_bucket_768 | 53.716 | 2.401 | 4.621 | 0.000 | 23.915 | 30.938 | 46.474 |
| unpad_d128_crop_1_bucket_3072 | 70.094 | 19.153 | 14.392 | 0.000 | 35.849 | 69.393 | 9.100 |
| unpad_d128_nz_weights_crop_0_bucket_768 | 54.644 | 2.395 | 4.432 | 0.000 | 24.334 | 31.162 | 49.635 |
| unpad_d128_nz_weights_crop_1_bucket_3072 | 70.546 | 19.206 | 14.250 | 0.000 | 36.727 | 70.182 | 10.908 |

## Evidence and limitations

- No 310P run was performed; these results cannot establish its NZ performance or explain its 12–24× kernel gap.
- Both grouped variants fail the intended all-NZ operator-input condition despite successful execution and NZ parameters. Their existing weight preparation was preserved; no new packing algorithm or API contract was substituted.
- Approximate-precision mode is 310P-only and was not run on 910B. Unpad remains eager, matching the previous executable contract.
- No synthetic matmul calibration was substituted for full vision. Earlier isolated calibration evidence remains separate.
- No new full-page OCR or FP32-reference run; numerical drift and end-to-end quality are different questions.
- Before/after receipts record device health, system load, CPU counts, power, temperature and visible jobs for every lane. See `HOST_CONTEXT.md`; snapshots cannot rule out transient clock or scheduling effects.
- Original captures and their warmed baseline cache were reused, with content/model hash checks. Candidate cache keys include benchmark/runtime source, config, model hashes, environment version and device. The original baseline cache is deliberately preserved to check capture parity.

`raw_evidence.tar.gz` contains immutable launch receipts, logs, JSON results, copied capture manifest/config, launch driver and processed profiler CSVs. It excludes model/capture tensors, compile caches and raw profiler databases/traces. No historical receipt was rewritten. The matrix completes with exit code 0.

Run the existing generic analyzer on the extracted archive, then the adjacent paired analyzer:

```bash
python3 11_mineru_2_5_pro_inference/analyze_vision_diagnostics.py \
  --run-dir /path/to/extracted --profile-forwards 3 \
  --output /path/to/extracted/diagnostic_analysis.json
python3 11_mineru_2_5_pro_inference/references/vision_nz_pairs_910b_20261007/analyze_pairs.py /path/to/extracted
```

Archive SHA-256 (matched to the server):

```text
fefe9e64649869eb487f8ac6d571c823fa7dded6f5ccd2321a4fe140c51c6b3a
```

Server run root:

```text
/workspace/repos/paddle_ocr_vl_npu_vision_diagnostics_retry/tmp/11_mineru_2_5_pro_inference/vision_nz_pairs_910B_20261007T140703Z_a55d24df
```
