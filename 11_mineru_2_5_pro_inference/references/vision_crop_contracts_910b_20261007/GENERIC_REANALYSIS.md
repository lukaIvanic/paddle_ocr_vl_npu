# Additive generic reanalysis — 2026-10-07

The portable analyzer was run on a fresh extraction of the committed raw_evidence.tar.gz. All 12 lanes match the existing analysis.json total, attention and linear kernel duration sums within 1e-6 ms. The original receipts, archive and analysis.json are unchanged. This adds fields previously omitted from the report; it does not replace the original 910B conclusions.

All durations and wait sums below are **milliseconds per full encoder forward**, dividing the three-forward profile by three. Block/Mix/Core histograms count calls across all three forwards. Wait is a separate CSV counter, not a causal host-time attribution, and must not be added to kernel duration as elapsed time. Block Num is a launch-grid indicator, not a utilization measurement.

| 910B lane / route | Kernel type | Kernel ms | Wait ms | Block Num histogram | Mix Block Num | Accelerator Core |
|---|---|---:|---:|---|---|---|
| baseline / crop_0_bucket_768 | MatMulV2 | 4.537323 | 0.006800 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| baseline / crop_0_bucket_768 | PromptFlashAttention | 2.658526 | 0.038891 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| baseline / crop_0_bucket_768 | TOTAL WAIT, separate | — | 0.455995 | — | — | — |
| baseline / crop_1_bucket_3072 | MatMulV2 | 2.116970 | 0.001649 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| baseline / crop_1_bucket_3072 | MatMulV3 | 11.522336 | 0.005075 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| baseline / crop_1_bucket_3072 | PromptFlashAttention | 16.780939 | 0.037281 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| baseline / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 0.438802 | — | — | — |
| eager_pfa / crop_0_bucket_768 | MatMulV2 | 4.735054 | 7.022519 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| eager_pfa / crop_0_bucket_768 | PromptFlashAttention | 2.823016 | 10.863005 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| eager_pfa / crop_0_bucket_768 | TOTAL WAIT, separate | — | 51.907049 | — | — | — |
| eager_pfa / crop_1_bucket_3072 | MatMulV2 | 2.217304 | 0.002039 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| eager_pfa / crop_1_bucket_3072 | MatMulV3 | 12.205444 | 0.099965 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| eager_pfa / crop_1_bucket_3072 | PromptFlashAttention | 16.838503 | 10.956439 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| eager_pfa / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 17.800649 | — | — | — |
| pfa_nz_weights / crop_0_bucket_768 | MatMulV2 | 5.250357 | 0.007542 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| pfa_nz_weights / crop_0_bucket_768 | PromptFlashAttention | 2.687980 | 0.038560 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| pfa_nz_weights / crop_0_bucket_768 | TOTAL WAIT, separate | — | 0.498490 | — | — | — |
| pfa_nz_weights / crop_1_bucket_3072 | MatMulV2 | 15.299860 | 0.007870 | {'24': 288, '21': 96} | {'0': 384} | {'AI_CORE': 384} |
| pfa_nz_weights / crop_1_bucket_3072 | PromptFlashAttention | 16.695467 | 0.037115 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| pfa_nz_weights / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 0.488193 | — | — | — |
| unpad_d128 / crop_0_bucket_768 | MatMulV2 | 4.616642 | 7.994650 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| unpad_d128 / crop_0_bucket_768 | UnpadFlashAttentionNdKernel | 2.376020 | 5.159329 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| unpad_d128 / crop_0_bucket_768 | TOTAL WAIT, separate | — | 54.206743 | — | — | — |
| unpad_d128 / crop_1_bucket_3072 | MatMulV2 | 2.200511 | 0.033820 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| unpad_d128 / crop_1_bucket_3072 | MatMulV3 | 12.179771 | 0.227332 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| unpad_d128 / crop_1_bucket_3072 | UnpadFlashAttentionNdKernel | 19.293505 | 5.022689 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| unpad_d128 / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 17.848239 | — | — | — |
| unpad_d128_nz_weights / crop_0_bucket_768 | MatMulV2 | 4.410976 | 8.043127 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| unpad_d128_nz_weights / crop_0_bucket_768 | UnpadFlashAttentionNdKernel | 2.451000 | 5.318254 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| unpad_d128_nz_weights / crop_0_bucket_768 | TOTAL WAIT, separate | — | 50.110997 | — | — | — |
| unpad_d128_nz_weights / crop_1_bucket_3072 | MatMulV2 | 2.297387 | 0.001461 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| unpad_d128_nz_weights / crop_1_bucket_3072 | MatMulV3 | 11.873884 | 0.144654 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| unpad_d128_nz_weights / crop_1_bucket_3072 | UnpadFlashAttentionNdKernel | 19.326912 | 4.720870 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| unpad_d128_nz_weights / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 14.838010 | — | — | — |
| pfa_d128 / crop_0_bucket_768 | MatMulV2 | 4.494101 | 0.005759 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| pfa_d128 / crop_0_bucket_768 | PromptFlashAttention | 2.758663 | 0.087999 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| pfa_d128 / crop_0_bucket_768 | TOTAL WAIT, separate | — | 0.725570 | — | — | — |
| pfa_d128 / crop_1_bucket_3072 | MatMulV2 | 2.114163 | 0.001859 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| pfa_d128 / crop_1_bucket_3072 | MatMulV3 | 11.550877 | 0.005140 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| pfa_d128 / crop_1_bucket_3072 | PromptFlashAttention | 16.803776 | 0.091513 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| pfa_d128 / crop_1_bucket_3072 | TOTAL WAIT, separate | — | 0.758448 | — | — | — |

The compressed [generic JSON](generic_analysis.json.gz) retains every attention/matmul call, all type aggregates, HF32 eligibility, shapes, input/output formats, conversion directions, unclassified remaining types and per-stream observed gaps. No separate format-conversion kernels were found in this archive. That does not rule out repacking inside another kernel. Historical per-lane telemetry remains unavailable as documented in HISTORICAL_HOST_CONTEXT.md.

Reproduce from the repository root:

```bash
CHECK=$(mktemp -d)
tar -xzf 11_mineru_2_5_pro_inference/references/vision_crop_contracts_910b_20261007/raw_evidence.tar.gz -C "$CHECK"
python3 11_mineru_2_5_pro_inference/analyze_vision_diagnostics.py \
  --run-dir "$CHECK" --profile-forwards 3 --output "$CHECK/generic_analysis.json" \
  --compare-analysis 11_mineru_2_5_pro_inference/references/vision_crop_contracts_910b_20261007/analysis.json
```
