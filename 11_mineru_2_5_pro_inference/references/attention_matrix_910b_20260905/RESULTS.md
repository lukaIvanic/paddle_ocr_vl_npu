# MinerU attention ablations: 910B2, 2026-09-05

## Scope and validation

Main matrix commit `6c7238d3`, physical 910B2 NPU 4. Exact inputs, rotary
factors, masks and baseline features were captured from the first 16 real
production pages. The model and original static vision implementation were
unchanged. The diagnostic reuses the production forward bytecode in an isolated
namespace and replaces only its attention helper; existing D128 padding is used.

Routes: direct S768 (640 real rows), packed S768 (480+192 real rows), direct
S5632 (5476 real rows). Padding remains its own independent full-attention
component: 128, 96 and 156 rows respectively. Mask validation checks every entry
before replacing it by unpadded sequence lengths. No synthetic substitute or
resized warmup input was used.

15 runnable cases completed; six approximate-mode cases were recorded as
310P-only and skipped before model load/compile. No claim of a 910B
approximate-softmax execution is made. All completed cases were finite and
repeat-exact. There was no model-default change or downstream OCR accuracy run.

Each main timing has 10 warm unprofiled samples. Profiles have three complete
forwards (96 attention kernels), normalized using duration_us, not aicore_time.
The S768 baseline has one 26.2-ms outlier, so its mean/p99 are not a stable tail
estimate. Raw samples and parity statistics are retained in `results.json`.

## Warm full-block latency

Mean device-event milliseconds, all 32 production vision blocks. Events can
include host launch gaps; raw_eager includes per-layer dispatch and metadata
handling. Patch embedding, H2D, merger, text/decode and model/cache setup are out
of scope. These are not page-throughput measurements.

| Route | Compiled D80 | Compiled D128 | Eager PromptFA | Eager unpad D80 | Eager unpad D128 |
|---|---:|---:|---:|---:|---:|
| Direct S768 | 18.120 | 20.321 | 50.985 | 47.286 | 51.700 |
| Packed S768 | 18.255 | 19.061 | 62.878 | 44.690 | 50.137 |
| Direct S5632 | 105.071 | 112.109 | 116.388 | 146.502 | 129.496 |

Baseline graph first-call/cache-loading times were 4.825–5.371 s. New D128
graph first calls took 40.0–45.5 s. All are excluded from warm timings.
Existing baseline cache roots were retained. Candidate graphs used separate
cache identities. Each lane ran in a fresh process, sequentially on NPU 4.

## Attention kernel time, summed across 32 layers

| Route | Compiled PFA D80 | Compiled PFA D128 | Unpad D80 | Unpad D128 |
|---|---:|---:|---:|---:|
| Direct S768 | 2.644 ms | 2.822 ms | 2.634 ms | 2.191 ms |
| Packed S768 | 2.699 ms | 2.730 ms | 2.460 ms | 1.967 ms |
| Direct S5632 | 54.806 ms | 56.223 ms | 80.961 ms | 55.845 ms |

The ATB kernel reports `UnpadFlashAttentionNdKernel`; PFA reports
`PromptFlashAttention`. Each row uses 96 observed calls divided by three.

D128 improves the unpad S5632 kernel by 31.0% versus unpad D80. It does not
beat current PromptFA there. The eager unpad full encoder also has more
layout/padding and unfused work, so it is not evidence for the performance of
an optimized compiled/eager hybrid. At small shapes its event interval exceeds
summed device kernels substantially: host submission matters.

## Numerical findings

Compiled D128 and eager PromptFA are bit-exact against the captured baseline
on all three routes. Unpad D80/D128 have the following full-encoder drift on
real rows (both variants report the same metrics):

| Route | Relative L2 | Mean absolute | Max absolute | Cosine |
|---|---:|---:|---:|---:|
| Direct S768 | 0.0106754 | 0.0159018 | 252 | 0.999943 |
| Packed S768 | 0.00368429 | 0.0104021 | 72 | 0.999993 |
| Direct S5632 | 0.00671001 | 0.0134967 | 264 | 0.999978 |

Same-input first-layer relative L2 is
0.0002443, 0.0002056 and 0.0003018 respectively. Small first-layer differences
therefore propagate; finite output or high cosine is not sufficient to approve
OCR quality. The elementwise 0.05/0.05 allclose diagnostic fails for all three
unpad full-encoder comparisons.

## Final harness gate

Commit `7ced563a` tightened baseline exactness and deterministic-replay gates,
recorded all vision projection formats, and made deadline cleanup apply to the
lane's own process group. A final 30-sample S5632 baseline/unpad-D128 run on
physical NPU 4 passed both strict gates without a new graph compilation:

- baseline 106.025 ms mean / 105.998 ms p50;
- unpad D128 129.578 ms mean / 129.547 ms p50;
- all 128 vision projection weights observed as native format 2 in both lanes;
- baseline bit-exact; unpad relative L2 0.00671001; both repeat-exact and finite.

Nine CPU contract tests passed; these cover mask component extraction and
packing semantics, not NPU inference. Real NPU executions above are the hardware
validation. The approximate converter mechanism is the previously owned Paddle
lab path; its actual mode4 execution remains to be tested on 310P.

## Evidence locations

Compact checked-in main matrix: `results.json` beside this report.
Remote detailed results, command records, profiles and captured tensors:

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_matrix_910b_r1/
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_matrix_910b_final_gate/
/tmp/mineru_attention_matrix_910b_r1.log
/tmp/mineru_attention_final_gate.log
```

The 310P handoff must keep these as contextual 910B data, not expected speedups.
Its mode4 lanes require feature checks and subsequent OCR quality validation
before any promotion. Do not infer that 310P will follow the 910B ranking.
