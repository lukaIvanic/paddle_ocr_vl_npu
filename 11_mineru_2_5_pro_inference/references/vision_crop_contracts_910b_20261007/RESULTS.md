# Real-crop full MinerU vision: 910B2, 2026-10-07

This is the vision investigation. The separate full-model KV benchmark measures
text decode and does not diagnose Luka's reported 310P vision bottleneck.

Source: `82575a912a6e9b1ab38233bfbf91c059c6111475`, physical NPU 3,
Ascend910B2, FP16, all 32 vision blocks. Twelve lanes completed successfully,
each with 30 warm unprofiled forwards and three separately profiled forwards.
Source, versions, raw results and available occupancy/exit records are retained.
The reconstructed top-level command file has a documented correction; see
[RECEIPT_NOTE.md](RECEIPT_NOTE.md) before treating it as launch evidence.

## Real inputs and measurement boundary

The actual repository crops were processed with the slow checkpoint image
processor at max_pixels=602112, then passed through patch embedding, positions,
the compiled blocks and merger. The captured inputs/rotary/masks/features come
from those real forwards, with model/image/tensor hashes in `capture_manifest.json`.

- `hotswap_001_code_txt_p0001_box_id_3.png`: grid [1,24,30], 720 real raw tokens, bucket768, attention components [720,48], merger output [180,896].
- `hotswap_002_code_txt_p1474_11.png`: grid [1,46,66], 3036 real raw tokens, bucket3072, attention components [3036,36], merger output [759,896].

These are recognition crops, not MinerU's own full-page layout image. Each warm
sample executes the complete 32-block stack; useful tokens count the real raw
patch tokens once per complete forward, excluding filler. Patch embedding,
initial position preparation, merger, CPU processing/H2D, model load, format
conversion and compile/cache loading are outside the replay timer. This is the
production vision-block boundary, not complete crop or page throughput.

## Full-encoder warm performance

Mean synchronized wall milliseconds and useful raw vision tokens/s. NPU-event
samples, percentiles and physical padded-token rates remain in each result JSON.
The two crops are separate diagnostics; these are not corpus-average rates.

| Complete vision path | 720 tokens: wall ms | Useful tok/s | 3036 tokens: wall ms | Useful tok/s |
|---|---:|---:|---:|---:|
| Compiled PromptFA D80 / ND weights | 17.037 | 42,262 | 48.513 | 62,581 |
| Compiled PromptFA D80 / NZ weights | 18.712 | 38,477 | 50.797 | 59,768 |
| Compiled PromptFA D128 / ND weights | 18.942 | 38,010 | 51.816 | 58,592 |
| Eager PromptFA D80 / ND weights | 57.512 | 12,519 | 58.658 | 51,758 |
| Eager stock-op unpad D128 / ND weights | 53.526 | 13,451 | 69.760 | 43,521 |
| Eager stock-op unpad D128 / NZ weights | 55.246 | 13,033 | 69.981 | 43,383 |

Compare weights within the same path. The small-crop eager lanes are
**host-bound** and are not an unpad-versus-PromptFA performance comparison.
Their raw measurements remain visible above; do not report a relative speedup
or slowdown between those small-crop eager paths. Larger-crop operator-path
comparisons use the eager controls. A compiled-versus-eager ratio includes fusion and dispatch differences.
CPU sequence lengths for unpad are prepared once before replay; the op's
per-layer CPU metadata consumption remains timed. This is the actual stock
operator contract inside our vision stack, not stock vLLM engine throughput or
an exact reproduction of its per-layer `torch.diff(...).to("cpu")` wrapper.

## Profile attribution: larger crop

All values use kernel Duration(us) summed over three full forwards and divided
by 3000. Every capture has 96 attention kernels and 384 projection matmuls.
These are summed kernel durations, not PMU engine times or an independent
measurement of elapsed forward time. Remaining kernels include rotary, manual
FP32 normalization, layout, padding, activation and residual work; mixed fusions
are retained by type rather than assigned to a guessed semantic category.

| Path | Attention ms/forward | Linears ms/forward | Remaining ms/forward | Kernel sum ms/forward |
|---|---:|---:|---:|---:|
| Compiled PromptFA D80 / ND weights | 16.781 | 13.639 | 17.770 | 48.190 |
| Compiled PromptFA D80 / NZ weights | 16.695 | 15.300 | 18.569 | 50.564 |
| Compiled PromptFA D128 / ND weights | 16.804 | 13.665 | 20.848 | 51.317 |
| Eager PromptFA D80 / ND weights | 16.839 | 14.423 | 26.791 | 58.052 |
| Eager stock-op unpad D128 / ND weights | 19.294 | 14.380 | 35.842 | 69.516 |
| Eager stock-op unpad D128 / NZ weights | 19.327 | 14.171 | 36.223 | 69.721 |

Compiled baseline attention is 34.8% of summed kernel duration on the larger
crop and 15.7% on the smaller crop. NZ compiled weights increase linear time
from 13.639 to 15.300 ms on the larger crop. D128 PromptFA attention itself is
nearly unchanged there, while the complete padded encoder is slower.

Small-crop eager kernel sums are 25.534 ms (PromptFA) and 31.029 ms (unpad),
versus unprofiled wall means of 57.512 and 53.526 ms. This is consistent with
host-bound execution; these are different profiled/unprofiled windows,
so their subtraction is not an independently measured host-time bucket.

## Actual formats and feature differences

Every native lane stores all 128 block projections in format2; every NZ lane
retains format29, with exact logical values verified after conversion. All 384
profiled matmuls consume ND weights in native lanes and FRACTAL_NZ weights in
NZ lanes. No separate TransData kernel was observed. This does not establish
absence of internal packing. Compiled large-crop native matmuls include V3;
the NZ compiled lane uses V2, so format selection also changes kernel selection.

**“Stock vLLM-Ascend” here means its stock op inside our eager stack, not the
vLLM-Ascend vision path.** The private op executes `UnpadFlashAttentionNdKernel` with ND Q/K/V tensors
[S,16,128], both with ND and NZ projection weights. Vision has no persistent
KV cache; a weight-format change does not turn these fresh activations into
an autoregressive NZ cache. The source audit of vLLM-Ascend is pinned in
`environment.json` to revision 80610e4438dba05011b05f89fc45d91e96992671.

Timing is retained for every finite deterministic candidate. Feature drift is
a separate measurement; no candidate throughput was hidden for failing exact
feature parity. Every baseline replay is bit-exact with its production capture,
and every lane has zero new graphs/recompilation warnings during warm timing.

| Path | Relative L2, 720 tokens | Relative L2, 3036 tokens | Cosine, 3036 tokens |
|---|---:|---:|---:|
| Compiled PromptFA D80 / ND weights | 0.000000 | 0.000000 | 1.000000 |
| Compiled PromptFA D80 / NZ weights | 0.007526 | 0.012792 | 0.999918 |
| Compiled PromptFA D128 / ND weights | 0.000000 | 0.000000 | 1.000000 |
| Eager PromptFA D80 / ND weights | 0.000000 | 0.000000 | 1.000000 |
| Eager stock-op unpad D128 / ND weights | 0.007199 | 0.021529 | 0.999768 |
| Eager stock-op unpad D128 / NZ weights | 0.007199 | 0.021529 | 0.999768 |

**Unpad drift is unexplained.** On the larger crop its layer-0 relative L2 is
about 2.8e-4, but full-encoder relative L2 reaches 0.021529 (2.15%). No FP32
reference investigation has been performed, so this cannot be attributed to
benign accumulation or approved as OCR-quality equivalence. Timing remains
reported independently.

Max/mean absolute errors and allclose diagnostics are retained in the raw
results; no downstream OCR quality run or production-default change was made.

## What this establishes

The harness now measures actual full-stack vision. On these 910B inputs,
NZ vision weights and D128 padding do not improve the compiled full encoder.
The stock unpad D128 eager path is slower than eager PromptFA on the larger
crop. These results do not establish the 310P ranking or explain its reported
3.5k rate. The matched 310P run must compare attention, linears and the remaining
kernels, with runtime differences recorded. The 310P-only approximate-precision
mode has no 910B execution claim.

Use [the vision-specific 310P handoff](../../VISION_CROP_310P_HANDOFF.md), which
discovers that server's environment and prohibits tracked source edits.

## Host context and receipt limitations

[Historical host context](HISTORICAL_HOST_CONTEXT.md) lists missing per-lane
load, CPU-count, other-job and device-state records. They are not reconstructed
from later snapshots. Small-crop eager comparisons are host-bound as described
above. New runs collect these fields before and after every lane.

## Evidence

- `analysis.json`: normalized types/counts/durations, actual formats, timing and feature summaries.
- `matrix/*/result.json` and `padding_matrix/*/result.json`: every warm sample, gate, format and feature result.
- `raw_evidence.tar.gz`: exact commands/logs/exit codes, occupancy, capture manifest, processed profiles, operator/kernel CSVs and environment receipts. Raw NPU profiler databases, binary input captures and compiler caches are excluded.
- `analyze_910b_evidence.py`: stdlib validation of extracted receipts; run with extracted directory and output JSON as its two arguments.

Full captures and compiler artifacts remain on the 910B container at:

```text
/workspace/repos/paddle_ocr_vl_npu_mineru_kv_probe/tmp/11_mineru_2_5_pro_inference/vision_crop_contracts_910B_20261007T115828Z_82575a91/
```
