# Exact d4e7fdd, full live cap3072 reproduction — 910B2

Inference source: `d4e7fdd0efd90bf6408c21efab320cc3292f07a4`, clean detached
checkout. Original production inference files were not modified. Runtime:
torch 2.10.0+cpu, torch-npu 2.10.0, Transformers 5.5.4,
mineru-vl-utils 1.0.5. This matches the retained higher-cap 910B reference's
package versions. Device: physical NPU 3, healthy and empty before launch.

## Actual full-model result

All figures in this document are **910B**, unless explicitly labelled 310P.
The input is the 1,651 OmniDocBench page images, with fresh live PP-DocLayoutV3
layout and MinerU recognition. PDF rendering and image/chart analysis are off,
as in the original handoff. This is the custom engine, not vLLM-Ascend.

| 910B original precision | Measured result |
|---|---:|
| Completed / failed / skipped pages | 1,651 / 0 / 0 |
| Live layout calls / recognition crops | 1,651 / 32,051 |
| Pipeline wall | 1,685.602498 s |
| E2E throughput | **0.979472 pages/s** |
| Setup, separately | 30.293181 s |
| Generated tokens, including EOS | 1,804,363 |
| Actual input vision tokens | 16,809,232 |
| Physical vision positions | 18,887,040 |
| Extra padding positions | 2,077,808 |
| Padding / physical positions | **11.0012%** |
| Padding / actual tokens | 12.3611% |
| Full 32-block vision event time | 378.759687 s |
| Actual vision tokens / vision event time | **44,379.7 tokens/s** |

32,025 crops ended at EOS; **26 reached their original length limits**:
10 equations, 10 text regions, 4 tables and 2 footers. Their outputs are retained
and included in the quality evaluation, without retries or changed generation
limits. These are crop completion limits, separate from the zero page failures.
See `generation_accounting.json` and the raw generation trace.

## Full-run accuracy

The frozen evaluator commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`
completed successfully on all 1,651 pages. It scored 665 table samples and
2,352 formula samples, with **zero** page-match fallback timeouts, metric
timeouts, errors or exceptions. Predictions were not rewritten for evaluation.

| 910B full live lower-cap metric | Score |
|---|---:|
| Text accuracy, page aggregation | 96.234624482 |
| Table TEDS, page aggregation | 93.520374665 |
| Formula CDM, page aggregation | 97.054050863 |
| Overall, arithmetic mean | **95.603016670** |

These are the full live run's own scores. Older selective-replay scores remain
unchanged in their original evidence. The new scores' near agreement does not
establish per-crop token parity with that older run.

Useful vision throughput divides the sum of actual crop-input tokens by the sum
of full-encoder event regions. These regions include launch gaps and first-use
cached-graph loads, and direct routes include padding preparation. Packed mask
construction is outside the event. This is not isolated kernel timing. All
first-use graph loads remain inside the production pipeline wall; no warmup
pages or after-the-fact subtraction were used.

## Exact strategy

FP16 MinerU; fast image processor, min_pixels=25,088, max_pixels=602,112.
Compiled 32-block vision, native D80 PromptFA in BNSD, full boolean component
mask, sparse_mode=1, stock GE innerPrecise=1, manual FP32 vision LayerNorm,
ordinary linear projections with native ND parameters, internal formats enabled.
Layout is eager NPU FP32 PP-DocLayoutV3 with graph capture off.

Nine vision buckets are configured: 384, 512, 768, 1024, 1536, 2048, 3072,
4224, 5632. The cap makes only the first **seven** reachable. Small crops are
sorted by descending length within lookahead 32 and first-fit packed to a
768-token target. Multiple crops share a B1 sequence, with mask components
isolating each crop and padding. Singletons use the smallest fitting bucket;
larger crops also run individually in the smallest fitting bucket. No actual
crop tokens are discarded by batching: padding adds empty positions.

Arithmetic on the measured per-crop lengths gives 22,742,912 physical positions
if each crop ran separately in its smallest fitting bucket, versus the measured
18,887,040 with packing. Packing therefore avoids **3,855,872 positions
(16.9542%)** relative to that counterfactual. This is a token-position count,
not a benchmark of separate-crop execution or a latency-speedup claim. Padding
percentage alone also does not quantify attention's total computation.
See `packing_accounting.json`.

Packed text prefill uses buckets 128, 256, 512, 1024, max 32 members. Continuous
decode uses B32, KV4096, IncreFA pse_sentinel_310p, NZ decoder weights and
npu_apply rotary. Page window 32, CPU prepare depth 64, vision lookahead 32.
The decode mode's name does not make this a 310P measurement.

## Actual full-workload vision distribution

Bins below use the **sum of useful tokens in each vision call**. Packed rows
contain multiple crops; their useful lengths are not individual crop lengths.
Means include first-use outliers. Rates are sum(tokens)/sum(seconds).

| 910B useful tokens / kind | Calls / crops | Vision time share | Mean / p50 / p99 ms | Useful tok/s |
|---|---:|---:|---:|---:|
| 128–384 single | 231 / 231 | 1.18% | 19.278 / 14.488 / 14.536 | 8,025.5 |
| 256–384 packed | 198 / 400 | 0.92% | 17.546 / 17.545 / 17.594 | 16,352.6 |
| 392–504 single | 58 / 58 | 0.49% | 31.946 / 15.532 / 425.234 | 13,520.5 |
| 388–512 packed | 198 / 585 | 0.92% | 17.540 / 17.544 / 17.581 | 24,741.8 |
| 520–768 single | 1,544 / 1,544 | 8.02% | 19.679 / 17.685 / 26.171 | 35,419.5 |
| 516–768 packed | 7,955 / 23,480 | 37.37% | 17.792 / 17.543 / 18.587 | 40,588.0 |
| 776–1024 single | 1,668 / 1,668 | 9.69% | 22.002 / 20.944 / 21.009 | 40,479.9 |
| 1032–1536 single | 1,503 / 1,503 | 10.79% | 27.201 / 26.436 / 26.514 | 45,731.3 |
| 1540–2048 single | 761 / 761 | 7.23% | 36.006 / 34.517 / 34.588 | 49,084.1 |
| 2064–3072 single | 1,821 / 1,821 | 23.39% | 48.656 / 48.003 / 48.059 | 57,387.3 |

## Environment and limits of comparison

Full-run before/after 1/5/15-minute loads were
24.55/25.94/26.01 and 32.28/32.43/30.86, with 192 visible CPUs. Other devices
were occupied; their process table is in the immutable snapshots. The selected
card reported no device errors. Clock data was not exposed by the successful
queried interfaces; raw power/sensor/work-mode responses are retained.

The retained higher-cap 910B run used max_pixels=1,103,872 (5,632 raw tokens),
not 1.6M. Its 19,114,444 actual vision tokens versus this run's 16,809,232 is
**12.0601% fewer** tokens. Its 0.932911 pages/s versus this run's 0.979472 is a
historical comparison, not an interleaved controlled timing pair. The former
0.993678 lower-cap estimate remains an estimate; this run is the actual measured
lower-cap full-page result. No original receipt for the user's rounded **310P
0.289 pages/s** claim has been recovered; this run cannot confirm that history.

## Reduced-precision integration check

The authoring harness is `run_page_pipeline_vision_precision.py` at
`0e794023`, importing production inference from the untouched d4 checkout.
It reuses the existing converter in
`09_persistent_page_engine/scripts/vision_matmul_lab.py`, scopes the override to
the first lowering of each vision graph, and restores the original converter
in a finally block. No different precision method was substituted.

| 910B validation | Outcome |
|---|---|
| Production fast-processor real-crop selection | All seven reachable buckets found |
| Original-precision real-crop prewarm | All seven passed; finite model features |
| Mode4 live two-page compatibility attempt | Failed at first vision graph, bucket 768 |
| Audited vision PromptFA lowerings | 32; all BNSD, sparse_mode=1, innerPrecise=4 |
| CANN causal error | `not support APPROXIMATE_COMPUTATION when curShortSocName is Atlas A2` |
| Causal source location | `prompt_flash_attention_tiling.cpp:3924` |
| Reduced-precision throughput / output comparison | Unavailable: the graph did not execute |

This is a demonstrated installed-stack limitation on 910B, not a speed result.
No 310P run occurred here. The updated 310P handoff requires a mode4 seven-bucket
prewarm, smoke, full live run and accuracy evaluation; that full integration is
still unvalidated. The compatibility attempt has cold compilation inside its
wall time and must never be treated as an E2E throughput comparison.

## Evidence

`original_raw_evidence.tar.gz` contains untouched smoke/full commands,
before/after snapshots, process/exit receipts, summaries, raw per-call vision
timings, all 32,051 generation traces, crop selection and derived distribution.
`run_summary_shard_00.json`, `command.json`, `before.json`, `after.json`,
`exit.json`, and `vision_distribution.json` are convenient unmodified copies.
See `RECEIPT_NOTE.md` for the corrected coordinator hash check and the retained
derived gate's cosmetic source-key mistake.

`page_raw_evidence.tar.gz` adds all page predictions/content lists, layout
metadata and smoke/full inference logs. `precision_raw_evidence.tar.gz` keeps
the real-crop prewarm and failed mode4 attempt, including all GE precision rows,
the exact error and immutable launch/device/load snapshots.
`precision_validation.json` is a derived summary checked against those audits.
`evaluation_raw_evidence.tar.gz` keeps the frozen evaluator receipts/runtime
verification, prep manifest, raw scores, per-sample results and error accounting.
The convenient score/stage copies and `accuracy_summary.json` retain the exact
main scores, auxiliary metrics and stage counts. `SHA256SUMS` covers all four
raw archives.

Reproduce the original full-model distribution with the committed reporter:

```bash
REPO="$(git rev-parse --show-toplevel)"
EVIDENCE_ROOT="$(mktemp -d)"
tar -xzf original_raw_evidence.tar.gz -C "$EVIDENCE_ROOT"
tar -xzf page_raw_evidence.tar.gz -C "$EVIDENCE_ROOT"
python3 "$REPO/11_mineru_2_5_pro_inference/report_e2e_vision_precision.py" \
  --original "$EVIDENCE_ROOT/full1651" --chip 910B \
  --output "$EVIDENCE_ROOT/reproduced.json"
cmp vision_distribution.json "$EVIDENCE_ROOT/reproduced.json"
```

Run from this evidence directory. The
reproduced JSON was compared with the original and matched exactly.

Remote run root:
`/workspace/repos/paddle_ocr_vl_npu_mineru_lengths/tmp/11_mineru_2_5_pro_inference/d4_live_cap3072_910B_20261008T083246Z`.
