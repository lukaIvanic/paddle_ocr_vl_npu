# MinerU production vision timing on 910B2, first 384 pages

Run commit `9bc93294179af403117fbba92a3de37772e95e85`, 2026-09-05,
physical 910B2 NPU4. Torch 2.10.0+cpu / torch-npu 2.10.0.
The preserved `command.sh` and run summary are the exact invocation/settings.

384/384 pages completed, zero failures or skips. Hot wall time was
362.942790 s (1.058018 pg/s). The two-page/all-bucket warmup took 18.479726 s
and was excluded. All vision first calls restored existing graphs in
0.804–0.986 s. No replacement cache root was used.

The preset matches the preceding 1651-page 910B run and the 310P handoff:
FP16, B32/KV4096, 32-page global stream, prefetch64, packed text,
compiled manual-FP32 LayerNorm + nn.Linear vision, native D80 PromptFA,
packed-768 vision, NZ decode weights, NPU rotary and
`pse_sentinel_310p` (the IncreFA `pse_shift` path).

## Validation and scope

47 local tests passed. The real production run's validation passed: all 2,749
raw call records reconcile exactly with route counts, real/physical tokens,
and the existing vision-transformer device-event total (106.487983 s).
All 11 routes were exercised. There are 5,490 request traces and 384 predictions.
Every Markdown page is byte-identical to the first 384 pages of the previous
full 1651 run. All 384 layout token sequences are exact. The two recognition
trace differences are verified table-image placeholder renamings with unchanged
Markdown/content JSON. The 18 pre-existing length stops remain unchanged.
See `comparison.json`. No new accuracy evaluation was run.

The instrumentation tags existing event pairs. It adds no event pairs and no
per-call synchronization. It resolves at the existing window completion point.
Warmup samples are cleared with the measurement counters. Percentile aggregation
and raw-file serialization occur after the hot timer.

These timings cover the production `vision_transformer_blocks` event region.
Device events can include host submission gaps. Single-image routes include
padding/mask preparation; packed routes construct their mask before the event.
Patch embedding, rotary-position preparation and merger have separate existing
counters. The rates are not isolated attention-kernel performance. Comparisons
between direct and packed routes must retain these boundary differences.

## Per-route results

Rates are sums of real or physical tokens divided by summed event seconds.
Latencies are per-call, with linear-interpolated percentiles at `(n-1)*q`.

| Route | Calls | Time s | Time share | Real tok/s | Physical tok/s | p50 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 57 | 0.812 | 0.76% | 11,589 | 26,950 | 14.130 | 15.376 | 15.379 |
| 512 | 9 | 0.134 | 0.13% | 29,807 | 34,441 | 14.585 | 15.925 | 15.942 |
| 768 direct | 213 | 3.723 | 3.50% | 39,506 | 43,937 | 17.562 | 22.253 | 32.087 |
| 768 packed | 1,353 | 22.538 | 21.16% | 42,124 | 46,105 | 16.600 | 17.715 | 18.457 |
| 1024 | 185 | 3.848 | 3.61% | 42,657 | 49,231 | 20.768 | 21.873 | 21.975 |
| 1536 | 205 | 5.391 | 5.06% | 47,617 | 58,410 | 26.269 | 27.327 | 27.397 |
| 2048 | 96 | 3.231 | 3.03% | 53,202 | 60,844 | 33.639 | 34.592 | 34.757 |
| 3072 | 102 | 4.893 | 4.60% | 50,484 | 64,038 | 47.930 | 48.925 | 50.523 |
| 4224 | 45 | 3.310 | 3.11% | 49,029 | 57,431 | 73.436 | 74.691 | 74.717 |
| 5632 | 413 | 43.810 | 41.14% | 51,199 | 53,094 | 105.935 | 107.248 | 118.585 |
| Eager overflow | 71 | 14.799 | 13.90% | 36,550 | 36,550 | 197.494 | 285.071 | 286.537 |

The JSON also includes mean, population std, min, p90 and p95 for every route
and each exact overflow sequence length. S512 has only nine samples; its p99
is descriptive of this run and not a stable tail estimate. Many exact overflow
shapes have only one or two samples.

## Findings

- S5632, packed S768 and overflow account for 76.20% of vision time. S5632's
  p50/p99 are close, so its dominant cost is consistent repeated work.
- Packed S768 processed 4,094 crops in 1,353 calls: 3.026 crops/call,
  16.658 ms mean/call or 5.505 ms/crop. This is observed amortization across
  the production crop mix, not a controlled packing speedup measurement.
- Overflow is only 71/2749 calls (2.58%) but takes 13.90% of vision time.
  All 44 exact sequence lengths, spanning S5684–S8160, are preserved.
  Even equal shapes vary: S8064 has seven calls, 183.129–269.845 ms;
  S7872 has five calls, 183.958–267.068 ms. These event regions alone do not
  establish whether that variation comes from host dispatch or operator work.
- Physical throughput peaks at S3072 (64.0k tok/s) and decreases to 53.1k
  at S5632. Useful throughput also depends on padding: S384 is only 43.0%
  real tokens, but its total cost is just 0.812 s, so it is a low-impact target.
- Vision transformer total: 4,895,824 real / 5,300,436 physical tokens,
  45,975 real tok/s and 49,775 physical tok/s, 92.37% useful tokens.
- Text transformer prefill: 1,361,206 real tokens / 74.332 s = 18,312.5 tok/s.
- Decode: 463,447 effective tokens / 63.690 s = 7,276.6 tok/s;
  raw 484,000 slots = 7,599.3 tok/s; 15,125 graph calls, 4.211 ms mean graph,
  96.888% active-slot occupancy. This is a 384-page result; use the matching
  310P 384-page run for comparison rather than the full-dataset 0.17 pg/s.

## Artifacts and reproduction

`run_summary_shard_00.json` includes `vision_timing.by_route`,
`vision_timing.by_exact_shape`, and 20 slowest records.
`vision_timing_shard_00.jsonl` contains every resolved vision call, including
execution-order index, internal request IDs, members/member lengths, hidden
sequence shape, attention head count/dimension, dtype and device seconds.

Explain the saved reference locally with:

```bash
python3 11_mineru_2_5_pro_inference/vision_timing_report.py \
  11_mineru_2_5_pro_inference/references/vision_timing_384_910b/run_summary_shard_00.json
```

The complete remote run, including logs, predictions and token traces, is:
`/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/vision_timing_384_9bc93294_20260905`.

The new 310P handoff is `WORK_SERVER_310P_MINERU_VISION_TIMING_384.md`.
It requires explaining the previous full-run summary and explicit approval
from Luka before starting its 384-page run.
