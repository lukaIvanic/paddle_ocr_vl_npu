# Frozen 60,416-row head versus saved 16k/full controls

Completed 2026-09-11 on physical NPU6, one Ascend 910B2. Runtime commit:
`fb6532c1f779a39ee807b2e3051a13c161de4974`.

Only the new 60,416-row variants were run. The 16,384- and 103,424-row controls
are the earlier same-day runs in `../lm_head_ab_20260911/`, runtime `2c04ecc7`.
This is a comparison against saved controls, not an interleaved variance study.

## Real-serving results

Same random-100 seed-1 order from the 665-table corpus, independent client,
closed-loop C=B. No request batching wait. Actual submission-to-response wall
latency, with setup and one real warmup request outside the measured 100.
Decode-device timing enabled equally across old and new runs.

| B/C | Head rows | Tables/s | Mean s | P50 s | P90 s | P95 s | P99 s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/2 | 16,384 (saved) | 2.957706 | 0.663769 | 0.394391 | 1.309363 | 1.889210 | 4.027872 | 4.277204 |
| 2/2 | **60,416** | **2.927094** | **0.670218** | **0.402869** | **1.318579** | **1.927554** | **4.151748** | **4.278569** |
| 2/2 | 103,424 (saved) | 2.833003 | 0.692523 | 0.417800 | 1.364497 | 1.993757 | 4.257685 | 4.446377 |
| 8/8 | 16,384 (saved) | 6.090250 | 1.174274 | 0.694363 | 2.384824 | 3.237947 | 7.461281 | 8.374919 |
| 8/8 | **60,416** | **5.787574** | **1.229111** | **0.749415** | **2.520271** | **3.333576** | **7.778751** | **8.758016** |
| 8/8 | 103,424 (saved) | 5.645766 | 1.256543 | 0.744877 | 2.606625 | 3.527710 | 8.195074 | 8.948973 |

Relative to 16k, 60k throughput decreased **1.035% at B2 / 4.970% at B8**;
P95 increased **2.030% / 2.953%**. Relative to full, throughput increased
**3.321% / 2.512%**. Small differences, particularly B2, should not be treated as
precisely repeatable effect sizes from this single saved-control comparison.

All 200 measured requests succeeded and stopped at EOS. Each run generated
41,120 tokens, the same aggregate token count as each saved control. No crop,
input-token-count, image-token-count or completion-reason differences were found.

## Raw output checks

- B2 expanded versus B2 full: **100/100 native token streams identical**.
- B8 expanded versus B8 full: **100/100 native token streams identical**.
- Expanded versus 16k: **98/100 identical** at each B, with the same two table
  differences observed in the earlier full-head comparison:
  `page_000542_table_0` and `page_001417_table_2`.

Formatted outputs match the full controls 100/100 too. This is observed parity
on this 100-table sample, not a full-corpus OCR-quality guarantee or fresh TEDS
evaluation. Comparisons use actual recorded generation IDs, never retokenization.

## Decode event metrics

These service aggregates include the real warmup request: **101 requests**, not
only the measured 100. Event intervals can include submission gaps; they are not
pure kernel-active time.

| B | Head | ms/call | Calls/s | Physical slots/s | Useful decode tokens/s |
|---|---|---:|---:|---:|---:|
| 2 | 16k saved | 1.166181 | 857.500 | 1715.000 | 1492.933 |
| 2 | 60k | 1.188296 | 841.541 | 1683.082 | 1467.400 |
| 8 | 16k saved | 1.523285 | 656.476 | 5251.808 | 3570.423 |
| 8 | 60k | 1.626934 | 614.653 | 4917.225 | 3329.153 |

## Frozen configuration and execution evidence

- Map: `19_table_ocr_serving/presets/table_compact_vocab/native_han_core_60416.json`.
- Mapping SHA256: `c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637`.
- 60,352 protected native/Unicode IDs plus 64 remaining mapped IDs in ascending
  ID order. Original 16k row order preserved. Exact fillers and provenance are
  recorded in the mapping. The runtime default remains 16k; opt in with
  `--expanded-decode-lm-head`. No per-request vocabulary switching.
- Same model, FP16, KV4096/output4096, first-token full head, preprocessing,
  scheduling, GC freeze, and instrumentation as the previous comparison.
- Vision, text-prefill and decode model source files unchanged from `2c04ecc7`.
  Only startup map selection/CLI wiring changed. Head size/map hash separates
  decode caches; two new decode graphs were built, B2 and B8. Prefill caches reused.
- Every variant used the same compile/warm process, then a fresh cached process
  with real-request warmup before measurement. No synthetic warmup redesign.
- Same locked `be691de1` client and sample order digest:
  `944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b`.
- **126 ownership snapshots clean**. Each phase stopped only its owned server
  and recorded the device free afterward. Final NPU6 check was free.
- Source committed/pushed locally, then checksum-verified bundle followed by
  remote `git pull --ff-only`, because private GitHub authentication failed.
- Result archive SHA256:
  `3ffc0e602cb01bd1bb99c3f2577e4e7e59959817b948c22fbfb70641a09b8d97`.

`driver.py`, all commands, readiness records, logs, native responses and service
summaries are preserved here. `analyze.py` generates `comparison.json` from these
artifacts and the saved controls. No further inference processes remain running.
