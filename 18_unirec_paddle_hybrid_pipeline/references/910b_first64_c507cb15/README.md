# Initial 910B integration validation

Inference source: `c507cb15`. This is a first-64-page integration check, not
a representative full-corpus/hot throughput measurement. Dataset ordering is
the existing OmniDocBench.json annotation order, not sorted filenames.

All runs use PP-DocLayoutV3 and produce the same 1,015 recognition crops.
The hybrid routes 741 OCR crops to UniRec and 272 formulas plus two tables
to Paddle. Both models remain resident on one 910B2. No inference ran on CPU.

| Route | Pages | Pipeline seconds | Pages/s | Setup seconds | Peak Torch allocated GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| All UniRec | 64 | 46.4126 | 1.3789 | 17.7920 | 9.0935 |
| All Paddle | 64 | 63.6954 | 1.0048 | 38.3503 | 15.4319 |
| Hybrid | 64 | 62.1871 | 1.0292 | 48.1155 | 23.8025 |

Torch allocation is not physical HBM consumption. Hybrid peak Torch reserved
was 25.2434 GB. Neither measurement establishes fit on 310P.

## Output comparison

- UniRec hybrid vs all-UniRec: **741/741 exact token-ID sequences**.
- Paddle hybrid vs all-Paddle: **272/274 exact token-ID sequences**.
- Hybrid stopped 1,012 crops on EOS, one via Paddle's existing repetition
  guard, and two at the UniRec length limit.
- Both UniRec length-limited cases have exactly the same token IDs in the
  all-UniRec control: `page_000003_block_000009` and
  `page_000012_block_000005`. Their text contains repetition; they were not
  introduced by switching recognizers.
- All 64 Markdown and 64 structured JSON pages were written.
- Full OmniDocBench accuracy evaluation has **not** been performed.

Earlier two-page gates passed for both individual adapters, 25 crops each.
All-Paddle matched 24/25 token sequences in the historical full-run reference
`910b_full_cap4096_text05_b64_pse_target768_898ced7`; the difference was a
formula generation. This historical comparison has different grouping and
runtime history; do not call it strict parity.

## Performance scope

This is a formula-heavy prefix, not the average document mix. The hybrid's
UniRec active token-slot utilization is only 15.68%, and Paddle's is 53.95%.
These include the complete drain and the long UniRec generations. The run is
not evidence for the earlier approximately 3 pages/s planning estimate.

Hybrid coordinator action times (non-overlapping host intervals):

- Shared layout/crop preparation/writing at that boundary: 13.454 s.
- UniRec prefill: 11.937 s; UniRec decode turns: 13.737 s.
- Paddle prefill: 13.870 s; Paddle decode turns: 9.107 s.

Engine run-window statistics can include time paused for the other recognizer;
do not sum them as additional work. The coordinator's action intervals and
whole-pipeline wall provide the cross-model accounting.

## Artifacts and tests

Remote root:
`/workspace/repos/paddle_ocr_vl_npu/tmp/18_unirec_paddle_hybrid_pipeline/smoke_c507cb15/`

Lanes: `unirec64_serial`, `paddle64_serial`, `hybrid64`. Each contains predictions,
`recognition_trace.jsonl` with exact IDs, and `run_summary.json`. Logs are
adjacent to the lane directories. Compact summaries and the exact-token
comparison are retained in `validation.json`. All three ran on physical NPU 4,
without concurrent controls. The earlier `unirec64`/`paddle64` controls raced
on device selection and overlapped on NPU 4; their timings are excluded here.

Existing regression tests passed on the configured 910B host:
13 Paddle continuous scheduling tests and seven UniRec persistent-queue tests.
Seven experiment-18 CPU policy/report tests pass locally, including a service that
drains a partial request, stays open, accepts another request, and closes.

Initial Paddle setup populated missing vision entries in the existing cache
root (271.6 s); the second setup reused them (8.1 s). A smoke-discovered
inference-mode initialization error was corrected in `c507cb15` before the
successful Paddle/hybrid tests. No fresh cache root was selected.
