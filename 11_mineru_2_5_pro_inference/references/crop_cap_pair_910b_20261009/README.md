# 910B full MinerU E2E repeat and lower crop-cap comparison — 2026-10-09

Status: baseline inference and full evaluation passed. Lower-cap smoke/full
inference launched on the same physical card 2 after evaluation exited.

## Baseline rerun result — 910B2

All 1,651 pages and 32,051 recognition requests completed; zero failed/skipped
pages. Pipeline wall: 1,632.567217 s, **1.011291 pages/s**. Setup: 29.275795 s
separately. The prior reference was 0.979472 pages/s; this rerun is 3.25% faster.
Vision: 16,809,232 useful tokens in 365.340176 s, **46,009.8 useful tokens/s**.
Physical positions: 18,887,040; padding is unchanged from the prior run.
All 32,051 generated token sequences, all 1,651 Markdown files, input prompt
tokens, crop hashes, stop reasons and layout metadata exactly match the prior
run. Generated-token count remains 1,804,363, including EOS.
Fresh baseline evaluation exactly reproduced the prior scores: text 96.234624,
Page TEDS 93.520375, Page CDM 97.054051, overall 95.603017. All 1,651 pages
matched, 665 table and 2,352 formula samples; zero matching timeouts and zero
metric timeouts/errors/exceptions. Evaluation wall was 884.14 s, separate from
inference.
After-run device checks report no device error. After-run host load was
37.23 / 34.04 / 32.52. These timings are complete-run results, not estimates.

User request: reproduce the near-1-page/s full E2E result, then reduce the
maximum crop pixel count and check accuracy. Both lanes use all 1,651 original
OmniDocBench page images, live PP-DocLayoutV3, fresh MinerU recognition, unchanged
predictions, and the frozen full evaluator.

The first lower cap is an author-selected experiment: 401,408 pixels / 2,048 raw
vision tokens, compared with 602,112 pixels / 3,072 tokens. Only that maximum
changes in the inference command. Minimum pixels stay 25,088. This reduces the
maximum by one third; actual total token savings must be measured.

The existing generation policy in native_custom_backend.py::_finish_generation
limits max_new_tokens to the remaining KV4096 capacity after the input. A smaller
image input can therefore increase a request's permitted output length. This
existing policy is preserved; changed output limits, actual generated-token
counts and length-capped outputs must be reported. Accuracy differences are full
pipeline effects of changing the pixel cap, not an isolated resize-only test.

Inference uses the clean exact d4e7fdd0efd90bf6408c21efab320cc3292f07a4 checkout
for both runs, original vision precision, FP16 MinerU, FP32 eager layout,
compiled vision/text prefill, B32/KV4096 decode, and the same existing cache roots.
The coordinator gains an optional --processor-max-pixels argument and validates
the selected cap; its default remains the original 602,112. No model or processor
implementation changes are part of this experiment. Baseline uses the coordinator
at aef97137; the lower-cap coordinator revision is 18b8de13 (published before launch).

Verified environment: torch 2.10.0+cpu, torch-npu 2.10.0, Transformers 5.5.4,
mineru-vl-utils 1.0.5, kornia-rs 0.1.14, shapely 2.1.2. Physical 910B2 card 2 was
healthy and free at preflight; the prior reference used card 3. The host has 192
CPUs with affinity covering all 192 and no container CPU quota (cfs_quota_us=-1);
observed initial load was about 27.

Each lane runs a two-page smoke then a fresh full run with no resumed pages or
warmup-page subtraction. Record setup separately, host/device state before and
after, exact commands, durable exits, real/padded vision tokens, generated-token
counts, stage times, and output length caps. Run evaluation after inference, not
concurrently with the next timed lane. Compare text accuracy, Page TEDS, Page CDM,
overall, available category scores and page-level regressions; an unchanged
aggregate alone does not establish that no pages degraded.

Baseline run root:
`/workspace/repos/paddle_ocr_vl_npu_mineru_lengths/tmp/11_mineru_2_5_pro_inference/d4_cap3072_repeat_910b_20261009T110942Z`

All new measurements in this report are 910B-only. The 310P handoff is unchanged.

Lower-cap run root:
`/workspace/repos/paddle_ocr_vl_npu_mineru_lengths/tmp/11_mineru_2_5_pro_inference/d4_cap2048_910b_20261009T115709Z`

## CDM evaluation parallelism correction

Luka explicitly requested the second evaluation achieve roughly 20–40 formula
samples/s; the ~3.6–3.7 samples/s at 12 workers in the old and repeated baseline
must not be treated as the expected performance target. The inference run is
unaffected. The second evaluation will request **CDM_WORKERS=96**, after inference
finishes, and record observed pool size and aggregate samples/s.

Verified frozen evaluator implementation: src/runtime/concurrency.py reads the
metric config's cdm_workers before OMNIDOCBENCH_CDM_WORKERS; src/metrics/cal_metric.py
uses that resolved count in ProcessPoolExecutor. The existing preparation wrote
cdm_workers: 12, so changing the evaluator environment variable alone would not
have overridden it. run_serving_accuracy.sh now forwards CDM_WORKERS to the
existing prepare_serving_eval.py --cdm-workers option. Its default is preserved;
this run explicitly requests 96. Metric implementations, matching, rendering,
TEDS workers and timeouts are unchanged. These helper sources still match d4.

96 is a chosen setting for this 192-CPU host, not a measured throughput result
or a universal setting for 310P hosts. Confirm actual workers and achieved rate.
