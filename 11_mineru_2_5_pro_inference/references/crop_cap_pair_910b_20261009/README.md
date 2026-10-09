# 910B full MinerU E2E repeat and lower crop-cap comparison — 2026-10-09

Status: both full inference runs and both full evaluations passed. All inference
results below are **910B2**, physical card 2. No 310P run was performed.

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
unaffected. The second evaluation requested **CDM_WORKERS=96**, after inference
finished, and recorded observed pool size and aggregate samples/s.

Verified frozen evaluator implementation: src/runtime/concurrency.py reads the
metric config's cdm_workers before OMNIDOCBENCH_CDM_WORKERS; src/metrics/cal_metric.py
uses that resolved count in ProcessPoolExecutor. The existing preparation wrote
cdm_workers: 12, so changing the evaluator environment variable alone would not
have overridden it. run_serving_accuracy.sh now forwards CDM_WORKERS to the
existing prepare_serving_eval.py --cdm-workers option. Its default is preserved;
this run explicitly requests 96. Metric implementations, matching, rendering,
TEDS workers and timeouts are unchanged. These helper sources still match d4.

96 was selected for this 192-CPU host; it is not a universal setting for 310P hosts.
The measured result follows below.

## Lower-cap inference result — 910B2

All 1,651 pages / 32,051 requests completed, zero failed/skipped pages.
Wall 1,658.862277 s (0.995260 pages/s), setup 29.390122 s separately.
Real vision tokens fell from 16,809,232 to 15,253,444 (9.26% fewer); physical
positions fell from 18,887,040 to 17,013,376. Vision time fell from 365.340176
to 351.083919 s, but full pipeline throughput fell 1.59%. This single pair
does not demonstrate an E2E speedup. The original-cap repeat itself varied
by 3.25% from the previous run, so do not treat a small timing difference as
a stable cap effect. Device health passed before and after; lower-cap host
load was 37.31 / 39.69 / 38.49 before and 30.29 / 30.57 / 32.85 after.
Actual command comparison found only max pixels and output-directory changes.

CDM completed with 96 requested workers and 96 actual Python child processes
observed in the evaluator's process group. It scored all 2,352 formulas in
1m54s at **20.56 samples/s**, versus the fresh baseline's 12 workers and
**3.62 samples/s** (10m49s). This meets the requested 20–40 samples/s target.
The measurements compare separate full evaluations of the two cap outputs;
they are not a controlled worker-scaling benchmark on identical predictions.
The same frozen CDM implementation was used.

## Completed comparison — 910B2

| Metric | Original cap 602,112 / 3,072 tokens | Lower cap 401,408 / 2,048 tokens |
|---|---:|---:|
| Full pages / requests | 1,651 / 32,051 | 1,651 / 32,051 |
| Pipeline wall (s), excludes setup/evaluation | 1,632.5672 | 1,658.8623 |
| Pages/s | 1.011291 | 0.995260 |
| Real vision tokens | 16,809,232 | 15,253,444 |
| Vision blocks event time (s) | 365.3402 | 351.0839 |
| Useful vision tokens/s | 46,009.8 | 43,446.7 |
| Text accuracy | 96.234624 | 96.313265 |
| Page TEDS | 93.520375 | 93.019968 |
| Page CDM | 97.054051 | 96.953065 |
| Overall | 95.603017 | 95.428766 |
| Evaluation wall (s), separate from inference | 884.14 | 348.58 |

Lower cap: **9.26% fewer vision tokens, no demonstrated E2E speedup, and accuracy
loss**: TEDS -0.500407 percentage points, CDM -0.100986 pp, overall -0.174251 pp.
Text increased +0.078640 pp. This does not support replacing the original cap
as a free speed/quality improvement on 910B. A small single-pair timing difference
is not evidence of a stable slowdown either, given observed repeat variation.

| Page metric | Scored pages | Worse | Better | Equal | Worse by >1 pp |
|---|---:|---:|---:|---:|---:|
| Text | 1,557 | 26 | 31 | 1,500 | 8 |
| TEDS | 458 | 85 | 44 | 329 | 35 |
| CDM | 313 | 12 | 7 | 294 | 5 |

Individual losses are much larger than the averages: worst TEDS page
`page-f6b6cfb8-3bb5-4393-b3fb-70efef72aca5.png` went from 100 to 30.68; worst CDM
page `page-1ce383cc-63a3-43c9-b6f2-7498a952b65c.png` went from 99.91 to 41.12.
The worst/best ten pages and all available category scores are in comparison.json.
These are observed output/metric changes; no visual diagnosis of their causes
has been performed.

Both evaluators matched all 1,651 pages with zero fallbacks/timeouts; all 665
TEDS and 2,352 CDM samples completed with zero metric errors/exceptions/timeouts.
For 2,189 formula matches with identical ground truth and prediction, every CDM
score was identical between evaluations, despite the worker-count change.
The same holds for 520 unchanged table matches. Matching can change when the
prediction changes; these counts are not the total sample denominators.

Layout metadata and all crop image hashes were unchanged. Processor input lengths
and permitted output limits changed for 1,821 requests; every changed output limit
increased under the preserved KV-capacity policy. 474 generated token sequences
changed. Generated tokens increased from 1,804,363 to 1,811,204; length-capped
requests increased from 26 to 28. Thus the experiment measures the existing full
pipeline's response to a smaller image cap, including its existing output-limit
policy. Only max pixels/output directory differ in the actual inference commands;
configuration_comparison.json records that check. No model implementation changed.

Stage diagnostics: text prefill was 241.65 vs 242.81 s, decode 247.67 vs 245.27 s,
layout host span 207.69 vs 224.54 s. These event/host spans overlap; they are not
an additive exclusive breakdown. Vision event regions include launch gaps and
first-use work, as in the original full-run harness. We did not subtract outliers
or replace full-run throughput with steady-only measurements.

## Reproduce the analysis from saved evidence

Archives preserve the original commands, logs, exits, device/host snapshots,
full predictions, generation traces and evaluation results. Formula rendering
scratch directories were omitted from evaluation archives; scores, per-sample
inputs/results, runtime records and logs are retained. SHA256SUMS checks all
four archives. Evaluation source is frozen at
2b161d010d2e3aff77a0edef359ea3a6411d23cd; the second evaluation launcher is 9408fd1a.
The inference worktree remains exact d4e7fdd for both runs.

```bash
mkdir -p /tmp/mineru-cap-baseline /tmp/mineru-cap-lower
for phase in inference evaluation; do
  tar --warning=no-timestamp -xzf baseline_${phase}_raw.tar.gz -C /tmp/mineru-cap-baseline
  tar --warning=no-timestamp -xzf lower_${phase}_raw.tar.gz -C /tmp/mineru-cap-lower
done
python3 analyze_pair.py --baseline /tmp/mineru-cap-baseline/full1651 \
  --comparison /tmp/mineru-cap-lower/full1651 --output /tmp/mineru-cap-comparison.json
```

Run from this evidence directory. The analysis reaggregates page scores from raw
samples and checks them against evaluator aggregates. It also compares output
limits, tokens, layout metadata, category scores and actual CDM process observations.
Remote wall clocks were ahead of local extraction time; original receipt times
were preserved. Tar's timestamp warning is suppressed above; throughput uses the
original monotonic measured durations.
