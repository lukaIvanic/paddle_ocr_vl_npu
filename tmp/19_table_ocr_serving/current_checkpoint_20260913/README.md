# Current-code validation before runtime redesign

Product baseline: `e268dae4` (numbered files and definition-order pass).
No further product changes are part of this run. The launcher is versioned
separately with its exact commit recorded in `plan.json`.

One Ascend 910B2, B8, 6-QPS Poisson arrivals, all 665 table crops plus 335
repeats in the exact historical globally shuffled 1,000-request schedule.
The frozen `be691de1` client, warmup and device-ownership harness are reused.
Server arguments adapt to the current `p01_serve.py` CLI only.

Current defaults: 60,416-row head, full-head first-token selection, Kornia-RS
resizing, uint8 CPU patches/NPU normalization, KV4096, unchanged pixel limits,
`metrics_level=scheduling`. Actual submission-to-response latency includes
queueing; scheduled-arrival latency is retained too. No output/result caching.

First process: new cache root, normal constructor compilation and one complete
real-request warmup, zero measured requests. Stop it, then restart from those
caches, warm with the same request and measure 1,000 requests. Every run keeps
commands, logs, readiness, complete responses, timings and ownership checks.

The old step-3 1,000-request run is a historical comparison, not an isolated
control: it predates the larger vocabulary and preprocessing/postprocessing
changes. Compare native output IDs and input shapes by occurrence, retain
all errors/cap stops and report discrepancies rather than hiding them. The
new measured run is the starting reference for subsequent invasive refactors.

Launch on the direct host from the updated main checkout:

```sh
python3 -u tmp/19_table_ocr_serving/current_checkpoint_20260913/driver.py \
  --expected-commit <full-committed-HEAD> --npu 6
```

The launcher refuses existing run/cache directories. It checks device ownership
before startup and throughout both phases, and stops only its own processes.
Output lives under the historical checkout's matching `tmp/...` directory;
this preserves frozen-client path resolution. Retrieve and retain it here.

## Completed measurement

2026-09-13, one Ascend 910B2, physical NPU6. Runtime/launcher commit:
`ce7a92b1150d456b2c83a3f35e04a6b0b4471f44`; the six product scripts are unchanged
from `e268dae4`. Fresh constructor setup took 369.626 s (vision 192.243 s,
text prefill 143.512 s, decode 10.854 s). One real warmup succeeded, then that
process stopped. Cached-process setup took 33.382 s, followed by the same real
warmup and the measured run. No inference-source fix was made during validation.

| Metric | Historical step 3 | Current checkpoint |
|---|---:|---:|
| Mean latency (s) | 1.317758 | 1.236842 |
| P50 (s) | 0.919765 | 0.817832 |
| P95 (s) | 3.790191 | 3.716353 |
| P99 (s) | 7.264818 | 7.244143 |
| Maximum (s) | 8.695539 | 8.439036 |
| Completed tables/s including drain | 5.733937 | 5.734273 |
| Errors / measured requests | 0 / 1000 | 0 / 1000 |
| KV4096 stops, retained | 10 | 10 |

The schedule is byte-identical. Its finite offered arrival rate is
5.763293819/s, despite a target of 6 QPS. Completed throughput here is not
maximum supported capacity. Current scheduled-arrival P95 is 3.716609 s;
maximum outstanding requests is 18. Neither CPU overlap nor queueing was
subtracted. Mean fell 6.14%, P95 fell 1.95%, but this is not an isolated
refactor comparison because the historical configuration differs as described above.

112 compilation-phase and 97 cached-phase ownership checks found no foreign
NPU6 processes. Both phases ended with the device free; direct-host checks
confirmed release. Setup GC froze 1,327,679 objects in the compiling process
and 625,160 after cached restart; GC remained enabled.

## Output differences and accuracy

This is **not an exact-output-parity pass** against historical step 3:

- 961/1000 native token streams match; 39 differ (25 unique tables).
- 966/1000 raw texts match; 34 differ (21 unique tables). Five token-stream
  differences therefore do not change the decoded text.
- 226 formatted outputs differ. Of those, 192 have identical raw generation
  and are explained by the intentional postprocessing change. Replaying the
  current converter reproduces every current formatted output exactly.
- All occurrence IDs, prompts, crop types, crop dimensions, input-token counts,
  projected-image-token counts and completion reasons match.
- All repeated occurrences of each input produce identical native tokens
  within the current run; there are 665 unique inputs.

Raw changes include character/sign recognition and some table-structure changes;
they are not all formatting. `output_audit.json` retains their per-table edits.
The old 16k head, preprocessing and postprocessing differ from current settings,
so this run cannot identify which earlier change caused each discrepancy.

As an additional check, all 100 shared inputs from the previous B8/6-QPS
Kornia+uint8/60k-head run match the current native token streams and raw text
exactly, with matching stopping reasons and input shapes/counts. This is a
shared-input output check, not a same-load performance comparison: the short
run had a different request schedule and detailed device instrumentation.

Both saved output sets were CPU-scored against the same ground truth and
evaluator, counting each unique table once (665 tables across 458 pages):

| Accuracy metric | Historical | Current |
|---|---:|---:|
| Page-TEDS | 95.455281% | 95.365599% |
| Table-average TEDS | 94.972558% | 94.930519% |
| Page structure-only TEDS | 97.811782% | 97.780307% |
| Table structure-only TEDS | 97.521605% | 97.510669% |

Zero evaluator timeouts and errors in either score run. Page-TEDS declined
0.089682 percentage points; table-average TEDS declined 0.042039 percentage
points. No regression tolerance is invented here: accuracy is not identical,
and invasive refactoring is paused for review of these differences.

Of the Page-TEDS decrease, 0.061496 percentage points comes from tables with
identical raw text (postprocessing only); 0.028186 points comes from tables
whose raw generation changed (which can also include postprocessing effects).
Across unique tables, 31 TEDS scores improve and 64 worsen. Aggregate closeness
does not mean every changed table is equivalent.

CPU scorer commit: `4fa62e73`; dedicated existing Python environment:
`/workspace/venvs/omnidocbench_py310/bin/python`. Evaluator checkout:
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`. Dataset SHA256:
`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
The scoring helper imports the existing process-isolated scorer from experiment
09; it does not run any NPU inference or change the serving code.

## Evidence

`compile/` and `cached/` retain exact commands, logs, readiness, warmup/results,
service summaries and ownership. `comparison.json` checks the historical run;
`analyze.py` regenerates `output_audit.json`. `accuracy/` contains complete
per-table/per-page scores and the paired comparison; `accuracy.log` preserves
the scorer output. These are table-crop results, not full page-parser validation.

Source was committed/pushed locally, then delivered by checksum-verified Git
bundle and pulled fast-forward because remote GitHub credentials were unavailable.
No tracked remote source was hand-edited. Archive SHA256 values:

- Run evidence: `8a46abe88688440dd49d6a9221d93db26dea6554de040f7ecdd9cae57d566782`.
- Accuracy evidence: `d3c971718a20ef47ace3063edc5428a2208f2b3cb7245fc74eea50b98107430b`.
