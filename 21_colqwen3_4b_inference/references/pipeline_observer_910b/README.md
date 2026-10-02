# Complete-pipeline observer validation — 910B2

Runtime commit: `2b1cafc3`. Date: 2026-10-02. One physical 910B2 (device 7,
mapped to `npu:0`), FP16, B1 sequential; existing compatible caches retained.
Environment: `/workspace/venvs/colqwen3_hf_py312/bin/python`, torch/torch-npu
2.10, Transformers 4.57.1. This is **not** 310P validation.

Every run encodes 111 real HR pages and 32 English queries, scores the selected
corpus, ranks results and computes metrics. Exact IDs are in
[workload.json](final/workload.json). Restricted-corpus development metrics are
not comparable with the published full-HR score. No full-HR run was launched.
No embeddings were serialized; embeddings remained in RAM through scoring.

## Observation overhead

The private development control omits events/logging/heartbeat, but retains the
same computational path, validation and shared workload metadata. Each ABBA run
uses a fresh process. [Final comparison](final/comparison.json):

| Wall-time measure | Control mean | Observed mean | Difference |
|---|---:|---:|---:|
| 111 pages | 39.449 s | 39.732 s | +0.72% |
| 32 queries | 7.882 s | 7.718 s | −2.09% |
| Encoding phase | 47.331 s | 47.450 s | +0.25% |
| Scoring | 0.303 s | 0.447 s | +47.63% / +0.144 s |
| Total job | 71.058 s | 72.510 s | +2.04% |

Observed page throughput was 2.845 and 2.744 pg/s. Control page throughput was
2.876 and 2.754 pg/s. These rates are wall-based **image-to-usable-embedding**
throughput, not full offline-job throughput. Totals include setup/evaluation;
`subprocess_s` additionally includes process startup/teardown.

There is meaningful run variance: do not interpret negative deltas as an
optimization or claim exactly 0.72% intrinsic observer overhead. The
[earlier ABBA](pre_pool_comparison.json), before timing-event pooling, measured
−1.13% page time and −0.03% total time, while scoring grew by 0.143 s. The final
pooling run does **not** prove pooling improved latency. Both repeats reveal
the small scoring phase's substantial relative observation cost.

All four final score matrices and ID lists were exactly equal. The observer
does not insert device synchronizations between subsections. NPU event pairs
are recycled only after completion and elapsed-time resolution.

## Log and accounting checks

[Representative result](final/result.json), [items](final/items.jsonl), and
[events](final/events.jsonl) preserve the full timing evidence:

- 111 page, 32 query, 111 scoring, two setup and one finalization completion.
- 1,962 section starts and matching finishes; 1,048 device resolutions.
- Zero pending device events at shutdown.
- 14 heartbeats; intervals 5.000–5.053 seconds.
- Each completion is immediately printed/flushed, independently of heartbeat.
- Full stage timing/token records and exact-shape/route aggregates are retained.
- Device intervals can contain launch gaps/waits; they are not pure active
  kernel time. Host submission spans and device intervals must not be summed.
- Unattributed host time and outer-loop time are reported explicitly.

31 local CPU contract tests passed, including deferred-event behavior, safe
event reuse, immediate flush, deterministic workload selection and ragged
MaxSim. These are bookkeeping tests, not local inference validation.

## Optional profiler

[Profiler result](profile/result.json): completed development run, exit 0,
2.404 pg/s page encoding. Scores exactly equal the unprofiled observed run.
Three traces captured actual second page/query/scoring items, with the preceding
real items used for warmup. No standalone stage was replayed. All 1,962 sections
finished, and no events remained pending. Profiler timings include profiling
and export overhead and are not clean throughput measurements.

Heavy traces remain in the container under:

`/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/observer_pool_2b1cafc3/profile/profiler/`

Trace directories (each contains `ASCEND_PROFILER_OUTPUT/trace_view.json`):

- `liteserver-c001-4_91303_20261002061517723_ascend_pt`
- `liteserver-c001-4_91303_20261002061603925_ascend_pt`
- `liteserver-c001-4_91303_20261002061618157_ascend_pt`

The full run directory also retains ABBA commands/logs, raw outputs and exit
codes. Older `observer_validation_final` retains the pre-pooling comparison.
