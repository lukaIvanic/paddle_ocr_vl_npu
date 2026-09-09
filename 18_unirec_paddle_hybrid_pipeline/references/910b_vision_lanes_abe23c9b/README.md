# UniRec vision lanes inside the hybrid — 910B2 comparison

2026-09-09, one physical 910B2 NPU 6. Implementation `1387cfd2`;
worker-join-before-executor-release safeguard `abe23c9b`.
First64 matrix ran at 1387cfd2; first384 at abe23c9b.

Experiment18 uses experiment12's existing BoundedVisionOwner, with an added
in-memory input entry point instead of a spool round trip. Persistent threads
dispatch independent vision bucket keys on stable per-key streams. All lanes
join before text prefill or another model/phase runs. There is no same-key
graph cloning, no new weight copy, no CPU worker-count change, and no layout,
prefill grouping, routing, decode scheduling, resolution or ready-KV change.
`--unirec-vision-lanes 0` is the original sequential control; 1/2/4 use the
existing executor with the specified concurrency. All graph objects remain
resident. These controls use the same cache roots and compiled model sources.

## First384 results

| Vision mode | Pipeline wall s | pg/s | Joined vision wall s | Torch peak allocated GiB | Torch peak reserved GiB | Sampled whole-device peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original sequential | 176.412 | 2.1767 | 45.458 | 17.177 | 18.514 | 22.322 |
| Executor, 1 lane | 178.586 | 2.1502 | 51.273 | 18.133 | 20.180 | 24.037 |
| Executor, 2 lanes | 182.540 | 2.1036 | 51.814 | 18.133 | 20.180 | 24.037 |
| Executor, 4 lanes | 173.051 | 2.2190 | 43.394 | 18.133 | 20.180 | 24.037 |

Four lanes saved 3.361 seconds overall (1.9% wall reduction / 1.94% throughput
increase), and 2.064 seconds in the vision owner envelope (4.54%). One run per
case on a shared server is not a statistically established speedup.
The original default remains selected; the small observed gain does not justify
silently adding memory to the 310P preset that recently encountered OOM.

The external sampler read 22,858 MB for the control versus 24,614 MB for every
executor variant. These whole-device values include a 3,424/3,426 MB baseline,
not just this process. Difference: 1,756 MB = 1.71484 GiB. Torch allocation
increased 0.95542 GiB; these are different counter scopes and must not be added.
Sampling had no errors (243/239/244/238 samples for 0/1/2/4 respectively).

The vision owner interval includes CPU canvas packing, host submission, cache
load/first-use work and waits, not just kernel execution. Each case used existing
disk caches in a fresh process; no real-page resident warmup was excluded.
Setup is recorded separately in comparison.json. Do not compare these rates
directly to standalone UniRec's warmup-excluded hot service figure.

The 385 joined vision turns had p50/p99/max:

| Mode | p50 ms | p99 ms | max s |
| --- | ---: | ---: | ---: |
| Original | 48.43 | 1044.40 | 16.056 |
| 1 lane | 55.34 | 1153.14 | 18.283 |
| 2 lanes | 53.07 | 1077.41 | 21.132 |
| 4 lanes | 49.79 | 670.10 | 16.082 |

The long maximum remains; its cause is not established by the aggregate data.
All cases processed 1,717 compiled bucket calls and nine existing eager-overflow
crops. The overflow route is inherited unchanged, not a newly introduced
fallback. Four-lane dispatch widths were 1:67, 2:104, 3:141, 4:145 groups.
These are submitted group widths, not proof that every group's kernels achieved
that degree of device overlap. Per-key lane host spans overlap and cannot be
summed as critical-path time. The non-overlapping owner partition reconciled
within floating-point precision in every run (largest residual 7.1e-15 s).

## Smoke and correctness

First64 original/1/2/4 pipeline wall: 53.994 / 53.196 / 52.988 / 53.688 s.
Joined vision: 8.718 / 9.077 / 8.444 / 7.214 s. The larger run therefore did
not reproduce the smoke's approximately 17% vision-envelope reduction.

All eight runs exited zero and completed all requested pages. Every candidate
exactly matched its same-size sequential control's crop token IDs, text and
stop reasons: 1,015 crops for first64 (741 UniRec / 274 Paddle) and 4,346 for
first384 (3,657 UniRec / 689 Paddle). Crop IDs/metadata were checked. Vision
bucket calls/real rows and both models' prefill token totals were identical.
This is output parity, not a new independent accuracy evaluation.

Local authoring tests: 32 passed, four torch-dependent tests skipped. The new
wrapper tests cover direct input ownership, retained-owner settings, summary
accounting, shutdown and failure propagation. Real NPU behavior is validated by
the production matrices, not inferred from those CPU tests. No 310P run or
memory-fit claim is made.

## Reproduction and evidence

Run `run_910b_vision_lanes_matrix.sh` after `source npu-setup`, with a new
`RUN_ROOT` and `PAGE_LIMIT=64` or `384`. It runs 0/1/2/4 serially, using the same
models/dataset/cache roots and `run_with_process_tree_memory.py`, then checks
exact crop parity before continuing. Comparison.json retains each exact command,
all exclusive timing distributions, lane timing/group distributions, parity,
token/shape invariants and raw peak npu-smi tables.

Full original artifacts remain on the 910B container under:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/18_unirec_paddle_hybrid_pipeline/vision_lanes_1387cfd2_64/
/workspace/repos/paddle_ocr_vl_npu/tmp/18_unirec_paddle_hybrid_pipeline/vision_lanes_abe23c9b_384/
```

Each root contains lanes0/lanes1/lanes2/lanes4, with command.txt, run.log,
exit_code.txt, parity.json, memory.json, and output/{run_summary.json,
timing_trace.json,recognition_trace.jsonl,predictions/}. The matrices were
detached and continued during a conversation interruption; no runs were
restarted. The old SSH master disappeared and was re-established for retrieval.
