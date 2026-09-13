# Single-engine runtime integration — Ascend 910B2

Runtime source: `52cf35a07b79778353b846ded22ccf151875eac7`.
Naming-only checkpoint: `a191a2a2` (retained separately).
The comparison utility was added in `e9800b84`; product code is identical.

## Scope

- One `ContinuousRecognizer` owns CPU preparation and the actual continuous
  decoding loop. No separate preparation/scheduler owners or iterable adapters.
- `PreparedCrop` and `DecodeRequest` replace five internal crop records.
  The latter owns a concrete prefill-cache lease; admission clears its device
  references and releases that lease at the existing event boundary.
- One `ServingSummary` retains the previous JSON summary fields and formulas.
- p02: 2,888 → 2,544 lines; 32 → 24 classes; 70 → 54 methods. This is removal
  of internal boundaries, not code moved into additional product files.
- p01 and p03–p06 are byte-identical to the naming checkpoint. No model,
  preprocessing, formatting, HTTP or graph-cache identity changes.

All 79 CPU tests passed in the container's existing Python environment.
`cpu-tests.log` includes exact test names; these tests are not NPU validation.
The historical naming receipt remains pinned to `a191a2a2`, rather than being
rewritten to describe this structural integration as a rename.

## Saved Poisson100

One 910B2 (physical NPU6), B8, target 6 QPS, 60,416-row head, KV4096,
Kornia-RS/uint8, detailed instrumentation. Exact saved request order and Poisson
schedule, no client concurrency cap. Cached startup and one real warmup outside
measurement. No new graph namespace; setup completed in 28.956 s.

| Metric | Accepted pre-integration | Integrated |
| --- | ---: | ---: |
| Mean request latency | 1.187888 s | 1.182164 s |
| P95 request latency | 2.980455 s | 3.021262 s |
| Completed requests/s | 5.258506 | 5.259351 |
| Errors | 0 | 0 |

100/100 native token streams, raw text, formatted HTML, completion reasons,
crop sizes and token counts match exactly. Every request ended at EOS.
The mixed latency changes are small; this single pair is not evidence of a
speedup. Completion rate includes the fixed arrival schedule and final drain,
not saturated serving capacity.

The final service summary retains exactly the same fields. It counts 101
requests including warmup; the measured client counts 100. The private cache
pool acquired/released 101 leases and finished with 0 active / 40 free slots.
The ownership log reports no processes left on NPU6 after shutdown.

`poisson100/plan.json`, `b8/server_command.txt` and `b8/measured/command.txt`
retain configuration and exact commands. `poisson100/comparison.json` checks
the accepted `refactor_poisson100_20260913` outputs, not the older 16K/Pillow
configuration.

## Final 1,000-request confirmation

Passed using the existing cached B8 / 6-QPS / 1,000-request protocol with
scheduling metrics (as in its reference, not Poisson100's detailed metrics).
Compared to `current_checkpoint_20260913/cached`, replaying restored math
formatting on the reference only; native tokens and raw text are compared
unchanged. `poisson1000/comparison.json` contains the complete comparison.

| Metric | Accepted pre-integration | Integrated |
| --- | ---: | ---: |
| Mean request latency | 1.236842 s | 1.239532 s |
| P95 request latency | 3.716353 s | 3.709361 s |
| Completed requests/s | 5.734273 | 5.733571 |
| Errors | 0 | 0 |
| KV-limit stops | 10 | 10 |

1,000/1,000 native token streams, raw texts, formatted HTML, completion reasons,
crop sizes and input/projected-token counts match. The saved schedule is
byte-identical and covers all 665 distinct tables. All 1,000 emitted HTML
results also match the predictions previously scored at **95.427620% Page-TEDS**
in `refactor_poisson100_20260913/accuracy_restored_math/predictions.jsonl`.
This is exact scored-input parity, not a fresh TEDS evaluation.

The service summary counts 1,001 requests including warmup. All 1,001 private
cache leases were released; the pool finished with 0 active / 40 free slots.
All 96 ownership checks were clean; the final check found no NPU6 processes.

The observed mean change is +0.22% and P95 change −0.19%; these single-run
differences do not establish a performance change. HTTP/process behavior,
continuous CPU/NPU overlap, KV storage and completion rules remain unchanged.
Live logging and further diagnostic cleanup remain deferred.
