# Smaller Paddle ready KV pool — full 910B2 validation

Inference commit `8b60af95` (production change `eb3264e1`), 2026-09-09.
One 910B2, physical NPU 6. Baseline: `4a01bda5` full timing run.
No changes to weights, image limits, task routing, prefill grouping, decode
batches, generation limits, or CPU/NPU concurrency.

## Change

Experiment 18 defaults to 64 ready rows of 1,536 positions instead of 96 rows of
4,096 positions. Paddle's active decode arena remains B64 x 4,096; UniRec is
unchanged. These are ready-storage settings, not shortened generation limits.

Admission extends the existing experiment09 `DecodeArena.admit`: the same
`torch._foreach_copy_` copies the full shorter source row into a prefix view
of the active row. Slot controls, hot-swap synchronization and the existing
cache-release/event-reuse protocol remain in place. No hybrid-specific copy
runtime or custom kernel is introduced. The finite destination suffix remains
masked by the real cache position and gets overwritten as decode progresses.

Packed prefill uses its existing scratch graphs and valid-prefix redistribution.
The 18 individual text-prefill routes above the packed limit use a shared
B1 x 4,096 scratch cache, preserving the old compiled call shape, then copy their
valid prefixes into ready storage. This extra scratch is about 72 MiB.
Compiled model sources and cache roots were unchanged.

Experiment 09 standalone constructor defaults remain unchanged. Experiment 18
offers `--paddle-ready-cache-rows` (default: Paddle batch size) and
`--paddle-ready-cache-length` (default: 1536). The baseline storage can be selected
with `--paddle-ready-cache-rows 96 --paddle-ready-cache-length 4096`.
Prompts exceeding the configured ready length fail explicitly before prefill,
without truncation or automatic alternate paths.

## Full 1,651-page result

| Measurement | Previous | Smaller ready pool |
| --- | ---: | ---: |
| Processing wall | 637.722 s | 632.005 s |
| Processing throughput | 2.5889 pg/s | 2.6123 pg/s |
| Peak Torch allocated | 22.289 GiB | 17.297 GiB |
| Peak Torch reserved | 23.619 GiB | 18.557 GiB |
| Ready pool allocated | 6.750 GiB | 1.6875 GiB |
| Paddle admission device-event envelope | 0.972 s | 2.381 s |
| Paddle admission host enqueue | 1.453 s | 2.924 s |

The pool saves 5.0625 GiB. Full-run peak allocation falls 4.9917 GiB, consistent
with the additional reusable individual-prefill scratch and small other
allocation differences. Performance numbers are one run per configuration on
a shared machine, not a statistically established speedup.

The existing grouped copy becomes a strided-destination copy. Despite moving
less storage (171 GiB -> 64.125 GiB across 2,432 admissions), its admission envelope
increased 1.409s across the corpus, about 0.22% of the previous processing time.
This envelope also includes slot-control operations and is not pure memory
bandwidth. Do not claim shorter copies are faster; the measured result is a
large memory saving with a small admission-cost increase and no observed E2E
throughput regression.

## Correctness, capacities, and coverage

- Full run exit 0; exactly 1,651 JSON and1,651 Markdown files, all byte-identical
  to the baseline.
- Exact same 30,557 unique crop requests, metadata, token sequences, output
  texts and stop reasons:28,125 UniRec and2,432 Paddle.
- Same per-model prefill token counts, prefill-chunk-size histograms, and
  decode graph counts (14,933 UniRec / 9,217 Paddle).
- Pool:64 rows,1,536 positions,1,811,939,328 bytes, peak 64 leased rows.
  All 2,432 acquisitions released; 2,368 reuses; zero live leases at drain.
- Maximum observed prompt length:1,036, leaving 500 positions of headroom.
  This is a measured workload maximum, not a universal model/input limit.
- Existing timing partition still reconciles with zero error.
- First64 smoke: all1,015 crop outputs matched, pool peak 64 rows, max prompt 1021,
  peak allocation 17.105 GiB versus 22.168 GiB. Smoke wall 54.022s versus 52.237s;
  the full-corpus measurement is the relevant throughput comparison.
- Four new tests passed on actual910B tensors: full-row admission, compact
  admission/reuse with an unchanged dirty suffix and neighboring slot,
  oversized-source rejection and prompt/source-bound rejection.
- Thirteen existing Paddle scheduler CPU regression tests passed on the server.
  Local authoring tests: 29 passed,4 torch-dependent tests skipped; the four
  were exercised on NPU as described above.
- Initial test-only launch failed before allocating tensors because NPU backend
  registration followed device construction. Commit 8b60af95 fixes that order.

These results validate 910B only. No 310P execution or new ground-truth accuracy
score is claimed. Exact saved-output parity preserves the baseline outputs.
See command.txt, comparison.json, smoke64.json and prediction_comparison.json.
Full raw traces/logs/page artifacts remain at the remote output root in
command.txt; aggregate evidence only is checked in here.
