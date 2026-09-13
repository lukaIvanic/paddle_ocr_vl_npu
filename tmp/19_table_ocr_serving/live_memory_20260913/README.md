# Live memory, native-output comparison, and retention fix

Measured **before** changing the runtime, on the user's existing warmed server:
Ascend **910B2, physical NPU6**, runtime `59392313`, B8, selected vocabulary
60,416, Kornia-RS/uint8, KV4096, detailed logs. Container worker PID 124053,
HTTP PID 124050, host NPU process PID 334505. The server was not restarted.

Replayed the exact saved 1,000-table / 6-QPS schedule three times. Each client
awaited all responses, followed by a 20-second idle period. A 20-second idle
baseline preceded the first round. Client crop preparation is outside each
measured arrival window. This is a warmed, long-lived process experiment,
not three fresh-start benchmarks.

## Actual host RAM and HBM

142 samples: Linux `/proc/PID/smaps_rollup` for the HTTP process, worker and
server descendants; `npu-smi info` for physical NPU6. The sampler does not import
torch or read device tensors. It checks that the only NPU6 process remains
PID 334505 on every sample. No ownership changes or monitor errors occurred.
Sampling waits five seconds after each sample; actual intervals also include
the time taken by smaps/npu-smi. Short-lived peaks between samples can be missed.

Numbers below are medians of the idle samples at each boundary. Host RAM uses
**MiB** (1,024 KiB); NPU memory retains the tool's reported **MB** label.

| After additional requests | Worker RSS (MiB) | HTTP RSS (MiB) | NPU process memory (MB) | Device HBM (MB) |
| --- | ---: | ---: | ---: | ---: |
| 0, idle baseline | 5,734.8 | 137.8 | 7,439 | 10,798 |
| 1,000 | 5,830.9 | 137.9 | 7,439 | 10,798 |
| 2,000 | 5,806.8 | 137.8 | 7,439 | 10,798 |
| 3,000 | 5,837.8 | 137.3 | 7,439 | 10,798 |

Worker RSS ended **103.0 MiB above baseline**; sampled peak increase was
**162.8 MiB**. The HTTP process did not retain a rising resident footprint.
NPU process memory stayed at 7,439 MB in every sample; total device HBM varied
between 10,797 and 10,798 MB, with no measured upward trend.

Host RSS is **not monotonic**: it falls after round 2. These measurements cannot
identify every allocation's owner or justify a reliable time-to-10-GB forecast.
The earlier 12.6-KB/request estimate measures reconstructed retained Python
objects, not live RSS. Allocator reuse/reclamation and temporary/native
allocations can change RSS independently. We do not attribute the entire
103-MiB difference to the completion list, or claim an HBM leak.

Raw per-process RSS/PSS/private-page readings, timestamps, phase labels and
original npu-smi output are in `before/memory.jsonl`. Summed server-tree PSS is
also retained: it avoids double-counting shared pages as a summed RSS would,
although sharing with other processes can still change its value.

## Native-output and performance comparison

The user's completed run (`tmp/manual-poisson1000.Yrltbw/results`) and all
three measurement rounds each have **1,000/1,000 exact matches** against the
accepted integrated-runtime 1,000-request reference for:

- Saved schedule/table occurrence order.
- Native generated token IDs (not retokenized text).
- Raw text and formatted output.
- Stopping reasons, crop dimensions, prompt tokens and projected image tokens.

Each run has zero request failures, 990 EOS completions and the same ten
KV4096-limit completions. No fresh ground-truth scoring was needed to establish
output parity with this reference; this is not an independent quality test.

| Run | Client mean (s) | Client P95 (s) | Completed requests/s |
| --- | ---: | ---: | ---: |
| Accepted integrated-runtime reference | 1.239532 | 3.709361 | — |
| User's manual run | 1.207931 | 3.663674 | 5.734465 |
| Memory round 1 | 1.202130 | 3.667519 | 5.734556 |
| Memory round 2 | 1.205359 | 3.685428 | 5.733772 |
| Memory round 3 | 1.215049 | 3.670204 | 5.734488 |

These use `request_latency_s` (client attempt through response), rather than
the slightly larger scheduled-arrival latency printed by the client's final
DONE line. Fixed-load completion rate is not maximum serving capacity.
Full comparisons and source-result hashes: `before/comparison.json`.

## Fix, deliberately made after the measurement

Commit `db0a3f92` removes the serving-lifetime completion list and submitted-ID
history. It updates request/token totals when each crop finishes and retains
only the IDs of unfinished requests. Duplicate checks therefore cover unfinished
work, not completed historical IDs. The public interfaces already assign a
unique internal ID per request; reusing a client-visible ID remains supported.

Delivered results retain their own token lists; the fix does not clear or mutate
response data. No explicit garbage collection is added. Summary calculations,
stopping rules, NPU operations, cache admission, transfers and synchronization
are unchanged. Only the main loop method's bookkeeping changes. p03–p06 are
byte-identical to the measured source, including the compiled-graph identity.

The new CPU regression test observes weak references to completed records
**while serving is still running**, with first-token EOS and normal decode/slot
reuse. It checks that the engine stops retaining previous completions and that
summary request/token counts remain exact. This is not an after-fix live-RSS
measurement: the NPU runs above intentionally use the old code.

The fix passed **100 CPU tests** on Python 3.12.13 / Torch 2.10.0. The first
attempt could not resolve historical Git fixtures because the detached worktree
used host-absolute Git paths inside the container. Setting `GIT_DIR` and
`GIT_WORK_TREE` to the container paths resolved this without source changes.
Both logs are retained; `cpu-tests-memory-fix.log` is the successful run.

## Product-facing follow-up

Commit `cbb31f14` adds the configurable `heartbeat_interval_s` / CLI
`--heartbeat-interval-s`, the purpose-first vocabulary explanation, and the
self-contained `benchmark_tables.py`. The user's shortened README is preserved
with curl examples followed by 100/1,000-table commands. Requirements now include
the installed Torch/Torch-NPU versions and omit unused direct dependencies.

The new client was exercised with the real OmniDocBench dataset and the same
running 910B2 server: **100 successful requests, zero failures**, maximum 14
concurrent requests, mean 1.092 s, P95 2.929 s. This validates the client and its
data handling, not an after-fix memory run. The new seeded schedule is deliberately
self-contained; it does not claim to reproduce the historical chart's order.
Both the schedule and raw results are retained under `product_client100/`.

Local CPU tests cover dataset cropping, ignored regions, balanced selection,
repeatable arrivals, percentiles, concurrent HTTP/error handling, and warmup
exclusion. API tests cover the configurable heartbeat during load/idle and reject
nonpositive/nonfinite intervals. The complete remote suite initially flagged
one historical expected-CLI-field list; `695a94ee` adds the new supported field.
The final rerun at `695a94ee` passes **all 106 CPU tests** on Python 3.12.13 /
Torch 2.10.0. See `cpu-tests-106.log`; the earlier fixture failure is preserved
separately in `cpu-tests-cli-field-before-update.log`.

## Evidence and reproduction

- Measurement/comparison scripts: `../handoff_audit_20260913/measure_live_memory.py`
  and `summarize_live_memory.py`.
- Each round's exact command, complete client log, schedule, results and summary:
  `before/round1`, `before/round2`, `before/round3`.
- User's original results retained at `tmp/manual-poisson1000.Yrltbw/results`.
- Download archive SHA256:
  `eb9bf14fa41f2642bfec74add76cfb1fc5000b00c622b1d7b662f0a12bef57e9`.

The user's HTTP server was left running, still on its original loaded source.
The main remote source checkout was fast-forwarded to the tested `695a94ee`;
restart the server for the fix and heartbeat option to take effect. The product bundle
and additional text/formula/Python NPU tests remain outside this task.
