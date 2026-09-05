# Keep ordinary decode running during CPU preparation

Source `02584647`, Ascend 910B2, physical NPU6. Same frozen random-100 seed1
and smaller prefill buckets as `table_prefill_buckets_aecebe0d_20260905`.
Model/greedy selection, images, real tokens, KV4096 and output/stopping limits
are unchanged. No speculative decoding or oracle/source-ID routing is used.

The runtime previously submitted CPU preparation to a background worker and
immediately called `future.result()` even for a non-blocking source poll.
It now retains unfinished preparation and returns `None` on non-blocking polls.
The scheduler continues decoding and retries; idle blocking pulls still wait.
CPU work never adds an extra request beyond the client's whole-pipeline cap.

| Same development set | Completed tables/s | P95 client latency |
|---|---:|---:|
| Previous B4/C3, smaller buckets | 3.243711481695539 | 2.396639176888857 s |
| Non-blocking CPU, B4/C3 | 3.4967696410273454 | 2.2994508231990034 s |
| Non-blocking CPU, B2/C2 | 2.808024198737056 | 1.9147474726370988 s |

Neither new run meets both targets. No second-100 or 1,000-request gate was run.
The historical B2 control was 2.4832859099 tables/s and P95 2.151053 s; it
predates both smaller buckets and the CPU-poll fix, so it does not isolate this fix.

All 100 outputs match their respective controls' native token streams exactly.
All 100 complete with EOS; image sizes and real vision/text token counts match.
`analyze.py` verifies the selection hash, completion set, client outstanding
limit, real input counts, native outputs, and the NPU6 process-monitor samples.
Full request latencies include all CPU work, idle waits, prefills and decoding.
Stage/CPU totals are diagnostics, not amounts subtracted from request latency.

For B4/C3, inference-thread CPU-future wait totals 0.000380 s, down from
4.248183 s. CPU preparation service itself totals 5.250199 s versus 4.536122 s
before: background work is not free, and host contention/batch occupancy change.
The measured wall saving is therefore smaller than the removed wait sum.

Each independent server used one complete `--set warm --count 1` request outside
timing, followed by the random-100 client. C3 was stopped and NPU6 released before
launching B2. Both clients and servers exit zero; both service summaries are saved.
The host monitor brackets the measured windows and reports only the corresponding
worker PID within each: 1760664 (C3), 1770509 (C2). Snapshots every approximately
five seconds cannot rule out arbitrarily short inter-sample activity. A MinerU
profile appears earlier in the log, before our C3 server was started; our run waited
for it to finish and NPU6 to become free. We did not modify it.

Both table servers were stopped after their results were saved. Direct host
confirmed NPU6 free at 2026-09-06 01:59:59 +08:00; the owned monitor PID1756377
was stopped afterward. The user's later hardware amendment permits another
verified idle card, with NPU6 preferred if another card gives anomalous results.
These measurements all remained on NPU6.

## Subsequent CPU-only normalization probe

`normalization_probe.py` / `normalization_cpu_probe.json` evaluate the bit-exact
lookup-table normalization candidate from `b66fc153` on the first eight requests
of the same development selection. This candidate was NOT present in the C2/C3
serving results above. It retains Pillow resizing and the reference FP32 arithmetic
rounding for every uint8 channel value. Complete resize/normalize/patchify outputs
were bit-exact for all eight. With two warm pairs and five alternating measured
pairs, the sum of per-crop medians was 0.333624 s reference versus 0.273601 s lookup.
That 18% CPU-preprocessor reduction is not a serving-latency result. The crop-PNG
payloads were prepared before this CPU-only timing, and no model or NPU was run.
