# Matched B4 decode: maximum in flight 3 versus 4

Source `1e32b233`, one Ascend 910B2, physical NPU6. Actual ordinary crop API,
optimized decoder, KV4096, selected native 16384 vocabulary, exact CPU lookup
and background image preparation, smaller prefill buckets, weight-padded vision
and joint FP32 RoPE. Normal decode control and device timing are enabled.
Neither experimental event removal nor compact control is selected.

Both runs use the exact frozen random100 seed1 membership/order, already-cropped
images preloaded before submission, and full submission-to-response latency.
The same server runs C3 then C4. Only client max-in-flight changes: B4 is fixed.
No collection-to-fill rule, synchronized completion, admission-delay subtraction,
oracle routing, reused outputs/features/KV, or quality tradeoff is introduced.

| Metric | B4 / C3 | B4 / C4 |
|---|---:|---:|
| Completed tables/s | 3.6711111239184526 | 4.339326623943828 |
| Mean | 0.7850533229822758 s | 0.871326564514311 s |
| P50 | 0.5324899100232869 s | 0.5471028130268678 s |
| P90 | 1.5428502660943209 s | 1.7981259426218459 s |
| P95 | 2.214141868660226 s | 2.559495626681018 s |
| P99 | 4.971084423201392 s | 5.5192258579365445 s |
| Maximum | 5.085484492010437 s | 5.570783469942398 s |
| Measured wall | 27.239709348068573 s | 23.04505022696685 s |

All 100 requests in each run finish with EOS; C3/C4 are 100/100 native-token
identical with matching inputs and real-token counts. C4 gains ~18.2% throughput
but raises P95 ~15.6%. Both miss the P95 goal, so no validation gate follows.

The initial API initialization needed a B4 graph compile (decode setup
13.962954 s). That setup-only instance received no requests and was stopped.
The measured server loaded the cache, received one complete `--set warm`
request outside timing, then the two tests. Both setup lifecycles are saved;
there was no result-dependent restart or discarded measured run. Service
lifetime metrics include warmup and both tests, unlike each client's timing.

`analyze.py` validates identical configurations, frozen selection hash,
request-count/outputs and actual client concurrency over the entire pipeline.
Direct-host evidence maps measured worker1808655 to container2483736 and the
owned API parent1808653. Approximately five-second device snapshots contain
only this worker during the two measurements (5/4 samples). This is sampled
monitoring, not continuous tracing. Setup worker1806403 had already exited.
All processes exit zero. At 2026-09-06 03:29:30 CST the host confirms NPU6 free
and the measured server/worker gone; owned monitor1806104 was stopped.
