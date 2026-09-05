# Exact B3 decode, C3 admission: no useful tail improvement

Source `b0a0f44d`, Ascend 910B2 physical NPU6. Same frozen random100 seed1
as the matched B4/C3 run at `table_b4_c3_c4_1e32b233_20260905`.
Only static decoder batch size changes from four to three. Batch-derived
allocation/reservoir capacities follow that size; client admission stays C3.
All inputs, real vision tokens, model weights, selected native vocabulary,
greedy policy, KV4096, output cap and prefill buckets are preserved.

The motivation was to remove B4's unused fourth row at C3, without reducing
admission concurrency. The original engine constructor rejected non-power-of-two
batches before model loading. Inspection found no corresponding ordinary
IncreFA/arena dependency. The positive-size validation replaces that historical
restriction; the actual scheduler CPU suite passes 12 tests, including B3
refill/completion and fixed-shape/token accounting.

## Real server result

| Metric | Prior B4/C3 | Exact B3/C3 |
|---|---:|---:|
| Completed tables/s | 3.6711111239184526 | 3.713472279160384 |
| P95 response wall seconds | 2.214141868660226 | 2.246692303806776 |
| Mean seconds | 0.7850533229822758 | 0.7757340987597127 |
| P99 seconds | 4.971084423201392 | 4.857735629737145 |
| Maximum seconds | 5.085484492010437 | 5.066967656952329 |

100/100 complete with EOS, no errors/unsent requests, and 100/100 native token
streams match B4/C3. Peak outstanding requests is three. The development gate
still fails P95; neither independent100 nor validation1000 was run.

The service-lifetime model/argmax event average is 1.268 ms/call for B3,
versus 1.313 ms/call for the saved B4 server. These scopes include warmup;
the B4 lifetime includes both C3 and C4 runs, so this is supporting diagnostic
evidence, not a matched microbenchmark. The small graph gain does not produce
a meaningful end-to-end tail gain. B3 is not promoted as the winning route.

## Setup and ownership

The first B3 setup compiled through the actual API runtime and completed in
91.773219 s. That setup-only instance served no requests, was shut down cleanly,
and was restarted from cache (34.658416 s setup). One complete warm request
outside timing preceded the measured 100-request client run. Initial fill and
final drain remain timed. All progress/results were saved incrementally.

Host PID1820267/container2486891 owned the setup-only instance. The measured
worker was host1822896/container2488522, parent1822894/container2488520.
Direct-host ownership checks and five-second samples found only our worker
on NPU6. This is sampled monitoring, not continuous proof. Both instances
exited cleanly; at 2026-09-06 03:59:49 CST the host showed NPU6 free and the
measured worker/parent gone. Owned monitor1820085 was stopped. No other card
or user's process was used or modified.

`analyze.py` audits the frozen manifest, all requests, concurrency, selected
runtime settings, native outputs, real tokens and monitoring evidence; it
writes `comparison.json`. Commands, stage logs, full results and service
summaries are retained alongside it. No profiled latency is presented as a
qualifying serving result.
