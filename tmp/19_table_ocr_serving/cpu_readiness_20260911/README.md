# CPU readiness on the prefill critical path

## Scope and reproducibility

Runtime `ec75ac3033faa9937fbccb5fff263620da9badc4`, Ascend 910B2 physical
NPU6, 2026-09-11. Current experiment-19 default 60,416-token decode head.
One background CPU-preparation worker; unchanged preprocessing and scheduler.
Closed-loop B2/C2 and B8/C8, 100 requests each, identical random-100 order to
the saved 60k comparison in `../lm_head_60416_20260911`.
Ordered-ID SHA256: `944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b`.

`driver.py`, per-run `server_command.txt`, client `command.txt`, plans and
identities preserve the launch details. A complete real-request warmup is
outside each measured client window. Device timing was enabled as in the
control. Historical directory labels `expanded` refer to the current default
60,416-token head, not an additional supported configuration.

## What was measured

`scheduling_metrics.cpu_readiness.prefill_blocked_s` is the wall time from the
first scheduler observation that this request is FIFO head with an unreserved
decode slot, until its CPU inputs finish preparation. Already-ready inputs
contribute zero. It excludes time waiting for occupied slots or earlier
requests' prefills. CPU service covers image decoding, image/prompt preparation,
mRoPE construction and pinning; it excludes device transfers and prefills.

The metric splits CPU-executor queue time from CPU service. A separate
`scheduler_idle_blocked_s` subset starts when the scheduler has no active or
ready decode request and makes a blocking pull. This is not a device-traced
idle measurement: outstanding device work can still overlap the host span.

These are direct readiness delays, **not exact counterfactual E2E savings**.
Faster CPU preparation changes prefill interruptions, batching and subsequent
scheduling. Do not subtract these times from request latency to predict a new
distribution. The eligibility boundary is observed at scheduler polls, and
propagated FIFO delays to later requests are not separately attributed.

## Results (seconds)

All distribution statistics include all 100 requests, including zeros.
Percentiles use linear interpolation at `(n-1)*p`.

| Configuration | Requests delayed >0.001 s | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| B2/C2 | 95/100 | 0.054530 | 0.046225 | 0.113368 | 0.123597 | 0.332935 |
| B8/C8 | 72/100 | 0.038645 | 0.031628 | 0.083080 | 0.096229 | 0.270199 |

Among only affected requests, mean delays were 0.057400 s and 0.053673 s.
Almost all direct delay was CPU service, not executor queueing: the direct
queue component averaged 0.000173 s (B2) and 0.000008 s (B8).
Total CPU service averaged approximately 0.0587 s / 0.0585 s respectively.

B8's *total* executor queue wait averaged 0.034490 s, but almost none remained
once that request became the FIFO head eligible for prefill. Total queue wait
and exposed readiness delay must therefore not be conflated.

| Scheduler-empty subset | B2/C2 | B8/C8 |
|---|---:|---:|
| Requests with >0.001 s wait | 9 | 1 |
| Mean per measured request | 0.003033 | 0.000144 |
| P95 | 0.027114 | 0 |
| Maximum | 0.068041 | 0.014405 |
| Total | 0.303313 | 0.014405 |
| Fraction of measured client wall | 0.8873% | 0.0855% |

CPU readiness can delay an individual request while other requests continue
decoding. These results show little complete scheduler starvation; they do not
prove that CPU preparation has no throughput effect through slot underuse.

## End-to-end tails and controls

| Configuration | Tables/s | E2E mean | E2E P95 | E2E max | Mean CPU-readiness delay among five slowest requests |
|---|---:|---:|---:|---:|---:|
| B2/C2 | 2.925457 | 0.670790 | 1.934260 | 4.288241 | 0.089680 |
| B8/C8 | 5.939070 | 1.197974 | 3.290554 | 8.569607 | 0.045941 |

The largest CPU delay belonged to `page_000188_table_box_id_6`: 0.332935 s
at B2 and 0.270199 s at B8. Its E2E latency was only 0.629023 / 0.777318 s;
the worst CPU case was not an E2E tail case. Raw per-request timings, including
the five slowest E2E and largest CPU delays, are in `analysis.json` and
`per_request.json`.

Both runs had 100/100 successes, matching native token streams, image/input
counts and completion reasons against the previous same-head run. Ownership
monitoring recorded 37 / 34 clean snapshots, including final release. Direct
host inspection after completion also found no process on NPU6.

No model graph source was changed for this instrumentation. Cached decode
startup was 0.210247 / 0.229879 s, with no fresh-compilation delay observed.
The B8 throughput difference from its prior 5.787574 tables/s control is not
an optimization result: this change adds measurement only.

## Analysis

Run `python3 tmp/19_table_ocr_serving/cpu_readiness_20260911/analyze.py`.
It checks sample alignment, CPU timing identities, completion reasons, native
outputs, ownership and final release, then regenerates the analysis files.
Raw archive SHA256:
`82b9077e0fdbc2567ed577c6ba2e8b7db92989d29076c1e1dd4c3f387ab80580`.
