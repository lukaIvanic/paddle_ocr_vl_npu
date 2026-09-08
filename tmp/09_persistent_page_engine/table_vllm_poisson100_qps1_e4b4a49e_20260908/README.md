# vLLM-Ascend Poisson 100-request control

2026-09-08, one Ascend 910B2 (physical NPU6), source e4b4a49e.
This is an existing-software anchor, not a maximum-capacity measurement.

The saved first 100 tables and arrival timestamps from the custom frontier's
1,000-request B2 / 1-QPS schedule were replayed exactly. Membership, metadata,
order and timestamps compare equal. This is not the older random-100 cohort.
There is no client concurrency cap. Server max sequences is 16, shared
batched-token budget 4096, per-request context 4096. Async scheduling and
chunked prefill are explicitly enabled; FULL_AND_PIECEWISE is retained.

| Metric | Result |
|---|---:|
| Completed / submitted | 100 / 100 |
| Errors | 0 |
| Target Poisson QPS | 1.000 |
| Completed requests / full wall time | 0.917276 |
| Mean request latency | 3.627446 s |
| P50 | 2.303079 s |
| P90 | 7.255522 s |
| P95 | 11.570333 s |
| P99 | 21.112396 s |
| Maximum | 24.449546 s |
| Measured wall time including drain | 109.018387 s |
| Maximum client outstanding requests | 9 |
| Maximum client dispatch lag | 0.002992 s |

Latency is actual request attempt through complete HTTP response, including
connection establishment. Scheduled-arrival latency is retained separately.
Image crop PNG/base64 preparation occurs before the measured arrival window,
as in the earlier comparison contract. Every response and native token stream
is saved immediately in measured/results.jsonl.

## Native counters, excluding warmup

Differences between metrics_after.txt and metrics_before.txt:

| Metric | Mean per request |
|---|---:|
| Scheduler queue time | 0.00001226 s |
| Time to first token | 0.219705 s |
| Prefill time | 0.120521 s |
| Decode-phase elapsed time | 3.401569 s |

Zero preemptions; 36,429 generated tokens. TTFT contains prefill, and
decode-phase elapsed times include scheduling interference: these are not
isolated NPU kernel timings and should not be added as disjoint stages.

The maximum 9 outstanding requests is below the server's 16-sequence cap.
The bottleneck at this offered load is therefore predominantly decode-phase
latency, not scheduler admission queueing. This does not establish the outcome
of changing the token budget at a higher offered rate.

## Validity and setup

Both prefix and multimodal processor caching are disabled. The installed
renderer explicitly makes request-specific image UUIDs in this configuration,
preventing reuse of image embeddings between repeated inputs. Effective
processor min/max pixels were checked as 28,224 / 802,816 despite a warning
at another processor layer. Global KV allocation keeps the previous 0.92
memory-utilization setting; it is separate from the 4096 per-request limit.

Real-request warmups completed before measurement. Startup captured 16 mixed
prefill/decode and 16 full-decode graphs. Native logs retain setup/compilation.
The monitor recorded only our EngineCore host PID 1513788 on NPU6. API host PID
1509637 / container PID 3227 was stopped after result collection; a direct
host check at 18:02:30 CST confirmed no process on NPU6. No other jobs were stopped.
