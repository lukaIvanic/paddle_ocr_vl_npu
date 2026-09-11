# B8, 6-QPS Poisson, 100 tables: preprocessing A/B

2026-09-11, Ascend 910B2 physical NPU6, runtime/launcher commit
`24f07aedae467b032c14ee263a0b8cea165071f4`.

Two sequential runs compare the former Pillow/CPU-float32 preparation with
the new Kornia-RS/uint8 defaults (NPU normalization). Same HTTP endpoint,
60,416-token head, KV4096, pixel limits 28,224–802,816, eight decode slots,
encoded-image uploads, compiler caches and instrumentation. One complete real
warmup request per server is outside measurement. The benchmark-only wrapper
selects existing constructor options; no scheduler or model operations changed.

The client replays `schedule.jsonl`: the exact prior random-100 table order,
with 6-QPS exponential interarrival draws from seed 1. The last arrival is
16.851962 s after start, giving this realization 5.934027 scheduled arrivals/s.
The same table IDs **and timestamps** were verified in both saved schedules.
Order SHA256: `944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b`.

The current committed Poisson client adds saved-schedule replay support to the
frozen-client machinery used for server lifecycle and warmup. Exact commands
are saved under each run. Although inherited driver status text says B8/C8,
there is **no client concurrency cap**: eight is the server's decode capacity,
not a limit on outstanding HTTP requests. Maximum outstanding was 18 in both
runs; each had 14 actual arrivals in its busiest rolling one-second window.

## Results

Seconds unless stated otherwise. Main latency is actual request-attempt to
complete response, including connection establishment and server queueing.

| Metric | Former preprocessing | Kornia + uint8 |
|---|---:|---:|
| E2E mean | 1.334952 | 1.221411 |
| E2E P50 | 0.970338 | 0.932179 |
| E2E P90 | 2.438668 | 2.245760 |
| E2E P95 | 3.465478 | 3.127210 |
| E2E P99 | 7.670305 | 6.963397 |
| E2E maximum | 8.407342 | 7.835798 |
| Scheduled-arrival P95 | 3.467542 | 3.127526 |
| Completed tables/s including drain | 5.213921 | 5.242709 |
| Drain after last scheduled arrival | 2.327463 | 2.222147 |
| Mean CPU preparation service | 0.060993 | 0.038395 |
| Mean CPU-executor queue wait | 0.028719 | 0.012467 |
| CPU-executor queue wait P95 | 0.183917 | 0.075573 |
| Mean exposed CPU-readiness delay | 0.026838 | 0.020544 |
| Exposed CPU-readiness delay P95 | 0.116728 | 0.081381 |

Mean E2E fell 8.50%; P95 fell 9.76% (0.338268 s). This is a single short
screening pair, not a sustained-load capacity measurement or a larger
validation. Completion throughput is constrained by the fixed arrival schedule
and includes the drain; it must not be called maximum supported QPS.

Client dispatch lag stayed small: mean 0.000865 / 0.000802 s, maximum
0.002557 / 0.002206 s. Scheduled latency independently confirms the improvement.
CPU-readiness delay is observed FIFO-head/free-slot waiting for CPU inputs, not
an exact counterfactual E2E saving. Faster CPU also changes admission timing,
decode batching and interruption patterns; do not subtract stage totals from E2E.

## Checks

- 100/100 successful requests in each run; all EOS, 41,120 generated tokens.
- **100/100 native token streams match** between the two runs.
- Identical per-request crop size, input-token count and projected image-token
  count. No request was dropped and no pixel/output limit was reduced.
- 33 + 32 clean ownership snapshots; final snapshots and direct-host `npu-smi`
  both confirm NPU6 released. All benchmark servers stopped.
- Cached startup was used; no fresh model-graph compilation delay was observed.
- No pixel-limit or decoded-PIL-input experiment is part of this comparison.

`analyze.py` regenerates `analysis.json` from raw per-request results and checks
the exact schedule, outputs, shapes, errors and ownership. Readiness,
server/client commands, stage logs, service summaries and all responses are
retained. Archive SHA256:
`e51bf70eec56e611d991e03e5112a99a4b125bedc65d51c39bb87636229c6107`.
