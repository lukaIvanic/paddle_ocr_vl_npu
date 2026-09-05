# Smaller full-table prefill buckets: real ordinary serving, 910B2

Source `aecebe0d`. One physical NPU6, one persistent server at a time.
Ordinary B4 decoding, client maximum three outstanding requests (B4/C3).
No speculative decoding, oracle routing, image rotation or benchmark-identity
lookup is involved in this serving path. The source ID is bookkeeping only.

Same frozen random-100 seed1, selection SHA256
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
Each fresh server receives the same one complete `--set warm --count 1`
request outside measurement, then the random-100 workload. Server initialization
uses the existing runtime's per-shape compile/cache-load invocation; no separate
lab warmup/build program was run. Every response is saved immediately.

## The only configuration change

- Control vision buckets: `4096`; text-prefill buckets: `1152`.
- Candidate vision buckets: `256,384,512,640,768,1408,1920,2048,2944,4096`.
- Candidate text-prefill buckets: `128,256,512,1024,1152`.

These are existing runtime options and existing compiled shapes, selected by
the real input length. Images, min/max pixels (28,224/802,816), FP16 checkpoint,
IncreFA decode preset, selected-vocabulary native ID map, KV4096, stopping rules,
B4 decode and C3 admission are unchanged. No request waits to form a prefill batch.
`command.txt` records the shared command and the two configurations.

## Results

| Metric | Fresh control | Smaller buckets |
|---|---:|---:|
| Completed tables/s | 2.915749957618936 | 3.243711481695539 |
| P95 client wall latency | 2.658448041841620 s | 2.396639176888857 s |
| Total measured wall | 34.29649368207902 s | 30.82888245896902 s |
| Vision transformer device total | 6.195001 s | 4.248541 s |
| Text transformer device total | 1.404731 s | 1.117610 s |

Use the JSON's unrounded values as authority. Device stage totals are diagnostic
and are not added to/subtracted from client latency. The larger wall saving also
includes host scheduling effects; it is not attributed entirely to device compute.
The fresh control closely reproduces the historical B4/C3 2.916 tables/s,
P95 2.655 s result.

**This does not meet the goal:** throughput passes, P95 does not. Do not run the
second-100 or 1,000-request validation gate yet.

## Validity and cleanup

`analyze.py` regenerates `comparison.json` from the original client records:

- All 100 requests finish with EOS in each run; all 100 native outputs match.
- Per-table crop size, real vision tokens, real text tokens and projected image
  tokens are identical. Same client selection/crop-payload code, no resampling
  setting change; no request input work was moved outside the existing timer.
- Vision physical slots fall from 409,600 to 270,592; real tokens stay 243,600.
- Client submission/completion events verify at most three outstanding requests.
  Because the server only receives these requests, the whole pipeline cannot
  process more than three measured requests at once. Server decode still uses B4.
- Direct-host process snapshots bracket both measured windows, with only the
  owned worker PID in every measured snapshot: control 1748367, candidate 1751056.
  Sampling approximately every five seconds cannot prove absence of an arbitrarily
  short inter-sample process; it is process monitoring, not continuous kernel tracing.
- Both clients and both servers exit zero. Servers stopped after saving their
  final summaries. Direct host confirmed NPU6 free at 2026-09-06 01:37:52 +08:00.
  The owned monitor (PID1748092) was stopped afterward. No benchmark remains running.

No production defaults were changed by this experiment. The measured configuration
is retained here for adoption in the next candidate.
