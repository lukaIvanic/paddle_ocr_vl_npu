# Synthetic graph warmup only: first versus second Poisson100 pass

## Protocol

One fresh server process, one physical Ascend 910B2 (NPU6), B8, 60,416-row
vocabulary, Kornia-RS/uint8 preprocessing, KV4096, detailed instrumentation.
Runtime commit: `e0a7b3e5f6a1fd79e360a99b4e72fd55d9e5a611`. Its six production
files are unchanged from the accepted integrated runtime at `935dbb8f`.
The local, uncommitted HTTP/Python API changes were deliberately NOT deployed.

Existing synthetic graph calls remain unchanged: ten vision buckets, five text
prefill buckets and one B8 decode entrypoint. This is a cached-graph startup,
using namespace `270729d5a752`, not a fresh graph-compilation experiment.
Model setup reported 34.091 seconds (excluding interpreter/import startup).

No real-request warmup. Replay the saved 100-table order and exact 6-QPS Poisson
arrival timestamps, await all responses, then replay on the same server.
The client generates/loads its inputs before timed arrivals; the second client
does that normally after the first has exited. No explicit drain command, server
restart, EOS override or production warmup implementation was introduced.
The service summary and HTTP log both contain exactly 200 requests.

Both runs offered arrivals across 16.8519622975 seconds. The client measures
actual-request-attempt-to-response latency, including connection establishment;
scheduled-arrival latency and dispatch lag are recorded separately.

Run on the bare-metal host, after pulling the committed benchmark harness:

```sh
python3 -u /data1/lukaiv/workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/refactor_poisson100_20260913/driver.py \
  --expected-commit e0a7b3e5f6a1fd79e360a99b4e72fd55d9e5a611 \
  --npu 6 --compare-first-use \
  --output-dir tmp/19_table_ocr_serving/first_use_poisson100_20260913/run
```

The driver refuses an existing output directory. Use a new directory for a
repeat and pin the actual deployed commit. Exact server/client commands and
readiness configuration are retained under `run/b8/`.

## Results

| Metric | First pass: graph warmup only | Second pass: same server |
| --- | ---: | ---: |
| Mean request latency | 1.220951 s | 1.166608 s |
| P95 request latency | 2.983749 s | 2.967298 s |
| Maximum request latency | 7.883175 s | 7.645826 s |
| Completed requests / full run wall time | 5.240471/s | 5.263577/s |
| Successful requests | 100 | 100 |
| Errors | 0 | 0 |
| Peak outstanding requests | 18 | 17 |

All 100 native token streams, raw texts, formatted HTML outputs, stopping
reasons, input token counts and image token counts match between passes.
Both also match the accepted integrated Poisson100 reference exactly; see
`run/first_vs_accepted.json` and `run/second_vs_accepted.json`. All requests
finished with EOS. No new ground-truth scoring was needed for this unchanged
output comparison; this is not a new full-benchmark accuracy claim.

## Individual differences

Positive differences below mean the first pass was slower.

- Mean paired difference: +54.344 ms; median +15.396 ms; P95 +204.286 ms;
  maximum +780.911 ms. P95 of paired differences is not the difference between
  the two runs' P95s (+16.451 ms).
- First ten requests: +215.029 ms average; remaining ninety: +36.490 ms.
- 24 requests were more than 100 ms slower on the first pass; 6 were more than
  200 ms slower. Fifteen were faster on the first pass.
- The largest difference is request 1, `page_000287_table_box_id_8`:
  5.268237 s versus 4.487326 s, with identical generated tokens.

Request 1's recorded stages:

| Stage | First pass | Second pass |
| --- | ---: | ---: |
| Background CPU preparation service | 131.064 ms | 49.604 ms |
| Vision + text prefill wall time | 198.923 ms | 72.488 ms |
| Time to first token | 332.961 ms | 124.114 ms |
| Vision embedding event region | 95.253 ms | 0.737 ms |
| Vision input preparation event region | 26.117 ms | 1.213 ms |
| Compiled vision prefill event region | 52.582 ms | 54.067 ms |
| Compiled text prefill event region | 12.860 ms | 13.261 ms |
| Decode slot residency | 4.918874 s | 4.351627 s |

These regions overlap with their parent timings; do not add all rows together.
Event-region duration is not necessarily pure kernel execution time: host
submission gaps can occur inside it. This test does not isolate the exact
first-use mechanism or prove that a particular operator compiled lazily.

There is measurable first-use work outside the warmed graph bodies, but no
multi-second first-prefill stall in this run. The first request's +209 ms time
to first token is clearer evidence of that work than assigning its entire
+781 ms end-to-end difference to startup. Its subsequent decoding overlaps
other crops, and changed timing changes contention and queueing. A single
paired run cannot separate all first-use effects from ordinary run variation.

This supports keeping graph-only warmup if that initial-request penalty is
acceptable. It does not establish zero first-use cost or cover untested inputs,
full-vocabulary mode, different batch sizes, or fresh graph compilation.

## Evidence and analysis

- `run/per_request_latency.md`: all 100 paired request latencies in arrival order.
- `run/paired_comparison.json`: paired latency, dispatch and per-stage timings.
- `run/b8/{first,second}/results/`: original requests, schedules and summaries.
- `run/b8/server.log`: setup and exactly 200 HTTP OCR responses.
- `run/ownership.jsonl`: 43 ownership checks, no foreign NPU6 process; final
  check shows no process on NPU6 after our server shut down.

Remote result root:
`/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910/tmp/19_table_ocr_serving/first_use_poisson100_20260913/run`.

Retrieved archive SHA256:
`87bf922633cbe7fa7a57cd33b59d59d72a893e7f7f1f06ac0b4968afd6abcd27`.

Recreate the paired report locally from the retained results:

```sh
python3 tmp/19_table_ocr_serving/first_use_poisson100_20260913/analyze.py \
  --root tmp/19_table_ocr_serving/first_use_poisson100_20260913/run
```
