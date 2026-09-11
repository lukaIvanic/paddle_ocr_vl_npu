# Step 3: cleaned serving implementation, flagship validation

Completed 2026-09-11. Runtime source: `0976fa33367f33ba9051d47bcb1e9a2a2b20ffaa`.
One Ascend 910B2, physical NPU6. No inference-code fix was needed during testing.

## Results versus cached step 2

| Metric | Step 2 (`dc755584`) | Step 3 (`0976fa33`) |
|---|---:|---:|
| Mean request latency (s) | 1.283549087 | 1.317757560 |
| P50 (s) | 0.862832514 | 0.919765120 |
| P90 (s) | 2.713301178 | 2.747258450 |
| P95 (s) | 3.779593972 | 3.790191471 |
| P99 (s) | 7.259671545 | 7.264818482 |
| Maximum (s) | 8.545706200 | 8.695539450 |
| Completed requests/s | 5.733948247 | 5.733937215 |
| Errors / requests | 0 / 1000 | 0 / 1000 |
| KV4096-limit stops, retained | 10 | 10 |
| Maximum outstanding client requests | 19 | 20 |

P95 +0.280%, mean +2.665%, P50 +6.598%, throughput essentially unchanged.
Output parity passes and P95/throughput closely reproduce the anchor. The mean
and P50 increases are real observations; one run does not separate a cleanup
effect from runtime variation. No optimization or repeat was added to hide them.
This open-loop replay is not a new maximum-capacity measurement.

## Workload and accuracy

Exact frozen `be691de1` client and benchmark harness, target 6 QPS, B8,
`--cohort all --max-requests 1000 --seed 1 --shuffle-all`. The sequence covers
all 665 tables plus 335 repeated occurrences, globally shuffled. Arrival rate
for the finite schedule is 5.763293819/s, unchanged from the control.

- Schedule files are byte-identical.
- Ordered request-ID SHA256:
  `97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9`.
- Join by occurrence `sequence`, not repeated table ID or completion order.
- All 1000 native token streams, raw text, formatted text/HTML, stop reasons,
  prompts, crop types, input-token counts, projected-image-token counts and crop
  dimensions are identical to the cached step-2 run.
- All responses, including the ten unchanged KV-cap outputs, remain counted.
- No independent Page-TEDS rerun: exact output equality establishes parity with
  the control, not a new ground-truth evaluation.

Latency is actual client submission to complete response. Queueing and server
processing remain included; no subtraction of CPU overlap or interruptions.
The frozen client's source-page crop/PNG preparation takes place before the
arrival clock, as in the historical control. Scheduled-arrival latency is also
retained in raw records. The client DONE line uses its scheduled-latency summary;
the table above uses `request_latency_s`, consistently with step 2.

## Startup and ownership

First process: normal constructor compilation and one complete real warmup;
zero measured requests. Vision setup 199.749 s, text prefill 152.121 s, decode
12.162 s; total recognizer setup 397.929 s. GC froze 1,337,152 objects. Warmup
returned HTTP 200, then the owned process stopped and the device was checked free.

Second process: same code, cached graph startup, same real warmup, then the
1000 requests. Vision setup 6.437 s, text prefill 3.288 s, decode 0.217 s;
total 35.888 s. GC collected 10 objects and froze 633,533 (step 2: 636,274).
No separate synthetic compilation implementation or cache-identity bypass was
introduced; existing constructor setup behavior was preserved.

Measured-process worker host PID181549, server parent179552. All 96 ownership
snapshots contain only owned NPU6 PIDs (or no PID). Final snapshot is free at
2026-09-11 17:56:38 CST / 11:56:38 Europe/Zagreb. Both benchmark drivers exited;
no follow-on experiment was started. Source HEAD and clean tracked experiment-19
state were verified after the run.

## Reproduction and artifacts

- `step3_b8qps6_0976fa33_20260911_compile/`: setup/warmup-only phase and ownership.
- `step3_b8qps6_0976fa33_20260911_cached/`: measured run, complete outputs,
  schedule, readiness, service summary, exact commands and ownership.
- `step3_b8qps6_0976fa33_20260911/driver.py`: phase adapter over frozen harness.
- `step3_b8qps6_0976fa33_20260911/compare.py` and `comparison.json`: comparison.

On the direct host, the driver uses the existing historical checkout
`/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910` for clients and
artifacts, and the main checkout's experiment-19 server. Its source pin is
explicit. Run `--phase compile-warm`, wait for success and device release, then
`--phase measured`. Existing output directories deliberately prevent accidental
overwrites. Exact Docker/server/client commands are saved per phase. Only
entrypoint/ownership names, fingerprint coverage and removed pixel CLI selectors
are adapted; the endpoint hardcodes those same validated pixel values.

Local source was committed/pushed; because remote private-GitHub credentials
were unavailable, the exact Git commit was delivered as a verified bundle and
pulled with `--ff-only`. One premature pull encountered an incomplete transfer
(`early EOF`); it changed no source and launched no model. The completed bundle
was checksum-verified before the successful pull. No tracked remote files were
hand-edited and no packages/drivers were changed.

Evidence archive SHA256 after remote creation and local transfer:
`4457267a75303b038b9b393e77040a13b7853a9d7974d43300c910ed9c858821`.
The unpacked evidence is committed; duplicate archive and source bundle are not.

Readiness differences are the intentionally removed preset/selection metadata,
renamed independent-crop grouping, source/cache identities and setup timings.
Operational and accuracy-sensitive settings remain the locked contract.

Recompute the comparison from the repository root:

```sh
python3 tmp/19_table_ocr_serving/step3_b8qps6_0976fa33_20260911/compare.py \
  tmp/19_table_ocr_serving/step2_b8qps6_dc755584_20260910_cached \
  tmp/19_table_ocr_serving/step3_b8qps6_0976fa33_20260911_cached
```
