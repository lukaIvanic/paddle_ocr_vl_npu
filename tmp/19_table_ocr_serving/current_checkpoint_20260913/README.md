# Current-code validation before runtime redesign

Product baseline: `e268dae4` (numbered files and definition-order pass).
No further product changes are part of this run. The launcher is versioned
separately with its exact commit recorded in `plan.json`.

One Ascend 910B2, B8, 6-QPS Poisson arrivals, all 665 table crops plus 335
repeats in the exact historical globally shuffled 1,000-request schedule.
The frozen `be691de1` client, warmup and device-ownership harness are reused.
Server arguments adapt to the current `p01_serve.py` CLI only.

Current defaults: 60,416-row head, full-head first-token selection, Kornia-RS
resizing, uint8 CPU patches/NPU normalization, KV4096, unchanged pixel limits,
`metrics_level=scheduling`. Actual submission-to-response latency includes
queueing; scheduled-arrival latency is retained too. No output/result caching.

First process: new cache root, normal constructor compilation and one complete
real-request warmup, zero measured requests. Stop it, then restart from those
caches, warm with the same request and measure 1,000 requests. Every run keeps
commands, logs, readiness, complete responses, timings and ownership checks.

The old step-3 1,000-request run is a historical comparison, not an isolated
control: it predates the larger vocabulary and preprocessing/postprocessing
changes. Compare native output IDs and input shapes by occurrence, retain
all errors/cap stops and report discrepancies rather than hiding them. The
new measured run is the starting reference for subsequent invasive refactors.

Launch on the direct host from the updated main checkout:

```sh
python3 -u tmp/19_table_ocr_serving/current_checkpoint_20260913/driver.py \
  --expected-commit <full-committed-HEAD> --npu 6
```

The launcher refuses existing run/cache directories. It checks device ownership
before startup and throughout both phases, and stops only its own processes.
Output lives under the historical checkout's matching `tmp/...` directory;
this preserves frozen-client path resolution. Retrieve and retain it here.

Status: prepared; no new NPU result claimed yet.
