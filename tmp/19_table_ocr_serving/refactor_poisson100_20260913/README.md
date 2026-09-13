# Refactoring check: B8, saved 100-table Poisson schedule at 6 QPS

Restore the pre-removal math-delimiter replacements before OTSL-to-HTML,
without restoring repetition trimming or generation-time repetition stopping.
Native token IDs and raw text remain unmodified. CPU replay on the 1,000 saved
historical raw generations reproduces their historical formatted text exactly.

Compare current code with `preprocess_poisson100_20260911/b8_both_measured`:
60,416-row decode head, full first-token head, Kornia-RS/uint8 preparation,
KV4096, fixed pixel bounds, encoded HTTP images and B8. Scheduling and decode
device timings are enabled in both (`metrics_level=detailed` now).

The client is SHA-pinned to the old client; it replays the identical saved
100-table IDs and arrival timestamps. There is no client concurrency cap.
The frozen `be691de1` server-lifecycle harness retains startup, one full real
warmup outside measurement, ownership monitoring, deadlines and cleanup.
Only server filename, CLI options, output path and current-source fingerprint
coverage are adapted. The old preprocessing selector wrapper is unnecessary:
its selected behavior is now the production implementation.

Reuse the current checkpoint's compiled graphs: model-source files p04/p05/p06
are asserted byte-identical. Runtime setup timings/logs must still be checked;
no cache checks are bypassed. Output has its own new directory; existing
evidence is not overwritten. Readiness verifies B8/KV4096, vocabulary hash
and both timing settings before the measured client starts.

Run on the direct 910B host after committing/pushing and pulling locally
authored source into the container checkout:

```sh
python3 -u tmp/19_table_ocr_serving/refactor_poisson100_20260913/driver.py \
  --expected-commit <full-committed-HEAD> --npu 6
```

Prior Poisson control: mean 1.221411 s, P95 3.127210 s,
completed throughput 5.242709 requests/s including drain.
Compare native IDs, raw text, formatted HTML, stopping reasons, input sizes
and counts as well as performance. Do not use the closed-loop control's
6.150543 requests/s as this Poisson run's reference.
