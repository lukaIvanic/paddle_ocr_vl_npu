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

## Completed result

2026-09-13, one Ascend 910B2 physical NPU6; runtime/launcher commit
`f4b672b50d170fd34f4b873c2efec4616d9bd2f8`. Four focused crop/HTTP-output
tests passed before inference. The exact client command and server command
are retained below `b8/`; the launch/deployment receipt is `command.txt`.

| Metric | Saved 60k/Kornia Poisson control | Current refactoring check |
|---|---:|---:|
| Mean latency (s) | 1.221411 | 1.187888 |
| P50 (s) | 0.932179 | 0.848305 |
| P95 (s) | 3.127210 | 2.980455 |
| P99 (s) | 6.963397 | 6.742145 |
| Maximum (s) | 7.835798 | 7.770258 |
| Scheduled-arrival P95 (s) | 3.127526 | 2.981550 |
| Completed requests/s, including drain | 5.242709 | 5.258506 |
| Drain after last arrival (s) | 2.222147 | 2.164847 |
| Maximum outstanding requests | 18 | 18 |
| Failed requests | 0 | 0 |

Mean decreased 2.7446%, P95 decreased 4.6928%. This is one short paired-workload
screening comparison, not a variance study or maximum-capacity measurement.
Request IDs and arrival timestamps match the original saved schedule exactly.

**100/100 native token streams, raw text, formatted HTML, stopping reasons,
crop sizes, input-token counts and projected-image-token counts match exactly.**
All 100 requests stop at EOS and generate 41,120 native tokens including EOS.
There is no output regression on this sample. No fresh TEDS scoring was needed
to establish identical scored inputs; this is not a new full-665-table score.

Cached startup: vision runtime 6.229143 s, text prefill runtime 3.236412 s,
decode runtime 0.210425 s, total recognizer setup 28.680924 s. These match the
cached-start pattern, not the prior 190/140-second compilation stages. The
model source files and graph namespace were unchanged. One full real warmup
request precedes measurement and is retained separately.

32 device-ownership snapshots were clean; the final snapshot has no process
on NPU6. Only our own server was stopped. Driver and measured client exit
codes are zero. All raw artifacts are retained unchanged. `analyze.py`
reproduces the paired output/performance checks in `comparison.json`.
Downloaded evidence archive SHA256:
`92232ae49a07091d7b0cc334f8c1f990b1d7f68188d85c20d83e89657884acfb`.

## Full saved-generation accuracy after math restoration

CPU-only replay applies the actual production math normalizer and HTML
converter to the saved current 1,000-request run. Its 665 unique table
generations are scored against the same ground truth/evaluator as before;
this does not run inference again or rescore only the 100-table subset.

| Metric (%) | Historical 16k/Pillow | Current before restoration | Current restored math |
|---|---:|---:|---:|
| Page-TEDS | 95.455281 | 95.365599 | 95.427620 |
| Table-average TEDS | 94.972558 | 94.930519 | 94.980620 |
| Page structure-only TEDS | 97.811782 | 97.780307 | 97.780307 |

The remaining Page-TEDS difference is -0.027661 percentage points, not exact
parity. There are 18 changed table TEDS scores (6 higher, 12 lower), all on
tables with changed raw generation. No table with identical raw generation
has a changed TEDS score after restoration. The older vocabulary and
preprocessing differ, so the remaining changes are not isolated refactor
effects. Page-TEDS weights pages equally, whereas table-average TEDS weights
tables equally; their net changes can therefore have different signs.

All 665 tables across 458 pages scored with zero errors/timeouts.
`accuracy_restored_math/` retains predictions, per-table/page scores,
comparison, input/source hashes and command. `score_restored_math.py` is the
reproduction helper (commit `eb2dfcc6`). No repetition handling was restored.
