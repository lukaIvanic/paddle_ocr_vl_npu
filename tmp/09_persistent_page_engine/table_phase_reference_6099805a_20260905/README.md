# Two-table reference implementation and sequential baseline recovery

Hardware: Ascend 910B2, physical NPU 6, 2026-09-05. One owned NPU worker
at a time. Direct-host process samples showed only the owning worker on NPU 6.
All benchmark workers were stopped and NPU 6 was checked free after the runs.

## Baseline successfully recovered

The authoritative reproduction is `original_fixed/`, not the experimental
reference scheduler. It ran the original height-routed U8 adaptive-K HTTP API,
with the decode Python code-identity fix from `96f6291c`, at source commit
`1a6b5b8e`. The graphs were loaded from the corrected caches in a fresh process.
The complete cold request and two additional client warmups were outside timing.

| Metric | Historical manual-verifier run | Reproduction |
|---|---:|---:|
| Completed requests | 100 | 100 |
| Completed tables/s | 1.686509 | 1.735458 |
| Mean client latency | 0.592043 s | 0.575363 s |
| P50 | 0.530739 s | 0.506465 s |
| P90 | 1.114316 s | 1.060186 s |
| P95 | 1.281931 s | 1.274381 s |
| P99 | 1.522730 s | 1.440144 s |
| Maximum | 2.522712 s | 2.436486 s |

These are one-run closed-loop measurements, not proof of a sustainable
open-loop arrival rate. Small differences should not be interpreted as a
repeated speedup measurement.

The selected `tables.jsonl` is byte-identical to the historical run and both
earlier experimental controls. SHA-256:
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
All 100 native output-ID lists match the historical optimized-spec outputs.
No generated text was re-encoded for comparison.

## What was implemented, and what is NOT established

`--interleaved-tables 1|2` opts into an experimental step-level API worker.
It keeps independent target and eight-row draft KV slots per table, batches
ready matching phases/query shapes, and alternates different phases fairly.
It does not wait to fill batches. Adaptive K remains per request; unlike the
older batched-verifier lab, changing K does not copy historical KV between
arenas. Host action intervals distinguish own work, other-request work, and
the combination of live decode phases. Device-event times are overlapping
diagnostics, not additional latency. Initial queue/HTTP time can remain outside
the action attribution and must be retained as the client-latency residual.

The normal API is still the default. No completed random-100 C2 benchmark is
claimed. Two live warmup requests completed successfully at C2, but that is a
correctness/termination smoke, not a latency result. Thirteen CPU policy,
accounting, and native acceptance tests pass, as does the existing API metrics
test.

The experimental C1 controls were slower and are not accepted replacements:

- `c1/`: 1.359661 tables/s, P95 1.792871 s, 100/100 historical output parity.
- `fixed_c1/`: 1.306290 tables/s, P95 1.901813 s, 100/100 output parity. This
  used the two-slot reference process at client concurrency one, in the same
  process that had just created the new graphs. It is NOT the cache-loaded
  original API reproduction above.

The original API itself also exposed a cold-cache compiler collision:
`TextDecodeStage.forward` shared one Dynamo code identity between B8 and B1.
The diagnostic log explicitly reported `input_ids` batch mismatch, expected
8 versus actual 1, followed by repeated TorchAir cache-rejection warnings.
`96f6291c` assigns a separate code identity per decode B/KV shape, following
the already-established verifier technique. The fixed process had zero such
warnings. This changes compiler dispatch, not model operations or weights.
Do not attribute every experimental C1 regression solely to this warning:
the reference also changes host sequencing, and its cached-path calibration
remains pending.

## Reproduction commands

Inside the initialized research container, with physical NPU 6 checked free:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/serve_table_speculative_api.py \
  --allow-compile --host 127.0.0.1 --port 8765 \
  --request-timeout-s 3600 --queue-capacity 64 \
  --service-summary-output tmp/09_persistent_page_engine/table_phase_reference_6099805a_20260905/original_fixed_service.json
```

Client warmup: `table_closed_loop_api_client.py --set warm --count 2
--max-in-flight 1`, with the same API URL and a separate output directory.
Measured client:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8765/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 1 \
  --client-label original-fixed \
  --output-dir tmp/09_persistent_page_engine/table_phase_reference_6099805a_20260905/original_fixed
```

The historical authority is
`../table_spec_closed_loop_random100_c1_manual_postrope_90b3b3c0_20260904/`.
`analyze.py` compares locally available actual result files and writes
`comparison.json`; it does not simulate generation or timing.
