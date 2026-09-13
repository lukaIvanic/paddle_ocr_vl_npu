# Basic/detailed logging validation — Ascend 910B2

Validated source: `593923139db191189335179d576c7e12c1782fb6`.
Final source changes after this run are documentation/comment-only.
All **99 CPU tests pass** in the installed validation environment; see
`cpu-tests.log`. These include output formatting, asynchronous API/lifecycle,
idle heartbeat, dropped-log/file-failure handling, absence of NPU profiling,
dependency-event preservation, decode-slot reuse and output-token accounting.

## Final fixed-load comparison

One physical 910B2, NPU6. B8, selected 60,416-token vocabulary, Kornia-RS/uint8,
KV4096, identical saved 100-table order and 6-QPS Poisson arrival schedule.
Each run starts a fresh server, loads the same populated graph cache, performs
the existing one-real-request warmup outside measurement, then sends 100 tables.
The server itself still has only synthetic graph warmup at startup.

| Mode | Mean latency | P95 latency | Completed requests/s |
| --- | ---: | ---: | ---: |
| Logging-disabled control | 1.166635 s | 2.955730 s | 5.261253 |
| Basic | 1.153371 s | 2.939240 s | 5.260913 |
| Detailed | 1.163348 s | 2.952097 s | 5.263755 |

No observed slowdown in this short test. Basic/detailed mean latencies were
1.14%/0.28% below the control, respectively; this is not evidence that logging
improves performance. Small differences in one run each do not establish zero
overhead. Completed requests/s at a fixed offered load is not maximum capacity.
Latency is client request-attempt to response, including HTTP connection time.

All three final runs match all 100 native token streams, raw texts, formatted
HTML outputs, stopping reasons, image dimensions and input/image-token counts
against the accepted integrated-runtime reference. All measured requests ended
with EOS; no request errors. No retokenization was used for comparison.

The control uses `logging_control.py`, a benchmark-only launcher that suppresses
delivery to the writer. The same inference implementation, heartbeat counters,
process communication, admission policy and startup/warmup still run. There is
no additional production CLI option or serving implementation for this control.

## Log checks

Both final logged runs contain:

- 101 accepted requests, 101 finished requests and 101 successful HTTP writes
  (one warmup plus 100 measured requests), with matching unique request IDs.
- 57 setup progress events, startup/readiness and shutdown events.
- Four heartbeat snapshots: initial, periodic 15-second snapshots and final.
  Rates use the actual elapsed snapshot interval; initial rates are null.
- Exactly 42,072 retained output tokens including EOS and prefill's first token,
  matching the sum of finished request token counts.
- No logging warnings, dropped-event indications, request errors or false
  inference-failure events. JSON event streams in console and file match exactly.
- Detailed per-request timings only in detailed logs; no OCR text, image bytes
  or native token arrays in operational request logs.

Detailed heartbeat latency statistics are worker-side completions over the last
60 seconds, including warmup when it falls in that window. They are not the
client's isolated 100-request benchmark statistics. Overlapping timing fields
must not be summed into an additive request breakdown.

Each final run has 32–34 ownership checks with no foreign NPU6 process. Final
checks show no NPU process after our server exits; the card is released.

## Compilation and preliminary checks

The new source fingerprint is `653d42c51f0d`, under:
`/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/19_current_checkpoint_20260913`.
Ten vision, five text-prefill and one B8 decode entrypoint were prepared.
Graph progress reporting changes setup source but not forward computations,
compiler arguments, warmup inputs or their computation order.

`control/`, `basic/` and `detailed/` outside `final/` are preliminary runs at
`3eb8fc28`. The first populated the new cache (the inherited harness's generic
"cached startup" label must not be interpreted as a cache-hit claim). Inspection
found a false `inference_failed` log on clean shutdown. That was fixed, covered
by tests and all three modes were rerun at `59392313` under `final/`. The
preliminary outputs remain as evidence but are not the final comparison above.

## Reproduction and locations

Bare-metal host checkout:
`/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu`.
Container: `research_vllm_ascend_021_external_workspace`.
Interpreter: `/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python`.
The harness checks device ownership before startup, sources `npu-setup` in the
container, selects physical NPU6 and stops only its own process tree.

The final invocations were the following, from the bare-metal host. The current
checkout must match `--expected-commit`; a repeat must use new output paths.
Each run's exact server and client commands are saved under its `b8/` folder.
The benchmark retains the historical 3,600-second request timeout override;
the product default remains 60 seconds. Admission capacity was 64 throughout.

```sh
python3 /data1/lukaiv/workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/refactor_poisson100_20260913/driver.py \
  --expected-commit 593923139db191189335179d576c7e12c1782fb6 --npu 6 \
  --logging-validation --logging-disabled-control --metrics-level basic \
  --output-dir tmp/19_table_ocr_serving/logging_20260913/final/control_cached

python3 /data1/lukaiv/workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/refactor_poisson100_20260913/driver.py \
  --expected-commit 593923139db191189335179d576c7e12c1782fb6 --npu 6 \
  --logging-validation --metrics-level basic \
  --output-dir tmp/19_table_ocr_serving/logging_20260913/final/basic

python3 /data1/lukaiv/workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/refactor_poisson100_20260913/driver.py \
  --expected-commit 593923139db191189335179d576c7e12c1782fb6 --npu 6 \
  --logging-validation --metrics-level detailed \
  --output-dir tmp/19_table_ocr_serving/logging_20260913/final/detailed
```

Original remote outputs are under the frozen harness checkout:
`/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910/tmp/19_table_ocr_serving/logging_20260913`.
The frozen harness supplies lifecycle/ownership checks; it is not the model
implementation being benchmarked. Model source is the current checkout above.

Result archive SHA256:
`c5709c817c6487ca1339ee94593d2c72293d000ee9039c38cad8911d856bd358`.

Recreate the checked report locally:

```sh
python3 tmp/19_table_ocr_serving/logging_20260913/analyze.py \
  --root tmp/19_table_ocr_serving/logging_20260913/final \
  --reference tmp/19_table_ocr_serving/integrated_runtime_20260913/poisson100/b8/measured/results
```

See `final/comparison.json` for full metrics, event counts and heartbeat data.
This validates table serving under this load on 910B2, not 310P, arbitrary
failure modes or full-dataset text/formula quality.
