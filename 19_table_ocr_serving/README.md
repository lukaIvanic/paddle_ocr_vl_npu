# Experiment 19: table OCR serving

**Step 2 complete: mechanical relocation validated on one 910B2. Not yet simplified.**

This self-contained runtime is copied from experiment 09 at
`be691de190ae099d1a9b0ba80865006b122ecc00`. It does not import experiment 09.
The original is preserved unchanged. The same B8 / 6 QPS 1,000-request benchmark
has now been repeated before any cleanup.

## Structure

- `serve.py`: HTTP API, operational CLI, process lifecycle and readiness.
- `serving_runtime.py`: preparation, prefill and continuous-decode coordination.
- `paddle_ocr_vl_1_6_modeling.py`: checkpoint loading and model composition.
- `vision_prefill.py`: vision encoder/projector and its existing runtime.
- `text_prefill_and_decode.py`: both original text implementations, joined without
  changing computation; the prefill-only `_linear_tokenwise` helper is renamed
  `_prefill_linear_tokenwise` to avoid changing either implementation.
- `crop_processing.py`: original image/prompt preprocessing.
- `_support/`: temporary original dependencies, including the decode scheduler,
  configuration/preset alternatives, operator wrappers, timing and table-output
  formatting. These are retained deliberately until relocation parity passes.
- `presets/`: unchanged 16,384-row native-token vocabulary mapping.

The eventual product has one selected inference implementation, but removing
the inherited alternatives belongs to step 3. No new scheduling, warmup or
metrics design is introduced here. Synthetic constructor compilation, real
request warmup, asynchronous token transfer, KV4096 stopping behavior, CPU
lookahead, prefill admission and setup GC freeze remain intact.

## Provenance checks

`relocation.json` maps copied source files to their original Git blobs and their
new names. The CPU-only migration receipt regenerates the expected mechanical
copy in memory and checks function/class bodies:

```sh
python3 19_table_ocr_serving/tests/check_relocation.py
python3 19_table_ocr_serving/serve.py --help
```

The receipt needs repository Git history, but the serving runtime does not.
`--materialize` is one-time bulk-copy tooling, not part of serving or warmup.
Do not use it after beginning intentional simplification.

## Runtime environment and benchmark

Use the existing 910B2 environment from the reproduction lock, with
`source npu-setup` before starting the worker. No packages need to be installed
or upgraded for this relocation. See `requirements.txt` for Python dependencies;
Ascend torch-npu, CANN and TorchAir must come from the already validated runtime.

The serving entrypoint accepts the same flags as the original. Its cache
directory defaults use the experiment-19 namespace, so moving files cannot
silently masquerade as the old source fingerprint. The benchmark explicitly
passes the locked configuration and saves resolved readiness information.

The benchmark clients remain external validation tools: use the unchanged
historical clients and exact step-1 input sequence/Poisson schedule, not a newly
invented request generator. No benchmark metadata is used by the server.

Step-1 anchor: `tmp/09_persistent_page_engine/step1_b8qps6_20260910/README.md`.
It recorded 1.282711 s mean, 3.781594 s P95, 5.734013 completed requests/s,
zero errors and 1,000/1,000 native outputs identical to the original chart run.

## Step-2 validation

Runtime source commit: `dc755584`. Physical NPU6, same environment, historical
clients, 1,000-request sequence, Poisson schedule, warmup and B8 settings.

| Run | Mean (s) | P95 (s) | Completed requests/s |
|---|---:|---:|---:|
| Step 1, historical cached runtime | 1.282711 | 3.781594 | 5.734013 |
| Experiment 19, process that compiled fresh graphs | 1.360155 | 3.917355 | 5.730880 |
| Experiment 19, restarted with cached graphs | 1.283549 | 3.779594 | 5.733948 |

Both experiment-19 runs retained all 1,000 responses, zero errors, the same ten
KV-limit stops and **1,000/1,000 identical native token streams, raw text and
formatted outputs** compared with step 1. Schedules are byte-identical. Input
token counts, projected-image-token counts, crop dimensions and completion
reasons match for every occurrence. No new TEDS evaluation was run: complete
output equality establishes parity with the control.

The cached run differs from step 1 by +0.065% mean and -0.053% P95. Ownership
monitoring and direct-host checks confirmed no competing NPU6 process; our
server was stopped and NPU6 released after each completed benchmark.

**Startup qualification:** both runs used the same synthetic-constructor setup
and complete real-request warmup. The first process compiled new graphs; the
repeat loaded them from cache. The performance gap disappeared after restart
without code/settings changes. Setup froze 1,346,978 objects in the first
process versus 636,274 in the cached process (step 1: 636,312). This establishes
a startup-state difference, not proof that the object count itself caused the
gap. Keep the fresh-compile result; discuss startup behavior during step 3.

Evidence: `tmp/19_table_ocr_serving/STEP2_VALIDATION.md`, with all response,
configuration, command and ownership records in the adjacent run directories.
