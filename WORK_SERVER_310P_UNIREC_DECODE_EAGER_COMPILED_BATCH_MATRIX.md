# 310P UniRec eager versus compiled decode matrix

## Goal

Run the same B8, B32, and B128 decode comparison that passed on 910B2. Each
batch compares true raw eager execution with the TorchAir graph.

The fixed contract is:

- cross-KV 256 and self-KV 256;
- cache position 32 and 56 valid cross-KV tokens;
- FP16 IncreFlashAttention with six heads;
- 49 FRACTAL_NZ decoder weights;
- 57,344 LM-head rows, sliced to the 56,371-token semantic vocabulary;
- separate Q, K, and V projections;
- all six decoder layers, argmax, and state advance.

Prefill and scheduling stay outside the test. Timing uses 100 queued decode
steps followed by one device synchronization. It does not include a per-step
token D2H wait.

## 910B2 reference

The exact JSON files are under
`12_unirec_0_1b_inference/references/unirec_910b_decode_eager_compiled_b8_b32_b128_2241889/`.

| Batch | Raw eager | Compiled | Speedup | Compiled step |
|---:|---:|---:|---:|---:|
| B8 | 1,210 tok/s | 11,525 tok/s | 9.52x | 0.694 ms |
| B32 | 4,936 tok/s | 29,063 tok/s | 5.89x | 1.101 ms |
| B128 | 19,702 tok/s | 68,973 tok/s | 3.50x | 1.856 ms |

All three passed eight-step token parity. These ran on physical 910B2 NPU 7
at commit `2241889`.

## Rules

- Pull only. Do not edit tracked files, create branches, commit, or push.
- Use one free physical 310P from 0 through 3. This server has no `npu-setup`.
- Preserve the validated `python_nosym` path. Never apply `readlink -f` to it.
- Use the OpenDoc `unirec-0.1b/model.pth` checkpoint.
- Do not run profiling, lane B, prefill, or OmniDocBench.
- Missing B8, B32, and B128 graphs may compile once each.
- Reuse the default deterministic cache directory. Do not delete it after a
  partial run. A second run must reuse finished graphs.

Cache growth is expected. It is not a failure in this matrix. The runner saves
the before and after OM inventory and reports counts every five seconds.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ASCEND_RT_VISIBLE_DEVICES=0

bash 12_unirec_0_1b_inference/run_310p_decode_eager_vs_compiled_batch_matrix_background.sh
```

Select an actually free device. The script returns immediately. Send Luka the
printed absolute `RUN_LOG` and `TAIL_COMMAND` before doing anything else.

The default cache is:

```text
.runtime_cache/12_unirec_0_1b_inference/decode_eager_compiled_batch_310p_2241889
```

Override `CACHE_DIR` only if this exact matrix already has a known cache
elsewhere. Do not point it at an unrelated vision or full-production cache.

## Monitor

```bash
tail -f "$RUN_LOG"
```

The heartbeat includes the current batch, elapsed time for that batch, active
compiler count, OM count, compiled-module count, and the last lab event.

The normal cold sequence for each batch is:

1. model load;
2. NZ weight conversion;
3. `compiled_first_call_begin` while TorchAir loads or compiles the shape;
4. `compiled_first_call_end`;
5. eager warmup;
6. ten alternating timing rows;
7. token and logit validation.

Do not restart the task because a first call takes several minutes. Check the
heartbeat. If one batch remains on `compiled_first_call_begin` for ten minutes,
send Luka the current batch, compiler count, OM counts, and the last 100 log
lines. Leave the owned process running unless an actual device error appears.

Stop only the owned process if the log reports an AICore timeout, OOM, or other
device error. Preserve the cache and run directory.

## Required report

Wait for `exit_code.txt`. A successful run prints:

```text
UNIREC_310P_DECODE_EAGER_COMPILED_BATCH_MATRIX: PASS
UNIREC_310P_BATCH_MATRIX_WORKER_END status=0
```

Return:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/cache_counts.txt"
```

Also report:

- commit, physical NPU, torch, torch-npu, and CANN versions;
- all five eager and compiled timing rounds for every batch;
- median step time and raw tok/s for every lane;
- compiled speedup for B8, B32, and B128;
- token parity and logit max, mean, and cosine differences;
- excluded first-call time for each batch;
- compiler and cache-count progression;
- absolute run root and log.

Timing does not fail the task. A token mismatch, recompile warning after a graph
has already loaded, or device error does.
