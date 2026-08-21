# 310P UniRec B1 raw-eager versus compiled decode

## Goal

Measure true raw eager against the production-style TorchAir graph for one
UniRec lane-A B1 decode step. Both lanes use the same optimized model state.

Fixed contract:

- B1, cross-KV 256, self-KV 256;
- cache position 32 and 56 valid cross-KV tokens;
- FP16 IncreFlashAttention with six semantic heads;
- 49 FRACTAL_NZ decoder weights;
- 57,344 LM-head rows, sliced back to the 56,371-token vocabulary;
- separate Q, K, and V projections;
- complete six-layer decode, argmax, and state advance;
- prefill and scheduling excluded.

`raw_eager` means direct Python/PyTorch execution. It does not mean
`torch.compile(..., backend="eager")`. The compiled lane loads the existing
TorchAir GE graph. This task must compile zero new graphs.

## 910B2 reference

The same script passed on physical Ascend 910B2 NPU 7 at commit `52de30b`.

| Lane | Median step | Median raw tok/s |
|---|---:|---:|
| Raw eager | 6.461 ms | 154.77 |
| TorchAir compiled | 0.504 ms | 1,983.66 |

Compiled speedup was 12.82x. Five rounds were stable. Eight validation tokens
matched exactly. Final-logit cosine was 0.9999897, max absolute difference was
0.02344, and mean absolute difference was 0.004813. The cached first call took
0.145 seconds and was excluded. OM inventory stayed at one.

## Rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P from 0 through 3. This server has no `npu-setup`.
- Preserve the validated `python_nosym` path. Do not apply `readlink -f` to it.
- Reuse the warmed `vocab57344` B1 cache from the completed aligned-LM-head or
  full production run. Do not point at the cache parent above `vocab57344`.
- Do not delete, rename, repair, or rebuild the cache.
- The runner stops if OM or compiled-module inventory changes.
- The runner stops if the compiled first call remains open for over 60 seconds.
- Do not run profiling, larger batches, lane B, prefill, or OmniDocBench.

## Find the exact warmed cache

Start with the known aligned-head cache:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
find "$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference" \
  -type d -path '*/vocab57344/decode_selfkv256_cross256_increfa_all_wnz' \
  -print
```

Set `CACHE_DIR` to the directory immediately above
`decode_selfkv256_cross256_increfa_all_wnz`. It should end in `vocab57344`.
Before launch, require at least one `compiled_module` and one `.om` below it.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export CACHE_DIR=/absolute/path/to/warmed/vocab57344
export ASCEND_RT_VISIBLE_DEVICES=0

test "$(basename "$CACHE_DIR")" = vocab57344
test "$(find "$CACHE_DIR" -type f -name compiled_module | wc -l)" -ge 1
test "$(find "$CACHE_DIR" -type f -name '*.om' | wc -l)" -ge 1

bash 12_unirec_0_1b_inference/run_310p_decode_eager_vs_compiled_b1_background.sh
```

Choose an actually free 310P. The runner returns immediately. Send Luka the
printed absolute `RUN_LOG` and `TAIL_COMMAND` before doing anything else.

## Monitor and stop conditions

```bash
tail -f "$RUN_LOG"
```

The runner prints a heartbeat every five seconds with elapsed time, compiler
count, OM count, compiled-module count, and the last phase event.

Expected sequence:

1. model load, about a few seconds;
2. NZ weight conversion, several seconds;
3. cached compiled first call, expected well below 60 seconds;
4. raw-eager warmup;
5. ten alternating measurement rows, five per lane;
6. validation and final report.

Stop only the owned process and report immediately if:

- OM or compiled-module inventory changes;
- the first compiled call exceeds 60 seconds;
- `Skip cache as ... recompiled` appears;
- an AICore timeout or device error appears.

Do not wait through compilation. No graph needs to compile for this task.

## Required report

Wait for `exit_code.txt`. Success requires zero and:

```text
UNIREC_310P_DECODE_EAGER_VS_COMPILED_B1: PASS
UNIREC_310P_DECODE_EAGER_COMPILED_B1_WORKER_END status=0
```

Return:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/cache.diff"
```

Also provide:

- commit, physical NPU, torch and torch-npu versions;
- all five raw-eager and five compiled round times and tok/s;
- median step time and median tok/s for each lane;
- compiled speedup;
- token parity and all three logit-difference metrics;
- cached first-call duration;
- before and after OM and compiled-module counts;
- compiler-process observations;
- absolute run root and run log.

Timing never fails the task. Any token mismatch, cache mutation, recompile, or
device error does.
