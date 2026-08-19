# 310P UniRec B1 decode weight-format test

## Goal

Compare native ND decoder weights against persistent FRACTAL_NZ decoder
weights on one production-shaped lane-A decode step:

- batch 1;
- cross-KV 256;
- self-KV 256;
- cache position 32;
- 56 valid cross-attention tokens;
- FP16 and IncreFA;
- the complete six-layer decoder plus LM head.

This test excludes the encoder and prefill. Run ND and NZ in separate Python
processes. Running both module instances in one process causes a TorchDynamo
code-object cache collision and invalidates the comparison.

The 910B2 control was numerically exact. Its loop timing was effectively tied:
ND 0.531 ms and NZ 0.523 ms. The single-step profile showed no TransData in
either lane. NZ increased profiled MatMulV2 time from 191.50 to 213.34 us and
the unaligned 56,371-row LM-head MatMul from 46.00 to 74.08 us. Do not assume
that 310P behaves the same. The purpose of this run is to measure that directly.

## Run

Use the existing validated UniRec environment. Preserve the venv's real
`python_nosym` path. Do not apply `readlink -f` to it. The 310P server has four
devices and no `npu-setup` command.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ASCEND_RT_VISIBLE_DEVICES=0  # select one free physical device 0-3
export CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/decode_weight_b1_310p"

bash 12_unirec_0_1b_inference/run_310p_decode_weight_b1_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG`. Tail it every 10 to 15
seconds. The heartbeat prints the active lane, elapsed time, compiler-process
count, OM count, and last phase. If a compile lasts more than 60 seconds, say
which lane is compiling and how many OMs exist. Do not wait silently.

The first run can create at most two decode graphs: one ND graph and one NZ
graph. Later runs must reuse those caches. Do not delete the cache merely
because `cache.diff` is non-empty on the first run.

The runner uses 100 measured steps. Do not increase this without also resetting
the decode state. Lane A starts at self-KV position 32 and has capacity 256.
The lab now rejects any phase that could advance past the cache boundary.

Expected wall time:

- one to six minutes when both graphs are cold;
- under two minutes when both caches are warm.

## Report

Require:

```text
UNIREC_310P_DECODE_WEIGHT_B1: PASS
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_310P_DECODE_WEIGHT_B1_RESULT` line;
3. ND and NZ first-call time, steady `step_ms`, raw tok/s, production-like wall
   time, and D2H wait;
4. parity: max absolute error, mean absolute error, cosine, and argmax match;
5. ND versus NZ TransData, MatMulV2, MatMul, and IncreFlashAttention counts and
   device time;
6. `cache.diff`, compiler-process observations, and whether later calls loaded
   the cache without a `Skip cache as ... recompiled` warning.

Stop after this B1 result. Do not run B64, B128, prefill, or full OmniDocBench.
