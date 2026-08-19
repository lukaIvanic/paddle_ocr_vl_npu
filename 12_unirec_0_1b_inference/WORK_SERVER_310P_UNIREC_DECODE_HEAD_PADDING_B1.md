# 310P UniRec B1 attention-head padding test

## Goal

Compare the native six trained heads against eight physical heads on lane-A B1
decode. The candidate preserves the six 128-wide trained heads and adds two
zero heads. It pads Q/K/V output width from 768 to 1,024 and pads the attention
output projection input from 768 to 1,024. It does not slice the result.

Both lanes use ND weights, C256, S256, cache position 32, 56 valid cross-KV
tokens, FP16, and IncreFA. Run them in separate Python processes.

The 910B2 control was bit-identical. Six heads measured 0.549 ms and eight heads
measured 0.551 ms. Eight heads reduced IncreFA time from 184.48 to 170.94 us,
but widened MatMulV2 time from 208.98 to 234.28 us. Measure the 310P balance
directly.

## Run

Use the validated UniRec environment. Preserve the venv `python_nosym` path.
Do not apply `readlink -f` to it. This server has four devices and no
`npu-setup` command.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ASCEND_RT_VISIBLE_DEVICES=0  # select one free physical device 0-3
export CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/decode_headpad_b1_310p"

bash 12_unirec_0_1b_inference/run_310p_decode_head_padding_b1_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG`. Tail it every 10 to 15
seconds. The heartbeat prints the active lane, elapsed time, compiler count, OM
count, and last phase. Report any first call longer than 60 seconds instead of
waiting silently.

The first run can create two logical graphs. Warm repeats must reuse them with
no `Skip cache as ... recompiled` warning. Each measured phase uses 100 steps
and stays within S256.

## Report

Require:

```text
UNIREC_310P_DECODE_HEAD_PADDING_B1: PASS
```

Paste back:

1. commit, physical NPU, run root, run log, and process wall time;
2. the complete `UNIREC_310P_DECODE_HEAD_PADDING_B1_RESULT` line;
3. six-head and eight-head steady step time, raw tok/s, and production-like
   wall time;
4. cross-lane max error, mean error, cosine, and argmax match;
5. IncreFlashAttention, MatMulV2, MatMul, and TransData counts and device time;
6. cache diff, compiler observations, and any cache-skip warning.

Stop after B1. Do not run B64, B128, prefill, or OmniDocBench.
