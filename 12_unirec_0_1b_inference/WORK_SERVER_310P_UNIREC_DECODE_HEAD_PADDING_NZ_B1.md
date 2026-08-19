# 310P UniRec B1 NZ plus attention-head padding test

## Goal

Compare six heads against eight physical heads after preformatting every
decode-step weight to FRACTAL_NZ. This is the combined experiment. Do not use
the earlier ND head-padding result to answer it.

Both lanes use C256, S256, cache position 32, 56 valid cross-KV tokens, FP16,
and IncreFA. The candidate pads the trained six 128-wide heads with two zero
heads. It pads the attention projections before casting the weights to NZ.

## Run

Use the validated UniRec environment. Preserve `python_nosym`. Do not apply
`readlink -f` to it. The server has four devices and no `npu-setup` command.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ASCEND_RT_VISIBLE_DEVICES=0  # select one free physical device 0-3
export WEIGHTS_NZ=1
export CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/decode_headpad_nz_b1_310p"

bash 12_unirec_0_1b_inference/run_310p_decode_head_padding_b1_background.sh
```

Immediately give Luka the absolute `RUN_LOG`. Tail it every 10 to 15 seconds.
Report the active lane, elapsed time, compiler count, OM count, and last event.
Do not wait silently through a first call longer than 60 seconds.

The first run can create two logical NZ graphs. Warm repeats must reuse them
without a `Skip cache as ... recompiled` warning. Each measured phase uses 100
steps and remains within S256.

## Report

Require:

```text
UNIREC_310P_DECODE_HEAD_PADDING_B1: PASS
```

Paste back:

1. commit, physical NPU, run root, run log, and process wall time;
2. the complete result line;
3. six-head-NZ and eight-head-NZ step time, raw tok/s, and production-like wall
   time;
4. cross-lane max error, mean error, cosine, and argmax match;
5. IncreFlashAttention, MatMulV2, MatMul, and TransData counts and time;
6. confirmation that both lanes report `weights_nz=true` and 49 NZ tensors;
7. cache diff, compiler observations, and any cache-skip warning.

Stop after B1. Do not run larger batches, prefill, or OmniDocBench.
