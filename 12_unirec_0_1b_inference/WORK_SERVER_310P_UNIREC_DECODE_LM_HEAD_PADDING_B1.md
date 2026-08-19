# 310P UniRec aligned LM-head B1 test

## Goal

Measure whether aligning the UniRec LM-head output from 56,371 rows to 57,344
rows speeds up lane-A B1 decode on 310P.

Both lanes use six attention heads, C256, S256, cache position 32, 56 valid
cross-KV tokens, FP16, IncreFA, and 49 FRACTAL_NZ decode weights. The candidate
pads the LM-head weight with 973 zero rows. The compiled module slices logits
back to the real 56,371-token vocabulary before argmax.

The required correctness gate is compiled baseline versus compiled aligned
head. The wrapper checks the first-step logits and a 100-token generated
sequence. The per-lane NZ eager comparison is diagnostic only. Raw eager calls
over internal-format NZ parameters are not the parity reference.

## 910B2 reference

The exact committed wrapper passed on physical NPU 3 at commit `36a920f`.

| Metric | 56,371 rows | 57,344 rows |
|---|---:|---:|
| Clean step | 0.5542 ms | 0.5344 ms |
| Clean raw throughput | 1,804.5 tok/s | 1,871.2 tok/s |
| Production-like wall | 0.9799 ms | 1.0074 ms |
| LM-head MatMul | 75.48 us | 63.68 us |
| StridedSliceD | 0 | 2.08 us |

The clean loop improved 3.69%. The production-like measurement regressed
2.73% because sampled-token D2H wait increased in that run. Compiled cross-lane
logits were bit-identical, and all 100 generated tokens matched.

Run root:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/310p_decode_lm_headpad_b1_36a920f_20260819T165640
```

## Run

Use the validated UniRec environment. Preserve `python_nosym`. Do not apply
`readlink -f` to the Python executable. The server has four devices and no
`npu-setup` command.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ASCEND_RT_VISIBLE_DEVICES=0  # select one free physical device 0-3
export CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/decode_lm_headpad_b1_310p"

bash 12_unirec_0_1b_inference/run_310p_decode_lm_head_padding_b1_background.sh
```

Immediately give Luka the absolute `RUN_LOG`. Tail it every 10 to 15 seconds.
Report the active lane, elapsed time, compiler count, OM count, and last event.
Do not wait silently through a first call longer than 60 seconds.

The first run can create two logical graphs. A warm repeat must reuse both
without a `Skip cache as ... recompiled` warning. Each measured phase uses 100
steps and remains within S256.

## Report

Require:

```text
UNIREC_310P_DECODE_LM_HEAD_PADDING_B1: PASS
```

Paste back:

1. commit, physical NPU, run root, run log, and process wall time;
2. the complete final result line;
3. each lane's clean step time, raw tok/s, queued device time, and
   production-like wall time;
4. cross-lane max error, mean error, cosine, argmax match, and 100-token match;
5. the exact LM-head MatMul shape and duration for each lane;
6. StridedSliceD, IncreFlashAttention, MatMulV2, MatMul, and TransData counts
   and durations;
7. confirmation that both lanes use 49 NZ tensors and return 56,371 logits;
8. cache diff, compiler observations, and any cache-skip warning.

Stop after B1. Do not run larger batches, prefill, or OmniDocBench.
