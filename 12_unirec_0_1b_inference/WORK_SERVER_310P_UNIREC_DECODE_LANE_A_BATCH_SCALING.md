# 310P UniRec optimized lane-A batch scaling profile

## Goal

Measure and profile the current optimized lane-A decode graph at B1, B16, B64,
and B128 on 310P.

Every point uses C256, S256, cache position 32, 56 valid cross-KV tokens, six
attention heads, separate self-QKV projections, FP16, IncreFA, 49 FRACTAL_NZ
decode weights, and a 57,344-row LM head that returns only the real 56,371
logits.

This is not a QKV-fusion experiment. Do not change the graph between points.

## 910B2 reference

The exact committed wrapper passed on physical NPU 3 at commit `b4a5c7e`.

| Batch | Step ms | Raw tok/s | IncreFA us | MatMulV2 us | LM head us | ArgMax us | Peak allocated GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5537 | 1,806 | 190.72 | 214.50 | 67.32 | 8.84 | 0.457 |
| 16 | 0.8577 | 18,654 | 315.88 | 292.74 | 153.62 | 11.54 | 0.545 |
| 64 | 1.3679 | 46,786 | 758.94 | 340.30 | 84.86 | 22.28 | 1.018 |
| 128 | 1.8909 | 67,694 | 1,197.62 | 295.48 | 92.84 | 31.90 | 1.652 |

The B1, B16, B64, and B128 profiles had zero TransData calls. StridedSliceD was
2.24, 4.58, 9.74, and 14.58 us respectively on 910B2.

Run root:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/310p_decode_lane_a_scaling_b4a5c7e_20260819T171242
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
export CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/decode_lane_a_scaling_310p"

bash 12_unirec_0_1b_inference/run_310p_decode_lane_a_batch_scaling_background.sh
```

Immediately give Luka the absolute `RUN_LOG`. Tail it every 10 to 15 seconds.
Report the active batch, elapsed time, compiler count, OM count, and last event.
Do not wait silently through a first call longer than 60 seconds. If a first
call exceeds 60 seconds, inspect compiler processes, cache logs, and OM count
before continuing to wait.

The wrapper runs each batch in a separate process. This releases its HBM before
the next point. A cold run can create four logical graphs. Warm repeats must
reuse them without a `Skip cache as ... recompiled` warning.

## Report

Require:

```text
UNIREC_310P_DECODE_LANE_A_SCALING: PASS
```

Paste back:

1. commit, physical NPU, run root, run log, and process wall time;
2. the complete final result line;
3. for every batch, clean step time, raw tok/s, queued device time,
   production-like wall time, and peak allocated/reserved HBM;
4. for every batch, total kernel time and counts/times for
   IncreFlashAttention, MatMulV2, MatMul, ArgMaxV2, StridedSliceD, and
   TransData;
5. the exact LM-head MatMul shape and duration at every batch;
6. confirmation that every point uses 49 NZ tensors, separate QKV, six heads,
   and returns 56,371 logits;
7. cache diff, compiler observations, and any cache-skip warning;
8. a direct 310P versus 910B2 table using the reference numbers above.

Stop after these four points. Do not run full generation, prefill, or
OmniDocBench.
