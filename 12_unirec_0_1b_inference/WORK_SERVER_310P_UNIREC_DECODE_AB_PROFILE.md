# 310P UniRec decoder A/B profile

## Goal

Profile one warmed compiled forward for both production decoder shapes:

- A: B128, self-KV 256, cross-KV 256, cache position 32;
- B: B128, self-KV 2048, cross-KV 1320, cache position 1023.

The production-cadence measurements are already complete: approximately 13.2k
raw token slots/s for A and 11.4k for B. This task explains why shrinking both
caches saves only about 1.53 ms per step on 310P. Do not rerun production
replay, prefill, layout, evaluation, or the full dual-lane pipeline.

The matching 910B2 profiles used the same cached GE graphs and recorded:

| | A | B |
|---|---:|---:|
| total kernel time | 2.021 ms | 5.930 ms |
| IncreFlashAttention, 12 calls | 1.153 ms | 4.889 ms |
| MatMulV2, 48 calls | 0.291 ms | 0.363 ms |
| MatMul, 1 LM-head call | 0.093 ms | 0.165 ms |
| AddLayerNorm, 19 calls | 0.198 ms | 0.209 ms |
| Scatter, 12 calls | 0.116 ms | 0.113 ms |

Both profiles had 111 kernel rows. Nearly all 910B A/B savings came from the 12
IncreFA calls. The 310P report must show whether fixed kernels dominate there or
whether short-cache IncreFA fails to become cheaper.

## Run

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export DECODE_CACHE_PARENT=/same/cache/parent/used/by/the/passed/dual-decode/lab
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; select a free device 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$DECODE_CACHE_PARENT"
bash 12_unirec_0_1b_inference/run_310p_decode_ab_profile_background.sh
```

Immediately give Luka the absolute `RUN_LOG` path printed by the launcher.
Monitor every 10--15 seconds:

```bash
tail -f "$RUN_LOG"
```

The runner prints ten-second heartbeats including the latest phase and active
compiler-process count. Both exact graphs must already exist. Expected total
wall time is about two to five minutes. Any compiler process, compilation text,
OM change, traceback, or NPU timeout invalidates the result; report it without
repairing or deleting caches.

## Completion

Require:

```text
UNIREC_310P_DECODE_AB_PROFILE: PASS
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_310P_DECODE_AB_PROFILE_RESULT` line;
3. both first-call/cache-load times and measured step/raw-token rates;
4. `om.diff` status and compiler/recompile grep result;
5. `a/profile.md` and `b/profile.md` top kernel-type tables;
6. the top 15 individual kernels by duration for A and B.

Stop after this report.
