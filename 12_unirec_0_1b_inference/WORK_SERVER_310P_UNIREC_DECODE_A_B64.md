# 310P UniRec lane-A B64 test

## Goal

Measure whether lane A improves raw decode throughput when its physical batch
is reduced from 128 to 64. Use C256/S256, cache position 32, and the same
realistic source-length distribution as the completed B128 sweep.

The B64 graph is new and may require exactly one cold compilation. Report this
separately. Never describe compile/cache-load time as decode time.

The measured state must use production-style inference tensors. The runner
hard-fails if any `Skip cache as ... recompiled` warning appears. An earlier
runner revision allocated ordinary tensors and produced an invalid cache-skip
measurement; do not use results from commit `1dc1565`.

## Run

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ARTIFACT_DIR=/same/canonical/first128/artifact/used/by/the/mask/sweep
export DECODE_CACHE_PARENT=/same/decode/cache/parent/used/by/the/mask/sweep
export BASELINE_A_RESULT=/absolute/path/to/the/completed/mask/sweep/a/result.json
export ASCEND_RT_VISIBLE_DEVICES=0  # example; select a free device 0-3

bash 12_unirec_0_1b_inference/run_decode_lane_a_b64_background.sh
```

Immediately give Luka the absolute `RUN_LOG`. Tail it every 10--15 seconds.
The heartbeat reports the current setup/point event, compiler-process count,
and elapsed time. If compilation occurs, report its start time, end time, and
whether it created only the expected B64 cache. An IncreFA stall, NPU timeout,
or fully masked row is a hard failure.

Expected wall time:

- two to six minutes if B64 must compile;
- under two minutes if the B64 cache already exists.

The steady measurement performs 10 warmups and 50 measured steps for realistic
and fully-valid masks. The raw-throughput break-even is B64 step latency below
half the measured B128 step latency, approximately 4.85 ms from the prior run.

## Report

Require:

```text
UNIREC_DECODE_A_B64: PASS
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_DECODE_A_B64_RESULT` line;
3. the two `UNIREC_DECODE_MASK_SWEEP_POINT` lines;
4. `b64_cache_preexisted`, `b64_first_call_s`, and `om.diff`;
5. whether compiler processes appeared and how long compilation lasted.

Stop after this result. Do not run a full pipeline or a second profile.
