# 310P UniRec decode mask/occupancy sweep

## Goal

Determine whether 310P decoder speed depends on active rows and attention-mask
contents. Run the existing A and B graphs across:

- active rows: 16, 32, 64, 96, 128;
- realistic source-length masks and fully valid cross-KV masks;
- A cache position 32;
- B cache positions 32 and 1023.

This is a value-only sweep. Tensor shapes and compiled graphs never change.
There must be zero compilation.

310P safety rule: no IncreFA batch row may be fully masked. The sweep mirrors
production static-batch padding by giving every inactive logical row the final
active row's valid source length. Inactive rows still do not count toward
effective tokens. The runner checks this before every forward and fails before
entering IncreFA if the rule is violated.

The complete 910B controls are committed under:

`12_unirec_0_1b_inference/references/unirec_decode_mask_profile_910b_20260819/`

That directory includes both full kernel-profile JSONs and all occupancy-sweep
points. On 910B, occupancy and masks made essentially no difference:

- A: about 57.5--58.8k raw token slots/s;
- B: about 20.7--21.2k raw token slots/s;
- A full/realistic B128 step ratio: 0.997;
- B full/realistic B128 step ratio: 1.005.

Its one-step kernel profiles recorded 111 rows for both graphs. A/B total kernel
time was 2.021/5.930 ms, of which the 12 IncreFA calls were 1.153/4.889 ms.

## Run

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ARTIFACT_DIR=/same/canonical/first128/artifact/used/by/the/passed/B/replay
export DECODE_CACHE_PARENT=/same/cache/parent/used/by/the/passed/dual-decode/lab
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; select a free device 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -s "$ARTIFACT_DIR/crops.jsonl"
test -d "$DECODE_CACHE_PARENT"
bash 12_unirec_0_1b_inference/run_310p_decode_mask_occupancy_sweep_background.sh
```

Immediately give Luka the absolute `RUN_LOG` printed by the launcher. Monitor
every 10--15 seconds. The runner prints ten-second heartbeats with the latest
setup/point event and active compiler-process count. A `point_begin` without a
completed point for more than 30 seconds is abnormal; report it immediately.

Expected wall time is approximately two to five minutes. A compiler process,
compile/recompile message, OM change, traceback, or NPU timeout invalidates the
run. Report it without deleting or repairing caches.

## Report

Require:

```text
UNIREC_310P_DECODE_MASK_OCCUPANCY_SWEEP: PASS
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_310P_DECODE_MASK_OCCUPANCY_RESULT` line;
3. `om.diff` status and compiler/recompile grep result;
4. the A and B point lines from their logs;
5. from the already completed 310P A/B profile, compare `a/profile.json` and
   `b/profile.json` directly with the committed 910B `a_profile.json` and
   `b_profile.json`; report top kernel-type counts/times and A/B deltas.

Stop after this report. Do not launch another profile or production pipeline.
