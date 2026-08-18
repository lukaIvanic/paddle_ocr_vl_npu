# 310P K20 representative-128 W4 gate

## Goal

Measure the practical K20 benefit on the distribution-matched 128-page set.
Run only after the K20 cache builder passes. This is one hot W4/T8 lane, not an
A/B rerun: it compares against the already completed W4 K10 summary.

The 910B2 control at commit `6deceef` showed that K20 works mechanically:

- physical pixels: 276,267,008 to 232,640,512, down 15.79%;
- pixel efficiency: 66.94% to 79.49%;
- graph calls: 996 to 963;
- zero fallback, zero new first calls, and unchanged hot OM inventory;
- enclosing prefill wall: 12.831 to 12.874 seconds, effectively neutral on
  910B2.

K20 was fitted to the empirical 310P latency curve. The 310P hot result is the
decision point; do not infer it from the neutral 910B2 control.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one free physical 310P device, 0-3. This server has no `npu-setup`.
- Use the real venv `python_nosym`; do not resolve it through `readlink -f`.
- Reuse the completed K20 recognition cache and warmed optimized layout cache.
- `K10_REFERENCE_SUMMARY` must be the prior representative-128 W4/T8
  `310p_k10_l4_all` `run_summary.json`, not W1, W2, first-128, or a trace lane.
- Do not delete a cache or automatically rerun after any failure.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export COMPILE_CACHE=/absolute/path/to/the/completed/K20/cache/parent
export LAYOUT_CACHE_ROOT=/absolute/path/to/the/warmed/optimized/layout/cache
export K10_REFERENCE_SUMMARY=/absolute/path/to/prior/W4/K10/output/run_summary.json
export ASCEND_RT_VISIBLE_DEVICES=0
export TASKSET_CPUS=0-63

bash 12_unirec_0_1b_inference/run_310p_k20_rep128_w4_background.sh
```

The device is an example, not a reservation. Select a free device from 0-3.
The launcher prints absolute `RUN_ROOT`, `RUN_LOG`, and `PID`; give Luka the
exact `RUN_LOG` immediately for `tail -f`.

## Monitor

Inspect every 15-30 seconds:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E 'UNIREC_310P_K20_REP128_(BEGIN|END)|UNIREC_LAYOUT_PROCESS_(HEARTBEAT|PAGE)|Traceback|ERROR|recompil' "$RUN_LOG" | tail -14
  sleep 15
done
```

Setup should only load the 20 existing K20 caches. The measured page phase
should take about 15-25 seconds if healthy. If a graph compilation message or
new OM appears, preserve the evidence; do not call the lane hot.

## Required report

Exit zero requires:

```text
UNIREC_310P_K20_HOT_OM_INVENTORY_UNCHANGED
UNIREC_310P_K20_REP128: PASS
```

Paste only:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/hot_om.diff"
```

Also give the absolute `RUN_ROOT`, `run.log`, and `comparison.json` paths and
externally observed peak HBM. State plainly whether K20 improves measured
prefill wall time and recognition-prefill service time over the exact W4 K10
reference.
