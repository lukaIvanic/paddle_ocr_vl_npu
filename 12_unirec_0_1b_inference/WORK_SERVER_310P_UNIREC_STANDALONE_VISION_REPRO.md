# 310P standalone synthetic vision divergence gate

## Goal

Run one self-contained synthetic stage-3 suffix to determine whether the
catastrophic `1024x704` TorchAir divergence survives without the UniRec model,
checkpoint, image processor, pipeline, or any project import.

The only executed source file is:

```text
12_unirec_0_1b_inference/standalone_vision_torchair_divergence.py
```

It imports only PyTorch/TorchNPU, NumPy-free standard library code, and TorchAir
from the installed environment. It sets eager JIT compilation off internally.

## 910B2 control

Commit `278be84`, physical 910B2 NPU 7:

```text
start_stage=3
input=[1,768,22,32], valid=20x30
one new graph compile/first call=11.542 s
steady eager=4.420 ms
steady compiled=1.741 ms
valid max_abs=0.000732421875
valid RMSE=0.0001481314
valid cosine=0.99999988
```

## Run on 310P

Pull only. Do not edit tracked files, create a branch, or create a worktree.
Use one free physical 310P device in `0,1,2,3`; this server has no `npu-setup`.
Use the validated real `python_nosym` executable and existing CANN shell setup.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git merge-base --is-ancestor 278be84 HEAD

: "${PYTHON_BIN:?validated venv/bin/python_nosym}"
: "${ASCEND_RT_VISIBLE_DEVICES:?one free physical 310P device 0-3}"
case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) exit 2 ;; esac

RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_standalone_vision_s3_$(date +%Y%m%dT%H%M%S)"
CACHE_ROOT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/standalone_vision_divergence"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT"

(
  set +e
  started="$(date +%s)"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" \
    12_unirec_0_1b_inference/standalone_vision_torchair_divergence.py \
      --cache-root "$CACHE_ROOT" \
      --output "$RUN_ROOT/report.json" \
      --start-stage 3 \
      --weight-format torchair_internal \
      --timing-repeats 3
  status=$?
  printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
  printf '%s\n' "$(($(date +%s) - started))" >"$RUN_ROOT/process_wall_s.txt"
) >"$RUN_ROOT/run.log" 2>&1 &
printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"

echo "RUN_ROOT=$RUN_ROOT"
echo "tail -f '$RUN_ROOT/run.log'"
```

This is a deliberately cold deterministic cache. Exactly one TorchAir graph is
allowed to compile. The graph is much smaller than the full vision encoder.
Inspect the log every 15 seconds; do not wait silently:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E 'UNIREC_STANDALONE_VISION_PHASE|Traceback|ERROR' "$RUN_ROOT/run.log" | tail -10
  sleep 15
done
```

## Report and stop

Paste:

```bash
cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/report.json"
```

Interpret only the eager-versus-compiled comparison:

- severe divergence: the bug is reproduced inside the two-block stage-3 suffix;
- clean result: stop. The next run will use the same script with
  `--start-stage 2`.

Do not automatically run stage 2 or any other lane.
