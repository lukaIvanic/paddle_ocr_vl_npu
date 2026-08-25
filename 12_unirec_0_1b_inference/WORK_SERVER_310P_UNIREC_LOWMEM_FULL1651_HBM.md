# 310P full-1651 low-memory PSS and HBM run

## Goal

Run the validated low-memory UniRec pipeline over all 1,651 OmniDocBench v1.6
pages on one Atlas 310P. Measure external process-tree PSS/RSS and physical-NPU
HBM. Compare every generated token and text with the accepted 310P K20 plus
compiled-FP32 full run.

Run one candidate only. Do not run OmniDocBench evaluation. Exact trace parity
inherits the accepted run's accuracy and avoids another CDM evaluation.

The matched 910B2 result at commit `4fc7311` was:

- 4.410 GB peak host process-tree PSS;
- 3,415 MiB idle HBM, 16,205 MiB absolute peak, and 12,790 MiB above idle;
- 414.968 seconds internal pipeline wall and 3.9786 pages/s;
- exact parity with the prior validated 32,110-crop trace.

## Work-server constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0 through 3. This server has four
  devices and no `npu-setup` command.
- Use the existing CANN shell environment from the passed UniRec runs. Confirm
  `npu-smi info` works before launch.
- Preserve the venv's real `python_nosym` path. Never apply `readlink -f` to
  `PYTHON_BIN`.
- Do not use `nproc` to infer available CPUs. It reports 1 in this environment.
  The runner uses `taskset -c 0-63` and records the effective affinity.
- `/dev/shm` exposes about 64 GiB. Record it but do not reject the run because
  of that limit.
- Reuse the current K20, compiled-FP32 B2 layout, and B128 C1320 S2048 decode
  caches. The decode directory may contain more than one complete TorchAir
  identity. The runner requires matching nonzero module and OM counts. New OMs
  or recompilation fail the experiment.
- Do not delete, rename, repair, or rebuild a cache after failure.
- The inference run should take roughly 10 to 15 minutes. The runner prints a
  heartbeat every 15 seconds. If the marker does not change for 30 seconds,
  inspect the process, compiler count, HBM, and last log line before waiting.

## Pull and recover the exact accepted run

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse HEAD
test -f 12_unirec_0_1b_inference/run_310p_lowmem_full1651_hbm_background.sh

# Reuse the validated inference environment. These are the same values used by
# the accepted K20 plus compiled-FP32 full run.
export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
test "$(basename "$PYTHON_BIN")" = python_nosym
test -x "$PYTHON_BIN"

# Select the newest passed single-lane K20 plus compiled-FP32 full run. Do not
# select a representative-128 run or the later dual-decode run.
BASE_PREFLIGHT="$(
  find "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
    -type f -name preflight.log -print0 2>/dev/null \
    | xargs -0 grep -l '^run_variant=optimized_k20_l4_compiled_fp32$' \
    | xargs -r ls -1t | head -n 1
)"
test -s "$BASE_PREFLIGHT"
BASE_ROOT="$(dirname "$BASE_PREFLIGHT")"
test -s "$BASE_ROOT/final_report.txt"
grep -q 'UNIREC_310P_FULL1651_W4T8_EVAL: PASS' "$BASE_ROOT/final_report.txt"
BASE_COMMAND="$BASE_ROOT/command.sh"
export CANONICAL_TRACE="$BASE_ROOT/output/recognition_trace.jsonl"
test -s "$BASE_COMMAND"
test -s "$CANONICAL_TRACE"

# The low-memory runner uses NZ decoder weights and a 57,344-row LM head. The
# accepted single-lane accuracy run predates those decode optimizations. Recover
# the exact current K20, layout, and Lane-B caches from the later passed
# dual-restart full run while retaining the accepted trace above as the output
# reference.
DECODE_PREFLIGHT="$(
  find "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
    -type f -name preflight.log -print0 2>/dev/null \
    | xargs -0 grep -l '^run_variant=optimized_k20_l4_compiled_fp32_dual_restart$' \
    | xargs -r ls -1t | head -n 1
)"
test -s "$DECODE_PREFLIGHT"
DECODE_RUN_ROOT="$(dirname "$DECODE_PREFLIGHT")"
test -s "$DECODE_RUN_ROOT/final_report.txt"
grep -q 'UNIREC_310P_FULL1651_W4T8_EVAL: PASS' \
  "$DECODE_RUN_ROOT/final_report.txt"
DECODE_COMMAND="$DECODE_RUN_ROOT/command.sh"
DECODE_SUMMARY="$DECODE_RUN_ROOT/output/run_summary.json"
test -s "$DECODE_COMMAND"
test -s "$DECODE_SUMMARY"

extract_flag() {
  "$PYTHON_BIN" - "$BASE_COMMAND" "$1" <<'PY'
import shlex
import sys

words = shlex.split(open(sys.argv[1]).read())
flag = sys.argv[2]
positions = [index for index, word in enumerate(words) if word == flag]
if len(positions) != 1 or positions[0] + 1 >= len(words):
    raise SystemExit(f"expected exactly one {flag}, found {len(positions)}")
print(words[positions[0] + 1])
PY
}

BASE_COMMAND="$DECODE_COMMAND"
export OPENOCR_ROOT="$(extract_flag --openocr-root)"
export MODEL="$(extract_flag --model-path)"
export LAYOUT_MODEL="$(extract_flag --layout-model)"
export IMAGES_DIR="$(extract_flag --input)"
export COMPILE_CACHE="$(extract_flag --compile-cache-dir)"
export LAYOUT_CACHE_ROOT="$(extract_flag --layout-cache-dir)"

# Recover the base directory above decode_weight_nz_lmhead57344_semantic56371
# from the actual passed Lane-B cache. Do not guess this path.
export DECODE_CACHE_PARENT="$(
  "$PYTHON_BIN" - "$DECODE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

data = json.load(open(sys.argv[1]))
found = []

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "torchair_cache_dir" and isinstance(child, str):
                if "decode_selfkv2048_cross1320_increfa_all_b128_wnz" in child:
                    found.append(Path(child))
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(data)
parents = {
    path.parent.parent
    for path in found
    if path.parent.name == "decode_weight_nz_lmhead57344_semantic56371"
}
if len(parents) != 1:
    raise SystemExit(f"expected one decode cache parent, found {sorted(map(str, parents))}")
print(next(iter(parents)))
PY
)"

export ASCEND_RT_VISIBLE_DEVICES=0  # example only, select a free physical 0-3
export CPUSET=0-63
export NPU_HBM_INTERVAL_MS=2000

test -d "$OPENOCR_ROOT"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -d "$IMAGES_DIR"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"
test -d "$DECODE_CACHE_PARENT"
printf 'BASE_ROOT=%s\nCANONICAL_TRACE=%s\nCOMPILE_CACHE=%s\nLAYOUT_CACHE_ROOT=%s\nDECODE_CACHE_PARENT=%s\n' \
  "$BASE_ROOT" "$CANONICAL_TRACE" "$COMPILE_CACHE" \
  "$LAYOUT_CACHE_ROOT" "$DECODE_CACHE_PARENT"
printf 'DECODE_RUN_ROOT=%s\n' "$DECODE_RUN_ROOT"
```

If this recovery block fails, stop and report the failed line and the candidate
paths. Do not substitute another cache or run.

## Launch in the background

```bash
LAUNCH_OUTPUT="$(
  bash 12_unirec_0_1b_inference/run_310p_lowmem_full1651_hbm_background.sh
)"
printf '%s\n' "$LAUNCH_OUTPUT"
RUN_ROOT="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^PID=//p')"
test -n "$RUN_ROOT" && test -n "$RUN_LOG" && test -n "$PID"
export RUN_ROOT RUN_LOG PID
printf 'For Luka: tail -f %q\n' "$RUN_LOG"
```

Give Luka the absolute `RUN_LOG` path immediately.

## Monitor progress and time stragglers

The log already receives a heartbeat every 15 seconds. Inspect it every 15 to
30 seconds. Do not wait silently.

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" \
    -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E \
    'UNIREC_310P_LOWMEM_HEARTBEAT|UNIREC_LOWMEM_(LAYOUT_PROGRESS|FRONTEND_END|DECODE_PROGRESS|VISION_END|SUMMARY)|UNIREC_GRAPH_WARMUP|recompil|Traceback|ERROR' \
    "$RUN_LOG" | tail -25
  sleep 20
done
```

Expected markers are layout pages, frontend completion, graph warmup, crop
decode progress, vision completion, and the final summary. Decode warmup pass 2
should be a fast replay. Any new OM, visible recompilation, NPU timeout, OOM, or
sampler parse error is a stop condition. Preserve the run root and logs.

## Completion report

Require all three lines:

```text
UNIREC_310P_LOWMEM_OM_INVENTORY_UNCHANGED
UNIREC_310P_LOWMEM_FULL1651_HBM: PASS
exit_code.txt = 0
```

Paste these files:

```bash
cat "$RUN_ROOT/preflight.txt"
cat "$RUN_ROOT/cache_locator.log"
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/exit_code.txt"
```

Also report:

- project commit, physical NPU, CANN, Torch, Torch-NPU, CPU affinity,
  `/dev/shm`, and bare-metal `MemAvailable`;
- internal and external wall time and pages/s;
- peak process-tree PSS/RSS, sample count, peak time, and largest processes;
- HBM idle baseline, absolute peak, peak above baseline, peak time, sample
  count, and the selected-device process-memory row at the peak;
- layout, frontend, vision, text-prefill, decode, raw/effective token/s, and
  crop counts;
- trace SHA values, request-ID parity, mismatch count, and first mismatches;
- OM inventory result, compiler activity, and any cache/recompile warnings;
- absolute run root and log path.

Stop after this run. Do not launch deferred Markdown writing, OmniDocBench
evaluation, another memory lane, or a baseline.
