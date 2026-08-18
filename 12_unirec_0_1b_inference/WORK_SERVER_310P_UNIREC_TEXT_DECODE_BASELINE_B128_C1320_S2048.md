# 310P text-decode baseline: B128 / self-2048 / cross-1320

## Goal

After the current full-1651 K20 + compiled-FP32 run completes, validate the
decoder-only lab at its exact production graph shape:

- batch 128;
- self-KV 2048;
- cross-KV 1320;
- initial cache position 1023;
- compiled `increfa_all`, per-step mask, full six-layer decoder and LM head;
- 300 clean timing steps plus 100 production-like sampled-token D2H steps.

This is one baseline only. Do not sweep cache sizes yet. Do not run eager,
profiling, prefill, layout, or OmniDocBench evaluation.

The matching 910B2 control at commit `366d024` measured:

- clean graph: 5.77673 ms/step, 22,157.88 raw token slots/s;
- production-like sampled-token D2H: 6.11955 ms/step, 20,916.58 raw token
  slots/s;
- latest full production decode: 20,499.53 raw and 18,594.15 effective
  tokens/s, with 90.705% slot efficiency;
- full production raw throughput was 98.006% of the lab D2H ceiling.

910B2 artifact:

`/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/text_decode_lab_b128_self2048_cross1320_current_366d024_20260819/result.json`

## Constraints

- Wait for the current full-1651 job to finish and release its NPU.
- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0-3. This server has no `npu-setup`.
- Preserve the validated venv `python_nosym`; never apply `readlink -f` to it.
- Reuse the exact decode cache reported by the completed full run.
- There must be zero graph compilations and no OM changes.
- Do not delete, repair, rename, or copy caches after a failure.

## Resolve the exact production cache

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export FULL_RUN_ROOT=/absolute/path/to/completed/full1651/K20/compiled-FP32/run
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; select a free device 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
FULL_SUMMARY="$FULL_RUN_ROOT/output/run_summary.json"
test -s "$FULL_SUMMARY"

readarray -t CACHE_PATHS < <(
  "$PYTHON_BIN" - "$FULL_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

d = json.load(open(sys.argv[1]))
assert d["status"] == "ok"
assert d["decode_batch_size"] == 128
assert d["self_cache_length"] == 2048
assert d["cross_cache_length"] == 1320
shape = Path(d["decode"]["compile"]["torchair_cache_dir"])
assert shape.name == "decode_selfkv2048_cross1320_increfa_all_b128"
print(shape.parent)
print(shape)
PY
)
test "${#CACHE_PATHS[@]}" -eq 2
CACHE_PARENT="${CACHE_PATHS[0]}"
SHAPE_CACHE="${CACHE_PATHS[1]}"
test -d "$CACHE_PARENT"
test -d "$SHAPE_CACHE"
test "$(find "$SHAPE_CACHE" -type f -name compiled_module | wc -l)" -eq 1
test "$(find "$SHAPE_CACHE" -type f -name '*.om' | wc -l)" -eq 1
printf 'CACHE_PARENT=%s\nSHAPE_CACHE=%s\n' "$CACHE_PARENT" "$SHAPE_CACHE"
```

## Launch in background

```bash
SHORT="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_text_decode_lab_b128_self2048_cross1320_${SHORT}_${STAMP}"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

find "$SHAPE_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > "$RUN_ROOT/om_before.txt"

COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/12_unirec_0_1b_inference/text_decode_lab.py"
  --model "$MODEL"
  --device npu:0 --dtype float16
  --batch-size 128 --self-cache-length 2048 --cross-cache-length 1320
  --cache-position 1023
  --warmup-steps 8 --measure-steps 300 --validation-steps 8
  --profile-steps 0 --profile-compiled-steps 0
  --backends increfa_all --compiled-timing-steps 100 --graph-mode ge
  --cache-dir "$CACHE_PARENT"
  --output "$RUN_ROOT/result.json"
)
printf '%q ' env UNIREC_STATIC_CACHE_LEN=2048 "${COMMAND[@]}" \
  > "$RUN_ROOT/command.sh"
printf '\n' >> "$RUN_ROOT/command.sh"
printf 'project_commit=%s\nphysical_npu=%s\nfull_run_root=%s\n' \
  "$(git rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$FULL_RUN_ROOT" \
  > "$RUN_ROOT/preflight.txt"

nohup bash -c '
  run_root="$1"
  shift
  set +e
  "$@"
  status="$?"
  printf "%s\n" "$status" > "$run_root/exit_code.txt"
  exit "$status"
' _ "$RUN_ROOT" env PYTHONUNBUFFERED=1 UNIREC_STATIC_CACHE_LEN=2048 \
  ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
  "${COMMAND[@]}" > "$RUN_ROOT/run.log" 2>&1 &
PID="$!"
printf '%s\n' "$PID" > "$RUN_ROOT/pid.txt"
printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
  "$RUN_ROOT" "$RUN_ROOT/run.log" "$PID"
```

Immediately give Luka the absolute `RUN_LOG` path. Expected wall time is about
30-90 seconds. Check every 10 seconds. If it exceeds 90 seconds, report the
latest event and whether compilation text or a compiler process exists before
waiting longer.

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$PID" -o pid,etime,stat,%cpu,%mem --no-headers || true
  tail -n 12 "$RUN_ROOT/run.log"
  sleep 10
done
STATUS="$(cat "$RUN_ROOT/exit_code.txt")"
printf 'STATUS=%s\n' "$STATUS"
```

## Cache and result gate

```bash
find "$SHAPE_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > "$RUN_ROOT/om_after.txt"
diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
  > "$RUN_ROOT/om.diff"
! grep -Eqi 'recompil|skip cache|compile graph|start.*compil|Traceback|ERROR' \
  "$RUN_ROOT/run.log"
test "$(cat "$RUN_ROOT/exit_code.txt")" -eq 0

"$PYTHON_BIN" - "$RUN_ROOT/result.json" "$FULL_SUMMARY" <<'PY' \
  | tee "$RUN_ROOT/final_report.txt"
import json
import sys

lab = json.load(open(sys.argv[1]))
full = json.load(open(sys.argv[2]))
assert lab["kind"] == "unirec_text_decode_lab"
assert lab["shape"] == {
    "batch_size": 128,
    "self_cache_length": 2048,
    "cross_cache_length": 1320,
    "initial_cache_position": 1023,
    "lm_head_rows": None,
    "synthetic_head": False,
}
lane = lab["lanes"]["increfa_all"]
meta = lane["compile"]
assert meta["self_attention_backend"] == "increfa_all"
assert meta["mask_mode"] == "per_step"
assert meta["batch_size"] == 128
assert meta["static_self_kv_len"] == 2048
assert meta["static_cross_kv_len"] == 1320
measure = lane["measure"]
d2h = lane["compiled_timing"]["production_like_d2h"]
full_raw = full["throughput"]["decode_raw_token_slots_per_s"]
full_effective = full["throughput"]["decode_effective_tokens_per_s"]
d2h_raw = 128000.0 / d2h["wall_step_ms"]
print("UNIREC_310P_TEXT_DECODE_BASELINE: PASS")
print(
    "TEXT_DECODE_RESULT "
    f"clean_step_ms={measure['step_ms']:.6f} "
    f"clean_raw_tok_s={measure['raw_tok_s']:.3f} "
    f"d2h_step_ms={d2h['wall_step_ms']:.6f} "
    f"d2h_raw_tok_s={d2h_raw:.3f} "
    f"full_raw_tok_s={full_raw:.3f} "
    f"full_effective_tok_s={full_effective:.3f} "
    f"full_over_lab_d2h={full_raw / d2h_raw:.6f}"
)
PY
```

## Required report

Paste:

1. `RUN_ROOT`, `RUN_LOG`, physical NPU, commit, and process wall time;
2. `UNIREC_310P_TEXT_DECODE_BASELINE: PASS` and `TEXT_DECODE_RESULT`;
3. clean step ms/raw tok/s and production-like D2H step ms/raw tok/s;
4. current full-run raw/effective tok/s and full-production divided by lab-D2H;
5. first-call/cache-load seconds;
6. `om.diff`, compile/recompile grep result, and peak HBM.

Do not interpret synthetic lab tokens as OCR output accuracy. This gate only
validates the physical decoder graph throughput before the cache-shape sweep.
