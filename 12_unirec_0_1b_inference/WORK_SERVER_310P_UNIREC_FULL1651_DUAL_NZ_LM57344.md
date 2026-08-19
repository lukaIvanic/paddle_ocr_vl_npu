# 310P full-1651 dual decode with NZ weights and aligned LM head

## Goal

Run one optimized candidate over all 1,651 OmniDocBench v1.6 pages. Do not
run an A/B baseline.

Use the established production configuration:

- W4, recognition preprocessing T8, CPU affinity `0-63`;
- compiled FP32 layout, B2, threshold 0.5, FP32 reading order;
- four-page lookahead and `310p_k20_l4` vision buckets;
- `constant_grouped_all` vision depthwise weights and `torchair_internal`
  vision weights;
- Lane A: B128, cross-KV 256, self-KV/max length 256;
- Lane B: B128, cross-KV 1320, self-KV/max length 2048;
- 16-token quanta with a 3A:1B schedule;
- Lane A overflow restarts from token zero in Lane B;
- six attention heads;
- all decoder matmul weights preformatted to FRACTAL_NZ;
- LM head padded from 56,371 to 57,344 rows, with returned logits restricted
  to the real 56,371-token vocabulary;
- the frozen OmniDocBench evaluator, with HTML image tags removed only from
  evaluator copies.

The 910B reference artifacts are committed under
`12_unirec_0_1b_inference/references/unirec_910b_full1651_dual_nz_lm57344_20260819/`.
The 910B result was:

- prefill 121.781 s, 13.557 pages/s;
- decode graph time 80.805 s, scheduler wall 108.300 s;
- sequential core 254.613 s, 6.484 pages/s;
- effective decode 27,916.7 token/s by graph time and 20,829.3 token/s by
  scheduler wall;
- text edit 0.053837, Page CDM 0.921385, Page TEDS 0.838087, reading edit
  0.145533, Overall 90.1878;
- 32,110 crops, zero rejected crops, 50 established oversized eager vision
  fallbacks, 466 Lane A restarts;
- both decode cache gates passed from a fresh process with one OM each and no
  recompilation. Inference created no OMs.

## Constraints

- Pull only. Do not edit tracked files, create branches, commit, or push.
- Use one free physical 310P device from 0 through 3. The server has no
  `npu-setup` command.
- Preserve the validated venv's real `python_nosym` path. Never apply
  `readlink -f` to it.
- Reuse the passed K20 vision and compiled-FP32 B2 layout caches.
- Build only the two new decode graphs listed below.
- Do not delete or repair a cache after a failure.
- `/dev/shm` can expose only 64 GiB. Use `ALLOW_LOW_HOST_MEMORY=1` and preserve
  a real OOM if one occurs.
- Use the repository-local frozen evaluator runtime at
  `$WORK_SERVER_REPO/.runtime_cache/omnidocbench_eval/tools`. Ambient TeX Live
  2022 is invalid for Page CDM.
- Record the time spent in every cache attempt. If an attempt has no new log
  line for 30 seconds, report its elapsed time, compiler-process count, OM
  count, and last log line before waiting longer.

## Prepare the environment

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse HEAD

# Reuse these exact values from the last passed full-1651 K20/compiled-FP32
# run. Do not discover new model or evaluator installations.
export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export EVALUATOR_ROOT=/absolute/path/to/clean/OmniDocBench/evaluator
export EVAL_PYTHON=/absolute/path/to/frozen/evaluator/python
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; choose a free 0-3

test "$(basename "$PYTHON_BIN")" = python_nosym
test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -d "$IMAGES_DIR"
test -f "$DATASET_JSON"

# Recover the two production cache roots from the actual last passed full run.
LAST_FULL_REPORT="$(
  find "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
    -type f \( -name final_report.txt -o -name final_report_replayed.txt \) \
    -print0 2>/dev/null \
  | xargs -0 -r grep -l 'UNIREC_310P_FULL1651_W4T8_EVAL: PASS' \
  | xargs -r ls -1t | head -n 1
)"
test -s "$LAST_FULL_REPORT"
LAST_FULL_ROOT="$(dirname "$LAST_FULL_REPORT")"
LAST_FULL_COMMAND="$LAST_FULL_ROOT/command.sh"
test -s "$LAST_FULL_COMMAND"

extract_flag() {
  "$PYTHON_BIN" - "$LAST_FULL_COMMAND" "$1" <<'PY'
import shlex
import sys
words = shlex.split(open(sys.argv[1]).read())
flag = sys.argv[2]
positions = [i for i, word in enumerate(words) if word == flag]
if len(positions) != 1 or positions[0] + 1 >= len(words):
    raise SystemExit(f"expected one {flag}, found {len(positions)}")
print(words[positions[0] + 1])
PY
}

export COMPILE_CACHE="$(extract_flag --compile-cache-dir)"
export LAYOUT_CACHE_ROOT="$(extract_flag --layout-cache-dir)"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"

export OMNIDOCBENCH_EVAL_TOOLS_ROOT="$WORK_SERVER_REPO/.runtime_cache/omnidocbench_eval/tools"
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/texlive/2025/bin/aarch64-linux/pdflatex"
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/imagemagick-7.1.1-47/bin/magick"

export OPTIMIZED_DECODE_CACHE_PARENT="$WORK_SERVER_REPO/.runtime_cache/12_unirec_0_1b_inference/production_dual_decode_nz_lmhead57344"
mkdir -p "$OPTIMIZED_DECODE_CACHE_PARENT"
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$OPTIMIZED_DECODE_CACHE_PARENT"

printf 'LAST_FULL_ROOT=%s\nCOMPILE_CACHE=%s\nLAYOUT_CACHE_ROOT=%s\nDECODE_CACHE=%s\n' \
  "$LAST_FULL_ROOT" "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" \
  "$OPTIMIZED_DECODE_CACHE_PARENT"
```

## Build and fresh-process gate the two decode graphs

Use this function for each attempt. It runs the exact production cache probe in
the background so cache or compiler delays stay visible.

```bash
PROBE="$WORK_SERVER_REPO/12_unirec_0_1b_inference/probe_production_decode_cache_contract.py"
BUILD_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_dual_nz_lm57344_cache_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$BUILD_ROOT"

run_probe_attempt() {
  local label="$1" attempt="$2" self_kv="$3" cross_kv="$4"
  local log="$BUILD_ROOT/${label}_attempt_${attempt}.log"
  local result="$BUILD_ROOT/${label}_attempt_${attempt}.json"
  local started now compiler_count om_count status
  started="$(date +%s)"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PROBE" \
    --model-path "$MODEL" \
    --compile-cache-dir "$COMPILE_CACHE" \
    --device npu:0 \
    --batch-size 128 \
    --self-cache-length "$self_kv" \
    --cross-cache-length "$cross_kv" \
    --decode-weight-format nz \
    --decode-lm-head-rows 57344 \
    --passes 2 \
    --output "$result" >"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now="$(date +%s)"
    compiler_count="$(pgrep -af 'atc|ccec|compiler|tbe' | wc -l)"
    om_count="$(find "$OPTIMIZED_DECODE_CACHE_PARENT" -name '*.om' | wc -l)"
    printf 'CACHE_PROGRESS lane=%s attempt=%s elapsed_s=%s compilers=%s oms=%s last=%s\n' \
      "$label" "$attempt" "$((now-started))" "$compiler_count" "$om_count" \
      "$(tail -n 1 "$log" 2>/dev/null || true)"
  done
  set +e
  wait "$pid"
  status=$?
  set -e
  cat "$log"
  printf 'CACHE_END lane=%s attempt=%s status=%s wall_s=%s result=%s\n' \
    "$label" "$attempt" "$status" "$(( $(date +%s) - started ))" "$result"
  [[ "$status" == 0 || "$status" == 4 ]]
  return "$status"
}

build_and_gate() {
  local label="$1" self_kv="$2" cross_kv="$3" status
  set +e
  run_probe_attempt "$label" 1 "$self_kv" "$cross_kv"
  status=$?
  set -e
  if [[ "$status" == 4 ]]; then
    run_probe_attempt "$label" 2 "$self_kv" "$cross_kv"
    status=$?
  fi
  test "$status" -eq 0
}

build_and_gate a 256 256
build_and_gate b 2048 1320
```

Expected behavior for a missing graph is exit 4 on attempt 1, then exit 0 in a
fresh process on attempt 2. A pre-existing valid graph can pass on attempt 1.
Each passing JSON must show `cache_before=1/1`, `cache_after=1/1`, and
`recompiled=false`. The optimized cache must contain exactly two OMs in total.

If Lane B OOMs, stop. Report peak HBM, process state, the last marker, and the
absolute log. Do not switch to B64 or delete caches.

## Launch the full run

```bash
export CPUSET=0-63
export LAYOUT_CPU_THREADS=16
export MATCH_WORKERS=64
export TEDS_WORKERS=64
export CDM_WORKERS=64
export PROGRESS_EVERY_PAGES=16
export REQUIRE_WARM_VISION_CACHE=1
export DECODE_CACHE_GATE_ATTEMPTS=1
export ALLOW_LOW_HOST_MEMORY=1
export RUN_VARIANT=optimized_k20_l4_compiled_fp32_dual_restart
export DECODE_WEIGHT_FORMAT=nz
export DECODE_LM_HEAD_ROWS=57344

LAUNCH_OUTPUT="$(
  bash 12_unirec_0_1b_inference/run_310p_full1651_w4t8_accuracy_background.sh
)"
printf '%s\n' "$LAUNCH_OUTPUT"
RUN_ROOT="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_ROOT=//p' | tail -n 1)"
RUN_LOG="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_LOG=//p' | tail -n 1)"
PID="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^PID=//p' | tail -n 1)"
test -n "$RUN_ROOT" && test -n "$RUN_LOG" && test -n "$PID"
export RUN_ROOT RUN_LOG PID
printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_LOG"
```

Give Luka the absolute `RUN_LOG` path immediately.

## Monitor wall-time and compilation

Check every 15 to 30 seconds. Do not wait silently.

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" \
    -o pid,etime,stat,%cpu,%mem --no-headers || true
  printf 'compiler_processes=%s om_count=%s\n' \
    "$(pgrep -af 'atc|ccec|compiler|tbe' | wc -l)" \
    "$(find "$OPTIMIZED_DECODE_CACHE_PARENT" -name '*.om' | wc -l)"
  grep -E \
    'UNIREC_310P_FULL1651_PHASE|UNIREC_310P_DECODE_CACHE_GATE|UNIREC_TWO_PHASE_(PREFILL|DECODE)|HEARTBEAT|page=.*1651|Skip cache|recompil|Traceback|ERROR' \
    "$RUN_LOG" | tail -30
  sleep 20
done
```

Expected broad timing is 4 to 6 minutes for prefill, 4 to 6 minutes for decode,
and 3 to 5 minutes for evaluation. Use measured phase markers instead of these
estimates. If a phase has no progress marker for 30 seconds, report the process
state and last marker before waiting longer.

## Completion report

Require:

```text
UNIREC_310P_FULL1651_OM_INVENTORY_UNCHANGED
UNIREC_310P_FULL1651_W4T8_EVAL: PASS
```

The full evaluator result is the accuracy gate. Compare it with the committed
910B result and the prior accepted 310P Overall near 90.13. Treat an Overall
drop greater than 0.3 points as a failed adoption gate.

Paste:

```bash
cat "$RUN_ROOT/preflight.log"
cat "$RUN_ROOT/evaluator_runtime_versions.txt"
cat "$RUN_ROOT/decode_cache_gate/a_passed.json"
cat "$RUN_ROOT/decode_cache_gate/b_passed.json"
cat "$RUN_ROOT/inference_process_wall_s.txt"
cat "$RUN_ROOT/evaluation_image_tags_stripped/eval_match_teds_wall_s.txt"
cat "$RUN_ROOT/evaluation_image_tags_stripped/cdm_wall_s.txt"
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/exit_code.txt"
```

Also report:

1. commit, physical NPU, CPU affinity, CANN, Torch, Torch-NPU, `/dev/shm`, bare
   RAM, and observed peak HBM;
2. setup, prefill, decode graph, decode scheduler wall, ordered-output decode,
   sequential-core time, pages/s, raw/effective decode token/s, and slot
   efficiency;
3. per-lane iterations, time, mean step time, refills, final completions,
   promotions, idle slots, and speculative discarded tokens;
4. crop/rejection counts, K20 bucket real/physical rows, compiled slot
   efficiency, and fallback rows;
5. text edit, Page CDM, Page TEDS, reading edit, Overall, removed image tags,
   and all timeout/error counts;
6. exact deltas versus the committed 910B reference and the prior accepted
   310P run;
7. absolute run root, log, and both decode cache-gate logs.

Stop after this full run. Do not begin another decode or prefill experiment.
