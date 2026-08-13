#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for the passed 310P environment}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${LAYOUT_MODEL:?export LAYOUT_MODEL for PP-DocLayoutV2}"
  : "${OPENOCR_ROOT:?export OPENOCR_ROOT for the passed OpenOCR checkout}"
  : "${IMAGES_DIR:?export IMAGES_DIR for OmniDocBench images}"
  : "${LAYOUT_CACHE:?export the passed optimized-layout cache parent}"
  : "${OPT_COMPILE_CACHE:?export the passed all-focal vision cache}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_FULL_PIPELINE_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"
  OPT_COMPILE_CACHE="$(readlink -f "$OPT_COMPILE_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -d "$OPT_COMPILE_CACHE"
  test -f "$RUNNER"
}

run_lane() {
  local lane="$1"
  local limit="$2"
  local output_dir="$RUN_ROOT/$lane/output"
  mkdir -p "$output_dir"
  command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output-dir "$output_dir"
    --device npu:0
    --dtype float16
    --offset 0
    --limit "$limit"
    --workers 8
    --warmup-pages 8
    --layout-execution torchair
    --layout-dtype float16
    --layout-reading-order-dtype float32
    --layout-batch-size 1
    --layout-depthwise-rewrite group16
    --layout-weight-format torchair_internal
    --layout-preformat-frozen-bn-buffers
    --layout-cache-dir "$LAYOUT_CACHE"
    --vision-page-lookahead 4
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
    --recognition-preprocess-threads 8
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 512
    --self-cache-length 1024
    --max-length 1024
    --decode-batch-size 64
    --compile-cache-dir "$OPT_COMPILE_CACHE"
    --decode-warmup-passes 2
    --decode-admission-prefetch-depth 0
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/$lane/command.sh"
  printf '\n' >>"$RUN_ROOT/$lane/command.sh"
  printf 'UNIREC_310P_FULL_PIPELINE_PHASE_BEGIN lane=%s limit=%s\n' \
    "$lane" "$limit"
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/$lane/run.log"
  test "${PIPESTATUS[0]}" = 0
  printf 'UNIREC_310P_FULL_PIPELINE_PHASE_END lane=%s limit=%s\n' \
    "$lane" "$limit"
}

check_gate() {
  SUMMARY="$RUN_ROOT/gate_first32/output/run_summary.json" \
  "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/gate_check.log"
import json
import os
from pathlib import Path

d = json.loads(Path(os.environ["SUMMARY"]).read_text())
assert d["status"] == "ok"
assert d["page_count"] == 32
assert d["decode_batch_size"] == 64
assert d["self_cache_length"] == 1024
assert d["cross_cache_length"] == 512
assert d["decode"]["submitted"] == d["crop_count"]
assert d["decode"]["completed"] == d["crop_count"]
print(
    "UNIREC_310P_B64_GATE: PASS "
    f"pages={d['page_count']} crops={d['crop_count']} "
    f"prefill={d['timing_s']['prefill_phase']:.3f}s "
    f"decode={d['timing_s']['decode_inference_including_ingress']:.3f}s "
    f"decode_graph={d['timing_s']['decode_graph']:.3f}s "
    f"raw_tok_s={d['throughput']['decode_raw_token_slots_per_s']:.3f} "
    f"effective_tok_s={d['throughput']['decode_effective_tokens_per_s']:.3f}"
)
PY
}

check_full_memory() {
  local minimum_bytes=$((40 * 1024 * 1024 * 1024))
  local shm_available mem_available
  shm_available="$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')"
  mem_available="$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))"
  printf 'UNIREC_310P_FULL_MEMORY shm_available=%s mem_available=%s required_each=%s\n' \
    "$shm_available" "$mem_available" "$minimum_bytes" \
    | tee "$RUN_ROOT/full_memory_check.log"
  if (( shm_available < minimum_bytes || mem_available < minimum_bytes )); then
    printf 'INSUFFICIENT_CPU_SHARED_MEMORY_FOR_FULL1651\n' >&2
    return 1
  fi
}

report_full() {
  SUMMARY="$RUN_ROOT/full1651/output/run_summary.json" \
  "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/full_summary.log"
import json
import os
from pathlib import Path

d = json.loads(Path(os.environ["SUMMARY"]).read_text())
assert d["status"] == "ok"
assert d["page_count"] == 1651
assert d["decode_batch_size"] == 64
assert d["decode"]["submitted"] == d["crop_count"]
assert d["decode"]["completed"] == d["crop_count"]
t = d["timing_s"]
q = d["throughput"]
print(
    "UNIREC_310P_FULL_PIPELINE_B64: PASS "
    f"pages={d['page_count']} crops={d['crop_count']} "
    f"prefill={t['prefill_phase']:.3f}s "
    f"decode={t['decode_inference_including_ingress']:.3f}s "
    f"decode_graph={t['decode_graph']:.3f}s "
    f"sequential_core={t['sequential_core_prefill_plus_decode']:.3f}s "
    f"pg_s={q['sequential_core_pages_per_s']:.3f} "
    f"raw_tok_s={q['decode_raw_token_slots_per_s']:.3f} "
    f"effective_tok_s={q['decode_effective_tokens_per_s']:.3f} "
    f"slot_eff={d['decode']['effective_decode_tokens'] / d['decode']['raw_decode_token_slots']:.4f}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\nmodel=%s\nlayout_model=%s\n' \
      "$PYTHON_BIN" "$MODEL" "$LAYOUT_MODEL"
    printf 'layout_cache=%s\ncompile_cache=%s\n' \
      "$LAYOUT_CACHE" "$OPT_COMPILE_CACHE"
    "$PYTHON_BIN" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
    df -h /dev/shm
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  run_lane gate_first32 32
  check_gate
  check_full_memory
  run_lane full1651 1651
  report_full
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1"
  local status=0
  local started="$SECONDS"
  set +e
  (
    set -e
    worker_main "$run_root"
  )
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_FULL_PIPELINE_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp minimum_bytes shm_available
  minimum_bytes=$((40 * 1024 * 1024 * 1024))
  shm_available="$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')"
  if (( shm_available < minimum_bytes )); then
    cat >&2 <<EOF
UNIREC_310P_SHM_PREFLIGHT_FAILED available=$shm_available required=$minimum_bytes
The full retained cross-KV bank requires at least 40 GiB available in /dev/shm.
Ask Luka to run this temporary repair on the Docker host (not in the container):
  CONTAINER=<container-name>
  PID=\$(docker inspect -f '{{.State.Pid}}' "\$CONTAINER")
  sudo nsenter -t "\$PID" -m -- mount -o remount,size=64G /dev/shm
  docker exec "\$CONTAINER" df -h /dev/shm
Then rerun this launcher. See WORK_SERVER_310P_UNIREC_FULL_PIPELINE_B64.md.
EOF
    return 1
  fi
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_full_pipeline_b64_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    OPT_COMPILE_CACHE="$OPT_COMPILE_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_FULL_PIPELINE_B64_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_entry "$2"
    ;;
  "") launch_main ;;
  *) printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
