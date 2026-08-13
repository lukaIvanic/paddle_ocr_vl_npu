#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/layout_detector_lab.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export the UniRec Python interpreter}"
  : "${LAYOUT_MODEL:?export PP-DocLayoutV2_safetensors}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench images directory}"
  : "${LAYOUT_CACHE:?export the warmed optimized-layout cache parent}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'LAYOUT_PRODUCTION_LAB_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"

  test -x "$PYTHON_BIN"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -f "$RUNNER"
}

run_lab() {
  local command=(
    "$PYTHON_BIN" "$RUNNER"
    --contract current_production
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output "$RUN_ROOT/result.json"
    --device npu:0
    --compile-cache-dir "$LAYOUT_CACHE"
    --offset 0
    --limit 128
    --warmup-pages 1
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  "${command[@]}"
}

report_result() {
  RESULT="$RUN_ROOT/result.json" "$PYTHON_BIN" - <<'PY' \
    | tee "$RUN_ROOT/report.log"
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["RESULT"]).read_text())
config = report["config"]
summary = report["summary"]
stages = summary["stages"]
assert config["contract"] == "current_production"
assert config["production_contract_verified"] is True
assert config["scheduling"] == "sequential_b1_same_process"
assert summary["page_count"] == 128

def total(name):
    return float(stages[name]["total_s"])

def mean_ms(name):
    return float(stages[name]["mean_ms"])

def p90_ms(name):
    return float(stages[name]["p90_ms"])

print(
    "UNIREC_LAYOUT_PRODUCTION_LAB: PASS "
    f"wall={summary['measured_page_wall_s']:.6f}s "
    f"pages_s={summary['pages_per_s']:.6f} "
    f"detector={total('detector_total_s'):.6f}s "
    f"model_forward={total('model_forward_s'):.6f}s "
    f"processor={total('processor_preprocess_s'):.6f}s "
    f"box_decode={total('hf_box_decode_s'):.6f}s "
    f"h2d={total('inputs_h2d_s'):.6f}s "
    f"outputs_d2h={total('outputs_d2h_s'):.6f}s "
    f"rgb_to_bgr={total('page_rgb_to_bgr_s'):.6f}s "
    f"image_decode={total('page_image_decode_s'):.6f}s"
)
print(
    "UNIREC_LAYOUT_PRODUCTION_FORWARD: "
    f"mean={mean_ms('model_forward_s'):.6f}ms "
    f"p90={p90_ms('model_forward_s'):.6f}ms "
    f"detector_share={stages['model_forward_s']['detector_share_pct']:.6f}%"
)
print(
    "UNIREC_LAYOUT_PRODUCTION_CONTRACT: "
    f"{json.dumps(config['resolved_model_contract'], sort_keys=True)}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\npython=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'layout_model=%s\nlayout_cache=%s\n' \
      "$LAYOUT_MODEL" "$LAYOUT_CACHE"
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print("torch=" + torch.__version__); print("torch_npu=" + torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_lab
  report_result
  npu-smi info >"$RUN_ROOT/npu_after.log" 2>&1 || true
}

worker_entry() {
  local run_root="$1"
  local status=0
  local started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_LAYOUT_PRODUCTION_LAB_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/layout_production_lab_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    LAYOUT_CACHE="$LAYOUT_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_LAYOUT_PRODUCTION_LAB_STARTED pid=%s physical=%s\n' \
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
