#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PROBE="$SCRIPT_DIR/bench_layout_b2_precision.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym}"
  : "${LAYOUT_MODEL:?export PP-DocLayoutV2_safetensors}"
  : "${IMAGES_DIR:?export OmniDocBench v1.6 images}"
  : "${FP32_LAYOUT_CACHE:?export a persistent FP32 layout cache root}"
  : "${FP16_LAYOUT_CACHE:?export the warmed optimized FP16 layout cache root}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device, 0-3}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  FP32_LAYOUT_CACHE="$(realpath -m "$FP32_LAYOUT_CACHE")"
  FP16_LAYOUT_CACHE="$(readlink -f "$FP16_LAYOUT_CACHE")"
  case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) echo 310P_DEVICE_MUST_BE_0_TO_3 >&2; exit 1;; esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  test -x "$PYTHON_BIN"
  test -d "$LAYOUT_MODEL"
  test -d "$IMAGES_DIR"
  test -d "$FP16_LAYOUT_CACHE"
  mkdir -p "$FP32_LAYOUT_CACHE"
  export PYTHON_BIN LAYOUT_MODEL IMAGES_DIR FP32_LAYOUT_CACHE FP16_LAYOUT_CACHE
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'layout_model=%s\nimages=%s\nfp32_cache=%s\nfp16_cache=%s\n' \
      "$LAYOUT_MODEL" "$IMAGES_DIR" "$FP32_LAYOUT_CACHE" "$FP16_LAYOUT_CACHE"
    "$PYTHON_BIN" -c 'import torch,torch_npu; print(torch.__version__); print(torch_npu.__version__)'
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  local command=(
    "$PYTHON_BIN" "$PROBE"
    --model-path "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output "$run_root/summary.json"
    --device npu:0
    --fp32-cache-dir "$FP32_LAYOUT_CACHE"
    --fp16-cache-dir "$FP16_LAYOUT_CACHE"
    --warmup 2
    --repeats 20
    --threshold 0.5
    --torch-cpu-threads 1
  )
  printf '%q ' "${command[@]}" >"$run_root/command.sh"
  printf '\n' >>"$run_root/command.sh"
  printf 'UNIREC_310P_LAYOUT_B2_BEGIN epoch_s=%s\n' "$(date +%s)"
  "${command[@]}" 2>&1 | tee "$run_root/probe.log"
  printf 'UNIREC_310P_LAYOUT_B2_END epoch_s=%s\n' "$(date +%s)"

  "$PYTHON_BIN" - "$run_root/summary.json" <<'PY' | tee "$run_root/final_report.txt"
import json, sys
p = json.load(open(sys.argv[1]))
assert p["schema"] == "unirec_layout_b2_precision_v1"
for name in ("eager_fp32", "compiled_fp32", "compiled_fp16_body_fp32_ro"):
    lane = p["lanes"][name]
    assert lane["batch_size"] == 2
    assert lane["forward"]["count"] == 20
    print(
        "UNIREC_310P_LAYOUT_B2_LANE "
        f"lane={name} mean_ms={lane['forward']['mean_ms']:.6f} "
        f"median_ms={lane['forward']['median_ms']:.6f} "
        f"p90_ms={lane['forward']['p90_ms']:.6f} "
        f"warmup_ms={lane['warmup_wall_ms']} "
        f"new_oms={len(lane['cache']['added'])} "
        f"boxes={lane['box_counts']}"
    )
print("UNIREC_310P_LAYOUT_B2_COMPARISON " + json.dumps(p["comparisons"], sort_keys=True))
print("UNIREC_310P_LAYOUT_B2_PRECISION: PASS")
PY
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_LAYOUT_B2_WORKER_END status=%s run_log=%s\n' "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_b2_precision_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" LAYOUT_MODEL="$LAYOUT_MODEL" IMAGES_DIR="$IMAGES_DIR" \
    FP32_LAYOUT_CACHE="$FP32_LAYOUT_CACHE" FP16_LAYOUT_CACHE="$FP16_LAYOUT_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
