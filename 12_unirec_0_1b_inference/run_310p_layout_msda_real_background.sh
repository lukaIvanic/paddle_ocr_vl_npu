#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAYOUT_LAB="$SCRIPT_DIR/layout_detector_lab.py"
PROFILE_RUNNER="$SCRIPT_DIR/profile_prefill_graph_suite.py"
ANALYZER="$SCRIPT_DIR/compare_layout_msda_ab.py"
EXTENSION_ROOT="$SCRIPT_DIR/custom_ops/layout_msda_aclnn/pytorch_extension"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'LAYOUT_MSDA_REAL_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the existing UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  : "${LAYOUT_CACHE_ROOT:?export the unique layout A/B cache root}"
  reject_bad_device

  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  LAYOUT_CACHE_ROOT="$(readlink -m "$LAYOUT_CACHE_ROOT")"
  MSDA_REBUILD_EXTENSION="${MSDA_REBUILD_EXTENSION:-0}"
  case "$MSDA_REBUILD_EXTENSION" in
    0)
      : "${MSDA_EXTENSION_SO:?export the completed binding probe extension SO}"
      MSDA_EXTENSION_SO="$(readlink -f "$MSDA_EXTENSION_SO")"
      ;;
    1)
      test -f "$EXTENSION_ROOT/setup.py"
      mkdir -p "$RUN_ROOT/extension" "$RUN_ROOT/extension_build"
      printf 'UNIREC_LAYOUT_MSDA_REAL_PHASE_BEGIN phase=extension_build\n'
      (
        cd "$EXTENSION_ROOT"
        MAX_JOBS=1 USE_NINJA=0 "$PYTHON_BIN" setup.py build_ext \
          --build-lib "$RUN_ROOT/extension" \
          --build-temp "$RUN_ROOT/extension_build"
      ) 2>&1 | tee "$RUN_ROOT/extension_build.log"
      mapfile -t extension_sos < <(
        find "$RUN_ROOT/extension" -type f -name '_C*.so' -print | sort
      )
      if [[ "${#extension_sos[@]}" != 1 ]]; then
        printf 'ERROR expected one rebuilt extension, found %s\n' \
          "${#extension_sos[@]}" >&2
        return 2
      fi
      MSDA_EXTENSION_SO="${extension_sos[0]}"
      printf '%s\n' "$MSDA_EXTENSION_SO" >"$RUN_ROOT/extension_so.txt"
      printf 'UNIREC_LAYOUT_MSDA_REAL_PHASE_END phase=extension_build so=%s\n' \
        "$MSDA_EXTENSION_SO"
      ;;
    *)
      printf 'ERROR MSDA_REBUILD_EXTENSION must be 0 or 1, got %s\n' \
        "$MSDA_REBUILD_EXTENSION" >&2
      return 2
      ;;
  esac
  MSDA_RUN_MODE="${MSDA_RUN_MODE:-full_ab}"
  case "$MSDA_RUN_MODE" in
    full_ab)
      MSDA_FORWARD_LIMIT="${MSDA_FORWARD_LIMIT:-128}"
      ;;
    candidate_compile_probe)
      MSDA_FORWARD_LIMIT="${MSDA_FORWARD_LIMIT:-1}"
      ;;
    *)
      printf 'ERROR unsupported MSDA_RUN_MODE=%s\n' "$MSDA_RUN_MODE" >&2
      exit 2
      ;;
  esac
  MSDA_WARMUP_PAGES="${MSDA_WARMUP_PAGES:-2}"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -f "$LAYOUT_MODEL/model.safetensors"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -f "$MSDA_EXTENSION_SO"
  test -f "$LAYOUT_LAB"
  test -f "$PROFILE_RUNNER"
  test -f "$ANALYZER"

  mapfile -d '' -t layout_images < <(
    find "$IMAGES_DIR" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
      -print0 | sort -z
  )
  if [[ "${#layout_images[@]}" -lt 128 ]]; then
    printf 'ERROR expected at least 128 layout images, found %s\n' \
      "${#layout_images[@]}" >&2
    exit 2
  fi
  LAYOUT_INPUT_IMAGE="${layout_images[0]}"
  mkdir -p "$LAYOUT_CACHE_ROOT/candidate"
  if [[ "$MSDA_RUN_MODE" == full_ab ]]; then
    mkdir -p "$LAYOUT_CACHE_ROOT/baseline"
  fi
}

run_phase() {
  local phase="$1" phase_log="$2"
  shift 2
  printf 'UNIREC_LAYOUT_MSDA_REAL_PHASE_BEGIN phase=%s command=' "$phase"
  printf '%q ' "$@"
  printf '\n'
  local started="$SECONDS" status=0
  set +e
  "$@" > >(tee "$phase_log") 2>&1
  status="$?"
  set -e
  printf 'UNIREC_LAYOUT_MSDA_REAL_PHASE_END phase=%s status=%s wall_s=%s\n' \
    "$phase" "$status" "$((SECONDS - started))"
  return "$status"
}

run_forward() {
  local name="$1" contract="$2" cache="$3"
  local output="$RUN_ROOT/forward_${name}.json"
  local command=(
    "$PYTHON_BIN" "$LAYOUT_LAB"
    --contract "$contract"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output "$output"
    --device npu:0
    --execution torchair
    --compile-cache-dir "$cache"
    --offset 0
    --limit "$MSDA_FORWARD_LIMIT"
    --warmup-pages "$MSDA_WARMUP_PAGES"
    --torch-cpu-threads 1
  )
  if [[ "$name" == candidate ]]; then
    command+=(--msda-extension-so "$MSDA_EXTENSION_SO")
  fi
  printf '%q ' "${command[@]}" >"$RUN_ROOT/forward_${name}_command.sh"
  printf '\n' >>"$RUN_ROOT/forward_${name}_command.sh"
  run_phase "forward_${name}" "$RUN_ROOT/forward_${name}.log" \
    "${command[@]}"
}

run_profile() {
  local name="$1" implementation="$2" cache="$3"
  local output="$RUN_ROOT/profile_${name}"
  local command=(
    "$PYTHON_BIN" "$PROFILE_RUNNER"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-input-image "$LAYOUT_INPUT_IMAGE"
    --layout-cache-dir "$cache"
    --recognition-cache-dir "$LAYOUT_CACHE_ROOT/unused_recognition"
    --output-dir "$output"
    --device npu:0
    --lane layout
    --layout-execution torchair
    --layout-dtype float16
    --layout-reading-order-dtype float16
    --layout-depthwise-rewrite constant_grouped
    --layout-weight-format torchair_internal
    --layout-preformat-frozen-bn-buffers
    --layout-msda-implementation "$implementation"
    --warmup 2
    --control-repeats 20
    --profile-steps 1
    --profile-metric pipe
    --parser-topn 120
    --torch-cpu-threads 1
  )
  if [[ "$name" == candidate ]]; then
    command+=(--layout-msda-extension-so "$MSDA_EXTENSION_SO")
  fi
  printf '%q ' "${command[@]}" >"$RUN_ROOT/profile_${name}_command.sh"
  printf '\n' >>"$RUN_ROOT/profile_${name}_command.sh"
  run_phase "profile_${name}" "$RUN_ROOT/profile_${name}.log" \
    "${command[@]}"
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  export UNIREC_LAYOUT_MSDA_HOST_INFER_MARKER="$RUN_ROOT/host_infer_marker.txt"
  {
    printf 'commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'model=%s\n' "$MODEL"
    printf 'layout_model=%s\n' "$LAYOUT_MODEL"
    printf 'images=%s\n' "$IMAGES_DIR"
    printf 'msda_extension_so=%s\n' "$MSDA_EXTENSION_SO"
    printf 'msda_rebuild_extension=%s\n' "$MSDA_REBUILD_EXTENSION"
    printf 'layout_cache_root=%s\n' "$LAYOUT_CACHE_ROOT"
    printf 'layout_profile_input=%s\n' "$LAYOUT_INPUT_IMAGE"
    printf 'msda_run_mode=%s\n' "$MSDA_RUN_MODE"
    printf 'msda_forward_limit=%s\n' "$MSDA_FORWARD_LIMIT"
    printf 'msda_warmup_pages=%s\n' "$MSDA_WARMUP_PAGES"
    "$PYTHON_BIN" -c \
      'import torch,torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
    npu-smi info
  } | tee "$RUN_ROOT/environment.txt"

  if [[ "$MSDA_RUN_MODE" == candidate_compile_probe ]]; then
    run_forward \
      candidate current_production_msda_aclnn "$LAYOUT_CACHE_ROOT/candidate"
  else
    run_forward baseline current_production "$LAYOUT_CACHE_ROOT/baseline"
    run_forward \
      candidate current_production_msda_aclnn "$LAYOUT_CACHE_ROOT/candidate"
    run_profile baseline decomposed "$LAYOUT_CACHE_ROOT/baseline"
    run_profile candidate aclnn "$LAYOUT_CACHE_ROOT/candidate"

    run_phase analyze "$RUN_ROOT/analyze.log" \
      "$PYTHON_BIN" "$ANALYZER" \
        --baseline-forward "$RUN_ROOT/forward_baseline.json" \
        --candidate-forward "$RUN_ROOT/forward_candidate.json" \
        --baseline-profile \
          "$RUN_ROOT/profile_baseline/profile_suite_summary.json" \
        --candidate-profile \
          "$RUN_ROOT/profile_candidate/profile_suite_summary.json" \
        --output "$RUN_ROOT/comparison_summary.json"
  fi
  npu-smi info >"$RUN_ROOT/npu_after.txt" 2>&1 || true
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  npu-smi info >"$run_root/npu_after.txt" 2>&1 || true
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_REAL_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  reject_bad_device
  : "${PYTHON_BIN:?export the existing UniRec Python interpreter}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench image directory}"
  MSDA_REBUILD_EXTENSION="${MSDA_REBUILD_EXTENSION:-0}"
  if [[ "$MSDA_REBUILD_EXTENSION" != 1 ]]; then
    : "${MSDA_EXTENSION_SO:?export the completed binding probe extension SO}"
  fi

  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_msda_real_${commit_short}_${timestamp}}"
  LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_layout_msda_real_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  LAYOUT_CACHE_ROOT="$(realpath -m "$LAYOUT_CACHE_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" \
    IMAGES_DIR="$IMAGES_DIR" \
    MSDA_EXTENSION_SO="${MSDA_EXTENSION_SO:-}" \
    MSDA_REBUILD_EXTENSION="$MSDA_REBUILD_EXTENSION" \
    LAYOUT_CACHE_ROOT="$LAYOUT_CACHE_ROOT" \
    MSDA_RUN_MODE="${MSDA_RUN_MODE:-full_ab}" \
    MSDA_FORWARD_LIMIT="${MSDA_FORWARD_LIMIT:-}" \
    MSDA_WARMUP_PAGES="${MSDA_WARMUP_PAGES:-}" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid" 2>/dev/null || test -f "$RUN_ROOT/exit_code.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_REAL_STARTED pid=%s physical=%s\n' \
    "$pid" "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$RUN_ROOT" "$RUN_ROOT/run.log"
  printf 'TAIL_COMMAND=tail -f %q\n' "$RUN_ROOT/run.log"
  printf 'EXIT_CODE_FILE=%s\n' "$RUN_ROOT/exit_code.txt"
}

case "${1:-}" in
  --worker)
    test "$#" -eq 2
    worker_entry "$2"
    ;;
  "")
    launch_main
    ;;
  *)
    printf 'usage: %s [--worker RUN_ROOT]\n' "$0" >&2
    exit 2
    ;;
esac
