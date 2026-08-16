#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_PROBE="$SCRIPT_DIR/probe_layout_msda_runtime.py"

reject_bad_device() {
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"
  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'MSDA_PREFLIGHT_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
}

resolve_python() {
  : "${PYTHON_BIN:?export the existing UniRec Python interpreter}"
  if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
  else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  fi
  test -x "$PYTHON_BIN"
  test -f "$RUNTIME_PROBE"
}

record_roots() {
  local candidate resolved
  : >"$RUN_ROOT/toolkit_roots.txt"
  for candidate in \
    "${ASCEND_HOME_PATH:-}" \
    "${ASCEND_TOOLKIT_HOME:-}" \
    "${ASCEND_OPP_PATH:-}" \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/latest; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    resolved="$(readlink -f "$candidate")"
    printf '%s\n' "$resolved" >>"$RUN_ROOT/toolkit_roots.txt"
  done
  sort -u "$RUN_ROOT/toolkit_roots.txt" -o "$RUN_ROOT/toolkit_roots.txt"
}

inspect_headers() {
  : >"$RUN_ROOT/header_hits.txt"
  local root candidate
  {
    while IFS= read -r root; do
      for candidate in \
        "$root/include/aclnnop/aclnn_multi_scale_deformable_attn_function.h" \
        "$root/aarch64-linux/include/aclnnop/aclnn_multi_scale_deformable_attn_function.h" \
        "$root/include/aclnnop/aclnn_multi_scale_deformable_attn.h" \
        "$root/aarch64-linux/include/aclnnop/aclnn_multi_scale_deformable_attn.h"; do
        [[ -f "$candidate" ]] && printf '%s\n' "$candidate" || true
      done
    done <"$RUN_ROOT/toolkit_roots.txt"
    true
  } | sort -u >"$RUN_ROOT/header_hits.txt"
}

inspect_symbols() {
  : >"$RUN_ROOT/symbol_hits.txt"
  local root library
  if ! command -v nm >/dev/null 2>&1; then
    printf 'NM_NOT_INSTALLED\n' >"$RUN_ROOT/symbol_hits.txt"
    return
  fi
  {
    while IFS= read -r root; do
      for library in \
        "$root/aarch64-linux/lib64/libopapi.so" \
        "$root/aarch64-linux/lib64/libopapi_nn.so" \
        "$root/lib64/libopapi.so" \
        "$root/lib64/libopapi_nn.so"; do
        [[ -f "$library" ]] || continue
        nm -D "$library" 2>/dev/null \
          | grep -E 'aclnnMultiScaleDeformableAttn(Function)?(GetWorkspaceSize)?$' \
          | sed "s#^#$library #" || true
      done
    done <"$RUN_ROOT/toolkit_roots.txt"
    true
  } | sort -u >"$RUN_ROOT/symbol_hits.txt"
}

inspect_operator_packages() {
  : >"$RUN_ROOT/operator_metadata_hits.txt"
  : >"$RUN_ROOT/operator_metadata_310p_hits.txt"
  local toolkit_root opp_root
  {
    while IFS= read -r toolkit_root; do
      for opp_root in \
        "$toolkit_root/opp" \
        "$toolkit_root/aarch64-linux/opp"; do
        [[ -d "$opp_root" ]] || continue
        find "$opp_root" -type f \
          \( -iname '*multi*scale*deform*attn*' \
             -o -name 'binary_info_config.json' \) \
          -print 2>/dev/null || true
      done
      case "$toolkit_root" in
        */opp|*/opp/*)
          find "$toolkit_root" -type f \
            \( -iname '*multi*scale*deform*attn*' \
               -o -name 'binary_info_config.json' \) \
            -print 2>/dev/null || true
          ;;
      esac
    done <"$RUN_ROOT/toolkit_roots.txt"
    true
  } | sort -u >"$RUN_ROOT/operator_metadata_candidates.txt"

  {
    while IFS= read -r candidate; do
      if [[ "$candidate" == *[Mm]ulti*[Ss]cale*[Dd]eform* ]]; then
        printf '%s\n' "$candidate"
      elif grep -q -E 'MultiScaleDeformableAttn(Function)?' "$candidate" 2>/dev/null; then
        printf '%s\n' "$candidate"
      fi
    done <"$RUN_ROOT/operator_metadata_candidates.txt"
    true
  } | sort -u >"$RUN_ROOT/operator_metadata_hits.txt"

  grep -Ei 'ascend310p|310p' "$RUN_ROOT/operator_metadata_hits.txt" \
    >"$RUN_ROOT/operator_metadata_310p_hits.txt" || true
}

emit_verdict() {
  local header_count symbol_count metadata_count metadata_310p_count runtime_status verdict
  header_count="$(grep -c . "$RUN_ROOT/header_hits.txt" || true)"
  symbol_count="$(grep -c 'aclnnMultiScale' "$RUN_ROOT/symbol_hits.txt" || true)"
  metadata_count="$(grep -c . "$RUN_ROOT/operator_metadata_hits.txt" || true)"
  metadata_310p_count="$(grep -c . "$RUN_ROOT/operator_metadata_310p_hits.txt" || true)"
  runtime_status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_ROOT/runtime_probe.json")"

  if [[ "$runtime_status" == pass ]]; then
    verdict=VERIFIED_RUNTIME_SUPPORTED
  elif grep -qiE 'does not support opType|not a registered function|unsupported.*310p' \
      "$RUN_ROOT/runtime_probe.json"; then
    verdict=VERIFIED_RUNTIME_UNSUPPORTED
  elif (( header_count > 0 && symbol_count > 0 && metadata_310p_count > 0 )); then
    verdict=READY_FOR_MINIMAL_BINDING_PROBE
  elif (( header_count > 0 && symbol_count > 0 )); then
    verdict=NEEDS_BINDING_RUNTIME_PROBE
  else
    verdict=BLOCKED_MISSING_NATIVE_API
  fi

  {
    printf 'verdict=%s\n' "$verdict"
    printf 'header_count=%s\n' "$header_count"
    printf 'symbol_count=%s\n' "$symbol_count"
    printf 'operator_metadata_count=%s\n' "$metadata_count"
    printf 'operator_metadata_310p_count=%s\n' "$metadata_310p_count"
    printf 'runtime_status=%s\n' "$runtime_status"
  } >"$RUN_ROOT/summary.env"
  printf 'UNIREC_LAYOUT_MSDA_PREFLIGHT verdict=%s headers=%s symbols=%s metadata=%s metadata_310p=%s runtime=%s\n' \
    "$verdict" "$header_count" "$symbol_count" "$metadata_count" \
    "$metadata_310p_count" "$runtime_status"
}

worker_main() {
  RUN_ROOT="$1"
  reject_bad_device
  resolve_python
  printf 'UNIREC_LAYOUT_MSDA_PHASE_BEGIN phase=environment\n'
  {
    printf 'commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-}"
    printf 'ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-}"
    printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-}"
    uname -a
    "$PYTHON_BIN" -c 'import torch,torch_npu; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
    npu-smi info
  } | tee "$RUN_ROOT/environment.txt"
  printf 'UNIREC_LAYOUT_MSDA_PHASE_END phase=environment\n'

  printf 'UNIREC_LAYOUT_MSDA_PHASE_BEGIN phase=static_inventory\n'
  record_roots
  inspect_headers
  inspect_symbols
  inspect_operator_packages
  printf 'UNIREC_LAYOUT_MSDA_HEADERS\n'; sed -n '1,40p' "$RUN_ROOT/header_hits.txt"
  printf 'UNIREC_LAYOUT_MSDA_SYMBOLS\n'; sed -n '1,80p' "$RUN_ROOT/symbol_hits.txt"
  printf 'UNIREC_LAYOUT_MSDA_310P_METADATA\n'; sed -n '1,80p' "$RUN_ROOT/operator_metadata_310p_hits.txt"
  printf 'UNIREC_LAYOUT_MSDA_PHASE_END phase=static_inventory\n'

  printf 'UNIREC_LAYOUT_MSDA_PHASE_BEGIN phase=runtime_wrapper\n'
  "$PYTHON_BIN" "$RUNTIME_PROBE" --output "$RUN_ROOT/runtime_probe.json"
  printf 'UNIREC_LAYOUT_MSDA_PHASE_END phase=runtime_wrapper\n'
  emit_verdict
  npu-smi info >"$RUN_ROOT/npu_after.txt" 2>&1 || true
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_PREFLIGHT_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  reject_bad_device
  resolve_python
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_layout_msda_preflight_${commit_short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}" \
    ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-}" \
    ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-}" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid" 2>/dev/null || test -f "$RUN_ROOT/exit_code.txt"
  printf 'UNIREC_310P_LAYOUT_MSDA_PREFLIGHT_STARTED pid=%s physical=%s\n' \
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
