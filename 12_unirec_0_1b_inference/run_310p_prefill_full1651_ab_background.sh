#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_prefill_export.py"

resolve_inputs() {
  : "${PYTHON_BIN:?export PYTHON_BIN for the passed 310P environment}"
  : "${MODEL:?export MODEL for the UniRec model directory}"
  : "${LAYOUT_MODEL:?export LAYOUT_MODEL for PP-DocLayoutV2}"
  : "${OPENOCR_ROOT:?export OPENOCR_ROOT for the passed OpenOCR checkout}"
  : "${IMAGES_DIR:?export IMAGES_DIR for OmniDocBench images}"
  : "${LAYOUT_CACHE:?export the passed optimized-layout cache parent}"
  : "${BASELINE_RECOGNITION_CACHE:?export the passed native five-graph cache}"
  : "${OPT_RECOGNITION_CACHE:?export a dedicated all-focal cache path}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup before launching}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'UNIREC_PREFILL_AB_REQUIRES_ONE_NPU=%s\n' \
      "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi

  # Keep the venv interpreter leaf. readlink -f can bypass pyvenv.cfg.
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
  BASELINE_RECOGNITION_CACHE="$(readlink -f "$BASELINE_RECOGNITION_CACHE")"
  OPT_RECOGNITION_CACHE="$(realpath -m "$OPT_RECOGNITION_CACHE")"

  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$LAYOUT_CACHE"
  test -d "$BASELINE_RECOGNITION_CACHE"
  test -f "$RUNNER"
}

common_args() {
  printf '%s\0' \
    --openocr-root "$OPENOCR_ROOT" \
    --model-path "$MODEL" \
    --layout-model "$LAYOUT_MODEL" \
    --input "$IMAGES_DIR" \
    --artifact-storage discard \
    --dtype float16 \
    --cross-cache-length 512 \
    --vision-full-batches \
    --recognition-input-contract compact_uint8_hwc \
    --recognition-preprocess-threads 8 \
    --vision-page-lookahead 4 \
    --no-retain-shared-images \
    --progress-every-pages 1 \
    --progress-heartbeat-s 15
}

run_lane() {
  local lane="$1"
  shift
  local output_dir="$RUN_ROOT/$lane/output"
  mkdir -p "$output_dir"
  mapfile -d '' common < <(common_args)
  command=("$PYTHON_BIN" "$RUNNER" "${common[@]}" --output-dir "$output_dir" "$@")
  printf '%q ' "${command[@]}" >"$RUN_ROOT/$lane/command.sh"
  printf '\n' >>"$RUN_ROOT/$lane/command.sh"
  printf 'UNIREC_310P_PREFILL_PHASE_BEGIN lane=%s\n' "$lane"
  "${command[@]}" 2>&1 | tee "$RUN_ROOT/$lane/run.log"
  test "${PIPESTATUS[0]}" = 0
  printf 'UNIREC_310P_PREFILL_PHASE_END lane=%s\n' "$lane"
}

report_results() {
  BASELINE="$RUN_ROOT/baseline/output/summary.json" \
  CANDIDATE="$RUN_ROOT/candidate/output/summary.json" \
  "$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/comparison.log"
import json
import os
from pathlib import Path

b = json.loads(Path(os.environ["BASELINE"]).read_text())
c = json.loads(Path(os.environ["CANDIDATE"]).read_text())
for row in (b, c):
    assert row["status"] == "ok"
    assert row["validation"]["passed"] is True
    assert (row["offset"], row["limit"], row["workers"]) == (0, 1651, 8)
    assert row["artifact_storage"] == "discard"
    assert row["artifact"]["page_count"] == 1651

contract = ("crop_count", "rejected_crop_count", "real_source_tokens")
workload_match = all(
    b["artifact"][key] == c["artifact"][key] for key in contract
)
b_wall = float(b["producer_wall_s"])
c_wall = float(c["producer_wall_s"])
print(
    "UNIREC_310P_PREFILL_FULL1651_AB: PASS "
    f"baseline={b_wall:.3f}s candidate={c_wall:.3f}s "
    f"speedup={b_wall / c_wall:.3f}x "
    f"baseline_pg_s={b['throughput']['pages_per_s']:.3f} "
    f"candidate_pg_s={c['throughput']['pages_per_s']:.3f} "
    f"workload_match={str(workload_match).lower()} "
    f"crops={b['artifact']['crop_count']}/{c['artifact']['crop_count']} "
    f"rejected={b['artifact']['rejected_crop_count']}/{c['artifact']['rejected_crop_count']} "
    f"tokens={b['artifact']['real_source_tokens']}/{c['artifact']['real_source_tokens']} "
    f"baseline_fallback={b['worker_summary']['vision_batching']['fallback_rows']} "
    f"candidate_fallback={c['worker_summary']['vision_batching']['fallback_rows']}"
)
for name, row in (("baseline", b), ("candidate", c)):
    stages = row["worker_summary"]["stage_s"]
    print(
        f"UNIREC_310P_PREFILL_FULL1651_STAGE lane={name} "
        f"layout_sum={stages['worker_detector_call_sum_s']:.3f}s "
        f"cpu_prepare_sum={stages['worker_recognition_input_prepare_sum_s']:.3f}s "
        f"prefill_sum={stages['worker_recognition_prefill_sum_s']:.3f}s "
        f"crops_per_s={row['throughput']['crops_per_s']:.3f} "
        f"tokens_per_s={row['throughput']['real_source_tokens_per_s']:.3f} "
        f"worker_max={max(row['worker_summary']['worker_busy_s']):.3f}s "
        f"setup={row['setup_s']:.3f}s warmup={row['warmup']['wall_s']:.3f}s"
    )
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  mkdir -p "$OPT_RECOGNITION_CACHE"
  {
    git -C "$REPO" rev-parse HEAD
    printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\nmodel=%s\nlayout_model=%s\n' \
      "$PYTHON_BIN" "$MODEL" "$LAYOUT_MODEL"
    printf 'layout_cache=%s\nbaseline_recognition_cache=%s\noptimized_recognition_cache=%s\n' \
      "$LAYOUT_CACHE" "$BASELINE_RECOGNITION_CACHE" "$OPT_RECOGNITION_CACHE"
    "$PYTHON_BIN" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
    npu-smi info
  } >"$RUN_ROOT/preflight.log" 2>&1

  run_lane baseline \
    --offset 0 --limit 1651 --workers 8 --warmup-pages 8 --warmup-repeats 1 \
    --layout-execution eager --layout-dtype float32 --layout-batch-size 2 \
    --layout-depthwise-rewrite native --layout-weight-format native \
    --layout-cache-dir "$LAYOUT_CACHE" \
    --recognition-cache-dir "$BASELINE_RECOGNITION_CACHE" \
    --vision-focal-depthwise-rewrite native --vision-weight-format native

  # Seed all five candidate vision graphs through one process. The subsequent
  # W8 run must load them; it must not have eight cold cache writers.
  run_lane candidate_seed \
    --offset 0 --limit 1 --workers 1 --warmup-pages 1 --warmup-repeats 1 \
    --layout-execution eager --layout-dtype float32 --layout-batch-size 1 \
    --layout-depthwise-rewrite native --layout-weight-format native \
    --layout-cache-dir "$LAYOUT_CACHE" \
    --recognition-cache-dir "$OPT_RECOGNITION_CACHE" \
    --vision-focal-depthwise-rewrite constant_grouped_all \
    --vision-weight-format torchair_internal

  run_lane candidate \
    --offset 0 --limit 1651 --workers 8 --warmup-pages 8 --warmup-repeats 1 \
    --layout-execution torchair --layout-dtype float16 --layout-batch-size 1 \
    --layout-depthwise-rewrite constant_grouped \
    --layout-weight-format torchair_internal \
    --layout-preformat-frozen-bn-buffers \
    --layout-cache-dir "$LAYOUT_CACHE" \
    --recognition-cache-dir "$OPT_RECOGNITION_CACHE" \
    --vision-focal-depthwise-rewrite constant_grouped_all \
    --vision-weight-format torchair_internal

  report_results
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
  printf 'UNIREC_310P_PREFILL_FULL1651_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local commit_short timestamp
  commit_short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_prefill_full1651_ab_${commit_short}_${timestamp}}"
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
    BASELINE_RECOGNITION_CACHE="$BASELINE_RECOGNITION_CACHE" \
    OPT_RECOGNITION_CACHE="$OPT_RECOGNITION_CACHE" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" --worker "$RUN_ROOT" \
    >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$RUN_ROOT/pid.txt"
  sleep 1
  kill -0 "$pid"
  printf 'UNIREC_310P_PREFILL_FULL1651_STARTED pid=%s physical=%s\n' \
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
