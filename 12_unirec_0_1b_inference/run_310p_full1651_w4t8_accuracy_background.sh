#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
PREP="$SCRIPT_DIR/prepare_unirec_subset_eval.py"
SUMMARIZER="$SCRIPT_DIR/summarize_completed_unirec_eval.py"
EVAL_ENV="$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
EVALUATOR_COMMIT=2b161d010d2e3aff77a0edef359ea3a6411d23cd

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then
    command -v "$value"
    return
  fi
  local directory basename
  directory="$(dirname "$value")"
  basename="$(basename "$value")"
  printf '%s/%s\n' "$(cd "$directory" && pwd -P)" "$basename"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P inference Python}"
  : "${MODEL:?export the UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout root}"
  : "${IMAGES_DIR:?export the OmniDocBench images directory}"
  : "${DATASET_JSON:?export the full OmniDocBench.json path}"
  : "${COMPILE_CACHE:?export the existing production compile-cache parent}"
  : "${EVALUATOR_ROOT:?export the OmniDocBench evaluator checkout}"
  : "${EVAL_PYTHON:?export the CDM-capable evaluator Python}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup and select one free 310P}"
  : "${CDM_WORKERS:=64}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'REQUIRES_EXACTLY_ONE_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  if ! [[ "$CDM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'INVALID_CDM_WORKERS=%s\n' "$CDM_WORKERS" >&2
    exit 1
  fi

  # Preserve the final venv launcher symlink. Dereferencing it with readlink -f
  # silently runs the base interpreter without the venv's site-packages.
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  DATASET_JSON="$(readlink -f "$DATASET_JSON")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  EVALUATOR_ROOT="$(readlink -f "$EVALUATOR_ROOT")"
  EVAL_PYTHON="$(absolute_executable_path "$EVAL_PYTHON")"

  export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
  # Select the repository-local frozen runtime on 310P and the workspace-local
  # frozen runtime on 910B. Never inherit ambient TeX/ImageMagick silently.
  # shellcheck source=../09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
  source "$EVAL_ENV"
  EVAL_PYTHON="$OMNIDOCBENCH_EVAL_PYTHON"
  EVALUATOR_ROOT="$OMNIDOCBENCH_EVALUATOR_ROOT"

  test -x "$PYTHON_BIN"
  test -x "$EVAL_PYTHON"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -f "$DATASET_JSON"
  test -d "$COMPILE_CACHE"
  test -f "$EVALUATOR_ROOT/pdf_validation.py"
  test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
  test -f "$RUNNER"
  test -f "$PREP"
  test -f "$SUMMARIZER"
  test -f "$CDM_RUNNER"
  "$PYTHON_BIN" -c 'import kornia_rs, torch, torch_npu'
  PYTHONPATH="$EVALUATOR_ROOT" "$EVAL_PYTHON" -c \
    'import Levenshtein; import src.metrics.cal_metric'
  PYTHONPATH="$(dirname "$CDM_RUNNER")" "$EVAL_PYTHON" -c \
    'from run_cdm_from_matched_formulas import _configure_cdm_runtime; print(_configure_cdm_runtime())'
}

prepare_clean_evaluator() {
  local source_root="$EVALUATOR_ROOT"
  local canonical_root="$RUN_ROOT/evaluator_clean"
  git clone --quiet --no-checkout "$source_root" "$canonical_root"
  git -C "$canonical_root" checkout --quiet --detach "$EVALUATOR_COMMIT"
  test -z "$(git -C "$canonical_root" status --porcelain=v1 --untracked-files=all)"
  EVALUATOR_ROOT="$canonical_root"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$canonical_root"
}

run_inference() {
  local output="$RUN_ROOT/output"
  mkdir -p "$output"
  command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-execution eager
    --layout-dtype float32
    --layout-reading-order-dtype float32
    --layout-weight-format native
    --layout-depthwise-rewrite native
    --layout-threshold 0.5
    --input "$IMAGES_DIR"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 1651
    --workers 4
    --warmup-pages 8
    --layout-batch-size 2
    --vision-page-lookahead 4
    --vision-focal-depthwise-rewrite native
    --vision-weight-format native
    --recognition-preprocess-threads 8
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --decode-batch-size 128
    --compile-cache-dir "$COMPILE_CACHE"
    --decode-warmup-passes 2
    --decode-admission-prefetch-depth 0
    --progress-every-pages 1
    --progress-heartbeat-s 15
  )
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  local started_ns ended_ns
  started_ns="$(date +%s%N)"
  printf 'UNIREC_310P_FULL1651_INFERENCE_BEGIN\n'
  "${command[@]}"
  ended_ns="$(date +%s%N)"
  "$EVAL_PYTHON" -c \
    'import sys; print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1e9:.6f}")' \
    "$started_ns" "$ended_ns" >"$RUN_ROOT/inference_process_wall_s.txt"
  printf 'UNIREC_310P_FULL1651_INFERENCE_END\n'
}

run_evaluation() {
  local evaluation="$RUN_ROOT/evaluation_image_tags_stripped"
  "$EVAL_PYTHON" "$PREP" \
    --dataset-json "$DATASET_JSON" \
    --output "$RUN_ROOT/output" \
    --evaluation-root "$evaluation" \
    --offset 0 --limit 1651 --strip-image-tags

  cd "$evaluation/work"
  ulimit -n 65536
  local started="$SECONDS"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" \
    "$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py" \
    --config config.yaml \
    --evaluator-root "$EVALUATOR_ROOT" \
    --match-workers 12 --teds-workers 12 \
    --page-timeout-sec 120 \
    --fallback-timeout-sec 180 \
    --fallback-latex-timeout-sec 30
  printf '%s\n' "$((SECONDS - started))" >"$evaluation/eval_match_teds_wall_s.txt"

  mkdir -p "$evaluation/cdm"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" \
    "$CDM_RUNNER" \
    --input "$evaluation/work/result/predictions_quick_match_display_formula_result.json" \
    --output-dir "$evaluation/cdm" \
    --evaluator-root "$EVALUATOR_ROOT" \
    --workers "$CDM_WORKERS" \
    >"$evaluation/cdm/run.log" 2>&1

  "$EVAL_PYTHON" "$SUMMARIZER" \
    --lane full1651_w4t8_b128_cross1320_self2048_t05_image_tags_stripped \
    --metric-result "$evaluation/work/result/predictions_quick_match_metric_result.json" \
    --stage-execution "$evaluation/work/result/predictions_quick_match_stage_execution.json" \
    --cdm-summary "$evaluation/cdm/cdm_run_summary.json" \
    --output "$evaluation/full_eval_summary.json" \
    | tee "$evaluation/full_eval_sentence.txt"
}

report() {
  RUN_ROOT="$RUN_ROOT" "$EVAL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
run = json.loads((root / "output/run_summary.json").read_text())
evaluation = root / "evaluation_image_tags_stripped"
score = json.loads((evaluation / "full_eval_summary.json").read_text())
transform = json.loads((evaluation / "transform_summary.json").read_text())
stage = json.loads(
    (evaluation / "work/result/predictions_quick_match_stage_execution.json").read_text()
)
assert run["status"] == "ok"
assert (run["page_count"], run["offset"], run["workers"]) == (1651, 0, 4)
assert run["recognition_preprocess_threads"] == 8
assert run["decode_batch_size"] == 128
assert run["cross_cache_length"] == 1320
assert run["self_cache_length"] == 2048
assert transform["page_count"] == 1651 and transform["strip_image_tags"] is True
assert stage["page_match"]["page_count"] == 1651
assert stage["page_match"]["fallbacks"]["page_timeout"]["count"] == 0
assert stage["page_match"]["fallbacks"]["quick_match_timeout"]["count"] == 0
assert stage["metrics"]["table"]["TEDS"]["timeout_case_count"] == 0
assert score["cdm_debug"]["timeout_case_count"] == 0
t = run["timing_s"]
q = run["throughput"]
inference_process_wall_s = float(
    (root / "inference_process_wall_s.txt").read_text()
)
slot_eff = run["decode"]["effective_decode_tokens"] / run["decode"]["raw_decode_token_slots"]
print(
    "UNIREC_310P_FULL1651_W4T8_EVAL: PASS "
    f"pages=1651 crops={run['crop_count']} "
    f"cold_process_wall_s={inference_process_wall_s:.6f} "
    f"cold_process_pg_s={1651 / inference_process_wall_s:.6f} "
    f"lifecycle_s={t['lifecycle']:.6f} "
    f"prefill_s={t['prefill_phase']:.6f} "
    f"decode_s={t['decode_inference_including_ingress']:.6f} "
    f"sequential_core_s={t['sequential_core_prefill_plus_decode']:.6f} "
    f"warmed_pipeline_pg_s={q['sequential_core_pages_per_s']:.6f} "
    f"raw_tok_s={q['decode_raw_token_slots_per_s']:.3f} "
    f"effective_tok_s={q['decode_effective_tokens_per_s']:.3f} "
    f"slot_eff={slot_eff:.6f} "
    f"removed_img_tags={transform['removed_image_tags']} "
    f"text_edit={score['text_block_page_edit']:.6f} "
    f"page_cdm={score['display_formula_page_cdm']:.6f} "
    f"page_teds={score['table_page_teds']:.6f} "
    f"reading_edit={score['reading_order_page_edit']:.6f} "
    f"overall={100 * score['official_overall']:.4f} "
    f"run_root={root}"
)
PY
}

worker_main() {
  RUN_ROOT="$1"
  resolve_inputs
  prepare_clean_evaluator
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'cdm_workers=%s\n' "$CDM_WORKERS"
    printf 'evaluator_root=%s\nevaluator_commit=%s\n' \
      "$EVALUATOR_ROOT" "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)"
    printf 'cann_home=%s\n' "${ASCEND_HOME_PATH:-unavailable}"
    if [[ -f "${ASCEND_HOME_PATH:-}/opp/version.info" ]]; then
      grep -E '^(Version|version_dir|timestamp)=' \
        "$ASCEND_HOME_PATH/opp/version.info"
    fi
    "$PYTHON_BIN" -c \
      'import torch, torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")'
    df -h /dev/shm
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  } >"$RUN_ROOT/preflight.log" 2>&1
  run_inference
  run_evaluation
  report | tee "$RUN_ROOT/final_report.txt"
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (
    set -e
    worker_main "$run_root"
  )
  status="$?"
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local shm_available mem_available minimum_shm minimum_mem
  minimum_shm=$((64 * 1024 * 1024 * 1024))
  minimum_mem=$((96 * 1024 * 1024 * 1024))
  shm_available="$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')"
  mem_available="$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))"
  if (( shm_available < minimum_shm || mem_available < minimum_mem )); then
    printf 'INSUFFICIENT_HOST_MEMORY shm=%s/%s ram=%s/%s\n' \
      "$shm_available" "$minimum_shm" "$mem_available" "$minimum_mem" >&2
    exit 1
  fi
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_full1651_w4t8_eval_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" IMAGES_DIR="$IMAGES_DIR" \
    DATASET_JSON="$DATASET_JSON" COMPILE_CACHE="$COMPILE_CACHE" \
    EVALUATOR_ROOT="$EVALUATOR_ROOT" EVAL_PYTHON="$EVAL_PYTHON" \
    OMNIDOCBENCH_EVAL_TOOLS_ROOT="$OMNIDOCBENCH_EVAL_TOOLS_ROOT" \
    OMNIDOCBENCH_TOOL_ROOT="$OMNIDOCBENCH_TOOL_ROOT" \
    CDM_PDFLATEX="$CDM_PDFLATEX" CDM_KPSEWHICH="$CDM_KPSEWHICH" \
    CDM_WORKERS="$CDM_WORKERS" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then
  worker_entry "$2"
else
  launch_main
fi
