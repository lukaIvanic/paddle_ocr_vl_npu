#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
MATERIALIZER="$SCRIPT_DIR/materialize_page_subset.py"
MANIFEST="$SCRIPT_DIR/references/unirec_representative_128_v1.json"
PREP="$SCRIPT_DIR/prepare_unirec_subset_eval.py"
MD_COMPARE="$SCRIPT_DIR/compare_unirec_markdown_subset.py"
SUMMARIZER="$SCRIPT_DIR/summarize_completed_unirec_eval.py"
EVAL_RUNNER="$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py"
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
EVAL_ENV="$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
EVALUATOR_COMMIT=2b161d010d2e3aff77a0edef359ea3a6411d23cd

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

phase() {
  printf 'UNIREC_310P_REP128_FP32_PHASE phase=%s epoch_s=%s\n' "$1" "$(date +%s)"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym executable}"
  : "${MODEL:?export the OpenDoc UniRec model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 image directory}"
  : "${DATASET_JSON:?export OmniDocBench.json}"
  : "${COMPILE_CACHE:?export the completed K20 recognition/decode cache parent}"
  : "${LAYOUT_FP32_CACHE:?export the warmed compiled-FP32 B2 layout cache}"
  : "${K20_FP16_REFERENCE_SUMMARY:?export the completed 8.32-pages/s K20 FP16 run summary}"
  : "${EAGER_FP32_FULL_OUTPUT:?export the canonical full1651 eager-FP32 output directory}"
  : "${EVALUATOR_ROOT:?export the clean OmniDocBench evaluator checkout}"
  : "${EVAL_PYTHON:?export the CDM-capable evaluator Python}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device, 0-3}"
  : "${TASKSET_CPUS:?export the known-good 64-CPU taskset mask}"
  : "${MATCH_WORKERS:=12}"
  : "${TEDS_WORKERS:=12}"
  : "${CDM_WORKERS:=64}"

  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  EVAL_PYTHON="$(absolute_executable_path "$EVAL_PYTHON")"
  for variable in MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR DATASET_JSON \
    COMPILE_CACHE LAYOUT_FP32_CACHE K20_FP16_REFERENCE_SUMMARY \
    EAGER_FP32_FULL_OUTPUT EVALUATOR_ROOT; do
    printf -v "$variable" '%s' "$(readlink -f "${!variable}")"
  done
  case "$ASCEND_RT_VISIBLE_DEVICES" in
    0|1|2|3) ;;
    *) printf '310P_DEVICE_MUST_BE_0_TO_3\n' >&2; exit 1 ;;
  esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  for worker_setting in MATCH_WORKERS TEDS_WORKERS CDM_WORKERS; do
    [[ "${!worker_setting}" =~ ^[1-9][0-9]*$ ]]
  done
  test -x "$PYTHON_BIN"
  test -x "$EVAL_PYTHON"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -f "$DATASET_JSON"
  test -d "$COMPILE_CACHE"
  test -d "$LAYOUT_FP32_CACHE"
  test -s "$K20_FP16_REFERENCE_SUMMARY"
  test -s "$EAGER_FP32_FULL_OUTPUT/run_summary.json"
  test -f "$EVALUATOR_ROOT/pdf_validation.py"
  test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
  local evaluator_dirty
  evaluator_dirty="$(
    git -C "$EVALUATOR_ROOT" status --porcelain=v1 --untracked-files=all \
      -- . ':(exclude)result/**'
  )"
  if [[ -n "$evaluator_dirty" ]]; then
    printf 'UNIREC_REP128_FP32_EVALUATOR_DIRTY_OUTSIDE_RESULT\n%s\n' \
      "$evaluator_dirty" >&2
    exit 1
  fi
  CPU_AFFINITY_COUNT="$(taskset -c "$TASKSET_CPUS" "$PYTHON_BIN" -c \
    'import os; print(len(os.sched_getaffinity(0)))')"
  if (( CPU_AFFINITY_COUNT < 64 )); then
    printf 'UNIREC_REP128_FP32_CPU_AFFINITY_TOO_SMALL count=%s mask=%s\n' \
      "$CPU_AFFINITY_COUNT" "$TASKSET_CPUS" >&2
    exit 1
  fi

  export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
  # shellcheck source=../09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
  source "$EVAL_ENV"
  EVAL_PYTHON="$OMNIDOCBENCH_EVAL_PYTHON"
  EVALUATOR_ROOT="$OMNIDOCBENCH_EVALUATOR_ROOT"
  [[ "$("$CDM_PDFLATEX" --version | head -n 1)" == *"1.40.28 (TeX Live 2025)"* ]]
  [[ "$("$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 1)" == *"ImageMagick 7.1.1-47"* ]]
  [[ "$(gs --version)" == "9.55.0" ]]
  test -n "$("$CDM_KPSEWHICH" CJK.sty)"
  test -n "$("$CDM_KPSEWHICH" c70gkai.fd)"
  test -n "$("$CDM_KPSEWHICH" xcolor.sty)"

  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR DATASET_JSON
  export COMPILE_CACHE LAYOUT_FP32_CACHE K20_FP16_REFERENCE_SUMMARY
  export EAGER_FP32_FULL_OUTPUT EVALUATOR_ROOT EVAL_PYTHON TASKSET_CPUS
  export CPU_AFFINITY_COUNT MATCH_WORKERS TEDS_WORKERS CDM_WORKERS
}

om_inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'recognition %p %s %T@\n'
    find "$LAYOUT_FP32_CACHE" -type f -name '*.om' -printf 'layout %p %s %T@\n'
  } | sort >"$output"
}

run_candidate() {
  local run_root="$1"
  local output="$run_root/candidate/output"
  mkdir -p "$output"
  local command=(
    "$PYTHON_BIN" "$RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --layout-execution torchair
    --layout-dtype float32
    --layout-reading-order-dtype float32
    --layout-weight-format native
    --layout-depthwise-rewrite native
    --layout-threshold 0.5
    --layout-cache-dir "$LAYOUT_FP32_CACHE"
    --input "$run_root/representative_128_v1_images"
    --output-dir "$output"
    --device npu:0
    --dtype float16
    --offset 0
    --limit 128
    --workers 4
    --warmup-pages 8
    --layout-batch-size 2
    --layout-cpu-threads 16
    --vision-page-lookahead 4
    --vision-bucket-preset 310p_k20_l4
    --vision-focal-depthwise-rewrite constant_grouped_all
    --vision-weight-format torchair_internal
    --recognition-preprocess-threads 8
    --recognition-input-contract compact_uint8_hwc
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --decode-batch-size 128
    --compile-cache-dir "$COMPILE_CACHE"
    --decode-warmup-passes 2
    --decode-admission-prefetch-depth 0
    --progress-every-pages 8
    --progress-heartbeat-s 15
  )
  printf '%q ' taskset -c "$TASKSET_CPUS" "${command[@]}" >"$run_root/candidate/command.sh"
  printf '\n' >>"$run_root/candidate/command.sh"
  phase candidate_inference_begin
  local started="$SECONDS"
  taskset -c "$TASKSET_CPUS" "${command[@]}" 2>&1 | tee "$run_root/candidate/run.log"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/candidate/process_wall_s.txt"
  phase candidate_inference_end
}

run_eval() {
  local lane="$1" source_output="$2" evaluation="$3"
  phase "${lane}_eval_prepare_begin"
  "$EVAL_PYTHON" "$PREP" \
    --dataset-json "$DATASET_JSON" \
    --output "$source_output" \
    --evaluation-root "$evaluation" \
    --offset 0 --limit 128 \
    --page-manifest "$MANIFEST" \
    --strip-image-tags
  phase "${lane}_eval_prepare_end"

  cd "$evaluation/work"
  ulimit -n 65536
  phase "${lane}_match_teds_begin"
  local started="$SECONDS"
  PYTHONUNBUFFERED=1 taskset -c "$TASKSET_CPUS" "$EVAL_PYTHON" \
    "$EVAL_RUNNER" \
    --config config.yaml \
    --evaluator-root "$EVALUATOR_ROOT" \
    --match-workers "$MATCH_WORKERS" --teds-workers "$TEDS_WORKERS" \
    --page-timeout-sec 120 \
    --fallback-timeout-sec 180 \
    --fallback-latex-timeout-sec 30 \
    >"$evaluation/match_teds.log" 2>&1
  printf '%s\n' "$((SECONDS - started))" >"$evaluation/match_teds_wall_s.txt"
  phase "${lane}_match_teds_end"

  mkdir -p "$evaluation/cdm"
  phase "${lane}_cdm_begin"
  started="$SECONDS"
  PYTHONUNBUFFERED=1 taskset -c "$TASKSET_CPUS" "$EVAL_PYTHON" \
    "$CDM_RUNNER" \
    --input "$evaluation/work/result/predictions_quick_match_display_formula_result.json" \
    --output-dir "$evaluation/cdm" \
    --evaluator-root "$EVALUATOR_ROOT" \
    --workers "$CDM_WORKERS" \
    >"$evaluation/cdm/run.log" 2>&1
  printf '%s\n' "$((SECONDS - started))" >"$evaluation/cdm_wall_s.txt"
  phase "${lane}_cdm_end"

  "$EVAL_PYTHON" "$SUMMARIZER" \
    --lane "$lane" \
    --metric-result "$evaluation/work/result/predictions_quick_match_metric_result.json" \
    --stage-execution "$evaluation/work/result/predictions_quick_match_stage_execution.json" \
    --cdm-summary "$evaluation/cdm/cdm_run_summary.json" \
    --output "$evaluation/full_eval_summary.json" \
    | tee "$evaluation/full_eval_sentence.txt"
  cd "$REPO"
}

write_report() {
  local run_root="$1"
  RUN_ROOT="$run_root" "$EVAL_PYTHON" - <<'PY' | tee "$run_root/final_report.txt"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
candidate = json.loads((root / "candidate/output/run_summary.json").read_text())
fp16 = json.load(open(os.environ["K20_FP16_REFERENCE_SUMMARY"]))
baseline_score = json.loads((root / "evaluation_baseline/full_eval_summary.json").read_text())
candidate_score = json.loads((root / "evaluation_candidate/full_eval_summary.json").read_text())
markdown = json.loads((root / "markdown_comparison.json").read_text())

assert candidate["status"] == fp16["status"] == "ok"
for key, expected in {
    "page_count": 128,
    "workers": 4,
    "recognition_preprocess_threads": 8,
    "layout_batch_size": 2,
    "layout_cpu_threads": 16,
    "layout_execution": "torchair",
    "layout_dtype": "float32",
    "layout_reading_order_dtype": "float32",
    "layout_weight_format": "native",
    "layout_depthwise_rewrite": "native",
    "layout_threshold": 0.5,
    "vision_page_lookahead": 4,
    "vision_bucket_preset": "310p_k20_l4",
    "vision_focal_depthwise_rewrite": "constant_grouped_all",
    "vision_weight_format": "torchair_internal",
    "cross_cache_length": 1320,
    "self_cache_length": 2048,
    "decode_batch_size": 128,
}.items():
    assert candidate[key] == expected, (key, candidate[key])
assert fp16["page_count"] == 128
assert fp16["vision_bucket_preset"] == "310p_k20_l4"
assert fp16["layout_dtype"] == "float16"
assert fp16["layout_execution"] == "torchair"

for score in (baseline_score, candidate_score):
    assert score["page_match"]["page_count"] == 128
    assert score["page_match"]["fallbacks"]["page_timeout"]["count"] == 0
    assert score["page_match"]["fallbacks"]["quick_match_timeout"]["count"] == 0
    assert score["table_teds_execution"]["timeout_case_count"] == 0
    assert score["cdm_debug"]["timeout_case_count"] == 0

def perf(run):
    stages = run["prefill_phase_summary"]["stage_s"]
    return {
        "crops": run["crop_count"],
        "prefill_s": run["timing_s"]["prefill_phase"],
        "prefill_pages_s": run["throughput"]["prefill_pages_per_s"],
        "layout_service_s": stages["worker_detector_call_sum_s"],
        "decode_s": run["timing_s"].get("decode_inference_including_ingress"),
        "sequential_core_s": run["timing_s"].get("sequential_inference_core"),
        "sequential_pages_s": run["throughput"].get("sequential_inference_core_pages_per_s"),
    }

def accuracy(score):
    return {
        "text_edit": score["text_block_page_edit"],
        "page_cdm": score["display_formula_page_cdm"],
        "page_teds": score["table_page_teds"],
        "reading_edit": score["reading_order_page_edit"],
        "overall_percent": 100.0 * score["official_overall"],
    }

candidate_perf = perf(candidate)
fp16_perf = perf(fp16)
baseline_accuracy = accuracy(baseline_score)
candidate_accuracy = accuracy(candidate_score)
report = {
    "schema": "unirec_310p_k20_rep128_compiled_fp32_accuracy_v1",
    "candidate_compiled_fp32": candidate_perf,
    "performance_reference_compiled_fp16": fp16_perf,
    "candidate_over_fp16_prefill_throughput": (
        candidate_perf["prefill_pages_s"] / fp16_perf["prefill_pages_s"]
    ),
    "baseline_eager_fp32_accuracy": baseline_accuracy,
    "candidate_compiled_fp32_accuracy": candidate_accuracy,
    "accuracy_delta_candidate_minus_baseline": {
        key: candidate_accuracy[key] - baseline_accuracy[key]
        for key in baseline_accuracy
    },
    "markdown_parity": {
        key: markdown[key]
        for key in (
            "page_count",
            "raw_exact_count",
            "raw_exact_fraction",
            "image_tag_stripped_exact_count",
            "image_tag_stripped_exact_fraction",
            "differing_stems",
        )
    },
}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_K20_REP128_COMPILED_FP32_ACCURACY: PASS")
print("UNIREC_310P_REP128_FP32_PERFORMANCE " + json.dumps(candidate_perf, sort_keys=True))
print("UNIREC_310P_REP128_FP16_REFERENCE " + json.dumps(fp16_perf, sort_keys=True))
print("UNIREC_310P_REP128_BASELINE_ACCURACY " + json.dumps(baseline_accuracy, sort_keys=True))
print("UNIREC_310P_REP128_CANDIDATE_ACCURACY " + json.dumps(candidate_accuracy, sort_keys=True))
print("UNIREC_310P_REP128_ACCURACY_DELTA " + json.dumps(report["accuracy_delta_candidate_minus_baseline"], sort_keys=True))
print("UNIREC_310P_REP128_MARKDOWN_PARITY " + json.dumps(report["markdown_parity"], sort_keys=True))
print(f"UNIREC_310P_REP128_COMPARISON_JSON={root / 'comparison.json'}")
PY
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" "$PYTHON_BIN"
    printf 'taskset=%s\ncpu_affinity_count=%s\n' "$TASKSET_CPUS" "$CPU_AFFINITY_COUNT"
    printf 'compile_cache=%s\nlayout_fp32_cache=%s\nfp16_reference=%s\neager_fp32_full_output=%s\n' \
      "$COMPILE_CACHE" "$LAYOUT_FP32_CACHE" "$K20_FP16_REFERENCE_SUMMARY" "$EAGER_FP32_FULL_OUTPUT"
    printf 'evaluator_root=%s\nevaluator_commit=%s\neval_tools=%s\n' \
      "$EVALUATOR_ROOT" "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
    "$PYTHON_BIN" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
    "$CDM_PDFLATEX" --version | head -n 2
    "$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 2
    printf 'ghostscript=%s\n' "$(gs --version)"
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  "$PYTHON_BIN" "$MATERIALIZER" --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" --output-dir "$run_root/representative_128_v1_images"
  om_inventory "$run_root/om_before.txt"
  run_candidate "$run_root"
  om_inventory "$run_root/om_after.txt"
  if diff -u "$run_root/om_before.txt" "$run_root/om_after.txt" >"$run_root/hot_om.diff"; then
    printf 'UNIREC_310P_REP128_FP32_HOT_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_310P_REP128_FP32_HOT_OM_INVENTORY_CHANGED\n' >&2
    cat "$run_root/hot_om.diff" >&2
    return 1
  fi

  "$PYTHON_BIN" "$MD_COMPARE" \
    --manifest "$MANIFEST" \
    --baseline-output "$EAGER_FP32_FULL_OUTPUT" \
    --candidate-output "$run_root/candidate/output" \
    --output "$run_root/markdown_comparison.json"

  run_eval representative128_eager_fp32_baseline \
    "$EAGER_FP32_FULL_OUTPUT" "$run_root/evaluation_baseline"
  run_eval representative128_compiled_fp32_candidate \
    "$run_root/candidate/output" "$run_root/evaluation_candidate"
  write_report "$run_root"
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_REP128_FP32_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_k20_rep128_compiled_fp32_accuracy_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT/candidate"
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" LAYOUT_MODEL="$LAYOUT_MODEL" \
    OPENOCR_ROOT="$OPENOCR_ROOT" IMAGES_DIR="$IMAGES_DIR" \
    DATASET_JSON="$DATASET_JSON" COMPILE_CACHE="$COMPILE_CACHE" \
    LAYOUT_FP32_CACHE="$LAYOUT_FP32_CACHE" \
    K20_FP16_REFERENCE_SUMMARY="$K20_FP16_REFERENCE_SUMMARY" \
    EAGER_FP32_FULL_OUTPUT="$EAGER_FP32_FULL_OUTPUT" \
    EVALUATOR_ROOT="$EVALUATOR_ROOT" EVAL_PYTHON="$EVAL_PYTHON" \
    OMNIDOCBENCH_EVAL_TOOLS_ROOT="$OMNIDOCBENCH_EVAL_TOOLS_ROOT" \
    OMNIDOCBENCH_TOOL_ROOT="$OMNIDOCBENCH_TOOL_ROOT" \
    CDM_PDFLATEX="$CDM_PDFLATEX" CDM_KPSEWHICH="$CDM_KPSEWHICH" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    TASKSET_CPUS="$TASKSET_CPUS" MATCH_WORKERS="$MATCH_WORKERS" \
    TEDS_WORKERS="$TEDS_WORKERS" CDM_WORKERS="$CDM_WORKERS" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == worker ]]; then worker_entry "$2"; else launch_main; fi
