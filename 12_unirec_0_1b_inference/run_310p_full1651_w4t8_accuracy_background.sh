#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNNER="$SCRIPT_DIR/run_two_phase_batched_unirec.py"
DECODE_CACHE_PROBE="$SCRIPT_DIR/probe_production_decode_cache_contract.py"
PREP="$SCRIPT_DIR/prepare_unirec_subset_eval.py"
SUMMARIZER="$SCRIPT_DIR/summarize_completed_unirec_eval.py"
EVAL_ENV="$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
EVAL_RUNTIME_VERIFY="$REPO/09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py"
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

phase() {
  printf 'UNIREC_310P_FULL1651_PHASE phase=%s epoch_s=%s\n' \
    "$1" "$(date +%s)"
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
  : "${MATCH_WORKERS:=12}"
  : "${TEDS_WORKERS:=12}"
  : "${RUN_VARIANT:=accuracy_anchor}"
  : "${REQUIRE_WARM_VISION_CACHE:=0}"
  : "${DECODE_CACHE_GATE_ATTEMPTS:=3}"
  : "${LAYOUT_CPU_THREADS:=1}"
  : "${PROGRESS_EVERY_PAGES:=1}"
  : "${ALLOW_LOW_HOST_MEMORY:=0}"
  : "${CPUSET:=}"

  case ",${ASCEND_RT_VISIBLE_DEVICES}," in
    *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
  esac
  if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    printf 'REQUIRES_EXACTLY_ONE_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  for worker_setting in CDM_WORKERS MATCH_WORKERS TEDS_WORKERS \
      LAYOUT_CPU_THREADS PROGRESS_EVERY_PAGES DECODE_CACHE_GATE_ATTEMPTS; do
    if ! [[ "${!worker_setting}" =~ ^[1-9][0-9]*$ ]]; then
      printf 'INVALID_POSITIVE_INTEGER %s=%s\n' \
        "$worker_setting" "${!worker_setting}" >&2
      exit 1
    fi
  done
  case "$RUN_VARIANT" in
    accuracy_anchor|optimized_k10_l4|optimized_k10_l4_aligned) ;;
    *) printf 'INVALID_RUN_VARIANT=%s\n' "$RUN_VARIANT" >&2; exit 1 ;;
  esac
  case "$ALLOW_LOW_HOST_MEMORY" in
    0|1) ;;
    *) printf 'INVALID_ALLOW_LOW_HOST_MEMORY=%s\n' "$ALLOW_LOW_HOST_MEMORY" >&2; exit 1 ;;
  esac
  case "$REQUIRE_WARM_VISION_CACHE" in
    0|1) ;;
    *) printf 'INVALID_REQUIRE_WARM_VISION_CACHE=%s\n' "$REQUIRE_WARM_VISION_CACHE" >&2; exit 1 ;;
  esac

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
  if [[ -n "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:-}" ]]; then
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$(
      readlink -f "$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE"
    )"
    local exact_decode_cache
    exact_decode_cache="$UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE/decode_selfkv2048_cross1320_increfa_all_b128"
    test "$(find "$exact_decode_cache" -name compiled_module | wc -l)" -eq 1
    test "$(find "$exact_decode_cache" -name '*.om' | wc -l)" -eq 1
    export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE
  elif [[ "$RUN_VARIANT" == optimized_k10_l4_aligned ]]; then
    printf 'ALIGNED_RUN_REQUIRES_PASSED_DECODE_CACHE_OVERRIDE\n' >&2
    exit 1
  fi

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
  test -f "$DECODE_CACHE_PROBE"
  test -f "$PREP"
  test -f "$SUMMARIZER"
  test -f "$CDM_RUNNER"
  test -f "$EVAL_RUNTIME_VERIFY"
  "$PYTHON_BIN" -c 'import kornia_rs, torch, torch_npu'
  PYTHONPATH="$EVALUATOR_ROOT" "$EVAL_PYTHON" -c \
    'import Levenshtein; import src.metrics.cal_metric'
  PYTHONPATH="$(dirname "$CDM_RUNNER")" "$EVAL_PYTHON" -c \
    'from run_cdm_from_matched_formulas import _configure_cdm_runtime; print(_configure_cdm_runtime())'

  # CDM scores are environment-sensitive. Refuse the ambient 310P TeX 2022
  # installation and require the frozen cross-host runtime used for the
  # canonical 910B/310P same-host replay.
  [[ "$("$CDM_PDFLATEX" --version | head -n 1)" == *"1.40.28 (TeX Live 2025)"* ]]
  [[ "$("$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 1)" == *"ImageMagick 7.1.1-47"* ]]
  [[ "$(gs --version)" == "9.55.0" ]]
  test -n "$("$CDM_KPSEWHICH" CJK.sty)"
  test -n "$("$CDM_KPSEWHICH" c70gkai.fd)"
  test -n "$("$CDM_KPSEWHICH" xcolor.sty)"
}

vision_cache_inventory() {
  local output="$1"
  COMPILE_CACHE="$COMPILE_CACHE" SCRIPT_DIR="$SCRIPT_DIR" \
    "$PYTHON_BIN" - "$output" <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["SCRIPT_DIR"])
import vision_full_batch

slots = {
    "448x384_b2": 1,
    "512x64_b4": 2,
    "512x192_b2": 0,
    "960x64_b4": 3,
    "960x128_b2": 4,
    "960x256_b1": 5,
    "960x512_b1": 9,
    "960x1024_b1": 7,
    "1024x704_b1": 8,
    "1024x1408_b1": 6,
}
root = Path(os.environ["COMPILE_CACHE"])
legacy_hash = vision_full_batch._source_hash()
flat_hash = vision_full_batch._flat_global_context_source_hash()
flat_keys = set(vision_full_batch.FLAT_GLOBAL_CONTEXT_BUCKET_KEYS)
report = {}
for key, slot in slots.items():
    use_flat = key in flat_keys
    source_hash = flat_hash if use_flat else legacy_hash
    method = (
        f"_forward_flat_bucket_slot_{slot}"
        if use_flat
        else f"_forward_bucket_slot_{slot}"
    )
    directories = sorted(root.glob(
        f"vision_full_bucket_{key}_float16_src{source_hash}_"
        "dwconstant_grouped_all*wtorchair_internal*"
    ))
    modules = []
    oms = []
    for directory in directories:
        found = list(directory.glob(f"**/{method}/compiled_module"))
        modules.extend(found)
        for module in found:
            oms.extend(module.parent.glob("*.om"))
    report[key] = {
        "slot": slot,
        "method": method,
        "source_hash": source_hash,
        "global_context_mode": "direct_2d" if use_flat else "legacy_two_stage",
        "target_compiled_module_count": len(set(modules)),
        "target_om_count": len(set(oms)),
        "target_compiled_modules": [str(path) for path in sorted(set(modules))],
        "target_oms": [str(path) for path in sorted(set(oms))],
    }
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
missing = [key for key, row in report.items() if not row["target_compiled_module_count"]]
print(
    "UNIREC_310P_FULL1651_VISION_CACHE "
    f"legacy_hash={legacy_hash} flat_hash={flat_hash} "
    f"missing={len(missing)} keys={missing}"
)
if missing:
    raise SystemExit(1)
PY
}

om_inventory() {
  local output="$1"
  find "$COMPILE_CACHE" -type f -name '*.om' -printf '%p %s %T@\n' \
    | sort >"$output"
}

gate_decode_cache() {
  local gate_root="$RUN_ROOT/decode_cache_gate"
  mkdir -p "$gate_root"
  local attempt status result log
  for ((attempt = 1; attempt <= DECODE_CACHE_GATE_ATTEMPTS; attempt++)); do
    result="$gate_root/attempt_${attempt}.json"
    log="$gate_root/attempt_${attempt}.log"
    set +e
    "$PYTHON_BIN" "$DECODE_CACHE_PROBE" \
      --model-path "$MODEL" \
      --compile-cache-dir "$COMPILE_CACHE" \
      --device npu:0 \
      --batch-size 128 \
      --self-cache-length 2048 \
      --cross-cache-length 1320 \
      --passes 2 \
      --output "$result" \
      >"$log" 2>&1
    status="$?"
    set -e
    cat "$log"
    printf 'UNIREC_310P_DECODE_CACHE_GATE attempt=%s status=%s result=%s\n' \
      "$attempt" "$status" "$result"
    if [[ "$status" == 0 ]]; then
      cp "$result" "$gate_root/passed.json"
      return 0
    fi
    if [[ "$status" != 3 && "$status" != 4 ]]; then
      return "$status"
    fi
  done
  printf 'UNIREC_310P_DECODE_CACHE_GATE_FAILED attempts=%s\n' \
    "$DECODE_CACHE_GATE_ATTEMPTS" >&2
  return 1
}

verify_evaluator_source() {
  test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
  local non_result_status
  non_result_status="$(
    git -C "$EVALUATOR_ROOT" status --porcelain=v1 --untracked-files=all \
      -- . ':(exclude)result/**'
  )"
  if [[ -n "$non_result_status" ]]; then
    printf 'UNIREC_EVALUATOR_SOURCE_DIRTY_OUTSIDE_RESULT\n%s\n' \
      "$non_result_status" >&2
    return 1
  fi
  printf 'UNIREC_EVALUATOR_SOURCE_VERIFIED commit=%s allowed_dirty=result_only\n' \
    "$EVALUATOR_COMMIT"
}

verify_evaluator_runtime() {
  {
    printf 'tools_root=%s\n' "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
    "$CDM_PDFLATEX" --version | head -n 2
    "$CDM_KPSEWHICH" --version | head -n 2
    "$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 2
    printf 'ghostscript=%s\n' "$(gs --version)"
    for resource in CJK.sty c70gkai.fd xcolor.sty; do
      printf '%s=%s\n' "$resource" "$("$CDM_KPSEWHICH" "$resource")"
    done
  } >"$RUN_ROOT/evaluator_runtime_versions.txt"
  "$EVAL_PYTHON" "$EVAL_RUNTIME_VERIFY" \
    --evaluator-root "$EVALUATOR_ROOT" \
    >"$RUN_ROOT/evaluator_runtime_smoke.json"
  printf 'UNIREC_EVALUATOR_RUNTIME_VERIFIED tools_root=%s\n' \
    "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
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
    --layout-cpu-threads "$LAYOUT_CPU_THREADS"
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
    --progress-every-pages "$PROGRESS_EVERY_PAGES"
    --progress-heartbeat-s 15
  )
  if [[ "$RUN_VARIANT" == optimized_k10_l4 ]]; then
    command+=(
      --vision-bucket-preset 310p_k10_l4_all
      --vision-focal-depthwise-rewrite constant_grouped_all
      --vision-weight-format torchair_internal
    )
  elif [[ "$RUN_VARIANT" == optimized_k10_l4_aligned ]]; then
    command+=(
      --vision-bucket-preset 310p_k10_l4_aligned
      --vision-focal-depthwise-rewrite constant_grouped_all
      --vision-weight-format torchair_internal
    )
  fi
  if [[ -n "$CPUSET" ]]; then
    command=(taskset -c "$CPUSET" "${command[@]}")
  fi
  printf '%q ' "${command[@]}" >"$RUN_ROOT/command.sh"
  printf '\n' >>"$RUN_ROOT/command.sh"
  local started_ns ended_ns
  started_ns="$(date +%s%N)"
  phase inference_begin
  "${command[@]}"
  ended_ns="$(date +%s%N)"
  "$EVAL_PYTHON" -c \
    'import sys; print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1e9:.6f}")' \
    "$started_ns" "$ended_ns" >"$RUN_ROOT/inference_process_wall_s.txt"
  phase inference_end
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
  local eval_prefix=()
  if [[ -n "$CPUSET" ]]; then
    eval_prefix=(taskset -c "$CPUSET")
  fi
  phase evaluation_match_teds_begin
  PYTHONUNBUFFERED=1 "${eval_prefix[@]}" "$EVAL_PYTHON" \
    "$REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py" \
    --config config.yaml \
    --evaluator-root "$EVALUATOR_ROOT" \
    --match-workers "$MATCH_WORKERS" --teds-workers "$TEDS_WORKERS" \
    --page-timeout-sec 120 \
    --fallback-timeout-sec 180 \
    --fallback-latex-timeout-sec 30
  printf '%s\n' "$((SECONDS - started))" >"$evaluation/eval_match_teds_wall_s.txt"
  phase evaluation_match_teds_end

  mkdir -p "$evaluation/cdm"
  local cdm_started="$SECONDS"
  phase evaluation_cdm_begin
  PYTHONUNBUFFERED=1 "${eval_prefix[@]}" "$EVAL_PYTHON" \
    "$CDM_RUNNER" \
    --input "$evaluation/work/result/predictions_quick_match_display_formula_result.json" \
    --output-dir "$evaluation/cdm" \
    --evaluator-root "$EVALUATOR_ROOT" \
    --workers "$CDM_WORKERS" \
    >"$evaluation/cdm/run.log" 2>&1
  printf '%s\n' "$((SECONDS - cdm_started))" >"$evaluation/cdm_wall_s.txt"
  phase evaluation_cdm_end

  "$EVAL_PYTHON" "$SUMMARIZER" \
    --lane "full1651_${RUN_VARIANT}_w4t8_b128_cross1320_self2048_t05_image_tags_stripped" \
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
variant = os.environ["RUN_VARIANT"]
if variant in {"optimized_k10_l4", "optimized_k10_l4_aligned"}:
    expected_preset = (
        "310p_k10_l4_aligned"
        if variant == "optimized_k10_l4_aligned"
        else "310p_k10_l4_all"
    )
    assert run["vision_bucket_preset"] == expected_preset
    assert run["vision_focal_depthwise_rewrite"] == "constant_grouped_all"
    assert run["vision_weight_format"] == "torchair_internal"
    assert run["prefill_phase_summary"]["vision_batching"]["fallback_rows"] == 0
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
  verify_evaluator_source
  phase evaluator_runtime_gate_begin
  verify_evaluator_runtime
  phase evaluator_runtime_gate_end
  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'run_variant=%s\ncpuset=%s\n' "$RUN_VARIANT" "${CPUSET:-unrestricted}"
    printf 'match_workers=%s\nteds_workers=%s\n' "$MATCH_WORKERS" "$TEDS_WORKERS"
    printf 'cdm_workers=%s\n' "$CDM_WORKERS"
    printf 'evaluator_root=%s\nevaluator_commit=%s\n' \
      "$EVALUATOR_ROOT" "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)"
    printf 'evaluator_tools_root=%s\n' "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
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
  export UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG=1
  if [[ "$REQUIRE_WARM_VISION_CACHE" == 1 ]]; then
    phase warm_vision_cache_gate_begin
    vision_cache_inventory "$RUN_ROOT/vision_cache_before.json"
    phase warm_vision_cache_gate_end
  fi
  om_inventory "$RUN_ROOT/om_before_inference.txt"
  phase decode_cache_gate_begin
  gate_decode_cache
  phase decode_cache_gate_end
  om_inventory "$RUN_ROOT/om_after_decode_gate.txt"
  if ! diff -u "$RUN_ROOT/om_before_inference.txt" \
      "$RUN_ROOT/om_after_decode_gate.txt" >"$RUN_ROOT/decode_gate_om.diff"; then
    printf 'UNIREC_310P_FULL1651_DECODE_GATE_CHANGED_OM_INVENTORY\n' >&2
    cat "$RUN_ROOT/decode_gate_om.diff" >&2
    return 1
  fi
  run_inference
  om_inventory "$RUN_ROOT/om_after_inference.txt"
  if diff -u "$RUN_ROOT/om_before_inference.txt" \
      "$RUN_ROOT/om_after_inference.txt" >"$RUN_ROOT/inference_om.diff"; then
    printf 'UNIREC_310P_FULL1651_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_310P_FULL1651_OM_INVENTORY_CHANGED\n' >&2
    cat "$RUN_ROOT/inference_om.diff" >&2
    return 1
  fi
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
    if [[ "$ALLOW_LOW_HOST_MEMORY" == 1 ]]; then
      printf 'LOW_HOST_MEMORY_PROCEEDING shm=%s/%s ram=%s/%s\n' \
        "$shm_available" "$minimum_shm" "$mem_available" "$minimum_mem" >&2
    else
      printf 'INSUFFICIENT_HOST_MEMORY shm=%s/%s ram=%s/%s\n' \
        "$shm_available" "$minimum_shm" "$mem_available" "$minimum_mem" >&2
      exit 1
    fi
  fi
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_full1651_${RUN_VARIANT}_w4t8_eval_${short}_${timestamp}}"
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
    CDM_WORKERS="$CDM_WORKERS" MATCH_WORKERS="$MATCH_WORKERS" \
    TEDS_WORKERS="$TEDS_WORKERS" RUN_VARIANT="$RUN_VARIANT" \
    REQUIRE_WARM_VISION_CACHE="$REQUIRE_WARM_VISION_CACHE" \
    DECODE_CACHE_GATE_ATTEMPTS="$DECODE_CACHE_GATE_ATTEMPTS" \
    LAYOUT_CPU_THREADS="$LAYOUT_CPU_THREADS" \
    PROGRESS_EVERY_PAGES="$PROGRESS_EVERY_PAGES" \
    ALLOW_LOW_HOST_MEMORY="$ALLOW_LOW_HOST_MEMORY" CPUSET="$CPUSET" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:-}" \
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
