#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/min_pixels_compare_${RUN_ID}}"
export BASELINE_MIN_PIXELS="${BASELINE_MIN_PIXELS:-112896}"
export CANDIDATE_MIN_PIXELS_LIST="${CANDIDATE_MIN_PIXELS_LIST:-50176 28224}"
export PAGE_START="${PAGE_START:-0}"
export NUM_PAGES="${NUM_PAGES:-8}"
export ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
export CACHE_LENGTH="${CACHE_LENGTH:-2048}"
export CROP_CHUNK_SIZE="${CROP_CHUNK_SIZE:-0}"
export VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
export DECODE_BACKEND="${DECODE_BACKEND:-torchair}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-0}"
export CHECK_PADDLE_IMPORT="${CHECK_PADDLE_IMPORT:-0}"
export FAIL_ON_MISMATCH="${FAIL_ON_MISMATCH:-1}"
export FAIL_ON_LENGTH_CAP="${FAIL_ON_LENGTH_CAP:-0}"
export EXPECT_LAYOUT_SOURCE="${EXPECT_LAYOUT_SOURCE:-omnidocbench_gt}"
export STRICT_KNOWN_FIRST64_GT_MANIFEST="${STRICT_KNOWN_FIRST64_GT_MANIFEST:-0}"
export TORCHAIR_CACHE_DIR="${TORCHAIR_CACHE_DIR:-${SCRIPT_DIR}/outputs/torchair_cache_page_pipeline_npu}"

mkdir -p "${OUTPUT_DIR}"

run_case() {
  local name="$1"
  local min_pixels="$2"
  local output_path="${OUTPUT_DIR}/${name}_min${min_pixels}.json"
  echo "MIN_PIXELS_COMPARE_RUN name=${name} min_pixels=${min_pixels} output=${output_path}"
  PREPROCESSOR_MIN_PIXELS="${min_pixels}" \
  OUTPUT_PATH="${output_path}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RUN_ID="${RUN_ID}_${name}_min${min_pixels}" \
  bash "${SCRIPT_DIR}/run_npu_page_pipeline_smoke.sh"
}

run_case "baseline" "${BASELINE_MIN_PIXELS}"

candidate_paths=()
for min_pixels in ${CANDIDATE_MIN_PIXELS_LIST}; do
  run_case "candidate" "${min_pixels}"
  candidate_paths+=("${OUTPUT_DIR}/candidate_min${min_pixels}.json")
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}/baseline_min${BASELINE_MIN_PIXELS}.json" "${candidate_paths[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_output_path"] = str(Path(path))
    return data


def avg(values):
    return None if not values else float(sum(values) / len(values))


def summarize_run(data: dict) -> dict:
    rows = data.get("output_fingerprints", [])
    vision_tokens = [int(row.get("vision_tokens", 0) or 0) for row in rows]
    projected_tokens = [int(row.get("projected_image_tokens", 0) or 0) for row in rows]
    input_tokens = [int(row.get("input_tokens", 0) or 0) for row in rows]
    trimmed_tokens = [int(row.get("trimmed_token_count", 0) or 0) for row in rows]
    pre = data.get("preprocessor", {})
    phase = data.get("phase_timing_s", {})
    throughput = data.get("throughput", {})
    rough = data.get("rough_ground_truth_accuracy", {})
    metrics = data.get("omnidocbench_metrics_without_cdm", {})
    return {
        "path": data.get("_output_path"),
        "min_pixels": pre.get("min_pixels"),
        "min_projected_image_tokens": pre.get("min_projected_image_tokens"),
        "recognizer_crop_count": data.get("recognizer_crop_count"),
        "fingerprints_sha256": data.get("output_fingerprint_summary", {}).get("fingerprints_sha256"),
        "generated_texts_sha256": data.get("output_fingerprint_summary", {}).get("generated_texts_sha256"),
        "token_ids_sha256": data.get("output_fingerprint_summary", {}).get("token_ids_sha256"),
        "vision_tokens_sum": int(sum(vision_tokens)),
        "vision_tokens_avg": avg(vision_tokens),
        "projected_image_tokens_sum": int(sum(projected_tokens)),
        "projected_image_tokens_avg": avg(projected_tokens),
        "input_tokens_avg": avg(input_tokens),
        "trimmed_tokens_sum": int(sum(trimmed_tokens)),
        "pages_per_s": throughput.get("pages_per_s_measured_e2e"),
        "crops_per_s": throughput.get("crops_per_s_measured_e2e"),
        "prefill_crops_per_s": throughput.get("prefill_crops_per_s"),
        "effective_decode_tokens_per_s": throughput.get("effective_decode_tokens_per_s"),
        "recognizer_cpu_input_build_s": phase.get("recognizer_cpu_input_build"),
        "recognizer_ready_bank_build_s": phase.get("recognizer_ready_bank_build"),
        "text_decode_queue_s": phase.get("text_decode_queue"),
        "rough_exact_rate": rough.get("normalized_exact_rate"),
        "rough_avg_sequence_ratio": rough.get("avg_sequence_ratio"),
        "available_non_cdm_component_mean_score_percent": metrics.get("available_non_cdm_component_mean_score_percent"),
        "text_table_conclusion_mean_score_percent": metrics.get("text_table_conclusion_mean_score_percent"),
        "correctness": data.get("correctness"),
    }


def compare_rows(base: dict, candidate: dict) -> dict:
    base_rows = {str(row.get("id")): row for row in base.get("output_fingerprints", [])}
    cand_rows = {str(row.get("id")): row for row in candidate.get("output_fingerprints", [])}
    common_ids = sorted(set(base_rows) & set(cand_rows))
    missing_in_candidate = sorted(set(base_rows) - set(cand_rows))
    extra_in_candidate = sorted(set(cand_rows) - set(base_rows))
    token_mismatches = []
    text_mismatches = []
    shape_changes = []
    for item_id in common_ids:
        lhs = base_rows[item_id]
        rhs = cand_rows[item_id]
        if lhs.get("token_ids_sha256") != rhs.get("token_ids_sha256"):
            token_mismatches.append(item_id)
        if lhs.get("generated_text_sha256") != rhs.get("generated_text_sha256"):
            text_mismatches.append(item_id)
        if (
            int(lhs.get("vision_tokens", 0) or 0) != int(rhs.get("vision_tokens", 0) or 0)
            or int(lhs.get("projected_image_tokens", 0) or 0) != int(rhs.get("projected_image_tokens", 0) or 0)
            or int(lhs.get("input_tokens", 0) or 0) != int(rhs.get("input_tokens", 0) or 0)
        ):
            shape_changes.append(
                {
                    "id": item_id,
                    "layout_label": lhs.get("layout_label"),
                    "crop_size": lhs.get("crop_size"),
                    "baseline_vision_tokens": lhs.get("vision_tokens"),
                    "candidate_vision_tokens": rhs.get("vision_tokens"),
                    "baseline_projected_image_tokens": lhs.get("projected_image_tokens"),
                    "candidate_projected_image_tokens": rhs.get("projected_image_tokens"),
                    "baseline_input_tokens": lhs.get("input_tokens"),
                    "candidate_input_tokens": rhs.get("input_tokens"),
                }
            )
    reduction_rows = [
        int(base_rows[item_id].get("vision_tokens", 0) or 0) - int(cand_rows[item_id].get("vision_tokens", 0) or 0)
        for item_id in common_ids
    ]
    changed_reductions = [value for value in reduction_rows if value != 0]
    return {
        "common_item_count": len(common_ids),
        "missing_in_candidate_count": len(missing_in_candidate),
        "extra_in_candidate_count": len(extra_in_candidate),
        "token_mismatch_count": len(token_mismatches),
        "text_mismatch_count": len(text_mismatches),
        "shape_change_count": len(shape_changes),
        "vision_token_reduction_sum": int(sum(reduction_rows)),
        "vision_token_reduction_avg_all_common": avg(reduction_rows),
        "vision_token_reduction_avg_changed_only": avg(changed_reductions),
        "vision_token_reduction_p50_changed_only": (
            None if not changed_reductions else float(statistics.median(changed_reductions))
        ),
        "sample_token_mismatch_ids": token_mismatches[:16],
        "sample_text_mismatch_ids": text_mismatches[:16],
        "sample_shape_changes": shape_changes[:16],
    }


baseline_path = sys.argv[1]
candidate_paths = sys.argv[2:]
baseline = load(baseline_path)
summary = {
    "baseline": summarize_run(baseline),
    "candidates": [],
}
for candidate_path in candidate_paths:
    candidate = load(candidate_path)
    summary["candidates"].append(
        {
            "run": summarize_run(candidate),
            "vs_baseline": compare_rows(baseline, candidate),
        }
    )
print("MIN_PIXELS_COMPARE_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "MIN_PIXELS_COMPARE_OUTPUT_DIR=${OUTPUT_DIR}"
