#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/workspace/.hf_home/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e}"
DATASET_DIR="${DATASET_DIR:-/workspace/data/OmniDocBench}"
PAGE_START="${PAGE_START:-0}"
NUM_PAGES="${NUM_PAGES:-4}"
DEVICE="${DEVICE:-cuda:0}"
LAYOUT_DEVICE="${LAYOUT_DEVICE:-cpu}"
LAYOUT_SOURCE="${LAYOUT_SOURCE:-omnidocbench_gt}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
CACHE_LENGTH="${CACHE_LENGTH:-2048}"
PREPROCESSOR_MIN_PIXELS="${PREPROCESSOR_MIN_PIXELS:--1}"
PREPROCESSOR_MAX_PIXELS="${PREPROCESSOR_MAX_PIXELS:--1}"
DECODE_BACKEND="${DECODE_BACKEND:-raw_eager}"
NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
VALIDATION_ITEMS="${VALIDATION_ITEMS:--1}"
VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-manual}"
VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
PAGE_CHUNK_SIZE="${PAGE_CHUNK_SIZE:-0}"
CROP_CHUNK_SIZE="${CROP_CHUNK_SIZE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/page_pipeline_cuda_smoke}"
TORCHAIR_CACHE_DIR="${TORCHAIR_CACHE_DIR:-${OUTPUT_DIR}/torchair_cache}"
EXPECT_LAYOUT_SOURCE="${EXPECT_LAYOUT_SOURCE:-}"
EXPECTED_RECOGNIZER_CROPS="${EXPECTED_RECOGNIZER_CROPS:-}"
MIN_RECOGNIZER_CROPS="${MIN_RECOGNIZER_CROPS:-}"
EXPECT_GT_CROP_MANIFEST="${EXPECT_GT_CROP_MANIFEST:-}"
STRICT_KNOWN_FIRST64_GT_MANIFEST="${STRICT_KNOWN_FIRST64_GT_MANIFEST:-0}"
INCLUDE_IGNORED_GT="${INCLUDE_IGNORED_GT:-0}"
INCLUDE_EMPTY_GT="${INCLUDE_EMPTY_GT:-0}"
FAIL_ON_MISMATCH="${FAIL_ON_MISMATCH:-1}"
FAIL_ON_LENGTH_CAP="${FAIL_ON_LENGTH_CAP:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/page_pipeline_${RUN_ID}_p${NUM_PAGES}_b${ACTIVE_BATCH_SIZE}_${DECODE_BACKEND}.json}"
LAYOUT_CACHE_JSON="${LAYOUT_CACHE_JSON:-${OUTPUT_DIR}/layout_cache_first${NUM_PAGES}_${RUN_ID}.json}"
CHILD_OUTPUT_DIR="${CHILD_OUTPUT_DIR:-${OUTPUT_DIR}/chunks_${RUN_ID}_p${NUM_PAGES}_b${ACTIVE_BATCH_SIZE}_${DECODE_BACKEND}}"
REUSE_LAYOUT_CACHE="${REUSE_LAYOUT_CACHE:-0}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-1}"
CHECK_PADDLE_IMPORT="${CHECK_PADDLE_IMPORT:-1}"

export PADDLE_OCR_VL_VISION_ATTENTION="${VISION_ATTENTION_IMPL}"
export PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT}"

if (( PAGE_CHUNK_SIZE > 0 && CROP_CHUNK_SIZE > 0 )); then
  echo "PAGE_CHUNK_SIZE and CROP_CHUNK_SIZE are mutually exclusive. Prefer CROP_CHUNK_SIZE for NPU VRAM control." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${DOWNLOAD_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_omnidocbench_pages.py" \
    --out-dir "${DATASET_DIR}" \
    --page-start "${PAGE_START}" \
    --num-pages "${NUM_PAGES}"
fi

if [[ "${LAYOUT_SOURCE}" == "omnidocbench_gt" ]]; then
  if [[ -z "${EXPECT_GT_CROP_MANIFEST}" && "${STRICT_KNOWN_FIRST64_GT_MANIFEST}" == "1" && "${PAGE_START}" == "0" && "${NUM_PAGES}" == "64" && "${INCLUDE_IGNORED_GT}" == "0" && "${INCLUDE_EMPTY_GT}" == "0" ]]; then
    EXPECT_GT_CROP_MANIFEST="${SCRIPT_DIR}/expected_omnidocbench_first64_gt_crops.json"
  fi
  GT_AUDIT_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/count_omnidocbench_gt_crops.py"
    --dataset-dir "${DATASET_DIR}"
    --page-start "${PAGE_START}"
    --num-pages "${NUM_PAGES}"
    --json
  )
  if [[ "${INCLUDE_IGNORED_GT}" == "1" ]]; then
    GT_AUDIT_CMD+=(--include-ignored-gt)
  fi
  if [[ "${INCLUDE_EMPTY_GT}" == "1" ]]; then
    GT_AUDIT_CMD+=(--include-empty-gt)
  fi
  if [[ -n "${EXPECTED_RECOGNIZER_CROPS}" ]]; then
    GT_AUDIT_CMD+=(--expect-count "${EXPECTED_RECOGNIZER_CROPS}")
  fi
  if [[ -n "${EXPECT_GT_CROP_MANIFEST}" ]]; then
    GT_AUDIT_CMD+=(--expect-manifest "${EXPECT_GT_CROP_MANIFEST}")
  fi
  GT_CROP_AUDIT="$("${GT_AUDIT_CMD[@]}")"
  echo "GT_CROP_AUDIT ${GT_CROP_AUDIT}"
  if [[ -z "${EXPECTED_RECOGNIZER_CROPS}" ]]; then
    EXPECTED_RECOGNIZER_CROPS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.loads(sys.argv[1])["recognizer_crop_count"]))' "${GT_CROP_AUDIT}")"
  fi
fi

if [[ "${CHECK_PADDLE_IMPORT}" == "1" && "${LAYOUT_SOURCE}" == "official" && "${REUSE_LAYOUT_CACHE}" != "1" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY'
import paddle
print("PADDLE_IMPORT_OK", paddle.__version__)
PY
  then
    cat >&2 <<'MSG'
PADDLE_IMPORT_FAILED
The official PaddleOCR/PaddleX layout detector cannot run in this environment.
On the current RTX 3060 Vast box this is expected with the CPU PaddlePaddle 3.x
wheel on a non-AVX host CPU: `import paddle` exits with SIGILL before layout
inference starts. Use an AVX-capable/Docker-capable GPU instance, or run with
REUSE_LAYOUT_CACHE=1 against a layout JSON produced elsewhere.
MSG
    exit 90
  fi
fi

if (( PAGE_CHUNK_SIZE > 0 )); then
  CHUNK_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_chunked_page_pipeline_e2e.py"
    --python-bin "${PYTHON_BIN}"
    --model "${MODEL}"
    --dataset-dir "${DATASET_DIR}"
    --page-start "${PAGE_START}"
    --num-pages "${NUM_PAGES}"
    --page-chunk-size "${PAGE_CHUNK_SIZE}"
    --child-output-dir "${CHILD_OUTPUT_DIR}"
    --output-json "${OUTPUT_PATH}"
    --layout-source "${LAYOUT_SOURCE}"
    --layout-device "${LAYOUT_DEVICE}"
    --device "${DEVICE}"
    --dtype fp16
    --decode-backend "${DECODE_BACKEND}"
    --active-batch-size "${ACTIVE_BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --cache-length "${CACHE_LENGTH}"
    --preprocessor-min-pixels "${PREPROCESSOR_MIN_PIXELS}"
    --preprocessor-max-pixels "${PREPROCESSOR_MAX_PIXELS}"
    --npu-jit-compile "${NPU_JIT_COMPILE}"
    --torchair-cache-dir "${TORCHAIR_CACHE_DIR}"
    --validation-items "${VALIDATION_ITEMS}"
  )
  if [[ "${REUSE_LAYOUT_CACHE}" == "1" ]]; then
    CHUNK_CMD+=(--reuse-layout-cache)
  fi
  if [[ "${INCLUDE_IGNORED_GT}" == "1" ]]; then
    CHUNK_CMD+=(--include-ignored-gt)
  fi
  if [[ "${INCLUDE_EMPTY_GT}" == "1" ]]; then
    CHUNK_CMD+=(--include-empty-gt)
  fi
  if [[ -n "${EXPECT_LAYOUT_SOURCE}" ]]; then
    CHUNK_CMD+=(--expect-layout-source "${EXPECT_LAYOUT_SOURCE}")
  fi
  if [[ -n "${EXPECTED_RECOGNIZER_CROPS}" ]]; then
    CHUNK_CMD+=(--expected-recognizer-crops "${EXPECTED_RECOGNIZER_CROPS}")
  fi
  if [[ -n "${MIN_RECOGNIZER_CROPS}" ]]; then
    CHUNK_CMD+=(--min-recognizer-crops "${MIN_RECOGNIZER_CROPS}")
  fi
  if [[ "${FAIL_ON_MISMATCH}" == "1" ]]; then
    CHUNK_CMD+=(--fail-on-mismatch)
  fi
  if [[ "${FAIL_ON_LENGTH_CAP}" == "1" ]]; then
    CHUNK_CMD+=(--fail-on-length-cap)
  fi

  echo "CHUNKED_COMMAND ${CHUNK_CMD[*]}"
  "${CHUNK_CMD[@]}"
  "${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
  echo "WROTE ${OUTPUT_PATH}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_page_pipeline_e2e.py"
  --model "${MODEL}"
  --dataset-dir "${DATASET_DIR}"
  --page-start "${PAGE_START}"
  --num-pages "${NUM_PAGES}"
  --layout-source "${LAYOUT_SOURCE}"
  --layout-device "${LAYOUT_DEVICE}"
  --layout-cache-json "${LAYOUT_CACHE_JSON}"
  --device "${DEVICE}"
  --dtype fp16
  --decode-backend "${DECODE_BACKEND}"
  --active-batch-size "${ACTIVE_BATCH_SIZE}"
  --crop-chunk-size "${CROP_CHUNK_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --cache-length "${CACHE_LENGTH}"
  --preprocessor-min-pixels "${PREPROCESSOR_MIN_PIXELS}"
  --preprocessor-max-pixels "${PREPROCESSOR_MAX_PIXELS}"
  --npu-jit-compile "${NPU_JIT_COMPILE}"
  --torchair-cache-dir "${TORCHAIR_CACHE_DIR}"
  --validation-items "${VALIDATION_ITEMS}"
  --json
)

if [[ "${REUSE_LAYOUT_CACHE}" == "1" ]]; then
  CMD+=(--reuse-layout-cache)
fi
if [[ "${INCLUDE_IGNORED_GT}" == "1" ]]; then
  CMD+=(--include-ignored-gt)
fi
if [[ "${INCLUDE_EMPTY_GT}" == "1" ]]; then
  CMD+=(--include-empty-gt)
fi
if [[ -n "${EXPECT_LAYOUT_SOURCE}" ]]; then
  CMD+=(--expect-layout-source "${EXPECT_LAYOUT_SOURCE}")
fi
if [[ -n "${EXPECTED_RECOGNIZER_CROPS}" ]]; then
  CMD+=(--expected-recognizer-crops "${EXPECTED_RECOGNIZER_CROPS}")
fi
if [[ -n "${MIN_RECOGNIZER_CROPS}" ]]; then
  CMD+=(--min-recognizer-crops "${MIN_RECOGNIZER_CROPS}")
fi

echo "COMMAND ${CMD[*]}"
"${CMD[@]}" | tee "${OUTPUT_PATH}"
"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8")
start = raw.find("{")
if start < 0:
    raise SystemExit(f"no JSON object found in {path}")
data = json.loads(raw[start:])
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
"${PYTHON_BIN}" -m json.tool "${OUTPUT_PATH}" >/dev/null
"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("error"):
    print("PAGE_PIPELINE_ERROR", json.dumps(data, sort_keys=True)[:4000])
    raise SystemExit(2)

correctness = data.get("correctness", {})
phase = data.get("phase_timing_s", {})
throughput = data.get("throughput", {})
crop_summary = data.get("crop_summary", {})
decode = data.get("decode_summary", {})
ready_timing = data.get("ready_item_timing_summary_s", {})
rough_accuracy = data.get("rough_ground_truth_accuracy", {})
omnidoc_metrics = data.get("omnidocbench_metrics_without_cdm", {})
print("PAGE_PIPELINE_SUMMARY", json.dumps({
    "page_count": data.get("page_count"),
    "page_load": data.get("page_load"),
    "recognizer_crop_count": data.get("recognizer_crop_count"),
    "crop_count_contract": data.get("crop_count_contract"),
    "uses_ground_truth_layout_boxes": data.get("uses_ground_truth_layout_boxes"),
    "doc_layout_model_measured": data.get("doc_layout_model_measured"),
    "layout_source": data.get("layout", {}).get("source"),
    "layout_device": data.get("layout", {}).get("device"),
    "recognizer_device": data.get("device"),
    "decode_backend": data.get("decode_backend"),
    "decode_attention": data.get("decode_attention"),
    "decode_cache_update": data.get("decode_cache_update"),
    "vision_attention": data.get("vision_attention"),
    "vision_prompt_fa_layout": data.get("vision_prompt_fa_layout"),
    "npu_jit_compile": data.get("npu_jit_compile"),
    "active_batch_size": data.get("active_batch_size"),
    "prefill_batch_size": data.get("prefill_batch_size"),
    "crop_chunk_size": data.get("crop_chunk_size"),
    "crop_chunking": data.get("crop_chunking"),
    "pages_per_s_measured_e2e": throughput.get("pages_per_s_measured_e2e"),
    "cache_length": data.get("cache_length"),
    "max_new_tokens": data.get("max_new_tokens"),
    "preprocessor": data.get("preprocessor"),
}, sort_keys=True))
print("SETUP_TIMING_S", json.dumps(data.get("setup_timing_s", {}), sort_keys=True))
print("PHASE_TIMING_S", json.dumps(phase, sort_keys=True))
print("THROUGHPUT", json.dumps(throughput, sort_keys=True))
print("DECODE_WARMUP", json.dumps(data.get("decode_warmup", {}), sort_keys=True))
print("OUTPUT_FINGERPRINT_SUMMARY", json.dumps(data.get("output_fingerprint_summary", {}), sort_keys=True))
print("PREFILL_STAGE_TIMING_S", json.dumps({
    "native_resolution_visual_encoder_total": ready_timing.get("native_resolution_visual_encoder_total"),
    "vision_encoder": ready_timing.get("vision_encoder"),
    "adaptive_mlp_projector": ready_timing.get("adaptive_mlp_projector"),
    "vision_total": ready_timing.get("vision_total"),
    "vision_projector_total": ready_timing.get("vision_projector_total"),
    "text_prefill": ready_timing.get("text_prefill"),
    "prefill_lm_head": ready_timing.get("prefill_lm_head"),
    "prefill_argmax": ready_timing.get("prefill_argmax"),
    "ready_item_total_excluding_device_transfer": ready_timing.get("ready_item_total_excluding_device_transfer"),
    "ready_item_total_with_device_transfer": ready_timing.get("ready_item_total_with_device_transfer"),
}, sort_keys=True))
print("CROP_SUMMARY", json.dumps({
    "layout_box_count": crop_summary.get("layout_box_count"),
    "recognizer_crop_count": crop_summary.get("recognizer_crop_count"),
    "skipped_count": crop_summary.get("skipped_count"),
    "label_counts": crop_summary.get("label_counts"),
    "prompt_counts": crop_summary.get("prompt_counts"),
    "per_page_counts": crop_summary.get("per_page_counts"),
}, sort_keys=True))
print("DECODE_SUMMARY", json.dumps({
    "decode_calls": decode.get("decode_calls"),
    "raw_decode_token_calls": decode.get("raw_decode_token_calls"),
    "effective_decode_token_calls": decode.get("effective_decode_token_calls"),
    "eos_hit_count": decode.get("eos_hit_count"),
    "length_cap_hit_count": decode.get("length_cap_hit_count"),
    "swap_event_count": decode.get("swap_event_count"),
    "total_swapped_in_items": decode.get("total_swapped_in_items"),
}, sort_keys=True))
print("CORRECTNESS", json.dumps(correctness, sort_keys=True))
print("OMNIDOCBENCH_METRICS_WITHOUT_CDM", json.dumps({
    "is_official_omnidocbench_metric": omnidoc_metrics.get("is_official_omnidocbench_metric"),
    "scope": omnidoc_metrics.get("scope"),
    "matched_scored_items": omnidoc_metrics.get("matched_scored_items"),
    "leaderboard_overall": omnidoc_metrics.get("leaderboard_overall"),
    "leaderboard_overall_unavailable_reason": omnidoc_metrics.get("leaderboard_overall_unavailable_reason"),
    "available_non_cdm_component_mean_score_percent": omnidoc_metrics.get("available_non_cdm_component_mean_score_percent"),
    "available_non_cdm_component_mean_note": omnidoc_metrics.get("available_non_cdm_component_mean_note"),
    "text_table_conclusion_mean_score_percent": omnidoc_metrics.get("text_table_conclusion_mean_score_percent"),
    "text_table_conclusion_components": omnidoc_metrics.get("text_table_conclusion_components"),
    "text_block_Edit_dist": omnidoc_metrics.get("text_block_Edit_dist"),
    "table_Edit_dist": omnidoc_metrics.get("table_Edit_dist"),
    "table_TEDS": omnidoc_metrics.get("table_TEDS"),
    "table_TEDS_structure_only": omnidoc_metrics.get("table_TEDS_structure_only"),
    "reading_order_Edit_dist": omnidoc_metrics.get("reading_order_Edit_dist"),
    "reported_paddleocr_vl_1_6_reference": omnidoc_metrics.get("reported_paddleocr_vl_1_6_reference"),
}, sort_keys=True))
print("FORMULA_DIAGNOSTICS_NOT_FOR_CONCLUSION", json.dumps({
    "display_formula_Edit_dist": omnidoc_metrics.get("display_formula_Edit_dist"),
    "display_formula_BLEU_1_4": omnidoc_metrics.get("display_formula_BLEU_1_4"),
    "display_formula_CDM": omnidoc_metrics.get("display_formula_CDM"),
}, sort_keys=True))
print("ROUGH_GT_ACCURACY", json.dumps({
    "enabled": rough_accuracy.get("enabled"),
    "is_official_omnidocbench_metric": rough_accuracy.get("is_official_omnidocbench_metric"),
    "scope": rough_accuracy.get("scope"),
    "matched_text_items": rough_accuracy.get("matched_text_items"),
    "normalized_exact_count": rough_accuracy.get("normalized_exact_count"),
    "normalized_exact_rate": rough_accuracy.get("normalized_exact_rate"),
    "avg_sequence_ratio": rough_accuracy.get("avg_sequence_ratio"),
    "by_layout_label": rough_accuracy.get("by_layout_label"),
}, sort_keys=True))
print("TEXT_SAMPLE", repr(data.get("texts", {}).get("sample", [])))
print("OUTPUT_JSON", path)

expected_gt = data.get("layout", {}).get("source") == "omnidocbench_gt"
if bool(data.get("uses_ground_truth_layout_boxes")) != bool(expected_gt):
    raise SystemExit(f"unexpected uses_ground_truth_layout_boxes={data.get('uses_ground_truth_layout_boxes')} expected {expected_gt}")
if not data.get("crop_count_contract", {}).get("passed", True):
    raise SystemExit("crop count/layout source contract failed")
invalid_token_count = int(correctness.get("invalid_token_count", 0) or 0)
length_cap_hit_count = int(correctness.get("length_cap_hit_count", 0) or 0)
validation_enabled = bool(correctness.get("enabled", False))
fail_on_mismatch = os.environ.get("FAIL_ON_MISMATCH", "1") == "1"
fail_on_length_cap = os.environ.get("FAIL_ON_LENGTH_CAP", "0") == "1"
if invalid_token_count:
    raise SystemExit(f"invalid token IDs generated: {invalid_token_count}")
if fail_on_length_cap and length_cap_hit_count:
    raise SystemExit(f"length cap hit: {length_cap_hit_count}")
if validation_enabled and fail_on_mismatch and not correctness.get("all_required_checks_passed", False):
    raise SystemExit("validation/correctness failed")
PY
echo "WROTE ${OUTPUT_PATH}"
