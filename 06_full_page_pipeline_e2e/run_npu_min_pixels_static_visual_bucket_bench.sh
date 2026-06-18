#!/usr/bin/env bash
set -euo pipefail

# Microbenchmark compiled static_visual PromptFA speed across min_pixels settings.
#
# Contract:
# - Select one real OmniDocBench GT crop per natural projected-token bucket.
# - For each selected crop, run the same crop id with each min_pixels value.
# - Each measurement compiles exactly that crop shape, runs one post-compile warmup,
#   then times BENCHMARK_REPEATS forwards with one sync before and one sync after.
# - No profiler, no text prefill, no decode, no OCR validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python" ]]; then
    export PYTHON_BIN="/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

if [[ -z "${MODEL:-}" ]]; then
  for candidate in \
    "/home/lukaiv/models/paddle_ocr_0_9b_v_1_6" \
    "/workspace/.hf_home/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e"
  do
    if [[ -f "${candidate}/config.json" && -f "${candidate}/tokenizer.json" ]]; then
      export MODEL="${candidate}"
      break
    fi
  done
fi
export MODEL="${MODEL:-/home/lukaiv/models/paddle_ocr_0_9b_v_1_6}"

if [[ -z "${DATASET_DIR:-}" ]]; then
  for candidate in \
    "/home/lukaiv/datasets/OmniDocBench_current" \
    "/home/lukaiv/data/OmniDocBench_current" \
    "/home/lukaiv/data/OmniDocBench" \
    "/home/lukaiv/datasets/OmniDocBench" \
    "/workspace/data/OmniDocBench"
  do
    if [[ -f "${candidate}/OmniDocBench.json" ]]; then
      export DATASET_DIR="${candidate}"
      break
    fi
  done
fi
export DATASET_DIR="${DATASET_DIR:-/home/lukaiv/datasets/OmniDocBench_current}"

export DEVICE="${DEVICE:-npu:0}"
export PAGE_START="${PAGE_START:-0}"
export NUM_PAGES="${NUM_PAGES:-64}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export MIN_PIXELS_LIST="${MIN_PIXELS_LIST:-112896 50176 28224}"
export BUCKET_SPECS="${BUCKET_SPECS:-0-15 32-63 64-95 96-127 128-143 144-191 256-511}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-100}"
export WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
export VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL:-prompt_flash_attention}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND:-torchair}"
export VISION_FORWARD_BOUNDARY="${VISION_FORWARD_BOUNDARY:-static_visual}"
export STATIC_VISUAL_PAD_MODE="${STATIC_VISUAL_PAD_MODE:-mask_pad_one}"
export VISION_COMPILE_VALIDATE="${VISION_COMPILE_VALIDATE:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/min_pixels_static_visual_bucket_bench_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${OUTPUT_DIR}"

SELECTED_JSON="${OUTPUT_DIR}/selected_buckets.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV PYTHON_BIN=${PYTHON_BIN}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV MODEL=${MODEL}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV DATASET_DIR=${DATASET_DIR}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV DEVICE=${DEVICE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV PAGE_START=${PAGE_START} NUM_PAGES=${NUM_PAGES}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV MIN_PIXELS_LIST=${MIN_PIXELS_LIST}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV BUCKET_SPECS=${BUCKET_SPECS}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV BENCHMARK_REPEATS=${BENCHMARK_REPEATS} WARMUP_ITEMS=${WARMUP_ITEMS}"
echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_ENV VISION_ATTENTION_IMPL=${VISION_ATTENTION_IMPL} VISION_COMPILE_BACKEND=${VISION_COMPILE_BACKEND} VISION_FORWARD_BOUNDARY=${VISION_FORWARD_BOUNDARY} STATIC_VISUAL_PAD_MODE=${STATIC_VISUAL_PAD_MODE}"

"${PYTHON_BIN}" - "${SCRIPT_DIR}" "${MODEL}" "${DATASET_DIR}" "${PAGE_START}" "${NUM_PAGES}" "${BUCKET_SPECS}" "${SELECTED_JSON}" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

script_dir = Path(sys.argv[1]).resolve()
model_arg = sys.argv[2]
dataset_dir = Path(sys.argv[3])
page_start = int(sys.argv[4])
num_pages = int(sys.argv[5])
bucket_specs_raw = sys.argv[6]
selected_json = Path(sys.argv[7])

repo_root = script_dir.parent
exp5_dir = repo_root / "05_full_recognizer_optimizations"
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(exp5_dir))

from bench_page_pipeline_e2e import (  # noqa: E402
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    load_pages_result,
    resolve_dataset_dir,
)
from local_modeling_paddleocr_vl import _resolve_model_dir  # noqa: E402
from run_local_recognition import load_preprocessor_config, smart_resize  # noqa: E402


def parse_bucket(spec: str) -> tuple[str, int, int | None]:
    spec = spec.strip()
    if spec.startswith(">="):
        return spec, int(spec[2:]), None
    lo_s, hi_s = spec.split("-", 1)
    return spec, int(lo_s), int(hi_s)


model_dir = _resolve_model_dir(model_arg)
pre_cfg = load_preprocessor_config(model_dir)
factor = int(pre_cfg["patch_size"]) * int(pre_cfg["merge_size"])
max_pixels = int(pre_cfg["max_pixels"])

page_load = load_pages_result(resolve_dataset_dir(dataset_dir), page_start=page_start, num_pages=num_pages)
layout_pages, _ = build_omnidocbench_gt_layout_pages(page_load.pages, include_ignored=False, include_empty_gt=False)
args = SimpleNamespace(
    layout_source="omnidocbench_gt",
    crop_padding=0,
    min_crop_side=4,
    skip_labels="",
)
crops, crop_summary, _ = build_detected_crops(pages=page_load.pages, layout_pages=layout_pages, args=args)

rows = []
for index, crop in enumerate(crops):
    width, height = [int(value) for value in crop.entry.get("crop_size", [0, 0])]
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=0,
        max_pixels=max_pixels,
    )
    natural_projected = int(resized_h // factor) * int(resized_w // factor)
    rows.append(
        {
            "id": str(crop.entry.get("id")),
            "source_index": int(index),
            "page_index": int(crop.entry.get("page_index", 0)),
            "layout_label": str(crop.entry.get("layout_label", "")),
            "crop_size": [int(width), int(height)],
            "raw_area": int(width * height),
            "natural_resized_hw": [int(resized_h), int(resized_w)],
            "natural_projected_image_tokens": int(natural_projected),
        }
    )

selected = []
for spec in bucket_specs_raw.replace(",", " ").split():
    name, lo, hi = parse_bucket(spec)
    candidates = [
        row for row in rows
        if int(row["natural_projected_image_tokens"]) >= lo
        and (hi is None or int(row["natural_projected_image_tokens"]) <= hi)
    ]
    if not candidates:
        selected.append({"bucket": name, "missing": True})
        continue
    candidates = sorted(candidates, key=lambda row: (int(row["natural_projected_image_tokens"]), str(row["id"])))
    chosen = candidates[(len(candidates) - 1) // 2]
    chosen = dict(chosen)
    chosen["bucket"] = name
    chosen["bucket_candidate_count"] = int(len(candidates))
    chosen["missing"] = False
    selected.append(chosen)

payload = {
    "selection_scope": {
        "dataset_dir": str(resolve_dataset_dir(dataset_dir)),
        "page_start": int(page_start),
        "num_pages": int(num_pages),
        "crop_count": int(len(rows)),
        "crop_summary": crop_summary,
    },
    "natural_resize_contract": {
        "factor": int(factor),
        "min_pixels": 0,
        "max_pixels": int(max_pixels),
        "official_min_pixels": int(pre_cfg["min_pixels"]),
        "official_min_projected_image_tokens": int(pre_cfg["min_pixels"]) // int(factor * factor),
    },
    "bucket_specs": bucket_specs_raw.replace(",", " ").split(),
    "selected": selected,
}
selected_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print("SELECTED_BUCKETS", json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY

mapfile -t SELECTED_LINES < <("${PYTHON_BIN}" - "${SELECTED_JSON}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in data["selected"]:
    if row.get("missing"):
        print(f"SKIP\t{row['bucket']}\t")
    else:
        print(f"RUN\t{row['bucket']}\t{row['id']}")
PY
)

for line in "${SELECTED_LINES[@]}"; do
  IFS=$'\t' read -r action bucket crop_id <<<"${line}"
  if [[ "${action}" == "SKIP" ]]; then
    echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_SKIP bucket=${bucket} reason=no_crop_in_bucket"
    continue
  fi
  safe_bucket="$(printf '%s' "${bucket}" | tr -c 'A-Za-z0-9_' '_')"
  for min_pixels in ${MIN_PIXELS_LIST}; do
    output_path="${OUTPUT_DIR}/bucket_${safe_bucket}_min${min_pixels}.json"
    echo "MIN_PIXELS_STATIC_VISUAL_BUCKET_RUN bucket=${bucket} crop_id=${crop_id} min_pixels=${min_pixels} output=${output_path}"
    PAGE_START="${PAGE_START}" \
    NUM_PAGES="${NUM_PAGES}" \
    MAX_CROPS=0 \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    NPU_JIT_COMPILE="${NPU_JIT_COMPILE}" \
    VISION_ATTENTION_IMPL="${VISION_ATTENTION_IMPL}" \
    VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT}" \
    VISION_COMPILE_BACKEND="${VISION_COMPILE_BACKEND}" \
    VISION_FORWARD_BOUNDARY="${VISION_FORWARD_BOUNDARY}" \
    STATIC_VISUAL_PAD_MODE="${STATIC_VISUAL_PAD_MODE}" \
    VISION_COMPILE_VALIDATE="${VISION_COMPILE_VALIDATE}" \
    MODES="unsynced_loop" \
    CROP_SAMPLE="all" \
    CROP_IDS="${crop_id}" \
    PREPROCESSOR_MIN_PIXELS="${min_pixels}" \
    PREPROCESSOR_MAX_PIXELS="-1" \
    BENCHMARK_REPEATS="${BENCHMARK_REPEATS}" \
    WARMUP_ITEMS="${WARMUP_ITEMS}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    OUTPUT_PATH="${output_path}" \
      bash "${SCRIPT_DIR}/run_npu_vision_prefill_only.sh"
  done
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SELECTED_JSON}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
selected_json = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
selected = json.loads(selected_json.read_text(encoding="utf-8"))
bucket_by_id = {
    str(row.get("id")): str(row.get("bucket"))
    for row in selected.get("selected", [])
    if not row.get("missing")
}

rows = []
for path in sorted(output_dir.glob("bucket_*_min*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = data.get("modes", {}).get("unsynced_loop", {})
    samples = mode.get("samples") or []
    sample = samples[0] if samples else {}
    compile_info = data.get("vision_compile", {}) or {}
    pre = data.get("preprocessor", {}) or {}
    forward_count = int(mode.get("count") or 0)
    total_s = float(mode.get("total_s") or 0.0)
    crop_id = sample.get("id") or compile_info.get("crop_id")
    row = {
        "path": str(path),
        "bucket": sample.get("crop_sample_bucket") or bucket_by_id.get(str(crop_id)),
        "crop_id": crop_id,
        "layout_label": sample.get("layout_label"),
        "crop_size": ((data.get("crop_id_filter") or {}).get("selected") or [{}])[0].get("crop_size"),
        "min_pixels": pre.get("min_pixels"),
        "min_projected_image_tokens": pre.get("min_projected_image_tokens"),
        "image_grid_thw": sample.get("image_grid_thw") or compile_info.get("image_grid_thw"),
        "vision_tokens": sample.get("vision_tokens") or compile_info.get("vision_tokens"),
        "projected_image_tokens": sample.get("projected_image_tokens"),
        "static_visual_physical_seq_len": compile_info.get("static_visual_physical_seq_len"),
        "static_visual_pad_tokens": compile_info.get("static_visual_pad_tokens"),
        "compile_first_call_s": compile_info.get("compiled_first_call_s"),
        "compile_wrapper_s": compile_info.get("compile_wrapper_s"),
        "warmup_s": (data.get("warmup") or {}).get("elapsed_s"),
        "forward_count": forward_count,
        "total_s": total_s,
        "steady_ms_per_forward": (1000.0 * total_s / forward_count) if forward_count else None,
        "items_per_s": mode.get("items_per_s"),
        "vision_tokens_per_s": mode.get("vision_tokens_per_s"),
        "projected_image_tokens_per_s": mode.get("projected_image_tokens_per_s"),
    }
    rows.append(row)

payload = {
    "selected_buckets": selected,
    "summary_contract": {
        "timed_region": "unsynced_loop only: one device sync before repeated forwards and one after; no profiler",
        "compile_first_call_s": "reported separately and not included in total_s",
        "warmup_s": "one post-compile warmup forward by default and not included in total_s",
    },
    "rows": rows,
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print("MIN_PIXELS_STATIC_VISUAL_BUCKET_SUMMARY", json.dumps(payload, ensure_ascii=False, sort_keys=True))
print(f"MIN_PIXELS_STATIC_VISUAL_BUCKET_OUTPUT_DIR={output_dir}")
print(f"MIN_PIXELS_STATIC_VISUAL_BUCKET_SUMMARY_JSON={summary_json}")
PY
