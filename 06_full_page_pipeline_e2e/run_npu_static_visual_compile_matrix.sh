#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Correctness-first discriminator for the static_visual compile boundary.
# It runs the same real small OmniDocBench crop through:
#   1. PromptFA eager static_visual
#   2. PromptFA TorchAir static_visual
#   3. manual-attention eager static_visual
#   4. manual-attention TorchAir static_visual
#
# Use this before profiling more compiled vision kernels. If PromptFA compiled
# diverges but manual compiled does not, the issue is PromptFA under TorchAir. If
# both compiled cases diverge, the issue is broader visual-encoder TorchAir
# lowering/fusion numerics.

export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/static_visual_compile_matrix_npu}"
export DEVICE="${DEVICE:-npu:0}"
export PAGE_START="${PAGE_START:-0}"
export NUM_PAGES="${NUM_PAGES:-8}"
export MAX_CROPS="${MAX_CROPS:-0}"
export WARMUP_ITEMS="${WARMUP_ITEMS:-1}"
export DTYPE="${DTYPE:-fp16}"
export NPU_JIT_COMPILE="${NPU_JIT_COMPILE:-off}"
export VISION_PROMPT_FA_LAYOUT="${VISION_PROMPT_FA_LAYOUT:-bnsd}"
export VISION_COMPILE_VALIDATE="${VISION_COMPILE_VALIDATE:-1}"
export VISION_FORWARD_BOUNDARY="${VISION_FORWARD_BOUNDARY:-static_visual}"
export MODES="${MODES:-unsynced_loop}"
export CROP_SAMPLE="${CROP_SAMPLE:-small_only}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-5}"
export PROFILE_DIR=""
export INCLUDE_IGNORED_GT="${INCLUDE_IGNORED_GT:-0}"
export INCLUDE_EMPTY_GT="${INCLUDE_EMPTY_GT:-0}"

if [[ "${VISION_FORWARD_BOUNDARY}" != "static_visual" ]]; then
  echo "ERROR: this matrix is specifically for VISION_FORWARD_BOUNDARY=static_visual" >&2
  exit 2
fi
if [[ "${CROP_SAMPLE}" != "small_only" ]]; then
  echo "ERROR: this matrix is shape-specialized and expects CROP_SAMPLE=small_only" >&2
  exit 2
fi
if [[ "${VISION_COMPILE_VALIDATE}" != "1" ]]; then
  echo "ERROR: this matrix must run with VISION_COMPILE_VALIDATE=1" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
RUN_PREFIX="${RUN_PREFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"

run_case() {
  local case_name="$1"
  local attention_impl="$2"
  local compile_backend="$3"
  local output_path="${OUTPUT_DIR}/${RUN_PREFIX}_${case_name}.json"

  echo "STATIC_VISUAL_MATRIX_RUN case=${case_name} attention=${attention_impl} backend=${compile_backend} output=${output_path}"
  RUN_ID="${RUN_PREFIX}_${case_name}" \
  OUTPUT_PATH="${output_path}" \
  VISION_ATTENTION_IMPL="${attention_impl}" \
  VISION_COMPILE_BACKEND="${compile_backend}" \
  "${SCRIPT_DIR}/run_npu_vision_prefill_only.sh"
}

run_case "promptfa_eager" "prompt_flash_attention" "none"
run_case "promptfa_torchair" "prompt_flash_attention" "torchair"
run_case "manual_eager" "manual" "none"
run_case "manual_torchair" "manual" "torchair"

"${PYTHON_BIN:-python3}" - "${OUTPUT_DIR}" "${RUN_PREFIX}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
run_prefix = sys.argv[2]
cases = [
    "promptfa_eager",
    "promptfa_torchair",
    "manual_eager",
    "manual_torchair",
]

rows = []
for case in cases:
    path = output_dir / f"{run_prefix}_{case}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = (data.get("modes") or {}).get("unsynced_loop") or {}
    compile_info = data.get("vision_compile") or {}
    validation = compile_info.get("validation") or {}
    static_vs_original = validation.get("static_visual_vs_original_visual") or {}
    rows.append(
        {
            "case": case,
            "attention": data.get("vision_attention"),
            "backend": compile_info.get("backend"),
            "enabled": compile_info.get("enabled"),
            "compile_first_call_s": compile_info.get("compiled_first_call_s"),
            "allclose": validation.get("allclose_atol_5e_2_rtol_5e_2"),
            "max_abs_diff": validation.get("max_abs_diff"),
            "mean_abs_diff": validation.get("mean_abs_diff"),
            "static_vs_original_allclose": static_vs_original.get("allclose_atol_5e_2_rtol_5e_2"),
            "static_vs_original_max_abs_diff": static_vs_original.get("max_abs_diff"),
            "items_per_s": mode.get("items_per_s"),
            "vision_tokens_per_s": mode.get("vision_tokens_per_s"),
            "output_path": str(path),
        }
    )

print("STATIC_VISUAL_COMPILE_MATRIX_SUMMARY", json.dumps(rows, ensure_ascii=False, sort_keys=True))
PY
