#!/usr/bin/env bash

# The CANN/ATB environment scripts are not safe under nounset/errexit.
source npu-setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO"

case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2; exit 1 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
MODEL="${MODEL:-/workspace/models/unirec-0.1b}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV2_safetensors}"
OPENOCR_ROOT="${OPENOCR_ROOT:-/workspace/repos/OpenOCR}"
INPUT="${INPUT:-$REPO/tmp/12_unirec_0_1b_inference/910b_rep128_k10_l1_7cd0f82_20260817T150836/representative_128_v1_images}"
COMPILE_CACHE="${COMPILE_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf}"
LAYOUT_CACHE="${LAYOUT_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/representative128_layout_compiled_fp32_optimized_b2_6deceef}"
DECODE_CACHE="${DECODE_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/dual_lane_decode_173eb50}"
CPUSET="${CPUSET:-0-63}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$INPUT"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE"
test -d "$DECODE_CACHE/decode_selfkv1408_cross384_increfa_all_b128"
test -d "$DECODE_CACHE/decode_selfkv2048_cross1320_increfa_all_b128"

STAMP="$(date +%Y%m%dT%H%M%S)"
COMMIT="$(git rev-parse --short HEAD)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/910b_dual_lane_rep128_${COMMIT}_${STAMP}"
OUTPUT="$RUN_ROOT/output"
RUN_LOG="$RUN_ROOT/run.log"
mkdir -p "$OUTPUT"

inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'vision %p %s %T@\n'
    find "$LAYOUT_CACHE" -type f -name '*.om' -printf 'layout %p %s %T@\n'
    find "$DECODE_CACHE" -type f -name '*.om' -printf 'decode %p %s %T@\n'
  } | sort >"$output"
}

command=(
  taskset -c "$CPUSET"
  env
  PYTHONUNBUFFERED=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$DECODE_CACHE"
  "$PYTHON_BIN" "$SCRIPT_DIR/run_two_phase_batched_unirec.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --layout-execution torchair
  --layout-dtype float32
  --layout-reading-order-dtype float32
  --layout-weight-format native
  --layout-depthwise-rewrite native
  --layout-threshold 0.5
  --layout-cache-dir "$LAYOUT_CACHE"
  --input "$INPUT"
  --output-dir "$OUTPUT"
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
  --decode-lane-mode dual
  --decode-a-batch-size 128
  --decode-a-cross-cache-length 384
  --decode-a-self-cache-length 1408
  --decode-a-max-length 1408
  --decode-b-batch-size 128
  --decode-quantum-steps 16
  --decode-max-skipped-quanta 8
  --decode-a-overflow-policy finish_at_cap
  --compile-cache-dir "$COMPILE_CACHE"
  --decode-warmup-passes 2
  --decode-admission-prefetch-depth 0
  --progress-every-pages 8
  --progress-heartbeat-s 15
)

inventory "$RUN_ROOT/om_before.txt"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'started=$(date +%%s)\n'
  printf 'set +e\n'
  printf '%q ' "${command[@]}"
  printf '\n'
  printf 'status=$?\n'
  printf 'ended=$(date +%%s)\n'
  printf 'printf "%%s\\n" "$((ended - started))" >%q\n' \
    "$RUN_ROOT/process_wall_s.txt"
  printf '{\n'
  printf '  find %q -type f -name "*.om" -printf "vision %%p %%s %%T@\\n"\n' \
    "$COMPILE_CACHE"
  printf '  find %q -type f -name "*.om" -printf "layout %%p %%s %%T@\\n"\n' \
    "$LAYOUT_CACHE"
  printf '  find %q -type f -name "*.om" -printf "decode %%p %%s %%T@\\n"\n' \
    "$DECODE_CACHE"
  printf '} | sort >%q\n' "$RUN_ROOT/om_after.txt"
  printf 'diff -u %q %q >%q\n' \
    "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" "$RUN_ROOT/om.diff"
  printf 'printf "%%s\\n" "$status" >%q\n' "$RUN_ROOT/exit_code.txt"
  printf 'exit "$status"\n'
} >"$RUN_ROOT/command.sh"
chmod +x "$RUN_ROOT/command.sh"

nohup "$RUN_ROOT/command.sh" >"$RUN_LOG" 2>&1 &
PID="$!"
printf '%s\n' "$PID" >"$RUN_ROOT/pid.txt"
{
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'input=%s\n' "$INPUT"
  printf 'compile_cache=%s\n' "$COMPILE_CACHE"
  printf 'layout_cache=%s\n' "$LAYOUT_CACHE"
  printf 'decode_cache=%s\n' "$DECODE_CACHE"
} >"$RUN_ROOT/preflight.txt"

printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'RUN_LOG=%s\n' "$RUN_LOG"
printf 'PID=%s\n' "$PID"
printf 'PHYSICAL_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
