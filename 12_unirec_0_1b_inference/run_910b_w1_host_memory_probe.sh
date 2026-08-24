#!/usr/bin/env bash

# Reproduce the accuracy-safe W1 streaming path and attribute host memory.
# The run is backgrounded. All evidence is written below RUN_ROOT.

source npu-setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO"

case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*)
    printf 'REJECTED_PHYSICAL_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
MODEL="${MODEL:-/workspace/models/unirec-0.1b}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV2_safetensors}"
OPENOCR_ROOT="${OPENOCR_ROOT:-/workspace/repos/OpenOCR}"
INPUT="${INPUT:-$REPO/tmp/12_unirec_0_1b_inference/910b_rep128_k10_l1_7cd0f82_20260817T150836/representative_128_v1_images}"
COMPILE_CACHE="${COMPILE_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf}"
LAYOUT_CACHE="${LAYOUT_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/representative128_layout_compiled_fp32_optimized_b2_6deceef}"
DECODE_CACHE_PARENT="${DECODE_CACHE_PARENT:-$REPO/.runtime_cache/12_unirec_0_1b_inference/production_dual_decode_nz_lmhead57344_0b707c1}"
REFERENCE_OUTPUT="${REFERENCE_OUTPUT:-}"
CPUSET="${CPUSET:-0-63}"
LIMIT="${LIMIT:-128}"
TORCH_WARM_POOL_VALUE="${TORCH_WARM_POOL_VALUE:-0}"
TE_PARALLEL_COMPILER_VALUE="${TE_PARALLEL_COMPILER_VALUE:-1}"
CANN_KNOWLEDGE_BANK_PROCESS_NUM_VALUE="${CANN_KNOWLEDGE_BANK_PROCESS_NUM_VALUE:-0}"
MEMORY_SAMPLE_INTERVAL_S="${MEMORY_SAMPLE_INTERVAL_S:-10}"
PROCESS_SNAPSHOT_INTERVAL_S="${PROCESS_SNAPSHOT_INTERVAL_S:-60}"
RUN_LABEL="${RUN_LABEL:-nowarmpool}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$INPUT"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE"
test -d "$DECODE_CACHE_PARENT"

STAMP="$(date +%Y%m%dT%H%M%S)"
COMMIT="$(git rev-parse --short HEAD)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/w1_hostmem_${RUN_LABEL}_${COMMIT}_${STAMP}"
OUTPUT="$RUN_ROOT/output"
RUN_LOG="$RUN_ROOT/run.log"
mkdir -p "$OUTPUT"

inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'vision %p %s %T@\n'
    find "$LAYOUT_CACHE" -type f -name '*.om' -printf 'layout %p %s %T@\n'
    find "$DECODE_CACHE_PARENT" -type f -name '*.om' -printf 'decode %p %s %T@\n'
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
  "TORCH_WARM_POOL=$TORCH_WARM_POOL_VALUE"
  "TE_PARALLEL_COMPILER=$TE_PARALLEL_COMPILER_VALUE"
  "CANN_KNOWLEDGE_BANK_PROCESS_NUM=$CANN_KNOWLEDGE_BANK_PROCESS_NUM_VALUE"
  "UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=$DECODE_CACHE_PARENT"
  "$PYTHON_BIN" "$SCRIPT_DIR/run_opendoc_batched_unirec.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-transformers-model "$LAYOUT_MODEL"
  --layout-backend transformers_npu
  --layout-execution torchair
  --layout-dtype float32
  --layout-reading-order-dtype float32
  --layout-weight-format native
  --layout-depthwise-rewrite native
  --layout-threshold 0.5
  --layout-compile-cache-dir "$LAYOUT_CACHE"
  --layout-process-workers 1
  --prefill-in-layout-workers
  --shared-cross-kv-budget-gib 3.5
  --input "$INPUT"
  --output-dir "$OUTPUT"
  --device npu:0
  --dtype float16
  --offset 0
  --limit "$LIMIT"
  --pipeline-warmup-pages 8
  --layout-batch-size 2
  --layout-cpu-threads 16
  --vision-page-lookahead 4
  --vision-bucket-preset 310p_k20_l4
  --vision-focal-depthwise-rewrite constant_grouped_all
  --vision-weight-format torchair_internal
  --recognition-preprocess-threads 8
  --recognition-input-contract compact_uint8_hwc
  --vision-prefill-mode compiled_full_buckets
  --text-prefill-mode compiled_packed_s1024
  --decode-mode compiled_ifa
  --decode-scheduling continuous
  --decode-batch-size 128
  --cross-cache-length 1320
  --self-cache-length 2048
  --max-length 2048
  --decode-weight-format nz
  --decode-lm-head-rows 57344
  --decode-admission-prefetch-depth 0
  --decode-live-arena-warmup-passes 2
  --page-decode-workers 4
  --page-image-decoder opencv
  --page-prepare-workers 1
)

inventory "$RUN_ROOT/om_before.txt"
printf 'unix_s\telapsed_s\tprocesses\trss_kib\tpss_kib\tprivate_kib\tshm_bytes\thbm_used_mb\n' \
  >"$RUN_ROOT/memory.tsv"
printf '%q ' "${command[@]}" >"$RUN_ROOT/command.txt"
printf '\n' >>"$RUN_ROOT/command.txt"

started="$(date +%s.%N)"
setsid "${command[@]}" >"$RUN_LOG" 2>&1 &
PID="$!"
printf '%s\n' "$PID" >"$RUN_ROOT/pid.txt"
printf '%s\n' "$started" >"$RUN_ROOT/start.txt"

monitor() {
  local peak_pss=0
  local last_snapshot_s=0
  local pids pid rss_value now elapsed shm_bytes hbm_used
  local process_count rss pss private values
  while :; do
    pids="$(ps -o pid= --sid "$PID" 2>/dev/null | xargs)"
    test -n "$pids" || break
    process_count=0
    rss=0
    pss=0
    private=0
    for pid in $pids; do
      test -r "/proc/$pid/smaps_rollup" || continue
      process_count=$((process_count + 1))
      rss_value="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null)"
      values="$(awk '
        /^Pss:/ {pss=$2}
        /^Private_Clean:/ {private_clean=$2}
        /^Private_Dirty:/ {private_dirty=$2}
        END {print pss+0, private_clean+private_dirty}
      ' "/proc/$pid/smaps_rollup" 2>/dev/null)"
      set -- $values
      rss=$((rss + ${rss_value:-0}))
      pss=$((pss + ${1:-0}))
      private=$((private + ${2:-0}))
    done
    shm_bytes="$(df -B1 --output=used /dev/shm | tail -n 1 | tr -d ' ')"
    hbm_used="$(/usr/local/bin/npu-status 2>/dev/null \
      | sed -n "s/^NPU $ASCEND_RT_VISIBLE_DEVICES:.*HBM=\([0-9]*\)\/.*$/\1/p")"
    now="$(date +%s.%N)"
    elapsed="$(awk -v now="$now" -v start="$started" \
      'BEGIN {printf "%.6f", now-start}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$now" "$elapsed" "$process_count" "$rss" "$pss" "$private" \
      "${shm_bytes:-0}" "${hbm_used:-0}" >>"$RUN_ROOT/memory.tsv"
    if (( pss > peak_pss )); then
      peak_pss="$pss"
    fi
    if awk -v elapsed="$elapsed" -v last="$last_snapshot_s" \
      -v interval="$PROCESS_SNAPSHOT_INTERVAL_S" \
      'BEGIN {exit !((elapsed-last) >= interval)}'; then
      last_snapshot_s="$elapsed"
      {
        printf 'captured_elapsed_s=%s total_pss_kib=%s\n' "$elapsed" "$pss"
        printf 'pid\tppid\trss_kib\tpss_kib\tprivate_kib\tcommand\n'
        for pid in $pids; do
          test -r "/proc/$pid/smaps_rollup" || continue
          rss_value="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null)"
          values="$(awk '
            /^Pss:/ {pss=$2}
            /^Private_Clean:/ {private_clean=$2}
            /^Private_Dirty:/ {private_dirty=$2}
            END {print pss+0, private_clean+private_dirty}
          ' "/proc/$pid/smaps_rollup" 2>/dev/null)"
          set -- $values
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$pid" "$(awk '/^PPid:/ {print $2}' "/proc/$pid/status")" \
            "${rss_value:-0}" "${1:-0}" "${2:-0}" \
            "$(tr '\0' ' ' <"/proc/$pid/cmdline")"
        done
      } >"$RUN_ROOT/process_memory_peak.tsv"
    fi
    sleep "$MEMORY_SAMPLE_INTERVAL_S"
  done
  inventory "$RUN_ROOT/om_after.txt"
  diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
    >"$RUN_ROOT/om.diff" || true
  awk -F '\t' '
    NR > 1 {
      if ($3 > processes) processes=$3
      if ($4 > rss) rss=$4
      if ($5 > pss) pss=$5
      if ($6 > private) private=$6
      if ($7 > shm) shm=$7
      if ($8 > hbm) hbm=$8
    }
    END {
      printf "processes_peak=%d\nrss_kib_peak=%d\npss_kib_peak=%d\n", processes, rss, pss
      printf "private_kib_peak=%d\nshm_bytes_peak=%d\nhbm_mb_peak=%d\n", private, shm, hbm
    }
  ' "$RUN_ROOT/memory.tsv" >"$RUN_ROOT/memory_summary.txt"
  if test -n "$REFERENCE_OUTPUT" && test -d "$REFERENCE_OUTPUT"; then
    diff -qr --exclude '*.json' --exclude '_pipeline_warmup' \
      "$REFERENCE_OUTPUT" "$OUTPUT" >"$RUN_ROOT/output.diff" || true
  fi
}

monitor >"$RUN_ROOT/monitor.log" 2>&1 &
MONITOR_PID="$!"
printf '%s\n' "$MONITOR_PID" >"$RUN_ROOT/monitor_pid.txt"

{
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'torch_warm_pool=%s\n' "$TORCH_WARM_POOL_VALUE"
  printf 'te_parallel_compiler=%s\n' "$TE_PARALLEL_COMPILER_VALUE"
  printf 'cann_knowledge_bank_process_num=%s\n' \
    "$CANN_KNOWLEDGE_BANK_PROCESS_NUM_VALUE"
  printf 'memory_sample_interval_s=%s\n' "$MEMORY_SAMPLE_INTERVAL_S"
  printf 'reference_output=%s\n' "$REFERENCE_OUTPUT"
} >"$RUN_ROOT/preflight.txt"

printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'RUN_LOG=%s\n' "$RUN_LOG"
printf 'PID=%s\n' "$PID"
printf 'MONITOR_PID=%s\n' "$MONITOR_PID"
printf 'PHYSICAL_NPU=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
