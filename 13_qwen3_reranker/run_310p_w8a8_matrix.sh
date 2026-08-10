#!/usr/bin/env bash

set -uo pipefail

REPO="$(git rev-parse --show-toplevel)" || exit 1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODEL_06B_DIR="${MODEL_06B_DIR:-}"
MODEL_4B_DIR="${MODEL_4B_DIR:-}"
DEVICE="${DEVICE:-npu:0}"
BATCH_06B="${BATCH_06B:-16}"
BATCH_4B="${BATCH_4B:-4}"
WARMUPS="${WARMUPS:-3}"
REPEATS="${REPEATS:-20}"
CONTINUATION_LENGTHS="${CONTINUATION_LENGTHS:-128,384}"
PHASES="${PHASES:-all}"
COMMIT="$(git rev-parse --short=12 HEAD)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/tmp/13_qwen3_reranker/310p_w8a8_${COMMIT}_${RUN_STAMP}}"

if [[ -z "$MODEL_06B_DIR" || -z "$MODEL_4B_DIR" ]]; then
  echo "MODEL_06B_DIR and MODEL_4B_DIR must point to local model directories" >&2
  exit 2
fi
if [[ ! -f "$MODEL_06B_DIR/config.json" || ! -f "$MODEL_4B_DIR/config.json" ]]; then
  echo "both model directories must contain config.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
overall_status=0

should_run() {
  local phase="$1"
  [[ "$PHASES" == "all" || ",$PHASES," == *",$phase,"* ]]
}

run_phase() {
  local phase="$1"
  shift
  local phase_dir="$OUTPUT_ROOT/$phase"
  mkdir -p "$phase_dir"
  {
    echo "git_commit=$(git rev-parse HEAD)"
    echo "hostname=$(hostname)"
    echo "python_bin=$PYTHON_BIN"
    echo "device=$DEVICE"
    echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } > "$phase_dir/command.txt"
  echo "PHASE_START $phase"
  set +e
  "$@" 2>&1 | tee "$phase_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$phase_dir/exit_code.txt"
  echo "PHASE_END $phase exit_code=$status"
  if [[ $status -ne 0 ]]; then
    overall_status=$status
  fi
}

if should_run environment; then
  run_phase environment "$PYTHON_BIN" -c \
    'import json,sys,torch,torch_npu; torch.npu.set_compile_mode(jit_compile=False); d=torch.device(sys.argv[1]); torch.npu.set_device(d); x=torch.arange(8,dtype=torch.float16,device=d); print(json.dumps({"torch":torch.__version__,"torch_npu":torch_npu.__version__,"device":torch.npu.get_device_name(d),"device_count":torch.npu.device_count(),"smoke":(x+1).cpu().tolist(),"ops":{n:callable(getattr(torch_npu,n,None)) for n in ("npu_quantize","npu_quant_matmul","npu_trans_quant_param","npu_format_cast")}},sort_keys=True))' \
    "$DEVICE"
fi

if should_run op_06b; then
  run_phase op_06b "$PYTHON_BIN" "$REPO/13_qwen3_reranker/probe_310p_w8a8_ops.py" \
    --model-dir "$MODEL_06B_DIR" --device "$DEVICE" --tokens 512 \
    --warmups "$WARMUPS" --repeats "$REPEATS" \
    --compile-cache-dir "$OUTPUT_ROOT/cache/op_06b" \
    --json-out "$OUTPUT_ROOT/op_06b/result.json"
fi

if should_run op_4b; then
  run_phase op_4b "$PYTHON_BIN" "$REPO/13_qwen3_reranker/probe_310p_w8a8_ops.py" \
    --model-dir "$MODEL_4B_DIR" --device "$DEVICE" --tokens 512 \
    --warmups "$WARMUPS" --repeats "$REPEATS" \
    --compile-cache-dir "$OUTPUT_ROOT/cache/op_4b" \
    --json-out "$OUTPUT_ROOT/op_4b/result.json"
fi

if should_run op_4b_doc_layout; then
  run_phase op_4b_doc_layout "$PYTHON_BIN" "$REPO/13_qwen3_reranker/probe_310p_w8a8_ops.py" \
    --model-dir "$MODEL_4B_DIR" --device "$DEVICE" --tokens 512 \
    --warmups "$WARMUPS" --repeats "$REPEATS" \
    --weight-format fractal_nz_inference_doc \
    --compile-cache-dir "$OUTPUT_ROOT/cache/op_4b_doc_layout" \
    --json-out "$OUTPUT_ROOT/op_4b_doc_layout/result.json"
fi

run_model_benchmark() {
  local model_key="$1"
  local model_dir="$2"
  local batch_size="$3"
  local weight_mode="$4"
  local phase="model_${model_key}_${weight_mode}"
  if ! should_run "$phase"; then
    return
  fi
  local ffn_mode="dense"
  if [[ "$weight_mode" == "w8a8" ]]; then
    ffn_mode="gate_up_w8a8"
  fi
  run_phase "$phase" "$PYTHON_BIN" "$REPO/13_qwen3_reranker/benchmark_prefix_cache_throughput.py" \
    --model-dir "$model_dir" --device "$DEVICE" \
    --batch-sizes "$batch_size" --continuation-lengths "$CONTINUATION_LENGTHS" \
    --batch-sweep-continuation 128 --length-sweep-batch "$batch_size" --matrix axes \
    --lanes prefix_promptfa_compiled --warmups "$WARMUPS" --repeats "$REPEATS" \
    --compile-cache-dir "$OUTPUT_ROOT/cache/${model_key}_${weight_mode}" \
    --prefill-optimizations combined_bsnd --linear-weight-format fractal_nz \
    --enable-internal-format --ffn-weight-mode "$ffn_mode" \
    --json-out "$OUTPUT_ROOT/$phase/result.json"
}

run_model_benchmark 06b "$MODEL_06B_DIR" "$BATCH_06B" dense
run_model_benchmark 06b "$MODEL_06B_DIR" "$BATCH_06B" w8a8
run_model_benchmark 4b "$MODEL_4B_DIR" "$BATCH_4B" dense
run_model_benchmark 4b "$MODEL_4B_DIR" "$BATCH_4B" w8a8

if should_run summary; then
  run_phase summary "$PYTHON_BIN" "$REPO/13_qwen3_reranker/summarize_310p_w8a8_matrix.py" \
    --run-root "$OUTPUT_ROOT"
fi

echo "OUTPUT_ROOT $OUTPUT_ROOT"
exit "$overall_status"
