#!/usr/bin/env bash
# Existing caches, sequential jobs, same reduced-ready configuration.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup first}"
: "${RUN_ROOT:?new output root required}"
PYTHON_BIN=/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python
PAGE_LIMIT=${PAGE_LIMIT:-64}
MODES=${MODES:-"serial streamed"}
export CANN_KNOWLEDGE_BANK_PROCESS_NUM=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
npu-smi info > "$RUN_ROOT/npu_before.txt"
for mode in $MODES; do
    options=()
    case "$mode" in
        serial) options=(--unirec-vision-lanes 0) ;;
        streamed) options=(--unirec-streamed --unirec-vision-lanes 4 --unirec-cpu-workers 4 --unirec-cpu-threads 8) ;;
        *) exit 2 ;;
    esac
    stage="$RUN_ROOT/$mode"
    mkdir -p "$stage"
    command=("$PYTHON_BIN" -u 12_unirec_0_1b_inference/run_with_process_tree_memory.py
        --output "$stage/memory.json" --interval-ms 1000
        --npu-id "$ASCEND_RT_VISIBLE_DEVICES" --npu-interval-ms 1000 --
        "$PYTHON_BIN" -u 18_unirec_paddle_hybrid_pipeline/run_pipeline.py
        --input /workspace/datasets/OmniDocBench/images
        --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json
        --unirec-model-path /workspace/models/unirec-0.1b
        --openocr-root /workspace/repos/OpenOCR
        --unirec-vision-cache .runtime_cache/12_unirec_0_1b_inference/vision_opt_allfocal_internal_b0c5c6e
        --unirec-decode-cache .runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf
        --unirec-ready-capacity 64 --paddle-ready-capacity 32
        "${options[@]}" --limit "$PAGE_LIMIT" --output-dir "$stage/output")
    printf '%q ' "${command[@]}" > "$stage/command.txt"
    printf '\n' >> "$stage/command.txt"
    printf 'START mode=%s pages=%s stage=%s\n' "$mode" "$PAGE_LIMIT" "$stage"
    status=0
    "${command[@]}" > "$stage/run.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$stage/exit_code.txt"
    test "$status" -eq 0
    "$PYTHON_BIN" - "$RUN_ROOT" "$mode" "$PAGE_LIMIT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, '18_unirec_paddle_hybrid_pipeline')
from compare_traces import compare, read_trace
root, mode, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
stage=root/mode
r=json.loads((stage/'output/run_summary.json').read_text())
assert r['pages'] == count
assert abs(r['detailed_timing']['owner_partition_error_s']) < 1e-6
assert r['engines']['unirec']['compact_ready_kv']['rows'] == 0
assert r['engines']['paddle']['ready_kv_pool']['active_slots'] == 0
if mode == 'streamed':
    assert r['engines']['unirec']['streamed_execution']['vision_text_buffer_final_rows'] == 0
base=root/'serial/output/recognition_trace.jsonl'
if base.exists():
    a,b=read_trace(base),read_trace(stage/'output/recognition_trace.jsonl')
    assert a.keys() == b.keys()
    c=compare(a,b)
    (stage/'parity.json').write_text(json.dumps(c,indent=2)+'\n')
    print('PARITY',json.dumps(c),flush=True)
print('PASS',mode,r['pages'],r['wall_s'],r['pages_per_s'],
      'allocated_GiB',r['peak_torch_allocated_bytes']/2**30,flush=True)
PY
    "$PYTHON_BIN" 18_unirec_paddle_hybrid_pipeline/summarize_run.py "$stage/output" > "$stage/summary_printed.json"
    npu-smi info > "$stage/npu_after.txt"
done
printf 'MATRIX_COMPLETE pages=%s\n' "$PAGE_LIMIT"
