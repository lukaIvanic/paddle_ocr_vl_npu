#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${ASCEND_RT_VISIBLE_DEVICES:?source npu-setup first}"
: "${RUN_ROOT:?new output root required}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/mineru_pro_vllm_py312/bin/python}
PAGE_LIMIT=${PAGE_LIMIT:-64}
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
command=("$PYTHON_BIN" -u 12_unirec_0_1b_inference/run_with_process_tree_memory.py
    --output "$RUN_ROOT/memory.json" --interval-ms 1000
    --npu-id "$ASCEND_RT_VISIBLE_DEVICES" --npu-interval-ms 1000 --
    "$PYTHON_BIN" -u 20_unirec_mineru_hybrid_pipeline/run_pipeline.py
    --input /workspace/datasets/OmniDocBench/images
    --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json
    --unirec-model-path /workspace/models/unirec-0.1b
    --openocr-root /workspace/repos/OpenOCR
    --unirec-vision-cache .runtime_cache/12_unirec_0_1b_inference/vision_opt_allfocal_internal_b0c5c6e
    --unirec-decode-cache .runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf
    --limit "$PAGE_LIMIT" --output-dir "$RUN_ROOT/output" "$@")
printf '%q ' "${command[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"
status=0
"${command[@]}" > "$RUN_ROOT/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
test "$status" -eq 0
"$PYTHON_BIN" - "$RUN_ROOT/output/run_summary.json" "$PAGE_LIMIT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['pages']==int(sys.argv[2])
assert abs(r['detailed_timing']['owner_partition_error_s'])<1e-6
for name,engine in r['engines'].items():
    if name=='mineru': assert engine['ready_storage']['live']==0
    if name=='unirec': assert engine['compact_ready_kv']['rows']==0
print('COMPLETION_PASS', r['pages'], r['wall_s'], r['pages_per_s'], flush=True)
PY
