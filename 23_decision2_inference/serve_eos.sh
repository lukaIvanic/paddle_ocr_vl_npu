#!/usr/bin/env bash
set -euo pipefail
# Physical 7 is reserved for Decision2 (T2 resumed on 0,1,2,3,6).
# No other process is stopped or framework/package files changed by this script.
run_dir=${1:?usage: serve_eos.sh RUN_DIRECTORY}
mkdir -p "$run_dir"
exec >"$run_dir/server.log" 2>&1
set +eu
source npu-setup
setup_status=$?
set -eu
if [[ "$setup_status" != 0 ]]; then
    printf 'npu-setup failed with status %s\n' "$setup_status"
    exit "$setup_status"
fi
export ASCEND_RT_VISIBLE_DEVICES=7
printf 'Authorized Decision2 device: physical NPU %s as logical npu:0\n' "$ASCEND_RT_VISIBLE_DEVICES"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 DECISION2_ASCEND_ENABLE=1
unset VLLM_PLUGINS
python=/workspace/venvs/decision2_vllm_py312/bin/python
model=${DECISION_MODEL:-/workspace/models/Decision-2.0-Eos-0.8B-vllm-view}
"$python" -c 'import torch, torch_npu; torch.npu.set_device(0); free,total=torch.npu.mem_get_info(); print({"free":free,"total":total},flush=True); assert free>12*1024**3, "Not enough free NPU memory to share safely"'
args=("$python" -m vllm.entrypoints.openai.api_server
    --model "$model" --served-model-name "${DECISION_SERVED_NAME:-eos-0.8b}"
    --runner pooling --dtype bfloat16 --mamba-ssm-cache-dtype float32
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill
    --max-model-len 2048 --max-num-batched-tokens "${EOS_BATCH_TOKENS:-2048}" --max-num-seqs "${EOS_MAX_SEQS:-4}"
    --gpu-memory-utilization "${DECISION_MEMORY_FRACTION:-0.12}" --host 127.0.0.1 --port 18423)
if [[ -n "${EOS_DIAGNOSTICS_DIR:-}" ]]; then
    args+=(--profiler-config.profiler torch
        --profiler-config.torch_profiler_dir "$EOS_DIAGNOSTICS_DIR/traces"
        --profiler-config.torch_profiler_with_stack false
        --profiler-config.ignore_frontend true)
fi
{ git rev-parse HEAD; hostname; printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"; printf '%q ' "${args[@]}"; printf '\n'; } >"$run_dir/command.txt"
set +e
"${args[@]}" &
server_pid=$!
printf '%s\n' "$server_pid" >"$run_dir/server_pid.txt"
wait "$server_pid"
status=$?
printf '%s\n' "$status" >"$run_dir/exit_code.txt"
exit "$status"
