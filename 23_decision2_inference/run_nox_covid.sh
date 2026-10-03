#!/usr/bin/env bash
# Reserved development device only. The five-NPU Qwen reranker job is untouched.
set -euo pipefail
run=tmp/23_decision2_inference/nox_covid_dc7e53e9
bundle=/workspace/models/Decision-2.0-Nox-4B
view=/workspace/models/Decision-2.0-Nox-4B-vllm-view
prepared=$run/prepared_v2
mkdir -p "$run"
exec >"$run/run.log" 2>&1
trap 'status=$?; echo "$status" > "$run/exit_code.txt"' EXIT
{ git rev-parse HEAD; hostname; printf 'Physical device 7; %s\n' "$0"; } >"$run/command.txt"
printf 'Phase: await verified model download and pinned dataset preparation\n'
for attempt in {1..180}; do
    if [[ -f tmp/23_decision2_inference/nox_download_ranges/exit_code.txt && -f "$prepared/manifest.json" ]]; then
        break
    fi
    sleep 5
done
[[ $(cat tmp/23_decision2_inference/nox_download_ranges/exit_code.txt) == 0 ]]
[[ -f "$prepared/smoke_cases.json" ]]
set +eu
source npu-setup
status=$?
set -eu
[[ "$status" == 0 ]] || exit "$status"
export ASCEND_RT_VISIBLE_DEVICES=7 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset EOS_DIAGNOSTICS_DIR
if curl -sf --max-time 2 http://127.0.0.1:18423/health; then
    printf 'Port already occupied; refusing to replace its owner\n'
    exit 1
fi
printf 'Phase: native BF16 Nox with original FP32 head\n'
timeout 900 /workspace/venvs/decision2_eval_py312/bin/python -u 23_decision2_inference/run_npu_smoke.py \
    --model "$bundle" --dtype bf16 --cases "$prepared/smoke_cases.json" --output "$run/native.json" --concurrent-workload none \
    >"$run/native.log" 2>&1
printf 'Phase: verified serving view\n'
/workspace/venvs/decision2_vllm_py312/bin/python 23_decision2_inference/prepare_serving_model.py \
    --bundle "$bundle" --output "$view"
printf 'Phase: start Nox serving on reserved NPU7\n'
EOS_MAX_SEQS=32 EOS_BATCH_TOKENS=16384 DECISION_MAX_LENGTH=8192 \
    DECISION_MODEL="$view" DECISION_SERVED_NAME=nox-4b DECISION_MEMORY_FRACTION=0.35 \
    bash 23_decision2_inference/serve_eos.sh "$run/server" &
launcher_pid=$!
ready=0
for attempt in {1..180}; do
    kill -0 "$launcher_pid" || { tail -n 60 "$run/server/server.log"; exit 1; }
    if curl -sf --max-time 2 http://127.0.0.1:18423/health; then ready=1; break; fi
    sleep 5
done
[[ "$ready" == 1 ]]
printf 'Phase: real native-versus-HTTP serving parity\n'
/workspace/venvs/decision2_vllm_py312/bin/python -u 23_decision2_inference/test_serving.py \
    --bundle "$bundle" --reference "$run/native.json" --output "$run/serving_parity.json" \
    --served-model nox-4b --cases "$prepared/smoke_cases.json" --max-length 8192 \
    --max-reference-probability-delta 0.03 --concurrent-workload none >"$run/serving_parity.log" 2>&1
printf 'Phase: full CovidRetrieval, identical saved embedding top100\n'
/workspace/venvs/decision2_vllm_py312/bin/python -u 23_decision2_inference/evaluate_retrieval.py run \
    --bundle "$bundle" --prepared "$prepared" --output "$run/evaluation" \
    --server-command "$run/server/command.txt" --concurrency 64 --max-length 8192 \
    >"$run/evaluation.log" 2>&1
printf 'Phase: pinned retrieval metrics\n'
/workspace/venvs/qwen3_embedding_eval_py312/bin/python -u 23_decision2_inference/evaluate_retrieval.py metrics \
    --prepared "$prepared" --output "$run/evaluation"
printf 'Finished. Nox server remains available on physical NPU7.\n'
