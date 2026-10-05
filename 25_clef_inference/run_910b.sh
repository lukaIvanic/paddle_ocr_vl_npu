#!/usr/bin/env bash
# Default to the existing free-device selector; do not terminate other workloads.
mode=${1:-transformers}
case "$mode" in
    transformers) runner=run_transformers_smoke.py; extra=(--dtype bf16) ;;
    local) runner=run_local_smoke.py; extra=() ;;
    *) echo "Usage: $0 [transformers|local]" >&2; exit 2 ;;
esac
source npu-setup || exit 1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${ASCEND_RT_VISIBLE_DEVICES:?No free NPU selected}"
: "${RUN_ROOT:?Choose a fresh run directory}"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
python=/workspace/venvs/clef_transformers_py312/bin/python
{
    git rev-parse HEAD
    hostname
    printf 'Physical NPU: %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'Command: RUN_ROOT=%q bash %q %q\n' "$RUN_ROOT" "$0" "$mode"
    "$python" -m pip show torch torch-npu transformers accelerate
    npu-smi info
} > "$RUN_ROOT/command.txt"
set +e
timeout 900 "$python" -u "25_clef_inference/$runner" \
    --model /workspace/models/clef-flash "${extra[@]}" \
    --output "$RUN_ROOT/result.json" 2>&1 | tee "$RUN_ROOT/run.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
