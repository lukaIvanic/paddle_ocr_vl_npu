#!/usr/bin/env bash
# Source npu-setup once; run every variant sequentially on that physical card.
set -e
cd "$(dirname "$0")/.."
if source npu-setup; then
  :
elif [ -n "${TEXT_PHYSICAL_NPU:-}" ]; then
  # Explicit user-reserved card may contain an idle reservation process, so
  # automatic free-card selection can fail before sourcing the vendor env.
  source /usr/local/Ascend/cann-9.0.0/set_env.sh || true
  source /usr/local/Ascend/nnal/atb/set_env.sh || true
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
  export LD_LIBRARY_PATH="/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
else
  exit 1
fi
if [ -n "${TEXT_PHYSICAL_NPU:-}" ]; then
  [[ "$TEXT_PHYSICAL_NPU" =~ ^[0-7]$ ]] || exit 2
  export ASCEND_RT_VISIBLE_DEVICES="$TEXT_PHYSICAL_NPU"
  printf 'Using explicitly user-reserved physical NPU %s\n' "$ASCEND_RT_VISIBLE_DEVICES"
fi
# Vendor environment scripts reference optional unset variables internally.
set -euo pipefail
TEXT_VARIANT_ROOT="${TEXT_VARIANT_ROOT:?Set a fresh evidence directory}"
TEXT_FROZEN_ROOT=tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817
TEXT_ANCHOR=tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/hr_hf/output/image_00.pt
TEXT_PY=/workspace/venvs/colqwen3_hf_py312/bin/python
test ! -e "$TEXT_VARIANT_ROOT"
mkdir -p "$TEXT_VARIANT_ROOT/cache/baseline"
cp -a "$TEXT_FROZEN_ROOT/cache/." "$TEXT_VARIANT_ROOT/cache/baseline/"
git rev-parse HEAD > "$TEXT_VARIANT_ROOT/source_commit.txt"
printf '%s\n' "$ASCEND_RT_VISIBLE_DEVICES" > "$TEXT_VARIANT_ROOT/physical_npu.txt"
npu-smi info -t proc-mem -i "$ASCEND_RT_VISIBLE_DEVICES" > "$TEXT_VARIANT_ROOT/npu_before.txt"
for TEXT_VARIANT in baseline bsnd rotary_bnsd apply_bnsd apply_bsnd swiglu apply_bsnd_swiglu baseline_end; do
  TEXT_VARIANT_NAME="$TEXT_VARIANT"
  TEXT_LANES=(raw_eager torchair)
  if [ "$TEXT_VARIANT" = baseline_end ]; then
    TEXT_VARIANT_NAME=baseline
    TEXT_LANES=(torchair)
  fi
  for TEXT_LANE in "${TEXT_LANES[@]}"; do
    TEXT_OUTPUT="$TEXT_VARIANT_ROOT/$TEXT_VARIANT/$TEXT_LANE"
    mkdir -p "$TEXT_OUTPUT"
    TEXT_CMD=("$TEXT_PY" -u 21_colqwen3_4b_inference/profile_warm_text.py
      --model /workspace/models/Ops-Colqwen3-4B --anchor "$TEXT_ANCHOR"
      --frozen-inputs "$TEXT_FROZEN_ROOT/raw_eager/output/text_inputs.pt"
      --variant "$TEXT_VARIANT_NAME" --execution "$TEXT_LANE" --diagnostic-parity
      --output-dir "$TEXT_OUTPUT/output" --cache-root "$TEXT_VARIANT_ROOT/cache/$TEXT_VARIANT_NAME"
      --warmups 5 --repeats 30 --profile-steps 3 --metrics pipe)
    {
      git rev-parse HEAD
      hostname
      printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
      printf '%q ' "${TEXT_CMD[@]}"
      printf '\n'
    } > "$TEXT_OUTPUT/command.txt"
    printf 'START %s %s\n' "$TEXT_VARIANT" "$TEXT_LANE"
    set +e
    "${TEXT_CMD[@]}" > "$TEXT_OUTPUT/run.log" 2>&1
    TEXT_EXIT=$?
    set -e
    printf '%s\n' "$TEXT_EXIT" > "$TEXT_OUTPUT/exit_code.txt"
    printf 'FINISH %s %s exit=%s\n' "$TEXT_VARIANT" "$TEXT_LANE" "$TEXT_EXIT"
    if [ "$TEXT_EXIT" != 0 ]; then
      tail -n 18 "$TEXT_OUTPUT/run.log"
      if [ "$TEXT_VARIANT" = baseline ]; then exit "$TEXT_EXIT"; fi
    fi
  done
done
npu-smi info -t proc-mem -i "$ASCEND_RT_VISIBLE_DEVICES" > "$TEXT_VARIANT_ROOT/npu_after.txt"
"$TEXT_PY" 21_colqwen3_4b_inference/analyze_text_variants.py --run-dir "$TEXT_VARIANT_ROOT"
