# Work-server 310P MinerU manual-FP32 vision default smoke

This brief is for the AI agent on Luka's Atlas 310P work server. The preceding
S=1024 matrix established that the two manual-FP32 LayerNorm lanes compiled in
roughly 50 to 70 seconds, while every `nn.LayerNorm` lane either timed out or
failed in `MatmulLayerNormReduce`.

## Goal

Pull the production default change and run the real two-page MinerU pipeline.
The compiled 32-block vision path now always expands its 64 encoder LayerNorm
calls into explicit FP32 math. This applies to every vision bucket and every
chip. It retains ordinary `nn.Linear` and native D80 PromptFA.

No LayerNorm flag is required. The purpose of this smoke is to prove that the
normal production entry point selects the new default, compiles or loads every
vision bucket it actually needs, and completes two pages.

## Rules

- This checkout is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard changes.
- Do not modify packages, CANN, TorchAir, torch-npu, model files, datasets, or
  evaluator files.
- Reuse the already verified Experiment-11 Python environment, model, dataset,
  and production cache root from the preceding diagnosis.
- Do not delete or rename any cache. The manual-FP32 graph has a distinct cache
  identity and can coexist under the same vision cache root with old module-LN
  graphs.
- Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Use one healthy free 310P. Never terminate another user's process and never
  fall back to CPU or CUDA.
- A first call taking roughly 50 to 70 seconds is normal. A ten-minute timeout
  with no `first_call_finish` is not normal compilation.
- If the smoke fails, reply directly to Luka in plain text. Do not create a
  Markdown report and do not propose or apply a source change.

## Phase 1: pull and verify

```bash
set -eo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet
git diff --cached --quiet
git pull --ff-only origin main
git diff --quiet
git diff --cached --quiet
git merge-base --is-ancestor \
  eed292c9f3eab05d85fc42f3ee9fbf66a95adf1d HEAD
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE.md
git rev-parse HEAD
```

Source the same server-owned CANN activation used by the completed matrix.
Do this before `set -u`. Select one healthy free physical 310P, then export:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHON_BIN=/absolute/path/to/the_verified_exp11_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the_1651_images
export CACHE_ROOT=/home/lukaiv/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/310p_cache_rebuild_b32_kv4096_ab925dd51fbe_20260904T095205
```

Use the replacement cache root recorded by the earlier agent only if the exact
root above does not exist. Do not invent a new production cache root.

```bash
set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$CACHE_ROOT/production_increfa_real_nz_compile"
test -d "$CACHE_ROOT/vision_prefill_b1_fp16"
test -d "$CACHE_ROOT/text_prefill_packed_fp16"

(
  cd 11_mineru_2_5_pro_inference
  PYTHONPYCACHEPREFIX=/tmp/mineru_manual_fp32_default_pycache \
    "$PYTHON_BIN" -m unittest discover -p 'test_*.py' -v
)
```

All tests must pass, including
`test_compiled_vision_defaults_to_manual_fp32_layer_norm`.

Acquire the existing production cache lock:

```bash
exec 9>"$CACHE_ROOT/manual_fp32_default_smoke.lock"
flock -n 9
```

## Phase 2: real two-page production smoke

```bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export SMOKE_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_manual_fp32_default_smoke_${COMMIT_SHORT}_${RUN_TAG}"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
git rev-parse HEAD >"$SMOKE_ROOT/commit.txt"

SMOKE_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$SMOKE_ROOT/output"
  --offset 0 --limit 2 --warmup-pages 0 --no-resume --fail-fast
  --batch-size 32 --page-batch-size 2 --global-request-stream
  --layout-image-size 1036 1036 --processor-min-pixels 25088
  --local-dtype float16 --local-compiled-cache-length 4096
  --local-decode-attention increfa --local-decode-weight-format decode_nz
  --local-decode-rotary-impl npu_apply --local-prepare-prefetch-depth 64
  --local-prefill-metrics --local-text-backend torchair-packed
  --local-text-buckets 128,256,512,1024 --local-text-max-members 32
  --local-text-torchair-cache-dir "$CACHE_ROOT/text_prefill_packed_fp16"
  --local-vision-attention prompt_flash_attention
  --local-vision-backend torchair
  --local-vision-buckets 384,512,768,1024,1536,2048,3072,4224,5632
  --local-vision-pack-target 768 --local-vision-lookahead 32
  --local-vision-torchair-cache-dir "$CACHE_ROOT/vision_prefill_b1_fp16"
  --local-torchair-cache-dir "$CACHE_ROOT/production_increfa_real_nz_compile"
  --local-decode-increfa-length-mode pse_sentinel_310p
  --token-trace --hash-model-files
  --streaming-pages --streaming-page-window 2
)

printf '%q ' "${SMOKE_COMMAND[@]}" >"$SMOKE_ROOT/command.sh"
printf '\n' >>"$SMOKE_ROOT/command.sh"
set +e
timeout --signal=TERM --kill-after=30s 3600s \
  "${SMOKE_COMMAND[@]}" >"$SMOKE_ROOT/run.log" 2>&1
SMOKE_STATUS=$?
set -e
printf '%s\n' "$SMOKE_STATUS" >"$SMOKE_ROOT/exit_code.txt"
test "$SMOKE_STATUS" = 0
```

The log must show only manual-FP32 compiled vision graphs:

```bash
grep 'MINERU_VISION_COMPILE' "$SMOKE_ROOT/run.log"
grep -q 'layer_norm=manual_fp32' "$SMOKE_ROOT/run.log"
! grep -q 'MINERU_VISION_COMPILE .*layer_norm=module' "$SMOKE_ROOT/run.log"
```

Run the smoke gate:

```bash
export SMOKE_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SMOKE_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
decode = summary["streaming"]["decode"]
vision = summary["local_compiled_vision"]
assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["local_decode_attention"] == "increfa"
assert summary["batch_size"] == 32
assert summary["streaming_page_window"] == 2
assert vision["layer_norm_impl"] == "manual_fp32"
assert vision["projection_impl"] == "linear"
assert vision["attention"] == "prompt_flash_attention"
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["idle_rows_with_ready_work"] == 0
assert len(list((root / "output/predictions").glob("*.md"))) == 2
assert len(list((root / "output/content_lists").glob("*.json"))) == 2
print("310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE: PASS")
print("vision", vision)
print("pipeline_wall_s", summary["pipeline_wall_s"])
PY
```

## Required reply

Reply directly to Luka with:

```text
310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE: PASS or FAIL
commit / host / exact 310P / torch / torch-npu / CANN:
unit-test result:
smoke command / exit / elapsed:
ordered MINERU_VISION_COMPILE markers:
per-bucket first-call times and cache paths:
vision LayerNorm / projection / attention metadata:
completed / failed / skipped:
prediction and content-list counts:
first causal error and last unmatched marker, if failed:
smoke root:
```

Stop after this smoke and reply. Do not start the 1,651-page run until Luka
reviews the result.
