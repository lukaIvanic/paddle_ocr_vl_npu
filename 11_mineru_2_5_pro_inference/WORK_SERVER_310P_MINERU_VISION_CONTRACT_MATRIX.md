# Work-server 310P MinerU S=1024 vision contract matrix

This brief is for the AI agent on Luka's Atlas 310P work server. Run it after
the earlier production diagnosis found a stall while compiling the recognition
vision bucket with 800 real tokens and 224 padded tokens.

## Goal

Run the real MinerU 32-block vision boundary at physical S=1024 while changing
one compiler contract at a time:

1. stock `nn.LayerNorm` plus stock `nn.Linear`;
2. a D80-to-D96 pad/PromptFA/slice boundary after RoPE;
3. explicit FP32 LayerNorm math;
4. `npu_grouped_matmul` with a 3D weight for QKV;
5. the same grouped MatMul for QKV and LayerNorm-fed MLP FC1;
6. the combined historical Paddle 310P workarounds;
7. full manual attention lowered through explicit 3D BMMs.

This is not a standalone PromptFA operator probe. Every lane loads the real
model and executes image processing, patch embedding, RoPE, all 32 vision
blocks, the real/padding mask, patch merger, text prefill, and eight generated
tokens. The input is resized to produce exactly 800 real vision tokens and is
padded to S=1024.

## Rules

- This checkout is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard changes.
- Do not modify packages, CANN, TorchAir, torch-npu, model files, or production
  caches.
- Use the same verified Experiment-11 Python environment and CANN activation
  as the preceding 310P diagnosis.
- Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` before Python imports torch-npu.
- Use one healthy free 310P. Never terminate another user's process and never
  fall back to CPU or CUDA.
- The matrix requires new graphs because the compiler contracts differ. Use the
  dedicated diagnostic cache root below. Do not delete or rename the existing
  production cache.
- Each lane runs in a separate Python process and has a separate cache identity.
  A failed lane must not reuse another lane's partial cache.
- Do not call a missing `first_call_finish` slow compilation after its
  ten-minute timeout. Record it as a timeout at the exact unmatched marker.
- Do not create a Markdown report. Reply directly to Luka in plain text with
  the requested table and evidence paths.
- Do not propose or apply a source change. If setup differs from this brief,
  stop and report the mismatch.

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
  396a36de34c2d48c62ae8d140a2fc2f68c7b1834 HEAD
test -f \
  11_mineru_2_5_pro_inference/run_vision_prefill_contract_matrix.sh
test -f \
  11_mineru_2_5_pro_inference/test_vision_prefill_contracts.py
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_VISION_CONTRACT_MATRIX.md
git rev-parse HEAD
```

Source the same server-owned CANN activation used by the completed diagnosis.
Do this before `set -u`. Select one healthy free physical 310P, then export:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHON_BIN=/absolute/path/to/the_verified_exp11_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
```

Resolve the values from the earlier successful Experiment-11 run. Do not guess
or install a replacement environment.

```bash
set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$MODEL_DIR/config.json"
test -f "$WORK_SERVER_REPO/crops/crop_01_text_block_en.png"

(
  cd 11_mineru_2_5_pro_inference
  PYTHONPYCACHEPREFIX=/tmp/mineru_vision_contract_pycache \
    "$PYTHON_BIN" -m unittest -v test_vision_prefill_contracts.py
)
```

All five tests must pass.

## Phase 2: exact 800-real-token S=1024 matrix

Run this in one persistent Bash or tmux coordinator session:

```bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export MATRIX_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_vision_contract_matrix_${COMMIT_SHORT}_${RUN_TAG}"
export MATRIX_CACHE="$WORK_SERVER_REPO/.runtime_cache/11_mineru_2_5_pro_inference/310p_vision_contract_matrix_${COMMIT_SHORT}_${RUN_TAG}"
test ! -e "$MATRIX_ROOT"
test ! -e "$MATRIX_CACHE"
mkdir -p "$MATRIX_ROOT" "$MATRIX_CACHE"
git rev-parse HEAD >"$MATRIX_ROOT/commit.txt"

set +e
PYTHON="$PYTHON_BIN" \
MODEL="$MODEL_DIR" \
IMAGE="$WORK_SERVER_REPO/crops/crop_01_text_block_en.png" \
BUCKET=1024 \
LAYOUT_SIZE_W=1133 \
LAYOUT_SIZE_H=140 \
WARMUP=0 \
REPEATS=1 \
GENERATION_TOKENS=8 \
LANE_TIMEOUT_S=600 \
RUN_TAG="$RUN_TAG" \
CACHE_ROOT="$MATRIX_CACHE" \
OUT_ROOT="$MATRIX_ROOT/lanes" \
bash 11_mineru_2_5_pro_inference/run_vision_prefill_contract_matrix.sh \
  >"$MATRIX_ROOT/run.log" 2>&1
MATRIX_STATUS=$?
set -e
printf '%s\n' "$MATRIX_STATUS" >"$MATRIX_ROOT/exit_code.txt"
```

The matrix intentionally returns nonzero if any lane fails. Do not stop merely
because `MATRIX_STATUS` is nonzero. Inspect every lane.

For each lane with `result.json`, require:

```bash
for result in "$MATRIX_ROOT"/lanes/*/result.json; do
  test "$(jq -r '.real_vision_tokens' "$result")" = 800
  test "$(jq -r '.physical_vision_tokens' "$result")" = 1024
  test "$(jq -r '.feature_comparison.nonfinite_compiled' "$result")" = 0
  test "$(jq -r '.generation_comparison.exact' "$result")" = true
done
```

For a failed lane, preserve its `run.log` and `exit_code.txt`. Report:

- the final `MINERU_VISION_COMPILE` marker;
- whether `cache_compile_finish` occurred;
- whether `first_call_finish` occurred;
- the first Python, TorchAir, GE, CANN, or AICore error;
- whether the shell exit was timeout 124 or another code;
- whether an `.om` file appeared in that lane's cache directory.

If an AICore execution error or device-health error occurs, stop after that
lane and report it. Do not continue on a poisoned device. Ordinary compile
timeouts and GE compile errors do not require rerunning earlier lanes.

## Required reply

Reply directly to Luka with one row per lane:

```text
lane | exit | last compile marker | first_call seconds or >600 | real/physical tokens | nonfinite count | feature cosine | token exact | first causal error | cache path
```

Then include:

```text
commit / host / exact 310P / torch / torch-npu / CANN:
matrix root:
matrix cache root:
which individual change first made S=1024 terminate:
whether the combined Paddle-style lane terminated:
whether the explicit 3D-BMM manual-attention control terminated:
what is proven:
what remains unproven:
```

Do not run the 1,651-page benchmark in this phase. The purpose of this matrix is
to identify the smallest production-graph contract change that compiles safely
on 310P.
