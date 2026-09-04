# Work-server 310P MinerU PSE boundary validation and full chain

This brief is for the AI agent on Luka's Atlas 310P work server. Read
`CLAUDE.md`, `AGENTS.md`, and
`11_mineru_2_5_pro_inference/INCREFA_310P_BOUNDARY.md` first. Run all shell
blocks in one persistent Bash or tmux coordinator session.

## Goal

Validate the source-derived MinerU IncreFA PSE sentinel in this order:

1. run the full B32/KV4096 production decode graph at every physical cache
   position 0 through 4095;
2. replay the same sweep from its warm persistent cache;
3. run the real two-page production pipeline and prove that step 13 at
   effective length 1408 completes;
4. only if all gates pass, run all 1,651 pages and the frozen OmniDocBench
   evaluation exactly as in the existing chained handoff.

Do not replace the position sweep with a standalone IncreFA call. The committed
sweep executes the complete packed-projection, 24-layer, NZ-weight, NPU-RoPE,
IncreFA, static-KV, and LM-head graph used by production.

## Rules

- This checkout is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard changes.
- Do not modify `/vllm-workspace`, packages, CANN, TorchAir, torch-npu, model,
  dataset, evaluator, or installed libraries.
- Reuse the already verified Experiment-11 environment, inputs, and cache root.
  Do not delete the old decode cache and do not create a fresh cache root.
- The existing normal graph keeps its old subdirectory. The new PSE graph uses
  one distinct subdirectory under the same decode cache root. Its first run
  must compile once; its immediate replay must load warm.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Use one free 310P. Never terminate another user's process. Never fall back to
  CPU or CUDA for inference.
- Normal logs, output artifacts, PID files, and evaluator files under the run
  root are allowed. Do not create a Markdown report.
- If a phase fails, stop the chain and reply directly to Luka in plain text.
  Give the phase, exact command, exit code, first causal error, last unmatched
  phase marker, and artifact paths. Do not propose or apply a source change.

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
  b57f7418bbaebbc922654136ff603a4f1e5bdb07 HEAD
test -f \
  11_mineru_2_5_pro_inference/production_decode_position_sweep.py
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_PSE_BOUNDARY_CHAIN.md
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
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the_1651_images
export OMNIDOCBENCH_REPO=/absolute/path/to/opendatalab/OmniDocBench
export CACHE_ROOT=/home/lukaiv/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/310p_cache_rebuild_b32_kv4096_ab925dd51fbe_20260904T095205
```

Use another cache root only if the exact verified prior root above does not
exist and the earlier report identifies its replacement. Do not invent one.

```bash
set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$MODEL_DIR/.msc"
test -f "$MODEL_DIR/.mv"
test -f "$MODEL_DIR/configuration.json"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$OMNIDOCBENCH_REPO/.git"
test -d "$CACHE_ROOT/production_increfa_real_nz_compile"
test -d "$CACHE_ROOT/vision_prefill_b1_fp16"
test -d "$CACHE_ROOT/text_prefill_packed_fp16"

"$PYTHON_BIN" \
  17_mineru_vllm_ascend_baseline/verify_310p_artifacts.py \
  --model-dir "$MODEL_DIR" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO"

PYTHONPYCACHEPREFIX=/tmp/mineru_pse_pycache \
"$PYTHON_BIN" -m unittest discover \
  -s 11_mineru_2_5_pro_inference -p 'test_*.py' -v

exec 9>"$CACHE_ROOT/mineru_pse_boundary_chain.lock"
flock -n 9
```

The artifact verifier must report `PASS`. All unit tests must pass.

## Phase 2: cold PSE graph and all-position sweep

```bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export CHAIN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_pse_chain_${COMMIT_SHORT}_${RUN_TAG}"
export SWEEP_ROOT="$CHAIN_ROOT/position_sweep_cold"
test ! -e "$CHAIN_ROOT"
mkdir -p "$SWEEP_ROOT"
git rev-parse HEAD >"$CHAIN_ROOT/commit.txt"

SWEEP_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/production_decode_position_sweep.py"
  --model "$MODEL_DIR"
  --device npu:0 --dtype float16
  --batch-size 32 --cache-length 4096
  --start-position 0 --end-position 4095
  --increfa-length-mode pse_sentinel_310p
  --cache-dir "$CACHE_ROOT/production_increfa_real_nz_compile"
  --output "$SWEEP_ROOT/result.json"
)
printf '%q ' "${SWEEP_COMMAND[@]}" >"$SWEEP_ROOT/command.sh"
printf '\n' >>"$SWEEP_ROOT/command.sh"

set +e
timeout --signal=TERM --kill-after=30s 1800s \
  "${SWEEP_COMMAND[@]}" >"$SWEEP_ROOT/run.log" 2>&1
SWEEP_STATUS=$?
set -e
printf '%s\n' "$SWEEP_STATUS" >"$SWEEP_ROOT/exit_code.txt"
test "$SWEEP_STATUS" = 0
```

The script prints one flushed start and finish for every graph call. If it
stops, report the last `MINERU_POSITION_SWEEP` line. Do not call a missing
finish "slow compilation" after the first call has already finished.

```bash
export SWEEP_ROOT
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["SWEEP_ROOT"])
x = json.loads((root / "result.json").read_text())
assert x["all_positions_completed"] is True
assert x["positions_tested"] == 4096
assert x["ascend_310p_inner_tile_size"] == 1408
assert x["exact_tile_effective_lengths"] == [1408, 2816]
assert x["compile"]["cache_was_warm"] is False
log = (root / "run.log").read_text()
for position, effective in ((1407, 1408), (2815, 2816), (4095, 4096)):
    rows = [line for line in log.splitlines()
            if f'"position": {position}' in line and
               '"event": "decode_step_graph"' in line]
    assert any('"phase": "start"' in line for line in rows)
    assert any('"phase": "finish"' in line for line in rows)
print("310P_MINERU_PSE_ALL_POSITIONS_COLD: PASS")
print(x["timing"])
print(x["compile"]["torchair_cache_dir"])
PY
```

## Phase 3: immediate warm-cache replay

Repeat the exact command with only the output paths changed:

```bash
export WARM_ROOT="$CHAIN_ROOT/position_sweep_warm"
mkdir -p "$WARM_ROOT"
WARM_COMMAND=("${SWEEP_COMMAND[@]}")
for i in "${!WARM_COMMAND[@]}"; do
  [[ "${WARM_COMMAND[$i]}" == "$SWEEP_ROOT/result.json" ]] && \
    WARM_COMMAND[$i]="$WARM_ROOT/result.json"
done
printf '%q ' "${WARM_COMMAND[@]}" >"$WARM_ROOT/command.sh"
printf '\n' >>"$WARM_ROOT/command.sh"
set +e
timeout --signal=TERM --kill-after=30s 600s \
  "${WARM_COMMAND[@]}" >"$WARM_ROOT/run.log" 2>&1
WARM_STATUS=$?
set -e
printf '%s\n' "$WARM_STATUS" >"$WARM_ROOT/exit_code.txt"
test "$WARM_STATUS" = 0

export WARM_ROOT
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
x = json.loads((Path(os.environ["WARM_ROOT"]) / "result.json").read_text())
assert x["all_positions_completed"] is True
assert x["positions_tested"] == 4096
assert x["compile"]["cache_was_warm"] is True
assert x["compile"]["compile_wrapper_s"] < 1.0
print("310P_MINERU_PSE_ALL_POSITIONS_WARM: PASS")
print(x["timing"])
print(x["compile"])
PY
```

## Phase 4: real two-page production smoke

Use Phase 3 of
`WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md`, with these required
changes and no others:

```text
SMOKE_ROOT=$CHAIN_ROOT/smoke_n2_pse
add: --local-decode-increfa-length-mode pse_sentinel_310p
add: --local-decode-diagnostic-steps 64
add: --local-decode-diagnostic-sync
add: --local-decode-diagnostic-boundary-period 1408
```

After its existing smoke gate passes, also require:

```bash
grep -q '"stage": "decode_step_state", "step": 13' "$SMOKE_ROOT/run.log"
grep -q '"boundary_rows": \[0, 1\].*"step": 13' "$SMOKE_ROOT/run.log"
grep -q '"stage": "decode_step_graph", "step": 13' "$SMOKE_ROOT/run.log"

export SMOKE_ROOT
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["SMOKE_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["local_decode_increfa_length_mode"] == "pse_sentinel_310p"
decode = summary["streaming"]["decode"]
assert decode["compile"]["cache_was_warm"] is True
assert decode["compile"]["decode_increfa_length_mode"] == "pse_sentinel_310p"
assert decode["decode_diagnostic_boundary_period"] == 1408
print("310P_MINERU_PSE_PRODUCTION_SMOKE: PASS")
print("requests", decode["request_count"])
print("pipeline_wall_s", summary["pipeline_wall_s"])
PY
```

The two active rows must have effective length 1408 at step 13, and that
step's synchronized `decode_step_graph` must have a matching finish. The smoke
must complete 2/2 pages with non-empty Markdown and content lists.

## Phase 5: full 1,651 pages and frozen evaluation

Only after Phases 2 through 4 pass, continue with Phases 4 through 10 of
`WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md` in the same persistent
coordinator session. Use its existing evaluator preflight, detached inference,
four-hour-or-longer monitoring calls, 1,651-output structural gate, detached
evaluation, TeX Live 2025, ImageMagick 7.1.1-47, and final metric table.

For its `FULL_COMMAND`, add exactly:

```text
--local-decode-increfa-length-mode pse_sentinel_310p
```

Use `RUN_ROOT=$CHAIN_ROOT/full1651_pse`. Before launch, require:

```bash
printf '%q ' "${FULL_COMMAND[@]}" | \
  grep -q -- '--local-decode-increfa-length-mode pse_sentinel_310p'
```

In the Phase-7 structural gate, add:

```python
assert summary["local_decode_increfa_length_mode"] == "pse_sentinel_310p"
assert decode["compile"]["decode_increfa_length_mode"] == "pse_sentinel_310p"
assert decode["compile"]["cache_was_warm"] is True
```

Do not stop after launching the full run or evaluator. Monitor each detached
process until its own exit-code file exists and all gates complete. Reply to
Luka with the existing Phase-10 final table plus:

```text
310P_MINERU_PSE_ALL_POSITIONS_COLD=PASS
310P_MINERU_PSE_ALL_POSITIONS_WARM=PASS
310P_MINERU_PSE_PRODUCTION_SMOKE=PASS
pse_cache_dir=
cold_sweep_timing=
warm_sweep_timing=
step13_effective_lengths=
step13_graph_elapsed_s=
```

## Issue reply

If any phase fails, reply directly in plain text:

```text
310P MINERU PSE BOUNDARY CHAIN: ISSUE
phase=
command=
exit_code=
first_causal_error=
last_unmatched_phase_marker=
chain_root=
sweep_root=
smoke_root=
full_run_root=
evaluation_root=
cache_root=
```

Do not add a proposed patch, workaround, rerun, or Markdown report.
