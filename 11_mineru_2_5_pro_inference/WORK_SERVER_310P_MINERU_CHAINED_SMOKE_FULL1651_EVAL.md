# Work-server 310P MinerU smoke, full run, and evaluation

This is one continuous execution brief for the AI agent on Luka's Atlas 310P
work server. Read `CLAUDE.md` and `AGENTS.md` first. Run every shell block in
Bash.

## Goal

Run the corrected custom MinerU2.5-Pro pipeline on two OmniDocBench v1.6 pages.
If, and only if, that smoke passes, run all 1,651 pages, wait until inference
has fully exited, and evaluate the unchanged predictions with the server's
frozen TeX Live 2025 and ImageMagick 7.1.1-47 runtime.

Do not stop after launching a background process. Keep this task open until the
full inference and evaluation have both finished, or until one of them fails.

Run Phases 1 through 10 from one persistent Bash coordinator session so the
selected NPU, CANN environment, path variables, and cache lock remain active.
If the shell tool does not preserve a session between calls, open a dedicated
tmux shell and send each phase to that same shell. The full inference and
evaluation still run as separate detached children of that coordinator.

The fix under test replaces zero-KV inactive B32 rows with copies of one valid
real row. The scheduler copies KV, token, cache position, and rope delta before
the first IncreFA call. It retains valid state in an inactive row until a new
request replaces that row.

The same two-page command passed on one 910B2 at commit `bf8f7673` with the
existing warm caches:

```text
completed=2
failed=0
requests=34
initial_inactive_filler_rows=30
layout_token_exact=2/2
recognition_token_exact=32/32
byte_identical_pages=2/2
```

That 910B result checks the source and graph contract. It does not prove 310P
compatibility. This handoff must run the real 310P test.

## Execution rules

- The repository is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard another person's changes.
- Do not edit `/vllm-workspace`, installed frameworks, shared libraries, the
  model, the dataset, or the OmniDocBench evaluator.
- Do not install or replace PyTorch, torch-npu, TorchAir, CANN, drivers,
  firmware, TeX Live, ImageMagick, Ghostscript, or Python packages.
- This pipeline does not use vLLM or vLLM-Ascend. Do not import or modify them.
- Reuse the already verified Experiment 11 Python environment, model, corpus,
  evaluator checkout, rendering tools, and persistent 310P graph cache.
- Do not delete or rebuild the cache merely because the source commit changed.
  The scheduler fix does not change the compiled decode graph.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The name is inherited from the
  framework, but this also prevents initialized torch-npu state from being
  forked by helper processes.
- Use one free physical Atlas 310P device. Never terminate another user's
  process. Never fall back to CPU or CUDA for inference.
- Run the two-page smoke once. Run the full inference once. Run the evaluation
  once. Preserve partial artifacts after any failure.
- Normal run scripts, PID files, logs, outputs, and evaluation artifacts under
  the run root are allowed. Do not create a separate Markdown report.
- If a phase fails, reply directly to Luka in plain text. Give the phase,
  command, exit code, first causal error, and paths. Do not propose a patch,
  workaround, or rerun.

## Long-running tool calls

Launch full inference and evaluation with `nohup` and `setsid`, as specified
below. Monitor each detached process until its exit-code file exists.

For each monitoring shell call, request a tool timeout of at least four hours:

```text
14,400 seconds
14,400,000 milliseconds
```

Use the unit accepted by the tool. If the tool imposes a smaller maximum, use
its maximum and immediately reattach to the same PID and log when it returns.
Do not treat a tool timeout as process completion. Do not return control to Luka
while inference or evaluation is still running.

## Phase 1: update and verify the pull-only checkout

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
  bf8f76737872849c4268a63e680861c8289312cd HEAD
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md
git rev-parse HEAD
```

Untracked logs and build outputs are allowed. Tracked changes are not. If
tracked changes prevent the pull, stop without modifying the checkout.

## Phase 2: reuse the verified environment and inputs

Source the same server-owned CANN activation used by the earlier 310P attempt.
Source it before enabling `set -u`. Do not source another user's project file.
Then select one free physical 310P device and set:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
```

Set these variables to the paths already found and verified on this server:

```bash
export PYTHON_BIN=/absolute/path/to/mineru_custom_exp11_310p_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the/1651/images
export OMNIDOCBENCH_REPO=/absolute/path/to/opendatalab/OmniDocBench
export CACHE_ROOT=/absolute/path/to/the/existing/exp11/310p/cache
```

Do not create a new venv or cache root. Validate the reused paths:

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
test -d "$CACHE_ROOT"

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import torch
import torch_npu
import torchair
import transformers
from mineru_vl_utils import MinerUClient
from torchair.inference import cache_compile

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", torchair.__file__)
print("transformers", transformers.__version__)
print("mineru-vl-utils", metadata.version("mineru-vl-utils"))
print("httpx-retries", metadata.version("httpx-retries"))
assert metadata.version("mineru-vl-utils") == "1.0.5"
assert metadata.version("httpx-retries") == "0.6.0"
assert torch.npu.is_available()
PY

"$PYTHON_BIN" \
  17_mineru_vllm_ascend_baseline/verify_310p_artifacts.py \
  --model-dir "$MODEL_DIR" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO"
```

The artifact verifier must finish with `status: PASS`. The required identities
are:

```text
model manifest SHA-256: 5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea
model.safetensors SHA-256: abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0
OmniDocBench.json SHA-256: a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496
1651-image manifest SHA-256: 34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe
evaluator commit: 2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

Acquire one cache lock and retain it through smoke and full inference:

```bash
exec 9>"$CACHE_ROOT/chained_smoke_full1651.lock"
flock -n 9 || {
  echo 'Another process owns the Experiment 11 310P cache.' >&2
  exit 2
}
```

## Phase 3: run the two-page smoke

```bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export CHAIN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_chained_${COMMIT_SHORT}_${RUN_TAG}"
export SMOKE_ROOT="$CHAIN_ROOT/smoke_n2"
test ! -e "$CHAIN_ROOT"
mkdir -p "$SMOKE_ROOT"
git rev-parse HEAD >"$CHAIN_ROOT/commit.txt"

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
  --token-trace --hash-model-files
  --streaming-pages --streaming-page-window 2
)

printf '%q ' "${SMOKE_COMMAND[@]}" >"$SMOKE_ROOT/command.sh"
printf '\n' >>"$SMOKE_ROOT/command.sh"
set +e
"${SMOKE_COMMAND[@]}" 2>&1 | tee "$SMOKE_ROOT/run.log"
SMOKE_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$SMOKE_STATUS" >"$SMOKE_ROOT/exit_code.txt"
test "$SMOKE_STATUS" = 0
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
assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["local_decode_attention"] == "increfa"
assert summary["batch_size"] == 32
assert summary["streaming_page_window"] == 2
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["initial_inactive_filler_rows"] == 30
assert decode["initial_filler_source_slot"] is not None
assert decode["idle_rows_with_ready_work"] == 0
assert len(list((root / "output/predictions").glob("*.md"))) == 2
assert len(list((root / "output/content_lists").glob("*.json"))) == 2
assert all(path.stat().st_size > 0 for path in
           (root / "output/predictions").glob("*.md"))
print("310P_MINERU_CHAINED_SMOKE: PASS")
print("requests", decode["request_count"])
print("initial_inactive_filler_rows", decode["initial_inactive_filler_rows"])
print("pipeline_wall_s", summary["pipeline_wall_s"])
PY
```

If the smoke fails or hangs, stop. Do not start the full run.

## Phase 4: verify the frozen evaluator before the long run

Use the existing repository-local 310P evaluator runtime when present. Do not
install or repair it:

```bash
export OMNIDOCBENCH_EVAL_TOOLS_ROOT="${OMNIDOCBENCH_EVAL_TOOLS_ROOT:-$WORK_SERVER_REPO/.runtime_cache/omnidocbench_eval/tools}"
export OMNIDOCBENCH_EVALUATOR_ROOT="$OMNIDOCBENCH_REPO"
export OMNIDOCBENCH_EVAL_PYTHON=/absolute/path/to/the/existing/evaluator/python

(
  set -euo pipefail
  source "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
  test -x "$OMNIDOCBENCH_EVAL_PYTHON"
  test "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" rev-parse HEAD)" = \
    2b161d010d2e3aff77a0edef359ea3a6411d23cd
  test -z "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" status --porcelain --untracked-files=no -- . ':!result')"
  [[ "$("$CDM_PDFLATEX" --version | head -n 1)" == *"1.40.28 (TeX Live 2025)"* ]]
  [[ "$("$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" --version | head -n 1)" == *"ImageMagick 7.1.1-47"* ]]
  test "$(gs --version)" = 9.55.0
  test -n "$("$CDM_KPSEWHICH" CJK.sty)"
  test -n "$("$CDM_KPSEWHICH" c70gkai.fd)"
  "$OMNIDOCBENCH_EVAL_PYTHON" \
    "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py" \
    --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" \
    >"$CHAIN_ROOT/evaluator_runtime_smoke.json"
  grep -q '"status": "pass"' "$CHAIN_ROOT/evaluator_runtime_smoke.json"
)
```

Ambient TeX Live 2022, ImageMagick 6, or another Ghostscript version is not an
acceptable substitute.

## Phase 5: launch the full 1,651-page inference in the background

```bash
export RUN_ROOT="$CHAIN_ROOT/full1651"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

FULL_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$RUN_ROOT/output"
  --offset 0 --limit 1651 --warmup-pages 2 --no-resume --fail-fast
  --batch-size 32 --page-batch-size 32 --global-request-stream
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
  --token-trace --hash-model-files
  --streaming-pages --streaming-page-window 32
)

printf '%q ' "${FULL_COMMAND[@]}" >"$RUN_ROOT/command.sh"
printf '\n' >>"$RUN_ROOT/command.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf '/usr/bin/time -f %%e -o %q ' "$RUN_ROOT/process_wall_s.txt"
  printf '%q ' "${FULL_COMMAND[@]}"
  printf ' >%q 2>&1\n' "$RUN_ROOT/run.log"
  printf 'status=$?\n'
  printf 'printf "%%s\\n" "$status" >%q\n' "$RUN_ROOT/exit_code.txt"
  printf 'exit "$status"\n'
} >"$RUN_ROOT/run_full.sh"
chmod +x "$RUN_ROOT/run_full.sh"

nohup setsid "$RUN_ROOT/run_full.sh" \
  </dev/null >"$RUN_ROOT/launcher.log" 2>&1 &
printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
printf 'FULL_RUN_ROOT=%s\nFULL_RUN_LOG=%s\nPID=%s\n' \
  "$RUN_ROOT" "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
```

Send the absolute `FULL_RUN_LOG` path to Luka immediately. Do not wait for the
next phase before sending that progress update.

## Phase 6: monitor full inference until it exits

Run the following as one shell tool call with a timeout of at least
14,400,000 ms. If the tool returns before `exit_code.txt` exists, run the same
block again. The inference stays detached.

```bash
set -euo pipefail
tick=0
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  tick=$((tick + 1))
  pid="$(cat "$RUN_ROOT/pid.txt")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Full inference exited without writing exit_code.txt.' >&2
    tail -n 160 "$RUN_ROOT/run.log" >&2 || true
    exit 2
  fi
  if (( tick <= 5 || tick % 5 == 0 )); then
    date -Ins
    ps -p "$pid" -o pid,etime,stat,%cpu,%mem --no-headers || true
    if [[ -f "$RUN_ROOT/output/progress_shard_00.jsonl" ]]; then
      printf 'completed_records=%s/1651\n' \
        "$(wc -l <"$RUN_ROOT/output/progress_shard_00.jsonl")"
    fi
    grep -E \
      '\[setup\]|\[warmup\]|\[stream\] completed=|\[summary\]|Traceback|ERROR|ACL_ERROR|E[0-9]{5}' \
      "$RUN_ROOT/run.log" | tail -30 || true
    npu-smi info
  fi
  sleep 60
done

FULL_STATUS="$(cat "$RUN_ROOT/exit_code.txt")"
printf 'FULL_INFERENCE_EXIT=%s\n' "$FULL_STATUS"
test "$FULL_STATUS" = 0
```

Do not infer completion from a quiet log or released NPU. Require the exit-code
file and the structural gate below.

## Phase 7: validate all 1,651 outputs

```bash
export RUN_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
output = root / "output"
summary = json.loads((output / "run_summary_shard_00.json").read_text())
decode = summary["streaming"]["decode"]
progress = [json.loads(line) for line in
            (output / "progress_shard_00.jsonl").read_text().splitlines()]
trace = [json.loads(line) for line in
         (output / "generation_trace.jsonl").read_text().splitlines()]

assert summary["completed"] == 1651
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["selected_pages"] == 1651
assert summary["batch_size"] == 32
assert summary["streaming_page_window"] == 32
assert summary["local_decode_attention"] == "increfa"
assert summary["local_decode_weight_format"] == "decode_nz"
assert summary["local_decode_rotary_impl"] == "npu_apply"
assert summary["warmup"]["executed_pages"] == 2
assert summary["streaming"]["remaining_inflight"] == 0
assert summary["streaming"]["source_closed"] is True
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["idle_rows_with_ready_work"] == 0
assert len(progress) == 1651
assert len({row["image"] for row in progress}) == 1651
assert len(trace) == summary["generation_trace"]["requests"]
assert len({row["request_id"] for row in trace}) == len(trace)
assert all(row["generated_token_ids"] for row in trace)
assert len(list((output / "predictions").glob("*.md"))) == 1651
assert len(list((output / "content_lists").glob("*.json"))) == 1651
assert not any((output / "failures").iterdir())
print("310P_MINERU_FULL1651_OUTPUT: PASS")
print("requests", len(trace))
print("hot_pages_per_s", summary["measured_group_pages_per_s"])
print("pipeline_wall_s", summary["pipeline_wall_s"])
PY
```

Do not start evaluation if this gate fails.

## Phase 8: launch evaluation in the background

The evaluation launcher must use the exact TeX Live 2025 and ImageMagick
7.1.1-47 root verified in Phase 4.

```bash
export EVAL_ROOT="$RUN_ROOT/evaluation_control"
mkdir -p "$EVAL_ROOT"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf 'cd %q\n' "$WORK_SERVER_REPO"
  printf 'export RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'export DATASET_JSON=%q\n' "$DATASET_JSON"
  printf 'export LIMIT=1651\n'
  printf 'export OMNIDOCBENCH_EVAL_TOOLS_ROOT=%q\n' "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
  printf 'export OMNIDOCBENCH_EVAL_PYTHON=%q\n' "$OMNIDOCBENCH_EVAL_PYTHON"
  printf 'export OMNIDOCBENCH_EVALUATOR_ROOT=%q\n' "$OMNIDOCBENCH_EVALUATOR_ROOT"
  printf 'bash 11_mineru_2_5_pro_inference/run_serving_accuracy.sh >%q 2>&1\n' \
    "$EVAL_ROOT/launcher.log"
  printf 'status=$?\n'
  printf 'printf "%%s\\n" "$status" >%q\n' "$EVAL_ROOT/exit_code.txt"
  printf 'exit "$status"\n'
} >"$EVAL_ROOT/run_eval.sh"
chmod +x "$EVAL_ROOT/run_eval.sh"

nohup setsid "$EVAL_ROOT/run_eval.sh" \
  </dev/null >"$EVAL_ROOT/nohup.log" 2>&1 &
printf '%s\n' "$!" >"$EVAL_ROOT/pid.txt"
printf 'EVAL_LOG=%s\nPID=%s\n' \
  "$RUN_ROOT/evaluation/run.log" "$(cat "$EVAL_ROOT/pid.txt")"
```

## Phase 9: monitor evaluation until it exits

Run this with a tool timeout of at least 14,400,000 ms. Reattach if the tool
returns early. Do not stop at `prepare_serving_eval.py`; wait for page matching,
CDM, TEDS, and the final evaluator exit.

```bash
set -euo pipefail
tick=0
while [[ ! -s "$EVAL_ROOT/exit_code.txt" ]]; do
  tick=$((tick + 1))
  pid="$(cat "$EVAL_ROOT/pid.txt")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Evaluation exited without writing exit_code.txt.' >&2
    tail -n 160 "$EVAL_ROOT/launcher.log" >&2 || true
    exit 2
  fi
  if (( tick <= 5 || tick % 5 == 0 )); then
    date -Ins
    ps -p "$pid" -o pid,etime,stat,%cpu,%mem --no-headers || true
    tail -n 40 "$EVAL_ROOT/launcher.log" || true
    test ! -f "$RUN_ROOT/evaluation/run.log" || \
      tail -n 40 "$RUN_ROOT/evaluation/run.log"
  fi
  sleep 60
done

EVAL_STATUS="$(cat "$EVAL_ROOT/exit_code.txt")"
printf 'EVALUATION_EXIT=%s\n' "$EVAL_STATUS"
test "$EVAL_STATUS" = 0
test "$(cat "$RUN_ROOT/evaluation/exit_code.txt")" = 0
```

## Phase 10: print and report the final result

Run this in the foreground and paste its output directly to Luka:

```bash
export RUN_ROOT
"$OMNIDOCBENCH_EVAL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
run = json.loads((root / "output/run_summary_shard_00.json").read_text())
metric = json.loads((
    root / "evaluation/work/result/predictions_quick_match_metric_result.json"
).read_text())
stage = json.loads((
    root / "evaluation/work/result/predictions_quick_match_stage_execution.json"
).read_text())

text_edit = float(metric["text_block"]["page"]["Edit_dist"]["ALL"])
formula_cdm = float(metric["display_formula"]["page"]["CDM"]["ALL"])
table_teds = float(metric["table"]["page"]["TEDS"]["ALL"])
structure_teds = float(metric["table"]["page"]["TEDS_structure_only"]["ALL"])
reading_edit = float(metric["reading_order"]["page"]["Edit_dist"]["ALL"])
overall = ((1.0 - text_edit) + formula_cdm + table_teds) / 3.0

reference = {
    "Hot pages/s": 0.813314141413167,
    "Overall": 95.11312625606774,
    "Text accuracy": 96.30632169786225,
    "Formula Page CDM": 96.72968076226105,
    "Table Page TEDS": 92.30337630807993,
    "Table structure Page TEDS": 95.10332546866125,
    "Reading-order edit": 0.12525933217652402,
}
candidate = {
    "Hot pages/s": float(run["measured_group_pages_per_s"]),
    "Overall": 100.0 * overall,
    "Text accuracy": 100.0 * (1.0 - text_edit),
    "Formula Page CDM": 100.0 * formula_cdm,
    "Table Page TEDS": 100.0 * table_teds,
    "Table structure Page TEDS": 100.0 * structure_teds,
    "Reading-order edit": reading_edit,
}

print("310P MINERU CHAINED SMOKE FULL1651 EVALUATION: PASS")
print("| Metric | 310P | 910B2 custom | Delta |")
print("|---|---:|---:|---:|")
for name, base in reference.items():
    value = candidate[name]
    print(f"| {name} | {value:.6f} | {base:.6f} | {value - base:+.6f} |")
print(f"run_root={root}")
print(f"pages={run['completed']} failed={run['failed']} skipped={run['skipped']}")
print(f"requests={run['generation_trace']['requests']}")
print(f"pipeline_wall_s={run['pipeline_wall_s']:.6f}")
print(f"inactive_filler_policy={run['streaming']['decode']['inactive_filler_policy']}")
print(f"page_match_fallbacks={sum(v['count'] for v in stage['page_match']['fallbacks'].values())}")
PY
```

Also report the repository commit, physical NPU ID and product, CANN, Python,
PyTorch, torch-npu, TorchAir, Transformers, evaluator Python, evaluator commit,
TeX Live, ImageMagick, and Ghostscript versions. Report evaluator timeouts,
exceptions, and fallbacks even when the evaluator exits successfully.

Do not impose token parity or an invented score threshold. Complete output,
successful evaluation, and honest reporting are the gates.

## Issue reply

If any phase fails, reply with only:

```text
310P MINERU CHAINED SMOKE FULL1651 EVALUATION: ISSUE
phase=
command=
exit_code=
first_causal_error=
chain_root=
smoke_root=
full_run_root=
evaluation_root=
cache_root=
paths_checked=
```

Do not add a proposed source change, workaround, rerun, or Markdown report.
