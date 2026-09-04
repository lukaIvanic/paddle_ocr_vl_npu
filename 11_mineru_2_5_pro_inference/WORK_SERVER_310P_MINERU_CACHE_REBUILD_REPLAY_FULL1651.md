# Work-server 310P MinerU cache rebuild, replay, and full run

This brief is for the AI agent on Luka's Atlas 310P work server. Read
`CLAUDE.md` and `AGENTS.md` first. Run every shell block in Bash.

## Goal

Replace the stalled Experiment 11 compile attempt with a clean, isolated cache
build. Run the same two OmniDocBench pages twice. The first process must build
the cache. The second process must load the same cache without changing it and
must reproduce the first process exactly. Only then run all 1,651 pages and the
frozen OmniDocBench evaluation.

Do not simulate a production batch. Use the real B32, KV4096 serving graph and
the existing real-row filler fix.

## Why this run is required

The prior 310P process spent hours inside Python-side graph tracing. It had not
submitted the decode graph to GE. AICore stayed at 0 percent, no compiler child
was active, and the cache stopped changing. Do not classify that state as a
normal cold compile.

The same test was run on one Ascend 910B2 with a completely new cache root:

```text
cold process exit: 0
cold completed pages: 2/2
cold requests: 34
cold generated tokens: 3303
cold process wall: 329.33 s
cold pipeline wall: 271.324535 s
cold decode first call: 16.512213 s
decode cache lock-to-OM: 3.088 s
cache artifacts: 9 OM files and 9 compiled modules
reference comparison: 34/34 token-exact, 2/2 byte-identical pages

fresh-process replay exit: 0
replay process wall: 65.61 s
replay pipeline wall: 11.998200 s
replay decode first call: 0.286108 s
replay cache inventory changes: 0
replay comparison: identical trace SHA-256, 0 changed pages
```

Those numbers are 910B2 evidence, not 310P performance targets. They prove that
the code creates cache artifacts automatically and loads them in a new process.

## Rules

- The repository is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard another person's changes.
- Do not edit `/vllm-workspace`, installed frameworks, shared libraries, the
  model, the dataset, or the evaluator.
- Do not install or replace Python packages, PyTorch, torch-npu, TorchAir,
  CANN, drivers, firmware, TeX Live, ImageMagick, or Ghostscript.
- This Experiment 11 pipeline does not use vLLM or vLLM-Ascend.
- Keep the stalled run root, old cache root, and logs. Do not delete them.
- Stop only the exact stalled process that this agent launched. Never use
  `pkill`, `killall`, or a broad command match. Never stop another user's job.
- Use one free physical Atlas 310P device. Do not fall back to CPU or CUDA.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Run all phases from one persistent Bash or tmux coordinator.
- Long inference and evaluation processes must use `nohup` and `setsid`.
- Keep monitoring until each exit-code file exists.
- Do not create a Markdown report on disk. Reply to Luka in plain text.
- If a phase fails, report the issue. Do not edit source or propose a patch.

## Phase 1: update and verify the checkout

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
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_CACHE_REBUILD_REPLAY_FULL1651.md
git rev-parse HEAD
```

Untracked run artifacts are allowed. Stop if tracked changes prevent the pull.

## Phase 2: restore the verified server environment

Source the same server-owned CANN activation used by the prior Experiment 11
attempt. Source it before `set -u`. Select one free physical 310P device.

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
```

Use the already verified paths from the prior attempt:

```bash
export PYTHON_BIN=/absolute/path/to/mineru_custom_exp11_310p_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the/1651/images
export OMNIDOCBENCH_REPO=/absolute/path/to/opendatalab/OmniDocBench
export OLD_CACHE_ROOT=/absolute/path/to/the/stalled/exp11/310p/cache
export OMNIDOCBENCH_EVAL_PYTHON=/absolute/path/to/the/existing/evaluator/python
```

Validate the environment and artifacts:

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
test -d "$OLD_CACHE_ROOT"
test -x "$OMNIDOCBENCH_EVAL_PYTHON"

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

The verifier must finish with `status: PASS`. Required identities:

```text
model manifest SHA-256: 5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea
model.safetensors SHA-256: abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0
OmniDocBench.json SHA-256: a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496
1651-image manifest SHA-256: 34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe
evaluator commit: 2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

## Phase 3: stop the exact stalled process

Set `STALLED_PID` to the PID that this agent launched. Set `STALLED_RUN_ROOT`
to that run's existing root. Do not guess the PID.

```bash
export STALLED_PID=<exact_pid_from_the_prior_attempt>
export STALLED_RUN_ROOT=/absolute/path/to/the/stalled/run/root
test -d "$STALLED_RUN_ROOT"
```

If the PID still exists, the command line must name the Experiment 11 runner
and the stalled run root. If it does not, stop this handoff and report the
mismatch. If the prior process already exited, record that fact and continue.
Otherwise, stop that exact PID:

```bash
if [[ -r "/proc/$STALLED_PID/cmdline" ]]; then
  tr '\0' ' ' <"/proc/$STALLED_PID/cmdline"
  ps -p "$STALLED_PID" -o pid,ppid,lstart,etime,stat,%cpu,%mem,args
  cmdline="$(tr '\0' ' ' <"/proc/$STALLED_PID/cmdline")"
  [[ "$cmdline" == *11_mineru_2_5_pro_inference* ]]
  [[ "$cmdline" == *"$STALLED_RUN_ROOT"* ]]
  kill -TERM "$STALLED_PID"
  for _ in $(seq 1 60); do
    kill -0 "$STALLED_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$STALLED_PID" 2>/dev/null; then
    kill -KILL "$STALLED_PID"
  fi
  ! kill -0 "$STALLED_PID" 2>/dev/null
else
  printf 'stalled_pid_already_exited=%s\n' "$STALLED_PID"
fi
```

Do not remove `STALLED_RUN_ROOT` or `OLD_CACHE_ROOT`.

## Phase 4: create one isolated cache and run root

```bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export CHAIN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_cache_rebuild_${COMMIT_SHORT}_${RUN_TAG}"
export CACHE_ROOT="$WORK_SERVER_REPO/.runtime_cache/11_mineru_2_5_pro_inference/310p_cache_rebuild_b32_kv4096_${COMMIT_SHORT}_${RUN_TAG}"
export COLD_ROOT="$CHAIN_ROOT/cold_smoke_n2"
export REPLAY_ROOT="$CHAIN_ROOT/cache_replay_n2"
test ! -e "$CHAIN_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$COLD_ROOT" "$REPLAY_ROOT" "$CACHE_ROOT"
git rev-parse HEAD >"$CHAIN_ROOT/commit.txt"
printf 'old_cache_root=%s\nnew_cache_root=%s\n' \
  "$OLD_CACHE_ROOT" "$CACHE_ROOT" >"$CHAIN_ROOT/cache_roots.txt"

exec 9>"$CACHE_ROOT/cache_rebuild.lock"
flock -n 9
```

Keep file descriptor 9 open in the coordinator until full inference exits.
Do not let another process write to this cache root.

## Phase 5: launch the cold two-page smoke

```bash
COLD_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$COLD_ROOT/output"
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

printf '%q ' "${COLD_COMMAND[@]}" >"$COLD_ROOT/command.sh"
printf '\n' >>"$COLD_ROOT/command.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf '/usr/bin/time -f %%e -o %q ' "$COLD_ROOT/process_wall_s.txt"
  printf '%q ' "${COLD_COMMAND[@]}"
  printf ' >%q 2>&1\n' "$COLD_ROOT/run.log"
  printf 'status=$?\n'
  printf 'printf "%%s\\n" "$status" >%q\n' "$COLD_ROOT/exit_code.txt"
  printf 'exit "$status"\n'
} >"$COLD_ROOT/run.sh"
chmod +x "$COLD_ROOT/run.sh"

nohup setsid "$COLD_ROOT/run.sh" \
  </dev/null >"$COLD_ROOT/launcher.log" 2>&1 &
printf '%s\n' "$!" >"$COLD_ROOT/pid.txt"
printf 'COLD_LOG=%s\nCOLD_PID=%s\n' \
  "$COLD_ROOT/run.log" "$(cat "$COLD_ROOT/pid.txt")"
```

Send the absolute cold log and cache-root paths to Luka immediately.

## Phase 6: monitor cold compilation with a stall gate

Monitor the detached process until `exit_code.txt` exists. Request at least a
four-hour tool timeout. Reattach if the tool returns early.

The 30-minute stall gate applies only when all of these conditions remain true:

- AICore stays at 0 percent.
- No CANN or GE compiler child is active.
- Cache file count, total bytes, and newest modification time do not change.
- The run log does not change.

Active cache growth resets the timer. A long compile that continues to create
artifacts is not this failure mode.

```bash
set -euo pipefail
last_signature=''
last_progress_epoch="$(date +%s)"
while [[ ! -s "$COLD_ROOT/exit_code.txt" ]]; do
  pid="$(cat "$COLD_ROOT/pid.txt")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Cold smoke exited without writing exit_code.txt.' >&2
    tail -n 160 "$COLD_ROOT/run.log" >&2 || true
    exit 2
  fi

  cache_count="$(find "$CACHE_ROOT" -type f | wc -l)"
  cache_bytes="$(find "$CACHE_ROOT" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
  cache_newest="$(find "$CACHE_ROOT" -type f -printf '%T@\n' | sort -nr | head -n 1)"
  log_size="$(stat -c %s "$COLD_ROOT/run.log" 2>/dev/null || printf 0)"
  signature="$cache_count:$cache_bytes:$cache_newest:$log_size"
  now="$(date +%s)"
  if [[ "$signature" != "$last_signature" ]]; then
    last_signature="$signature"
    last_progress_epoch="$now"
  fi

  date -Ins
  ps -p "$pid" -o pid,etime,stat,%cpu,%mem,args --no-headers || true
  ps --ppid "$pid" -o pid,ppid,etime,stat,%cpu,%mem,args || true
  printf 'cache_files=%s cache_bytes=%s cache_newest=%s log_bytes=%s idle_s=%s\n' \
    "$cache_count" "$cache_bytes" "$cache_newest" "$log_size" \
    "$((now - last_progress_epoch))"
  find "$CACHE_ROOT" -type f \
    \( -name '*.om' -o -name compiled_module -o -name '*.idx' -o -name '*.lock' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort | tail -30
  grep -E \
    'cache_compile|GE|graph|compile|\[setup\]|\[stream\]|Traceback|ERROR|ACL_ERROR|E[0-9]{5}' \
    "$COLD_ROOT/run.log" | tail -40 || true
  npu-smi info

  if (( now - last_progress_epoch >= 1800 )); then
    printf '%s\n' \
      'Cold smoke made no cache or log progress for 30 minutes.' \
      'Treat this as a tracing stall, not normal compilation.' >&2
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    exit 3
  fi
  sleep 60
done

COLD_STATUS="$(cat "$COLD_ROOT/exit_code.txt")"
printf 'COLD_EXIT=%s\n' "$COLD_STATUS"
test "$COLD_STATUS" = 0
```

If the stall gate fires, stop this handoff and use the issue reply at the end.

## Phase 7: validate the cold output and cache

```bash
export COLD_ROOT CACHE_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["COLD_ROOT"])
cache = Path(os.environ["CACHE_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
decode = summary["streaming"]["decode"]
decode_cache = (
    cache
    / "production_increfa_real_nz_compile"
    / "mineru_increfa_attention_decode_nz_rotary_npu_apply_bs32_cache4096"
)

assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["batch_size"] == 32
assert summary["streaming_page_window"] == 2
assert summary["local_decode_attention"] == "increfa"
assert decode["request_count"] == 34
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["initial_inactive_filler_rows"] == 30
assert decode["initial_filler_source_slot"] is not None
assert decode["idle_rows_with_ready_work"] == 0
assert len(list((root / "output/predictions").glob("*.md"))) == 2
assert len(list((root / "output/content_lists").glob("*.json"))) == 2
assert list(decode_cache.rglob("*.om"))
assert list(decode_cache.rglob("compiled_module"))
assert list((cache / "vision_prefill_b1_fp16").rglob("*.om"))
assert list((cache / "text_prefill_packed_fp16").rglob("*.om"))
print("310P_MINERU_COLD_CACHE_SMOKE: PASS")
print("pipeline_wall_s", summary["pipeline_wall_s"])
print("decode_first_call_s", decode["compiled_first_call_s"])
print("decode_cache", decode_cache)
print("om_files", len(list(cache.rglob("*.om"))))
print("compiled_modules", len(list(cache.rglob("compiled_module"))))
PY

find "$CACHE_ROOT" -type f -printf '%P %s %T@\n' | sort \
  >"$REPLAY_ROOT/cache_inventory_before.txt"
```

## Phase 8: run a fresh-process cache replay

Use the same command and cache root. Change only the output directory.

```bash
REPLAY_COMMAND=("${COLD_COMMAND[@]}")
for i in "${!REPLAY_COMMAND[@]}"; do
  if [[ "${REPLAY_COMMAND[$i]}" == "$COLD_ROOT/output" ]]; then
    REPLAY_COMMAND[$i]="$REPLAY_ROOT/output"
  fi
done
printf '%q ' "${REPLAY_COMMAND[@]}" >"$REPLAY_ROOT/command.sh"
printf '\n' >>"$REPLAY_ROOT/command.sh"

set +e
/usr/bin/time -f '%e' -o "$REPLAY_ROOT/process_wall_s.txt" \
  "${REPLAY_COMMAND[@]}" >"$REPLAY_ROOT/run.log" 2>&1
REPLAY_STATUS=$?
set -e
printf '%s\n' "$REPLAY_STATUS" >"$REPLAY_ROOT/exit_code.txt"
test "$REPLAY_STATUS" = 0

find "$CACHE_ROOT" -type f -printf '%P %s %T@\n' | sort \
  >"$REPLAY_ROOT/cache_inventory_after.txt"
diff -u \
  "$REPLAY_ROOT/cache_inventory_before.txt" \
  "$REPLAY_ROOT/cache_inventory_after.txt" \
  >"$REPLAY_ROOT/cache_inventory.diff"

"$PYTHON_BIN" \
  11_mineru_2_5_pro_inference/compare_generation_traces.py \
  "$COLD_ROOT/output" "$REPLAY_ROOT/output" \
  --first-pages 2 --allow-table-image-placeholders \
  --output "$REPLAY_ROOT/cold_compare.json"
```

Validate cache reuse and output parity:

```bash
export COLD_ROOT REPLAY_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

cold = Path(os.environ["COLD_ROOT"])
replay = Path(os.environ["REPLAY_ROOT"])
cold_summary = json.loads((cold / "output/run_summary_shard_00.json").read_text())
replay_summary = json.loads((replay / "output/run_summary_shard_00.json").read_text())
comparison = json.loads((replay / "cold_compare.json").read_text())
cold_first = float(cold_summary["streaming"]["decode"]["compiled_first_call_s"])
replay_first = float(replay_summary["streaming"]["decode"]["compiled_first_call_s"])

assert replay_summary["completed"] == 2
assert replay_summary["failed"] == 0
assert replay_summary["skipped"] == 0
assert not comparison["changed_pages"]
assert not comparison["differences"]
assert comparison["reference_requests"] == comparison["candidate_requests"] == 34
assert comparison["reference_generated_tokens"] == comparison["candidate_generated_tokens"]
assert replay_first < cold_first
assert (replay / "cache_inventory.diff").stat().st_size == 0
print("310P_MINERU_CACHE_REPLAY: PASS")
print("cold_decode_first_call_s", cold_first)
print("replay_decode_first_call_s", replay_first)
print("replay_pipeline_wall_s", replay_summary["pipeline_wall_s"])
print("trace_sha256", comparison["candidate_trace_sha256"])
PY
```

Do not start the full run unless Phases 7 and 8 both print `PASS`.

## Phase 9: verify the frozen evaluator

```bash
export OMNIDOCBENCH_EVAL_TOOLS_ROOT="${OMNIDOCBENCH_EVAL_TOOLS_ROOT:-$WORK_SERVER_REPO/.runtime_cache/omnidocbench_eval/tools}"
export OMNIDOCBENCH_EVALUATOR_ROOT="$OMNIDOCBENCH_REPO"

(
  set -euo pipefail
  source "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
  test "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" rev-parse HEAD)" = \
    2b161d010d2e3aff77a0edef359ea3a6411d23cd
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

Do not install or repair the evaluator if this gate fails.

## Phase 10: launch and monitor all 1,651 pages

```bash
export RUN_ROOT="$CHAIN_ROOT/full1651"
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
printf 'FULL_LOG=%s\nFULL_PID=%s\n' \
  "$RUN_ROOT/run.log" "$(cat "$RUN_ROOT/pid.txt")"
```

Send the absolute full-run log path to Luka. Monitor until the exit-code file
exists. Request at least a four-hour tool timeout and reattach as needed:

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
    test ! -f "$RUN_ROOT/output/progress_shard_00.jsonl" || \
      printf 'completed_records=%s/1651\n' \
        "$(wc -l <"$RUN_ROOT/output/progress_shard_00.jsonl")"
    grep -E \
      '\[setup\]|\[warmup\]|\[stream\] completed=|\[summary\]|Traceback|ERROR|ACL_ERROR|E[0-9]{5}' \
      "$RUN_ROOT/run.log" | tail -30 || true
    npu-smi info
  fi
  sleep 60
done
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
```

Validate the full output:

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

## Phase 11: launch and monitor evaluation

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
printf 'EVAL_LOG=%s\nEVAL_PID=%s\n' \
  "$EVAL_ROOT/launcher.log" "$(cat "$EVAL_ROOT/pid.txt")"
```

Monitor until evaluation writes its exit-code file. Do not stop at prediction
preparation. Wait for page matching, CDM, TEDS, and evaluator exit.

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
test "$(cat "$EVAL_ROOT/exit_code.txt")" = 0
test "$(cat "$RUN_ROOT/evaluation/exit_code.txt")" = 0
```

## Phase 12: print the final result

Run this in the foreground. Paste the output directly to Luka:

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

print("310P MINERU CACHE REBUILD REPLAY FULL1651 EVALUATION: PASS")
print("| Metric | 310P | 910B2 custom | Delta |")
print("|---|---:|---:|---:|")
for name, base in reference.items():
    value = candidate[name]
    print(f"| {name} | {value:.6f} | {base:.6f} | {value - base:+.6f} |")
print(f"run_root={root}")
print(f"cache_root={os.environ['CACHE_ROOT']}")
print(f"pages={run['completed']} failed={run['failed']} skipped={run['skipped']}")
print(f"requests={run['generation_trace']['requests']}")
print(f"pipeline_wall_s={run['pipeline_wall_s']:.6f}")
print(f"page_match_fallbacks={sum(v['count'] for v in stage['page_match']['fallbacks'].values())}")
PY
```

Also report the repository commit, physical NPU ID and product, CANN, Python,
PyTorch, torch-npu, TorchAir, Transformers, evaluator Python, evaluator commit,
TeX Live, ImageMagick, and Ghostscript versions. Report evaluator timeouts,
exceptions, and fallbacks even when evaluation succeeds.

## Issue reply

If any phase fails, reply in plain text with:

```text
310P MINERU CACHE REBUILD REPLAY FULL1651 EVALUATION: ISSUE
phase=
command=
exit_code=
first_causal_error=
stalled_run_root=
old_cache_root=
chain_root=
new_cache_root=
cold_root=
replay_root=
full_run_root=
evaluation_root=
last_cache_signature=
paths_checked=
```

Do not add a proposed source change, workaround, rerun, or report file.
