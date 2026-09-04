# Work-server 310P MinerU stall diagnosis

This brief is for the AI agent on Luka's Atlas 310P work server. Read CLAUDE.md
and AGENTS.md first. Run all blocks in one persistent Bash or tmux coordinator.

## Goal

Find the exact boundary where the real two-page B32/KV4096 MinerU smoke stops.
Run stock IncreFA variants in separate timeout-safe processes, then run the exact
production pipeline with structured phase events.

Do not treat stopped progress dots as slow compilation. The last unmatched
MINERU_PHASE event=start line is the active boundary. Do not start 1,651 pages
in this brief. Report the diagnosis to Luka first.

## Code and 910B control

Commit a3c49b6d5c31eeea5e8d5c975930890eba746793:

- isolates packed-text padding from real requests while giving every padding
  query at least one finite attention key;
- rejects invalid or fully masked decode rows before graph entry;
- prints flushed cache-wrapper and graph-first-call start/finish events;
- lets increfa_contract_probe.py run one variant per process.

The packed-text source hash changed, so corrected text graphs get new cache keys.
The vision and decode graph bodies did not change.

The same source passed on one 910B2:

~~~text
31 CPU tests passed
stock eager IncreFA B32/KV4096/14Q/2KV, valid length 1395:
  0.0139 s, cosine 0.99999982, max abs 0.0001526
two-page real pipeline:
  2/2 pages, 34/34 request token streams exact, 2/2 pages byte identical
  decode preflight: all 32 rows have 1395 valid keys
  decode first call: 0.294 s
  corrected text seq1024 first call: 46.404 s
  corrected text seq256 first call: 42.372 s
  pipeline wall: 99.346 s
fresh-process cache replay:
  pipeline wall: 12.274 s
  text graph first calls: 0.855 s and 0.845 s, cache_was_warm=true
  generation-trace SHA-256 identical
~~~

These are 910B controls, not 310P timing targets.

## Rules

- The checkout is pull-only. Do not edit tracked files, create a branch, commit,
  push, reset, stash, or discard changes.
- Do not modify /vllm-workspace, packages, CANN, TorchAir, torch-npu, model,
  corpus, or evaluator.
- Use the verified Experiment 11 environment and one free 310P. No CPU or CUDA
  fallback.
- Set VLLM_WORKER_MULTIPROC_METHOD=spawn before importing torch-npu.
- Preserve every old run and cache. Do not delete caches.
- Stop only the exact process launched by this agent. Never use pkill, killall,
  or a broad process match.
- Run every IncreFA variant in a fresh process with its own timeout.
- Do not write a Markdown report. Reply to Luka in plain text. On failure,
  report the issue; do not edit source or propose a patch.

## Phase 1: stop the exact old process and pull

Set these from the existing run. Do not guess:

~~~bash
export STALLED_PID=<exact_pid_launched_by_this_agent>
export STALLED_RUN_ROOT=/absolute/path/to/the_existing_stalled_run
test -d "$STALLED_RUN_ROOT"

if [[ -r "/proc/$STALLED_PID/cmdline" ]]; then
  cmdline="$(tr '\0' ' ' <"/proc/$STALLED_PID/cmdline")"
  printf '%s\n' "$cmdline"
  [[ "$cmdline" == *11_mineru_2_5_pro_inference* ]]
  ps -p "$STALLED_PID" -o pid,ppid,lstart,etime,stat,%cpu,%mem,args
  kill -TERM "$STALLED_PID"
  for _ in $(seq 1 30); do
    kill -0 "$STALLED_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$STALLED_PID" 2>/dev/null; then kill -KILL "$STALLED_PID"; fi
  ! kill -0 "$STALLED_PID" 2>/dev/null
else
  printf 'old_smoke_already_exited=%s\n' "$STALLED_PID"
fi

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
  a3c49b6d5c31eeea5e8d5c975930890eba746793 HEAD
test -f 11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_STALL_DIAGNOSIS.md
git rev-parse HEAD
~~~

Untracked artifacts are allowed. Stop if tracked changes prevent the pull.

## Phase 2: restore the verified environment

Source the same server-owned CANN activation used by the prior Experiment 11
attempt. Source it before set -u. Then set:

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHON_BIN=/absolute/path/to/the_verified_exp11_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the_1651_images
export CACHE_ROOT=/absolute/path/to/the_current_stalled_310p_cache_root
~~~

CACHE_ROOT must be the cache used by the current failed attempt, not an older
pre-filler cache.

~~~bash
set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$CACHE_ROOT"

"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", torchair.__file__)
print("npu_available", torch.npu.is_available())
assert torch.npu.is_available()
PY

PYTHONPYCACHEPREFIX=/tmp/mineru_diag_pycache \
"$PYTHON_BIN" -m unittest discover \
  -s 11_mineru_2_5_pro_inference -p 'test_*.py' -v
~~~

All tests must pass before NPU probes.

## Phase 3: independent stock IncreFA probes

~~~bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export DIAG_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_stall_diag_${COMMIT_SHORT}_${RUN_TAG}"
test ! -e "$DIAG_ROOT"
mkdir -p "$DIAG_ROOT/increfa"
git rev-parse HEAD >"$DIAG_ROOT/commit.txt"
printf 'cache_root=%s\nold_run_root=%s\n' \
  "$CACHE_ROOT" "$STALLED_RUN_ROOT" >"$DIAG_ROOT/inputs.txt"

set +e
for variant in \
  mask_b1 \
  mask_b1_actual_full \
  mask_bn \
  mha_mask_b1
do
  start_epoch="$(date +%s)"
  timeout --signal=TERM --kill-after=30s 120s \
    "$PYTHON_BIN" \
    11_mineru_2_5_pro_inference/increfa_contract_probe.py \
      --device npu:0 \
      --batch-size 32 --cache-length 4096 \
      --num-heads 14 --num-kv-heads 2 --head-dim 64 \
      --valid-length 1395 --variant "$variant" \
      --output "$DIAG_ROOT/increfa/$variant.json" \
      >"$DIAG_ROOT/increfa/$variant.log" 2>&1
  status=$?
  elapsed="$(( $(date +%s) - start_epoch ))"
  printf '%s\n' "$status" >"$DIAG_ROOT/increfa/$variant.exit_code.txt"
  printf '%s\n' "$elapsed" >"$DIAG_ROOT/increfa/$variant.elapsed_s.txt"
  printf 'variant=%s status=%s elapsed_s=%s\n' "$variant" "$status" "$elapsed"
  cat "$DIAG_ROOT/increfa/$variant.log"
  if pgrep -af 'increfa_contract_probe.py' | grep -F -- "$variant"; then
    echo "Probe child survived timeout: $variant" >&2
    exit 2
  fi
  npu-smi info
done
set -e
~~~

Interpret only as follows:

- mask_b1 passes: the exact stock GQA IncreFA call is not the standalone fault.
- only mask_b1_actual_full passes: this 310P stack needs explicit physical
  lengths.
- GQA variants fail but mha_mask_b1 passes: the fault is GQA-specific.
- all pass: isolate the full cached decode graph next.

Do not change production code from this matrix.

## Phase 4: real two-page run with the existing cache

The corrected text graphs get new source-hash directories automatically. Reuse
the current vision and decode caches.

~~~bash
export REUSE_ROOT="$DIAG_ROOT/existing_cache_smoke_n2"
mkdir -p "$REUSE_ROOT"

REUSE_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$REUSE_ROOT/output"
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

printf '%q ' "${REUSE_COMMAND[@]}" >"$REUSE_ROOT/command.sh"
printf '\n' >>"$REUSE_ROOT/command.sh"
set +e
start_epoch="$(date +%s)"
timeout --signal=TERM --kill-after=30s 1200s \
  "${REUSE_COMMAND[@]}" >"$REUSE_ROOT/run.log" 2>&1
REUSE_STATUS=$?
REUSE_ELAPSED="$(( $(date +%s) - start_epoch ))"
set -e
printf '%s\n' "$REUSE_STATUS" >"$REUSE_ROOT/exit_code.txt"
printf '%s\n' "$REUSE_ELAPSED" >"$REUSE_ROOT/elapsed_s.txt"
grep '^MINERU_PHASE ' "$REUSE_ROOT/run.log" >"$REUSE_ROOT/phase_events.jsonl" || true
tail -n 100 "$REUSE_ROOT/run.log"
~~~

Interpret the last phase:

- decode_cache_wrapper finish, then decode_graph_first_call start without finish:
  the full compiled decode invocation did not return. If its OM existed before
  the run, this is replay, not packed text.
- decode_graph_first_call finish: decode replay completed.
- packed_text_graph_first_call start without finish: the event names the exact
  text bucket, member lengths, real tokens, and padding tokens.
- page_pipeline finish plus exit 0: the stream drained.

If REUSE_STATUS is 0, skip Phase 5. If it stopped anywhere other than
decode_graph_first_call, also skip Phase 5 and report that exact event.

## Phase 5: real pipeline with only a new decode cache

Run only if mask_b1 passed and Phase 4 stopped after decode_graph_first_call
start with no matching finish. This keeps the real production pipeline and the
same vision/text caches. It tests only whether the saved decode graph is damaged
or incompatible.

~~~bash
export NEW_DECODE_ROOT="$DIAG_ROOT/new_decode_cache_smoke_n2"
export NEW_DECODE_CACHE="$DIAG_ROOT/new_decode_cache"
mkdir -p "$NEW_DECODE_ROOT" "$NEW_DECODE_CACHE"

NEW_DECODE_COMMAND=("${REUSE_COMMAND[@]}")
for i in "${!NEW_DECODE_COMMAND[@]}"; do
  if [[ "${NEW_DECODE_COMMAND[$i]}" == "$REUSE_ROOT/output" ]]; then
    NEW_DECODE_COMMAND[$i]="$NEW_DECODE_ROOT/output"
  elif [[ "${NEW_DECODE_COMMAND[$i]}" == "$CACHE_ROOT/production_increfa_real_nz_compile" ]]; then
    NEW_DECODE_COMMAND[$i]="$NEW_DECODE_CACHE"
  fi
done

printf '%q ' "${NEW_DECODE_COMMAND[@]}" >"$NEW_DECODE_ROOT/command.sh"
printf '\n' >>"$NEW_DECODE_ROOT/command.sh"
set +e
start_epoch="$(date +%s)"
timeout --signal=TERM --kill-after=30s 1200s \
  "${NEW_DECODE_COMMAND[@]}" >"$NEW_DECODE_ROOT/run.log" 2>&1
NEW_DECODE_STATUS=$?
NEW_DECODE_ELAPSED="$(( $(date +%s) - start_epoch ))"
set -e
printf '%s\n' "$NEW_DECODE_STATUS" >"$NEW_DECODE_ROOT/exit_code.txt"
printf '%s\n' "$NEW_DECODE_ELAPSED" >"$NEW_DECODE_ROOT/elapsed_s.txt"
grep '^MINERU_PHASE ' "$NEW_DECODE_ROOT/run.log" \
  >"$NEW_DECODE_ROOT/phase_events.jsonl" || true
find "$NEW_DECODE_CACHE" -type f \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' \
  | sort >"$NEW_DECODE_ROOT/decode_cache_inventory.txt"
tail -n 100 "$NEW_DECODE_ROOT/run.log"
~~~

Do not create another cache after this.

## Phase 6: validate a successful output

Set SUCCESS_ROOT to the successful Phase 4 or Phase 5 root.

~~~bash
export SUCCESS_ROOT=<absolute_successful_root>
test "$(cat "$SUCCESS_ROOT/exit_code.txt")" = 0

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["SUCCESS_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
decode = summary["streaming"]["decode"]
assert summary["completed"] == 2 and summary["failed"] == 0
assert decode["request_count"] == 34
assert decode["initial_inactive_filler_rows"] == 30
assert decode["idle_rows_with_ready_work"] == 0
assert len(list((root / "output/predictions").glob("*.md"))) == 2
print("310P_MINERU_DIAGNOSTIC_SMOKE: PASS")
print("pipeline_wall_s", summary["pipeline_wall_s"])
print("decode_first_call_s", decode["compiled_first_call_s"])
print("decode_cache", decode["compile"]["torchair_cache_dir"])
print("text_compile_records", summary["local_compiled_text_prefill"]["compile_records"])
PY

export REFERENCE_ROOT="$DIAG_ROOT/frozen_reference"
mkdir -p "$REFERENCE_ROOT"
tar -xzf \
  11_mineru_2_5_pro_inference/references/serving_streaming_1651_ae4c947c/streaming.tar.gz \
  -C "$REFERENCE_ROOT"
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/compare_generation_traces.py \
  "$REFERENCE_ROOT/output" "$SUCCESS_ROOT/output" \
  --first-pages 2 --allow-table-image-placeholders \
  --output "$SUCCESS_ROOT/reference_compare.json"
~~~

## Plain-text reply to Luka

Do not create a report file. Reply with:

1. commit, device ID, torch, torch-npu, TorchAir, and CANN versions;
2. exit code and elapsed time for every independent IncreFA variant;
3. max abs, mean abs, and cosine for each successful variant;
4. existing-cache smoke exit code and elapsed time;
5. ordered MINERU_PHASE lines through the last start/finish pair;
6. the exact last unmatched start event, if any;
7. the complete decode cache-position and valid-key-count lists;
8. whether the old decode cache was warm and whether first replay returned;
9. if Phase 5 ran, its status, cache inventory, and last phase;
10. if a smoke passed, page/request/token counts and reference comparison;
11. absolute paths to all logs, JSON, inventories, and DIAG_ROOT.

Then stop and wait for Luka.
