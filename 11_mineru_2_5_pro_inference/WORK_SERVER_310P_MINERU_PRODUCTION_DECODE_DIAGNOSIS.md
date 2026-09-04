# Work-server 310P MinerU production decode diagnosis

This brief is for the AI agent on Luka's Atlas 310P work server. Read
`CLAUDE.md` and `AGENTS.md` first. Run the real Experiment 11 two-page pipeline;
do not replace it with a standalone attention, scatter, or synthetic graph
probe.

## Goal

Locate the exact production decode step that stops on 310P. Record the full
32-row cache-position state and separate these boundaries:

1. compiled graph execution;
2. sampled-token D2H submission;
3. host control updates;
4. the wait for the previous sampled-token copy.

If the current inactive-row policy stops, repeat the same production run while
advancing inactive filler rows. This changes no active request state. Do not
start the 1,651-page run in this brief.

## Required commit and 910B control

Pull commit `65051504e94902d75a2c467eda1f19c1d13cb8f9` or a descendant. It adds
production-path diagnostics and a controlled inactive-filler position policy.
It does not change the default path.

The exact real two-page pipeline passed on one 910B2 with both policies:

- `retain`: after step 0, the two active rows had effective length 1396 and 30
  inactive filler rows remained at 1395;
- `advance`: after step 0, all 32 rows had effective length 1396;
- both runs completed 2/2 pages and 34/34 request streams;
- `generation_trace.jsonl` had the same SHA-256 in both runs;
- every prediction file was byte-identical;
- synchronized graph calls at steps 1-3 were approximately 4.3-4.7 ms.

These are 910B controls only. They do not prove the 310P behavior.

## Why the boundary-period field exists

The PaddleOCR-VL production decoder already has a 310P-specific workaround for
a masked GQA IncreFA deadlock at an exact 1280-token internal boundary. It
exposes one otherwise masked key and suppresses it with an FP16-min
`pse_shift`, which selects a different operator path without changing the
attention result.

MinerU has different head geometry: 14 query heads, 2 KV heads, head dimension
64. Therefore, 1280 is a diagnostic hypothesis, not an assumed MinerU period.
This brief logs `effective_length % 1280` and exact-boundary rows. Do not add a
PSE workaround unless the real production stop supplies boundary evidence.

## Rules

- The checkout is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard changes.
- Do not modify `/vllm-workspace`, packages, CANN, TorchAir, torch-npu, model,
  dataset, evaluator, or cache contents.
- Reuse the verified Experiment 11 environment, model, dataset, and the warm
  B32/KV4096 cache from the previous diagnosis. Do not select a fresh cache
  root and do not delete or rebuild caches.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Use one free 310P. Do not fall back to CPU or CUDA.
- Stop only a process launched by this agent after verifying its exact PID and
  command line. Never use `pkill`, `killall`, or a broad process match.
- Do not write a Markdown report or a report file. Reply to Luka in plain text.
  If a run fails, report the evidence; do not edit source or propose a patch.

## Phase 1: stop the previous exact process, if it still exists

Use the exact PID recorded by the previous attempt. Do not guess a PID. If the
process has already exited, continue.

~~~bash
export PREVIOUS_PID=<exact_pid_from_the_previous_attempt>

if [[ -r "/proc/$PREVIOUS_PID/cmdline" ]]; then
  previous_cmd="$(tr '\0' ' ' <"/proc/$PREVIOUS_PID/cmdline")"
  printf 'previous_cmd=%s\n' "$previous_cmd"
  [[ "$previous_cmd" == *11_mineru_2_5_pro_inference* ]]
  ps -p "$PREVIOUS_PID" -o pid,ppid,lstart,etime,stat,%cpu,%mem,args
  kill -TERM "$PREVIOUS_PID"
  for _ in $(seq 1 30); do
    kill -0 "$PREVIOUS_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
    kill -KILL "$PREVIOUS_PID"
  fi
  ! kill -0 "$PREVIOUS_PID" 2>/dev/null
else
  printf 'previous_process_already_exited=%s\n' "$PREVIOUS_PID"
fi
~~~

## Phase 2: pull and restore the verified environment

Source the same server-owned CANN activation used for the prior Experiment 11
attempt. Source it before `set -u`. Resolve the repository instead of
hardcoding it.

~~~bash
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
  65051504e94902d75a2c467eda1f19c1d13cb8f9 HEAD
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_PRODUCTION_DECODE_DIAGNOSIS.md
git rev-parse HEAD
~~~

Export the exact verified paths from the previous Experiment 11 attempt. Do not
substitute a new environment or cache. The expected cache from the previous
report is shown below; confirm it exists on this server.

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHON_BIN=/absolute/path/to/the_verified_exp11_python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the_1651_images
export CACHE_ROOT=/home/lukaiv/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/310p_cache_rebuild_b32_kv4096_ab925dd51fbe_20260904T095205

set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$CACHE_ROOT"
test -d "$CACHE_ROOT/production_increfa_real_nz_compile"
test -d "$CACHE_ROOT/vision_prefill_b1_fp16"
test -d "$CACHE_ROOT/text_prefill_packed_fp16"

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

PYTHONPYCACHEPREFIX=/tmp/mineru_production_diag_pycache \
"$PYTHON_BIN" -m unittest discover \
  -s 11_mineru_2_5_pro_inference -p 'test_*.py' -v
~~~

All tests must pass before the NPU run.

## Phase 3: prepare one production command

This is the prior two-page command with diagnostics added. It reuses all three
cache families. A diagnostic step count of 1024 covers the complete 731-call
910B control while remaining bounded.

~~~bash
cd "$WORK_SERVER_REPO"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export DIAG_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_production_decode_diag_${COMMIT_SHORT}_${RUN_TAG}"
test ! -e "$DIAG_ROOT"
mkdir -p "$DIAG_ROOT"
git rev-parse HEAD >"$DIAG_ROOT/commit.txt"
printf 'cache_root=%s\nmodel=%s\ndataset=%s\nimages=%s\n' \
  "$CACHE_ROOT" "$MODEL_DIR" "$DATASET_JSON" "$IMAGES_DIR" \
  >"$DIAG_ROOT/inputs.txt"

BASE_COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
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
  --local-decode-diagnostic-steps 1024
  --local-decode-diagnostic-sync
  --local-decode-diagnostic-boundary-period 1280
  --token-trace --hash-model-files
  --streaming-pages --streaming-page-window 2
)
~~~

## Phase 4: current production policy

`retain` is the current default. It leaves inactive filler rows at their last
cache position while active rows advance.

~~~bash
export RETAIN_ROOT="$DIAG_ROOT/retain"
mkdir -p "$RETAIN_ROOT"
RETAIN_COMMAND=(
  "${BASE_COMMAND[@]}"
  --output-dir "$RETAIN_ROOT/output"
  --local-decode-filler-control retain
)
printf '%q ' "${RETAIN_COMMAND[@]}" >"$RETAIN_ROOT/command.sh"
printf '\n' >>"$RETAIN_ROOT/command.sh"

set +e
start_epoch="$(date +%s)"
timeout --signal=TERM --kill-after=30s 600s \
  "${RETAIN_COMMAND[@]}" >"$RETAIN_ROOT/run.log" 2>&1
RETAIN_STATUS=$?
RETAIN_ELAPSED="$(( $(date +%s) - start_epoch ))"
set -e
printf '%s\n' "$RETAIN_STATUS" >"$RETAIN_ROOT/exit_code.txt"
printf '%s\n' "$RETAIN_ELAPSED" >"$RETAIN_ROOT/elapsed_s.txt"
grep '^MINERU_PHASE ' "$RETAIN_ROOT/run.log" \
  >"$RETAIN_ROOT/phase_events.jsonl" || true
tail -n 200 "$RETAIN_ROOT/run.log"
~~~

Read the last unmatched diagnostic event literally:

- `decode_step_graph start` without `finish`: the synchronized compiled graph
  did not complete. Report that step's preceding `decode_step_state`, including
  all positions, effective lengths, residues, and boundary rows.
- `decode_step_token_copy_submit start` without `finish`: the graph completed,
  but token-copy/event submission did not return.
- `decode_step_previous_token_copy_wait start` without `finish`: the current
  graph and copy submission completed, but waiting for the prior D2H copy did
  not return.
- all step markers finish but a later request phase does not: report that exact
  phase. Do not label it a decode-graph stall.
- `page_pipeline finish` and exit 0: the smoke passed.

If `RETAIN_STATUS=0`, do not run an unnecessary variant. Report success and the
first four step states and timings.

## Phase 5: advance inactive filler positions, only if retain stops

Run this only if Phase 4 stops inside a `decode_step_*` boundary. It uses the
same production graph and caches. Only inactive filler token and position
controls advance. Active request state and output collection are unchanged.

~~~bash
export ADVANCE_ROOT="$DIAG_ROOT/advance"
mkdir -p "$ADVANCE_ROOT"
ADVANCE_COMMAND=(
  "${BASE_COMMAND[@]}"
  --output-dir "$ADVANCE_ROOT/output"
  --local-decode-filler-control advance
)
printf '%q ' "${ADVANCE_COMMAND[@]}" >"$ADVANCE_ROOT/command.sh"
printf '\n' >>"$ADVANCE_ROOT/command.sh"

set +e
start_epoch="$(date +%s)"
timeout --signal=TERM --kill-after=30s 600s \
  "${ADVANCE_COMMAND[@]}" >"$ADVANCE_ROOT/run.log" 2>&1
ADVANCE_STATUS=$?
ADVANCE_ELAPSED="$(( $(date +%s) - start_epoch ))"
set -e
printf '%s\n' "$ADVANCE_STATUS" >"$ADVANCE_ROOT/exit_code.txt"
printf '%s\n' "$ADVANCE_ELAPSED" >"$ADVANCE_ROOT/elapsed_s.txt"
grep '^MINERU_PHASE ' "$ADVANCE_ROOT/run.log" \
  >"$ADVANCE_ROOT/phase_events.jsonl" || true
tail -n 200 "$ADVANCE_ROOT/run.log"
~~~

Interpretation:

- retain stops and advance passes: the fault depends on the mixed/stale
  inactive-row production state. This does not by itself distinguish IncreFA
  from the graph's scatter updates.
- both stop at the same graph step and same state: inactive filler positions
  are not the cause.
- a stop coincides with a row whose effective length is exactly divisible by
  1280: report it as evidence consistent with the known Paddle boundary issue,
  not as proof that MinerU has the same period.
- a stop does not coincide with that boundary: do not invoke the Paddle 1280
  explanation.

If both runs finish, compare them exactly:

~~~bash
sha256sum \
  "$RETAIN_ROOT/output/generation_trace.jsonl" \
  "$ADVANCE_ROOT/output/generation_trace.jsonl"
diff -qr "$RETAIN_ROOT/output/predictions" \
  "$ADVANCE_ROOT/output/predictions"
~~~

## Plain-text response to Luka

Reply with:

1. pulled commit, device, environment versions, and confirmed cache root;
2. retain exit code and elapsed time;
3. the last 20 ordered `MINERU_PHASE` events;
4. for the last decode step, the exact 32 cache positions, effective lengths,
   residues modulo 1280, boundary rows, and active slots;
5. whether graph execution, copy submission, control update, or prior-copy wait
   is the first unmatched boundary;
6. advance result, if Phase 5 was required, with the same fields;
7. whether either stop coincided with an exact 1280-token boundary;
8. if both passed, generation-trace hashes and prediction diff result;
9. NPU health after the last run.

Do not call a timeout a slow compile. Do not claim an operator root cause from
one full-graph stop. Do not continue to 1,651 pages until Luka reviews this
diagnosis.
