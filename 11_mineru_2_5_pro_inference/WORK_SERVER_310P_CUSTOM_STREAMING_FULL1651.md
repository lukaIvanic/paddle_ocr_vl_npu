# Work-server 310P custom MinerU full-1651 run and evaluation

This is the execution brief for the AI agent on Luka's Atlas 310P work server.
Read the repository `CLAUDE.md` and `AGENTS.md` first. Run every shell block in
Bash.

## Goal

Run the repository-owned custom MinerU2.5-Pro bounded-streaming pipeline over
all 1,651 OmniDocBench v1.6 pages on one Atlas 310P. Score the unchanged
predictions with the pinned OmniDocBench evaluator and frozen rendering tools.

Use the exact environment, model, corpus, and 310P-only cache root from the
passed two-page smoke in
`11_mineru_2_5_pro_inference/WORK_SERVER_310P_CUSTOM_STREAMING_SMOKE.md`.
The full run uses the accepted production contract:

- official MinerU frontend, processor, crop policy, and post-processing;
- custom local PyTorch model, with no vLLM or vLLM-Ascend engine;
- B32 continuous decode and KV4096;
- packed TorchAir text prefill with 128, 256, 512, and 1024-token buckets;
- TorchAir vision prefill with PromptFlashAttention and nine buckets;
- IncreFA decode, FRACTAL_NZ decode weights, and NPU rotary decode;
- 64-request CPU preparation queue, 32-request vision lookahead, global request
  stream, and 32-page bounded streaming window;
- two real warmup pages excluded from the hot timer;
- lossless token trace and unchanged Markdown evaluation.

The decode graph has no synthetic IncreFA warmup. Its first call follows real
request prefill. Before that call, every unused B32 physical row copies one
admitted real request's KV row, token, cache position, and rope delta. Completed
rows keep valid physical state until a new request replaces them. No inactive
row uses zero KV or an all-masked attention row.

Run one full inference and one evaluation. Do not run a baseline, ablation,
second hot-cache trial, or another corpus.

## Preconditions

Do not start unless the custom two-page 310P smoke passed. Its direct reply
must say:

```text
310P MINERU CUSTOM TWO-PAGE SMOKE: PASS
```

You need the exact values reported by that task:

```text
SMOKE_RUN_ROOT
SMOKE_CACHE_ROOT
PYTHON_BIN
MODEL_DIR
DATASET_JSON
IMAGES_DIR
OMNIDOCBENCH_REPO
```

If the smoke failed, any path is unavailable, or the smoke used a different
model or corpus identity, stop and report the issue. Do not reconstruct a new
environment or cache in this task.

## Rules

- The work-server repository is pull-only. Do not edit tracked files, create a
  branch, commit, push, reset, stash, or discard another person's changes.
- Do not edit `/vllm-workspace`, the OmniDocBench evaluator, installed
  frameworks, the custom model implementation, or any shared library.
- Do not install or replace Python packages, PyTorch, torch-npu, TorchAir,
  CANN, the driver, firmware, TeX Live, ImageMagick, or Ghostscript.
- Do not download a model, dataset, image, evaluator, or rendering tool.
- vLLM and vLLM-Ascend are not part of this pipeline. Do not inspect, import,
  change, or report them.
- Use one free physical Atlas 310P device. Never terminate another user's
  process. Do not fall back to CPU or CUDA for inference.
- Reuse the exact cache root from the passed smoke. Keep its existing files.
  Missing bucket shapes may compile during the excluded two-page warmup. Do
  not delete, copy, rename, repair, or manually prebuild a cache.
- Run full inference once. If it fails, preserve the partial output and cache.
  Do not retry, resume, change flags, or start evaluation.
- Evaluate only after all 1,651 pages pass the completion and trace gates. Do
  not transform the prediction Markdown or remove image tags.
- Normal command, log, output, evaluation, and cache evidence is allowed. Do
  not create a separate Markdown report or agent report.
- Send the absolute live inference log path to Luka immediately after launch.
  Report progress while the process runs. Do not wait silently.
- If any gate fails, stop and reply directly in plain text. Include the phase,
  exact command, exit code, first causal error, and relevant paths. Do not
  propose a source change, workaround, patch, or rerun.

## Reference identities and results

The smoke already verified these inputs. Verify them again before the long run:

```text
MinerU model files: 15
Path-independent model manifest SHA-256:
5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea
model.safetensors SHA-256:
abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0

OmniDocBench pages: 1651
OmniDocBench.json SHA-256:
a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496
Path-independent image manifest SHA-256:
34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe

OmniDocBench evaluator commit:
2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

The accepted 910B2 custom run used the same source and settings. Its measured
reference values are:

| Metric | 910B2 custom reference |
|---|---:|
| Hot pipeline wall | 2,029.96594 s |
| Hot throughput | 0.813314 pages/s |
| Overall | 95.1131% |
| Text accuracy | 96.3063% |
| Formula Page CDM | 96.7297% |
| Table Page TEDS | 92.3034% |
| Table structure Page TEDS | 95.1033% |
| Reading-order edit distance | 0.125259 |

These are comparison values, not 310P pass thresholds. The 310P evaluator
result is authoritative. Do not reject valid output because token IDs or scores
differ across chips.

## Phase 1: pull and recover the passed smoke

Start inside the repository clone:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet
git diff --cached --quiet
git pull --ff-only origin main
git rev-parse HEAD
git diff --quiet
git diff --cached --quiet
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_CUSTOM_STREAMING_FULL1651.md
test -f 11_mineru_2_5_pro_inference/run_serving_accuracy.sh
```

Set the exact paths from the passed smoke reply. Do not substitute another
environment or cache:

```bash
export SMOKE_RUN_ROOT=/absolute/path/from/the/passed/smoke
export CACHE_ROOT=/absolute/cache_root/from/the/passed/smoke
export PYTHON_BIN=/absolute/python/from/the/passed/smoke
export MODEL_DIR=/absolute/model_dir/from/the/passed/smoke
export DATASET_JSON=/absolute/dataset_json/from/the/passed/smoke
export IMAGES_DIR=/absolute/images_dir/from/the/passed/smoke
export OMNIDOCBENCH_REPO=/absolute/omnidocbench_repo/from/the/passed/smoke
```

Verify the smoke evidence and cache relationship:

```bash
test -d "$SMOKE_RUN_ROOT"
test -d "$CACHE_ROOT"
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/model.safetensors"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$OMNIDOCBENCH_REPO/.git"
test "$(cat "$SMOKE_RUN_ROOT/exit_code.txt")" = 0
test -s "$SMOKE_RUN_ROOT/output/run_summary_shard_00.json"
test -s "$SMOKE_RUN_ROOT/output/generation_trace.jsonl"
test -s "$SMOKE_RUN_ROOT/comparison_910b2_first2.json"
test -s "$SMOKE_RUN_ROOT/commit.txt"
grep -Fq -- "$CACHE_ROOT/" "$SMOKE_RUN_ROOT/command.sh"

SMOKE_COMMIT="$(cat "$SMOKE_RUN_ROOT/commit.txt")"
git merge-base --is-ancestor "$SMOKE_COMMIT" HEAD
git diff --quiet "$SMOKE_COMMIT" HEAD -- \
  11_mineru_2_5_pro_inference/config.py \
  11_mineru_2_5_pro_inference/fixed_batch_engine.py \
  11_mineru_2_5_pro_inference/generation_trace.py \
  11_mineru_2_5_pro_inference/local_modeling_mineru.py \
  11_mineru_2_5_pro_inference/native_custom_backend.py \
  11_mineru_2_5_pro_inference/prefill_timing.py \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py \
  11_mineru_2_5_pro_inference/streaming_decode.py \
  11_mineru_2_5_pro_inference/streaming_pipeline.py \
  11_mineru_2_5_pro_inference/text_prefill_compile.py \
  11_mineru_2_5_pro_inference/vision_prefill_compile.py
```

Run this gate:

```bash
export SMOKE_RUN_ROOT CACHE_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SMOKE_RUN_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
comparison = json.loads((root / "comparison_910b2_first2.json").read_text())
assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["backend"] == "official_mineru_local-continuous-client"
assert summary["batch_size"] == 32
assert summary["local_compiled_cache_length"] == 4096
assert summary["local_decode_attention"] == "increfa"
assert summary["local_decode_weight_format"] == "decode_nz"
assert summary["local_decode_rotary_impl"] == "npu_apply"
assert summary["local_text_backend"] == "torchair-packed"
assert summary["local_vision_attention"] == "prompt_flash_attention"
assert summary["local_vision_backend"] == "torchair"
decode = summary["streaming"]["decode"]
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["initial_inactive_filler_rows"] > 0
assert decode["initial_filler_source_slot"] is not None
for key in (
    "missing_pages",
    "extra_pages",
    "empty_pages",
    "missing_requests_with_unchanged_layout",
    "extra_requests_with_unchanged_layout",
    "unexpected_input_changes",
    "new_length_stops",
):
    assert not comparison[key], (key, comparison[key])
accounting = comparison["candidate_trace_accounting"]
assert accounting is not None and all(accounting.values()), accounting
print("CUSTOM_MINERU_SMOKE_PREREQUISITE: PASS")
PY
```

If this gate fails, stop. Do not rebuild the smoke environment or cache.

## Phase 2: activate the same NPU runtime

Use the same server-owned CANN activation used by the passed smoke. If the
shell is new, inspect and source the same setup file again. Do not source a
file from another user's project.

Run `npu-smi info`. Select one free physical Atlas 310P device. Prefer the same
physical device used by the smoke if it is free, but do not wait for or evict
another user's process:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

Verify the exact Python and one real NPU operation:

```bash
"$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
import torch_npu
import torchair
import transformers
from mineru_vl_utils import MinerUClient
from torchair.inference import cache_compile

assert torch.npu.is_available()
torch.npu.set_device("npu:0")
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(8, dtype=torch.float16, device="npu:0")
print("PYTHON", sys.executable)
print("RUNTIME", json.dumps({
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "torchair": torchair.__file__,
    "transformers": transformers.__version__,
    "device": torch.npu.get_device_name(0),
}, sort_keys=True))
print("NPU_PROBE", (x + 1).cpu().tolist())
PY
```

The device name must identify a 310P. Stop if the runtime or NPU probe differs
from the passed environment.

## Phase 3: create the run root and repeat artifact verification

Create a new run root. Keep the passed cache root unchanged:

```bash
cd "$WORK_SERVER_REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_custom_streaming_full1651_${COMMIT_SHORT}_${RUN_TAG}"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

exec 9>"$CACHE_ROOT/full1651.lock"
flock -n 9 || {
  echo 'Another process owns the passed Experiment 11 310P cache.' >&2
  exit 2
}

git rev-parse HEAD >"$RUN_ROOT/commit.txt"
find "$CACHE_ROOT" -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort >"$RUN_ROOT/om_before.txt"
{
  hostname
  uname -a
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  npu-smi info
  "$PYTHON_BIN" -m pip show \
    torch torch-npu torchair transformers mineru-vl-utils httpx-retries
  df -h /dev/shm
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
} >"$RUN_ROOT/environment.txt" 2>&1

printf 'RUN_ROOT=%s\nCACHE_ROOT=%s\n' "$RUN_ROOT" "$CACHE_ROOT"
```

Repeat the complete artifact verifier and retain its stdout:

```bash
set -o pipefail
"$PYTHON_BIN" \
  "$WORK_SERVER_REPO/17_mineru_vllm_ascend_baseline/verify_310p_artifacts.py" \
  --model-dir "$MODEL_DIR" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO" \
  2>&1 | tee "$RUN_ROOT/artifact_verification.log"
ARTIFACT_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$ARTIFACT_STATUS" >"$RUN_ROOT/artifact_verification_exit_code.txt"
test "$ARTIFACT_STATUS" = 0
grep -q '"status": "PASS"' "$RUN_ROOT/artifact_verification.log"
```

## Phase 4: verify the frozen evaluator before inference

Use the same clean evaluator checkout verified above. Find the existing frozen
evaluator Python and rendering-tool root used by earlier successful 310P
OmniDocBench evaluations. Do not clone or install anything.

Prefer the paths from a prior successful evaluator runtime fingerprint when it
exists. Otherwise inspect normal local venv and tool roots:

```bash
find "$WORK_SERVER_REPO/tmp" "$WORK_SERVER_REPO/temp" \
  -type f -name candidate_runtime_fingerprint.json -print 2>/dev/null \
  | sort

find "$WORK_SERVER_REPO/.runtime_cache" "$HOME" /workspace /data /data1 /mnt \
  -maxdepth 10 -type f -path '*/texlive/2025/bin/*/pdflatex' \
  -print 2>/dev/null | sort -u

find "$HOME" /workspace/venvs /opt \
  -maxdepth 5 -type f \( -name python -o -name python3 \) \
  -print 2>/dev/null | sort -u
```

Set the verified existing paths:

```bash
export EVAL_PYTHON=/absolute/path/to/frozen/evaluator/python
export EVALUATOR_ROOT="$OMNIDOCBENCH_REPO"
export OMNIDOCBENCH_EVAL_TOOLS_ROOT=/absolute/root/containing/texlive-and-imagemagick
export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
source "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
```

Require the pinned source and tools:

```bash
test -x "$EVAL_PYTHON"
test -d "$EVALUATOR_ROOT/.git"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd
test -z "$(git -C "$EVALUATOR_ROOT" status --porcelain --untracked-files=no -- . ':!result')"
test -x "$OMNIDOCBENCH_PDFLATEX"
test -x "$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick"
test "$(gs --version)" = 9.55.0

"$EVAL_PYTHON" \
  "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py" \
  --evaluator-root "$EVALUATOR_ROOT" \
  >"$RUN_ROOT/evaluator_runtime_smoke.json"
grep -q '"status": "pass"' "$RUN_ROOT/evaluator_runtime_smoke.json"

"$EVAL_PYTHON" - <<'PY' >"$RUN_ROOT/evaluator_python_versions.txt"
import importlib.metadata as m
import sys
print("python=", sys.executable)
for name in (
    "apted", "beautifulsoup4", "evaluate", "func-timeout", "Levenshtein",
    "lxml", "numpy", "pandas", "Pillow", "pylatexenc", "PyYAML", "scipy",
):
    print(f"{name}={m.version(name)}")
PY
```

The runtime smoke must render the representative CJK formula with TeX Live
2025/pdfTeX 1.40.28 and ImageMagick 7.1.1-47. Ambient TeX Live 2022 is invalid.
If the frozen evaluator is unavailable or fails, stop before the NPU run.

## Phase 5: launch the full inference once

Use the same production command as the accepted 910B2 full run. The two warmup
pages are excluded from `pipeline_wall_s` and
`measured_group_pages_per_s`. They also exercise every configured static vision
and text bucket with real-page tensors before the hot timer starts.

```bash
export PYTHONUNBUFFERED=1

COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$RUN_ROOT/output"
  --offset 0
  --limit 1651
  --warmup-pages 2
  --no-resume
  --fail-fast
  --batch-size 32
  --page-batch-size 32
  --global-request-stream
  --layout-image-size 1036 1036
  --processor-min-pixels 25088
  --local-dtype float16
  --local-compiled-cache-length 4096
  --local-decode-attention increfa
  --local-decode-weight-format decode_nz
  --local-decode-rotary-impl npu_apply
  --local-prepare-prefetch-depth 64
  --local-prefill-metrics
  --local-text-backend torchair-packed
  --local-text-buckets 128,256,512,1024
  --local-text-max-members 32
  --local-text-torchair-cache-dir "$CACHE_ROOT/text_prefill_packed_fp16"
  --local-vision-attention prompt_flash_attention
  --local-vision-backend torchair
  --local-vision-buckets 384,512,768,1024,1536,2048,3072,4224,5632
  --local-vision-pack-target 768
  --local-vision-lookahead 32
  --local-vision-torchair-cache-dir "$CACHE_ROOT/vision_prefill_b1_fp16"
  --local-torchair-cache-dir "$CACHE_ROOT/production_increfa_real_nz_compile"
  --token-trace
  --hash-model-files
  --streaming-pages
  --streaming-page-window 32
)

printf '%q ' "${COMMAND[@]}" >"$RUN_ROOT/command.sh"
printf '\n' >>"$RUN_ROOT/command.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf '/usr/bin/time -f %%e -o %q ' "$RUN_ROOT/inference_process_wall_s.txt"
  printf '%q ' "${COMMAND[@]}"
  printf '\nstatus=$?\n'
  printf 'printf "%%s\\n" "$status" > %q\n' "$RUN_ROOT/inference_exit_code.txt"
  printf 'printf "%%s\\n" "$status" > %q\n' "$RUN_ROOT/exit_code.txt"
  printf 'exit "$status"\n'
} >"$RUN_ROOT/run_inference.sh"
chmod +x "$RUN_ROOT/run_inference.sh"

nohup "$RUN_ROOT/run_inference.sh" >"$RUN_ROOT/run.log" 2>&1 &
PID="$!"
printf '%s\n' "$PID" >"$RUN_ROOT/pid.txt"
printf 'RUN_ROOT=%s\nPID=%s\nFor Luka: tail -f %q\n' \
  "$RUN_ROOT" "$PID" "$RUN_ROOT/run.log"
```

Give Luka the `tail -f` path immediately.

## Phase 6: monitor inference

Inspect progress every 15 to 30 seconds during setup and warmup, then at least
every five minutes during steady page processing:

```bash
while [[ ! -s "$RUN_ROOT/inference_exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" \
    -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E \
    '\[setup\]|\[warmup\]|\[stream\] completed=|\[summary\]|Traceback|ERROR|ACL_ERROR|E[0-9]{5}' \
    "$RUN_ROOT/run.log" | tail -25
  printf 'om_count=%s compiler_processes=%s\n' \
    "$(find "$CACHE_ROOT" -type f -name '*.om' | wc -l)" \
    "$(pgrep -af 'atc|ccec|compiler|tbe' | wc -l)"
  npu-smi info
  sleep 30
done
```

Expected sequence:

1. model setup completes;
2. the two-page warmup runs;
3. all nine vision buckets and all four text buckets execute in warmup;
4. warmup prints `DONE` and resets measurement counters;
5. streaming progress reaches 1,651 of 1,651;
6. the final summary reports zero failed and skipped pages;
7. the process writes exit code zero and releases the NPU.

Compiler activity and new OM files are allowed only while missing shapes are
loaded or compiled before the hot run reaches steady progress. Record what
happens. Do not call a cache load a compile, and do not call a compile a cache
load. Do not restart the process to obtain a cleaner hot result.

If progress stops for five minutes, inspect the latest log, process state, HBM,
and compiler processes. Do not kill or rerun automatically. If the process
exits nonzero or the NPU becomes unhealthy, preserve the first causal error and
send the issue reply.

## Phase 7: validate the complete inference

After exit, require success and record the cache inventory:

```bash
test "$(cat "$RUN_ROOT/inference_exit_code.txt")" = 0
test -s "$RUN_ROOT/inference_process_wall_s.txt"
test -s "$RUN_ROOT/output/run_summary_shard_00.json"
test -s "$RUN_ROOT/output/generation_trace.jsonl"

find "$CACHE_ROOT" -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort >"$RUN_ROOT/om_after.txt"
comm -13 "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
  >"$RUN_ROOT/om_created.txt" || true
wc -l "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
  "$RUN_ROOT/om_created.txt"
```

Run the full structural gate:

```bash
export RUN_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
output = root / "output"
summary = json.loads((output / "run_summary_shard_00.json").read_text())
trace = [json.loads(line) for line in
         (output / "generation_trace.jsonl").read_text().splitlines()]
progress = [json.loads(line) for line in
            (output / "progress_shard_00.jsonl").read_text().splitlines()]

assert summary["completed"] == 1651
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["selected_pages"] == 1651
assert summary["shard_pages"] == 1651
assert summary["shard_count"] == 1
assert summary["shard_index"] == 0
assert summary["batch_size"] == 32
assert summary["page_batch_size"] == 32
assert summary["streaming_page_window"] == 32
assert summary["local_compiled_cache_length"] == 4096
assert summary["local_decode_attention"] == "increfa"
assert summary["local_decode_weight_format"] == "decode_nz"
assert summary["local_decode_rotary_impl"] == "npu_apply"
assert summary["local_text_backend"] == "torchair-packed"
assert summary["local_vision_attention"] == "prompt_flash_attention"
assert summary["local_vision_backend"] == "torchair"
assert summary["warmup"]["executed_pages"] == 2
assert set(summary["warmup"]["static_bucket_warmup"]["vision"]) == {
    "384", "512", "768", "1024", "1536", "2048", "3072", "4224", "5632"
}
assert set(summary["warmup"]["static_bucket_warmup"]["text"]) == {
    "128", "256", "512", "1024"
}
assert len(progress) == 1651
assert len({row["image"] for row in progress}) == 1651
assert len(list((output / "predictions").glob("*.md"))) == 1651
assert len(list((output / "content_lists").glob("*.json"))) == 1651
assert len(trace) == summary["generation_trace"]["requests"]
assert len({row["request_id"] for row in trace}) == len(trace)
assert all(row["generated_token_ids"] for row in trace)
assert not any((output / "failures").iterdir())
assert summary["streaming"]["remaining_inflight"] == 0
assert summary["streaming"]["source_closed"] is True
assert summary["streaming"]["decode"]["idle_rows_with_ready_work"] == 0
decode = summary["streaming"]["decode"]
assert decode["inactive_filler_policy"] == "duplicate_first_real_row_retain_controls"
assert decode["initial_inactive_filler_rows"] >= 0
assert decode["initial_filler_source_slot"] is not None
print(
    "CUSTOM_MINERU_310P_FULL1651_OUTPUT: PASS",
    f"requests={len(trace)}",
    f"pipeline_wall_s={summary['pipeline_wall_s']:.6f}",
    f"hot_pages_s={summary['measured_group_pages_per_s']:.6f}",
)
PY
```

## Phase 8: compare the full 910B2 trace

Verify and extract the committed reference:

```bash
REFERENCE_BUNDLE="$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/references/serving_streaming_1651_ae4c947c"
REFERENCE_ROOT="$RUN_ROOT/reference_910b2"
mkdir -p "$REFERENCE_ROOT"
(
  cd "$REFERENCE_BUNDLE"
  sha256sum -c SHA256SUMS
)
tar -xzf "$REFERENCE_BUNDLE/streaming.tar.gz" -C "$REFERENCE_ROOT"

set +e
"$PYTHON_BIN" \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/compare_generation_traces.py" \
  "$REFERENCE_ROOT/output" \
  "$RUN_ROOT/output" \
  --allow-table-image-placeholders \
  --output "$RUN_ROOT/comparison_910b2_full1651.json"
TRACE_COMPARE_STATUS="$?"
set -e
printf '%s\n' "$TRACE_COMPARE_STATUS" \
  >"$RUN_ROOT/comparison_910b2_full1651_exit_code.txt"

export RUN_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
result = json.loads((root / "comparison_910b2_full1651.json").read_text())
for key in (
    "missing_pages",
    "extra_pages",
    "empty_pages",
    "missing_requests_with_unchanged_layout",
    "extra_requests_with_unchanged_layout",
    "unexpected_input_changes",
):
    assert not result[key], (key, result[key])
accounting = result["candidate_trace_accounting"]
assert accounting is not None and all(accounting.values()), accounting
print(
    "CUSTOM_MINERU_310P_FULL1651_TRACE: PASS",
    f"differing_requests={len(result['differences'])}",
    f"changed_pages={len(result['changed_pages'])}",
    f"new_length_stops={len(result['new_length_stops'])}",
)
PY
```

The comparison must have complete page and trace accounting. Missing or extra
pages, missing or extra requests under unchanged layouts, unexpected input
changes, or failed accounting stop the task before evaluation. Exact cross-chip
token identity, byte-identical Markdown, and new length stops are reported, not
required. The evaluator measures their quality effect.

## Phase 9: evaluate unchanged predictions

Confirm the NPU inference process has exited. Keep the frozen evaluator exports
from Phase 4. Run the committed evaluator launcher in the foreground:

```bash
cd "$WORK_SERVER_REPO"
export RUN_ROOT DATASET_JSON
export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
export OMNIDOCBENCH_EVAL_TOOLS_ROOT
export LIMIT=1651

set +e
bash 11_mineru_2_5_pro_inference/run_serving_accuracy.sh
EVAL_STATUS="$?"
set -e
printf '%s\n' "$EVAL_STATUS" >"$RUN_ROOT/evaluation_launcher_exit_code.txt"
test "$EVAL_STATUS" = 0
test "$(cat "$RUN_ROOT/evaluation/exit_code.txt")" = 0
```

`prepare_serving_eval.py` checks complete page membership, prediction hashes,
progress records, dataset identity, zero inference failures, and the unchanged
Markdown adapter before the evaluator starts. The evaluator uses 24 page-match
workers, 12 CDM workers, and 12 TEDS workers. It gives each page a 420-second
hard deadline and records all fallbacks, timeouts, errors, and exceptions.

## Phase 10: print the final comparison

Run this command and paste its stdout directly to Luka. Do not redirect it to a
new report file:

```bash
export RUN_ROOT
"$EVAL_PYTHON" - <<'PY'
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
comparison = json.loads((root / "comparison_910b2_full1651.json").read_text())

text_edit = float(metric["text_block"]["page"]["Edit_dist"]["ALL"])
page_cdm = float(metric["display_formula"]["page"]["CDM"]["ALL"])
page_teds = float(metric["table"]["page"]["TEDS"]["ALL"])
structure_teds = float(metric["table"]["page"]["TEDS_structure_only"]["ALL"])
reading_edit = float(metric["reading_order"]["page"]["Edit_dist"]["ALL"])
overall = ((1.0 - text_edit) + page_cdm + page_teds) / 3.0

generation = run["local_compiled_generation"]
prefill = generation["prefill_metrics"]
decode = run["streaming"]["decode"]
vision_s = float(prefill["vision_transformer_blocks"])
text_s = float(prefill["text_transformer_prefill"])
vision_tok_s = float(prefill["raw_vision_tokens"]) / vision_s
text_tok_s = float(prefill["text_prefill_tokens"]) / text_s
teds_debug = stage["metrics"]["table"]["TEDS"]
cdm_debug = stage["metrics"]["display_formula"]["CDM"]
fallbacks = stage["page_match"]["fallbacks"]

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
    "Formula Page CDM": 100.0 * page_cdm,
    "Table Page TEDS": 100.0 * page_teds,
    "Table structure Page TEDS": 100.0 * structure_teds,
    "Reading-order edit": reading_edit,
}

print("310P MINERU CUSTOM FULL1651 AND EVALUATION: PASS")
print("| Metric | 310P | 910B2 custom | Delta |")
print("|---|---:|---:|---:|")
for name in reference:
    value = candidate[name]
    base = reference[name]
    print(f"| {name} | {value:.6f} | {base:.6f} | {value - base:+.6f} |")
print()
print(f"repo_commit={run['git_commit']}")
print(f"run_root={root}")
print(f"inference_process_wall_s={(root / 'inference_process_wall_s.txt').read_text().strip()}")
print(f"setup_s={run['setup_s']:.6f}")
print(f"warmup_s={run['warmup']['wall_s']:.6f}")
print(f"hot_pipeline_wall_s={run['pipeline_wall_s']:.6f}")
print(f"pages=1651 failed={run['failed']} skipped={run['skipped']}")
print(f"requests={run['generation_trace']['requests']}")
print(f"length_stops={comparison['candidate_length_stops']}")
print(f"vision_raw_tokens={prefill['raw_vision_tokens']} vision_transformer_s={vision_s:.6f} vision_tok_s={vision_tok_s:.3f}")
print(f"text_prefill_tokens={prefill['text_prefill_tokens']} text_transformer_s={text_s:.6f} text_prefill_tok_s={text_tok_s:.3f}")
print(f"decode_effective_tokens={generation['decode_calls']} decode_s={generation['decode_s']:.6f} decode_tok_s={generation['decode_tok_s']:.3f}")
print(f"decode_active_slot_fraction={decode['active_slot_fraction']:.6f}")
print(f"inactive_filler_policy={decode['inactive_filler_policy']}")
print(f"initial_inactive_filler_rows={decode['initial_inactive_filler_rows']}")
print(f"initial_filler_source_slot={decode['initial_filler_source_slot']}")
print(f"reference_pages_byte_identical={comparison['byte_identical_pages']}/1651")
print(f"differing_requests={len(comparison['differences'])} changed_pages={len(comparison['changed_pages'])}")
print(f"page_match_fallbacks={sum(row['count'] for row in fallbacks.values())}")
print(f"cdm_samples={cdm_debug['sample_count']} cdm_timeouts={cdm_debug['timeout_case_count']} cdm_exceptions={cdm_debug['exception_case_count']}")
print(f"teds_samples={teds_debug['sample_count']} teds_timeouts={teds_debug['timeout_case_count']} teds_errors={teds_debug['error_case_count']} teds_exceptions={teds_debug['exception_case_count']}")
print(f"evaluation_wall_s={(root / 'evaluation/wall_s.txt').read_text().strip()}")
print(f"cache_root={os.environ.get('CACHE_ROOT', '<report the reused path>')}")
print(f"om_before={sum(1 for _ in (root / 'om_before.txt').open())}")
print(f"om_after={sum(1 for _ in (root / 'om_after.txt').open())}")
print(f"om_created={sum(1 for _ in (root / 'om_created.txt').open())}")
PY
```

Also include the host, physical NPU, exact NPU product, CANN, Python, PyTorch,
torch-npu, TorchAir, Transformers, evaluator Python, evaluator commit, and
frozen rendering-tool versions from the retained preflight files.

Do not impose token parity or a guessed accuracy threshold. Report every
difference and the measured evaluator score.

## Issue reply

If any phase fails, stop and reply with only:

```text
310P MINERU CUSTOM FULL1651 AND EVALUATION: ISSUE
phase=
command=
exit_code=
first_causal_error=
smoke_run_root=
run_root=
cache_root=
paths_checked=
```

Do not add a proposed change, workaround, patch, rerun, or Markdown report.
