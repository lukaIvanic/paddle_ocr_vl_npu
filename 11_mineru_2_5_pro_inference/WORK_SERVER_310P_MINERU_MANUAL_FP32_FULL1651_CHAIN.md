# Work-server 310P MinerU manual-FP32 smoke, full 1651, and evaluation

This brief is for the AI agent on Luka's Atlas 310P work server. It supersedes
the stopping point of
`WORK_SERVER_310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE.md`: the two-page smoke is
still mandatory, but a passing smoke must continue into the complete 1,651-page
run and frozen OmniDocBench evaluation.

## Goal

Complete this chain without returning early:

1. pull the unconditional manual-FP32 compiled-vision default;
2. run and gate the real two-page production smoke;
3. verify the existing TeX Live 2025 evaluation environment;
4. launch all 1,651 pages in the background;
5. monitor until inference has actually exited;
6. structurally validate all 1,651 outputs;
7. launch the frozen evaluation in the background;
8. monitor through page matching, CDM, TEDS, and evaluator exit;
9. print the 310P accuracy and throughput comparison.

## Rules

- Read `CLAUDE.md`, `AGENTS.md`,
  `WORK_SERVER_310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE.md`, and
  `WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md` first.
- This checkout is pull-only. Do not edit tracked files, create a branch,
  commit, push, reset, stash, or discard changes.
- Do not modify packages, CANN, TorchAir, torch-npu, models, datasets, or the
  evaluator.
- Reuse the verified Experiment-11 environment and production cache root. Do
  not delete caches and do not create a replacement production cache root.
- Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Use one healthy free 310P. Never terminate another user's process and never
  fall back to CPU or CUDA.
- Run the two-page smoke once, full inference once, and evaluation once.
- The full run and evaluation must be detached with `nohup` and `setsid`.
- Send the full-run log path to Luka immediately after launch, then keep
  monitoring. Do not treat a quiet log, released NPU, or tool timeout as
  completion. Require the child-written exit-code file.
- Use tool timeouts of at least 14,400,000 ms for the full inference and
  evaluation monitors. If a tool returns early, reattach and continue.
- Do not create a Markdown report. Report results directly to Luka in plain
  text.
- If any phase fails, stop the chain and report the exact command, exit code,
  first causal error, last unmatched marker, and artifact paths. Do not propose
  or apply a source change.

## Phase 1: pull, verify, and run the two-page smoke

Follow every command and gate in
`WORK_SERVER_310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE.md` through the end of its
Phase 2. In addition to that brief's commit check, require:

```bash
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only origin main
git merge-base --is-ancestor \
  eed292c9f3eab05d85fc42f3ee9fbf66a95adf1d HEAD
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_MINERU_MANUAL_FP32_FULL1651_CHAIN.md
```

The smoke must print:

```text
310P_MINERU_MANUAL_FP32_DEFAULT_SMOKE: PASS
```

Do not continue unless the smoke completed two pages with zero failures and its
`local_compiled_vision` metadata says:

```text
layer_norm_impl=manual_fp32
projection_impl=linear
attention=prompt_flash_attention
```

After the passing smoke, keep the same activated shell, device, cache lock,
`PYTHON_BIN`, `MODEL_DIR`, `DATASET_JSON`, `IMAGES_DIR`, `CACHE_ROOT`, and
`SMOKE_ROOT`. Define the continuation root:

```bash
export CHAIN_ROOT="$SMOKE_ROOT/continuation"
test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
```

## Phase 2: verify the frozen evaluator

Run Phase 4 of
`WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md` exactly. Resolve
`OMNIDOCBENCH_REPO` and `OMNIDOCBENCH_EVAL_PYTHON` from the previously verified
310P environment. Require all of these before inference:

- evaluator commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`;
- TeX Live 2025 with pdfTeX `1.40.28`;
- ImageMagick `7.1.1-47` from the repository-local evaluation tools root;
- Ghostscript `9.55.0`;
- `CJK.sty` and `c70gkai.fd` visible to the configured pdfTeX;
- `verify_omnidocbench_eval_runtime.py` reports `status: pass`.

Do not substitute ambient TeX Live 2022 or ImageMagick 6.

## Phase 3: launch all 1,651 pages

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
  --local-decode-increfa-length-mode pse_sentinel_310p
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

Send `FULL_RUN_LOG` to Luka immediately.

## Phase 4: monitor and validate inference

Run Phase 6 and Phase 7 of
`WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md` exactly. Keep monitoring
until `$RUN_ROOT/exit_code.txt` exists and equals zero. Require all original
1,651-page structural gates, plus:

```bash
export RUN_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
summary = json.loads((root / "output/run_summary_shard_00.json").read_text())
vision = summary["local_compiled_vision"]
assert vision["layer_norm_impl"] == "manual_fp32"
assert vision["projection_impl"] == "linear"
assert vision["attention"] == "prompt_flash_attention"
assert summary["completed"] == 1651
assert summary["failed"] == 0
assert summary["skipped"] == 0
print("310P_MINERU_MANUAL_FP32_FULL1651: PASS")
print("hot_pages_per_s", summary["measured_group_pages_per_s"])
print("pipeline_wall_s", summary["pipeline_wall_s"])
print("vision", vision)
PY
```

Do not begin evaluation unless both the original structural gate and this
manual-FP32 gate pass.

## Phase 5: evaluate and report

Run Phase 8, Phase 9, and Phase 10 of
`WORK_SERVER_310P_MINERU_CHAINED_SMOKE_FULL1651_EVAL.md` exactly. The evaluation
must use the already verified TeX Live 2025 and ImageMagick 7.1.1-47 roots.
Monitor until both the launcher and the evaluator's own `exit_code.txt` equal
zero.

The final reply must include the original comparison table for:

- hot pages/s;
- overall score;
- text accuracy;
- formula Page CDM;
- table Page TEDS;
- table-structure Page TEDS;
- reading-order edit distance.

Also include:

```text
310P MINERU MANUAL FP32 SMOKE FULL1651 EVALUATION: PASS
commit / host / exact NPU / torch / torch-npu / CANN:
two-page smoke elapsed and per-bucket compile/load times:
full completed / failed / skipped:
full request count:
hot pages/s and pipeline wall time:
vision LayerNorm / projection / attention metadata:
evaluation environment versions and evaluator commit:
evaluation timeouts / exceptions / page-match fallbacks:
smoke root:
full run root:
evaluation root:
```

Do not impose token parity or invent an accuracy threshold. Complete outputs,
successful evaluation, and honest metric reporting are the gates.

## Issue reply

If any phase fails, reply only with:

```text
310P MINERU MANUAL FP32 SMOKE FULL1651 EVALUATION: ISSUE
phase=
command=
exit_code=
first_causal_error=
last_unmatched_marker=
smoke_root=
full_run_root=
evaluation_root=
cache_root=
paths_checked=
```

Do not add a proposed patch, workaround, rerun, or Markdown report.
