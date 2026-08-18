# 310P full-1651 K20 + compiled-FP32 accuracy run

## Goal

Run one candidate-only production lane over all 1,651 OmniDocBench v1.6 pages.
This is the full follow-up to the passed representative-128 run that measured
6.98 prefill pages/s.

The exact candidate is:

- W4, recognition preprocessing T8, CPU affinity `0-63`;
- compiled TorchAir FP32 layout, B2, threshold 0.5, FP32 reading order;
- native layout weights and native layout depthwise operations;
- four-page lookahead with `310p_k20_l4` vision buckets;
- `constant_grouped_all` vision depthwise and `torchair_internal` weights;
- B128 compiled IncreFA decode, cross-KV 1320, self-KV/max length 2048;
- the frozen OmniDocBench evaluator, with HTML image tags removed only from
  evaluator copies.

Do not run an A/B baseline. The prior canonical eager-FP32 full run is the
reference: about 332 s prefill, 327 s decode, 2.50 sequential-core pages/s, and
90.2394 Overall. Preserve the new measured result even if it differs.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0-3. This server has no `npu-setup`.
- Preserve the venv launcher `python_nosym`; never apply `readlink -f` to it.
- Reuse the exact recognition/decode and compiled-FP32 layout caches from the
  successful representative-128 run. Do not compile new graphs.
- Do not delete, rename, or repair caches after a failure.
- `/dev/shm` can expose only 64 GiB. Proceed with `ALLOW_LOW_HOST_MEMORY=1`;
  preserve a real OOM if one occurs.
- Use the repository-selected frozen TeX Live 2025 runtime. Ambient TeX Live
  2022 produces different Page-CDM scores and is invalid.

## Prepare exact caches

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse --short HEAD

# Keep the same validated model, dataset, OpenOCR, evaluator, and Python values
# used by the successful representative-128 run.
export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export EVALUATOR_ROOT=/absolute/path/to/clean/OmniDocBench/evaluator
export EVAL_PYTHON=/absolute/path/to/frozen/evaluator/python

# Recover cache paths from the actual passed representative-128 command. Do
# not guess paths and do not select the older K10 cache.
REP128_REPORT="$(
  find "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
    -type f -name final_report.txt -print0 2>/dev/null \
    | xargs -0 grep -l \
      'UNIREC_310P_K20_REP128_COMPILED_FP32_ACCURACY: PASS' \
    | xargs -r ls -1t | head -n 1
)"
test -s "$REP128_REPORT"
REP128_ROOT="$(dirname "$REP128_REPORT")"
REP128_COMMAND="$REP128_ROOT/candidate/command.sh"
test -s "$REP128_COMMAND"

extract_flag() {
  "$PYTHON_BIN" - "$REP128_COMMAND" "$1" <<'PY'
import shlex
import sys

words = shlex.split(open(sys.argv[1]).read())
flag = sys.argv[2]
positions = [index for index, word in enumerate(words) if word == flag]
if len(positions) != 1 or positions[0] + 1 >= len(words):
    raise SystemExit(f"expected exactly one {flag}, found {len(positions)}")
print(words[positions[0] + 1])
PY
}

export COMPILE_CACHE="$(extract_flag --compile-cache-dir)"
export LAYOUT_CACHE_ROOT="$(extract_flag --layout-cache-dir)"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"
printf 'REP128_ROOT=%s\nCOMPILE_CACHE=%s\nLAYOUT_CACHE_ROOT=%s\n' \
  "$REP128_ROOT" "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT"
```

## Select the frozen evaluator runtime

Reuse the exact frozen runtime that produced the valid same-host CDM replay.
Do not download, reinstall, clone, or use the ambient TeX installation.

```bash
RUNTIME_FP="$(
  find "$WORK_SERVER_REPO/tmp" "$WORK_SERVER_REPO/temp" \
    -type f -name candidate_runtime_fingerprint.json -printf '%T@ %p\n' \
    2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -s "$RUNTIME_FP"
export OMNIDOCBENCH_EVAL_TOOLS_ROOT="$(
  "$EVAL_PYTHON" -c \
    'import json,sys; from pathlib import Path; d=json.load(open(sys.argv[1])); print(Path(d["tex_runtime"]["texlive_root"]).parents[1])' \
    "$RUNTIME_FP"
)"
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/texlive/2025/bin/aarch64-linux/pdflatex"
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/imagemagick-7.1.1-47/bin/magick"
printf 'RUNTIME_FP=%s\nOMNIDOCBENCH_EVAL_TOOLS_ROOT=%s\n' \
  "$RUNTIME_FP" "$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
```

## Launch in background

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; select an actually free 0-3
export CPUSET=0-63
export LAYOUT_CPU_THREADS=16
export MATCH_WORKERS=64
export TEDS_WORKERS=64
export CDM_WORKERS=64
export PROGRESS_EVERY_PAGES=1
export REQUIRE_WARM_VISION_CACHE=0
export DECODE_CACHE_GATE_ATTEMPTS=1
export ALLOW_LOW_HOST_MEMORY=1
export RUN_VARIANT=optimized_k20_l4_compiled_fp32

LAUNCH_OUTPUT="$(
  bash 12_unirec_0_1b_inference/run_310p_full1651_w4t8_accuracy_background.sh
)"
printf '%s\n' "$LAUNCH_OUTPUT"
RUN_ROOT="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_ROOT=//p' | tail -n 1)"
RUN_LOG="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_LOG=//p' | tail -n 1)"
PID="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^PID=//p' | tail -n 1)"
test -n "$RUN_ROOT" && test -n "$RUN_LOG" && test -n "$PID"
export RUN_ROOT RUN_LOG PID
```

Immediately give Luka the printed absolute `RUN_LOG` path for `tail -f`.

## Monitor time stragglers

Inspect every 15-30 seconds. Do not wait silently.

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" \
    -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E \
    'UNIREC_310P_FULL1651_PHASE|UNIREC_310P_DECODE_CACHE_GATE|UNIREC_TWO_PHASE_(PREFILL|DECODE|END)|HEARTBEAT|page=.*1651|compile|recompil|Traceback|ERROR' \
    "$RUN_LOG" | tail -30
  sleep 20
done
```

Expected behavior:

- evaluator runtime gate: under 20 seconds;
- model/cache startup and decode cache gate: about 1-2 minutes;
- full prefill: approximately 4-5 minutes, but use live page cadence;
- full decode: approximately 5-6 minutes;
- match/TEDS plus CDM evaluation: approximately 3-5 minutes;
- zero new OMs and no visible graph compilation.

If setup exceeds two minutes, page progress stops for 30 seconds, evaluation
exceeds five minutes, or compile/recompile text appears, report the active
phase, process state, latest marker, and elapsed time before waiting longer.
Do not automatically rerun.

## Completion and report

Required markers:

```text
UNIREC_310P_FULL1651_OM_INVENTORY_UNCHANGED
UNIREC_310P_FULL1651_W4T8_EVAL: PASS
```

Paste:

```bash
cat "$RUN_ROOT/preflight.log"
cat "$RUN_ROOT/evaluator_runtime_versions.txt"
cat "$RUN_ROOT/evaluator_runtime_smoke.json"
cat "$RUN_ROOT/decode_cache_gate/passed.json"
cat "$RUN_ROOT/inference_process_wall_s.txt"
cat "$RUN_ROOT/evaluation_image_tags_stripped/eval_match_teds_wall_s.txt"
cat "$RUN_ROOT/evaluation_image_tags_stripped/cdm_wall_s.txt"
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/inference_om.diff"
cat "$RUN_ROOT/exit_code.txt"
```

Also report:

1. commit, physical NPU, CPU affinity, CANN/Torch/Torch-NPU, `/dev/shm`, bare
   RAM, and externally observed peak HBM;
2. crop/rejection counts, all K20 bucket counts, physical/real rows, vision
   slot efficiency, and fallback rows;
3. inference process wall, lifecycle, prefill, decode including ingress,
   decode graph, sequential-core time and pages/s, raw/effective decode tok/s,
   and decode slot efficiency;
4. text edit, Page CDM, Page TEDS, reading-order edit, Overall, removed image
   tags, and all timeout/exception counts;
5. deltas versus the prior canonical values stated at the top;
6. absolute run root and log paths.

Do not impose a guessed accuracy threshold. The complete frozen evaluator is
the accuracy gate.
