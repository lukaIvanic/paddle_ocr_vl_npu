# 310P persistent resident-K20 full-1651 run

## Goal

Run the persistent UniRec service over all 1,651 OmniDocBench v1.6 pages on one
Atlas 310P. Keep all 20 K20 vision graphs resident. Measure the hot request
window after real-page warmup, then run the frozen accuracy evaluator.

Do not run a baseline. Do not build or repair a cache. Recover every path from
the already-passed 310P low-memory and accuracy runs.

The matched 910B2 result at commit `cae401d` was:

| Metric | 910B2 |
|---|---:|
| Hot serving time | 203.687 s |
| Hot throughput | 8.1056 pages/s |
| Text edit | 0.053826 |
| Page CDM | 92.1501% |
| Page TEDS | 83.7940% |
| Overall | 90.1872% |
| Peak host PSS | 9.654 GB |
| HBM baseline / peak | 3,396 / 16,490 MB |

The accepted earlier 310P low-memory run reached about 1.99 pages/s. The
accepted accuracy runs scored about 90.2% Overall. This run changes the serving
schedule and K20 residency. It does not change the model, threshold, K20 graph
implementations, decoder graph, or evaluator.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0 through 3. There is no `npu-setup`.
- Preserve the validated executable named `python_nosym`. Never apply
  `readlink -f` to `PYTHON_BIN`.
- Do not use `nproc`. Use CPU affinity `0-63`.
- Reuse the exact passed K20, compiled-FP32 B2 layout, and B128 C1320 S2048
  decode caches. Any new OM or visible recompilation invalidates the run.
- Keep the accepted 310P eager tall-crop fallback. This avoids a new graph and
  preserves the validated crop path.
- Use 512 real pages as excluded warmup. Warmup must exercise all 20 K20 graphs
  before the hot timer starts.
- Use the repository-local frozen evaluator tools. Ambient TeX Live 2022 is
  invalid. Require TeX Live 2025/pdfTeX 1.40.28.
- Sample full process-tree PSS every 200 ms and physical-device HBM every 1 s.
- Give Luka the absolute live log path immediately. Inspect progress every
  15-30 seconds.
- Do not create an additional final report file. Print the requested Markdown
  table directly to Luka.

## Pull and recover the passed 310P environment

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse HEAD
test -f 12_unirec_0_1b_inference/run_persistent_unirec_service_benchmark.py
test -f 12_unirec_0_1b_inference/run_with_process_tree_memory.py

# Recover paths from the last low-memory run that passed inference, PSS, HBM,
# and trace checks. A report-only recovery is valid when final_report.txt says
# PASS.
LOWMEM_REPORT="$(
  find "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
    -type f -name final_report.txt -print0 2>/dev/null \
    | xargs -0 grep -l 'UNIREC_310P_LOWMEM_FULL1651_HBM: PASS' \
    | xargs -r ls -1t | head -n 1
)"
test -s "$LOWMEM_REPORT"
LOWMEM_ROOT="$(dirname "$LOWMEM_REPORT")"
LOWMEM_PREFLIGHT="$LOWMEM_ROOT/preflight.txt"
LOWMEM_COMMAND="$LOWMEM_ROOT/command.sh"
test -s "$LOWMEM_PREFLIGHT"
test -s "$LOWMEM_COMMAND"
test -s "$LOWMEM_ROOT/output/run_summary.json"

read_preflight() {
  sed -n "s/^${1}=//p" "$LOWMEM_PREFLIGHT" | tail -n 1
}

export PYTHON_BIN="$(read_preflight python)"
export MODEL="$(read_preflight model)"
export LAYOUT_MODEL="$(read_preflight layout_model)"
export IMAGES_DIR="$(read_preflight images)"
export COMPILE_CACHE="$(read_preflight vision_cache)"
export LAYOUT_CACHE_ROOT="$(read_preflight layout_cache)"
export DECODE_CACHE_PARENT="$(read_preflight decode_cache_parent)"
export CPUSET="$(read_preflight cpuset)"

test "$(basename "$PYTHON_BIN")" = python_nosym
test -x "$PYTHON_BIN"
test "$CPUSET" = 0-63

extract_flag() {
  "$PYTHON_BIN" - "$LOWMEM_COMMAND" "$1" <<'PY'
import shlex
import sys
words = shlex.split(open(sys.argv[1]).read())
flag = sys.argv[2]
positions = [i for i, word in enumerate(words) if word == flag]
if len(positions) != 1 or positions[0] + 1 >= len(words):
    raise SystemExit(f"expected exactly one {flag}, found {len(positions)}")
print(words[positions[0] + 1])
PY
}

export OPENOCR_ROOT="$(extract_flag --openocr-root)"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"
test -d "$DECODE_CACHE_PARENT"

mapfile -t DATASET_JSON_CANDIDATES < <(
  find "$(dirname "$IMAGES_DIR")" -maxdepth 2 -type f \
    -name OmniDocBench.json -print
)
test "${#DATASET_JSON_CANDIDATES[@]}" = 1
export DATASET_JSON="${DATASET_JSON_CANDIDATES[0]}"
```

## Recover the frozen evaluator

Use the runtime fingerprint that repaired and validated the earlier 310P CDM
score. Do not download, reinstall, or clone anything.

```bash
RUNTIME_FP="$(
  find "$WORK_SERVER_REPO/tmp" "$WORK_SERVER_REPO/temp" \
    -type f -name candidate_runtime_fingerprint.json -printf '%T@ %p\n' \
    2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -s "$RUNTIME_FP"

export EVAL_PYTHON="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["platform"]["python_executable"])' \
    "$RUNTIME_FP"
)"
export EVALUATOR_ROOT="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["evaluator_root"])' \
    "$RUNTIME_FP"
)"
export OMNIDOCBENCH_EVAL_TOOLS_ROOT="$(
  "$PYTHON_BIN" -c \
    'import json,sys; from pathlib import Path; d=json.load(open(sys.argv[1])); print(Path(d["tex_runtime"]["texlive_root"]).parents[1])' \
    "$RUNTIME_FP"
)"
export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
source 09_persistent_page_engine/scripts/omnidocbench_eval_env.sh

test -x "$EVAL_PYTHON"
test -f "$EVALUATOR_ROOT/pdf_validation.py"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/texlive/2025/bin/aarch64-linux/pdflatex"
test -x "$OMNIDOCBENCH_EVAL_TOOLS_ROOT/imagemagick-7.1.1-47/bin/magick"
"$EVAL_PYTHON" 09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py \
  --evaluator-root "$EVALUATOR_ROOT"
```

## Validate caches

```bash
LOCATOR_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_resident_k20_locator_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOCATOR_ROOT"
"$PYTHON_BIN" 12_unirec_0_1b_inference/locate_unirec_production_caches.py \
  --search-root "$COMPILE_CACHE" \
  --search-root "$LAYOUT_CACHE_ROOT" \
  --output "$LOCATOR_ROOT/cache_locator.json" \
  | tee "$LOCATOR_ROOT/cache_locator.log"
grep -q 'buckets=20/20 missing=none' "$LOCATOR_ROOT/cache_locator.log"
grep -q 'UNIREC_PRODUCTION_CACHE_LOCATOR: PASS' "$LOCATOR_ROOT/cache_locator.log"

DECODE_SHAPE="$DECODE_CACHE_PARENT/decode_weight_nz_lmhead57344_semantic56371/decode_selfkv2048_cross1320_increfa_all_b128_wnz"
DECODE_MODULE_COUNT="$(
  find "$DECODE_SHAPE" -name compiled_module | wc -l | tr -d '[:space:]'
)"
DECODE_OM_COUNT="$(
  find "$DECODE_SHAPE" -type f -name '*.om' | wc -l | tr -d '[:space:]'
)"
test "$DECODE_MODULE_COUNT" = 1
test "$DECODE_OM_COUNT" -ge 1
```

If the locator fails, stop and report its output. Do not build or repair a
cache. The accepted run already proved these paths.

## Select one free 310P and launch

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; replace with a free 0-3
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]

RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_persistent_resident_k20_full1651_$(git rev-parse --short=12 HEAD)_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/spool"
export RUN_ROOT

find "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$DECODE_CACHE_PARENT" \
  -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort -u >"$RUN_ROOT/om_before.txt"

COMMAND=(
  taskset -c "$CPUSET"
  "$PYTHON_BIN" 12_unirec_0_1b_inference/run_with_process_tree_memory.py
  --output "$RUN_ROOT/process_tree_and_hbm.json"
  --interval-ms 200
  --npu-id "$ASCEND_RT_VISIBLE_DEVICES"
  --npu-interval-ms 1000
  --
  "$PYTHON_BIN" 12_unirec_0_1b_inference/run_persistent_unirec_service_benchmark.py
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output-dir "$RUN_ROOT/output"
  --spool-dir "$RUN_ROOT/spool"
  --layout-cache "$LAYOUT_CACHE_ROOT"
  --vision-cache "$COMPILE_CACHE"
  --decode-cache-parent "$DECODE_CACHE_PARENT"
  --device npu:0
  --offset 0
  --limit 1651
  --warmup-pages 512
  --workers 4
  --recognition-threads 8
  --layout-lanes 1
  --layout-batch-size 2
  --layout-threshold 0.5
  --vision-bucket-preset 310p_k20_l4
  --vision-lanes 4
  --vision-graph-residency all
  --require-all-warmup-vision-graphs
  --vision-same-key-shards 1
  --vision-sharded-key-count 0
  --vision-record-budget 128
  --vision-max-calls-per-key 64
  --vision-queue-size 128
  --vision-tall-fallback eager
  --decode-batch-size 128
  --cross-cache-length 1320
  --self-cache-length 2048
  --max-length 2048
  --ready-queue-size 128
  --progress-every 16
  --write-outputs
)
printf '%q ' "${COMMAND[@]}" >"$RUN_ROOT/command.sh"
printf '\n' >>"$RUN_ROOT/command.sh"

nohup env PYTHONUNBUFFERED=1 CANN_KNOWLEDGE_BANK_PROCESS_NUM=0 \
  "${COMMAND[@]}" >"$RUN_ROOT/run.log" 2>&1 &
PID="$!"
printf '%s\n' "$PID" >"$RUN_ROOT/pid.txt"
export PID
printf 'RUN_ROOT=%s\nPID=%s\nFor Luka: tail -f %q\n' \
  "$RUN_ROOT" "$PID" "$RUN_ROOT/run.log"
```

Give Luka the absolute `run.log` path immediately.

## Monitor

```bash
while kill -0 "$PID" 2>/dev/null; do
  date -Ins
  ps -p "$PID" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E \
    'UNIREC_SERVING_(READY|WARMUP_GRAPHS|WARMUP_END|PROGRESS|HOT_END)|recompil|compile|Traceback|ERROR' \
    "$RUN_ROOT/run.log" | tail -25
  printf 'compiler_processes=%s om_count=%s\n' \
    "$(pgrep -af 'atc|ccec|compiler|tbe' | wc -l)" \
    "$(find "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$DECODE_CACHE_PARENT" -type f -name '*.om' | wc -l)"
  sleep 20
done
```

Expected sequence:

1. setup completes;
2. excluded warmup prints `UNIREC_SERVING_WARMUP_GRAPHS used=20/20`;
3. `UNIREC_SERVING_WARMUP_END` starts the hot boundary;
4. measured progress reaches 1,651/1,651;
5. `UNIREC_SERVING_HOT_END` prints hot wall and pages/s;
6. excluded shutdown and Markdown writing finish.

If warmup does not use 20/20 graphs, stop. If setup or progress is unchanged
for 30 seconds, inspect the latest line, compiler processes, HBM, and process
state before waiting longer. Do not automatically rerun.

## Inference checks

```bash
test -s "$RUN_ROOT/process_tree_and_hbm.json"
test -s "$RUN_ROOT/output/run_summary.json"
test -s "$RUN_ROOT/output/recognition_trace.jsonl"
"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
memory = json.loads((root / "process_tree_and_hbm.json").read_text())
run = json.loads((root / "output/run_summary.json").read_text())
assert memory["exit_code"] == 0
assert run["status"] == "pass"
assert run["page_count"] == 1651
assert run["settings"]["workers"] == 4
assert run["settings"]["recognition_threads"] == 8
assert run["settings"]["layout_batch_size"] == 2
assert run["settings"]["vision_bucket_preset"] == "310p_k20_l4"
assert run["settings"]["vision_graph_residency"] == "all"
assert run["settings"]["decode_batch_size"] == 128
assert run["settings"]["cross_cache_length"] == 1320
assert run["settings"]["self_cache_length"] == 2048
warm = run["warmup_metrics"]["npu"]["vision_dispatch_details"]
warm_keys = {
    call["key"]
    for dispatch in warm
    for call in dispatch.get("lane_calls", [])
}
assert len(warm_keys) == 20, sorted(warm_keys)
measured = run["measurement"]["npu"]["vision_dispatch_details"]
released = sum(
    len(row.get("released_keys", []))
    for dispatch in measured
    for row in dispatch.get("residency", [])
)
assert released == 0
print(
    "UNIREC_310P_RESIDENT_INFERENCE: PASS "
    f"hot_s={run['hot_pipeline_wall_s']:.6f} "
    f"pages_s={run['hot_pages_per_s']:.6f} "
    f"crops={int(run['measurement']['frontend']['crop_count'])} "
    f"warm_graphs={len(warm_keys)} released={released}"
)
PY

find "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$DECODE_CACHE_PARENT" \
  -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort -u >"$RUN_ROOT/om_after.txt"
diff -u "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt"
COMPILE_SIGNALS="$(
  grep -aciE 'recompil|graph compile|warmup_graph_call' "$RUN_ROOT/run.log" \
    || true
)"
test "$COMPILE_SIGNALS" = 0
```

## Frozen full accuracy evaluation

```bash
EVAL_DIR="$RUN_ROOT/evaluation_image_tags_stripped"
export EVAL_DIR

"$EVAL_PYTHON" 12_unirec_0_1b_inference/prepare_unirec_subset_eval.py \
  --dataset-json "$DATASET_JSON" \
  --output "$RUN_ROOT/output" \
  --evaluation-root "$EVAL_DIR" \
  --offset 0 --limit 1651 --strip-image-tags

cd "$EVAL_DIR/work"
ulimit -n 65536
PYTHONUNBUFFERED=1 taskset -c "$CPUSET" "$EVAL_PYTHON" \
  "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/run_omnidocbench_eval.py" \
  --config config.yaml \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 64 --teds-workers 64 \
  --page-timeout-sec 120 \
  --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30

mkdir -p "$EVAL_DIR/cdm"
PYTHONUNBUFFERED=1 taskset -c "$CPUSET" "$EVAL_PYTHON" \
  "$WORK_SERVER_REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py" \
  --input "$EVAL_DIR/work/result/predictions_quick_match_display_formula_result.json" \
  --output-dir "$EVAL_DIR/cdm" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --workers 64 >"$EVAL_DIR/cdm/run.log" 2>&1

"$EVAL_PYTHON" \
  "$WORK_SERVER_REPO/12_unirec_0_1b_inference/summarize_completed_unirec_eval.py" \
  --lane 310p_persistent_resident_k20_full1651 \
  --metric-result "$EVAL_DIR/work/result/predictions_quick_match_metric_result.json" \
  --stage-execution "$EVAL_DIR/work/result/predictions_quick_match_stage_execution.json" \
  --cdm-summary "$EVAL_DIR/cdm/cdm_run_summary.json" \
  --output "$EVAL_DIR/full_eval_summary.json"
```

Do not score the original Markdown directory. The evaluator copy must strip
HTML image tags.

## Print the only final report

Run this command and paste its stdout directly to Luka. Do not write another
report file.

```bash
cd "$WORK_SERVER_REPO"
"$EVAL_PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
run = json.loads((root / "output/run_summary.json").read_text())
memory = json.loads((root / "process_tree_and_hbm.json").read_text())
score = json.loads(
    (root / "evaluation_image_tags_stripped/full_eval_summary.json").read_text()
)
hbm = memory["npu_hbm"]
cdm = score["cdm_debug"]
teds = score["table_teds_execution"]
rows = [
    ("Hot serving time", f"{run['hot_pipeline_wall_s']:.3f} s"),
    ("Hot-window throughput", f"{run['hot_pages_per_s']:.4f} pages/s"),
    ("Text edit", f"{score['text_block_page_edit']:.6f}"),
    ("Page CDM", f"{100 * score['display_formula_page_cdm']:.4f}%"),
    ("Page TEDS", f"{100 * score['table_page_teds']:.4f}%"),
    ("Overall", f"{100 * score['official_overall']:.4f}%"),
    ("Peak host PSS", f"{memory['peak']['total_pss_bytes'] / 1e9:.3f} GB"),
    ("HBM baseline", f"{hbm['baseline']['used_mb']:,} MB"),
    ("HBM peak", f"{hbm['peak']['used_mb']:,} MB"),
    ("HBM above baseline", f"{hbm['peak_increase_from_baseline_mb']:,} MB"),
]
print("| Metric | 310P result |")
print("|---|---:|")
for name, value in rows:
    print(f"| {name} | {value} |")
print()
print(
    "Validation: "
    f"pages={run['page_count']}, "
    f"crops={int(run['measurement']['frontend']['crop_count'])}, "
    f"CDM exceptions/timeouts={cdm['exception_case_count']}/{cdm['timeout_case_count']}, "
    f"TEDS errors/exceptions/timeouts="
    f"{teds['error_case_count']}/{teds['exception_case_count']}/{teds['timeout_case_count']}, "
    f"page-match fallbacks="
    f"{sum(v['count'] for v in score['page_match']['fallbacks'].values())}."
)
print(f"Run root: {root}")
PY
```

Stop after printing the table. Do not launch a baseline or another candidate.
