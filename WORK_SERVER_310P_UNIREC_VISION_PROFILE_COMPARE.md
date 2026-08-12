# 310P UniRec vision profile against the matched 910B2 reference

Run only this task after the first-128 background job has exited and released
its NPU. Reuse the exact model, OpenOCR checkout, full-run manifests, Python
environment, and passed warm five-graph cache from first-128.

This is a measurement and comparison task. Identify where 310P spends more
time than 910B2. Do not debug, change code, test fixes, or start other lanes.

The committed 910B2 reference was measured at commit `d629c87` on physical
Ascend 910B2 NPU 7. It contains two matched views:

- first 32 real pages through the exact one-worker production boundary,
  including compact H2D, NPU normalization/transpose, bucket routing, padding,
  five compiled graphs, and output compaction;
- the same five already-warmed compiled graphs in isolation, with ten NPU-event
  controls and one pipe profile per graph.

The 910B2 reference measured 32 pages / 186 crops in 545.459 ms wall p50 and
545.228 ms device p50: 341.0 crops/s, 58.67 pages/s, and 0.655 slot efficiency.
Its first-128 weighted isolated graph time was 2.094010 s.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Pull the commit named by Luka, or a descendant.
- Use one genuinely free physical 310P. Never use physical NPU 5.
- Do not run concurrently with first-128 or another process on the selected NPU.
- Reuse the passed warm graph cache. Do not copy a 910B cache to 310P.
- Do not enable per-operator JIT compilation or fall back to CPU.
- Run the committed detached bundle exactly once. Do not retry automatically.
- Profiler export time and fresh-process graph loading are setup, not inference
  latency. NPU-event controls and production p50 are the latency authorities.
- Stop after returning the comparison. Do not propose or test a fix.

## 1. Pull and export the passed environment

Use one Bash shell:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export PAGE_MANIFEST="${PAGE_MANIFEST:?reuse the passed full-run pages.jsonl}"
export CROP_MANIFEST="${CROP_MANIFEST:?reuse the passed full-run crops.jsonl}"
export VISION_CACHE="${VISION_CACHE:?reuse the passed warm five-graph cache}"

for path in "$PYTHON_BIN" "$MODEL/model.pth" \
  "$OPENOCR_ROOT/tools/infer_doc_onnx.py" "$PAGE_MANIFEST" \
  "$CROP_MANIFEST" "$VISION_CACHE"
do
  test -e "$path"
done
```

Use the exact successful first-128 values. Do not move artifacts to match the
defaults.

## 2. Launch in the background and report the log path immediately

```bash
launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_vision_profile_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"

RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
WORKER_PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_VISION_PROFILE_STARTED pid=//p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$WORKER_PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P VISION PROFILE STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

The first phase prints 15-second heartbeats. The isolated graph phase creates
one directory at a time under `$RUN_ROOT/graph_suite/`.

```bash
while ! test -f "$RUN_ROOT/exit_code.txt"; do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    printf 'OWNED_PROCESS_EXITED_WITHOUT_STATUS pid=%s\n' "$WORKER_PID"
    break
  fi
  tail -n 12 "$RUN_LOG"
  find "$RUN_ROOT/graph_suite" -maxdepth 1 -mindepth 1 -type d \
    -printf '%f\n' 2>/dev/null | sort
  sleep 15
done
```

Do not infer compilation from a quiet first call. Check the OM inventories.

## 3. Require a complete matched bundle

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
PRODUCTION="$RUN_ROOT/production/vision_production_lab.json"
GRAPHS="$RUN_ROOT/graph_suite/profile_suite_summary.json"
test -f "$PRODUCTION"
test -f "$GRAPHS"
cmp "$RUN_ROOT/om_before.tsv" "$RUN_ROOT/om_after.tsv"

grep 'UNIREC_PRODUCTION_VISION_LAB' "$RUN_LOG"
grep 'UNIREC_PREFILL_PROFILE_' "$RUN_LOG"
```

Stop with a failure report if the process failed, either JSON is absent, or the
OM inventory changed.

## 4. Compare automatically with the committed 910B2 profiles

```bash
ANALYSIS="$RUN_ROOT/vision_gap_analysis.json"
COMPARE_LOG="$RUN_ROOT/vision_gap_analysis.log"
"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/analyze_vision_profile_comparison.py" \
  --npu310-production "$PRODUCTION" \
  --npu310-graphs "$GRAPHS" \
  --output "$ANALYSIS" \
  --topn 12 2>&1 | tee "$COMPARE_LOG"
test "${PIPESTATUS[0]}" = 0
test -f "$ANALYSIS"
```

The analyzer requires the same page offset, page count, and page-group contract.
It permits crop and bucket-distribution drift, reports every difference, and
normalizes the production, graph, and surrounding ratios per crop using each
run's own bucket-call histogram. The five isolated graphs and their fixed
first-128 weighting remain exact comparisons. It reports:

- per-crop production-boundary ratio;
- per-crop isolated graph ratio and estimated graph share of the production gap;
- per-crop surrounding transfer/normalization/padding/compaction ratio;
- all five bucket latency ratios and first-128 weighted gaps;
- kernel counts and weighted cube utilization;
- kernel types and exact shape signatures ranked by added weighted time.

The surrounding split is an attribution estimate: synchronized production
device p50 minus the sum of isolated graph means using the exact first-32 call
histogram. Treat a large value as a target category, not an individual kernel.

## 5. Return a short bottleneck report, then stop

Return:

```text
310P UNIREC VISION PROFILE: PASS | PYTHON_FAILURE | HARD_PROCESS_EXIT

commit / physical NPU / runtime:
owned PID / absolute run.log:
exit / process wall / OM inventory unchanged:
310P production device p50 / wall p50 / crops/s / pages/s:
910B2 production device p50 / wall p50 / crops/s / pages/s:
production ratio / isolated-graph ratio / surrounding ratio:
estimated graph share of the production gap:
five bucket ratios, with 310P ms versus 910B2 ms:
kernel counts and cube utilization by bucket:
top three kernel-type gaps by weighted seconds:
top three exact shape-signature gaps by weighted seconds:
bottleneck highlight: <one or two measured categories only; no proposed fix>
analysis JSON / production JSON / graph JSON / evidence root:
```

Paste the analyzer's `UNIREC_VISION_*` lines after this report. Do not start a
debugging or optimization experiment.

## Completed-run recovery

If the NPU bundle already completed successfully but an older analyzer rejected
only a crop or bucket-distribution mismatch, do not rerun the NPU bundle. Pull
the new commit, keep the existing `$RUN_ROOT`, then repeat only section 4 and
return section 5. Report `UNIREC_VISION_WORKLOAD_COMPARISON` prominently.
