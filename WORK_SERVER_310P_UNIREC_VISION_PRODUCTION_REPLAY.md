# 310P UniRec production-vision real-crop replay

Run only this task, then stop. The graph-only crash probe has passed every
isolated and cumulative case on 310P. This task restores the exact real-crop
production vision boundary around those same five graphs.

The three lanes distinguish:

1. graph-probe cache plus real crop routing;
2. cold compilation plus real crop routing;
3. fresh-process reuse of the cache created by lane 2.

Do not run layout, text prefill, cross-KV export, decode, page assembly,
profiling, parity replay, or a larger page set.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use commit `389d926` or a descendant.
- Use one genuinely free physical 310P. Never use physical NPU 5.
- Use the exact model, OpenOCR checkout, manifests, Python environment, and
  NPU activation that passed the graph crash probe.
- Keep NPU operator JIT off. `vision_production_lab.py` sets this itself.
- Preserve every log and the first causal failure.
- Do not delete or modify the successful graph-probe cache.
- Do not interpret setup or graph loading as measured vision throughput.

## 910B2 control

Commit `389d926` passed this exact 20-page workload twice on one Ascend 910B2:

```text
pages / crops / page groups        20 / 109 / 5
lookahead                           4
warmup replays / measured replays  1 / 2
cold replay median                 352.238 ms
fresh-process warm replay median   353.026 ms
cold crops/s                        309.5
warm crops/s                        308.8
compiled slot efficiency            0.649
maximum reserved HBM               859,832,320 bytes
OM files after cold and warm        exactly 1 per bucket
```

The five cold first-call times were approximately 29.16, 22.19, 22.32, 20.65,
and 20.53 seconds. Fresh-process cache-load first calls were approximately
18.32, 14.00, 13.44, 13.53, and 13.09 seconds. These are 910B2 context, not
310P thresholds.

## 1. Pull and resolve the existing artifacts

Run with Bash from the work-server checkout:

```bash
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch
git merge-base --is-ancestor 389d926 HEAD

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
PAGE_MANIFEST="${PAGE_MANIFEST:?set the existing full-run pages.jsonl}"
CROP_MANIFEST="${CROP_MANIFEST:?set the matching full-run crops.jsonl}"
PROBE_CACHE="${PROBE_CACHE:?set the cache root from the passing five-graph crash probe}"

PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
MODEL="$(readlink -f "$MODEL")"
OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
PAGE_MANIFEST="$(readlink -f "$PAGE_MANIFEST")"
CROP_MANIFEST="$(readlink -f "$CROP_MANIFEST")"
PROBE_CACHE="$(readlink -f "$PROBE_CACHE")"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -f "$PAGE_MANIFEST"
test -f "$CROP_MANIFEST"
test -d "$PROBE_CACHE"

COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_vision_production_replay_$COMMIT_SHORT"
FRESH_CACHE="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_vision_production_replay_$COMMIT_SHORT"
test ! -e "$OUT"
test ! -e "$FRESH_CACHE"
mkdir -p "$OUT"
```

If the defaults do not match the existing passed environment, set the
variables to the passed paths. Do not move, reinstall, or redownload artifacts
only to match the defaults.

Record provenance before running:

```bash
{
  git rev-parse HEAD
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'model=%s\n' "$MODEL"
  printf 'openocr=%s\n' "$OPENOCR_ROOT"
  printf 'page_manifest=%s\n' "$PAGE_MANIFEST"
  printf 'crop_manifest=%s\n' "$CROP_MANIFEST"
  printf 'probe_cache=%s\n' "$PROBE_CACHE"
  printf 'physical_device=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("npu_available=", torch.npu.is_available())
PY
  npu-smi info
} >"$OUT/preflight.log" 2>&1
```

## 2. Run helper

Define this function in the same Bash shell:

```bash
run_lane() {
  lane="$1"
  cache="$2"
  lane_dir="$OUT/$lane"
  mkdir -p "$lane_dir"

  find "$cache" -name '*.om' -printf '%P %s bytes\n' 2>/dev/null \
    | sort >"$lane_dir/om_before.txt"
  npu-smi info >"$lane_dir/npu_before.log" 2>&1

  command=(
    "$PYTHON_BIN"
    "$REPO/12_unirec_0_1b_inference/vision_production_lab.py"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --page-manifest "$PAGE_MANIFEST"
    --crop-manifest "$CROP_MANIFEST"
    --cache-dir "$cache"
    --output-dir "$lane_dir/output"
    --page-offset 0
    --page-limit 20
    --page-lookahead 4
    --warmup-replays 1
    --repeats 2
    --parity-samples-per-route 0
    --profile-scope none
    --diagnostic-graph-log
  )

  printf '%q ' "${command[@]}" >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"

  set +e
  SECONDS=0
  "${command[@]}" >"$lane_dir/run.log" 2>&1
  lane_status="$?"
  lane_wall="$SECONDS"
  set -e

  printf '%s\n' "$lane_status" >"$lane_dir/exit_code.txt"
  printf '%s\n' "$lane_wall" >"$lane_dir/process_wall_s.txt"
  find "$cache" -name '*.om' -printf '%P %s bytes\n' 2>/dev/null \
    | sort >"$lane_dir/om_after.txt"
  npu-smi info >"$lane_dir/npu_after.log" 2>&1

  printf 'LANE=%s EXIT=%s WALL_S=%s\n' \
    "$lane" "$lane_status" "$lane_wall"
  tail -n 5 "$lane_dir/run.log"
  return "$lane_status"
}
```

## 3. Lane A: passed probe cache plus real crops

```bash
run_lane probe_cache_real_crops "$PROBE_CACHE"
```

This is the highest-value test. If it fails, stop immediately. Report the last
complete `UNIREC_VISION_GRAPH_DIAGNOSTIC` line, exit code or signal, NPU state,
and OM inventory. Do not run the fresh-cache lanes.

If it passes, require:

- `UNIREC_PRODUCTION_VISION_LAB status=ok`;
- 20 pages, 109 crops, and 5 page groups;
- one warm workload replay and two measured replays;
- no traceback, CANN error, AICore error, hard exit, or lost outputs.

## 4. Lane B: new cold cache

```bash
mkdir -p "$FRESH_CACHE"
run_lane fresh_cache_cold "$FRESH_CACHE"
```

If it fails, stop and report the same causal evidence. Do not delete the cache
or retry compilation with changed settings.

## 5. Lane C: fresh-process warm-cache reuse

Run only after lane B passes:

```bash
run_lane fresh_cache_warm "$FRESH_CACHE"
```

After lane C, record the OM count per bucket. Do not fail solely because a
bucket has more than one OM: the earlier 310P cache replay produced a second OM
without evidence of a second semantic specialization. Pair every count with
the cache path, graph identity, and first-call log.

## 6. Compact report

Extract the final summaries and graph preparation events:

```bash
for lane in probe_cache_real_crops fresh_cache_cold fresh_cache_warm; do
  test -d "$OUT/$lane" || continue
  printf '=== %s ===\n' "$lane"
  cat "$OUT/$lane/exit_code.txt"
  grep 'UNIREC_PRODUCTION_VISION_LAB' "$OUT/$lane/run.log" || true
  grep 'warmup_graph_call_end' "$OUT/$lane/run.log" || true
  grep 'lab_measurement_replay_end' "$OUT/$lane/run.log" || true
  printf 'om_before=%s om_after=%s\n' \
    "$(wc -l <"$OUT/$lane/om_before.txt")" \
    "$(wc -l <"$OUT/$lane/om_after.txt")"
done | tee "$OUT/compact_report.txt"
```

Return only:

```text
310P UNIREC PRODUCTION VISION REPLAY: PASS_ALL | PROBE_CACHE_FAILURE |
FRESH_COLD_FAILURE | FRESH_WARM_FAILURE

commit / physical NPU / runtime:
lane A exit / wall / final summary:
lane B exit / wall / final summary:
lane C exit / wall / final summary:
per-lane five graph first-call times:
per-lane replay times:
per-lane peak allocated/reserved HBM:
per-lane OM count before/after:
last complete diagnostic and first causal error, if any:
evidence root:
```

If all three lanes pass, conclude only that the earlier production-vision lab
failure is not reproduced at commit `389d926`. Then stop. Do not begin
profiling or a larger run in this task.
