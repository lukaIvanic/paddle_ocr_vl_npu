# 310P UniRec production vision: observable first-32 background run

Run only this task. Launch the first 32 pages in the background with the passed
warm five-graph cache. Immediately give Luka the absolute `run.log` path and a
ready-to-paste `tail -f` command. Then inspect the same log until the job exits.

This follows the successful 20-page three-lane replay. The goal is to extend
the exact production-vision workload to 32 pages while making every quiet
compile/load interval, page-group boundary, cache mutation, and failure point
visible.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use this commit or a descendant.
- Use one genuinely free physical 310P. Never use physical NPU 5.
- Reuse the warm cache from a successful 20-page production-vision lane.
- Run only offset 0, limit 32, lookahead 4, one warmup replay, and two measured
  replays.
- Do not run a cold-cache lane, layout, text prefill, decode, profiling, parity,
  or 128 pages.
- Do not stop the background job merely because graph loading is quiet. The log
  emits a phase heartbeat every 15 seconds.
- Stop only an owned PID from this task if intervention becomes necessary.

## 1. Pull and export the passed environment

Run in one Bash shell:

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

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export PAGE_MANIFEST="${PAGE_MANIFEST:?set the existing full-run pages.jsonl}"
export CROP_MANIFEST="${CROP_MANIFEST:?set the matching full-run crops.jsonl}"
export VISION_CACHE="${VISION_CACHE:?set the warm cache from a passed 20-page lane}"
```

Use the exact Python, model, OpenOCR, manifest, and cache paths that passed the
20-page work. Do not move or recreate them to match the defaults.

## 2. Launch and immediately report the full log path

```bash
launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_vision_production_32_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"

RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
WORKER_PID="$(printf '%s\n' "$launch_output" | sed -n 's/^UNIREC_VISION_BACKGROUND_STARTED pid=//p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$WORKER_PID"
test -f "$RUN_LOG"
```

Immediately send Luka this exact compact update before waiting:

```text
310P FIRST32 STARTED — pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

The launcher prints all three values. Do not shorten the path and do not report
a path relative to the repository. Luka will inspect this file independently.

## 3. Inspect the background log

The job is owned by the PID in `$RUN_ROOT/pid.txt`. It writes its final status
to `$RUN_ROOT/exit_code.txt`. Monitor without attaching the job to the current
terminal:

```bash
while ! test -f "$RUN_ROOT/exit_code.txt"; do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    printf 'OWNED_PROCESS_EXITED_WITHOUT_STATUS pid=%s\n' "$WORKER_PID"
    break
  fi
  printf '\n=== latest %s ===\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  tail -n 12 "$RUN_LOG"
  sleep 15
done
```

If the process exits without `exit_code.txt`, preserve the log and classify it
as a hard process exit. Do not retry automatically.

The useful log records are:

- `UNIREC_PRODUCTION_VISION_PROGRESS ... heartbeat`: current phase and elapsed
  time during quiet synchronous work;
- `... page_group_begin` / `... page_group_submitted`: current replay and page
  group, out of eight expected groups;
- `UNIREC_VISION_GRAPH_DIAGNOSTIC ... graph_registration_*`: graph wrapper
  registration;
- `... warmup_graph_call_begin/end`: exact bucket, cache inventory, and first
  synchronized compile/load duration;
- `... bucket_graph_call_begin`: production workload graph calls;
- `UNIREC_PRODUCTION_VISION_PROGRESS ... cache_inventory_checkpoint`: exact OM
  paths, sizes, modification times, and changes relative to the inventory after
  explicit graph warmup.

Interpret cache behavior carefully:

- An OM that appears during explicit graph warmup is compilation/cache
  preparation.
- Any OM path, size, or modification-time change after explicit graph warmup is
  reported as `unexpected_cache_mutation=true` and is evidence of later cache
  mutation worth investigating.
- Multiple OMs that already existed before launch and remain unchanged are not
  evidence of recompilation.
- Do not infer compilation only from a slow first call; fresh-process OM loading
  is also slow.

## 4. Final extraction and report

After the background process exits:

```bash
test -f "$RUN_ROOT/exit_code.txt"
status="$(cat "$RUN_ROOT/exit_code.txt")"
printf 'exit=%s\n' "$status"
tail -n 40 "$RUN_LOG"
grep 'UNIREC_PRODUCTION_VISION_LAB' "$RUN_LOG" || true
grep 'cache_inventory_checkpoint' "$RUN_LOG" || true
grep 'graph_warmup_cache_diff' "$RUN_LOG" || true
grep 'warmup_graph_call_end' "$RUN_LOG" || true
```

If exit is zero, also inspect:

```bash
RESULT_JSON="$RUN_ROOT/output/vision_production_lab.json"
test -f "$RESULT_JSON"
"$PYTHON_BIN" - "$RESULT_JSON" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "status": data["status"],
    "workload": data["workload"],
    "timing_ms": data["timing_ms"],
    "throughput": data["throughput"],
    "unexpected_cache_mutation_after_graph_warmup": data[
        "cache_diagnostics"
    ]["unexpected_mutation_after_graph_warmup"],
}, indent=2, sort_keys=True))
PY
```

Return:

```text
310P UNIREC VISION FIRST32: PASS | PYTHON_FAILURE | HARD_PROCESS_EXIT

commit / physical NPU / runtime:
owned PID:
absolute run.log path:
exit / process wall:
pages / crops / groups:
warmup and measured replay times:
pages/s / crops/s / slot efficiency / fallbacks:
five graph first-call durations:
OM count and inventory before graph warmup:
OM changes during graph warmup:
unexpected OM mutation after graph warmup: yes | no
peak allocated/reserved HBM:
last completed phase, replay, group, and graph event:
first causal error, if any:
result JSON / evidence root:
```

Then stop. Do not start 128 pages or profiling in this task.
