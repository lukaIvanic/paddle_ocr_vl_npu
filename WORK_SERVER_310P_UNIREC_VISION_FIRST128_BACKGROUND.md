# 310P UniRec production vision: first-128 background run

Run only this task. Extend the passed first-32 production-vision replay to the
original first 128 pages. Use the same model, manifests, five-graph cache,
runtime, and physical NPU policy as the passed first-32 run.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Pull the commit named by Luka, or a descendant.
- Use one genuinely free physical 310P. Never use physical NPU 5.
- Reuse the exact warm graph cache from the passed first-32 run.
- Run offset 0, limit 128, lookahead 4, one warmup replay, and two measured
  replays. The launcher fixes every setting except the page limit.
- Do not run layout, text prefill, decode, profiling, parity, or a cold-cache
  comparison.
- Launch in the background and immediately give Luka the absolute `run.log`
  path and ready-to-paste `tail -f` command.
- Do not retry automatically after a hard process exit.

## Launch

In one Bash shell, use the exact environment values that passed first-32:

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
export PAGE_MANIFEST="${PAGE_MANIFEST:?reuse the passed full-run pages.jsonl}"
export CROP_MANIFEST="${CROP_MANIFEST:?reuse the passed full-run crops.jsonl}"
export VISION_CACHE="${VISION_CACHE:?reuse the passed warm five-graph cache}"
export PAGE_LIMIT=128

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

Immediately report:

```text
310P FIRST128 STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

The detached job writes `exit_code.txt` when it completes. Monitor the log and
preserve it even if the process exits without that file:

```bash
while ! test -f "$RUN_ROOT/exit_code.txt"; do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    printf 'OWNED_PROCESS_EXITED_WITHOUT_STATUS pid=%s\n' "$WORKER_PID"
    break
  fi
  tail -n 12 "$RUN_LOG"
  sleep 15
done
```

## Required report

If the job exits zero, inspect
`$RUN_ROOT/output/vision_production_lab.json`. Return this compact report:

```text
310P UNIREC VISION FIRST128: PASS | PYTHON_FAILURE | HARD_PROCESS_EXIT

commit / physical NPU / runtime:
owned PID / absolute run.log path:
exit / process wall:
pages / crops / page groups:
warmup replay and both measured replay times:
pages/s / crops/s / slot efficiency / fallback count:
five graph first-call durations:
OM count before and after graph warmup:
unexpected OM mutation after graph warmup: yes | no
peak allocated/reserved HBM:
last completed phase, replay, group, and graph event:
first causal error, if any:
result JSON / evidence root:
```

Before declaring PASS, require all of the following:

- 128 pages were submitted;
- the warmup and both measured replays completed;
- fallback count is zero;
- `unexpected_mutation_after_graph_warmup` is false;
- exit code is zero.

Then stop. Do not start another workload.
