# 310P versus 910B2 UniRec representative-128 W1/T1 prefill trace

Pull the commit containing this brief and run this task only. It reproduces the
current accuracy-safe UniRec page-to-cross-KV producer on the committed
representative 128-page set, then compares the complete 310P result with the
committed 910B2 evidence.

The orchestrator runs two fresh Python processes sequentially on one physical
310P:

1. traced W1/T1 prefill-only run;
2. identical clean W1/T1 prefill-only run with tracing disabled;
3. exhaustive machine comparison against the committed 910B2 trace.

Do not run decode, evaluation, W2, W4, W8, T8, or the full benchmark.

## Exact inference contract

- OpenDoc `unirec-0.1b/model.pth` and PP-DocLayoutV2;
- the committed `unirec_representative_128_v1` page set;
- W1/T1, layout B2, four-page vision lookahead;
- eager FP32 layout, threshold 0.5, native layout weights and depthwise ops;
- FP16 recognition, native vision weights and depthwise ops;
- five compiled production full-vision buckets plus eager fallback shapes;
- compact uint8 HWC recognition inputs with Pillow bicubic resize;
- packed S1320 text prefill, cross-KV 1320, self-KV/max length 2048;
- complete cross-KV retained in page-scoped CPU shared memory;
- eight excluded warmup pages and 128 measured pages;
- stop immediately after prefill. No decode graph is loaded.

The trace keeps every cross-page vision batch as one event. It records its
exact member shapes, tokens, real rows, physical graph input, and device-stage
timings. It never divides one vision-call time among member crops.

## Restrictions

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Reuse the exact model, dataset, OpenOCR, and accuracy-safe native production
  compile-cache parent from the successful full W4/T8 run. Do not use an older
  `constant_grouped_all`, `torchair_internal`, or FP16-layout cache as the
  experiment contract.
- Do not copy a 910B cache to 310P. The committed 910B files are result data,
  not executable graph caches.
- `/dev/shm` must have at least 8 GiB available. The measured cross-KV bank is
  expected to occupy 3,328,745,472 bytes.
- Launch in the background and immediately give Luka the absolute run log and
  exact `tail -f` command.
- Do not retry with changed flags after a failure. The orchestrator attempts
  the clean twin even if the traced lane fails, then preserves both statuses.

## Committed 910B2 reference evidence

Source implementation commit: `4cf871c9433b6120a0741cd42d1021243e3d482b`.
The handoff/tooling commit changes no model or page-pipeline implementation.

Complete traced run:

```text
tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_trace_4cf871c_20260814T184415/
```

It contains the exact command, preflight, run log, terminal report, materialized
page manifest, run summary, all 3,623 iteration events, all 128 page records,
every distribution and shape histogram, exit status, process wall, and final
NPU state.

Clean twin:

```text
tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_clean_4cf871c_20260814T1850/
```

The clean reference contains its exact command, environment record, and
complete machine-readable run summary.

910B2 anchors:

```text
clean measured prefill       36.973793706 s
clean throughput              3.461911456 pages/s
traced measured prefill      38.078165345 s
trace overhead                2.986904%
setup, excluded              91.581453930 s clean / 92.170022594 s traced
warmup, excluded              1.130051636 s clean / 1.144418980 s traced
pages / crops / rejected    128 / 2487 / 0
real / physical tokens      180596 / 273240
cross-KV shared bytes       3328745472
trace events                3623
event workload digest       a7d189c2d3afeaf6333266038c4ce33aa29535c390ffb3b02323f60156dc5d2d
page workload digest        731caf71034070b653193d4af3a1e0dff0d629ebaac69ccefb68f461d19f82ae
```

The committed `prefill_distributions.json` is authoritative for every stage
count, sum, mean, min, P50, P75, P90, P95, P99, max, and shape histogram. Do
not transcribe selected values into a new parser.

## 1. Pull and resolve the validated 310P environment

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
  *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
esac
case "$ASCEND_RT_VISIBLE_DEVICES" in
  *,*) printf 'REQUIRES_EXACTLY_ONE_VISIBLE_310P=%s\n' \
         "$ASCEND_RT_VISIBLE_DEVICES" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench v1.6 images directory}"
export COMPILE_CACHE="${COMPILE_CACHE:?reuse the successful accuracy-safe native production compile-cache parent}"

PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
MODEL="$(readlink -f "$MODEL")"
LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR COMPILE_CACHE

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$COMPILE_CACHE"
test -f "$REPO/12_unirec_0_1b_inference/references/unirec_representative_128_v1.json"
test -f "$REPO/tmp/12_unirec_0_1b_inference/representative128_w1t1_prefill_trace_4cf871c_20260814T184415/output/prefill_iterations.jsonl"

SHM_AVAILABLE="$(df -B1 --output=avail /dev/shm | tail -n 1 | tr -d ' ')"
test "$SHM_AVAILABLE" -ge 8589934592
printf 'physical_npu=%s shm_available=%s commit=%s\n' \
  "$ASCEND_RT_VISIBLE_DEVICES" "$SHM_AVAILABLE" "$(git rev-parse HEAD)"
```

If `COMPILE_CACHE` is not already exported, recover it from the exact latest
successful accuracy-safe full-run `command.sh` or `preflight.log`. Do not search
for ONNX assets and do not substitute an optimized experimental cache lane.

## 2. Launch the complete cross-chip experiment

```bash
launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_representative128_w1t1_prefill_crosschip_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n 's/^PID=//p')"
TAIL_COMMAND="$(printf '%s\n' "$launch_output" | sed -n 's/^TAIL_COMMAND=//p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka exactly:

```text
310P REP128 W1T1 PREFILL CROSSCHIP STARTED - pid=<pid>; run_root=<absolute path>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Then follow only the owned run:

```bash
tail -f "$RUN_LOG"
```

The log prints every completed page. A 15-second heartbeat distinguishes graph
cache loading from a dead worker. The trace and clean lanes each have their own
`preflight.log`, `command.sh`, `run.log`, `report.log`, and output directory.

## 3. Completion and comparison report

```bash
test -f "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/lane_status.txt"
cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/crosschip_comparison.log"
printf 'RUN_ROOT=%s\nRUN_LOG=%s\nCOMPARISON_JSON=%s\n' \
  "$RUN_ROOT" "$RUN_LOG" "$RUN_ROOT/crosschip_comparison.json"
```

The comparator verifies before trusting any timing ratio:

- exact inference contract;
- exact traced-versus-clean contract and retained-bank workload on each chip;
- exact retained crop/token/cross-KV workload;
- exact page workload digest;
- exact timing-free 3,623-event workload digest;
- exact event counts;
- exact stage keys and sample counts;
- exact shape histograms for layout, crop transforms, vision calls, text packs,
  and cross-KV lengths.

It then reports **every** common stage distribution with candidate/reference
ratios for sum, mean, P50, P90, P95, P99, and max. The JSON preserves both full
histograms and both complete distribution rows.

Return to Luka:

1. `UNIREC_PREFILL_CROSSCHIP_CONTRACT`;
2. `UNIREC_PREFILL_CROSSCHIP_CLEAN`;
3. `UNIREC_PREFILL_CROSSCHIP_TRACE`;
4. all `UNIREC_PREFILL_CROSSCHIP_STAGE` and
   `UNIREC_PREFILL_CROSSCHIP_SHAPE` lines, preferably as the saved log path
   rather than manually retyping them;
5. the five largest positive stage `sum_s` gaps and five largest stage P90
   ratios, clearly noting that nested stages are not additive;
6. physical NPU, CANN, torch, torch_npu, setup/warmup/shutdown times, trace
   overhead, and absolute artifact paths.

If the contract gate fails, report the exact mismatched check and preserve the
comparison JSON. Do not quote cross-chip timing ratios as like-for-like.

If traced `NPUEvent.elapsed_time()` fails on 310P, report the first causal stack
trace and the last completed page. The clean lane should still run and be
reported. Do not disable event timing or change the experiment in place; a
separate event-free fallback would be a different measurement.

Then stop.
