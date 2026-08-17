# 310P full-1651 canonical inference baseline

## Goal

Re-establish the approximately 1.9 pages/s accuracy-sized 310P baseline with one
full 1,651-page inference run.  This run does not evaluate accuracy; the
first-128 canonical-native parity test already matched the saved canonical 310P
outputs exactly.

Exact inference contract:

- PP-DocLayoutV2, eager FP32, native weights/depthwise, threshold 0.5, B2;
- W4 page workers and T8 recognition crop preprocessing;
- `production_v1` vision buckets with native weights and native focal depthwise;
- four-page vision lookahead and compact uint8 HWC recognition input;
- cross-KV 1320, self-KV/max length 2048;
- continuous compiled IncreFA B128 decode;
- no live-arena warmup before real token 1;
- one physical 310P device, 0-3.

The decode cache is probed in a separate process before inference and must load
the existing OM.  Evaluation is deliberately omitted to minimize wall time.

## Inputs

Pull the commit named in Luka's message.  Do not edit tracked files, create a
branch, commit, or push.

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"

export PYTHON_BIN=/absolute/path/to/the/validated/venv/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export COMPILE_CACHE=/absolute/path/to/the/canonical/production/cache/parent
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/to/the/passed/B128/decode/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=0  # one free physical 310P, 0-3 only
export CPUSET=0-63
export LAYOUT_CPU_THREADS=1
```

Preserve the venv executable path.  Do not `readlink -f` a venv symlink into
the base interpreter.

## Run and follow

```bash
bash 12_unirec_0_1b_inference/run_310p_full1651_accuracy_anchor_inference_background.sh
```

Immediately give Luka the absolute launcher output and follow command:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

The run prints page-level progress plus a 15-second heartbeat.  If progress
stops for more than 30 seconds, report the last 100 log lines immediately; do
not wait silently.

Expected wall time is approximately 15 minutes from the historical baseline,
not 350 seconds.  Rough historical phase expectations were about 540 seconds
of native prefill plus about 350 seconds of decode including ingress.  These are
context only; report the new measurement.

## Hard stops

Stop and report if:

- the decode cache probe compiles instead of loading;
- TorchDynamo reports a recompile;
- the B128 decode OM hash changes;
- the NPU becomes unhealthy;
- a device outside physical 0-3 is selected;
- crop/page progress remains silent for over 30 seconds.

Do not switch to K10/internal/grouped vision, B64, smaller KV, or another CPU
configuration.  Do not start evaluation after inference.

## Required report

Paste back:

1. `RUN_ROOT`, `RUN_LOG`, PID, and exit code;
2. decode-cache probe result and `DECODE_CACHE_OM_INVENTORY_UNCHANGED`;
3. `UNIREC_310P_FULL1651_BASELINE: PASS`;
4. every `UNIREC_DECODE_HISTORY_ROW`;
5. from `output/run_summary.json`:
   - `timing_s`
   - `throughput`
   - `decode`
   - `prefill_phase_summary`
   - crop/page/rejection/fallback counts;
6. from `history.json`, recognition-trace generated-length, cap, and repeated-cap
   statistics;
7. exact `preflight.log` and `command.sh` paths.

The headline baseline is
`throughput.sequential_core_pages_per_s`, which excludes setup, graph-cache
loading, process startup, and final shutdown while retaining full prefill plus
decode/ingress work.
