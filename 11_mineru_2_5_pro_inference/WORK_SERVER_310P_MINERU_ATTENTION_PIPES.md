# 310P: deeper counters for the existing production attention matrix

Luka has authorized this diagnostic. Use your pull-only checkout and the
successful custom MinerU environment. Do not edit tracked code, install or
patch packages, change defaults, commit, push, or create branches. Report
issues directly to Luka in plain text, with the command and causal log excerpt;
do not write an agent report Markdown file. Preserve unrelated jobs and caches.

## Resolve and verify existing artifacts

1. Resolve the checkout using `git rev-parse --show-toplevel`, pull with
   `git pull --ff-only origin main`, record HEAD. Stop on conflicts.
2. Locate the successful **21-lane attention matrix** run from the preceding
   `WORK_SERVER_310P_MINERU_ATTENTION_MATRIX.md` brief. Use its `capture/`
   directory, `result.json` files, and existing vision cache root. Do not
   recapture pages, reconstruct Q/K/V, change masks, or clear/rebuild caches.
3. Use that run's own Python, CANN activation, model and dataset paths.
   Select one free healthy 310P and keep that physical device throughout.
   Record SOC, runtime versions and NPU health. Do not copy 910B paths.
4. The capture manifest and replay enforce model/tensor SHA256 checks.
   Expected model config SHA256:
   `22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28`;
   weights: `abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0`.
   Keep native vision weights, manual FP32 LayerNorm, FP16 and existing
   attention-mask semantics. This profiles the vision stack, not decode;
   the stored inputs came from the unchanged PSE-sentinel production path.

Resolve the placeholders, do not execute them literally:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
export PYTHON_BIN=/your/previous/successful/mineru/python
export OLD_MATRIX=/your/successful/21_lane_attention_matrix
export CAPTURE_DIR="$OLD_MATRIX/capture"
export VISION_CACHE=/your/existing/local-vision-torchair-cache-dir
export PIPE_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/attention_pipes_310p_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
cd "$WORK_SERVER_REPO"
test -x "$PYTHON_BIN" && test -s "$CAPTURE_DIR/manifest.json" && test -d "$VISION_CACHE"
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
PYTHONPATH=11_mineru_2_5_pro_inference "$PYTHON_BIN" -m unittest \
  11_mineru_2_5_pro_inference/test_attention_pipes.py \
  11_mineru_2_5_pro_inference/test_production_vision_attention.py
```

First extract existing pipe counters **without running inference**:

```bash
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/analyze_attention_pipes.py \
  "$OLD_MATRIX" --out "$OLD_MATRIX/attention_pipes_raw_calls.json"
```

This requires the original `kernel_details.csv` files, not only screenshots or
the old parsed summaries. If raw CSVs are missing, report that explicitly; do
not synthesize raw counters from old averages. Fresh collection below can still
proceed if the validated captures and caches are intact.

## Collect the supported metric groups in the real replay

```bash
nohup "$PYTHON_BIN" -u 11_mineru_2_5_pro_inference/profile_attention_pipes.py suite \
  --capture-dir "$CAPTURE_DIR" --cache-root "$VISION_CACHE" \
  --output-dir "$PIPE_ROOT" \
  --routes bucket_768,bucket_5632 \
  --variants baseline,eager_pfa,unpad_d80,unpad_d128,pfa_approx \
  --metrics pipe,arithmetic,memory,memory_l0,memory_ub,resource_conflict,l2,memory_access \
  --steps 10 --profile-steps 3 --timeout-s 900 \
  >"$PIPE_ROOT.driver.log" 2>&1 </dev/null &
PIPE_DRIVER_PID=$!
printf 'pid=%s log=%s\n' "$PIPE_DRIVER_PID" "$PIPE_ROOT.driver.log"
```

The wrapper loads one model/graph per variant/route, then collects separate
metric sessions around its warmed callable. It does not change the inference
harness source or cache identities. It verifies each profiled replay is exact
against that candidate's unprofiled output. Approximate candidates are compared
with themselves here; this does not establish OCR accuracy.

Stay engaged, using background execution and short polls, until `PIPES complete`.
Use a tool-session timeout above 120 minutes where available. There are ten
lanes, each with a 900-second deadline and 15-second heartbeats. Logs contain
`metric_start`/`metric_finish`; profiler parsing time is not graph compilation.
No overnight waits. No cache clearing.

Capability handling is deliberately explicit:

- Missing enum or an explicit runtime capability rejection is recorded as
  `unsupported_by_runtime`; other supported metrics continue.
- A completed collection without usable counter values is **missing data**,
  not zero activity and not a successful hardware measurement.
- Negative PMU traffic values are invalid exports: keep the raw values, but
  exclude them from derived statistics. The analyzer flags these and flags
  metric-pass duration more than 25% above the same lane's pipe pass. Do not
  use a heavily perturbed pass as hot performance evidence. On 910B the
  S5632 MemoryAccess pass exhibited both problems.
- An actual profiler/API failure stops the suite. Read the error, report it,
  and check health. If it unambiguously says a metric is unsupported on this
  device, you may rerun that lane alone with `mode=lane`, a new diagnostic
  output directory and that metric omitted, preserving the first failure log.
  Then run remaining variants/routes explicitly; do not repeat successful work.
- If math/parity succeeds but the device kernel export is absent, the current
  wrapper stops with `missing_kernel_csv`. Inspect the profiler logs and NPU
  health; if this is only missing profiler data, one fresh-process retry of
  the affected lane/metrics is permitted. Preserve both attempts. A repeated
  failure should be reported, not retried indefinitely.
- Never classify an NPU execution error, output mismatch, timeout or ambiguous
  collection failure as a harmless unsupported metric. Stop and report those.

## Analyze and explain to Luka

```bash
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/analyze_attention_pipes.py \
  "$PIPE_ROOT" --out "$PIPE_ROOT/attention_pipes_raw_calls.json"
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/analyze_attention_pipes.py \
  "$PIPE_ROOT" --out "$PIPE_ROOT/attention_pipes_compact.json" --omit-calls
```

Explain directly to Luka:

- Successful/unsupported/missing metric groups and exact runtime/device.
- Actual elapsed attention time separately from full encoder warm time.
- Absolute MAC, vector, scalar, MTE1/2/3 and FixPipe times per call, with
  mean/p50/p99/max and call counts. Keep AIC and AIV separate where exposed.
- Arithmetic FP16/FP32 counters; L0/UB traffic; memory bandwidth/access
  counters; L2 hit/miss counts; resource-conflict indicators. Preserve raw
  units and state missing fields rather than substitute cross-chip values.
- Baseline vs approximate D80: which measured counters change alongside the
  saved attention time? Baseline/eager differences? Unpad D80 vs D128?
- Shape S768 vs S5632: which costs grow? Check every variant against the same
  captured shape and precision contract.

PMU engine times overlap and are NOT additive wall-time components. AICore
time is not interchangeable with task duration. Each capture contains THREE
forwards, each with 32 attention calls: normalize totals accordingly. Do not
sum ratios/bandwidths, assume raw per-core bandwidth is chip-wide bandwidth,
or equate GM-interface traffic with physical DDR/HBM traffic (L2 intervenes).
High utilization or a profiler heuristic does not prove a critical bottleneck.

Keep all raw CSVs/logs/captures. No default promotion, page benchmark or
accuracy claim is authorized by this profiling run.

## Optional deeper tool capability check (read-only first)

Record `command -v msopprof`, `msopprof --help`, and the installed tool version
if exposed. Do not install missing tools. Detailed instruction/source timelines
may be unavailable on 310P; never infer support merely from the 910B help text.
Report availability and the remaining specific unanswered question to Luka.
Do not substitute random-tensor operator probes or rebuild vendor kernels.
Kernel replay can clear L2; any later operator-level probe must record replay
mode and avoid presenting cold-cache replay as production hot latency.

The 910B trial found two concrete restrictions: `TimelineDetail` rejected
`--replay-mode=application`, and kernel mode failed to create the Python MSTX
`attention_hot` range with this installed integration. No instruction timeline
was obtained. The tool returned shell exit 0 despite a failing child and
`Get profiling data failed`; inspect logs and actual artifacts, not exit status
alone. These are local tool/integration observations, not proof of a 310P
hardware restriction. The wrapper's `lane --operator-window` is an experimental
hook, not a validated instruction-timeline collection recipe. Keep the main
PMU suite independent of it.
