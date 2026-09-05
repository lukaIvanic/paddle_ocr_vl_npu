# 310P MinerU production-input attention matrix

Run this brief only when Luka hands it to you. You have no access to the Mac
or 910B. Work in your existing pull-only checkout and your successful MinerU
environment. Do not edit tracked files, install/patch packages, change defaults,
commit, push, or create branches. Report a failure directly to Luka in plain
text, with the command, first causal error, and log path. Do not write an agent
report Markdown file or propose speculative source changes.

## Scope

Test the same 32-layer production vision block path with:

- baseline compiled masked PromptFA D80;
- compiled masked PromptFA D128, retaining the D80 scale and slicing outputs;
- compiled D80 and D128 with experimental `innerPrecise=4`;
- eager PromptFA control;
- eager stock `_npu_flash_attention_unpad`, D80 and D128.

All linears stay native/ND, LayerNorm stays manual FP32, FP16 weights and
resolution stay unchanged. Decode remains on the existing PSE-sentinel path
during capture. No candidate output enters recognition or replaces production.
Unpad is intentionally eager across the full block stack: report device AND
host timings, including per-layer metadata handling and layout conversions.
It is not a ready-to-deploy fully compiled unpad implementation.

The accessible 910B validation and raw timing/parity data are in
`references/attention_matrix_910b_20260905/RESULTS.md` and the adjacent
`results.json`. All 15 runnable 910B lanes completed; the six approximate
lanes were explicitly skipped as 310P-only. Treat those rankings as context,
not predictions for 310P. The final 910B gate observed native format 2 on all
128 vision projection weights. Report your observed weight formats; do not
silently reformat weights if they differ.

The approximate mode is experimental and 310P-only. It changes softmax
precision, not weights. Full-encoder feature differences are measurements, not
proof of OCR accuracy. A finite result with drift may be reported; it must not
be promoted. Nonfinite values, NPU failures or timeouts stop the matrix.

## Preflight

1. Resolve the checkout with `git rev-parse --show-toplevel`, pull main with
   `git pull --ff-only origin main`, and record the resulting commit. Preserve
   unrelated work; stop on a pull conflict. Use the Python/CANN activation from
   the successful custom MinerU run, not the stock vLLM environment by guess.
2. Select one free healthy physical 310P, record its model/SOC and health,
   export it as `ASCEND_RT_VISIBLE_DEVICES`, and keep the same environment for
   every child process. Never kill unrelated jobs.
3. Locate `reference_command.sh` from the recent **MinerU six-route vision
   profiling** task. This is NOT the old Paddle Phase-50/51 task. It must use
   your own model/dataset/cache paths. Do not use the 910B command verbatim.
   If unavailable, follow section 2 of
   `WORK_SERVER_310P_MINERU_VISION_ROUTE_PROFILES.md` to regenerate it from
   your successful full-run summary and verify all listed asset hashes.
4. Confirm the command uses `local-continuous-client`, B32/KV4096, FP16,
   `pse_sentinel_310p`, manual-FP32/ordinary-linear/native-D80 vision,
   packed text, pack target768, layout1036, and existing usable caches.
   Retain that command's pixel settings and record them. Do not resize images,
   delete caches, or create a fresh baseline root. The harness creates separate
   candidate cache identities only. Baseline replay refuses a missing cache.

Model authority: config SHA256
`22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28`;
weights SHA256
`abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0`.
Dataset JSON authority:
`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
No `.msc`, `.mv` or `configuration.json` requirement. No downloads.

Resolve these values on YOUR machine:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
export PYTHON_BIN=/absolute/path/to/your/successful/mineru/python
export REFERENCE_COMMAND=/absolute/path/to/recent/mineru/profile/reference_command.sh
export VISION_CACHE=/absolute/path/from/that/command/local-vision-torchair-cache-dir
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/attention_matrix_310p_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test -x "$PYTHON_BIN" && test -s "$REFERENCE_COMMAND" && test -d "$VISION_CACHE"
test ! -e "$RUN_ROOT"
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$WORK_SERVER_REPO"
PYTHONPATH=11_mineru_2_5_pro_inference "$PYTHON_BIN" -m unittest \
  11_mineru_2_5_pro_inference/test_production_vision_attention.py
```

## Run and monitor until done

The driver first captures exact inputs, mask and baseline features from the
first 16 real production pages. It validates each mask is exactly contiguous
block-diagonal full attention before generating CPU sequence lengths. Filler
rows remain their own valid component. Nothing is replaced by synthetic inputs.

Each subsequent route/variant runs in a separate process, strictly sequentially.
Every lane has a 900-second deadline and 15-second heartbeat. New graph creation
is expected only for the D128/approximate candidates, not the baseline. A graph
timeout is a failure to diagnose, not permission to wait all night or clear caches.

```bash
nohup "$PYTHON_BIN" -u \
  11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --reference-command "$REFERENCE_COMMAND" \
  --cache-root "$VISION_CACHE" --output-dir "$RUN_ROOT" \
  --routes bucket_768,packed_768,bucket_5632 \
  --steps 30 --timeout-s 900 --profile \
  >"$RUN_ROOT.driver.log" 2>&1 </dev/null &
ATTENTION_DRIVER_PID=$!
printf 'pid=%s log=%s\n' "$ATTENTION_DRIVER_PID" "$RUN_ROOT.driver.log"
```

Use a tool-session timeout above 120 minutes or background commands with short
polls. Stay engaged until `MATRIX complete` and all 21 replay lanes completed.
Monitor the driver plus the current lane's log. On a stall, report the last
`ATTENTION_LAB` phase and elapsed time; do not infer compilation from CPU usage.
On an NPU error/timeout stop, inspect health and process ownership, and report
directly to Luka. Do not continue on a device left unhealthy.

All three capture routes must be present. Compare their `tags` and segment
lengths with the 910B reference: direct S768 real640, packed S768 real480+192,
and direct S5632 real5476. Differences are a cross-chip comparison caveat:
within-chip variants still use identical captures, but do not call differing
sample shapes an exact cross-chip match. If a route is missing, report it;
do not silently replace it with a random tensor.

Baseline capture/replay should agree; any baseline drift must be explained
before interpreting candidate parity. The `first_layer_eager_parity` field for
approximate lanes intentionally remains mode1; only GE lowering selects mode4.
Use `full_encoder_parity` for their actual approximate-mode feature result.

## Read and explain the results to Luka

```bash
"$PYTHON_BIN" 11_mineru_2_5_pro_inference/summarize_attention_matrix.py "$RUN_ROOT"
```

Report directly in plain text:

- device, runtime versions, exact commit, asset checks, capture shape/mask list;
- per route/variant: setup separately; warm device and host mean/p50/p99/max,
  useful and physical tok/s; speed ratio against the same captured baseline;
- full-encoder max/mean error, relative L2, cosine, nonfinite count and exactness;
- profile attention kernel names/counts/duration per full forward, total kernel
  time and visible layout/conversion work. Each profile has THREE forwards:
  use `duration_us / 3 / 1000`, never unnormalized totals or `aicore_time_us`;
- compare unpad with both eager PromptFA and compiled baseline. Faster attention
  with slower eager full encoder is evidence to consider a compiled/eager hybrid,
  not evidence the unpad kernel is slower;
- distinguish compiled-wrapper setup, compilation/first call, unprofiled warm
  timings and profiler collection time. Busy percentages do not establish
  optimality or compute-bound behavior;
- no claim of downstream OCR accuracy, changed defaults, or E2E page throughput.

Keep JSON, logs and raw captures on disk for follow-up. Do not run a full
OmniDocBench benchmark or change production defaults. Stop after the summary.
