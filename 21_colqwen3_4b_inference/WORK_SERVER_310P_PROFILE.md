# ColQwen3-4B: warmed 310P kernel profiling

You are an execution-only agent on the 310P machine. Run the committed capture
and analysis scripts, and report their results to Luka in chat. Do not write a
Markdown report, edit tracked code, commit, push, create branches, patch packages,
or change model settings. You do not need access to Luka's machine or blue zone.

The full English HR run already completed on 310P with matching retrieval
accuracy. Luka reported page encoding 2579 s for 1110 pages, query encoding 110 s,
scoring 8.8 s, and steady stage times about 1049 ms vision / 922 ms text. These
are user-relayed results, not independent measurements by this profiling script.
The task now is to identify kernels and model regions causing that runtime.
Do not repeat full HR or add a new accuracy-acceptance criterion.

## Measurement contract

- Same checkpoint, processor, FP16, native weights, B1, exact portable attention
  and padding as the successful HR run. No model modifications or alternatives.
- Real English HR page 5, original preprocessing: 5040 vision tokens and 1274
  real text positions (1280 inside text attention). This is a representative
  fixed-shape diagnostic, not a measurement over all 1110 pages.
- Separate fresh processes for `torchair` and `raw_eager`, in that order.
- Three windows: full forward (processed NPU input to NPU embeddings), vision
  graph only, and text graph only. Full includes preparation, mergers and
  projection. Vision excludes patch/position preparation and mergers. Text
  excludes mergers/rotary preparation/retrieval projection but **retains the
  production graph's padding and output trim**, unlike the older prealigned
  text microbenchmark. Do not silently substitute that older microbenchmark.
- Isolated text inputs are frozen from compiled vision in both processes. Their
  hashes must match before the automatic analyzer prints an eager/compiled
  comparison. Full eager still runs its own eager vision, as normal.
- Five warmups, 20 clean forwards before and after profiling, three recorded
  forwards per scope per metric, plus one profiler warmup. Compilation, file
  decode/preprocessing, transfers, validation and analysis are outside timings.
- Separate PipeUtilization and Memory captures, CPU+NPU traces, shapes and
  stacks. Profiled latency is diagnostic; use the clean timing for performance.
- Eager/compiled differences are recorded as diagnostics, not a new 0.002
  acceptance gate. Finite outputs and unchanged **same-lane replay** are checked.
  If profiling changes a replay, stop and retain the evidence; do not relax it.

## Get the source and recover the existing environment

Repository: `https://github.com/lukaIvanic/paddle_ocr_vl_npu.git`
Branch: `codex/colqwen-warm-forward-profile`. Use the exact profiling commit
provided by Luka with this brief when available; otherwise record fetched HEAD.

Start with `pwd`, `hostname`, `uname -a`, `command -v python3`, and
`git rev-parse --show-toplevel`. If needed locate the existing checkout with
`find /home /root /workspace /opt /data -maxdepth 5 -type d -name paddle_ocr_vl_npu -print 2>/dev/null`.
Verify its remote. Preserve unrelated changes; if tracked changes conflict,
report and stop. The existing successful checkout and run logs are preferred.

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git remote -v
git status --short --branch
git diff --quiet && git diff --cached --quiet || exit 1
git fetch origin codex/colqwen-warm-forward-profile
git checkout --detach FETCH_HEAD
git rev-parse HEAD
python3 21_colqwen3_4b_inference/discover_310p.py
```

If Luka provided an exact SHA, replace `FETCH_HEAD` with that SHA. Detached
checkout is allowed; it creates no branch. If there is no checkout, create an
unused directory, `git init`, add the remote above, fetch that branch, and
checkout detached; do not overwrite an existing directory.

Use the SAME container, interpreter, CANN/ATB setup, checkpoint and English HR
dataset as the successful full run. Inspect its `command.txt`, `result.json`
and the earlier handoff's run logs to recover these paths. The discovery script
prints candidate environments/assets without allocating a device; use `--root`
for another observed mount if discovery is truncated. Do not replace a virtual
environment path with its resolved system-Python symlink.

If working from the host, inspect `docker ps` and enter the existing successful
container; do not create or restart one. Recheck all paths inside that container.
If `npu-setup` exists, inspect it before sourcing; otherwise source the matching
setup scripts identified from the successful local run. Do not guess CANN paths,
mix versions, install packages, or run the repository's 910B environment setup.
If the previous run/environment cannot be recovered, follow environment
discovery in `WORK_SERVER_310P.md` only, not its full-HR execution steps.

Run `npu-smi info` and select one healthy idle 310P. Never terminate another
process or use the blue-zone NPU reservation. If no card is available, report
availability. Keep the cache exclusive to this sequential run. Prefer the
successful run's local graph cache; never copy 910B graphs to 310P.

## Execute

The assignments below are placeholders to replace with observed local paths.
Keep the same PROFILE_ROOT throughout; use an unused directory. Retain the old
full-HR outputs. Have at least 10 GiB scratch headroom for traces and exports.

```bash
export PYTHON_BIN=/actual/path/to/successful/colqwen/python
export COLQWEN_MODEL=/actual/path/to/Ops-Colqwen3-4B
export HR_DATASET=/actual/path/to/verified/english/hr/dataset
export COLQWEN_CACHE=/actual/path/to/exclusive/local/310p/cache
export ASCEND_RT_VISIBLE_DEVICES=ACTUAL_IDLE_PHYSICAL_ID
export PROFILE_ROOT="$WORK_SERVER_REPO/tmp/21_colqwen3_4b_inference/310p_profile_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EXPECTED_CHIP=310P
test -x "$PYTHON_BIN" && test -d "$COLQWEN_MODEL" && test -d "$HR_DATASET" || exit 1
df -h "$WORK_SERVER_REPO"
PYTHONPATH=21_colqwen3_4b_inference "$PYTHON_BIN" -m unittest -v \
  test_portable_profile_analysis test_forward_profile_analysis || exit 1
bash 21_colqwen3_4b_inference/run_portable_profile.sh
```

The shell runner records commands, source SHA, physical device, logs and exit
codes. It runs all captures and invokes analysis after each execution lane.
It prints `PROFILE_SUITE_COMPLETE` only after all requested work succeeds.
Do not add `--profile` to full HR, invent helper scripts, or manually parse CSVs.

Use a long-lived tool session. Stay engaged until it exits. Logs stream live;
every ten seconds a heartbeat gives the active phase and its elapsed time,
including compilation and profiler export. If backgrounding is necessary, poll
`tail -n 5 "$PROFILE_ROOT/torchair.log"` or the corresponding `raw_eager.log`
every 30–60 seconds. A heartbeat during export is not kernel-execution progress;
do not kill a healthy process just because export takes longer than inference.

If a capture or analysis fails, retain result.json, the entire raw capture,
command, log, traceback and exit code. Report the first causal failure. Do not
continue from a failed NPU stream. A new attempt needs a new process and root.
If hardware counters specifically are unsupported, a basic timing-only capture
is an explicitly allowed diagnostic fallback: in a NEW PROFILE_ROOT set
`PROFILE_METRICS=basic` and rerun the SAME shell runner. Label this as lacking
PMU evidence; it cannot settle compute-versus-bandwidth questions. Do not use
this fallback for model execution/parity errors or to hide an analysis bug.

For a second same-shape page only if the first timings are unstable or its
bottleneck is unclear, use NEW PROFILE_ROOT and `PROFILE_PAGE_INDEX=0`.
For an explicit retry of one scope/lane, the runner accepts
`PROFILE_EXECUTIONS=torchair` and `PROFILE_SCOPES=text` (or `vision`/`full`).
Keep defaults for the initial run; disclose all retry overrides.

The summary can be regenerated without an NPU and without rerunning inference:

```bash
"$PYTHON_BIN" 21_colqwen3_4b_inference/analyze_portable_profile.py --run-root "$PROFILE_ROOT"
```

## Report directly in chat

Use the generated `analysis_after_raw_eager.log` and `summary.json`. No new
analysis code is needed. Give Luka:

1. Host/container, device/chip, source SHA, interpreter/runtime versions, exact
   PROFILE_ROOT, successful phases and any fallback/missing counters.
2. `CLEAN` full/vision/text mean and p50 for both execution lanes; before/after
   means and `EAGER_COMPILED` ratios. Do not call full-forward time file-to-CPU
   throughput or compare profiled latency as a clean speed benchmark.
3. For compiled vision and text: top kernel types/shapes, ms/forward, percent
   of kernel sum, calls/forward, names and projection-role candidates; include
   `TARGETED_KERNELS` for transpose, strided slice, TransData, indexing,
   normalization, rotary and activation/gating work. Do not combine the two
   towers into an unlabelled list. Include full-forward hotspots outside them.
4. Available top-kernel pipe/memory counters, using their printed column units.
   Missing counters are unknown, never zero. State whether the evidence actually
   distinguishes compute/transfer pressure; don't infer it from duration alone.
5. Same-lane replay results, eager/compiled diagnostic differences, frozen-input
   identity checks, and paths to summary.json and raw profile directories.

Shape-derived projection roles are candidates (sometimes ambiguous). Family
buckets are heuristic, especially fused elementwise/norm work. Source filenames
and lines are stored in `model_map` for each projection role. Preserve unresolved
names/shapes rather than inventing an exact layer assignment. Percentages divide
by summed kernel duration, which may differ from elapsed wall time. Timeline
gaps include between-step boundaries and do not directly establish CPU stalls.

## Documentation and validation basis

Consulted Huawei's profiler API and CANN counter documentation:

- https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000289.html
- https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/devaids/Profiling/atlasprofiling_16_0016.html

The older PyTorch API page lists training products; it is not proof of 310P
API support. CANN documents inference-series counter collection but warns that
available fields vary by product. The installed 310P stack must validate actual
capture/export support. No claim of 310P profiling success is made in advance.

910B capture/analysis validation is pending while this brief is being authored;
Luka will provide the validated final commit with the handoff.
