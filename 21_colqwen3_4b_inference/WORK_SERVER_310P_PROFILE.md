# ColQwen3-4B: text-only native versus NZ profiling on 310P

You are an execution-only agent on the 310P machine. Run the committed capture
and analysis scripts, and report their results to Luka in chat. Do not write a
Markdown report, edit tracked code, commit, push, create branches, patch packages,
or change model settings beyond the explicit text-weight-format comparison below.
You do not need access to Luka's machine, blue zone, or any of its result files.

The full English HR run already completed on 310P with matching retrieval
accuracy. Luka reported page encoding 2579 s for 1110 pages, query encoding 110 s,
scoring 8.8 s, and steady stage times about 1049 ms vision / 922 ms text. These
are user-relayed results, not independent measurements by this profiling script.
The initial profiling has also completed. Luka reports that compiled text is
slower than eager on 310P, with matmul durations more than 50% higher on average,
similar MAC time and 2–2.5x vector time. These are relayed observations, not a
verified bandwidth or core-count diagnosis.

The current task is to test one explicit hypothesis: does preformatting only
text projection weights to FRACTAL_NZ improve warmed text forward latency and
matmul time? Run the existing scripts; do not implement an optimization yourself.
Do not repeat full HR or add a new accuracy-acceptance criterion.

## Measurement contract

- Keep the successful checkpoint, processor, FP16, B1, attention, masks, rotary,
  normalization, fusion and padding. The sole model variant is
  `TEXT_WEIGHT_FORMAT=native` versus `fractal_nz`.
- NZ uses the existing `npu_format_cast(weight, 29)` path, after projection
  fusion and before text compilation. It converts 144 text projection weights
  (four per layer), not activations, embeddings, normalization weights, retrieval
  projection or vision. All 96 vision projection weights stay native.
- Internal formats are enabled in BOTH controls. The runner records actual
  format histograms and rejects an NZ run unless every target text weight
  reports format 29. Native and NZ text graphs have distinct cache identities;
  the vision graph retains the same identity. Do not delete or mix caches.
- Use real English HR page 5 with original preprocessing: 5040 vision tokens,
  1274 real text positions, and 1280 positions inside the text graph. This is a
  fixed-shape diagnostic, not throughput over all 1110 pages.
- Measure/profile ONLY `PROFILE_SCOPES=text`. Text retains production padding
  and output trimming; it excludes vision, mergers, rotary-input preparation and
  retrieval projection. Do not substitute the older prealigned microbenchmark.
- Frozen text inputs come from compiled native vision in every process. Both
  vision and text input hashes must match across all six processes below.
  Vision still runs during setup and output diagnostics; it is not profiled.
- Initial order: native compiled, native eager, NZ compiled, NZ eager. Then
  confirm compiled performance in reversed format order, NZ then native, using
  fresh processes loading the same cached graphs. Use the same idle physical
  card throughout, with no simultaneous jobs of your own on that card.
- Each process uses five warmups and 20 clean forwards before and after
  profiling. Initial runs capture PipeUtilization and Memory separately, three
  active forwards each plus a profiler warmup. Cached confirmations capture
  PipeUtilization only. Use clean timings, not profiled latency, for speed.
- Loading, NZ conversion, compilation/cache loading, preprocessing, transfers,
  output diagnostics and analysis are outside the measured text forward.
- NZ diagnostics compare native eager text against NZ eager and NZ compiled
  text, and compare final embeddings using the same compiled vision. Record
  these differences; do not mistake a diagnostic `passed: false` at the existing
  0.002 tolerance for a new retrieval-quality rejection criterion. Finite outputs
  and exact SAME-LANE replay remain mandatory; the scripts enforce those.

## Get the source and recover the existing environment

Repository: `https://github.com/lukaIvanic/paddle_ocr_vl_npu.git`
Branch: `codex/colqwen-warm-forward-profile`. Use the exact profiling commit
provided by Luka with this brief when available; otherwise record fetched HEAD.
The NZ profiler implementation was validated at `cc2e8c68`; use this updated
handoff commit or a descendant containing that implementation, not an older
profiling checkout.

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
git merge-base --is-ancestor cc2e8c68 HEAD || exit 1
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

## Execute the comparison

Replace only the environment-specific path/device placeholders below with
observed local values. Keep the model, dataset, physical device, cache root and
experiment root fixed for the whole sequence. Use a new experiment root; keep
prior HR/profile outputs. Check available disk space before capture and retain
all raw traces. Ten GiB is the earlier minimum, not a guarantee that every
profiler version's six-process output fits; monitor free space during the run.

```bash
export PYTHON_BIN=/actual/path/to/successful/colqwen/python
export COLQWEN_MODEL=/actual/path/to/Ops-Colqwen3-4B
export HR_DATASET=/actual/path/to/verified/english/hr/dataset
export COLQWEN_CACHE=/actual/path/to/exclusive/local/310p/cache
export ASCEND_RT_VISIBLE_DEVICES=ACTUAL_IDLE_PHYSICAL_ID
export EXPERIMENT_ROOT="$WORK_SERVER_REPO/tmp/21_colqwen3_4b_inference/310p_text_nz_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EXPECTED_CHIP=310P PROFILE_PAGE_INDEX=5 PROFILE_SCOPES=text
export PROFILE_WARMUPS=5 PROFILE_REPEATS=20 PROFILE_STEPS=3
test -x "$PYTHON_BIN" && test -d "$COLQWEN_MODEL" && test -d "$HR_DATASET" || exit 1
test ! -e "$EXPERIMENT_ROOT" || exit 1
mkdir -p "$EXPERIMENT_ROOT" "$COLQWEN_CACHE"
df -h "$EXPERIMENT_ROOT" "$COLQWEN_CACHE"
PYTHONPATH=21_colqwen3_4b_inference "$PYTHON_BIN" -m unittest -v \
  test_portable_profile_analysis test_forward_profile_analysis || exit 1

# Four initial processes: both formats, each compiled then eager.
export PROFILE_METRICS='pipe memory' PROFILE_EXECUTIONS='torchair raw_eager'
for format in native fractal_nz; do
  export TEXT_WEIGHT_FORMAT="$format"
  export PROFILE_ROOT="$EXPERIMENT_ROOT/$format"
  bash 21_colqwen3_4b_inference/run_portable_profile.sh || exit 1
done

# Two fresh processes, reversed format order, reusing the SAME graph cache.
export PROFILE_METRICS=pipe PROFILE_EXECUTIONS=torchair
for format in fractal_nz native; do
  export TEXT_WEIGHT_FORMAT="$format"
  export PROFILE_ROOT="$EXPERIMENT_ROOT/${format}_cached"
  bash 21_colqwen3_4b_inference/run_portable_profile.sh || exit 1
done
```

The setup creates the chosen cache directory if needed; it does not clear an
existing cache. The runner adds the format to graph identity automatically,
even under one shared cache root. No two processes may write to it concurrently. Confirm
`om_present_before: true` for both graphs in each `_cached/torchair/result.json`.
If either is false, report the discrepancy; do not label that result cached.

The runner records commands, source SHA, physical device, logs and exit codes,
and invokes analysis after each lane. Each of the four output roots must print
`PROFILE_SUITE_COMPLETE`. Do not add `--profile` to full HR or invent helper
scripts. Existing analysis handles each root; read its JSON for comparisons
between formats. There is no automatic cross-format winner/quality decision.

Use a long-lived tool session and stay engaged until it exits. Logs stream live;
a ten-second heartbeat names the active phase and elapsed time, including setup,
compilation and export. If backgrounding is necessary, poll the active root's
`torchair.log` or `raw_eager.log` every 30–60 seconds. Continue reading the current
root when the loop advances. A heartbeat during export is not kernel progress;
do not kill a healthy export because it takes longer than inference.

If execution, conversion, replay or analysis fails, stop the sequence and retain
the command, result.json, raw capture, log, traceback and exit code. Report the
first causal failure and minimal proposed change; do not modify code or resume
on a failed NPU stream. A retry needs a new process and output root.

If hardware counters specifically are unsupported, the previously authorized
fallback is a NEW root with `PROFILE_METRICS=basic` and the same model settings.
Label it timing-only; it cannot answer MAC/vector/bandwidth questions. Do not
use it to bypass model, format, replay or analysis failures. Basic captures can
also lack types/shapes. Preserve that limitation instead of assigning kernels
by guesswork. Do not change page, batch size, precision or attention to make a
run succeed.

Regenerate a root's summary without inference using:

```bash
"$PYTHON_BIN" 21_colqwen3_4b_inference/analyze_portable_profile.py --run-root "$EXPERIMENT_ROOT/native"
```

Repeat with `fractal_nz`, `fractal_nz_cached` or `native_cached` as needed.

## Report directly in chat

Use `analysis_after_raw_eager.log` for initial roots,
`analysis_after_torchair.log` for cached roots, and their `summary.json` and
`result.json`. Read the JSON with existing file tools; no new parser is required.

1. Report source SHA, host/container, chip/device, interpreter/runtime versions,
   exact experiment/cache roots, completion status and missing counters/fallbacks.
2. Give one table: native versus NZ, with eager clean mean/p50, initial compiled
   mean/p50 and cached compiled mean/p50. Include before/after means when they
   drift. State whether NZ improves either lane and whether cached confirmation
   agrees. These are text-forward milliseconds, not full-model pages/s.
3. Give summed matmul ms/forward and each projection shape's ms/forward,
   calls/forward, kernel type/name and available pipe/memory counters for both
   formats and lanes. Include transpose, strided slice, TransData and casts if
   they change. `summary.json` retains all groups beyond the printed TOP rows.
   Report counter names/units exactly; missing values are unknown, not zero.
4. Report `weight_formats` (expected native: text 2×144, vision 2×96; NZ:
   text 29×144, vision 2×96), text options, and cache-hit records. The profiler
   JSON contains these; the analyzer also preserves formats/options in summary.
5. Compare `input_contract.vision_sha256` and `text_sha256` across all six
   result files. The analyzer checks eager/compiled hashes within each initial
   root automatically, but does NOT check between different format roots.
   A mismatch blocks an identical-input performance conclusion.
6. Report `nz_vs_native` text and final-embedding max/mean absolute difference,
   RMSE and cosine, `compiled_vs_eager`, finite outputs and same-lane replay.
   Do not claim retrieval accuracy from feature similarity or from the old
   native-weight HR accuracy result. No full-HR rerun is requested here.
7. Give paths to all four summaries and raw captures. Keep reporting in chat.

Projection-role labels are shape-derived candidates. NZ weights can appear as
four-dimensional blocked shapes, leaving role candidates empty; report those
shapes verbatim rather than inventing an assignment. An operator name alone
does not establish a kernel's implementation or core count. If launch/tiling
metadata is absent from existing outputs, say so; do not infer it from timing.
Higher vector time alone does not establish memory-bandwidth saturation.
Counter times can overlap; do not add them as a breakdown of elapsed time.

## Verified 910B NZ comparison (not a 310P prediction)

Implementation commit `cc2e8c68`, 2026-10-08, Ascend910B2 physical NPU 3,
torch/torch-npu 2.10.0, Transformers 4.57.1, same page and six-process procedure
above. All six processes completed with identical frozen-input hashes, finite
outputs and exact same-lane replay. Ten local analysis tests passed. Actual
compiled matmul inputs were ND/FRACTAL_NZ in the NZ run.

| Text weights | Eager mean | Initial compiled mean | Cached compiled mean |
|---|---:|---:|---:|
| Native | 77.67 ms | 60.67 ms | 62.21 ms |
| NZ | 78.08 ms | 63.17 ms | 63.14 ms |

NZ was 1.5–4.1% slower compiled in these comparisons. Cached kernel sums:

| Projection, summed over 36 layers | Native | NZ |
|---|---:|---:|
| QKV | 4.91 ms | 6.11 ms |
| Attention output | 4.20 ms | 3.94 ms |
| MLP gate/up | 16.06 ms | 16.06 ms |
| MLP down | 9.12 ms | 10.47 ms |
| Total matmuls | 34.30 ms | 36.58 ms |

QKV/output/down changed from MatMulV3 to MatMul in the compiled NZ profile.
This identifies a kernel-path change, not its internal bottleneck. NZ eager
matched native eager bit-for-bit. Compiled NZ final embeddings had maximum
absolute difference 0.00471377, mean absolute difference 0.0000716561,
RMSE 0.000111882 and cosine 0.999983977 versus native eager with fixed compiled
vision. Text hidden-state maximum absolute difference was 1.1328125, RMSE
0.0166635 and cosine 0.999983579. These diagnostics are not retrieval accuracy.
Native remains the default. The 310P result is still to be measured.

The 910B source followed the same simple Qwen3 reranker weight-format method:
`local_modeling_qwen3_reranker.py:prepare_reranker_linear_weight_format`, first
added in `223ff9c4`. ColQwen reuses its own existing `optimized_prefill.Linear`
constructor; no custom matmul, precision change or decode-cache optimization was
introduced. The runner's `--text-weight-format` controls only the text stage.

## Earlier full-scope profiler validation and documentation

Consulted Huawei's profiler API and CANN counter documentation:

- https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000289.html
- https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/devaids/Profiling/atlasprofiling_16_0016.html

The older PyTorch API page lists training products; it is not proof of 310P
API support. CANN documents inference-series counter collection but warns that
available fields vary by product. Reuse the installed stack that completed the earlier 310P profiling. The new
NZ runs still need their own successful capture/export and format verification.

Validated on **910B2 physical NPU 0**, 2026-10-08, with the unchanged capture
runner from `db770025`: all 12 default captures and automatic analysis passed
(three scopes × two metrics × two execution lanes). Same-lane replays were
bit-exact, frozen vision/text inputs matched between processes, and attention
counts were 24 vision / 36 text / 60 full per recorded forward. A separate
compiled-text `basic` capture and analysis also passed. This validates the
tooling on 910B, not target-chip profiler availability or 310P performance.

Clean means from that same-page 910B run:

| Scope | Compiled | Raw eager |
|---|---:|---:|
| Full model forward | 141.10 ms | 171.09 ms |
| Vision graph | 54.80 ms | 64.79 ms |
| Text graph, including production alignment/trim | 61.27 ms | 80.19 ms |

The parser was also tested with missing/NA counters, legitimate zero counters,
overlapping intervals, alternate CSV headers, malformed durations and ambiguous
projection shapes and absent Level0 operator metadata. All ten parser/accounting
tests passed. The concise final
console summary was regenerated from the real 910B CSVs. Compact validation
provenance and CSV hashes are in `references/portable_profile_910b/validation.json`.
