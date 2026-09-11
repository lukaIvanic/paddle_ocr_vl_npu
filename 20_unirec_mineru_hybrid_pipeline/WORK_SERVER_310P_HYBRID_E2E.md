# 310P: continuous PPv3 + UniRec/MinerU, halved MinerU decode B16

## Task, sequence and authority

Run experiment 20 on ONE Atlas 310P: original OmniDocBench v1.6 images ->
live PP-DocLayoutV3 -> UniRec text / MinerU tables and formulas -> page
Markdown and JSON. No MinerU layout or image/chart recognition. Preserve
experiment 18's continuous service, not page cohorts or all-text-first phases.

Luka has approved the full **1,651-page B16 reference run**. First run a
two-page smoke in a fresh root. If all checks pass, automatically continue
from offset zero through all1,651 pages in another fresh root, monitor until
actual exit, then evaluate every prediction with the frozen evaluator. No
additional384-page run or approval gate is required. Do not substitute a
165-page run or resume a suffix. A smoke pass is not a memory-fit guarantee.

Read CLAUDE.md, this experiment's README and this brief. The recent historical
references are experiment 18's `WORK_SERVER_310P_HALF_READY_FULL1651.md` and
`WORK_SERVER_310P_COMPLETED_FULL1651_ACCURACY.md`. This brief supersedes their
inference settings for this experiment, including their sequential UniRec mode.
It also supersedes this brief's original B32/384-page approval-gated version.

Inspect git status, preserve modifications, and `git pull --ff-only origin main`.
Require `git merge-base --is-ancestor 8a412cb1 HEAD` and this brief present.
The server is pull-only: no tracked-source, model or package edits, installs,
branches, commits, pushes, resets or stashes. If blocked, report the command,
first causal error and smallest proposed change; do not apply it. Generated
launch scripts, logs, metrics and predictions in a new run directory are allowed.
Do not access Luka's Mac or the 910B server. If GitHub access fails, ask Luka;
do not change repository visibility.

## Settings: 910B reference with ONLY MinerU decode batch halved

- UniRec B128, self-KV2048/cross-KV1320, K20 `310p_k20_l4`, NZ decode,
  LM-head57344, compact NPU ready capacity64.
- UniRec **streamed stages with four vision lanes**, four persistent CPU
  processes and eight resize threads per process. Its stages may overlap;
  the owner fences them before MinerU/layout. Do not silently substitute
  sequential vision from the older successful experiment-18 310P run.
- MinerU FP16, **B16/KV4096**, NPU ready capacity32/S4096, CPU capacity64.
  Active KV is **0.75 GiB**; ready KV remains **1.5 GiB**. This is NOT
  Paddle's 32/S1536 ready pool; do not copy Paddle ready-cache flags here.
- MinerU min/max pixels25088/602112 (3072 raw vision tokens), manual FP32
  vision LayerNorm + nn.Linear, compiled PromptFA vision, packed text prefill,
  NZ decode, `npu_apply` rotary and IncreFA `pse_sentinel_310p`.
- Live PPv3 converted safetensors via the shared experiment-09 compatibility
  path, layout graph capture off. Keep experiment20's inherited crop merging,
  formula margins, table placeholders and page assembly. Do NOT transplant
  the older standalone experiment11 handoff's different crop policy.
- Routes text=unirec, table=mineru, formula=mineru; decode turns32; detailed
  timing enabled. No output-length reduction, smaller active batch, resolution
  change, model unloading or fallback. The approved B32 -> B16 change is the
  only active-batch reduction; UniRec stays B128. CPU queues are not HBM ready pools.

### Code audit and expected memory effect (not NPU validation)

The existing `--mineru-batch-size 16` is sufficient; no model/scheduler edits
or global default changes are required. `streaming_decode.iter_decode_stream`
derives slot arrays, admission and compiled decode from `engine.batch_size`.
`ContinuousBatchDecodeEngine._arena_for_batch` allocates that many rows and
accepts batch sizes greater than1. The ready arena independently uses
`ready_capacity`; its32 prepared rows may feed the16 active slots across
multiple admissions. Remaining leases stay alive until copied/released.
Prefill packing stays at max32 members, vision lookahead32, and ready32.
These are not requirements for decode B32.

`torchair_cache_dir_for_shape` includes `_bs16_cache4096` in the decode cache
key. Reuse the existing parent root; allow this genuinely new graph to compile
without deleting B32 graphs. Vision/text prefill shapes are not changed by
this flag. Keep knowledge-bank processes1 during both compilation and replay.

Expected fixed active-KV saving is **805,306,368 bytes = 0.75 GiB**, from
1,610,612,736 to805,306,368 bytes. The ready allocation stays1,610,612,736 bytes.
Other batch-dependent buffers may shrink, but allocator/workspace behavior
means the external peak need not fall by exactly0.75 GiB. B16 throughput,
NPU lowering, accuracy and310P fit are UNVALIDATED until this run. Fewer
simultaneous decode rows may reduce raw tok/s; measure rather than assume
unchanged throughput. Keep all generation/context/pixel limits unchanged.

Export `CANN_KNOWLEDGE_BANK_PROCESS_NUM=1` for preflight AND every run,
including warm-cache replay. Never switch back to zero after compile. Reuse
existing compatible graph caches; this setting does not force recompilation.
Genuine missing graphs may compile normally in those roots. Do not delete,
rename or invalidate caches or run concurrent writers against the same roots.

## Preflight: recover paths, do not recreate the environment

Use an existing persistent Bash/tmux session. Source this server's validated
CANN/ATB environment before enabling `set -u`; the 910B `npu-setup` command
does not exist here. Recover the existing Python environment, actual asset
paths and cache roots from the successful experiment18 and experiment11 run
commands/manifests. Preserve a `python_nosym` executable path if used, rather
than resolving its symlink into a different environment.

One interpreter must import torch, torch_npu, torchair, transformers,
mineru_vl_utils, kornia_rs and shapely and support the reused UniRec imports.
Record its path and package/runtime versions. A missing dependency is a
blocker to report, not permission to install or switch packages. Do not try
to match a vLLM version: this is the custom pipeline.

Verify a free physical device, health, capacity and logical `npu:0` mapping.
Inspect actual CPU affinity with `os.sched_getaffinity(0)`; historical `nproc`
reported one despite a larger allowed CPU set. Reuse the validated affinity;
do not pin all producer/evaluator workers to one CPU. Do not disturb other jobs.

Verify the dataset, image hashes, PPv3 and UniRec assets against experiment18's
`references/910b_layout_cpu_de71a457/asset_hashes.json`: use UNIREC_MODEL,
LAYOUT_MODEL, OPENOCR_ROOT, dataset and images, NOT PADDLE_MODEL. Check the
OpenOCR revision and that no unvalidated `model.safetensors` shadows UniRec's
validated `model.pth`. Verify MinerU model/config/processor/tokenizer hashes
against experiment11's `references/live_paddle_full1651_910b/run_manifest_shard_00.json`
(`model_hashes`; optional ModelScope metadata is not a required asset).
Report config differences rather than editing a model to make it match.

Resolve these Bash variables to verified absolute local paths:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
# Set: PYTHON_BIN, IMAGES, DATASET_JSON, LAYOUT_MODEL, UNIREC_MODEL,
# OPENOCR_ROOT, MINERU_MODEL, U_VISION_CACHE, U_DECODE_CACHE,
# M_VISION_CACHE, M_TEXT_CACHE, M_DECODE_CACHE, PHYSICAL_NPU.
# U_DECODE_CACHE is the existing decode base expected by the runner.
# Preserve the previously validated UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE
# if present: it names the parent BEFORE decode_weight_nz_lmhead57344_semantic56371.
export CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
export TE_PARALLEL_COMPILER=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_NPU"
cd "$WORK_SERVER_REPO"
"$PYTHON_BIN" 20_unirec_mineru_hybrid_pipeline/run_pipeline.py --help
READY_CACHE_TEST_DEVICE=npu:0 "$PYTHON_BIN" -m unittest discover \
  -s 20_unirec_mineru_hybrid_pipeline/tests -v
```

All nine integration tests must pass. They include a real NPU admission-copy
check with this environment variable, but are not an end-to-end compile test.
Do not run `run_910b.sh`: it contains 910B-local paths. The three MinerU cache
roots must come from the passed 310P MinerU run, not copied `/workspace` paths.
Inspect all cache owners and do not overlap another validation job.

Before launch, run `npu-smi info` and test the existing sampler's
`query_npu_hbm('npu-smi', physical_id, 10)` from
`12_unirec_0_1b_inference/run_with_process_tree_memory.py`. Check the raw table
header: HBM versus DDR versus generic Memory Usage must be labelled accurately.
Confirm the sampler finds the intended device and a plausible baseline.

## Launch each stage once, detached and externally sampled

Use the following command in a generated Bash launcher under a new job
directory, with RUN_ROOT a not-yet-created child directory. Supply the
verified variables above in that launcher/environment; use
absolute RUN_ROOT. Save the environment choices, CPU affinity, source commit,
raw pre-run `npu-smi info` and exact shell-escaped command. Launch with the
server's existing tmux/nohup/setsid method so a tool timeout cannot kill it.
The launcher must write the child's exit status even on failure. Do not use
`set -e` in a way that skips that status write.

```bash
# First PAGE_LIMIT=2; after passing, PAGE_LIMIT=1651 and a DIFFERENT RUN_ROOT.
# Name both roots explicitly with mineru_b16 so they cannot be confused with B32.
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
command=("$PYTHON_BIN" -u 12_unirec_0_1b_inference/run_with_process_tree_memory.py
  --output "$RUN_ROOT/memory.json" --interval-ms 1000
  --npu-id "$PHYSICAL_NPU" --npu-interval-ms 1000 --
  "$PYTHON_BIN" -u 20_unirec_mineru_hybrid_pipeline/run_pipeline.py
  --input "$IMAGES" --dataset-json "$DATASET_JSON"
  --output-dir "$RUN_ROOT/output" --offset 0 --limit "$PAGE_LIMIT"
  --layout-model "$LAYOUT_MODEL"
  --unirec-model-path "$UNIREC_MODEL" --openocr-root "$OPENOCR_ROOT"
  --unirec-vision-cache "$U_VISION_CACHE" --unirec-decode-cache "$U_DECODE_CACHE"
  --unirec-batch-size 128 --unirec-ready-capacity 64
  --unirec-streamed --unirec-vision-lanes 4
  --unirec-cpu-workers 4 --unirec-cpu-threads 8
  --mineru-model-path "$MINERU_MODEL" --mineru-batch-size 16
  --mineru-ready-capacity 32 --mineru-cpu-capacity 64 --mineru-max-pixels 602112
  --mineru-vision-cache "$M_VISION_CACHE" --mineru-text-cache "$M_TEXT_CACHE"
  --mineru-decode-cache "$M_DECODE_CACHE"
  --text-model unirec --table-model mineru --formula-model mineru
  --decode-steps 32 --detailed-timing)
printf '%q ' "${command[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"
status=0
"${command[@]}" > "$RUN_ROOT/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$RUN_ROOT/exit_code.txt"
exit "$status"
```

Monitor the SAME job every 30–60 seconds: page progress, process state,
raw NPU-SMI memory and first error. Retain timestamped raw samples externally
while running; the sampler's `memory.json` is finalized after child exit.
Keep monitoring through output writing and teardown, not just the last page.
On tool timeout, inspect the existing PID/session; never launch a duplicate.
Do not send SIGUSR1. Compilation time is not throughput; report new/missing
graphs separately. A two-page pass alone does not prove table coverage or fit.

On OOM, compile failure, corruption or a failed check, STOP the sequence.
Preserve artifacts and report the first causal error, last page/operation,
device memory, active/ready allocations and command. Do not automatically
further shrink batches/pools, disable overlap, change kernels or rebuild caches.

## Completion checks and report

Use `output/run_summary.json`, NOT experiment11's `run_summary_shard_00.json`.
Require process exit0, expected page count, complete Markdown/JSON coverage of
the first N annotation-ordered images, unique crop request IDs, and all
requests completed exactly once. Verify the reported arguments/routes and
streamed UniRec execution really match the command. Check producer counts,
empty final pending work and ready pools; high-water marks must not exceed
their configured capacities. Cross-chip crop/token counts need not be exact,
but discrepancies must be reported, not silently discarded.

These summary checks mirror the completed 910B run:

```bash
"$PYTHON_BIN" - "$RUN_ROOT/output/run_summary.json" "$PAGE_LIMIT" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['pages'] == int(sys.argv[2])
assert abs(r['detailed_timing']['owner_partition_error_s']) < 1e-6
assert r['engines']['mineru']['ready_storage']['live'] == 0
assert r['engines']['mineru']['capacity'] == 16
assert r['engines']['mineru']['ready_storage']['capacity'] == 32
assert r['engines']['mineru']['ready_storage']['active_cache_bytes'] == 805306368
assert r['engines']['mineru']['ready_storage']['allocated_bytes'] == 1610612736
assert r['engines']['unirec']['compact_ready_kv']['rows'] == 0
print('COMPLETION_PASS', r['pages'], r['wall_s'], r['pages_per_s'])
PY
```

Explain results DIRECTLY TO LUKA in chat, not in a new Markdown report:

- Commit, chip/device, exact invocation/environment/cache paths and exit.
- Pages/crops by model and category; EOS/length stops; complete drain checks.
- Pipeline wall and pages/s; separately setup and externally sampled process
  lifetime/pages/s. Do not mix evaluation into inference throughput.
- Exclusive owner time split, CPU critical-path waits, event-envelope vision,
  text prefill and decode times, real/physical tokens and tok/s, decode graph
  calls/s and useful slot utilization. Do not sum nested or overlapping times.
- Ready capacity/high-water/final rows and bytes for both engines, active KV
  separately; whole-device memory baseline/peak/increase/headroom, actual
  memory-column label, sample count/errors and CPU process-tree RAM separately.
- Interesting differences versus the matched 910B references below. Label
  unavailable metrics unavailable; do not infer chip memory from torch alone.

## Full-run evaluation, after successful inference

Reuse the already successful 310P frozen evaluation environment/tools, not
the inference Python. Recover EVAL_PYTHON, EVALUATOR_ROOT and
OMNIDOCBENCH_EVAL_TOOLS_ROOT from the previous evaluation fingerprint and
command. Preserve its successful worker settings and real CPU affinity.
Do not rely on `nproc` or copy 910B's worker counts blindly. Formula CDM's
previous worker/thread bottleneck makes this check important.

Use experiment18's existing `run_completed_accuracy_eval.sh` with:

```bash
export HYBRID_OUTPUT="$RUN_ROOT/output" DATASET_JSON
# Set EVAL_JOB to a new absolute directory and create it before this call.
# Set/export EVAL_PYTHON, EVALUATOR_ROOT, OMNIDOCBENCH_EVAL_TOOLS_ROOT.
# Set/export the previously validated MATCH_WORKERS, TEDS_WORKERS, CDM_WORKERS.
export EVAL_LANE=hybrid_unirec_mineru_b16_310p_full1651
bash 18_unirec_paddle_hybrid_pipeline/run_completed_accuracy_eval.sh \
  > "$EVAL_JOB/run.log" 2>&1
```

Run detached and monitor actual worker processes/progress until exit. The
runner fixes OMP/BLAS/NumExpr/ImageMagick threads to1, pins evaluator commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`, TeX Live2025/pdfTeX1.40.28 and
ImageMagick7.1.1-47. It evaluates transformed copies and hash-checks that
original predictions remain untouched. Do not install tools, skip formulas,
drop length stops or edit predictions to improve scores. Report slow CDM
with affinity/worker evidence rather than changing evaluator semantics.

Require all1,651 pages represented, final hash checks, exit0 and actual
`HYBRID_EVAL_COMPLETE`. Inspect matching/TEDS/CDM timeout, recovery and error
counts. Report **text accuracy, Page Table TEDS, Page Formula CDM and overall**,
sample/page coverage, evaluator fingerprint and elapsed time. The legacy
`UNIREC_FULL_EVAL` output label does not mean all crops went through UniRec.

## Comparison anchors — measured on 910B, not promises for 310P

Use this experiment's committed **B32 910B** references, not standalone MinerU
scores. The new run changes both chip and decode batch; it is not a controlled
same-chip B32/B16 throughput comparison:

- `references/910b_full1651_5c3450b8/`: 1651 pages, 30557 crops; pipeline
  641.591s / **2.573290pg/s**; process721.992s / **2.286730pg/s**.
  UniRec28125 text; MinerU1681 formula and751 table. MinerU active slots86.113%.
- Whole-device sampled HBM baseline3396MiB, peak22952MiB (22.414GiB),
  increase19556MiB (19.098GiB). A higher instantaneous peak is possible.
  This was a64GiB910B; **310P fit is unproven**, especially with streamed stages.
- `references/910b_accuracy_5c3450b8/`: text95.028590, Page Table TEDS93.593239,
  Page Formula CDM96.662795, overall95.094875. 665 TEDS samples;2352 CDM
  samples/313 pages. This is NOT standalone MinerU+PPv3's95.603016 score.
- `references/910b_first384_67cd100b/`: first384 pages192.495s,1.99485pg/s.
  Compare equal subsets; don't treat that subset as full-corpus throughput.

After the smoke, continue to the approved full run. After full+eval, give one
self-contained final result and artifact locations directly to Luka, clearly
labelled 310P / MinerU B16 / ready32 / UniRec B128 streamed4. If anything fails,
report it and stop; do not claim the intended memory saving as a measured peak.
