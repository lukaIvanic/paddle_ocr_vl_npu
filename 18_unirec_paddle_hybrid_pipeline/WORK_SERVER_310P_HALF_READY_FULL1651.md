# 310P: full 1,651-page hybrid retry with halved NPU ready capacities

## Task and authority

Run experiment 18 on one Atlas 310P: original OmniDocBench page images -> live
PP-DocLayoutV3 -> UniRec text and Paddle tables/formulas -> page Markdown/JSON.
Run **all 1,651 pages from offset zero**, not 165 pages, not a resumed suffix,
and not saved-layout/crop replay. Luka has approved this full rerun. Monitor
the same detached job until it actually exits and validate completion, then
explain the interesting timing and memory results directly to Luka.

This is a NEW brief. It supersedes both earlier experiment-18 310P handoffs
for this attempt, particularly their knowledge-bank=0 instructions. The last
attempt already compiled and processed pages but OOMed around 364 completed
pages. Do not repeat the older smoke/replay chain or rebuild caches just
because that run OOMed. No new accuracy-evaluation job is requested here;
retain all outputs for later evaluation.

Read CLAUDE.md and experiment 18's README. Inspect status and pull with
`git pull --ff-only origin main`, preserving existing modifications. Require
`git merge-base --is-ancestor fc3774b9 HEAD` and this new brief present.
This agent is pull-only: no tracked-source/package/model edits, installs,
branches, commits, pushes, resets or stashes. If a change is needed, stop and
report the smallest proposed change, command and causal error. Generated
commands, logs, metrics and predictions under the new output root are allowed.

You cannot access Luka's Mac or the 910B server. Resolve everything on your
own 310P server from the previous hybrid run's saved command/environment and
validated assets. Do not use authoring-machine paths printed in references.
If the private GitHub repository is inaccessible, tell Luka; do not alter
visibility or invent an alternative source checkout.

## Exact changes from the previous attempt

```bash
export CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
# These arguments belong on the experiment-18 runner command:
--unirec-ready-capacity 64
--paddle-ready-capacity 32
--paddle-ready-cache-rows 32
--paddle-ready-cache-length 1536
--unirec-vision-lanes 0
```

Replace an old explicit `--paddle-ready-cache-rows 64` or `96`: leaving it
there would retain the larger physical allocation. Do not duplicate flags.
Knowledge-bank processes stay **1 during preflight and the full run**, even
with warm caches. Do not switch to 0 after compilation. This setting controls
knowledge-bank worker concurrency; it does not require compiling every graph
again. Reuse valid entries and allow genuine missing shapes to compile in
the normal production path into the same roots.

Keep everything else as in the last working hybrid attempt:

- Active decode: UniRec B128/self-KV2048/cross-KV1320; Paddle B64/KV4096.
- CPU preparation capacities: UniRec 128 / Paddle 64, unchanged. These are
  host-RAM queues, NOT the ready-KV rows we are reducing.
- Paddle ready rows: 32 x 1536, exactly 905,969,664 bytes (0.84375 GiB).
  The previous 64 x 1536 pool was 1.6875 GiB. Admission uses the existing
  grouped-copy/release path; cooperative admission fills the active arena
  across smaller ready groups without changing the graph's batch size.
- UniRec K20 `310p_k20_l4`, NZ decode weights/57344 LM-head rows, existing
  packed text prefill. Sequential vision (`--unirec-vision-lanes 0`).
- Paddle `combined_apply_pse_sentinel`, NZ vision weights, MLP width 4352,
  vision packing target 768 and existing production-group text prefill.
- PPv3 converted safetensors, existing experiment-09 layout compatibility
  path, layout graph capture off. Min/max pixels 28224/802816; unchanged
  Paddle OCR half-scaling. Do not add `--vision-promptfa-align-128`.
- Routing text=unirec, table=paddle, formula=paddle; 32 decode steps per turn,
  continuous cross-page scheduling, persistent CPU workers. No NPU-over-NPU
  concurrency, model unloading, active-batch reduction or generation truncation.

## 910B evidence — not a 310P fit guarantee

At source `3a45bf83`, both first-384 runs passed with exactly matching 4,346
crop records (tokens, text, stops and metadata). Fresh original vs half-ready:
185.498 vs 184.085 s, 2.0701 vs 2.0860 pg/s. No material throughput penalty
appeared in this pair. Torch peak allocated: 17.177 -> 16.334 GiB.
External whole-device peak: 22,859 -> 21,969 MB. The half-ready baseline was
3,419 MB, so its increase was 18,550 MB (~18.12 GiB), not 21.45 GiB on top
of the baseline. Do not add these overlapping counters together.

UniRec ready rows peaked at 128 -> 64, but logical ready cross-KV bytes peaked
at 118,996,992 bytes in BOTH runs. Almost all observed memory savings came
from Paddle's fixed pool. Do not promise a multi-GiB UniRec saving. Full details:
`references/910b_half_ready_3a45bf83/{README.md,comparison.json}`.
Later pages and 310P runtime/workspace overhead can still cause OOM.

## Preflight: reuse the established environment

Use one persistent Bash/tmux session. Source this server's existing CANN/ATB
environment before `set -u`; there is no blue-zone `npu-setup` here. Recover
the existing common interpreter that already ran PPv3 plus both recognizers.
Preserve its `python_nosym` path if used; do not resolve it to another Python.
Use a free device, verify physical/logical mapping and health with `npu-smi
info`, and never disturb another workload. Verify the existing CPU affinity
(historically 0-63); do not infer affinity from `nproc`.

Fill these variables with verified 310P-local paths, not the placeholders:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
export PYTHON_BIN=/absolute/path/to/validated/python_nosym
export UNIREC_MODEL=/absolute/path/to/unirec-0.1b
export PADDLE_MODEL=/absolute/path/to/PaddleOCR-VL-1.6
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export UNIREC_VISION_CACHE=/absolute/path/to/existing/K20/cache
export UNIREC_DECODE_BASE=/absolute/path/to/existing/decode/base
export CPUSET=0-63
export ASCEND_RT_VISIBLE_DEVICES=THE_VERIFIED_FREE_310P_ID
export CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -euo pipefail
test -x "$PYTHON_BIN"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/18_unirec_paddle_hybrid_pipeline/310p_half_ready_full1651_$(git rev-parse --short=12 HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
npu-smi info > "$RUN_ROOT/npu_before.txt"
taskset -c "$CPUSET" "$PYTHON_BIN" -c 'import os, torch, torch_npu, kornia_rs; print("affinity", sorted(os.sched_getaffinity(0))); print("knowledge_bank", os.environ["CANN_KNOWLEDGE_BANK_PROCESS_NUM"]); assert torch.npu.is_available(); torch.npu.set_device("npu:0"); print(torch.npu.get_device_name(0))' 2>&1 | tee "$RUN_ROOT/import_preflight.log"
READY_CACHE_TEST_DEVICE=cpu "$PYTHON_BIN" -m unittest discover -s 18_unirec_paddle_hybrid_pipeline/tests -v 2>&1 | tee "$RUN_ROOT/unit_tests.log"
```

Check assets against the prior successful hybrid preflight and committed
`references/910b_layout_cpu_de71a457/asset_hashes.json`, including original
1,651-image annotation order, configs/tokenizers and weights. Do not change
assets on mismatch. Confirm the common interpreter, OpenOCR checkout and
runtime versions; record them under RUN_ROOT. CPU tests are not inference.

Recover UniRec's exact passed decode cache parent and preserve the verified
`UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE` if previously required. The
parent is BEFORE `decode_weight_nz_lmhead57344_semantic56371/`; the runner
appends that variant itself. Do not accidentally nest it twice. Inspect and
report inherited experiment overrides before running; do not keep unexplained
diagnostic settings.

Paddle's four cache roots are fixed relative to this checkout:

```text
.runtime_cache/09_persistent_page_engine_torchair
.runtime_cache/09_persistent_page_engine_vision_torchair
.runtime_cache/09_persistent_page_engine_text_torchair
.runtime_cache/09_persistent_page_engine_text_packed_torchair
```

Verify the previous compiled graphs are there and no concurrent writer uses
them. Record cache paths/sizes/mtimes before/after. Do not delete, relocate or
replace caches; a new OUTPUT root is not a new cache root. No synthetic warmup
or repeated cold-cache retries. If valid caches are elsewhere and cannot be
selected by the existing interfaces, report it instead of patching source.

Verify the existing external sampler against the actual 310P memory header:

```bash
"$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/memory_preflight.log"
import os, sys
sys.path.insert(0, '12_unirec_0_1b_inference')
from run_with_process_tree_memory import query_npu_hbm
m = query_npu_hbm('npu-smi', int(os.environ['ASCEND_RT_VISIBLE_DEVICES']), 10)
print(m['raw_npu_smi'])
print('SELECTED_DEVICE_MEMORY_MB', m['used_mb'], '/', m['total_mb'])
assert m['total_mb'] > 0
PY
```

Confirm the printed device/column matches the raw table. Use its actual HBM,
DDR or Memory-Usage label in the report; `npu_hbm` is a historical JSON name.
If parsing fails, report the table and stop rather than inventing readings.

## Launch exactly one detached full run

No additional 64-page smoke is required by this brief: the previous full run
already exercised the production path, and this smaller-ready implementation
has passed real 910B first-384 validation. Start fresh from page zero.

```bash
COMMON=("$PYTHON_BIN" "$WORK_SERVER_REPO/18_unirec_paddle_hybrid_pipeline/run_pipeline.py"
  --input "$IMAGES_DIR" --dataset-json "$DATASET_JSON" --offset 0 --limit 1651
  --layout-model "$LAYOUT_MODEL" --paddle-model-path "$PADDLE_MODEL"
  --unirec-model-path "$UNIREC_MODEL" --openocr-root "$OPENOCR_ROOT"
  --unirec-vision-cache "$UNIREC_VISION_CACHE" --unirec-decode-cache "$UNIREC_DECODE_BASE"
  --text-model unirec --table-model paddle --formula-model paddle
  --unirec-batch-size 128 --paddle-batch-size 64 --decode-steps 32
  --unirec-ready-capacity 64 --paddle-ready-capacity 32
  --paddle-ready-cache-rows 32 --paddle-ready-cache-length 1536
  --unirec-vision-lanes 0 --detailed-timing --output-dir "$RUN_ROOT/output")
COMMAND=(taskset -c "$CPUSET" "$PYTHON_BIN"
  "$WORK_SERVER_REPO/12_unirec_0_1b_inference/run_with_process_tree_memory.py"
  --output "$RUN_ROOT/memory.json" --interval-ms 1000
  --npu-id "$ASCEND_RT_VISIBLE_DEVICES" --npu-interval-ms 1000 -- "${COMMON[@]}")
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/command.txt"
printf '\nCANN_KNOWLEDGE_BANK_PROCESS_NUM=1\nASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >> "$RUN_ROOT/command.txt"
nohup setsid env CANN_KNOWLEDGE_BANK_PROCESS_NUM=1 bash -c '
  root="$1"; shift
  set +e
  /usr/bin/time -f %e -o "$root/process_wall_s.txt" "$@" > "$root/run.log" 2>&1
  status=$?
  printf "%s\n" "$status" > "$root/exit_code.txt"
  npu-smi info > "$root/npu_after.txt" 2>&1
  exit "$status"
' bash "$RUN_ROOT" "${COMMAND[@]}" </dev/null > "$RUN_ROOT/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_ROOT/pid.txt"
printf 'RUN_ROOT=%s\nLIVE_LOG=%s/run.log\n' "$RUN_ROOT" "$RUN_ROOT"
```

Give Luka the live log path. Where a tool needs a timeout, use **more than
120 minutes**, preferably 14,400,000 ms (four hours), in that tool's actual
units. The job is detached; a monitoring timeout is not a failed inference
run. Reconnect to this same PID/run root, never launch a duplicate.

Monitor every 30-60 seconds until completion, with live `npu-smi` readings
as well as page counts and phase logs. Pay particular attention around
350-400 completed pages and then throughout the tail. Completion order is
not input order, so completed page 364 need not identify the same image on
two runs. Save the last completed IDs/phase when diagnosing failure.
`memory.json` is finalized only AFTER the child exits; its absence while
running is normal. Do not call 1651 page messages success before exit/gates.

## Completion gate

After the child exits:

```bash
"$PYTHON_BIN" - <<'PY' | tee "$RUN_ROOT/completion_gate.log"
import json, os
from pathlib import Path
root=Path(os.environ['RUN_ROOT']); out=root/'output'
assert (root/'exit_code.txt').read_text().strip() == '0'
r=json.loads((out/'run_summary.json').read_text())
m=json.loads((root/'memory.json').read_text())
assert m['exit_code'] == 0 and r['pages'] == 1651
assert r['routing'] == {'text':'unirec','table':'paddle','formula':'paddle'}
names=[Path(x['page_info']['image_path']).stem for x in json.loads(Path(os.environ['DATASET_JSON']).read_text())]
assert len(names) == len(set(names)) == 1651
for suffix in ('.md','.json'):
    assert {p.stem for p in (out/'predictions').glob('*'+suffix)} == set(names)
trace=[json.loads(line) for line in (out/'recognition_trace.jsonl').open()]
assert len({x['request_id'] for x in trace}) == len(trace)
for name, active, ready, cpu in [('unirec',128,64,128),('paddle',64,32,64)]:
    e=r['engines'][name]; c=e['cpu_preparation']
    assert e['capacity'] == active and e['ready_capacity'] == ready
    assert c['capacity'] == cpu
    assert c['submitted'] == c['consumed'] == sum(x['model']==name for x in trace)
    assert c['high_water_requests'] <= cpu
p=r['engines']['paddle']['ready_kv_pool']
assert p['capacity'] == 32 and p['cache_length'] == 1536
assert p['allocated_bytes'] == 905969664
assert p['high_water_active_slots'] <= 32
assert p['active_slots'] == 0 and p['free_slots'] == 32
assert p['acquisitions'] == p['releases'] == sum(x['model']=='paddle' for x in trace)
assert 0 < p['max_prompt_length'] <= 1536
u=r['engines']['unirec']['compact_ready_kv']
assert u['rows'] == u['bytes'] == 0 and u['high_water_rows'] <= 64
assert r['arguments']['unirec_vision_lanes'] == 0
assert r['detailed_timing']['enabled']
assert abs(r['detailed_timing']['owner_partition_error_s']) < 1e-6
h=m['npu_hbm']
assert h['baseline'] is not None and h['peak'] is not None and h['sample_count'] > 0
assert not h['errors'], h['errors']
print('COMPLETION_PASS', r['pages'], 'pages', len(trace), 'crops', r['wall_s'], 'seconds', r['pages_per_s'], 'pg/s')
print('UNIREC_COMPACT_READY_KV', json.dumps(u))
print('PADDLE_READY_KV', json.dumps(p))
print('DEVICE_MEMORY_MB baseline peak increase capacity', h['baseline']['used_mb'], h['peak']['used_mb'], h['peak_increase_from_baseline_mb'], h['peak']['total_mb'])
print('RAW_PEAK_NPU_SMI', h['peak']['raw_npu_smi'])
PY
"$PYTHON_BIN" 18_unirec_paddle_hybrid_pipeline/summarize_run.py "$RUN_ROOT/output" | tee "$RUN_ROOT/summary_printed.json"
```

Inspect representative text/table/formula Markdown for empty sections,
repetition or corruption. Retain original predictions; no post-hoc edits.
If a comparable previous trace exists, use the existing `compare_traces.py`
to report overlapping-crop parity, clearly labeling a partial comparison.
Do not require exact cross-chip token parity or invent an accuracy score.

## Failure handling and direct report to Luka

On OOM, graph/compiler failure or a persistent unexplained stall, preserve
logs and stop this attempt; no automatic retries, fallback, smaller active
batches, shorter KV, cache resets or model unloading. Capture the first causal
error, attempted allocation, Torch allocated/reserved, physical device capacity,
raw memory/process table, phase and completed count. CPU utilization or
elapsed time alone does not establish compile progress. Do not send SIGUSR1.
Terminate only this task's own process tree if a hang requires it, never an
unrelated workload. Do not wait all night calling an unverified stall compile.

After exit, verify the task process disappeared and memory returned toward
baseline. Explain results directly to Luka in plain text, not a narrative
Markdown report file. Include:

- Commit, runtime/device/health, resolved paths, exact settings including
  knowledge-bank=1, cache reuse/new shapes and full completion status.
- Pages, crops by model/task, wall time and pg/s; setup separately. Explain
  whether any cold work occurred inside the processing window.
- Whole-device baseline/peak/capacity/increase and post-exit usage, raw selected
  rows and process reading at peak, sample counts/errors/gaps. Keep Torch
  allocated/reserved separate. Host PSS/RSS is RAM, not device memory.
- UniRec logical ready row AND byte maxima, Paddle fixed allocation and peak
  leased rows, final leases/bytes, max prompt length. Maxima are not averages.
- CPU capacities/high-water marks (unchanged 128/64); active decode B128/B64,
  per-model slot utilization, useful/raw tok/s and graph counts; prefill
  real/physical tokens, density and execution envelopes.
- Non-overlapping critical-path owner split, CPU dependency waits and output
  work, p50/p99/max. Do not sum nested timers or call cooperative suspension
  decode overhead. Event envelopes are not necessarily kernel-active time.
- Compare with the prior 310P attempt where possible; explain whether the
  smaller pools actually fixed the full-run OOM. Label 910B numbers as 910B;
  the first-384 comparison is not a matching full-1651 speed baseline.
- Sanity/parity observations, stop reasons and exact log/output roots.

Stop after the completed result or first blocking issue. Preserve artifacts
for Luka's next decision; do not launch a separate evaluation or another run.
