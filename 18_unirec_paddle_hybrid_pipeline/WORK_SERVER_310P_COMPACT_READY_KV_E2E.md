# 310P: retry the hybrid after OOM with compact ready KV — smoke64 then full1651

## Task

Run experiment 18 on **one Atlas 310P**: original page images → live
PP-DocLayoutV3 → UniRec text / Paddle tables and formulas → page Markdown/JSON.
Run the first 64 pages as a smoke, inspect outputs, then automatically run all
1,651 pages if it passes. Monitor through actual completion. No separate
approval is required between these stages. This is the custom pipeline, not
vLLM, MinerU, the standalone UniRec service, or saved-layout/crop replay.

This is a **new handoff**, superseding `WORK_SERVER_310P_HYBRID_E2E.md` for this
retry. The previous 310P attempt reportedly compiled its graphs successfully,
then ran out of memory. Keep those caches; an OOM does not invalidate them.
Read CLAUDE.md and experiment 18's README for orientation, but **do not
execute any older handoff's runs or cache-rebuild recipes**.

The reduced-memory inference was validated on 910B at `8b60af95` (production
change `eb3264e1`); evidence and summary support are committed through
`29ee823d`, under `references/910b_readykv_8b60af95/`. Full result:
632.005 s, **2.61232 pg/s**, 30,557 crops. All per-crop token sequences, text,
stops and all page JSON/Markdown matched the previous 910B run exactly.
No weights, resolution, decode batch, generation limit, or scheduling change
was used to save memory. This is not a 310P result.

## Rules

- You can access only your own server and this Git repository. Do not try the
  authoring Mac, its /tmp files, SSH alias, or the 910B /workspace paths.
- Resolve `WORK_SERVER_REPO` with `git rev-parse --show-toplevel`. Inspect status,
  then `git pull --ff-only origin main`; preserve any existing modifications.
  Require `git merge-base --is-ancestor 29ee823d HEAD` and this new brief present.
  No tracked-file edits, package edits/installs, branches, commits, pushes,
  resets, stashes, model/config changes, or automatic workarounds.
- Use one free, healthy physical 310P. The historical server had devices 0–3;
  verify current inventory with `npu-smi info`. Do not disturb other workloads.
  There is no `npu-setup` on that server. Activate its established CANN/ATB
  environment before shell `set -u`, then select the device.
- Recover the exact environment/asset/cache paths from the previous hybrid
  attempt and its logs, plus the successful standalone Paddle/UniRec references.
  Compilation success and inference success are separate: reuse the hybrid's
  compiled graphs even though its run later OOMed. Verify paths; do not simply
  select the newest directory.
  Preserve the validated `python_nosym` executable path when applicable; never
  resolve its symlink into a base interpreter. CPU affinity was `0-63`; verify
  it, and do not infer it with `nproc` (historically reports 1).
- Both recognizers and PPv3 must import in **one existing interpreter**. If no
  existing environment satisfies that, report the missing dependencies and
  stop; do not install or silently switch runtimes. vLLM package version
  equality is irrelevant; this pipeline does not use vLLM.
- **Report directly to Luka in plain text**, including any issue. Do not write
  a separate narrative report or proposed patch. Generated launch commands,
  logs, PID/exit files, machine-readable metrics and predictions are allowed
  under the new run root. Never edit predictions to improve a score.

## Important memory and execution contract

**Only the private Paddle ready-KV storage is smaller: 64 rows x 1,536
positions, instead of 96 x 4,096.** The active Paddle decode arena remains
B64 x 4,096 and UniRec remains B128 with its original KV lengths. Admission
uses Paddle's existing grouped-copy and release-event path. Individual prefill
keeps one reusable full-size scratch cache so its compiled graph shape is
unchanged; packed prefill retains its existing redistribution.

The ready pool falls **6.75 -> 1.6875 GiB**. The 910B full run peaked at
**17.297 GiB Torch allocated / 18.557 GiB reserved**, down from
22.289 / 23.619 GiB. Its maximum Paddle prompt was **1,036 tokens**, not 1,306;
1,536 provides headroom for that measured workload. It is not a universal
input-length guarantee. An oversized prompt raises; do not truncate or
silently enlarge the pool.

**Do not equate PyTorch allocation with total device memory.** A separate
910B first-64 run sampled `npu-smi` at 22,740 MB whole-device usage (22.21 GiB),
against a 3,424 MB idle baseline (3.34 GiB). The process field was 19,367 MB
(18.91 GiB) at that sample, versus Torch 17.105 GiB allocated / 18.445 GiB
reserved. After exit it returned to baseline. These are different counter
scopes; do not add them together or assume the same overhead on 310P.

Record the actual 310P device capacity, baseline usage and existing process
list. Use the accelerator-memory label printed by its `npu-smi` (HBM/DDR/
Memory-Usage), not an assumed 910B memory type. A 64-page pass still does not
guarantee the full corpus fits; monitor both.

Try the prescribed smoke once on a free device. If it OOMs, stop and report
the phase, requested allocation, PyTorch allocated/reserved figures when
available, and raw `npu-smi` readings/process list. **Do not shrink active
batches, KV lengths or graph residency, unload models, split across devices,
or run the recognizers sequentially as separate pipelines to manufacture a
pass.** No repeated retries, cache deletion, or parameter hunting.

Required settings are the tested experiment-18 defaults, made explicit below:

- UniRec B128, self-KV/max length 2048, cross-KV 1320, K20 `310p_k20_l4`, NZ
  decoder weights and LM-head rows 57344; existing packed text prefill.
- Paddle active B64/KV4096, **ready pool 64 rows/KV1536**,
  `combined_apply_pse_sentinel`, NZ vision linear weights, MLP width 4352,
  vision pack target 768, production-group packed text prefill.
- PPv3 converted safetensors, eager NPU layout with capture **off**, the shared
  experiment-09 IndexPut compatibility path and existing mask/crop logic.
  This is PPv3, **not UniRec's PP-DocLayoutV2/B2 layout**.
- Existing min/max pixels 28224/802816; Paddle OCR-only crops retain half-scaling.
  No new resolution cap or `--vision-promptfa-align-128` flag in this run.
- <=32 decode steps per turn, full/ready-slot priority and alternating ties;
  continuous cross-page recognition, not fixed page cohorts.
- Persistent CPU recognition and page-input/crop workers, bounded queues.
  All NPU work/transfers stay on the coordinator. No NPU-over-NPU concurrency.

## Resolve local inputs and caches

Keep one persistent Bash/tmux session. Replace these placeholders with verified
310P-local paths recovered from successful runs; if ambiguous, ask Luka.

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
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -euo pipefail
test -x "$PYTHON_BIN"
taskset -c "$CPUSET" "$PYTHON_BIN" -c 'import os; print(sorted(os.sched_getaffinity(0)))'
export CHAIN_ROOT="$WORK_SERVER_REPO/tmp/18_unirec_paddle_hybrid_pipeline/310p_compact_ready_kv_$(git rev-parse --short=12 HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
git rev-parse HEAD > "$CHAIN_ROOT/commit.txt"
```

The UniRec decode base is the directory **before**
`decode_weight_nz_lmhead57344_semantic56371/`. Recover a passed B128/C1320/S2048
NZ cache and set `UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE` to that base
when needed to reuse it. The runner appends the variant directory itself. Do
not pass the leaf OM directory or nest the variant twice. Inspect and report
any inherited override rather than unknowingly taking a different cache path.

Paddle's four roots are currently fixed by `run_pipeline.py`, relative to your
checkout. They are not command-line options:

```text
.runtime_cache/09_persistent_page_engine_torchair
.runtime_cache/09_persistent_page_engine_vision_torchair
.runtime_cache/09_persistent_page_engine_text_torchair
.runtime_cache/09_persistent_page_engine_text_packed_torchair
```

Verify these contain your prior Paddle caches. If accepted caches live elsewhere,
report the paths and stop; do not copy, relocate, symlink, or patch paths yourself.
Never use MinerU caches, caches copied from 910B, or a fresh cache root. Ensure
there is no other process writing the same roots. Inventory OM paths/sizes/mtimes
before and after each run. A new **output** directory is required, not a new cache.

The expected smoke is a **cache-hit run**, since the previous attempt already
compiled its graphs. Use knowledge-bank processes 0. Inspect the cache inventory
before launch. Only if a genuine missing entry is established in advance,
use `CANN_KNOWLEDGE_BANK_PROCESS_NUM=1` for the first smoke and let the real
production path compile into the same existing root. Record why. Do not use
synthetic warmup or transplant compiled modules.

If that smoke compiled, run one fresh-process first-64 replay in a new output
directory with the same cache and knowledge-bank processes 0 before the full
run. For an unexpected compile/cache failure during an intended hit run,
stop and report instead of deleting caches or launching another compile.

Inspect inherited UniRec/Paddle experiment overrides before launch. A setting
left over from a diagnostic run is not part of this preset; report ambiguity
rather than unknowingly changing its kernels, shapes or input policy.

## Asset and import preflight

The committed manifest was read from the 910B source assets after validation.
It contains hashes only, not model files. Compare model/tokenizer/config files
and every dataset image. `.msc`, `.mv`, and ModelScope `configuration.json`
are not required. A mismatch is a reportable issue, never a reason to edit assets.

```bash
"$PYTHON_BIN" - <<'PY' | tee "$CHAIN_ROOT/preflight.log"
import hashlib, importlib.metadata as md, json, os, subprocess, sys
from pathlib import Path
ref = json.loads(Path('18_unirec_paddle_hybrid_pipeline/references/910b_layout_cpu_de71a457/asset_hashes.json').read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''): h.update(block)
    return h.hexdigest()
for variable, files in ref['assets'].items():
    root = Path(os.environ[variable]); print(variable, root)
    for name, expected in files.items():
        actual = sha(root/name)
        print(name, actual)
        assert actual == expected, f'{variable}/{name}: hash mismatch'
assert not (Path(os.environ['UNIREC_MODEL'])/'model.safetensors').exists(), 'Unexpected safetensors would shadow the validated model.pth'
assert subprocess.check_output(['git','-C',os.environ['OPENOCR_ROOT'],'rev-parse','HEAD'], text=True).strip() == ref['openocr_commit']
dataset = Path(os.environ['DATASET_JSON'])
assert sha(dataset) == ref['dataset_json_sha256']
names = [Path(r['page_info']['image_path']).name for r in json.loads(dataset.read_text())]
assert len(names) == len(set(names)) == len({Path(n).stem for n in names}) == 1651
assert set(names) == set(ref['images'])
for name in names:
    assert sha(Path(os.environ['IMAGES_DIR'])/name) == ref['images'][name], name
print('ASSET_HASHES: PASS images=1651')
import torch, torch_npu, transformers, torchvision, kornia_rs, shapely, cv2, numpy, safetensors
sys.path[:0] = ['09_persistent_page_engine', '12_unirec_0_1b_inference', os.environ['OPENOCR_ROOT']]
from pipeline.layout_frontend import OwnedLayoutFrontend
from modeling_optimized_unirec import OptimizedUniRecRunner
from tools import infer_doc_onnx
assert torch.npu.is_available()
torch.npu.set_device('npu:0')
print('device', torch.npu.get_device_name(0), 'physical', os.environ['ASCEND_RT_VISIBLE_DEVICES'])
for package in ('torch','torch-npu','transformers','torchvision','kornia-rs','shapely','numpy','safetensors'):
    print(package, md.version(package))
print('IMPORT_PREFLIGHT: PASS')
PY
READY_CACHE_TEST_DEVICE=cpu "$PYTHON_BIN" -m unittest discover -s 18_unirec_paddle_hybrid_pipeline/tests -v 2>&1 | tee "$CHAIN_ROOT/unit_tests.log"
```

Save the preflight output/CANN version/device inventory under the run root.
Before starting the model, verify that the existing external sampler parses
this server's actual memory table correctly (inspect the raw header and row):

```bash
"$PYTHON_BIN" - <<'PY' | tee "$CHAIN_ROOT/device_memory_preflight.log"
import os, sys
sys.path.insert(0, '12_unirec_0_1b_inference')
from run_with_process_tree_memory import query_npu_hbm
m = query_npu_hbm('npu-smi', int(os.environ['ASCEND_RT_VISIBLE_DEVICES']), 10)
print(m['raw_npu_smi'])
print('SELECTED_DEVICE_MEMORY_MB', m['used_mb'], '/', m['total_mb'])
assert m['total_mb'] > 0
PY
```

The sampler retains the historical field name `npu_hbm`; explain the actual
memory type from this server's header. If parsing fails or selects the wrong
column/device, report the raw table and stop; do not patch or fabricate readings.
CPU tests are not an inference pass. Environment versions may differ from 910B;
do not impose vLLM/vLLM-Ascend equality or install 910B packages to match them.

## Detached launch: smoke64, then full1651

```bash
COMMON=("$PYTHON_BIN" "$WORK_SERVER_REPO/18_unirec_paddle_hybrid_pipeline/run_pipeline.py"
  --input "$IMAGES_DIR" --dataset-json "$DATASET_JSON" --offset 0
  --layout-model "$LAYOUT_MODEL" --paddle-model-path "$PADDLE_MODEL"
  --unirec-model-path "$UNIREC_MODEL" --openocr-root "$OPENOCR_ROOT"
  --unirec-vision-cache "$UNIREC_VISION_CACHE" --unirec-decode-cache "$UNIREC_DECODE_BASE"
  --text-model unirec --table-model paddle --formula-model paddle
  --unirec-batch-size 128 --paddle-batch-size 64 --decode-steps 32
  --paddle-ready-cache-rows 64 --paddle-ready-cache-length 1536 --detailed-timing)
launch_stage() {
  local stage="$1" count="$2" knowledge="$3"
  export RUN_ROOT="$CHAIN_ROOT/$stage"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  local command=(taskset -c "$CPUSET" "$PYTHON_BIN"
    "$WORK_SERVER_REPO/12_unirec_0_1b_inference/run_with_process_tree_memory.py"
    --output "$RUN_ROOT/memory.json" --interval-ms 1000
    --npu-id "$ASCEND_RT_VISIBLE_DEVICES" --npu-interval-ms 500 --
    "${COMMON[@]}" --limit "$count" --output-dir "$RUN_ROOT/output")
  printf '%q ' "${command[@]}" > "$RUN_ROOT/command.txt"
  printf '\n' >> "$RUN_ROOT/command.txt"
  nohup setsid env CANN_KNOWLEDGE_BANK_PROCESS_NUM="$knowledge" bash -c '
    root="$1"; shift
    set +e
    /usr/bin/time -f %e -o "$root/process_wall_s.txt" "$@" > "$root/run.log" 2>&1
    status=$?
    printf "%s\n" "$status" > "$root/exit_code.txt"
    exit "$status"
  ' bash "$RUN_ROOT" "${command[@]}" </dev/null > "$RUN_ROOT/launcher.log" 2>&1 &
  printf '%s\n' "$!" > "$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nLIVE_LOG=%s/run.log\n' "$RUN_ROOT" "$RUN_ROOT"
}
launch_stage smoke64 64 0  # expected warm caches from the previous attempt
```

Give Luka the live log path immediately. Monitor and apply the gate below.
Inspect a few smoke Markdown files, including its tables and formulas; look for
empty outputs, loops, missing sections or obvious corruption. The 910B smoke
had 1,015 crops (741 UniRec text, 2 Paddle tables, 272 Paddle formulas). Report
cross-chip count differences; do not assume all output tokens must match 910B.

If the smoke compiled, use `launch_stage smoke64_replay 64 0`, gate it, and
compare its trace with smoke64 using `compare_traces.py`. Report differences
and verify cache reuse; do not repeatedly rerun in pursuit of exact tokens.
After the smoke/replay gate and sanity inspection pass:

```bash
launch_stage full1651 1651 0
```

Do not launch full inference if smoke failed, OOMed, or left an unresolved
cache/runtime issue. Never overlap the stages or use different settings to pass.

## Completion gate and metrics (run for each completed stage)

Keep `RUN_ROOT` pointing at the stage; set `EXPECTED_PAGES` to 64 or 1651.

```bash
export EXPECTED_PAGES=64  # 1651 for full1651
"$PYTHON_BIN" - <<'PY'
import json, os, statistics
from pathlib import Path
root=Path(os.environ['RUN_ROOT']); out=root/'output'; n=int(os.environ['EXPECTED_PAGES'])
assert (root/'exit_code.txt').read_text().strip() == '0'
assert json.loads((root/'memory.json').read_text())['exit_code'] == 0
r=json.loads((out/'run_summary.json').read_text())
assert r['pages'] == n
assert r['routing'] == {'text':'unirec','table':'paddle','formula':'paddle'}
p=r['page_preparation']
assert all(p['counts'][k] == n for k in ('submitted','input','detected','crops'))
assert p['high_water_pages'] <= 2
assert p['worker_threads_observed'] == {'input':1,'crops':1}
names=[Path(x['page_info']['image_path']).stem for x in json.loads(Path(os.environ['DATASET_JSON']).read_text())][:n]
for suffix in ('.md','.json'):
    assert {x.stem for x in (out/'predictions').glob('*'+suffix)} == set(names)
trace=[json.loads(line) for line in (out/'recognition_trace.jsonl').open()]
assert len({x['request_id'] for x in trace}) == len(trace)
for model, capacity in (('unirec',128),('paddle',64)):
    e=r['engines'][model]; c=e['cpu_preparation']
    assert e['capacity'] == capacity
    assert c['submitted'] == c['consumed'] == sum(x['model']==model for x in trace)
    assert c['high_water_requests'] <= c['capacity']
    assert c['worker_threads_observed'] == 1
pool=r['engines']['paddle']['ready_kv_pool']
assert pool['capacity'] == 64 and pool['cache_length'] == 1536
assert pool['allocated_bytes'] == 1811939328
assert pool['high_water_active_slots'] <= 64
assert pool['acquisitions'] == pool['releases'] == sum(x['model']=='paddle' for x in trace)
assert pool['active_slots'] == 0 and pool['free_slots'] == 64
assert 0 < pool['max_prompt_length'] <= 1536
t=r['detailed_timing']
assert t['enabled'] and abs(t['owner_partition_error_s']) < 1e-6
m=json.loads((root/'memory.json').read_text())
h=m['npu_hbm']
assert h['baseline'] is not None and h['peak'] is not None and h['sample_count'] > 0
if h['errors']:
    raise RuntimeError('External device-memory sampling errors: ' + repr(h['errors']))
print('READY_KV_POOL', json.dumps(pool))
gaps = [b['elapsed_s'] - a['elapsed_s'] for a, b in zip(h['samples'], h['samples'][1:])]
print('EXTERNAL_DEVICE_MEMORY', json.dumps({
    'physical_npu':h['physical_npu'], 'sample_count':h['sample_count'],
    'baseline_MB':h['baseline']['used_mb'], 'peak_MB':h['peak']['used_mb'],
    'capacity_MB':h['peak']['total_mb'],
    'increase_MB':h['peak_increase_from_baseline_mb'],
    'requested_interval_ms':h['interval_ms'],
    'actual_sample_gap_median_s':statistics.median(gaps) if gaps else None,
    'actual_sample_gap_max_s':max(gaps) if gaps else None,
}))
print('RAW_PEAK_NPU_SMI')
print(h['peak']['raw_npu_smi'])
print('HYBRID_COMPLETION: PASS', 'pages',n,'crops',len(trace),'wall_s',r['wall_s'],'pg_s',r['pages_per_s'])
PY
"$PYTHON_BIN" 18_unirec_paddle_hybrid_pipeline/summarize_run.py "$RUN_ROOT/output"
```

## Monitoring, failure handling, and reply

Set the tool execution timeout above 120 minutes (prefer 14,400,000 ms / four
hours, using that tool's units). Detached jobs survive tool sessions; reattach
to the **same job**, not another run. Inspect every 30–60 seconds until exit
and the gate pass. Keep Luka informed. Watch `HYBRID setup_finish`,
`HYBRID page_finish`, exceptions, first-call cache messages, and the child-written
exit file. The memory wrapper writes `memory.json` **after its child exits**, not
continuously; its absence during execution is normal. During the run inspect
`npu-smi info` directly as well. The wrapper samples throughout setup and
inference; query latency may make actual sampling slower than 500 ms. Save a
post-exit `npu-smi info` snapshot too, checking that the task's process disappears
and memory returns toward baseline. Out-of-order page completion is normal.

If progress is quiet, inspect the process, memory, compiler activity and last
phase message. CPU usage or elapsed time alone does not prove compilation is
progressing. Do not send SIGUSR1, clear caches, restart, or wait all night while
calling a stall "slow compilation". If stalled/failed, preserve artifacts and
report the exact command, phase, first causal error, status and relevant logs.
Terminate only this task's own job if necessary; never another workload.

After full completion, **read and explain** `output/run_summary.json` and the
printed summary directly to Luka. No narrative report file. Include:

- commit, device/health, runtime versions, verified asset hashes and resolved paths;
- smoke/full completion, crop counts by model/task, sanity observations and cache reuse;
- full `pages / wall_s`, setup separately, and including-setup rate; no invented
  PDF timing or comparison with UniRec's warmup-excluded hot service metric;
- layout owner wall, both prefill/decode owner walls, and detailed timing:
  non-overlapping owner scopes, CPU-dependency wait sets, output work, and
  count/mean/p50/p99/max. Nested inclusive timers are not additive. Legacy
  pause-inclusive engine timers are not exclusive decode or CPU overhead;
- per-model useful/raw decode tok/s, graph calls, slot utilization, text/vision
  real/physical tokens and packing density; explain the different timing bases;
- frontend CPU service, worker counts/high-water marks, recognition CPU service;
  overlapping worker spans are not additive to total wall;
- length/KV-cap/repetition stops; ready-pool capacity/bytes, high-water leases,
  final live leases, acquisitions/releases and maximum prompt length;
- Torch peak allocation/reservation and independent device-memory baseline,
  sampled whole-device peak/capacity, increase above baseline, process-memory
  reading at that peak, and post-exit reading. Include the raw selected-device
  rows, sample count, actual sample gaps and any errors. PSS/RSS are host RAM,
  not accelerator memory. Do not add overlapping memory counters;
- Paddle admission device-event and enqueue time; a shorter row's strided copy
  can be slower despite moving fewer bytes. The 910B full admission envelope
  was 2.381 seconds (previously 0.972), with about 4.99 GiB saved at peak;
- the interesting differences versus the committed 910B metrics: 2.61232 pg/s,
  UniRec/Paddle utilization 88.869%/69.650%, with chip labels and no assumed ratio;
- absolute log/output roots and any unresolved issue.

This handoff repeats the prior smoke/full task with the now-validated smaller
ready pool and detailed memory/timing reporting. The OOM's resolution is a
question for this 310P run, not assumed from 910B. It does **not** launch a new
accuracy evaluation or infer a hybrid accuracy score
from the separate models' results. Preserve all per-crop token/text traces and
page outputs for the subsequent frozen TeX-2025/ImageMagick evaluation task.
Stop after reporting the full result, or the first blocking issue.
