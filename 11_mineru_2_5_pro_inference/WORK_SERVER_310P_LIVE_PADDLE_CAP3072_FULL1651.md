# 310P full MinerU E2E: baseline versus the measured best host configuration

Updated 2026-10-09. **This is the single active handoff for this task.** It
supersedes this file's original-versus-approximate-precision instructions.
Both lanes below use **approximate vision precision, innerPrecise=4**. The
filename is retained so existing links keep working: “live Paddle” refers only
to live PP-DocLayoutV3 layout detection, followed by the custom MinerU recognizer.

## Requested work and limits

On one healthy, free **310P**, run all **1,651 OmniDocBench pages**, first baseline,
then the best configuration recorded in the committed 910B short-ladder evidence. Same commit, interpreter, device, model, inputs and processor cap
**602,112 pixels** (min 25,088; at most 3,072 raw vision tokens per crop).
Use live PP-DocLayoutV3 on the NPU, no layout graph capture, fresh recognition,
and normal page assembly. No saved layout/crop replay, lower-cap variant,
extra precision comparison, length sweep, profiling, GIL study or worker tuning.
No full-run evaluation is needed if generation and page outputs are identical.

| Flag | Effect | Default |
|---|---|---|
| `--local-vision-grid-device cpu` (C1) | CPU grid metadata; copy only position IDs | npu |
| `--local-input-transfer pinned-nonblocking` (C2) | pinned CPU FP32 inputs, nonblocking copies | blocking |
| `--no-local-prefill-metrics` (C5) | omit prefill timing synchronization | production metrics on |
| `--local-input-transfer pinned-thread` (C2b) | dedicated transfer thread/stream, event dependency | blocking |
| `--local-compact-uint8` (C3) | resized uint8 RGB transfer; exact lookup normalization and patchification on NPU | off |
| `--prepare-workers N`, `--frontend-workers N` (C4) | CPU pool sizes; FIFO request consumption retained | 1 / 2 |

The combined lane uses **only** the flags in `best_combination.json` below;
implemented flags are not automatically accepted optimizations. The file records
the fastest measured cumulative configuration, the completed single-pass times and each
incremental comparison; it does not claim performance for untested mixtures. No further
profiling/GIL work is requested. Model math, vision/text/decode kernels, layout
and decode KV-length handling remain unchanged. Pinned sources are retained
through prefill. The H2D thread uses the experiment-09 stream/event pattern.
C3 follows its lookup-table idea but builds the table through the installed
MinerU fast processor's own normalization function to preserve exact rounding.
It retains the processor's resize, transfers a single RGB frame, then performs
normalization, temporal duplication and patchification on NPU. It is restricted
to the verified single-image RGB Qwen2-VL contract; unsupported inputs fail.

Basis: Luka explicitly requested this pair. C2 reuses the pinning/nonblocking
mechanism in experiment 09 `paddleocr_vl/serving/engine.py`; experiment 11 owns
it in `host_input_staging.py`, `native_custom_backend.py` and
`fixed_batch_engine.py`. C1 is in `local_modeling_mineru.py` and the request
finalizer. This is not an inferred weight-format or attention optimization.

The authoring session validated these host changes on **910B2**, including
200 real crops across all reachable buckets and the packed-768 route, and
64-page exact-output checks. The first 910B comparison was a 384-page baseline/C1+C2+C5 pair repeated
twice. The subsequent requested ladder uses 32 pages, five configurations in
order once (Luka cancelled the second pass); C3 has a 200-crop byte-exact gate, and the full combination
has one final 64-page token check. Receipts and the selection are in
`references/host_overlap_short_910b_20261009/`. **No 310P run was performed by that
session.** On the 32-page ladder, first-use loading of already compiled graphs remains
inside pipeline wall time. These are short E2E measurements, not steady-state
or full-1,651-page throughput estimates. The earlier relayed 310P approximate full run was 5,193.62 s /
0.3179 pages/s. It is context, not the baseline for this new comparison.

## 910B selection to carry over, subject to 310P validation

The selected flags are **C1+C2+C5**: CPU grid, pinned/nonblocking copies on the
main thread, prefill metrics off, preparation/frontend workers **1 / 2**, uint8
off. The one completed 32-page pass measured baseline **42.504 s / 0.7529 pg/s**,
C1+C2+C5 **39.187 s / 0.8166 pg/s**, +H2D thread **43.553 s / 0.7347 pg/s**,
+uint8 **43.353 s / 0.7381 pg/s**, and +two preparation workers
**42.929 s / 0.7454 pg/s**. These are cumulative lanes. The extra flags are
implemented but are not selected on this evidence. There is no validated
32-page baseline spread because the second pass was cancelled. Layout host
spans varied between lanes, so this single pass does not isolate the cause of
the slower totals or establish tiny marginal gains.

The earlier **910B** 384-page ABAB comparison supports C1+C2+C5: baseline
279.312 / 274.871 s, combined 265.461 / 266.487 s; mean throughput improvement
**4.18%**, baseline spread **4.441 s (1.60%)**. These are different page counts;
do not use the 384-page spread as a 32-page statistical threshold.
C3 passed all 200 real crops including exact FP32 processor pixels and full
encoder results. With all flags enabled, the final 64-page check matched all
**1,020 crop token sequences and all 64 Markdown outputs**. The initial C3
attempt hit Ascend's rank-greater-than-eight copy limit; removing singleton
axes fixed it without changing element order. No numerical tolerance was used.

## Access, source and environment

Read `CLAUDE.md` and `LIVE_PAGE_PIPELINE.md`. You cannot reach the author's
910B machine. This work-server checkout is **pull-only**: never edit tracked
source, install/change packages, change model/data files, commit, push, create
branches, discard changes, or repair a failed run by changing its substance.
Report any required change, exact command and error to Luka instead.

With clean tracked source:

```bash
git status --short
git fetch origin claude/mineru-host-overlap
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 6a5f932b HEAD
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git rev-parse HEAD
```

Resolve these **310P-local** paths from the last successful approximate full
run's immutable `command.txt`, precision audit and summary. Do not use the
910B `/workspace` paths, guess a Transformers version, or silently change the
environment. Preserve the successful CANN activation and its process settings;
record torch, torch-npu, Transformers, mineru-vl-utils, kornia-rs and shapely
versions. The PPv3 source-contract guard must pass; do not bypass it. The layout
checkpoint must be the converted safetensors directory with `inference.yml`.

Source the server's established CANN environment **before shell nounset**.
Check `npu-smi info`; select a healthy, unoccupied physical 310P and retain it
for both runs. Never stop another user's process. Record health, load, CPU
count, affinity, visible devices and package versions before execution.

```bash
export PYTHON_BIN=/absolute/path/to/the/successful/mineru/python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export CACHE_DECODE=/absolute/path/to/the/successful/decode/cache
export CACHE_TEXT=/absolute/path/to/the/successful/text/cache
# Parent originally supplied to the precision wrapper, NOT its innerprecise4 child:
export CACHE_VISION=/absolute/path/to/the/successful/vision/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=THE_VERIFIED_FREE_310P_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/host_overlap_310p_$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
mkdir -p "$RUN_ROOT"
```

Before launching, compare the resolved production arguments below against the
successful 310P run's command. Expected: FP16, fast processor, cap602112/min25088,
batch/page-window32, decode KV4096 with `pse_sentinel_310p`, packed text prefill,
compiled D80 PromptFA vision, ordinary linear projections, manual FP32 vision
LayerNorm, internal formats enabled, pack target768/lookahead32, preparation
prefetch64, live eager NPU PPv3, no warmup pages. Record bucket lists and all
other overrides. If that successful command differs materially, report the
difference before execution; do not silently substitute 910B settings.
Use `run_page_pipeline.py` production defaults plus the explicit overrides
below. Compare model/data hashes from the two new manifests and the previous
successful manifest. Dataset must contain all 1,651 unique existing images.

## Cache and precision contract

Reuse the successful **310P approximate** graph caches, making independent
copies for baseline and combined; never clear the originals or use hard links
or concurrent writers. The coordinator below copies files outside the timer.
`CACHE_VISION` must contain the exact `vision_innerprecise4_<identity>` child
selected by the current precision wrapper. Its identity is computed from the
wrapper and `09_persistent_page_engine/scripts/vision_matmul_lab.py`.
Keep the successful prewarm audit proving innerPrecise=4. A warm run may have
no GE-lowering entries; its configuration still must say supported_310p=true,
and each first vision call must restore the stock converter for text/decode.
**Never use `--allow-unsupported-mode4-probe`.** No precision change is being
compared here, and no numerical drift allowance applies to these host changes.

Graph artifacts are hashed before/after each lane. If any graph is newly
compiled or changed inside the measured run, retain that run as invalid and
stop/report. Do not silently rerun until a favorable number appears. If the
required warm cache or audit is missing, report the specific missing item;
cache warming needs to be resolved before this measured pair is launched.
Warm graph loading is included in pipeline wall time; setup is reported
separately. No first pages, outliers or completion work are subtracted.

## Launch the two full runs in the background

Save the following coordinator as **generated run data**, not tracked source,
at `$RUN_ROOT/run_pair.py`. It reuses the committed receipt, cache-audit and
strict comparison helpers; it adds no inference implementation.

```python
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
repo = Path(os.environ['WORK_SERVER_REPO'])
sys.path.insert(0, str(repo/'11_mineru_2_5_pro_inference'))
from run_host_overlap_validation import cache_artifacts, exact
from vision_diagnostic_runner import run_lane, write_new
root = Path(os.environ['RUN_ROOT'])
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
wrapper = repo/'11_mineru_2_5_pro_inference/run_page_pipeline_vision_precision.py'
helper = repo/'09_persistent_page_engine/scripts/vision_matmul_lab.py'
identity = hashlib.sha256(wrapper.read_bytes()+helper.read_bytes()).hexdigest()[:16]
source = {key: Path(os.environ['CACHE_'+key.upper()]).resolve()
          for key in ['decode', 'vision', 'text']}
assert all(p.is_dir() for p in source.values())
assert (source['vision']/('vision_innerprecise4_'+identity)).is_dir(), 'Missing mode4 warm cache'
selection = json.loads((repo/'11_mineru_2_5_pro_inference/references/host_overlap_short_910b_20261009/best_combination.json').read_text())
flags = {
    'baseline': ['--local-vision-grid-device','npu','--local-input-transfer','blocking','--local-prefill-metrics'],
    'combined': selection['flags'],
}
rows = []
for lane in ['baseline', 'combined']:
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    stage = root/lane
    stage.mkdir(exist_ok=False)
    cache = {}
    for key, src in source.items():
        dst = root/'cache'/lane/key
        shutil.copytree(src, dst, symlinks=False)
        cache[key] = dst
    cmd = [sys.executable, '-u', str(wrapper), '--vision-inner-precise','4',
           '--precision-audit',str(stage/'precision.jsonl')]
    for flag, value in [
        ('--model',os.environ['MODEL_DIR']), ('--layout-model',os.environ['LAYOUT_MODEL']),
        ('--dataset-json',os.environ['DATASET_JSON']), ('--images-dir',os.environ['IMAGES_DIR']),
        ('--local-torchair-cache-dir',cache['decode']),
        ('--local-vision-torchair-cache-dir',cache['vision']),
        ('--local-text-torchair-cache-dir',cache['text']),
        ('--processor-max-pixels',602112), ('--offset',0), ('--limit',1651),
        ('--output-dir',stage/'output')]:
        cmd += [flag,str(value)]
    cmd += flags[lane]
    before = cache_artifacts(cmd)
    write_new(stage/'cache_before.json',before)
    run_lane(cmd, stage/'receipt', 10800, stage/'run.log', require_idle_card=True)
    after = cache_artifacts(cmd)
    write_new(stage/'cache_after.json',after)
    changed = [k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)]
    write_new(stage/'cache_audit.json',dict(unchanged=not changed,changed=changed))
    assert not changed, 'Graph changed during timer; result invalid, stop'
    audit = [json.loads(x) for x in (stage/'precision.jsonl').read_text().splitlines()]
    assert any(x.get('event')=='configuration' and x.get('supported_310p') and
               x.get('allow_internal_format') for x in audit)
    assert audit[-1].get('event')=='completed' and audit[-1].get('mode')==4
    assert all(x['stock_converter_restored'] for x in audit if x['event']=='first_vision_call_finish')
    out = stage/'output'
    s = json.loads((out/'run_summary_shard_00.json').read_text())
    assert (s['completed'],s['failed'],s['skipped'])==(1651,0,0)
    assert s['processor_max_pixels']==602112 and s['processor_min_pixels']==25088
    assert s['dtype']=='float16' and s['processor_fast']
    assert s['streaming']['layout_calls']==1651 and s['layout_backend']=='pp-doclayout-v3'
    assert not s['saved_layout_manifest'] and not s['crop_replay_manifest']
    assert not s['layout_graph_capture'] and not s['image_analysis']
    assert s['warmup']['executed_pages']==0
    assert s['local_decode_increfa_length_mode']=='pse_sentinel_310p'
    assert s['local_compiled_vision']['layer_norm_impl']=='manual_fp32'
    assert s['local_compiled_vision']['projection_impl']=='linear'
    assert s['local_compiled_vision']['route_counts'].get('eager_overflow',0)==0
    assert s['local_prefill_metrics']==(lane=='baseline')
    assert s['local_vision_grid_device']==('npu' if lane=='baseline' else 'cpu')
    if lane=='baseline':assert s['local_input_transfer']=='blocking'
    else:
        expected=selection['settings']
        for key,value in expected.items():assert s[key]==value,(key,s[key],value)
    parity = None
    if lane=='combined':
        parity = exact(root/'baseline/output',out,stage/'exactness.json')
        assert parity['exact_pages']==1651
        for folder in ['content_lists','layout_regions']:
            a = {p.name:p.read_bytes() for p in (root/'baseline/output'/folder).glob('*.json')}
            b = {p.name:p.read_bytes() for p in (out/folder).glob('*.json')}
            assert len(a)==1651 and a==b, f'{folder}: nonidentical page data'
        base = json.loads((root/'baseline/output/run_summary_shard_00.json').read_text())
        assert s['model_hashes']==base['model_hashes']
        assert s['layout_model_hashes']==base['layout_model_hashes']
    g, st = s['local_compiled_generation'], s['streaming']
    row = dict(chip='310P',lane=lane,pages=1651,wall_s=s['pipeline_wall_s'],
        pages_per_s=1651/s['pipeline_wall_s'],exactness=parity,
        request_h2d_submit_s=st['request_h2d_submit_s'],cpu_prepare_wait_s=st['cpu_prepare_wait_s'],
        prefill_s=g['prefill_s'],decode_s=g['decode_s'],layout_host_wall_s=st['layout_host_wall_s'],
        prefill_event_metrics=g['prefill_metrics'] if s['local_prefill_metrics'] else None)
    rows.append(row)
    (root/'pair_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('FULL_310P_RESULT '+json.dumps(row),flush=True)
```

```bash
nohup setsid bash -c '
  set +e
  "$PYTHON_BIN" -u "$RUN_ROOT/run_pair.py" > "$RUN_ROOT/coordinator.log" 2>&1
  status=$?
  printf "%s\n" "$status" > "$RUN_ROOT/exit_code.txt"
  exit "$status"
' </dev/null > "$RUN_ROOT/launcher.log" 2>&1 &
echo "$!" > "$RUN_ROOT/pid.txt"
```

Poll logs/exit files every 30–60 seconds, without restarting interrupted tool
connections. Each inference child has a **10,800-second deadline**; the runner
terminates only its owned process group on timeout. Stop/report on timeout,
nonzero exit, device error, occupied card, output mismatch, or in-timer compile.
Do not rerun, change settings or continue to the next lane after a failure.
A quiet log or released card is not success: require child receipt exit zero,
coordinator exit zero and both rows in `pair_results.json`. Keep all receipts,
commands, cache hashes, precision audits, logs, manifests and generation traces.

## Reply directly to Luka, with these tables

Label every result **310P**; include commit, card, interpreter/package versions,
cache provenance, precision audit and command paths. Provide:

1. Run table: lane, pages, requests, pipeline wall seconds, pages/s, separately
   reported setup time, exact crop-token count and exact page count.
2. Wall-time breakdown table with baseline/combined columns: vision encoder,
   text prefill, decode, live layout host span, vision position preparation,
   input-copy submission, CPU-preparation wait, total prefill host span, other
   named prefill event regions, and wall time. Show all seconds.
3. Baseline-versus-prior-production configuration table, including precision,
   buckets, cap, projection/LayerNorm modes and internal-format setting.
4. A plain list of failures, missing counters, skipped work and limitations.

**C5 deliberately disables prefill event measurements.** Mark combined vision,
text-prefill and position event timings **N/A (metrics disabled)**. Retain its
available copy, CPU-wait, prefill host, decode and layout counters, but do not
interpret a smaller prefill host span as equivalent saved device work: disabling
synchronizations moves where asynchronous work is waited for. Do not enable
metrics for the primary combined run or add another full run to fill the table.
Do not add overlapping counters together; prefill includes vision/text and some
host work, and layout/CPU preparation can overlap other work. Any wall-minus-
stages residual is an estimate containing unmeasured work, not measured NPU idle.

Report measured pg/s change honestly. This one 310P pair cannot establish a
noise floor. Exactness is binary for these host changes: any mismatch is a bug,
not an acceptable precision effect. Both lanes use the same approximate math.
