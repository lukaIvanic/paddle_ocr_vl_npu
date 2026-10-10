# 310P MinerU E2E, first 256 pages: per-route vision and text-prefill timing

Written 2026-10-10. **This is the single active handoff for this task.** It
replaces `WORK_SERVER_310P_LIVE_PADDLE_CAP3072_FULL1651.md` (full text in Git at
`c9503cca`). Run **one lane**: that brief's **baseline** lane, unchanged except
for `--limit 256`, at the new code commit. Nothing else is compared.

## Limits

- Same as the previous baseline: approximate vision precision (innerPrecise=4),
  cap 602,112 px (min 25,088), live PP-DocLayoutV3 on NPU, no warmup pages,
  production text buckets **128/256/512/1024** (no bucket overrides, so no graph
  recompilation), `--local-prefill-metrics` on, blocking input copies, NPU grids.
- This checkout is **pull-only**: do not edit tracked files, install packages,
  commit, push or create branches. If anything fails, stop and report the
  command, error and log tail. Do not rerun with changed settings.
- No card-idle check this time: `run_lane(..., require_idle_card=False)`.
  npu-smi IDs (0/32/32768/32800) do not match `ASCEND_RT_VISIBLE_DEVICES` (0–3)
  on this server. Do not add manual npu-smi checks.

## Source

The code commit is `75491e23`. HEAD may be a later commit on this branch, but
only if that commit changed documentation alone (the second check enforces this).

```bash
git status --short            # tracked files must be clean
git fetch origin claude/mineru-text-prefill-timing
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 75491e23 HEAD
git diff --quiet 75491e23 HEAD -- '*.py'
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git rev-parse HEAD
```

## Environment

Reuse **exactly** the values of the previous handoff's baseline lane: take them
from its `host_overlap_310p_*/baseline/receipt/command.json`. If that run does not
exist, use the last successful approximate full run's `command.txt`. Same
interpreter, CANN activation (source it before `set -u`), model, layout model,
dataset, original warm caches (`CACHE_VISION` is the parent that contains
`vision_innerprecise4_<identity>`) and `ASCEND_RT_VISIBLE_DEVICES`.

```bash
export PYTHON_BIN=/absolute/path/to/the/successful/mineru/python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export CACHE_DECODE=/absolute/path/to/the/successful/decode/cache
export CACHE_TEXT=/absolute/path/to/the/successful/text/cache
export CACHE_VISION=/absolute/path/to/the/successful/vision/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=SAME_AS_PREVIOUS_BASELINE
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/prefill_timing_first256_310p_$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
mkdir -p "$RUN_ROOT"
```

## Run

Save as generated run data (not tracked source) at `$RUN_ROOT/run_baseline.py`.
It copies the warm caches outside the timer, as before. Any in-timer graph
change makes the run invalid.

```python
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
repo = Path(os.environ['WORK_SERVER_REPO'])
sys.path.insert(0, str(repo/'11_mineru_2_5_pro_inference'))
from run_host_overlap_validation import cache_artifacts
from vision_diagnostic_runner import run_lane, write_new
root = Path(os.environ['RUN_ROOT'])
git = lambda *a: subprocess.check_output(['git', *a], text=True).strip()
assert not git('status', '--porcelain', '--untracked-files=no')
assert subprocess.run(['git','merge-base','--is-ancestor','75491e23','HEAD']).returncode == 0
assert subprocess.run(['git','diff','--quiet','75491e23','HEAD','--','*.py']).returncode == 0
commit = git('rev-parse', 'HEAD')
wrapper = repo/'11_mineru_2_5_pro_inference/run_page_pipeline_vision_precision.py'
helper = repo/'09_persistent_page_engine/scripts/vision_matmul_lab.py'
identity = hashlib.sha256(wrapper.read_bytes()+helper.read_bytes()).hexdigest()[:16]
source = {k: Path(os.environ['CACHE_'+k.upper()]).resolve() for k in ['decode', 'vision', 'text']}
assert all(p.is_dir() for p in source.values())
assert (source['vision']/('vision_innerprecise4_'+identity)).is_dir(), 'Missing mode4 warm cache'
stage = root/'baseline'
stage.mkdir(exist_ok=False)
cache = {}
for key, src in source.items():
    cache[key] = root/'cache'/'baseline'/key
    shutil.copytree(src, cache[key], symlinks=False)
cmd = [sys.executable, '-u', str(wrapper), '--vision-inner-precise', '4',
       '--precision-audit', str(stage/'precision.jsonl')]
for flag, value in [
    ('--model', os.environ['MODEL_DIR']), ('--layout-model', os.environ['LAYOUT_MODEL']),
    ('--dataset-json', os.environ['DATASET_JSON']), ('--images-dir', os.environ['IMAGES_DIR']),
    ('--local-torchair-cache-dir', cache['decode']),
    ('--local-vision-torchair-cache-dir', cache['vision']),
    ('--local-text-torchair-cache-dir', cache['text']),
    ('--processor-max-pixels', 602112), ('--offset', 0), ('--limit', 256),
    ('--output-dir', stage/'output')]:
    cmd += [flag, str(value)]
cmd += ['--local-vision-grid-device', 'npu', '--local-input-transfer', 'blocking',
        '--local-prefill-metrics']
before = cache_artifacts(cmd)
write_new(stage/'cache_before.json', before)
run_lane(cmd, stage/'receipt', 3600, stage/'run.log', require_idle_card=False)
after = cache_artifacts(cmd)
write_new(stage/'cache_after.json', after)
changed = [k for k in sorted(set(before)|set(after)) if before.get(k) != after.get(k)]
write_new(stage/'cache_audit.json', dict(unchanged=not changed, changed=changed))
assert not changed, 'Graph changed during timer; result invalid, stop'
audit = [json.loads(x) for x in (stage/'precision.jsonl').read_text().splitlines()]
assert any(x.get('event') == 'configuration' and x.get('supported_310p') and
           x.get('allow_internal_format') for x in audit)
assert audit[-1].get('event') == 'completed' and audit[-1].get('mode') == 4
assert all(x['stock_converter_restored'] for x in audit if x['event'] == 'first_vision_call_finish')
out = stage/'output'
s = json.loads((out/'run_summary_shard_00.json').read_text())
assert s['git_commit'] == commit, (s['git_commit'], commit)
assert (s['completed'], s['failed'], s['skipped']) == (256, 0, 0)
assert s['processor_max_pixels'] == 602112 and s['processor_min_pixels'] == 25088
assert s['dtype'] == 'float16' and s['processor_fast']
assert s['streaming']['layout_calls'] == 256 and s['layout_backend'] == 'pp-doclayout-v3'
assert not s['saved_layout_manifest'] and not s['crop_replay_manifest']
assert not s['layout_graph_capture'] and not s['image_analysis']
assert s['warmup']['executed_pages'] == 0
assert s['local_decode_increfa_length_mode'] == 'pse_sentinel_310p'
assert s['local_compiled_vision']['layer_norm_impl'] == 'manual_fp32'
assert s['local_compiled_vision']['projection_impl'] == 'linear'
assert s['local_compiled_vision']['route_counts'].get('eager_overflow', 0) == 0
assert s['local_text_buckets'] == '128,256,512,1024'
assert s['local_prefill_metrics'] and s['local_vision_grid_device'] == 'npu'
assert s['local_input_transfer'] == 'blocking'
assert s['vision_timing']['raw_samples_file'] == 'vision_timing_shard_00.jsonl'
assert s['text_prefill_timing']['raw_samples_file'] == 'text_prefill_timing_shard_00.jsonl'
g = s['local_compiled_generation']
row = dict(chip='310P', lane='baseline', pages=256, commit=commit, run_dir=root.name,
           wall_s=s['pipeline_wall_s'], pages_per_s=256/s['pipeline_wall_s'],
           prefill_s=g['prefill_s'], decode_s=g['decode_s'],
           prefill_event_metrics=g['prefill_metrics'])
(root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
print('FIRST256_310P_RESULT '+json.dumps(row), flush=True)
```

```bash
nohup setsid bash -c '
  set +e
  "$PYTHON_BIN" -u "$RUN_ROOT/run_baseline.py" > "$RUN_ROOT/coordinator.log" 2>&1
  status=$?
  printf "%s\n" "$status" > "$RUN_ROOT/exit_code.txt"
  exit "$status"
' </dev/null > "$RUN_ROOT/launcher.log" 2>&1 &
echo "$!" > "$RUN_ROOT/pid.txt"
```

Poll every 30–60 s. Success requires `exit_code.txt` = 0, receipt exit zero and
`result.json`. The child deadline is 3,600 s. On timeout, nonzero exit, device
error, an in-timer compile or a failed assertion, stop and report. Do not rerun.

## Per-route tables

```bash
cd "$RUN_ROOT/baseline/output" && "$PYTHON_BIN" - <<'EOF'
import json, statistics as S, collections as C
key = lambda k: (k.rsplit('_', 1)[0], int(k.rsplit('_', 1)[1]) if k.rsplit('_', 1)[1].isdigit() else 1 << 30)
for f in ['vision_timing_shard_00.jsonl', 'text_prefill_timing_shard_00.jsonl']:
    g = C.defaultdict(list)
    for line in open(f):
        r = json.loads(line); g[r['route']].append(r); g['ALL'].append(r)
    print(f'\n{f}\n| route | calls | members | real tok | phys tok | device s | median ms excl 1st | mean ms excl 1st | phys tok/s | real tok/s |\n|---|---|---|---|---|---|---|---|---|---|')
    for k in sorted(g, key=lambda k: (k == 'ALL', key(k) if k != 'ALL' else ())):
        rs = sorted(g[k], key=lambda r: r['call_index']); t = sum(r['device_s'] for r in rs)
        ms = [1e3 * r['device_s'] for r in rs[1:]] if k != 'ALL' else []
        R, P = sum(r['real_tokens'] for r in rs), sum(r['physical_tokens'] for r in rs)
        med, mean = (f'{S.median(ms):.3f}', f'{S.mean(ms):.3f}') if ms else ('n/a', 'n/a')
        print(f"| {k} | {len(rs)} | {sum(r['members'] for r in rs)} | {R} | {P} | {t:.3f} | {med} | {mean} | {P/t:.0f} | {R/t:.0f} |")
EOF
```

Median/mean ms exclude each route's first call. Token rates are total tokens
divided by total `device_s` over all calls in that route. `ALL` has no latency
columns. Device seconds are event regions that include launch gaps; they are not
pure kernel time.

## Report to Luka in chat

Label everything **310P**.

1. Run directory name (`$RUN_ROOT` basename), `git rev-parse HEAD`, and
   confirmation that it equals the summary's `git_commit` and contains `75491e23`
   with no `.py` changes after it.
2. Pages 256; `pipeline_wall_s`; pg/s (256 / wall).
3. The two tables printed above, **vision** and **text prefill**, verbatim.
4. Card (`ASCEND_RT_VISIBLE_DEVICES`), interpreter path, and the source of the
   environment values (which earlier receipt).
5. Any failure, warning, missing file or deviation, with the log tail.
