# 310P: profile real MinerU production vision routes and compare with 910B

This is a new, self-contained handoff for Luka's pull-only 310P work agent.
Run the profiling chain below and report directly to Luka. This brief does not
inherit older approval gates or smoke/full1651/evaluation chains. Do not run a
new 384-page benchmark, a full 1651-page benchmark, or accuracy evaluation.

## Task and constraints

Collect warm vision profiles for direct S384, S768, S1536, S3072, S5632, and
packed S768, through the existing production page/crop path. First run the two
endpoints on 16 pages; if successful, run the other four routes on 32 pages.
Keep the same healthy, free physical 310P for both stages when possible.

- Read `CLAUDE.md` and `AGENTS.md` for lane rules. You cannot access Luka's Mac
  or the 910B host; neither can access your machine. Everything needed here is
  in the pulled repo or your existing successful Experiment-11 run.
- Do not edit tracked source, branch, commit, push, reset, stash, or discard
  changes. Do not modify packages, CANN, torch-npu, TorchAir, model assets or
  datasets. No Huawei-trunk installation or attention replacement in this task.
- Reuse the successful manual-FP32/PSE-sentinel run's Python environment,
  activation, and vision/text/decode cache paths. No cache deletion or fresh
  cache root. A new **output** directory is required and is not a cache root.
- Do not run concurrent jobs against any of these caches. The driver takes a
  nonblocking vision-root lock, but this does not protect against unrelated
  launchers that do not honor that lock. Check process ownership/cache users.
- Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` before torch-npu imports. Never
  use CPU/CUDA fallback, terminate another user's process, or reset the device.
- This is the custom pipeline, not a stock-vLLM test. Do not impose matching
  vLLM and vLLM-Ascend version numbers or change your working environment.
- Report issues directly to Luka in plain text: exact command, first causal
  error, phase/last unmatched marker, exit status, device and artifact paths.
  Do not propose/apply a patch or invent a workaround.
- Do not create a separate narrative report file. The driver's generated JSON,
  logs, CSVs, traces, commands and automatic parser Markdown are expected
  profiling artifacts; your interpretation belongs in the reply to Luka.

## 1. Pull and resolve your own assets

Inspect tracked changes and preserve them. Then:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git merge-base --is-ancestor 9eb55dbee314389eba43383be1185a7a02dcc0b3 HEAD
test -f 11_mineru_2_5_pro_inference/profile_production_vision_routes.py
test -f 11_mineru_2_5_pro_inference/references/vision_profiles_910b_20260905/analysis.json
```

Stop and report a pull conflict rather than discarding work. Use the Python
environment and full-run summary from your successful manual-FP32/PSE-sentinel
1651-page run (the approximately 0.17 pg/s run). Resolve and export:

```bash
# Replace these placeholders with verified existing paths on YOUR machine.
export PYTHON_BIN=/absolute/path/to/the/successful/exp11/python
export REFERENCE_SUMMARY=/absolute/path/to/successful/full1651/output/run_summary_shard_00.json
test -x "$PYTHON_BIN"
test -s "$REFERENCE_SUMMARY"
```

Find them from your saved command, successful exit file and run artifacts,
not from the 910B reference paths. Ask Luka only if multiple runs remain
genuinely ambiguous. Check 1651 completed, zero failed/skipped, and matching
manual-FP32/linear/native-D80 vision and PSE-sentinel decode metadata.

Read the 910B comparison report in
`references/vision_profiles_910b_20260905/RESULTS.md`. The bundle contains the
report and merged numerical reference, not the 910B model, caches or raw traces.

Source your server-owned CANN/environment activation from the successful run
before enabling shell nounset. Inspect NPU health and process ownership; select
one free healthy 310P into `ASCEND_RT_VISIBLE_DEVICES`. Do not assume the 910B
`npu-setup` command or `/workspace` paths exist on your server. Record hostname,
physical device/SOC, health, Python executable, torch, torch-npu and CANN versions.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export PROFILE_CONTROL="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_vision_profiles_$(git rev-parse --short=12 HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$PROFILE_CONTROL"
mkdir -p "$PROFILE_CONTROL"
```

## 2. Generate a local reference command; verify asset hashes

Do not pass the committed 910B `command.sh` to the driver: it contains foreign
absolute paths. Use the existing validated command builder on your own full-run
summary. The following creates only a generated command artifact under tmp.

The 910B profiles intentionally retained the **older matched 384-page reference
resolution**, min_pixels=25088 and max_pixels=1605632, not the newer 1103872 cap.
Use 1605632 explicitly for this diagnostic comparison only. This does not change
the production default or reverse Luka's decision to use the cap.

```bash
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os, shlex, sys
from pathlib import Path
repo = Path(os.environ['WORK_SERVER_REPO']).resolve()
sys.path.insert(0, str(repo/'11_mineru_2_5_pro_inference'))
from run_vision_timing_production import build_command
reference = json.loads(Path(os.environ['REFERENCE_SUMMARY']).read_text())
root = Path(os.environ['PROFILE_CONTROL'])
command = build_command(reference, root/'unused_driver_replaces_output', 32,
                        max_pixels=1605632)
def value(flag):
    return Path(command[command.index(flag)+1])
model = value('--model')
required = ['config.json', 'model.safetensors', 'preprocessor_config.json',
            'tokenizer.json', 'tokenizer_config.json']
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()
observed = {}
for name in required:
    observed[name] = sha(model/name)
    assert observed[name] == reference['model_hashes'][name], f'changed asset: {name}'
observed['dataset_json'] = sha(value('--dataset-json'))
assert observed['dataset_json'] == reference['model_hashes']['dataset_json']
expected_910b = {
 'config.json': '22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28',
 'model.safetensors': 'abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0',
 'preprocessor_config.json': '7070ae84a684ce2eb8d239c2cb38ff848085075784b213ea28a5ef5b3cdb445f',
 'tokenizer.json': 'dceac5fc54a795ee7570d17902b47bd05412dc2afa62bdf325c3f97fcb5b87fe',
 'tokenizer_config.json': 'd762430b9c668b5c3ad95e26626e3982d5b2a59c18ff214d621d1de4318ff376',
 'dataset_json': 'a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496',
}
assert observed == expected_910b, 'assets differ from 910B reference; report to Luka'
assert value('--images-dir').is_dir()
dataset = json.loads(value('--dataset-json').read_text())
assert len(dataset) == 1651
for sample in dataset[:32]:
    image = value('--images-dir') / Path(sample['page_info']['image_path']).name
    assert image.is_file(), f'missing image: {image}'
vision = json.loads((model/'config.json').read_text())['vision_config']
assert (vision['depth'], vision['embed_dim'], vision['num_heads']) == (32, 1280, 16)
assert command[command.index('--offset')+1] == '0'
(root/'reference_command.sh').write_text(shlex.join(command)+'\n')
(root/'asset_hashes.json').write_text(json.dumps(observed, indent=2)+'\n')
print('PROFILE_PREFLIGHT: PASS')
print(shlex.join(command))
PY
```

The preflight also checks the first 32 image paths and actual vision config.
Do not download/convert a replacement dataset. Do not
search for `.msc`, `.mv`, or require optional hub metadata such as
`configuration.json`; those are not preflight requirements here.

The builder checks all kernel/scheduler settings and that model, dataset and
cache roots exist. Confirm the printed command retains B32/KV4096, FP16,
streaming window 32, preparation depth 64, lookahead 32, pack target 768,
PromptFA, packed text prefill, and
`--local-decode-increfa-length-mode pse_sentinel_310p` (the `pse_shift` path).
Check existing usable vision graph/module cache entries; do not launch into a
known missing cache tree. Mere `.lock` files do not demonstrate a usable cache.

## 3. Endpoint profiles — 16 pages

The driver overrides only the output directory, page limit and page warmup
count. It uses real production inputs; the first call and three additional
warmups precede each route's captures. `--warmup-pages 0` inside the driver
avoids synthetic resized all-bucket warmup, not warm timing. Do not modify it.

```bash
export ENDPOINT_ROOT="$PROFILE_CONTROL/endpoints"
test ! -e "$ENDPOINT_ROOT"
nohup setsid bash -c '
  exit_file=$1; shift
  "$@"
  result=$?
  printf "%s\n" "$result" > "$exit_file"
  exit "$result"
' _ "$PROFILE_CONTROL/endpoints.launcher_exit.txt" \
  timeout --signal=TERM --kill-after=60s 10800s \
  "$PYTHON_BIN" -u \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/profile_production_vision_routes.py" \
  --reference-command "$PROFILE_CONTROL/reference_command.sh" \
  --output-dir "$ENDPOINT_ROOT" --limit 16 \
  --routes bucket_384,bucket_5632 --metrics pipe,memory \
  --baseline-steps 30 --profile-steps 3 \
  </dev/null >"$PROFILE_CONTROL/endpoints.run.log" 2>&1 &
printf '%s\n' "$!" > "$PROFILE_CONTROL/endpoints.launcher_pid.txt"
```

Send Luka the log path immediately. Monitor through actual completion before
starting the next stage. Require both the launcher and driver exit files equal
0, `PROFILE_COMPLETE exit=0 missing=[]`, and both route results present.
Inspect the validation gates in section 5 now. If the endpoints fail or a route
is absent, stop and report; do not proceed to the middle stage.

## 4. Remaining routes — 32 pages

After endpoint success, retain the activated environment, device and caches.

```bash
export MIDDLE_ROOT="$PROFILE_CONTROL/middle"
test ! -e "$MIDDLE_ROOT"
nohup setsid bash -c '
  exit_file=$1; shift
  "$@"
  result=$?
  printf "%s\n" "$result" > "$exit_file"
  exit "$result"
' _ "$PROFILE_CONTROL/middle.launcher_exit.txt" \
  timeout --signal=TERM --kill-after=60s 10800s \
  "$PYTHON_BIN" -u \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/profile_production_vision_routes.py" \
  --reference-command "$PROFILE_CONTROL/reference_command.sh" \
  --output-dir "$MIDDLE_ROOT" --limit 32 \
  --routes bucket_768,bucket_1536,bucket_3072,packed_768 --metrics pipe,memory \
  --baseline-steps 30 --profile-steps 3 \
  </dev/null >"$PROFILE_CONTROL/middle.run.log" 2>&1 &
printf '%s\n' "$!" > "$PROFILE_CONTROL/middle.launcher_pid.txt"
```

## 5. Monitor and validate both stages

Use long tool timeouts (prefer at least 10,800,000 ms / 180 minutes). Detached
launches must survive short tool limits; keep reattaching and monitoring every
30–60 seconds until they actually exit. Keep Luka informed of completed route
captures and meaningful problems. A tool timeout is not task completion.

Follow `MINERU_PHASE`, `MINERU_VISION_COMPILE`, `PROFILE_ROUTE`,
`PROFILE_CAPTURE`, and `PROFILE_COMPLETE`. Profiler parsing can be CPU-only;
baseline samples on 310P can also take substantially longer than on 910B. Do
not infer a stall from low AI Core activity or a quiet log alone. If a stage
appears stuck, inspect its process tree, last marker, file timestamps, device
errors and health; do not repeatedly relaunch or clear caches. Report an actual
failure/timeout to Luka, preserving artifacts. The outer process timeout is
three hours per stage, not permission to ignore evidence of failure for hours.

Completion gates:

- Each `<stage>.launcher_exit.txt` and `<stage>/exit_code.txt` equals 0.
- Each result has no missing routes and `diagnostic_page_throughput_valid=false`.
- Every route has 30 before and 30 after unprofiled samples, three pipe-profile
  and three memory-profile executions, and matching tags/output shape.
- Require `parity.exact=true`, `max_abs=0`, `nonfinite=0`. The driver allows a
  small allclose tolerance; if exact equality differs, report it explicitly
  and stop the chain rather than claiming the 910B exact-replay gate passed.
- For each metric's `parsed_profile_summary.json`, examine `runs[*]` and require
  usable `kernel_details`. Across the relevant capture there should be 96
  attention calls (32 layers × 3 forwards) and 384 linear calls. Check the
  actual operator names rather than assuming 310P spellings match 910B.
  Missing CSVs, empty parsing or inconsistent counts are an issue, not a
  successful profile. Retain raw files for inspection.
- Confirm nested diagnostic page outputs completed 16/32 pages respectively,
  zero failed/skipped, with the same asset hashes and production settings.
  **Never report their pg/s as throughput:** profiling and extra replays distort
  both E2E timing and route counters. No accuracy evaluation is needed.

The profiler already writes parsed summaries; use them rather than inventing
replacement kernels or operator microbenchmarks. On 910B, one packed pipe
`step_trace_time` summary covered only one forward although kernel_details had
all three; check counts before using any trace total as a denominator. Treat
profiler-overhead gaps separately from normal unprofiled host/device timing.

## 6. Explain the findings directly to Luka

The committed numerical reference is
`references/vision_profiles_910b_20260905/analysis.json`. It has one row per
route, real/member lengths, 60-sample mean/p50/p99/max, useful/physical tok/s,
and pipe/memory kernel breakdowns. Do not try to access the Mac or 910B raw
paths mentioned in the reference report.

910B reference (all times per complete 32-block execution):

| Route | Real member lengths | Warm mean ms | PromptFA ms | Linear total ms | Rotary-attributed ms | LN-attributed ms |
|---|---|---:|---:|---:|---:|---:|
| Direct S384 | 152 | 14.525 | 1.719 | 3.226 | 4.483 | 2.607 |
| Direct S768 | 640 | 17.890 | 2.662 | 4.556 | 5.171 | 2.575 |
| Packed S768 | 480, 192 | 17.605 | 2.655 | 4.526 | 5.195 | 2.570 |
| Direct S1536 | 1088 | 26.012 | 5.683 | 7.290 | 6.160 | 3.456 |
| Direct S3072 | 2160 | 48.654 | 16.726 | 13.649 | 9.031 | 4.348 |
| Direct S5632 | 5476 | 105.007 | 54.781 | 23.061 | 13.930 | 6.326 |

Report:

1. Commit, host, exact device/SOC, environment versions, verified asset hashes,
   reused cache paths, both exit codes, route coverage and replay parity.
2. For each route: member lengths, useful/padded tokens, mean/p50/p99/max event
   latency and wall latency, useful/physical tok/s, and 310P/910B latency ratio.
   Combine all 60 raw samples for percentiles (linear interpolation); do not
   average two p99s. Report before/after means separately if they drift.
3. Kernel time and percentage for attention, QKV/output projections, MLP
   projections, rotary arithmetic/slices/casts, normalization, layout transforms,
   and remaining work. Compare operator-level slowdowns with whole-region
   slowdowns. Divide sums by the verified three executions, not 32 layers.
4. Raw operator names, representative shapes/formats/dtypes, call counts,
   significant PMU indicators and conversions. Keep source/shape-based semantic
   attribution (e.g. fused LayerNorm/RoPE pieces) distinct from exact kernel-type
   totals; do not force a 910B grouping onto a different 310P fusion pattern.
5. Patch embedding, initial position preparation and merger timings separately.
   Auxiliary samples can belong to different real crops within a bucket: do
   not sum them as one exact full-vision latency unless their inputs match.
6. Profiler overhead and capture integrity. Event intervals can include device
   idle gaps; kernel-duration sums can overlap. Do not call a high busy ratio
   proof of a bottleneck or per-core bandwidth whole-device bandwidth.
7. The interesting conclusion: which operations account for the extra 310P
   time, how that changes with length, whether the evidence favors attention,
   rotary/normalization, matmuls or host overhead, and what remains uncertain.

Different layout predictions/scheduling can select different first crops on
310P. If member lengths/padding differ from the table, label the comparison
shape-matched but not input-matched; do not attribute the entire ratio to
hardware. Do not manufacture matching tensors, modify packing, add pages or
rerun automatically to chase a match. Report the discrepancy to Luka.

Reply directly in plain text (compact tables are fine), not a new report file.
Retain all generated artifacts and give their paths. Stop after reporting;
no optimization experiments, D128 padding changes, `_unpad` migration, full
benchmark, or evaluator work is authorized by this brief.
