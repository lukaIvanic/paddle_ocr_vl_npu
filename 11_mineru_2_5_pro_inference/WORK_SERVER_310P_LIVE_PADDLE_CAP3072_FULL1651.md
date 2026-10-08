# 310P: live PP-DocLayoutV3 + MinerU cap3072, original versus approximate vision precision

This is a new execution handoff for Luka's pull-only 310P work agent. It replaces
older native-MinerU-layout or selective-replay instructions for this task.
Luka authorizes original-precision revalidation followed by approximate vision
precision: a two-page smoke, full 1,651-page run and evaluation for each mode.
Do not ask for another approval after a passing gate. This updates the handoff
that existed in `d4e7fdd`; that commit was not itself a receipt for 0.289 pg/s.

**This is the primary full E2E handoff.** It includes original-versus-approximate
vision throughput comparisons by length from the two actual full-page runs.
`VISION_LENGTH_310P_HANDOFF.md` is a supplementary fixed-input encoder replay
sweep; it is not launched here and is not a prerequisite. This brief reuses only
that script's CPU crop-selection command for cache warming. Its new --resume
option applies to the separate sweep, not these full-page jobs.

## Goal and scope

Run **all 1,651 original OmniDocBench images → live PP-DocLayoutV3 → crops →
custom MinerU recognition → Markdown**, on one 310P. Use
`--processor-max-pixels 602112` (3,072 raw vision tokens), min_pixels 25,088.
This cap applies to all recognition categories, including tables. Do not change
crop geometry, skip tables, add retries, or implement repetition workarounds.
Luka will use overall, page text accuracy, Page Table TEDS and Page Formula CDM
as the main comparison anchors. Report failures/regressions honestly, without
changing the experiment to optimize its scores.

This is **not** vLLM-Ascend and **not** the selective crop replay. Do not use
`--saved-layout-manifest`, `--crop-replay-manifest`, saved 910B crops, or the
Paddle recognizer. Every page gets live layout detection and fresh recognition.
Image/chart analysis remains off, as in the reference. Input is images, not
PDF originals; do not include or invent PDF rendering throughput.

Authoring validation on **910B2, 2026-10-08**, used a clean detached
`d4e7fdd0efd90bf6408c21efab320cc3292f07a4` checkout and the settings in this
brief. The full live run completed 1,651/1,651 pages and 32,051 crops, zero
failed/skipped pages: **1,685.602498 s pipeline wall, 0.979472 pages/s**;
setup 30.293181 s separately. It processed 16,809,232 actual vision tokens
in 378.759687 s of full-encoder event regions (**44,379.7 useful tokens/s**),
with 2,077,808 padding positions / 18,887,040 physical positions (**11.0012%**).
No warmup pages or first-use cached-graph loads were subtracted.
Evidence: `references/d4_cap3072_reproduction_910b_20261008/`.
Its complete 910B evaluation scored text **96.234624**, Page TEDS **93.520375**,
Page CDM **97.054051**, overall **95.603017**. All 1,651 pages were matched;
665 table and 2,352 formula samples, with zero evaluator timeouts/errors.
26 recognition crops hit their existing output length limits and remain in the
evaluation. This is a 910B reference, not a 310P quality guarantee.

The production fast-processor prewarm passed all seven reachable vision buckets
on 910B. The mode4 two-page compatibility attempt reached all 32 GE vision
PromptFA lowerings, each audited as innerPrecise=4, then CANN rejected bucket768:
`not support APPROXIMATE_COMPUTATION when curShortSocName is Atlas A2`
(`prompt_flash_attention_tiling.cpp:3924`). No approximate 910B throughput or
full-path validation exists. This failure does not predict 310P behavior; the
310P mode4 prewarm, smoke, full run and evaluation below remain required.

## Shared Paddle layout implementation and 310P reference

This does not reimplement the detector. `paddle_layout_source.load_paddle_frontend`
imports the exact `OwnedLayoutFrontend` from experiment 09. Compare these source
locations if there is a layout problem:

- `09_persistent_page_engine/scripts/run_omnidocbench.py`: Paddle's construction
  of `OwnedLayoutFrontend`, with its chosen device and graph-capture flag.
- `11_mineru_2_5_pro_inference/paddle_layout_source.py`: same constructor,
  logical `npu:0`, graph capture explicitly false for this run.
- `09_persistent_page_engine/pipeline/layout_frontend.py`: both use default
  `model_backend="transformers"`, checkpoint dtype (`model_dtype=None`) and
  `npu_indexput_compat=True`. Model loading, detector normalization, final-head
  decoder optimization, mask/polygon processing and rectangle fast path are
  shared. Both entrypoints disable NPU JIT compilation before model setup.
- `09_persistent_page_engine/pipeline/layout_model_runtime.py`:
  `_install_pp_doclayout_v3_npu_indexput_compat` installs the shared forward
  which builds/caches static shape metadata without NPU scalar IndexPut and
  replaces top-k advanced indexing with `gather`. Its source-contract guard
  raises if the installed Transformers forward is incompatible; do not bypass it.
- `WORK_SERVER_310P_EXP09_LADDER.md`, section **Current 310P layout route:
  eager NPU**, records the established settings: `--layout-device npu`,
  `--no-layout-graph-capture`, `npu_indexput_compat=true`. Prior ACLGraph failure
  107027 is not evidence that eager NPU layout fails. Do not repeat capture
  probes or revert to an older CPU-layout diagnostic recipe.

The downstream policy intentionally remains MinerU's validated hybrid policy:
no Paddle text half-scaling, formula-margin trimming or crop-image concatenation.
Those are recognition-input differences, not missing detector compatibility
fixes. Do not import them into this benchmark as a workaround.

If layout setup/inference fails, inspect the last successful Paddle command on
your own server: interpreter, Transformers/torch-npu versions, checkpoint path,
layout device, graph flag and IndexPut workaround. Include any differences in
the plain-text issue reply. The two projects share source, but different Python
environments can load different framework implementations. Do not silently
switch environments, packages, dtype, graph mode or source to force a pass.

## Rules and environment discovery

- Read `CLAUDE.md`, `AGENTS.md`, and `LIVE_PAGE_PIPELINE.md`. The Mac/910B
  environments and their files are inaccessible from your server; do not use
  SSH to them or copy their `/workspace` paths.
- The checkout is pull-only. Never edit tracked files, packages, `/vllm-workspace`,
  model configs/weights, datasets or evaluator source. No commits, pushes,
  branches, resets, stashes or discarding user changes.
- Inspect `git status --short`; preserve tracked changes. With clean tracked
  source, `git fetch origin codex/mineru-vision-length-sweep`, then
  `git checkout --detach FETCH_HEAD`. Require
  `git merge-base --is-ancestor 0e794023 HEAD`; record actual HEAD. Do not
  switch a dirty checkout. The baseline production files must match d4e7fdd:
  `git diff --exit-code d4e7fdd HEAD -- 11_mineru_2_5_pro_inference/run_page_pipeline.py 11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py 11_mineru_2_5_pro_inference/vision_prefill_compile.py 11_mineru_2_5_pro_inference/local_modeling_mineru.py 11_mineru_2_5_pro_inference/fixed_batch_engine.py 11_mineru_2_5_pro_inference/streaming_decode.py 11_mineru_2_5_pro_inference/text_prefill_compile.py 11_mineru_2_5_pro_inference/paddle_layout_source.py`.
- Resolve `WORK_SERVER_REPO` using `git rev-parse --show-toplevel`.
- Reuse the successful custom MinerU 310P Python/CANN environment and its three
  existing cache paths. Resolve them from a successful run's command/summary:
  `local_torchair_cache_dir`, `local_vision_torchair_cache_dir`, and
  `local_text_torchair_cache_dir`. Resolve repo-relative paths against your
  checkout. Do not clear caches or run concurrent cache owners. Original
  precision reuses those cache roots. The explicitly authorized new precision
  uses a separate vision subdirectory selected by the wrapper below; text and
  decode keep their original caches. A new output directory is required.
- Find the **converted PP-DocLayoutV3 safetensors folder** from the successful
  Paddle pipeline. It needs `inference.yml`, config/processor JSON and weights.
  A Paddle `.pdparams` directory is not a substitute. Verify hashes below.
- Reuse existing dependencies. The live frontend needs kornia-rs and shapely
  in the MinerU interpreter (910B used 0.1.14 / 2.1.2), plus the already used
  torch/torch-npu/transformers stack. Record versions; do not require vLLM and
  vLLM-Ascend version equality or install/change them. If imports are missing,
  stop and report exactly which dependency/path is missing; do not install it.
- Use one healthy free physical 310P; never terminate another user's process
  or fall back to CPU/CUDA. Source your established server-owned CANN activation
  **before** enabling shell nounset. Then select the device and export
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing torch-npu.
- Generated commands, PID/exit files, logs, predictions and evaluator artifacts
  under the run root are allowed. **Report directly to Luka in plain text**;
  do not write a separate narrative report file. If blocked, stop and report
  the issue, command, error and logs; do not propose or apply a code change.

Use one persistent Bash/tmux coordinator session throughout. Resolve and export
these variables to real **310P-local** paths, not the placeholders shown:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
export PYTHON_BIN=/absolute/path/to/verified/mineru/python
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV3_safetensors
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export CACHE_DECODE=/absolute/path/to/existing/decode/cache
export CACHE_VISION=/absolute/path/to/existing/vision/cache
export CACHE_TEXT=/absolute/path/to/existing/text/cache
export OMNIDOCBENCH_EVALUATOR_ROOT=/absolute/path/to/frozen/OmniDocBench/evaluator
export OMNIDOCBENCH_EVAL_PYTHON=/absolute/path/to/verified/eval/python
export OMNIDOCBENCH_EVAL_TOOLS_ROOT=/absolute/path/to/existing/2025/eval/tools
# Select the verified free physical device, not an assumed device number:
export ASCEND_RT_VISIBLE_DEVICES=THE_VERIFIED_FREE_310P_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
set -euo pipefail
```

If several possible model/environment paths remain ambiguous after inspecting
the existing successful run artifacts, ask Luka to identify the right one.

## Preflight: assets, packages and evaluator

The following compares to committed 910B **asset hashes**, not software versions
or inaccessible 910B paths. `.msc`, `.mv` and ModelScope `configuration.json`
are **not required** by this preflight. Do not modify a checkpoint to force a
hash match; report mismatches.

```bash
"$PYTHON_BIN" - <<'PY'
import hashlib, importlib.metadata as md, json, os
from pathlib import Path
ref = json.loads(Path('11_mineru_2_5_pro_inference/references/live_paddle_full1651_910b/run_manifest_shard_00.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
for key, hashes in [('MODEL_DIR', ref['model_hashes']), ('LAYOUT_MODEL', ref['layout_model_hashes'])]:
    root = Path(os.environ[key]).resolve()
    print(key, root)
    for name, expected in hashes.items():
        if name in ('dataset_json', 'configuration.json'): continue
        actual = sha(root/name)
        print(name, actual)
        assert actual == expected, f'{key}/{name}: hash mismatch'
dataset_path = Path(os.environ['DATASET_JSON'])
assert sha(dataset_path) == ref['model_hashes']['dataset_json']
rows = json.loads(dataset_path.read_text())
names = [Path(r['page_info']['image_path']).name for r in rows]
assert len(names) == len(set(names)) == len({Path(n).stem for n in names}) == 1651
assert all((Path(os.environ['IMAGES_DIR'])/n).is_file() for n in names)
for key in ('CACHE_DECODE', 'CACHE_VISION', 'CACHE_TEXT'):
    path = Path(os.environ[key]).resolve(); assert path.is_dir(); print(key, path)
import torch, torch_npu, kornia_rs, shapely, transformers
assert torch.npu.is_available()
print('device', torch.npu.get_device_name(0), 'visible', os.environ['ASCEND_RT_VISIBLE_DEVICES'])
for package in ('torch','torch-npu','transformers','mineru-vl-utils','kornia-rs','shapely'):
    print(package, md.version(package))
print('LIVE_ASSET_PREFLIGHT: PASS')
PY

source 09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
test "$(git -C "$OMNIDOCBENCH_EVALUATOR_ROOT" rev-parse HEAD)" = 2b161d010d2e3aff77a0edef359ea3a6411d23cd
"$OMNIDOCBENCH_EVAL_PYTHON" 09_persistent_page_engine/scripts/verify_omnidocbench_eval_runtime.py \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"
```

Require the established TeX Live 2025/pdfTeX 1.40.28, ImageMagick 7.1.1-47,
Ghostscript 9.55.0, CJK support and runtime `status: pass`. Do not substitute
ambient TeX Live 2022 or ImageMagick 6. Record CANN/driver/device health too.

## Original precision: smoke, then full run

### Precision-pair method (new, explicitly requested)

Keep the original page/crop/scheduler settings below. Original precision is the
stock GE PromptFA innerPrecise=1 path, FP16 D80; approximate precision is the
previously tested **310P** innerPrecise=4 path. It does not change weights,
vision LayerNorm, layout precision, text prefill or decode. It is not BF16.
The reference 310P3 full-32-block matrix reported +10.9% useful tok/s at
640/672 useful tokens (bucket768), and +35.2% at 5476 tokens (bucket5632 layout
image). The latter is valid historical evidence beyond this recognition cap.
No corresponding retained 1k–3k approximate-crop measurements exist.

The new experimental entrypoint is `run_page_pipeline_vision_precision.py`.
It reuses the owned `_register_promptfa_inner_precise_converter(4)` from
`09_persistent_page_engine/scripts/vision_matmul_lab.py`, scopes it to each
vision graph's first call, and restores the exact original converter in a
finally block. Text-prefill and decode retain stock converters. It records
each cold GE PromptFA lowering's actual requested precision in an immutable
JSONL audit. A warm-cache load may have no new GE-lowering rows; retain the
earlier prewarm audit proving how that cache was built.

This process-local interception is an experimental implementation choice,
not a claim that upstream MinerU exposes this precision flag. The pipeline has
one NPU owner thread; do not run concurrent compilation threads in this process.
No tracked production implementation or default is changed. Wrapper metadata
is a sidecar: run_summary alone does not establish the precision mode.

New precision requires its own vision-cache namespace, derived from the wrapper
and converter helper hashes. This is an explicit exception to the old brief's
ban on new cache roots. **Warm all seven reachable buckets before the measured
full runs**, using actual crops with the production fast processor. Otherwise a
new-precision full run could include cold compilation while the original reused
compiled graphs. First-use cache loading still remains inside each page run,
as in the original zero-warmup-page benchmark. Do not subtract it after the fact.

After COMMON, CHAIN_ROOT, the cache lock and launch_stage below are defined,
but **before the original smoke**, run CPU selection and the original-precision
real-crop prewarm. Run the approximate prewarm at its later step, after original
full inference and evaluation. The prewarm flag exits without page inference;
these are setup artifacts, never throughput results. Preserve shell variables
in the same coordinator session. Poll each job to durable exit 0 before the next.

The executable cache-warming sequence is placed after launch_stage below.
Original precision warms first; approximate precision warms only after original
full inference and evaluation complete.


Selection deadline 900 seconds; each prewarm deadline 3600 seconds, with short
polls. The launch_stage function below uses vision_diagnostic_runner.py to enforce
these deadlines and capture before/after health/load/CPU/job state. It stops
only the owned child process group on timeout. Preserve logs and report; do not
retry or substitute kernels. Keep runner.log and receipt/*.json with run.log.
After prewarming, require every reachable bucket's compile record and all
seven prewarm_crop_complete entries in both audits. Original and approximate
cache directories must differ; original text/decode paths must match.

Omit `--allow-unsupported-mode4-probe` on 310P. Approximate vision is selected by
`--vision-inner-precise 4`, which needs no device-guard override on a recognized
310P. The override only bypasses our script's device check for a deliberately
labelled two-page 910B attempt; it cannot add CANN support. The actual 910B
attempt was rejected by CANN as recorded above.

Use `run_page_pipeline.py`; do **not** use the 910B-specific shell launcher,
which has 910B paths and `npu-setup` assumptions. Its production defaults are
the validated B32/KV4096 stream, page window 32, prepare depth 64, manual-FP32
vision LayerNorm + nn.Linear, PromptFA, packed text and PSE-sentinel IncreFA
(`pse_shift`), NZ decode weights and NPU rotary. Do not alter them.

The layout checkpoint remains FP32/eager with layout graph capture **off**;
MinerU remains FP16. If this exact detector path is unsupported on your 310P,
report it rather than silently switching layout precision or hardware.

```bash
export CHAIN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_live_paddle_cap3072_$(git rev-parse --short=12 HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
git rev-parse HEAD > "$CHAIN_ROOT/commit.txt"
printf 'host=%s\nphysical_npu=%s\n' "$(hostname)" "$ASCEND_RT_VISIBLE_DEVICES" > "$CHAIN_ROOT/environment.txt"
exec 9>"$WORK_SERVER_REPO/.runtime_cache/11_mineru_2_5_pro_inference/serving_validation.lock"
flock -n 9 || { echo 'Existing MinerU cache owner is busy'; exit 2; }

COMMON=("$PYTHON_BIN" "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_page_pipeline.py"
  --model "$MODEL_DIR" --layout-model "$LAYOUT_MODEL" --layout-backend pp-doclayout-v3
  --no-layout-graph-capture --images-dir "$IMAGES_DIR"
  --processor-min-pixels 25088 --processor-max-pixels 602112
  --warmup-pages 0 --no-resume --fail-fast
  --local-torchair-cache-dir "$CACHE_DECODE"
  --local-vision-torchair-cache-dir "$CACHE_VISION"
  --local-text-torchair-cache-dir "$CACHE_TEXT")

launch_stage() {
  local stage_root="$1"; shift
  test ! -e "$stage_root"
  mkdir -p "$stage_root"
  printf '%q ' "$@" > "$stage_root/command.txt"
  printf '\n' >> "$stage_root/command.txt"
  ln -s receipt/run.log "$stage_root/run.log"
  nohup setsid bash -c '
    root="$1"; shift
    set +e
    /usr/bin/time -f %e -o "$root/process_wall_s.txt" \
      "$PYTHON_BIN" "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/vision_diagnostic_runner.py" \
      --output-dir "$root/receipt" --timeout-s "${STAGE_DEADLINE_S:-7200}" -- "$@" \
      > "$root/runner.log" 2>&1
    status=$?
    printf "%s\n" "$status" > "$root/exit_code.txt"
    exit "$status"
  ' bash "$stage_root" "$@" </dev/null > "$stage_root/launcher.log" 2>&1 &
  printf '%s\n' "$!" > "$stage_root/pid.txt"
  printf 'LOG=%s/run.log\n' "$stage_root"
}
```

```bash
export STAGE_DEADLINE_S=900
launch_stage "$CHAIN_ROOT/select_crops" "$PYTHON_BIN" \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/vision_length_sweep.py" select \
  --model "$MODEL_DIR" --processor fast --per-bucket 1 --output "$CHAIN_ROOT/selection.json"
while test ! -f "$CHAIN_ROOT/select_crops/exit_code.txt"; do sleep 15; done
test "$(cat "$CHAIN_ROOT/select_crops/exit_code.txt")" = 0 || exit 1
# Inspect all seven actual token grids before proceeding.
export STAGE_DEADLINE_S=3600
prewarm_precision() {
  local precision="$1"
  export STAGE_DEADLINE_S=3600
  launch_stage "$CHAIN_ROOT/prewarm_$precision" "$PYTHON_BIN" \
    "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_page_pipeline_vision_precision.py" \
    --vision-inner-precise "$precision" \
    --precision-audit "$CHAIN_ROOT/prewarm_$precision/precision.jsonl" \
    --prewarm-only "$CHAIN_ROOT/selection.json" "${COMMON[@]:2}" \
    --input-images "$IMAGES_DIR/page-573c437e-c309-4483-a038-ef2f440b104a.png" \
    --limit 1 --output-dir "$CHAIN_ROOT/prewarm_$precision/unused_page_output"
  while test ! -f "$CHAIN_ROOT/prewarm_$precision/exit_code.txt"; do
    tail -n 5 "$CHAIN_ROOT/prewarm_$precision/run.log"
    sleep 15
  done
  test "$(cat "$CHAIN_ROOT/prewarm_$precision/exit_code.txt")" = 0 || exit 1
}
prewarm_precision 1
export STAGE_DEADLINE_S=1800
```

```bash
export RUN_ROOT="$CHAIN_ROOT/smoke2"
launch_stage "$RUN_ROOT" "${COMMON[@]}" --limit 2 --output-dir "$RUN_ROOT/output" \
  --input-images "$IMAGES_DIR/page-573c437e-c309-4483-a038-ef2f440b104a.png" \
                 "$IMAGES_DIR/page-9bcba6da-bdb0-4403-97cb-8874898ac8ab.png"
```

These smoke pages contain known affected table/text crops, so this is stronger
than an arbitrary first-two-pages test whose crops may all be below the cap.
Monitor the smoke to child-written exit 0, then run the gate below with
`EXPECTED_PAGES=2`. Inspect the two Markdown outputs for sanity, including
tables/formulas if present; this is not a two-page quality score.

```bash
export EXPECTED_PAGES=2  # Change to 1651 when gating the full run.
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['RUN_ROOT']); n = int(os.environ['EXPECTED_PAGES'])
assert (r/'exit_code.txt').read_text().strip() == '0'
o = r/'output'; s = json.loads((o/'run_summary_shard_00.json').read_text())
assert (s['completed'],s['failed'],s['skipped']) == (n,0,0)
assert s['layout_backend'] == 'pp-doclayout-v3'
assert s['streaming']['layout_source'] == 'PP-DocLayoutV3_live'
assert s['streaming']['layout_calls'] == n
assert s['streaming']['layout_model_dtype'] == 'torch.float32'
assert not s['layout_graph_capture'] and not s['image_analysis']
assert not s['saved_layout_manifest'] and not s['crop_replay_manifest']
assert s['processor_max_pixels'] == 602112 and s['processor_min_pixels'] == 25088
assert s['warmup']['executed_pages'] == 0
assert s['local_decode_increfa_length_mode'] == 'pse_sentinel_310p'
assert s['local_compiled_vision']['layer_norm_impl'] == 'manual_fp32'
assert s['local_compiled_vision']['projection_impl'] == 'linear'
for folder, suffix in [('predictions','*.md'),('content_lists','*.json'),('layout_regions','*.json')]:
    assert len(list((o/folder).glob(suffix))) == n
cfg = json.loads((Path(os.environ['MODEL_DIR'])/'config.json').read_text())
ids = set(); counts = 0
for line in (o/'generation_trace.jsonl').open():
    row = json.loads(line); assert row['phase'] == 'recognition'
    assert row['request_id'] not in ids; ids.add(row['request_id']); counts += 1
    assert row['prompt_token_ids'].count(cfg['image_token_id'])*4 <= 3072
assert counts == s['generation_trace']['requests']
assert counts == s['streaming']['requests_completed_by_phase']['recognition']
print('LIVE_310P_GATE: PASS', 'pages',n,'requests',counts,'wall_s',s['pipeline_wall_s'])
PY
```

After a passing smoke, automatically launch the full run with the same device,
environment and caches. Reprocessing those two pages in the full run is intended.

```bash
export RUN_ROOT="$CHAIN_ROOT/full1651" STAGE_DEADLINE_S=7200
launch_stage "$RUN_ROOT" "${COMMON[@]}" --dataset-json "$DATASET_JSON" \
  --offset 0 --limit 1651 --output-dir "$RUN_ROOT/output"
```

Send the full log path to Luka immediately. Monitor until exit 0; run the same
gate with `EXPECTED_PAGES=1651`. Do not reuse predictions from another run.

## Monitoring and evaluation

Use detached jobs and keep monitoring live until both inference and evaluation
actually finish. Configure the tool's execution timeout above 120 minutes
(prefer 14,400,000 ms / four hours, using the tool's units). Poll/reattach in
30–60-second intervals, yielding progress to Luka. If the tool has a shorter
limit, keep reattaching to the same job; do not restart it. A released NPU,
quiet log, final progress line or tool timeout is not a successful exit.
Inspect durable progress, phase start/finish pairs and the child's exit file.
Do not call a pause "slow compilation" without phase/process evidence.

After each full inference gate passes, release the coordinator's cache lock
(`exec 9>&-`) and launch evaluation. Keep `RUN_ROOT` pointing at `full1651`:

```bash
export LIMIT=1651 STAGE_DEADLINE_S=14400
launch_stage "$CHAIN_ROOT/eval_launcher" bash \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_serving_accuracy.sh"
```

This uses your exported `DATASET_JSON`, `OMNIDOCBENCH_EVAL_PYTHON`, evaluator
root and TeX/ImageMagick tools. It structurally validates all outputs and runs
the frozen evaluator. Monitor both `eval_launcher/run.log` and
`full1651/evaluation/run.log` through page matching, CDM, TEDS and exit. Require
both `eval_launcher/exit_code.txt` and `full1651/evaluation/exit_code.txt` = 0.

## Final reply to Luka — directly in plain text, with tables for both modes

### Run the approximate-precision lane before the final reply

Only after original full inference and evaluation pass, reacquire the same
cache-owner lock. Retain original outputs. The prewarm audits above already
establish all seven approximate vision caches; do not clear them.

```bash
exec 9>"$WORK_SERVER_REPO/.runtime_cache/11_mineru_2_5_pro_inference/serving_validation.lock"
flock -n 9 || { echo 'Existing MinerU cache owner is busy'; exit 2; }
prewarm_precision 4
export STAGE_DEADLINE_S=1800
export RUN_ROOT="$CHAIN_ROOT/approx_smoke2"
launch_stage "$RUN_ROOT" "$PYTHON_BIN" \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_page_pipeline_vision_precision.py" \
  --vision-inner-precise 4 --precision-audit "$RUN_ROOT/precision.jsonl" \
  "${COMMON[@]:2}" --limit 2 --output-dir "$RUN_ROOT/output" \
  --input-images "$IMAGES_DIR/page-573c437e-c309-4483-a038-ef2f440b104a.png" \
                 "$IMAGES_DIR/page-9bcba6da-bdb0-4403-97cb-8874898ac8ab.png"
```

Wait for durable exit 0; apply the same smoke gate with EXPECTED_PAGES=2.
Compare crop hashes and prompt token IDs with original smoke, and report output
token/Markdown differences. Finite output drift is expected to be possible and
is reported separately from throughput; no token-equality gate is imposed on
approximate precision. Stop on nonfinite output or device errors.

```bash
export STAGE_DEADLINE_S=7200
export RUN_ROOT="$CHAIN_ROOT/approx_full1651"
launch_stage "$RUN_ROOT" "$PYTHON_BIN" \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_page_pipeline_vision_precision.py" \
  --vision-inner-precise 4 --precision-audit "$RUN_ROOT/precision.jsonl" \
  "${COMMON[@]:2}" --dataset-json "$DATASET_JSON" --offset 0 --limit 1651 \
  --output-dir "$RUN_ROOT/output"
```

Wait for exit 0; apply the same full gate with EXPECTED_PAGES=1651. Compare
layout_regions' crop_pixel_sha256 values, model/data hashes and recognition
prompt IDs against original full output before claiming matched real inputs.
Different scheduler packing may still change group counts; disclose it.
Require all mode4 vision cache paths to lie under the precision-specific
subdirectory, no eager-overflow crops, and the same text/decode caches as
original. The wrapper's sidecar audit must reach completed with
supported_310p=true and must show stock_converter_restored after each bucket's
first call. Keep the prewarm audit to establish cached graphs' GE precision.

```bash
exec 9>&-
export LIMIT=1651 STAGE_DEADLINE_S=14400
launch_stage "$CHAIN_ROOT/approx_eval_launcher" bash \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_serving_accuracy.sh"
```

Monitor through evaluator exit as for original. Do not obscure a measured speed
result because quality changed; report both. Do not treat execution success as
proof of equal OCR quality or a production benefit. No 310P result was produced
by the authoring session that prepared this update.

After both full runs complete, generate and paste the paired E2E and actual
length-distribution tables (the reporter requires complete live 1651-page runs):

```bash
"$PYTHON_BIN" "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/report_e2e_vision_precision.py" \
  --original "$CHAIN_ROOT/full1651" --approximate "$CHAIN_ROOT/approx_full1651" \
  --approximate-prewarm-audit "$CHAIN_ROOT/prewarm_4/precision.jsonl" \
  --chip 310P --output "$CHAIN_ROOT/precision_pair.json"
```

This compares actual crop geometry/pixel hashes, post-helper image hashes,
prompt IDs and model hashes, and reports changes in output tokens independently.
It shows every bin's group length distribution match flag beside its speed gain.
It does not suppress measured throughput when a flag is false; disclose why a
nonmatched distribution weakens the length-specific comparison. The reported
overall vision rate uses all actual input tokens divided by all vision event
time; the E2E page rate includes the actual full pipeline. Report both.
Report total generated tokens as well: approximate vision may change output
lengths and therefore decode work. An E2E gain alone cannot establish that
vision kernels got faster; the separately measured vision rates answer that.

Read `output/run_summary_shard_00.json` and
`evaluation/work/result/predictions_quick_match_metric_result.json`.
The score definitions are exactly:

```text
Text accuracy = 100 * (1 - result["text_block"]["page"]["Edit_dist"]["ALL"])
Page Table TEDS = 100 * result["table"]["page"]["TEDS"]["ALL"]
Page Formula CDM = 100 * result["display_formula"]["page"]["CDM"]["ALL"]
Overall = arithmetic mean of those three percentage scores
```

These use the evaluator's **page aggregation**, not a globally pooled mean over
all individual tables/formulas. Also report page table-structure TEDS and page
reading-order edit distance (lower is better), without changing the main overall.

Report:

- commit, host, exact 310P, package/CANN versions, resolved model/data/cache paths
  and preflight hash results;
- smoke/full completed, failed, skipped, live layout calls and recognition count;
- **1651 / pipeline_wall_s** for actual full E2E pg/s, wall time, setup separately;
  optionally completion-10-to-last pg/s = 1641 / (last completion elapsed minus
  tenth completion elapsed), clearly labeled. No separate warmup pages are used;
  first-use cache loads during page processing remain inside pipeline wall;
- the four score anchors, their deltas against the 910B scores below, and the
  auxiliary scores; evaluation sample counts, timeouts/errors/fallbacks;
- main vision/text-prefill/decode device times and layout host time separately;
- real and physical vision tokens, padding count and percentage, crop count,
  all per-bucket vision latency / useful tok/s / share of vision time from
  `vision_timing`; sum(tokens)/sum(seconds), never mean(per-call tok/s);
- original-versus-approximate E2E pg/s gain and per-bucket vision useful tok/s
  gain. Label packed_768 separately. Show input/packing distribution changes
  first, and do not claim a matched length comparison if the groups differ;
- precision audit paths and confirmation that only vision requested mode4;
  length-cap counts by crop type, mentioning repetition if observed;
- full inference/evaluation log and artifact roots.

Use the new **910B full live lower-cap** reference scores for the delta table:
text **96.234624**, Page TEDS **93.520375**, Page CDM **97.054051**, overall
**95.603017**. See `references/d4_cap3072_reproduction_910b_20261008/`.
The older selective replay/reconstruction reference remains in
`references/crop_cap3072_live_910b/`: text 96.234624, Page TEDS 93.520705,
Page CDM 97.053718, overall 95.603016. Its receipts have not been rewritten.

The measured 910B **higher-cap** live run was **0.932911 pg/s**. The former
lower-cap **0.993678 pg/s is only an estimate**. The new exact-d4 full live
lower-cap 910B reproduction measured **0.979472 pg/s**. These are 910B reference
points, not 310P predictions or a controlled interleaved cap comparison. Your
run will establish the real 310P E2E number.
No cross-chip token-parity gate or invented accuracy cutoff is imposed. If
scores differ substantially, report the facts; do not silently change settings.

On any failure: stop the chain, preserve outputs and report phase, exact command,
exit status, first causal error, last unmatched phase marker and log paths to
Luka. No separate report file, proposed patch, package changes or automatic
workaround. Do not stop merely after launching a successful background job.
