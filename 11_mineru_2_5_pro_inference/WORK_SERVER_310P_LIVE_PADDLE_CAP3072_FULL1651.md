# 310P: live PP-DocLayoutV3 + custom MinerU, cap 3072, full 1651 + evaluation

This is a new execution handoff for Luka's pull-only 310P work agent. It replaces
older native-MinerU-layout or selective-replay instructions for this task.
Luka authorizes the two-page smoke, then (if it passes) one full run and one
evaluation. Do not stop to ask for another approval after a passing smoke.

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

Authoring validation on 910B at inference commit `8528414f`: the exact two
affected smoke pages below passed with 2 live layout calls, 20 recognition
requests, zero failed pages and 13.215 s pipeline wall (23.972 s setup
separately). The targeted text/table crops used 2,904 / 2,976 raw vision tokens
and both completed at EOS. Evidence is in
`references/live_cap3072_handoff_smoke_910b/affected/`. This validates the live
lower-cap integration on 910B, **not** 310P compatibility or full-run speed.

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
- Inspect `git status --short`; pull with `git pull --ff-only origin main`.
  If changes conflict, preserve them and report the issue. Require
  `git merge-base --is-ancestor 8528414f HEAD` and this new brief to exist.
- Resolve `WORK_SERVER_REPO` using `git rev-parse --show-toplevel`.
- Reuse the successful custom MinerU 310P Python/CANN environment and its three
  existing cache paths. Resolve them from a successful run's command/summary:
  `local_torchair_cache_dir`, `local_vision_torchair_cache_dir`, and
  `local_text_torchair_cache_dir`. Resolve repo-relative paths against your
  checkout. Do not clear caches, use a fresh cache root, or run concurrent cache
  owners. A new output directory is required; that is not a new graph cache.
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

## Commands: smoke, then full run

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
  nohup setsid bash -c '
    root="$1"; shift
    set +e
    /usr/bin/time -f %e -o "$root/process_wall_s.txt" "$@" > "$root/run.log" 2>&1
    status=$?
    printf "%s\n" "$status" > "$root/exit_code.txt"
    exit "$status"
  ' bash "$stage_root" "$@" </dev/null > "$stage_root/launcher.log" 2>&1 &
  printf '%s\n' "$!" > "$stage_root/pid.txt"
  printf 'LOG=%s/run.log\n' "$stage_root"
}

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
export RUN_ROOT="$CHAIN_ROOT/full1651"
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

After the full inference gate passes, release the coordinator's cache lock
(`exec 9>&-`) and launch evaluation. Keep `RUN_ROOT` pointing at `full1651`:

```bash
export LIMIT=1651
launch_stage "$CHAIN_ROOT/eval_launcher" bash \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_serving_accuracy.sh"
```

This uses your exported `DATASET_JSON`, `OMNIDOCBENCH_EVAL_PYTHON`, evaluator
root and TeX/ImageMagick tools. It structurally validates all outputs and runs
the frozen evaluator. Monitor both `eval_launcher/run.log` and
`full1651/evaluation/run.log` through page matching, CDM, TEDS and exit. Require
both `eval_launcher/exit_code.txt` and `full1651/evaluation/exit_code.txt` = 0.

## Final reply to Luka — directly in plain text

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
  length-cap counts by crop type, mentioning repetition if observed;
- full inference/evaluation log and artifact roots.

Reference 910B lower-cap scores: text **96.234624**, Page TEDS **93.520705**,
Page CDM **97.053718**, overall **95.603016**. They come from hash-verified
selective recognition replay with full-page reconstruction/evaluation, not a
full lower-cap E2E timing run. See `references/crop_cap3072_live_910b/`.

The measured 910B **higher-cap** live run was **0.932911 pg/s**. The lower-cap
**0.993678 pg/s is only an estimate**, so do not present it as a measured
cross-chip speed baseline. Your run will establish the real 310P E2E number.
No cross-chip token-parity gate or invented accuracy cutoff is imposed. If
scores differ substantially, report the facts; do not silently change settings.

On any failure: stop the chain, preserve outputs and report phase, exact command,
exit status, first causal error, last unmatched phase marker and log paths to
Luka. No separate report file, proposed patch, package changes or automatic
workaround. Do not stop merely after launching a successful background job.
