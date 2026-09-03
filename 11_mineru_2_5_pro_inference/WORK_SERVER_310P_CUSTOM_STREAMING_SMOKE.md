# Work-server 310P custom MinerU two-page smoke

This is the execution brief for the AI agent on Luka's Atlas 310P work server.
Read the repository `CLAUDE.md` and `AGENTS.md` first. Then execute this brief
from top to bottom in Bash.

## Goal

Run the first two OmniDocBench v1.6 pages through the repository-owned custom
MinerU2.5-Pro pipeline on one Atlas 310P. Use the same model, processor,
frontend, post-processing, B32 decode, KV4096, packed text prefill, compiled
vision prefill, and bounded streaming scheduler as the accepted 1,651-page
910B2 run.

This custom pipeline does not import or invoke vLLM or vLLM-Ascend. Do not use
Experiment 17's stock engine runner. The only Experiment 17 component reused
here is the generic artifact verifier for the model, dataset, images, and
OmniDocBench repository.

This is a cold compatibility smoke. It is not a throughput benchmark or an
accuracy run. Compile only the graph shapes reached by these two real pages.
Do not continue to 128 pages or the full corpus.

## Rules

- The work-server checkout is pull-only. Do not edit tracked files, create a
  branch, commit, push, reset, stash, or discard another person's changes.
- Do not edit `/vllm-workspace` or any installed framework source.
- Do not install or replace PyTorch, torch-npu, TorchAir, CANN, the NPU driver,
  firmware, or system packages.
- vLLM and vLLM-Ascend versions do not matter for this run. Do not require them,
  import them, change them, or compare them with the 910B environment.
- You may create one experiment venv under `$HOME/.venvs`. Install only
  `mineru-vl-utils==1.0.5` and `httpx-retries==0.6.0` with `--no-deps` from the
  committed requirements file.
- Do not download a model, dataset, image corpus, or evaluator repository. Find
  and verify the copies already on the server.
- Use one free physical Atlas 310P device. Never terminate another user's
  process. Do not fall back to CPU or CUDA.
- Use fresh, 310P-only TorchAir caches. Do not copy, rename, delete, repair, or
  reuse a 910B cache. Do not run two processes against the same cache.
- Run the two-page command once. If it fails, do not retry it, change flags, or
  delete the partial cache.
- Normal command, log, cache, and output artifacts are allowed. Do not create a
  separate Markdown report or agent report.
- If a check fails, stop and reply to Luka directly in plain text. Include the
  phase, exact command, exit code, first causal error, and paths checked. Do not
  propose a fix or source change.

## Reference identities

Use these exact inputs:

```text
MinerU model: opendatalab/MinerU2.5-Pro-2605-1.2B
Model files: 15
Model bytes: 2328027938
Path-independent model manifest SHA-256:
5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea
model.safetensors SHA-256:
abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0

OmniDocBench revision: v1.6_full
Dataset pages: 1651
OmniDocBench.json SHA-256:
a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496
Path-independent 1651-image manifest SHA-256:
34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe

OmniDocBench Git remote: opendatalab/OmniDocBench
OmniDocBench Git commit:
2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

The model manifest includes `.msc`, `.mv`, and `configuration.json`. They are
part of the checkpoint identity checked by the verifier. They are not the
TorchAir graph cache used below.

The accepted 910B2 source run used commit
`ae4c947c15d43d9858dfe3608e25bdbb6fa43565`. Its complete 1,651-page
predictions and token trace are committed at:

```text
11_mineru_2_5_pro_inference/references/serving_streaming_1651_ae4c947c/
```

That run scored 95.1131 and reached 0.81331 hot pages/s on one 910B2. Those
numbers are context only. They are not expected 310P results and are not pass
thresholds for this cold two-page smoke.

## Phase 1: update the pull-only checkout

Start inside the existing repository clone:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet
git diff --cached --quiet
git pull --ff-only origin main
git rev-parse HEAD
git status --short --branch
git diff --quiet
git diff --cached --quiet
test -f \
  11_mineru_2_5_pro_inference/WORK_SERVER_310P_CUSTOM_STREAMING_SMOKE.md
test -f \
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py
test -f \
  11_mineru_2_5_pro_inference/references/serving_streaming_1651_ae4c947c/streaming.tar.gz
```

If tracked changes prevent the pull, stop. Do not alter the checkout.

## Phase 2: activate the server's NPU environment

Inspect the current shell and normal CANN locations:

```bash
command -v npu-smi || true
command -v python3 || true
command -v npu-setup || true
for candidate in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
do
  test -f "$candidate" && printf 'CANN_ENV_CANDIDATE=%s\n' "$candidate"
done
```

If this server has its own documented `npu-setup`, inspect and source it.
Otherwise source the installed CANN environment file found above. Do not source
a setup file from another user's project directory.

Run `npu-smi info`. Confirm that the product is Atlas 310P and inspect the
process table. Select one free physical device from the available 310P devices:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

If no device is free, stop and send the plain-text issue reply.

## Phase 3: find and verify the model and corpus

Search only normal storage roots. Ignore errors for roots that do not exist:

```bash
find /workspace /data /data1 /mnt /home \
  -maxdepth 8 -type f -name model.safetensors -size +2000M \
  -print 2>/dev/null | sort -u

find /workspace /data /data1 /mnt /home \
  -maxdepth 8 -type f -name OmniDocBench.json \
  -print 2>/dev/null | sort -u

find /workspace /data /data1 /mnt /home \
  -maxdepth 8 -type f \
  -name page-d1561665-5359-42fe-920c-d6e3bff81953.png \
  -print 2>/dev/null | sort -u

find /workspace /data /data1 /mnt /home \
  -maxdepth 7 -type d -name OmniDocBench \
  -print 2>/dev/null | sort -u
```

Set these variables to the matching existing copies:

```bash
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the/1651/images
export OMNIDOCBENCH_REPO=/absolute/path/to/opendatalab/OmniDocBench
```

Check the visible checkpoint files before hashing:

```bash
test -f "$MODEL_DIR/.msc"
test -f "$MODEL_DIR/.mv"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/configuration.json"
test -f "$MODEL_DIR/model.safetensors"
test -f "$MODEL_DIR/preprocessor_config.json"
test -f "$MODEL_DIR/tokenizer.json"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$OMNIDOCBENCH_REPO/.git"
```

Run the committed verifier in the foreground. It hashes the 2.3 GB checkpoint,
all 1,651 selected images, the dataset JSON, and the evaluator checkout. It
prints progress and writes no report file:

```bash
python3 \
  "$WORK_SERVER_REPO/17_mineru_vllm_ascend_baseline/verify_310p_artifacts.py" \
  --model-dir "$MODEL_DIR" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO"
```

Continue only if the final line contains:

```text
EXPERIMENT17_310P_ARTIFACTS {"dataset": ... "status": "PASS"}
```

The `EXPERIMENT17` prefix is only the verifier's historical name. It does not
run or import vLLM. Any `MISMATCH` or `ERROR` ends this task.

## Phase 4: create the custom-pipeline environment

Find a base Python from the server's existing NPU installation. It must import
PyTorch, torch-npu, TorchAir, and Transformers before any venv is created.
Inspect only normal interpreter and venv roots:

```bash
{
  test -x /usr/local/python3.12.13/bin/python3 && \
    printf '%s\n' /usr/local/python3.12.13/bin/python3
  test -x /usr/local/python3.12.13/bin/python && \
    printf '%s\n' /usr/local/python3.12.13/bin/python
  command -v python3 || true
  find "$HOME" /workspace/venvs /opt \
    -maxdepth 5 -type f \( -name python -o -name python3 \) \
    -print 2>/dev/null
} | awk '!seen[$0]++'
```

Test plausible candidates directly:

```bash
/absolute/candidate/python -c \
  'import sys,torch,torch_npu,torchair,transformers; from torchair.inference import cache_compile; print(sys.executable); print(torch.__version__); print(torch_npu.__version__); print(torchair.__file__); print(transformers.__version__); print(torch.npu.is_available())'
```

Choose `BASE_PYTHON` only when that command succeeds and NPU availability is
true. Record the actual versions. Do not force them to match the 910B versions
and do not change the core stack.

Create one experiment-specific venv over that working base stack:

```bash
export BASE_PYTHON=/absolute/verified/base/python
export CUSTOM_VENV="$HOME/.venvs/mineru_custom_exp11_310p_py312"

if test -e "$CUSTOM_VENV"; then
  test -x "$CUSTOM_VENV/bin/python"
else
  "$BASE_PYTHON" -m venv --system-site-packages "$CUSTOM_VENV"
fi

export PYTHON_BIN="$CUSTOM_VENV/bin/python"
"$PYTHON_BIN" -m pip install --no-deps \
  -r "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/requirements_official_vllm.txt"
```

The requirements filename is historical. It installs MinerU's Python frontend
utilities only. It does not install or run vLLM.

Verify every import used at the pipeline boundary:

```bash
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import json
import sys

import PIL
import torch
import torch_npu
import torchair
import transformers
from mineru_vl_utils import MinerUClient
from torchair.inference import cache_compile

versions = {}
for name in (
    "torch",
    "torch-npu",
    "torchair",
    "transformers",
    "tokenizers",
    "safetensors",
    "Pillow",
    "mineru-vl-utils",
    "httpx-retries",
):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print("PYTHON", sys.executable)
print("TORCH_SOURCE", torch.__file__)
print("TORCH_NPU_SOURCE", torch_npu.__file__)
print("TORCHAIR_SOURCE", torchair.__file__)
print("TRANSFORMERS_SOURCE", transformers.__file__)
print("PACKAGE_VERSIONS", json.dumps(versions, sort_keys=True))
assert versions["mineru-vl-utils"] == "1.0.5"
assert versions["httpx-retries"] == "0.6.0"
assert torch.npu.is_available()
PY
```

Run one real NPU operation before model loading:

```bash
"$PYTHON_BIN" -c \
  'import torch,torch_npu; torch.npu.set_device("npu:0"); torch.npu.set_compile_mode(jit_compile=False); x=torch.arange(8,dtype=torch.float16,device="npu:0"); print("NPU_PROBE",torch.npu.get_device_name(0),((x+1).cpu()).tolist())'
```

The device name must identify a 310P. Stop if environment creation, imports, or
the NPU operation fails.

## Phase 5: prepare isolated 310P caches and evidence

Use a persistent cache outside the Git checkout. Its identity includes the
repository commit and physical chip family:

```bash
cd "$WORK_SERVER_REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
RUN_TAG="$(date +%Y%m%dT%H%M%S)"
export CACHE_ROOT="$HOME/.cache/paddle_ocr_vl_npu/exp11_mineru_custom_310p/${COMMIT_SHORT}_${RUN_TAG}"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_custom_streaming_n2_${COMMIT_SHORT}_${RUN_TAG}"

test ! -e "$RUN_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
exec 9>"$CACHE_ROOT/smoke.lock"
flock -n 9 || {
  echo 'Another process owns the Experiment 11 310P caches.' >&2
  exit 2
}

find "$CACHE_ROOT" -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort >"$RUN_ROOT/om_before.txt"
git rev-parse HEAD >"$RUN_ROOT/commit.txt"
{
  hostname
  uname -a
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  npu-smi info
  "$PYTHON_BIN" -m pip show \
    torch torch-npu torchair transformers mineru-vl-utils httpx-retries
} >"$RUN_ROOT/environment.txt" 2>&1

printf 'RUN_ROOT=%s\nCACHE_ROOT=%s\n' "$RUN_ROOT" "$CACHE_ROOT"
```

Give Luka the absolute `RUN_ROOT/run.log` path before starting inference.

## Phase 6: run exactly two pages

Keep the same shell, selected device, environment, cache lock, and exported
paths. Use the accepted production settings. The only smoke-specific changes
are `limit=2`, no excluded warmup, and a two-page streaming window.

```bash
export PYTHONUNBUFFERED=1

COMMAND=(
  "$PYTHON_BIN"
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py"
  --backend local-continuous-client
  --model "$MODEL_DIR"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --output-dir "$RUN_ROOT/output"
  --offset 0
  --limit 2
  --warmup-pages 0
  --no-resume
  --fail-fast
  --batch-size 32
  --page-batch-size 2
  --global-request-stream
  --layout-image-size 1036 1036
  --processor-min-pixels 25088
  --local-dtype float16
  --local-compiled-cache-length 4096
  --local-decode-attention increfa
  --local-decode-weight-format decode_nz
  --local-decode-rotary-impl npu_apply
  --local-prepare-prefetch-depth 64
  --local-prefill-metrics
  --local-text-backend torchair-packed
  --local-text-buckets 128,256,512,1024
  --local-text-max-members 32
  --local-text-torchair-cache-dir "$CACHE_ROOT/text_prefill_packed_fp16"
  --local-vision-attention prompt_flash_attention
  --local-vision-backend torchair
  --local-vision-buckets 384,512,768,1024,1536,2048,3072,4224,5632
  --local-vision-pack-target 768
  --local-vision-lookahead 32
  --local-vision-torchair-cache-dir "$CACHE_ROOT/vision_prefill_b1_fp16"
  --local-torchair-cache-dir "$CACHE_ROOT/production_increfa_real_nz_compile"
  --token-trace
  --hash-model-files
  --streaming-pages
  --streaming-page-window 2
)

printf '%q ' "${COMMAND[@]}" >"$RUN_ROOT/command.sh"
printf '\n' >>"$RUN_ROOT/command.sh"
chmod +x "$RUN_ROOT/command.sh"

set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$RUN_STATUS" >"$RUN_ROOT/exit_code.txt"
test "$RUN_STATUS" = 0
```

This first 310P invocation may compile several graph shapes. Record the
behavior. Do not call a cache load a compile, and do not call a compile a cache
load. Do not launch a second process to measure a hot-cache result.

While it runs, inspect the active process, selected NPU, latest log output, and
cache file count every 15 to 30 seconds. A long first graph build is not by
itself a failure. If the process exits nonzero, preserve the first causal error
and stop. Do not retry.

## Phase 7: validate output and compare the stored token reference

Run the structural gate:

```bash
export RUN_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
summary_path = root / "output/run_summary_shard_00.json"
trace_path = root / "output/generation_trace.jsonl"
progress_path = root / "output/progress_shard_00.jsonl"
assert (root / "exit_code.txt").read_text().strip() == "0"
assert summary_path.is_file()
assert trace_path.is_file() and trace_path.stat().st_size > 0
assert progress_path.is_file() and progress_path.stat().st_size > 0

summary = json.loads(summary_path.read_text())
assert summary["backend"] == "official_mineru_local-continuous-client"
assert summary["completed"] == 2
assert summary["failed"] == 0
assert summary["skipped"] == 0
assert summary["batch_size"] == 32
assert summary["local_compiled_cache_length"] == 4096
assert summary["local_decode_attention"] == "increfa"
assert summary["local_decode_weight_format"] == "decode_nz"
assert summary["local_decode_rotary_impl"] == "npu_apply"
assert summary["local_text_backend"] == "torchair-packed"
assert summary["local_vision_attention"] == "prompt_flash_attention"
assert summary["local_vision_backend"] == "torchair"
assert summary["global_request_stream"] is True
assert summary["streaming_pages"] is True
assert summary["streaming_page_window"] == 2
assert summary["generation_trace"]["requests"] > 0

predictions = sorted((root / "output/predictions").glob("*.md"))
content_lists = sorted((root / "output/content_lists").glob("*.json"))
progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
assert len(predictions) == 2
assert len(content_lists) == 2
assert len(progress) == 2
assert len(trace) == summary["generation_trace"]["requests"]
assert all(path.stat().st_size > 0 for path in predictions)
assert all(isinstance(json.loads(path.read_text()), list) for path in content_lists)
assert all(row.get("generated_token_ids") for row in trace)
assert len({row["request_id"] for row in trace}) == len(trace)
print(
    "CUSTOM_MINERU_TWO_PAGE_OUTPUT: PASS",
    f"pages={len(predictions)}",
    f"requests={len(trace)}",
)
PY
```

Verify and extract the committed 910B2 reference under this run root:

```bash
REFERENCE_BUNDLE="$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/references/serving_streaming_1651_ae4c947c"
REFERENCE_ROOT="$RUN_ROOT/reference_910b2"
mkdir -p "$REFERENCE_ROOT"
(
  cd "$REFERENCE_BUNDLE"
  sha256sum -c SHA256SUMS
)
tar -xzf "$REFERENCE_BUNDLE/streaming.tar.gz" -C "$REFERENCE_ROOT"

"$PYTHON_BIN" \
  "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/compare_generation_traces.py" \
  "$REFERENCE_ROOT/output" \
  "$RUN_ROOT/output" \
  --first-pages 2 \
  --allow-table-image-placeholders \
  --output "$RUN_ROOT/comparison_910b2_first2.json"
```

The comparison must have no missing or extra pages, no missing or extra
requests under unchanged layouts, no unexpected input changes, no new length
stops, and valid trace accounting. Exact token IDs are informative across
chips, not a required pass condition. Report layout and recognition token-exact
counts, the first differing token position for each changed request, changed
page names, and whether final Markdown stayed byte-identical.

Record the final 310P cache inventory and release state:

```bash
find "$CACHE_ROOT" -type f -name '*.om' -printf '%p %s %T@\n' \
  | sort >"$RUN_ROOT/om_after.txt"
comm -13 "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
  >"$RUN_ROOT/om_created.txt" || true
wc -l "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" \
  "$RUN_ROOT/om_created.txt"
npu-smi info
```

Confirm that the inference process exited and released the selected device.

## Direct reply to Luka

Do not create another report file. Reply directly in plain text.

For success, include these fields:

```text
310P MINERU CUSTOM TWO-PAGE SMOKE: PASS
repo_commit=
hostname=
physical_npu=
npu_name=
cann_version=
python=
package_versions=
model_dir=
model_manifest_sha256=
model_safetensors_sha256=
dataset_json=
dataset_json_sha256=
dataset_pages=1651
images_dir=
image_manifest_sha256=
omnidocbench_repo=
omnidocbench_commit=
run_root=
cache_root=
completed=2
failed=0
request_count=
setup_s=
pipeline_wall_s=
cold_smoke_pages_per_s=
prefill_s=
decode_s=
decode_effective_tok_s=
decode_active_slot_fraction=
compiled_first_call_s=
om_before=
om_after=
om_created=
reference_pages_byte_identical=
layout_token_exact=
recognition_token_exact=
differing_requests=
changed_pages=
npu_released=yes
```

Label all timing and rate values as cold smoke values. Do not compare them with
the 910B2 hot-cache rate as if they were the same measurement.

If any phase fails, stop and reply with only:

```text
310P MINERU CUSTOM TWO-PAGE SMOKE: ISSUE
phase=
command=
exit_code=
first_causal_error=
run_root=
cache_root=
paths_checked=
```

Do not add a proposed change, workaround, patch, or Markdown report. Stop after
the success or issue reply.
