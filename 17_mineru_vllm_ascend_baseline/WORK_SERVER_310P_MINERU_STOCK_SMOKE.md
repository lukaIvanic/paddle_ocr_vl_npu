# Work-server 310P stock MinerU smoke

This is the execution brief for the AI agent on Luka's Atlas 310P work server.
Read the repository `CLAUDE.md` and `AGENTS.md` first. Then execute this brief.

## Goal

Set up an experiment-owned Python environment over the server's stock
`/vllm-workspace`, verify every required artifact against the 910B reference,
and run one real OmniDocBench page through stock vLLM-Ascend and MinerU.

Use `enable_static_kernel=False`. This is a compatibility smoke, not a
performance benchmark and not a static-kernel ablation. Do not continue to 128
pages or the full corpus in this task.

## Rules

- The repository is pull-only. Do not edit tracked files, create a branch,
  commit, push, or reset the checkout.
- Do not edit either stock repository under `/vllm-workspace`.
- Do not replace or upgrade vLLM, vLLM-Ascend, PyTorch, torch-npu, CANN, or
  system packages.
- You may create one experiment venv under `$HOME/.venvs`. Install only
  `mineru-vl-utils==1.0.5` and `httpx-retries==0.6.0` with `--no-deps` through
  the committed requirements file.
- Use one free physical Atlas 310P device. Do not terminate another user's
  process. Do not fall back to CPU or CUDA.
- Do not download a model, dataset, or evaluator repository. Find and verify
  the copies already present on the server.
- The committed runner may write its normal run evidence below `tmp/`. Do not
  create a separate report file, Markdown report, proposed patch, or proposed
  change.
- If any required check fails, stop. Reply to Luka directly in plain text with
  the failed phase, exact command, exit code, first causal error, and paths
  checked. Do not suggest a fix.

## Reference identities

The smoke must use these exact artifacts:

```text
MinerU model directory files: 15
MinerU model directory bytes: 2328027938
MinerU path-independent directory manifest SHA-256:
5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea
model.safetensors SHA-256:
abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0

OmniDocBench dataset revision: v1.6_full
OmniDocBench pages: 1651
OmniDocBench.json SHA-256:
a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496
Path-independent 1651-image manifest SHA-256:
34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe

OmniDocBench Git remote: opendatalab/OmniDocBench
OmniDocBench Git commit:
2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

The 310P work environment intentionally has its own vLLM and vLLM-Ascend
versions. Record their versions, source paths, Git commits, and remotes. Do not
compare them to the 910B versions and do not change them.

The committed verifier recalculates the model, dataset, image, and
OmniDocBench repository identities. Do not replace that verification with
directory names or file sizes.

## Phase 1: update the pull-only checkout

Start inside the repository clone:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet
git diff --cached --quiet
git pull --ff-only origin main
git rev-parse HEAD
git diff --quiet
git diff --cached --quiet
test -f 17_mineru_vllm_ascend_baseline/WORK_SERVER_310P_MINERU_STOCK_SMOKE.md
```

If tracked local changes prevent the pull, stop and send the plain-text issue
reply. Do not stash, discard, or overwrite them.

## Phase 2: activate the server's NPU environment

Inspect the current shell and only these normal CANN locations:

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

If the server has its own documented `npu-setup`, inspect it and source it.
Otherwise source the installed CANN environment file found above. Do not source
a file from another user's project directory.

Run `npu-smi info`. Confirm that the product is Atlas 310P and inspect the
process table. Select one free physical device from the server's 310P devices,
then export it:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<free_physical_310p_id>
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

If no device is free, stop and send the plain-text issue reply.

## Phase 3: find the stock source and all artifacts

Search only normal storage roots. Missing roots may print errors and can be
ignored.

```bash
find /vllm-workspace /workspace /data /data1 /mnt /home \
  -maxdepth 6 -type d -name vllm-workspace -print 2>/dev/null | sort -u

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

Set these variables to the candidates that contain the expected objects:

```bash
export VLLM_WORKSPACE=/absolute/path/to/vllm-workspace
export MODEL_DIR=/absolute/path/to/MinerU2.5-Pro-2605-1.2B
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export IMAGES_DIR=/absolute/path/to/the/1651/images
export OMNIDOCBENCH_REPO=/absolute/path/to/opendatalab/OmniDocBench
```

The stock workspace must contain both Git checkouts:

```bash
test -d "$VLLM_WORKSPACE/vllm/.git"
test -d "$VLLM_WORKSPACE/vllm-ascend/.git"
git -C "$VLLM_WORKSPACE/vllm" rev-parse HEAD
git -C "$VLLM_WORKSPACE/vllm-ascend" rev-parse HEAD
git -C "$VLLM_WORKSPACE/vllm" diff --quiet
git -C "$VLLM_WORKSPACE/vllm" diff --cached --quiet
git -C "$VLLM_WORKSPACE/vllm-ascend" diff --quiet
git -C "$VLLM_WORKSPACE/vllm-ascend" diff --cached --quiet
git -C "$VLLM_WORKSPACE/vllm" remote -v
git -C "$VLLM_WORKSPACE/vllm-ascend" remote -v
git -C "$VLLM_WORKSPACE/vllm" status --short
git -C "$VLLM_WORKSPACE/vllm-ascend" status --short
```

Tracked source must be clean. The commits may differ from the 910B reference.
Untracked files do not change the source identity, but include them in the
direct success reply.

Run the complete artifact verification in the foreground. It prints progress
while hashing the 2.3 GB model and all 1,651 images. It writes no report file.

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

Any `MISMATCH` or `ERROR` is an issue. Stop and reply in plain text.

## Phase 4: create and verify the experiment environment

Find the stock base Python that imports the two pinned source checkouts. Check
the normal interpreter and venv roots only:

```bash
{
  test -x /usr/local/python3.12.13/bin/python3 && \
    printf '%s\n' /usr/local/python3.12.13/bin/python3
  command -v python3 || true
  find "$HOME" /workspace/venvs /opt \
    -maxdepth 5 -type f \( -name python -o -name python3 \) \
    -print 2>/dev/null
} | awk '!seen[$0]++'
```

For each plausible interpreter, run this directly in the terminal:

```bash
/absolute/candidate/python -c \
  'import importlib.metadata as m,sys,vllm,vllm_ascend,torch,torch_npu; print(sys.executable); print(vllm.__file__); print(vllm_ascend.__file__); print(torch.__file__); print(torch_npu.__file__); print({n:m.version(n) for n in ("vllm","vllm-ascend","torch","torch-npu","transformers")})'
```

Choose `BASE_PYTHON` only when the command succeeds, vLLM resolves below
`$VLLM_WORKSPACE/vllm`, vLLM-Ascend resolves below
`$VLLM_WORKSPACE/vllm-ascend`, and these non-vLLM versions match:

```text
torch=2.10.0+cpu
torch-npu=2.10.0
transformers=5.5.4
```

Print and retain the installed vLLM and vLLM-Ascend versions. They are
work-server identities, not mismatches.

Create one fresh experiment venv if it does not already exist:

```bash
export BASE_PYTHON=/absolute/verified/base/python
export EXP17_310P_ENV="$HOME/.venvs/mineru_vllm_ascend_exp17_310p_py312"

if test -e "$EXP17_310P_ENV"; then
  test -x "$EXP17_310P_ENV/bin/python"
else
  "$BASE_PYTHON" -m venv --system-site-packages "$EXP17_310P_ENV"
  "$EXP17_310P_ENV/bin/python" -m pip install --no-deps \
    -r "$WORK_SERVER_REPO/11_mineru_2_5_pro_inference/requirements_official_vllm.txt"
fi

export PYTHON_BIN="$EXP17_310P_ENV/bin/python"
"$PYTHON_BIN" \
  "$WORK_SERVER_REPO/17_mineru_vllm_ascend_baseline/verify_environment.py" \
  --allow-vllm-version-drift

"$PYTHON_BIN" -c \
  'import sys,vllm,vllm_ascend,torch,torch_npu,mineru_vl_utils; print("PYTHON",sys.executable); print("VLLM_SOURCE",vllm.__file__); print("VLLM_ASCEND_SOURCE",vllm_ascend.__file__); print("TORCH_SOURCE",torch.__file__); print("TORCH_NPU_SOURCE",torch_npu.__file__); print("MINERU_UTILS_SOURCE",mineru_vl_utils.__file__)'
```

The final verification must also report:

```text
mineru-vl-utils=1.0.5
httpx-retries=0.6.0
```

The final venv must still resolve vLLM below `$VLLM_WORKSPACE/vllm` and
vLLM-Ascend below `$VLLM_WORKSPACE/vllm-ascend`. Otherwise stop and send the
plain-text issue reply.

Run a real NPU operation before loading the model:

```bash
"$PYTHON_BIN" -c \
  'import torch,torch_npu; assert torch.npu.is_available(); torch.npu.set_device("npu:0"); x=torch.arange(8,dtype=torch.float16,device="npu:0"); print("NPU_PROBE",torch.npu.get_device_name(0),((x+1).cpu()).tolist())'
```

The device name must identify a 310P. If environment creation, verification,
module-path validation, or the NPU operation fails, stop and send the
plain-text issue reply.

## Phase 5: run one real page with static kernels off

Run exactly one first-page smoke. Keep the selected device and environment in
the same shell. Run `npu-smi info` once more immediately before launch. If
another process has taken the selected device, stop and send the plain-text
issue reply.

```bash
cd "$WORK_SERVER_REPO"
npu-smi info

PYTHON="$PYTHON_BIN" \
MODEL_DIR="$MODEL_DIR" \
DATASET_JSON="$DATASET_JSON" \
IMAGES_DIR="$IMAGES_DIR" \
LIMIT=1 \
OFFSET=0 \
HASH_MODEL_FILES=1 \
STATIC_KERNEL=off \
EXP17_NPU_SETUP_ALREADY_SOURCED=1 \
bash 17_mineru_vllm_ascend_baseline/run_npu_reproduction.sh
```

Do not rerun automatically if it fails. Send the plain-text issue reply with
the first causal error from the printed log.

After exit zero, identify the run created by this command and validate it:

```bash
RUN_DIR="$(ls -1dt \
  "$WORK_SERVER_REPO"/tmp/17_mineru_vllm_ascend_baseline/compiled_async_static_kernel_off_n1_* \
  | head -n 1)"
export RUN_DIR
printf 'RUN_DIR=%s\n' "$RUN_DIR"

test "$(cat "$RUN_DIR/exit_code.txt")" = 0
test -s "$RUN_DIR/output/run_summary.json"
test "$(find "$RUN_DIR/output/predictions" -type f -name '*.md' | wc -l)" = 1
test "$(find "$RUN_DIR/output/content_lists" -type f -name '*.json' | wc -l)" = 1
find "$RUN_DIR/output/predictions" -type f -name '*.md' -exec test -s {} \;
find "$RUN_DIR/output/content_lists" -type f -name '*.json' -exec test -s {} \;

"$PYTHON_BIN" -c \
  'import json,os,pathlib; p=pathlib.Path(os.environ["RUN_DIR"])/"output"/"run_summary.json"; s=json.loads(p.read_text()); assert s["static_kernel"]=="off"; assert s["preset"]["enable_static_kernel"] is False; assert s["completed"]==1; assert s["failed"]==0; print("SMOKE_SUMMARY",json.dumps(s,sort_keys=True))'

grep -q "'enable_static_kernel': False" "$RUN_DIR/run.log"
if grep -q 'Starting static kernel compilation' "$RUN_DIR/run.log"; then
  printf 'unexpected static-kernel compilation in %s\n' "$RUN_DIR/run.log" >&2
  exit 1
fi
```

Finally, run `npu-smi info`. Confirm that the smoke process exited and released
the selected device.

## Direct reply to Luka

Do not create a report file. Reply directly in the chat as plain text.

For success, include exactly these fields:

```text
310P MINERU STOCK SMOKE: PASS
repo_commit=
hostname=
physical_npu=
npu_name=
cann_version=
python=
vllm_source=
vllm_commit=
vllm_ascend_source=
vllm_ascend_commit=
package_versions=
model_dir=
model_manifest_sha256=
model_safetensors_sha256=
dataset_json=
dataset_json_sha256=
dataset_pages=
images_dir=
image_manifest_sha256=
omnidocbench_repo=
omnidocbench_commit=
run_dir=
engine_setup_s=
inference_s=
benchmark_wall_s=
pages_per_s_smoke_only=
markdown_bytes=
content_json_bytes=
static_kernel=off
npu_released=yes
```

If any phase fails, stop and reply with only:

```text
310P MINERU STOCK SMOKE: ISSUE
phase=
command=
exit_code=
first_causal_error=
paths_checked=
```

Do not add a proposed change, patch, workaround, or Markdown report.
