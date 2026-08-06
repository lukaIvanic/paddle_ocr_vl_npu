# Work-server 310P UniRec validation ladder

This is the pull-only execution brief for the AI agent on Luka's Atlas 310P
server. Read `AGENTS.md` first. Do not edit tracked files, create branches,
commit, or push from the work server.

Run all shell blocks with Bash. Several commands use Bash `PIPESTATUS` to
preserve the exit code of a command whose output is sent through `tee`.

## Current task: run Phase U0 only

The immediate goal is deliberately narrow:

> Install the user-space dependencies for Experiment 12, download and verify
> the exact OpenDoc UniRec checkpoint, and prove that the repository-owned
> UniRec implementation performs eager FP16 recognition on the 310P.

Do not run OpenDoc, PP-DocLayoutV2, ONNX Runtime, page inference, batching,
TorchAir, `torch.compile`, ACLGraph, or performance sweeps in this phase. Those
are later gates. Do not install or replace PyTorch, torch-npu, CANN, the NPU
driver, or firmware.

Report progress after each numbered section. If a download or installation
makes no visible progress for five minutes, stop and report the exact command,
last output, and artifact path. Do not silently wait.

## Required checkpoint identity

Use this checkpoint and no other checkpoint:

```text
Repository: topdu/unirec-0.1b
Source:     https://huggingface.co/topdu/unirec-0.1b
Revision:   a377e00d62c01b6544603e2a90f2cffe2a0388e1
File:       model.pth
Size:       535901578 bytes
SHA-256:    b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef
```

Do **not** download `topdu/unirec_0_1b`. The underscore repository contains a
different safetensors checkpoint. It is not output-equivalent to the
`model.pth` checkpoint used by OpenDoc and the current Experiment 12 pipeline.

The complete pinned snapshot must include at least:

```text
config.json
model.pth
tokenizer.json
tokenizer_config.json
```

Known official metadata at the pinned revision:

```text
config.json    704 bytes
model.pth      535901578 bytes
tokenizer.json 9498873 bytes
```

## Phase U0.1: checkout and environment identity

Start from the existing clone. Do not assume its absolute path if the current
shell is already inside it:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD
```

If tracked changes prevent the fast-forward pull, stop. Do not discard them.

Create an evidence root:

```sh
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_eager_bootstrap_$COMMIT_SHORT"
mkdir -p "$OUT"
```

Record the host and NPU state:

```sh
{
  date --iso-8601=seconds 2>/dev/null || date
  hostname
  uname -a
  npu-smi info
  command -v python || true
  command -v python3 || true
  command -v npu-setup || true
} >"$OUT/environment_discovery.log" 2>&1
```

Use the same environment activation and free NPU selection mechanism already
validated for Experiment 09 on this server. Never terminate another user's
process. If no NPU is free, stop and report that blocker.

The expected existing base stack is:

```text
Python:    3.12.13
torch:     2.10.0+cpu
torch-npu: 2.10.0
CANN:      9.1.0-beta.1
```

These are expected values, not permission to install them. If the selected
runtime differs, report the exact versions before continuing.

Resolve the intended base Python. Prefer the already validated Experiment 09
Python. Use `/usr/local/python3.12.13/bin/python` only if it exists and imports
the current NPU stack:

```sh
if test -x /usr/local/python3.12.13/bin/python; then
  BASE_PYTHON=/usr/local/python3.12.13/bin/python
elif test -x /usr/local/python3.12.13/bin/python3; then
  BASE_PYTHON=/usr/local/python3.12.13/bin/python3
else
  BASE_PYTHON="$(command -v python)"
fi
test -x "$BASE_PYTHON"
```

After activating the NPU environment, run this exact gate:

```sh
"$BASE_PYTHON" - <<'PY' | tee "$OUT/npu_python_gate.log"
import platform
import sys

import torch
import torch_npu

print("python_executable=", sys.executable)
print("python_version=", sys.version.replace("\n", " "))
print("machine=", platform.machine())
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("npu_available=", torch.npu.is_available())
print("npu_count=", torch.npu.device_count())
print("device_0=", torch.npu.get_device_name(0))
assert torch.npu.is_available()
assert torch.npu.device_count() >= 1
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(8, dtype=torch.float16, device="npu:0")
print("npu_result=", (x + 1).cpu().tolist())
PY
```

The phase must stop if this gate fails. Do not create a different PyTorch
environment to hide a base-stack failure.

## Phase U0.2: isolated user-space environment

Create a system-site-packages venv. This reuses the working torch-npu stack and
isolates only the Python-level UniRec dependencies:

```sh
VENV_ROOT="${VENV_ROOT:-$HOME/venvs}"
VENV="$VENV_ROOT/unirec_310p_py312"
if ! test -x "$VENV/bin/python"; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
PYTHON_BIN="$VENV/bin/python"
```

Install the exact Experiment 12 dependency pins. Do not include `torch` or
`torch-npu` in the pip command:

```sh
"$PYTHON_BIN" -m pip install \
  transformers==4.49.0 \
  tokenizers==0.21.4 \
  huggingface-hub==0.36.2 \
  2>&1 | tee "$OUT/pip_install.log"
```

Verify the runtime after installation:

```sh
"$PYTHON_BIN" - <<'PY' | tee "$OUT/package_fingerprint.log"
import importlib.metadata as metadata
import sys

import numpy
import PIL
import torch
import torch_npu
import transformers

print("python=", sys.executable)
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("transformers=", transformers.__version__)
for name in ("tokenizers", "huggingface-hub", "numpy", "Pillow"):
    print(f"{name}=", metadata.version(name))
assert transformers.__version__ == "4.49.0"
assert torch.npu.is_available()
torch.npu.set_compile_mode(jit_compile=False)
PY
```

If NumPy or Pillow is missing, install only the missing package. Record the
selected version. Do not upgrade a working transitive dependency without a
specific import error.

## Phase U0.3: exact checkpoint download

Use a persistent model directory outside the Git checkout:

```sh
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="$MODEL_ROOT/unirec-0.1b"
mkdir -p "$MODEL_ROOT"
```

Download the complete pinned Hugging Face snapshot. This call gives progress,
pins the revision, and safely passes the absolute destination through the
environment:

```sh
MODEL="$MODEL" "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUT/model_download.log"
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="topdu/unirec-0.1b",
    revision="a377e00d62c01b6544603e2a90f2cffe2a0388e1",
    local_dir=os.environ["MODEL"],
)
print("snapshot_path=", path)
PY
```

Use the command as written. Do not use `main` as the revision.

Verify every required artifact before model loading:

```sh
test -f "$MODEL/config.json"
test -f "$MODEL/model.pth"
test -f "$MODEL/tokenizer.json"
test -f "$MODEL/tokenizer_config.json"
test "$(stat -c %s "$MODEL/model.pth")" = "535901578"
printf '%s  %s\n' \
  b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef \
  "$MODEL/model.pth" | sha256sum -c - | tee "$OUT/model_checksum.log"
find "$MODEL" -maxdepth 1 -type f -printf '%f %s bytes\n' \
  | sort >"$OUT/model_files.txt"
```

If Hugging Face is inaccessible but ModelScope is available, the official
mirror is `topdktu/unirec-0.1b`. It is acceptable only if the downloaded
`model.pth` passes the exact size and SHA-256 gates above. Do not accept a file
because its directory name looks correct.

## Phase U0.4: one-crop eager FP16 gate

Use the repository's first real recognition crop:

```sh
CROP="$REPO/crops/crop_01_text_block_en.png"
test -f "$CROP"
```

Run the local implementation. Keep NPU JIT compilation disabled and use eager
decode only:

```sh
set -o pipefail
"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/run_optimized.py" \
  --model-path "$MODEL" \
  --image "$CROP" \
  --device npu:0 \
  --dtype float16 \
  --max-length 64 \
  --decode-mode eager \
  --npu-jit-compile off \
  --output-json "$OUT/one_crop.json" \
  2>&1 | tee "$OUT/one_crop.log"
ONE_CROP_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$ONE_CROP_STATUS" >"$OUT/one_crop_exit_code.txt"
test "$ONE_CROP_STATUS" = 0
```

Pass criteria:

- process exit code is zero;
- output JSON has `status: ok`;
- device is `npu:0` and dtype is `float16`;
- the model loads `model.pth` from the verified directory;
- at least one token is generated;
- decoded text is non-empty and not an obvious repeated-token degeneration;
- no CPU/CUDA fallback, TorchAir compilation, or NPU runtime error occurs.

Stop at the first causal error. Preserve the complete traceback. Do not patch
the model or retry with another checkpoint.

## Phase U0.5: six-crop eager sanity set

Only after the one-crop gate passes, run the script without `--image`. This
uses the six committed repository crops:

```sh
set -o pipefail
"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/run_optimized.py" \
  --model-path "$MODEL" \
  --device npu:0 \
  --dtype float16 \
  --max-length 128 \
  --decode-mode eager \
  --npu-jit-compile off \
  --output-json "$OUT/six_crops.json" \
  2>&1 | tee "$OUT/six_crops.log"
SIX_CROP_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$SIX_CROP_STATUS" >"$OUT/six_crops_exit_code.txt"
test "$SIX_CROP_STATUS" = 0
```

For each crop, report:

- generated token count;
- generated text;
- prefill, decode, and total latency;
- decode tokens/s;
- whether the text is coherent for the crop type.

This is a compatibility gate. Do not interpret one cold eager run as final
310P performance.

## Final report

Write a compact report to:

```text
tmp/12_unirec_0_1b_inference/310p_eager_bootstrap_<commit>/agent_report.md
```

The report must include:

- project commit, host, NPU model, selected logical device;
- Python, torch, torch-npu, CANN, driver, and firmware versions;
- exact pip package versions;
- model repository, pinned revision, size, and SHA-256 result;
- exact commands and artifact paths;
- one-crop and six-crop exit codes;
- six generated texts and timing summaries;
- first causal error, or `none`;
- what is proven and what is not proven.

Then report back to Luka in one sentence only:

```text
UNIREC_310P_U0: PASS — exact model verified; eager FP16 passed 1/1 and 6/6; report=<path>
```

or:

```text
UNIREC_310P_U0: BLOCKED at U0.<section> — <first causal error>; report=<path>
```

Stop after Phase U0. Do not begin OpenDoc or layout setup until Luka supplies
the next phase.
