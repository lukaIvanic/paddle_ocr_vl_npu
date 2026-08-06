# Work-server 310P UniRec validation ladder

This is the pull-only execution brief for the AI agent on Luka's Atlas 310P
server. Read `AGENTS.md` first. Do not edit tracked files, create branches,
commit, or push from the work server.

Run all shell blocks with Bash. Several commands use Bash `PIPESTATUS` to
preserve the exit code of a command whose output is sent through `tee`.

## Current task: run Phase U1 only

Luka reported that Phase U0 passed. Do not repeat U0 unless a U1 prerequisite
gate shows that its environment or checkpoint is missing. Run Phase U1 below,
then stop.

Phase U0 remains in this file as the provenance record for the environment and
recognizer checkpoint.

## Phase U0 (complete): eager recognizer bootstrap

The U0 goal was deliberately narrow:

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

Phase U0 is complete. Continue only with the Phase U1 instructions below.

## Phase U1: faithful one-page OpenDoc eager comparison

### Goal

Run one complete OmniDocBench page through the official OpenDoc page contract:

```text
PP-DocLayoutV2 ONNX on CPU
  -> official OpenDoc crops and labels
  -> stock UniRec ONNX on CPU and local UniRec eager FP16 on NPU
  -> official OpenDoc postprocessing and page assembly
```

Both recognizers must receive each exact in-memory crop. The local result is
returned to the page assembler. The stock result is a comparison oracle only.

This phase proves the page-level integration contract. It is not a performance
benchmark. Keep `max_parallel_blocks=1`, eager decoding, and one page. Do not
use TorchAir, `torch.compile`, ACLGraph, batching, continuous decode, or the NPU
layout implementation.

Report progress after every U1 section. Stop if a command has no visible
progress for five minutes.

### U1.1: update and reuse the U0 environment

Run with Bash:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

PROJECT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_u1_page_compare_$COMMIT_SHORT"
mkdir -p "$OUT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/unirec_310p_py312/bin/python}"
MODEL="${MODEL:-$HOME/models/unirec-0.1b}"
IMAGES_DIR="${IMAGES_DIR:-/home/lukaiv/datasets/OmniDocBench/images}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$IMAGES_DIR"
test "$(stat -c %s "$MODEL/model.pth")" = "535901578"
printf '%s  %s\n' \
  b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef \
  "$MODEL/model.pth" | sha256sum -c -
```

Use the same NPU environment activation and free-device selection that passed
U0. Do not select a busy device. Run the U0 NPU tensor gate again only if the
new shell has not yet imported and exercised `npu:0`.

### U1.2: pinned OpenOCR source checkout

Use this exact upstream source:

```text
Repository: https://github.com/Topdu/OpenOCR.git
Commit:     0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
```

Keep this dependency in its own checkout. Do not alter an existing OpenOCR
checkout with unknown local changes:

```sh
DEPS_ROOT="${DEPS_ROOT:-$HOME/deps}"
OPENOCR_ROOT="$DEPS_ROOT/OpenOCR_0d522801"
mkdir -p "$DEPS_ROOT"

if ! test -d "$OPENOCR_ROOT/.git"; then
  git clone --filter=blob:none https://github.com/Topdu/OpenOCR.git "$OPENOCR_ROOT"
fi

git -C "$OPENOCR_ROOT" fetch origin 0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
git -C "$OPENOCR_ROOT" checkout --detach \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
test -z "$(git -C "$OPENOCR_ROOT" status --short)"
test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  "0d522801ec6dc1df852c6b6d4ed6a08f5127ed97"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
```

Do not install OpenOCR as a package. The repository runner imports this pinned
checkout directly.

### U1.3: minimal full-page Python dependencies

Reuse the U0 venv. First test the required imports:

```sh
set +e
"$PYTHON_BIN" - <<'PY' >"$OUT/u1_import_gate_before.log" 2>&1
import cv2
import numpy
import onnxruntime
import pydantic
import torch
import torch_npu
import transformers
from pydantic import BaseModel, computed_field, model_validator
print("imports=ok")
PY
IMPORT_STATUS=$?
set -e
cat "$OUT/u1_import_gate_before.log"
```

If `IMPORT_STATUS` is nonzero, install only these user-space packages into the
U0 venv. Do not install or replace torch, torch-npu, CANN, Transformers,
tokenizers, or huggingface-hub:

```sh
"$PYTHON_BIN" -m pip install \
  onnxruntime==1.27.0 \
  opencv-python-headless==4.10.0.84 \
  pydantic==2.13.4 \
  tqdm==4.68.2 \
  2>&1 | tee "$OUT/u1_pip_install.log"
```

Record the effective versions and enforce that U0's core versions did not
change:

```sh
"$PYTHON_BIN" - <<'PY' | tee "$OUT/u1_package_fingerprint.log"
import importlib.metadata as metadata
import sys

import cv2
import numpy
import onnxruntime
import pydantic
import torch
import torch_npu
import transformers

print("python=", sys.executable)
for name in (
    "torch", "torch-npu", "transformers", "tokenizers",
    "huggingface-hub", "numpy", "onnxruntime",
    "opencv-python-headless", "pydantic", "tqdm",
):
    try:
        print(f"{name}=", metadata.version(name))
    except metadata.PackageNotFoundError:
        print(f"{name}=<not-installed-as-distribution>")
print("cv2=", cv2.__version__)
assert torch.__version__ == "2.10.0+cpu"
assert torch_npu.__version__ == "2.10.0"
assert transformers.__version__ == "4.49.0"
assert torch.npu.is_available()
torch.npu.set_compile_mode(jit_compile=False)
PY
```

If the U0 report recorded a different base torch or torch-npu version, preserve
that passed U0 stack and report the difference instead of replacing it.

### U1.4: exact OpenDoc ONNX artifacts

Use these official Hugging Face repositories and revisions:

```text
Layout repository: topdu/PP_DoclayoutV2_onnx
Layout revision:   4de0e57bf2a3f7e818f1ad0cbf2961c33818a07b

Stock UniRec repository: topdu/unirec_0_1b_onnx
Stock UniRec revision:   e8f55f72013d544cbd6c27c576946ac63da247ed
```

Download into persistent cache directories:

```sh
OPENOCR_CACHE="${OPENOCR_CACHE:-$HOME/.cache/openocr}"
LAYOUT_ONNX_DIR="$OPENOCR_CACHE/PP_DoclayoutV2_onnx"
UNIREC_ONNX_DIR="$OPENOCR_CACHE/unirec_0_1b_onnx"
mkdir -p "$OPENOCR_CACHE"

LAYOUT_ONNX_DIR="$LAYOUT_ONNX_DIR" \
UNIREC_ONNX_DIR="$UNIREC_ONNX_DIR" \
"$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUT/u1_onnx_download.log"
import os
from huggingface_hub import snapshot_download

print("download=layout", flush=True)
print(snapshot_download(
    repo_id="topdu/PP_DoclayoutV2_onnx",
    revision="4de0e57bf2a3f7e818f1ad0cbf2961c33818a07b",
    local_dir=os.environ["LAYOUT_ONNX_DIR"],
), flush=True)

print("download=stock_unirec", flush=True)
print(snapshot_download(
    repo_id="topdu/unirec_0_1b_onnx",
    revision="e8f55f72013d544cbd6c27c576946ac63da247ed",
    local_dir=os.environ["UNIREC_ONNX_DIR"],
), flush=True)
PY
```

Set and verify the four runtime artifacts:

```sh
LAYOUT_ONNX="$LAYOUT_ONNX_DIR/PP-DoclayoutV2.onnx"
STOCK_ENCODER="$UNIREC_ONNX_DIR/unirec_encoder.onnx"
STOCK_DECODER="$UNIREC_ONNX_DIR/unirec_decoder.onnx"
STOCK_TOKENIZER="$UNIREC_ONNX_DIR/unirec_tokenizer_mapping.json"

test "$(stat -c %s "$LAYOUT_ONNX")" = "213963712"
test "$(stat -c %s "$STOCK_ENCODER")" = "164645880"
test "$(stat -c %s "$STOCK_DECODER")" = "554415797"
test "$(stat -c %s "$STOCK_TOKENIZER")" = "9843868"

printf '%s  %s\n' \
  2009fcb35e64085ab9f6f2b27aca550edc29a040a24f7d6a0f05b74a2f804860 \
  "$LAYOUT_ONNX" | sha256sum -c -
printf '%s  %s\n' \
  eb6b5d38f16c4f1abc39bab47e8d4ec83103b3246f642bde599edf6634289392 \
  "$STOCK_ENCODER" | sha256sum -c -
printf '%s  %s\n' \
  334326c877856d7595614424f1e15d6b1aab6ba24c26c769f09ab38901d5021c \
  "$STOCK_DECODER" | sha256sum -c -
printf '%s  %s\n' \
  769c66284765415c022ec213ea7b5e5266824aef5c5ea61d72b9ef087da525cc \
  "$STOCK_TOKENIZER" | sha256sum -c -
```

Stop on any size or hash mismatch. Do not compare mismatched checkpoints.

### U1.5: fixed one-page input

Use the same first sorted OmniDocBench page as the 910B reference:

```sh
PAGE="$IMAGES_DIR/PPT_1001115_eng_page_003.png"
test -f "$PAGE"
"$PYTHON_BIN" - "$PAGE" <<'PY' | tee "$OUT/u1_page_identity.log"
import hashlib
import sys
from pathlib import Path
from PIL import Image

path = Path(sys.argv[1]).resolve()
with Image.open(path) as image:
    print("path=", path)
    print("format=", image.format)
    print("size=", image.size)
    image.verify()
print("bytes=", path.stat().st_size)
print("sha256=", hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Do not substitute a crop, contact sheet, or another page. If this dataset page
is missing, stop and report `DATASET_MISMATCH`.

### U1.6: full-page eager comparison run

Record the exact expanded command, then execute it with unbuffered output:

```sh
RUN_OUTPUT="$OUT/output"
mkdir -p "$RUN_OUTPUT"

command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/run_opendoc_custom_unirec.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_ONNX"
  --layout-backend onnx_cpu
  --stock-encoder "$STOCK_ENCODER"
  --stock-decoder "$STOCK_DECODER"
  --stock-tokenizer-mapping "$STOCK_TOKENIZER"
  --input "$PAGE"
  --output-dir "$RUN_OUTPUT"
  --mode compare
  --device npu:0
  --dtype float16
  --max-length 256
  --decode-mode eager
  --limit 1
)

{
  printf 'project_commit=%s\n' "$PROJECT_COMMIT"
  printf 'openocr_commit=%s\n' "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$OUT/command.txt"

export PYTHONUNBUFFERED=1
set -o pipefail
SECONDS=0
"${command[@]}" 2>&1 | tee "$OUT/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$RUN_STATUS" >"$OUT/exit_code.txt"
printf '%s\n' "$SECONDS" >"$OUT/wall_seconds.txt"
test "$RUN_STATUS" = 0
```

Expected progress markers include setup begin/end, one page begin/end, six
`UNIREC_CROP_END` records, and `OPENDOC_CUSTOM_RUN_END`. If the count is not
six, report the actual count. Do not force it to six by changing layout output.

### U1.7: mechanical result analysis

Run this analyzer without modifying the outputs:

```sh
RUN_OUTPUT="$RUN_OUTPUT" "$PYTHON_BIN" - <<'PY' \
  | tee "$OUT/u1_result_analysis.log"
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_OUTPUT"])
summary = json.loads((root / "run_summary.json").read_text())
records = [
    json.loads(line)
    for line in (root / "recognition_comparison.jsonl").read_text().splitlines()
    if line.strip()
]

assert summary["status"] == "ok"
assert summary["mode"] == "compare"
assert summary["device"] == "npu:0"
assert summary["dtype"] == "float16"
assert summary["decode_mode"] == "eager"
assert summary["layout_backend"] == "onnx_cpu"
assert summary["page_count"] == 1
assert summary["crop_count"] == len(records) > 0
assert all("custom_error" not in record for record in records)
assert all(record["custom"]["text"].strip() for record in records)
assert all(record["preprocess"]["stock_shape"] == record["preprocess"]["custom_shape"] for record in records)
assert max(record["preprocess"]["max_abs"] for record in records) <= 2e-7

print("page_count=", summary["page_count"])
print("crop_count=", summary["crop_count"])
print("pipeline_wall_s=", summary["pipeline_wall_s"])
print("token_exact=", f'{summary["token_exact_count"]}/{len(records)}')
print("raw_text_exact=", f'{summary["raw_text_exact_count"]}/{len(records)}')
print("postprocessed_exact=", f'{summary["postprocessed_text_exact_count"]}/{len(records)}')
print("preprocess_max_abs=", max(record["preprocess"]["max_abs"] for record in records))
for record in records:
    comparison = record["comparison"]
    custom = record["custom"]
    print(json.dumps({
        "crop_index": record["crop_index"],
        "label": record["block_label"],
        "crop_size": record["crop_size"],
        "custom_tokens": custom["token_count"],
        "custom_total_s": custom["timing_s"]["total"],
        "custom_decode_tok_s": custom["decode_tokens_per_s"],
        "token_exact": comparison["token_exact"],
        "first_token_difference": comparison["first_token_difference"],
        "raw_text_exact": comparison["raw_text_exact"],
        "postprocessed_exact": comparison["postprocessed_text_exact"],
        "stock_text": record["stock"]["text"],
        "custom_text": custom["text"],
    }, ensure_ascii=False))
PY
```

On the known 910B reference, this page has six recognized crops and all six
stock/custom token sequences are exact at eager FP16. Exact token parity is a
useful 310P result, but it is not a hard pass criterion because NPU numerical
behavior can change an argmax without violating the integration contract.

Classify U1 as:

- `PASS_EXACT`: page completes and every crop is token-, raw-text-, and
  postprocessed-text-exact;
- `PASS_NUMERIC_DIFF`: page completes, every output is coherent, but one or
  more stock/custom generations differ;
- `FAIL_INTEGRATION`: crash, missing page output, empty/degenerate generation,
  preprocessing shape mismatch, or preprocessing max-abs above `2e-7`.

Do not hide token differences. Include both texts and the first differing token
index for each differing crop.

### U1 final report

Write:

```text
tmp/12_unirec_0_1b_inference/310p_u1_page_compare_<commit>/agent_report.md
```

Include:

- U0 report path and the statement that it passed;
- project and OpenOCR commits;
- host, NPU, Python, torch, torch-npu, CANN, driver, and firmware;
- dependency versions;
- all checkpoint revisions, sizes, and SHA-256 checks;
- page path, size, dimensions, and SHA-256;
- exact command, exit code, setup time, page wall time, and crop count;
- per-crop label, token count, latency, decode tokens/s, both texts, and parity;
- U1 classification;
- first causal error, or `none`;
- what is proven and what is not proven.

Report back to Luka in one sentence:

```text
UNIREC_310P_U1: PASS_EXACT — page=1 crops=6 token_exact=6/6 wall=<s>; report=<path>
```

or:

```text
UNIREC_310P_U1: PASS_NUMERIC_DIFF — page=1 crops=<n> token_exact=<x>/<n> coherent=<n>/<n>; report=<path>
```

or:

```text
UNIREC_310P_U1: FAIL_INTEGRATION — <first causal error>; report=<path>
```

Stop after U1. Do not start compilation, NPU layout, batching, multi-page
inference, or OmniDocBench evaluation.
