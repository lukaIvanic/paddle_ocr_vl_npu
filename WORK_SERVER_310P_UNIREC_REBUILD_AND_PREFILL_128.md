# Work-server 310P UniRec rebuild and eager-layout prefill-128

The previous work-server environment and its 310P graph cache are unavailable.
The replacement environment still has CANN, PyTorch, and torch-npu, and it has
the 1,651 OmniDocBench images. Reconstruct every other input from pinned
sources, compile a new recognition-prefill cache on this 310P, then run the
first-128 W8/T8 prefill-only benchmark with eager FP32 layout.

Do not retry compiled layout. Its prior run failed during worker-setup warmup
with `IndexByTensor` before page processing.

## Immutable provenance

Use these exact sources:

```text
UniRec repository: topdu/unirec-0.1b
UniRec revision:   a377e00d62c01b6544603e2a90f2cffe2a0388e1

Layout repository: PaddlePaddle/PP-DocLayoutV2_safetensors
Layout revision:   880e8971b88938518611c54fc0f59ad57849c9d4

OpenOCR repository: https://github.com/Topdu/OpenOCR.git
OpenOCR commit:     0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
```

The graph cache is hardware/runtime specific. Build it on the selected 310P.
Never copy a 910B cache or a cache from another environment.

## Restrictions

- Read `CLAUDE.md` and `AGENTS.md` first.
- The work-server checkout is pull-only. Do not edit tracked source, branch,
  commit, or push.
- Do not install or replace CANN, the driver, firmware, PyTorch, torch-npu, or
  torchvision.
- Use exactly one genuinely free physical 310P. Never use physical device 5.
- Never stop another user's process.
- Keep layout eager FP32 and recognition FP16.
- Stop on any revision, size, or SHA-256 mismatch.
- Stop if the inherited NPU stack cannot execute the tensor gate.
- Setup, downloads, compilation, and graph warmup are outside producer
  throughput.

## 1. Pull and create the evidence root

Run all blocks with Bash:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_rebuild_prefill128_$COMMIT_SHORT"
mkdir -p "$OUT"
```

If tracked changes block the pull, stop. Do not discard them.

## 2. Discover and validate the inherited NPU runtime

Activate the replacement server's intended CANN/torch-npu environment first.
Do not invent a new activation script. Select one free device and export it as
the only entry in `ASCEND_RT_VISIBLE_DEVICES`; reject device 5.

Resolve a base Python that already imports the installed torch-npu:

```sh
BASE_PYTHON="${BASE_PYTHON:-}"
if test -z "$BASE_PYTHON"; then
  for candidate in \
    /usr/local/python3.12.13/bin/python \
    /usr/local/python3.12.13/bin/python3 \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
  do
    if test -x "$candidate" && \
       "$candidate" -c 'import torch, torch_npu; assert torch.npu.is_available()' \
       >/dev/null 2>&1
    then
      BASE_PYTHON="$candidate"
      break
    fi
  done
fi
test -x "$BASE_PYTHON"
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",$ASCEND_RT_VISIBLE_DEVICES," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1
```

Record CANN instead of assuming its version:

```sh
{
  date --iso-8601=seconds 2>/dev/null || date
  hostname
  uname -a
  printf 'base_python=%s\n' "$BASE_PYTHON"
  printf 'visible_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  npu-smi info
  env | sort | grep '^ASCEND_' || true
  find /usr/local/Ascend -maxdepth 6 -type f \
    \( -name 'ascend_toolkit_install.info' -o -name 'version.info' \
       -o -name 'version.cfg' \) -print -exec sed -n '1,80p' {} \; 2>/dev/null || true
} >"$OUT/base_environment.log" 2>&1
```

Run the actual NPU gate:

```sh
"$BASE_PYTHON" - <<'PY' | tee "$OUT/base_npu_gate.log"
import importlib.metadata as metadata
import platform
import sys

import torch
import torch_npu
import torchvision

print("python=", sys.executable)
print("python_version=", sys.version.replace("\n", " "))
print("machine=", platform.machine())
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("torchvision=", torchvision.__version__)
print("npu_available=", torch.npu.is_available())
print("npu_count=", torch.npu.device_count())
print("device_0=", torch.npu.get_device_name(0))
assert torch.npu.is_available()
assert torch.npu.device_count() == 1
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(8, dtype=torch.float16, device="npu:0")
print("npu_result=", (x + 1).cpu().tolist())
PY
```

Do not require one specific CANN version. Record it, and label the final result
with it. The expected prior stack was torch `2.10.0+cpu`, torch-npu `2.10.0`,
and torchvision `0.25.0+cpu`. If these three differ, stop and report
`BASE_RUNTIME_MISMATCH`; do not replace them.

## 3. Rebuild the isolated user-space environment

Create a system-site-packages venv. This must inherit, not reinstall, the base
NPU stack:

```sh
VENV_ROOT="${VENV_ROOT:-$HOME/venvs}"
VENV="$VENV_ROOT/unirec_full_npu_310p_py312"
mkdir -p "$VENV_ROOT"
if ! test -x "$VENV/bin/python"; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
PYTHON_BIN="$VENV/bin/python"

"$PYTHON_BIN" -m pip install \
  numpy==1.26.4 \
  transformers==5.5.4 \
  tokenizers==0.22.2 \
  huggingface-hub==1.19.0 \
  safetensors==0.8.0 \
  onnxruntime==1.27.0 \
  opencv-python-headless==4.13.0.92 \
  pydantic==2.13.4 \
  tqdm==4.68.2 \
  kornia_rs==0.1.14 \
  Pillow==12.2.0 \
  PyYAML==6.0.2 \
  shapely==2.1.2 \
  2>&1 | tee "$OUT/pip_install.log"
```

Verify every effective dependency and ensure pip did not replace the inherited
NPU packages:

```sh
"$PYTHON_BIN" - <<'PY' | tee "$OUT/package_fingerprint.log"
import importlib.metadata as metadata
import sys

import cv2
import kornia_rs
import numpy
import onnxruntime
import PIL
import pydantic
import safetensors
import shapely
import torch
import torch_npu
import torchvision
import transformers
import yaml
from torchvision.io import decode_image

expected = {
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "torchvision": "0.25.0+cpu",
    "numpy": "1.26.4",
    "transformers": "5.5.4",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.19.0",
    "safetensors": "0.8.0",
    "onnxruntime": "1.27.0",
    "opencv-python-headless": "4.13.0.92",
    "pydantic": "2.13.4",
    "tqdm": "4.68.2",
    "kornia_rs": "0.1.14",
    "Pillow": "12.2.0",
    "PyYAML": "6.0.2",
    "shapely": "2.1.2",
}
print("python=", sys.executable)
for name, wanted in expected.items():
    actual = metadata.version(name)
    print(f"{name}={actual}")
    assert actual == wanted, (name, actual, wanted)
assert torch.npu.is_available()
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(4, dtype=torch.float32, device="npu:0")
print("npu_result=", (x * 2).cpu().tolist())
PY
```

## 4. Rebuild and verify pinned OpenOCR

```sh
DEPS_ROOT="${DEPS_ROOT:-$HOME/deps}"
OPENOCR_ROOT="$DEPS_ROOT/OpenOCR_0d522801"
mkdir -p "$DEPS_ROOT"

if ! test -d "$OPENOCR_ROOT/.git"; then
  git clone --filter=blob:none https://github.com/Topdu/OpenOCR.git "$OPENOCR_ROOT"
fi
git -C "$OPENOCR_ROOT" fetch origin \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
git -C "$OPENOCR_ROOT" checkout --detach \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97

test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  "0d522801ec6dc1df852c6b6d4ed6a08f5127ed97"
test -z "$(git -C "$OPENOCR_ROOT" status --short)"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
{
  printf 'commit=%s\n' "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)"
  printf 'tree=%s\n' "$(git -C "$OPENOCR_ROOT" rev-parse HEAD^{tree})"
  git -C "$OPENOCR_ROOT" status --short --branch
} | tee "$OUT/openocr_identity.log"
```

Do not install OpenOCR as a package. The runner imports this exact checkout.

## 5. Download and verify the exact UniRec checkpoint

```sh
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="$MODEL_ROOT/unirec-0.1b"
mkdir -p "$MODEL_ROOT"

MODEL="$MODEL" "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUT/unirec_download.log"
import os
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="topdu/unirec-0.1b",
    revision="a377e00d62c01b6544603e2a90f2cffe2a0388e1",
    local_dir=os.environ["MODEL"],
), flush=True)
PY
```

Require every critical file's exact size and SHA-256:

```sh
test "$(stat -c %s "$MODEL/config.json")" = 704
test "$(stat -c %s "$MODEL/model.pth")" = 535901578
test "$(stat -c %s "$MODEL/tokenizer.json")" = 9498873
test "$(stat -c %s "$MODEL/tokenizer_config.json")" = 8642053

printf '%s  %s\n' \
  cad5d58d58cd9aa268c334944d445a3b162e04bfd06a39cfa68aec7659aafe57 \
  "$MODEL/config.json" | sha256sum -c -
printf '%s  %s\n' \
  b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef \
  "$MODEL/model.pth" | sha256sum -c -
printf '%s  %s\n' \
  c08e9670bfdcd1dec8ec18b623e7c4adfbbac2f45f532e42044bcb0a15c5a7f5 \
  "$MODEL/tokenizer.json" | sha256sum -c -
printf '%s  %s\n' \
  42fc776afdea5659ad9b58cdc7b51f2522b2f734468d6718142ba0e65c3f1dc9 \
  "$MODEL/tokenizer_config.json" | sha256sum -c -
```

Do not substitute the underscore repository `topdu/unirec_0_1b`; it contains
a different checkpoint.

## 6. Download and verify the exact PP-DocLayoutV2 checkpoint

```sh
LAYOUT_MODEL="$MODEL_ROOT/PP-DocLayoutV2_safetensors"

LAYOUT_MODEL="$LAYOUT_MODEL" "$PYTHON_BIN" - <<'PY' \
  2>&1 | tee "$OUT/layout_download.log"
import os
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="PaddlePaddle/PP-DocLayoutV2_safetensors",
    revision="880e8971b88938518611c54fc0f59ad57849c9d4",
    local_dir=os.environ["LAYOUT_MODEL"],
), flush=True)
PY

test "$(stat -c %s "$LAYOUT_MODEL/config.json")" = 3787
test "$(stat -c %s "$LAYOUT_MODEL/preprocessor_config.json")" = 575
test "$(stat -c %s "$LAYOUT_MODEL/model.safetensors")" = 214798436

printf '%s  %s\n' \
  18a696b54c64c4fa582afcd3a41407c4b65a99dc7ab187ad2fed8af8e4128ad8 \
  "$LAYOUT_MODEL/config.json" | sha256sum -c -
printf '%s  %s\n' \
  56281a70c931a291dcaf653605fb4df713fd823f65e939aecd6005c26346a103 \
  "$LAYOUT_MODEL/preprocessor_config.json" | sha256sum -c -
printf '%s  %s\n' \
  e60f3725aeedc88fd319416ef166bda79171a41516a301c27cab9132dc2739d2 \
  "$LAYOUT_MODEL/model.safetensors" | sha256sum -c -
```

Load both model contracts on CPU before using the NPU:

```sh
MODEL="$MODEL" LAYOUT_MODEL="$LAYOUT_MODEL" "$PYTHON_BIN" - <<'PY' \
  | tee "$OUT/model_cpu_load_gate.log"
import os
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    PreTrainedTokenizerFast,
)

tokenizer = PreTrainedTokenizerFast.from_pretrained(os.environ["MODEL"])
processor = AutoImageProcessor.from_pretrained(os.environ["LAYOUT_MODEL"])
model = AutoModelForObjectDetection.from_pretrained(os.environ["LAYOUT_MODEL"])
print("tokenizer=", type(tokenizer).__name__)
print("layout_processor=", type(processor).__name__)
print("layout_model=", type(model).__name__)
print("layout_parameter_count=", sum(p.numel() for p in model.parameters()))
assert type(processor).__name__ == "PPDocLayoutV2ImageProcessor"
assert type(model).__name__ == "PPDocLayoutV2ForObjectDetection"
PY
```

Run the repository's four layout compatibility tests as a source gate:

```sh
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py" \
  | tee "$OUT/layout_compatibility_tests.log"
```

All four tests must pass before cache compilation.

## 7. Verify the 1,651-page input set

Set the actual replacement-server image directory, then verify it:

```sh
IMAGES_DIR="${IMAGES_DIR:-/home/lukaiv/datasets/OmniDocBench/images}"
IMAGES_DIR="$IMAGES_DIR" "$PYTHON_BIN" - <<'PY' | tee "$OUT/dataset_gate.log"
import hashlib
import os
from pathlib import Path
from PIL import Image

root = Path(os.environ["IMAGES_DIR"]).expanduser().resolve()
paths = sorted(
    path for path in root.iterdir()
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
)
print("root=", root)
print("count=", len(paths))
print("first=", paths[0].name)
print("last=", paths[-1].name)
assert len(paths) == 1651
page = root / "PPT_1001115_eng_page_003.png"
assert page.is_file()
with Image.open(page) as image:
    print("fixed_page_size=", image.size)
    image.verify()
print("fixed_page_sha256=", hashlib.sha256(page.read_bytes()).hexdigest())
PY
```

## 8. Build and validate six recognition graphs on the 310P

Use a new empty cache. One worker is the only cache writer. This pass compiles
and executes the five masked full-vision graphs plus packed S1024 cross-KV.
Layout stays eager.

```sh
CACHE_ROOT="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_rebuild_$COMMIT_SHORT"
RECOGNITION_CACHE="$CACHE_ROOT/recognition"
UNUSED_LAYOUT_CACHE="$CACHE_ROOT/unused_layout"
test ! -e "$CACHE_ROOT"
mkdir -p "$RECOGNITION_CACHE" "$UNUSED_LAYOUT_CACHE"

BUILD="$OUT/cache_build_w1_t16"
mkdir -p "$BUILD/output"
build_command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output-dir "$BUILD/output"
  --dtype float16
  --offset 0
  --limit 16
  --workers 1
  --warmup-pages 2
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution eager
  --layout-batch-size 1
  --cross-cache-length 512
  --layout-cache-dir "$UNUSED_LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --recognition-preprocess-threads 16
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)

printf '%q ' "${build_command[@]}" >"$BUILD/command.txt"
printf '\n' >>"$BUILD/command.txt"
export PYTHONUNBUFFERED=1
set -o pipefail
SECONDS=0
"${build_command[@]}" 2>&1 | tee "$BUILD/run.log"
BUILD_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$BUILD_STATUS" >"$BUILD/exit_code.txt"
printf '%s\n' "$SECONDS" >"$BUILD/wall_seconds.txt"
test "$BUILD_STATUS" = 0
test -f "$BUILD/output/summary.json"
```

Cold compilation can be quiet. Do not call it a hang while the owned process
or selected NPU remains active. Never terminate an unowned process.

Verify the exact current graph directories and require each to contain files:

```sh
HASHES="$OUT/current_recognition_source_hashes.txt"
(
  cd "$REPO/12_unirec_0_1b_inference"
  "$PYTHON_BIN" - <<'PY'
import text_packed_prefill
import vision_full_batch

print(f"VISION_SOURCE_HASH={vision_full_batch._source_hash()}")
print(f"TEXT_SOURCE_HASH={text_packed_prefill._source_hash()}")
PY
) | tee "$HASHES"
VISION_SOURCE_HASH="$(sed -n 's/^VISION_SOURCE_HASH=//p' "$HASHES")"
TEXT_SOURCE_HASH="$(sed -n 's/^TEXT_SOURCE_HASH=//p' "$HASHES")"

for graph in \
  "vision_full_bucket_960x64_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x256_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x256_b4_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x512_b8_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x512_b4_float16_src$VISION_SOURCE_HASH" \
  "text_prefill_packed_b1_s1024_float16_src$TEXT_SOURCE_HASH"
do
  test -d "$RECOGNITION_CACHE/$graph"
  test -n "$(find "$RECOGNITION_CACHE/$graph" -type f -print -quit)"
done
find "$RECOGNITION_CACHE" -maxdepth 1 -type d -printf '%f\n' \
  | sort | tee "$OUT/recognition_graph_dirs.txt"
du -sh "$RECOGNITION_CACHE" | tee "$OUT/recognition_cache_size.txt"
```

Require the build summary to report `status=ok`, 16 pages, eager layout, all
five full-vision graphs warmed, packed S1024 cross-KV warmed, nonzero crops,
and validation passed. If not, stop before W8.

## 9. Run the first-128 W8/T8 producer

```sh
RUN="$OUT/eager_layout_w8_t8_first128"
mkdir -p "$RUN/output"
command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output-dir "$RUN/output"
  --dtype float16
  --offset 0
  --limit 128
  --workers 8
  --warmup-pages 8
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution eager
  --layout-batch-size 1
  --cross-cache-length 512
  --layout-cache-dir "$UNUSED_LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --recognition-preprocess-threads 8
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$RUN/command.txt"

SECONDS=0
"${command[@]}" 2>&1 | tee "$RUN/run.log"
STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$STATUS" >"$RUN/exit_code.txt"
printf '%s\n' "$SECONDS" >"$RUN/wall_seconds.txt"
test "$STATUS" = 0
test -f "$RUN/output/summary.json"
```

If W8 setup OOMs, preserve the first causal failure and stop. Do not silently
reduce workers.

## 10. Validate and report

For the W8 result require `status=ok`, offset 0, 128 pages, W8/T8, eager FP32
layout, compact HWC, cross-KV 512, discard storage, all eight workers active,
all six recognition graphs loaded from the rebuilt cache, no unexpected graph
first calls after warmup, and validation passed.

Report producer wall, pages/s, crops, real source tokens/s, real/physical vision
rows, slot efficiency, fallback and rejected counts, worker page distribution,
layout time, CPU crop preparation, NPU recognition prefill, D2H, packing, IPC,
file I/O, setup, warmup, shutdown, total wall, peak HBM, maximum RSS, and the
discovered CANN/torch/torch-npu versions.

Compare against:

```text
310P prior eager-layout W1/T16: 2.78 pages/s
910B compiled-layout W8/T8:    27.3511137111 pages/s
910B real source rate:         13145.6290274 tokens/s
```

The 910B comparison is not a pure chip ratio because its layout was compiled.

Write `$OUT/agent_report.md`, then report one compact line:

```text
UNIREC_310P_REBUILD_PREFILL128: <PASS|BLOCKED_DOWNLOAD|BASE_RUNTIME_MISMATCH|HASH_MISMATCH|FAIL_COMPILE|OOM|FAIL_INTEGRATION> — cann=<v> torch=<v> torch_npu=<v> model_sha=pass layout_sha=pass openocr_commit=pass graphs=6 producer=<s> pg_s=<n> real_tok_s=<n> scale_vs_310p_w1=<n> ratio_vs_910b=<n> crops=<n> slot_eff=<n> fallback=<n> rejected=<n> layout=<s> setup=<s> peak_hbm=<MiB>; report=<path>
```

Then stop. Do not retry compiled layout, run decode, increase the page count,
or sweep worker counts.
