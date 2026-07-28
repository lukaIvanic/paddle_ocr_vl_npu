# Work-server 310P Experiment 09 validation ladder

This is the execution brief for the AI agent on Luka's replacement Atlas 310P
server. Read `CLAUDE.md` and `AGENTS.md` first.

## Goal

Prove that the real Experiment 09 pipeline runs on this 310P software stack,
identify which production optimizations work, and measure their approximate
effect on a small representative workload.

## Current layout IndexPut compatibility route

For this validation, keep every PaddleOCR-VL recognition stage on `npu:0` but
run PP-DocLayoutV3 on CPU by passing:

```text
--layout-device cpu
```

This is a device-boundary workaround for the current environment's missing NPU
IndexPut implementation. It does not change layout model code, recognition
model code, crops, prompts, scheduling, or OCR execution. The production summary
must show:

```text
configuration.device = "npu:0"
configuration.layout_device = "cpu"
layout_frontend.device = "cpu"
layout_frontend.graph_capture = false
```

Do not use the standalone Transformers one-crop example as the production
recognizer gate: that helper constructs MRoPE after moving its metadata tensors
to NPU. Experiment 09's serving preparation constructs MRoPE on CPU and copies
the completed tensor to NPU.

## Targeted NPU layout IndexPut compatibility check

Run this section separately after the existing Phase 1-7 results are preserved.
Do not change those completed runs. Pull `main`, confirm the checked-out commit
contains `043b957`, and restore the exact `PYTHON_BIN`, `LAYOUT_MODEL`,
`DATASET_JSON`, and `IMAGES_DIR` values already selected in Phase 0.

The patch changes one operation in Transformers 5.5.4 PP-DocLayoutV3 inference:
upstream allocates `spatial_shapes` on NPU and fills it through
`spatial_shapes[level, 0/1] = value`; the owned forward constructs the complete
device tensor with `full` and `stack`. It does not move layout computation to
CPU.

First isolate the two constructors without loading either model:

```sh
LAYOUT_PATCH_ROOT="$REPO/tmp/09_persistent_page_engine/310p_layout_indexput_patch"
mkdir -p "$LAYOUT_PATCH_ROOT"

run_targeted_layout() {
  local run_name="$1"
  shift
  local evidence_dir="$OUTPUT_ROOT/$run_name"
  mkdir -p "$evidence_dir"
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$evidence_dir/exit_code.txt"
  printf 'run=%s exit_code=%s log=%s\n' \
    "$run_name" "$status" "$evidence_dir/run.log"
  return "$status"
}

set +e
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_spatial_shapes.py" \
  --mode legacy_indexput \
  >"$LAYOUT_PATCH_ROOT/legacy_indexput.log" 2>&1
LEGACY_STATUS=$?
set -e
printf '%s\n' "$LEGACY_STATUS" \
  >"$LAYOUT_PATCH_ROOT/legacy_indexput.exit_code"

"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_spatial_shapes.py" \
  --mode single_constructor \
  >"$LAYOUT_PATCH_ROOT/single_constructor.log" 2>&1

"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_spatial_shapes.py" \
  --mode capture_constructor \
  >"$LAYOUT_PATCH_ROOT/capture_constructor.log" 2>&1

cat "$LAYOUT_PATCH_ROOT/legacy_indexput.log"
cat "$LAYOUT_PATCH_ROOT/single_constructor.log"
cat "$LAYOUT_PATCH_ROOT/capture_constructor.log"
```

Both replacement modes must print `"verdict": "PASS"`. The legacy mode is
diagnostic: preserve whether it passes or fails and its complete causal
traceback.

Then run exactly one real page through the patched layout model, first eager:

```sh
run_targeted_layout layout_indexput_patch_eager \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device npu \
  --layout-indexput-compat \
  --no-graph-capture \
  --offset 0 \
  --limit 1 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/layout_indexput_patch_eager/output"
```

Stop if eager fails. Preserve the first new causal traceback; do not add another
model patch. If eager passes, run the same page with production graph capture:

```sh
run_targeted_layout layout_indexput_patch_graph \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device npu \
  --layout-indexput-compat \
  --graph-capture \
  --offset 0 \
  --limit 1 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/layout_indexput_patch_graph/output"
```

For each model run report exit code, request count, request-manifest SHA256,
raw/filtered boxes, page wall, detector device time, and
`npu_indexput_compat`. Do not run multiple pages or change the production
ladder to NPU layout yet. Wait for Luka to compare the targeted result first.

## Immediate standalone IndexPut probe

Run this section by itself when Luka asks for the 310P advanced-indexing
environment probe. Do not load either model, run the Experiment 09 pipeline,
compile a graph, install or replace packages, or change source code. The point
is to test one bare NPU operation and preserve enough environment evidence for
Luka to compare separately.

First read `CLAUDE.md` and `AGENTS.md`, pull the current `main`, activate the
machine's intended NPU environment, and select the exact Python interpreter
that is meant to run Experiment 09:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_indexput_probe"
mkdir -p "$OUTPUT_ROOT"

PYTHON_BIN="$(command -v python)"
test -x "$PYTHON_BIN"
```

Record the software and resolved runtime paths. Missing optional files or
package-manager commands are evidence, not a reason to abort this collection:

```sh
{
  printf 'git_commit='
  git rev-parse HEAD
  printf 'hostname='
  hostname
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import os
import platform
import sys

import torch
import torch_npu

print("platform:", platform.platform())
print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_file:", torch.__file__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torch_npu_file:", torch_npu.__file__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
if torch.npu.is_available() and torch.npu.device_count():
    print("npu_0_name:", torch.npu.get_device_name(0))
for name in (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
):
    print(f"{name}={os.environ.get(name, '')}")
PY

  printf '\nresolved_paths:\n'
  for name in ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH; do
    eval "value=\${$name:-}"
    printf '%s=%s\n' "$name" "$value"
    test -n "$value" && readlink -f "$value" || true
  done

  printf '\npython_packages:\n'
  "$PYTHON_BIN" -m pip show torch torch-npu 2>&1 || true

  printf '\nascend_version_files:\n'
  for file in \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/firmware/version.info \
    "${ASCEND_HOME_PATH:-}/version.cfg" \
    "${ASCEND_HOME_PATH:-}/version.info" \
    "${ASCEND_OPP_PATH:-}/version.info"; do
    test -f "$file" || continue
    printf '\n--- %s ---\n' "$file"
    cat "$file"
  done

  printf '\ninstalled_ascend_packages:\n'
  command -v rpm >/dev/null &&
    rpm -qa | sort | grep -Ei 'ascend|cann|torch.npu' || true
  command -v dpkg-query >/dev/null &&
    dpkg-query -W 2>/dev/null | sort |
      grep -Ei 'ascend|cann|torch.npu' || true

  printf '\nnpu_smi:\n'
  command -v npu-smi >/dev/null && npu-smi info || true
} >"$OUTPUT_ROOT/environment.txt" 2>&1
```

Now run the exact minimal operation. The CPU execution is only a semantic
control; the NPU execution is the result of interest. The explicit synchronize
ensures an asynchronously reported kernel failure is captured inside this
probe:

```sh
set +e
"$PYTHON_BIN" - <<'PY' \
  >"$OUTPUT_ROOT/indexput_minimal_reproducer.txt" 2>&1
import traceback

import torch
import torch_npu


def run(device: str) -> torch.Tensor:
    destination = torch.zeros(
        (16, 16, 16), dtype=torch.float16, device=device
    )
    source = torch.ones(
        (1, 16, 16), dtype=torch.float16, device=device
    )
    destination[[5]] = source
    if device.startswith("npu"):
        torch.npu.synchronize()
    return destination.cpu()


cpu_result = run("cpu")
assert int(torch.count_nonzero(cpu_result).item()) == 16 * 16
print("CPU_CONTROL: PASS")

torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
try:
    npu_result = run("npu:0")
    torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    print("NPU_INDEXPUT: PASS")
except Exception:
    print("NPU_INDEXPUT: FAIL")
    traceback.print_exc()
    raise
PY
PROBE_EXIT_CODE=$?
set -e
printf '%s\n' "$PROBE_EXIT_CODE" \
  >"$OUTPUT_ROOT/indexput_minimal_reproducer_exit_code.txt"
```

Stop after this probe. Do not attempt a workaround. Report:

```text
310P STANDALONE INDEXPUT PROBE

Git commit:
Python executable:
CPU control: PASS | FAIL
NPU IndexPut: PASS | FAIL
Probe exit code:
First causal NPU error:
Artifact paths:
```

Include the complete contents of `environment.txt`,
`indexput_minimal_reproducer.txt`, and
`indexput_minimal_reproducer_exit_code.txt`. In particular, do not omit an
`aclnnIndexPutImpl`, `IndexPutV2`, missing operator binary, or dispatcher
message.

The required runner is:

```text
09_persistent_page_engine/scripts/run_omnidocbench.py
```

Its real path is:

```text
OmniDocBench page selection
  -> owned PaddleX-free layout frontend
  -> bounded page/crop production
  -> ContinuousRecognizer
  -> vision prefill
  -> text prefill and private KV construction
  -> persistent continuous-decode arena
  -> page assembly
  -> incremental artifacts and timeline
```

`run_offline_e2e.py` is an older alternative/diagnostic page pipeline. It does
not match the production frontend, scheduling, packing, artifact, or timing
path. Do not use it anywhere in this ladder and do not cite its success as
Experiment 09 validation.

## Scope and stopping point

- The largest OCR run is the first eight OmniDocBench pages.
- The only 32-page run is the isolated layout lab, which does not load the OCR
  recognizer.
- Do not run 256 OCR pages.
- Do not run all 1,651 OCR pages.
- Do not run OmniDocBench accuracy evaluation.
- Do not compile the complete default vision/text bucket ladders.
- Do not use the 910B2 profile-guided vision-routing table on 310P.
- Do not edit source code, install packages, or invent fallback implementations.

Stop after the layout check and write the required report.

## What the previous 310P server established

These results were manually relayed from the old server:

- the Experiment 09 model and native NPU operations could run;
- the NPU-side MRoPE `IndexPut` failure was avoided by the production CPU-MRoPE
  path;
- PromptFlashAttention worked after physical vision lengths were aligned to
  128;
- TorchAir B4 decode replay worked;
- a prior eight-page diagnostic workload measured roughly:
  - manual eager B1: 100 s wall, 54 s decode, 48 raw decode tokens/s;
  - eager B4: 64 s wall, 16 s decode, 164 raw decode tokens/s;
  - TorchAir B4: 51 s wall, 4 s decode, 650 raw decode tokens/s;
  - aligned PromptFA plus TorchAir B4: about 40 s wall;
- the isolated layout lab measured about 4.8 pages/s with one worker and
  7.54 pages/s with two workers.

Those are context, not baselines. The old raw artifacts and graph caches were
not committed, and the old measurements did not all use the standard production
runner. Recreate evidence locally on the replacement server.

## What is being tested

Keep these mechanisms distinct:

| Mechanism | Production control or behavior | Evidence |
|---|---|---|
| NPU runtime | `torch_npu`, CANN, native operators | real `npu:0` tensor operation and production run |
| TorchAir compiler | resolved by the repository helper, not `torch.backends` | `cache_compile` preflight, cache creation, fresh-process replay |
| Layout | owned PaddleX-free frontend | summary, request manifest, no loaded PaddleX modules |
| Vision attention | manual versus 128-aligned PromptFA | same-page output comparison and vision device time |
| Vision execution | eager smoke, then TorchAir | trace execution counts, cache replay, useful/physical tokens/s |
| Text prefill | production TorchAir path | trace execution counts, cache replay, useful/physical tokens/s |
| Decode | production TorchAir persistent arena | B4/B16/B32 cache, raw/effective rate, active-slot fraction |
| Vision packing | off, then arrival-order greedy at 1,920 | groups, calls, fill fraction, E2E |
| Text packing | off, then `production_group` after vision packing | groups, calls, KV redistribution, E2E |
| Recognition resolution | default, then `min_pixels=28,224` | real/physical token reduction and E2E |
| CPU preparation | built-in background worker | configuration and timeline overlap |
| Recognition H2D | dedicated transfer stream and events | timeline I/O spans |
| Page scheduling | bounded frontend and incremental page completion | completion order, artifacts, timeline |
| Layout worker mode | isolated lab only: W1 versus W2 | manifest comparison and pages/s |

The production runner always uses TorchAir for text prefill and decode. It has
no `--decode-backend` or `--text-backend` switch. Vision alone exposes
`--vision-backend`.

## Operating rules

- The work-server checkout is pull-only. Do not edit tracked files, commit,
  push, or create branches.
- Start with `git pull --ff-only origin main`.
- Do not install or replace PyTorch, torch-npu, TorchAir, CANN, model files, or
  system packages.
- Use one free physical 310P. Do not terminate another user's process.
- Use discovered absolute paths. Do not assume Blue Zone `/workspace` paths.
- Create every graph cache on this exact 310P software stack. Never copy a 910B
  or old-server cache.
- Preserve the expanded command, complete log, exit code, Git commit,
  environment fingerprint, cache inventory, and output artifacts for every
  lane.
- Put small evidence below:

  ```text
  tmp/09_persistent_page_engine/310p_exp09_ladder/
  ```

- Put compiler caches below `.runtime_cache/`, with `310p` and the graph shape
  in their names.
- Stop at the first failed dependency chain and preserve the first causal
  traceback. Do not change unrelated flags until the failed stage is known.

## Phase 0: environment and exact local paths

Pull and establish variables:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_exp09_ladder"
mkdir -p "$OUTPUT_ROOT" "$REPO/.runtime_cache"

PYTHON_BIN="<absolute compatible Python>"
RECOGNIZER_MODEL="<absolute PaddleOCR-VL-1.6 directory>"
LAYOUT_MODEL="<absolute PP-DocLayoutV3_safetensors directory>"
DATASET_JSON="<absolute OmniDocBench.json>"
IMAGES_DIR="<absolute OmniDocBench images directory>"
```

Do not run later commands with angle-bracket placeholders. If paths are
unknown, search existing server roots:

```sh
for root in "$HOME" /workspace /data /data1 /data2; do
  test -d "$root" || continue
  find "$root" -maxdepth 6 \
    \( -name 'PaddleOCR-VL-1.6' \
       -o -name 'PP-DocLayoutV3_safetensors' \
       -o -name 'OmniDocBench.json' \) \
    -print 2>/dev/null
done
```

Select the existing complete copies used by the earlier eager validation.
Validate:

```sh
test -x "$PYTHON_BIN"
test -d "$RECOGNIZER_MODEL"
test -f "$RECOGNIZER_MODEL/config.json"
test -d "$LAYOUT_MODEL"
test -f "$LAYOUT_MODEL/config.json"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"

"$PYTHON_BIN" - <<'PY'
import cv2
import numpy
import safetensors
import torch
import torch_npu
import transformers
print("core_python_imports: PASS")
PY
```

Verify the annotation count and decode only the first eight selected page files:

```sh
export DATASET_JSON IMAGES_DIR
"$PYTHON_BIN" - <<'PY' \
  >"$OUTPUT_ROOT/phase0_dataset_first8.txt" 2>&1
import json
import os
from pathlib import Path

import cv2
import numpy as np

dataset_json = Path(os.environ["DATASET_JSON"]).expanduser().resolve()
images_dir = Path(os.environ["IMAGES_DIR"]).expanduser().resolve()
annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
assert len(annotations) == 1651, len(annotations)

for index, annotation in enumerate(annotations[:8]):
    basename = Path(annotation["page_info"]["image_path"]).name
    image_path = images_dir / basename
    assert image_path.is_file(), image_path
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert image is not None, image_path
    print(index, image_path, tuple(int(value) for value in image.shape))

print("OMNIDOCBENCH_FIRST8: PASS")
PY

tail -n 1 "$OUTPUT_ROOT/phase0_dataset_first8.txt"
```

Activate the server's intended NPU environment before continuing. A working
`npu-smi` alone is insufficient; the selected Python must use NPU successfully.

Record the environment:

```sh
{
  printf 'git_commit='
  git rev-parse HEAD
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname='
  hostname
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import os
import sys
import torch
import torch_npu

print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_file:", torch.__file__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torch_npu_file:", torch_npu.__file__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
print("npu_0_name:", torch.npu.get_device_name(0))
print("ASCEND_HOME_PATH:", os.environ.get("ASCEND_HOME_PATH"))
print("ASCEND_OPP_PATH:", os.environ.get("ASCEND_OPP_PATH"))

assert torch.npu.is_available()
x = torch.arange(16, dtype=torch.float32, device="npu:0")
y = (x * 2).cpu()
torch.npu.synchronize()
assert y.tolist() == [float(index * 2) for index in range(16)]
print("basic_npu_tensor: PASS")
PY
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'layout_model=%s\n' "$LAYOUT_MODEL"
  printf 'dataset_json=%s\n' "$DATASET_JSON"
  printf 'images_dir=%s\n' "$IMAGES_DIR"
  df -h "$REPO" "$REPO/.runtime_cache"
  command -v npu-smi >/dev/null && npu-smi info
} >"$OUTPUT_ROOT/phase0_environment.txt" 2>&1
```

### Command and cache recording helpers

Use this function for every model or lab command:

```sh
run_and_record() {
  local run_name="$1"
  shift
  local evidence_dir="$OUTPUT_ROOT/$run_name"
  mkdir -p "$evidence_dir"
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$evidence_dir/exit_code.txt"
  printf 'run=%s exit_code=%s log=%s\n' \
    "$run_name" "$status" "$evidence_dir/run.log"
  return "$status"
}

record_cache_inventory() {
  local cache_root="$1"
  local output_file="$2"
  {
    printf 'cache_root=%s\n' "$cache_root"
    if test -d "$cache_root"; then
      du -sh "$cache_root"
      printf 'files='
      find "$cache_root" -type f | wc -l
      find "$cache_root" -maxdepth 2 -mindepth 1 -type d -print | sort
    else
      printf 'state=absent\n'
    fi
  } >"$output_file"
}
```

### Production-output validators

Use this validator for every same-configuration first/replay pair:

```sh
validate_production_pair() {
  local first_root="$1"
  local replay_root="$2"
  local expected_pages="$3"
  local expected_batch="$4"
  local expected_vision_packing="$5"
  local expected_text_packing="$6"
  local expected_min_pixels="$7"

  "$PYTHON_BIN" - \
    "$first_root" \
    "$replay_root" \
    "$expected_pages" \
    "$expected_batch" \
    "$expected_vision_packing" \
    "$expected_text_packing" \
    "$expected_min_pixels" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

first_root = Path(sys.argv[1])
replay_root = Path(sys.argv[2])
expected_pages = int(sys.argv[3])
expected_batch = int(sys.argv[4])
expected_vision_packing = sys.argv[5]
expected_text_packing = sys.argv[6]
expected_min_pixels = None if sys.argv[7] == "none" else int(sys.argv[7])

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_trace(root):
    return [
        json.loads(line)
        for line in (root / "recognition_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

def projection(records):
    return [
        (
            row["request_id"],
            row["token_ids"],
            row["text"],
            row["stop_reason"],
        )
        for row in records
    ]

roots = [first_root, replay_root]
summaries = [read_json(root / "run_summary.json") for root in roots]
traces = [read_trace(root) for root in roots]

for root, summary, trace in zip(roots, summaries, traces, strict=True):
    configuration = summary["configuration"]
    assert summary["result_count"] == expected_pages
    assert summary["prediction_count"] == expected_pages
    assert summary["paddlex_runtime_dependency"] is False
    assert summary["loaded_paddlex_modules"] == []
    assert configuration["batch_size"] == expected_batch
    assert configuration["vision_packing"] == expected_vision_packing
    assert configuration["text_packing"] == expected_text_packing
    assert configuration["preprocessor_min_pixels"] == expected_min_pixels
    assert summary["timeline"]["enabled"] is True
    assert (root / "timeline_trace.json").is_file()
    assert (root / "timeline.html").is_file()
    assert (root / "page_regions.jsonl").is_file()
    assert len(trace) > 0

first_projection = projection(traces[0])
replay_projection = projection(traces[1])
assert first_projection == replay_projection
payload = json.dumps(
    replay_projection,
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
print("requests:", len(replay_projection))
print("output_projection_sha256:", hashlib.sha256(payload).hexdigest())
print("PRODUCTION_PAIR: PASS")
PY
}
```

Use this comparison only when attention/backend settings intentionally differ:

```sh
compare_production_outputs() {
  local first_root="$1"
  local second_root="$2"

  "$PYTHON_BIN" - "$first_root" "$second_root" <<'PY'
import json
import sys
from pathlib import Path

roots = [Path(sys.argv[1]), Path(sys.argv[2])]

def projection(root):
    records = [
        json.loads(line)
        for line in (root / "recognition_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return [
        (
            row["request_id"],
            row["token_ids"],
            row["text"],
            row["stop_reason"],
        )
        for row in records
    ]

first = projection(roots[0])
second = projection(roots[1])
print("first_requests:", len(first))
print("second_requests:", len(second))
print("exact:", first == second)
if first != second:
    differences = [
        (index, left, right)
        for index, (left, right) in enumerate(zip(first, second, strict=True))
        if left != right
    ]
    print("different_requests:", len(differences))
    print("first_difference:", differences[0])
else:
    print("PRODUCTION_OUTPUT_COMPARISON: EXACT")
PY
}
```

### TorchAir: the correct availability check

Do not inspect `torch.backends`. TorchAir is a separate compiler package, or an
embedded module supplied through `torch_npu.dynamo`. Experiment 09 calls:

```text
paddleocr_vl.model.compile_utils.import_torchair()
  -> torchair.inference.cache_compile
```

The helper first tries direct `import torchair`, then falls back to
`from torch_npu.dynamo import torchair`.

Run the exact repository resolver:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" - \
  >"$OUTPUT_ROOT/phase0_torchair_preflight.txt" 2>&1 <<'PY'
import importlib.metadata
import sys

import torch
import torch_npu

from paddleocr_vl.model.compile_utils import import_torchair

print("python_executable:", sys.executable)
print("torch_version:", torch.__version__)
print("torch_npu_version:", getattr(torch_npu, "__version__", "<missing>"))
print("npu_0_name:", torch.npu.get_device_name(0))

torchair, CompilerConfig = import_torchair()
print("torchair_module_name:", torchair.__name__)
print("torchair_file:", getattr(torchair, "__file__", "<namespace>"))
try:
    print("torchair_distribution:", importlib.metadata.version("torchair"))
except importlib.metadata.PackageNotFoundError:
    print("torchair_distribution: <embedded module>")

configuration = CompilerConfig()
print("CompilerConfig:", type(configuration))
print(
    "cache_compile_callable:",
    callable(getattr(torchair.inference, "cache_compile", None)),
)
print(
    "get_npu_backend_callable:",
    callable(getattr(torchair, "get_npu_backend", None)),
)
assert callable(getattr(torchair.inference, "cache_compile", None))
print("TORCHAIR_EXP09_PREFLIGHT: PASS")
PY

tail -n 1 "$OUTPUT_ROOT/phase0_torchair_preflight.txt"
```

Proceed only if the last line is:

```text
TORCHAIR_EXP09_PREFLIGHT: PASS
```

If it fails, preserve both direct-import and embedded-fallback exceptions. Run:

```sh
"$PYTHON_BIN" -m pip show torch torch-npu torchair \
  >"$OUTPUT_ROOT/phase0_python_packages.txt" 2>&1 || true
"$PYTHON_BIN" - <<'PY' \
  >"$OUTPUT_ROOT/phase0_python_paths.txt" 2>&1
import site
import sys
print("executable:", sys.executable)
print("sys.path:")
print("\n".join(sys.path))
print("site-packages:")
print("\n".join(site.getsitepackages()))
PY
```

Do not install anything. Report whether the direct route, embedded route, or
both failed.

## Shared production configuration

Use the first eight dataset pages for every OCR performance comparison:

```text
--offset 0 --limit 8
```

The one-page correctness comparison uses the first page:

```text
--offset 0 --limit 1
```

Use these compact singleton buckets:

```text
vision: 640,768,1408,2944,4992
text:   176,208,384,768,1280,1312
```

They cover the expected small test while avoiding the complete default compile
ladder. Any sequence above the largest bucket uses the explicit eager-overflow
route and must be reported from `recognition_trace.jsonl`.

Create fresh local caches:

```sh
COMMIT_SHORT="$(git rev-parse --short HEAD)"
DECODE_B4_CACHE="$REPO/.runtime_cache/310p_decode_b4_k4096_$COMMIT_SHORT"
TEXT_CACHE="$REPO/.runtime_cache/310p_text_prefill_compact_$COMMIT_SHORT"
VISION_PFA_CACHE="$REPO/.runtime_cache/310p_vision_pfa_align128_compact_$COMMIT_SHORT"

test ! -e "$DECODE_B4_CACHE"
test ! -e "$TEXT_CACHE"
test ! -e "$VISION_PFA_CACHE"
```

Define the production command shared by all lanes:

```sh
PRODUCTION_BASE=(
  "$PYTHON_BIN"
  "$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL"
  --layout-device cpu
  --recognizer-model "$RECOGNIZER_MODEL"
  --dtype fp16
  --cache-length 4096
  --text-padding bucket
  --text-torchair-cache-dir "$TEXT_CACHE"
  --timeline
)

DEFAULT_TEXT_BUCKETS=(
  --text-buckets 176,208,384,768,1280,1312
)

PACKING_OFF=(
  --vision-packing off
  --text-packing off
)
```

The production runner hardcodes TorchAir for decode and text prefill. The
decode cache is selected by `--torchair-cache-dir`; text uses
`--text-torchair-cache-dir`.

## Phase 1: real production smoke and graph replay

This is the first model run. It uses the actual owned layout/frontend,
`ContinuousRecognizer`, continuous decode, page writer, and timeline. Vision is
manual eager only to avoid compiling vision before the complete system has
passed once.

Use one page and an eight-token cap:

```sh
record_cache_inventory \
  "$DECODE_B4_CACHE" \
  "$OUTPUT_ROOT/phase1_decode_cache_before.txt"
record_cache_inventory \
  "$TEXT_CACHE" \
  "$OUTPUT_ROOT/phase1_text_cache_before.txt"

run_and_record phase1_production_smoke_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 1 \
  --batch-size 4 \
  --max-new-tokens 8 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --output-dir "$OUTPUT_ROOT/phase1_production_smoke_first/output"

record_cache_inventory \
  "$DECODE_B4_CACHE" \
  "$OUTPUT_ROOT/phase1_decode_cache_after_first.txt"
record_cache_inventory \
  "$TEXT_CACHE" \
  "$OUTPUT_ROOT/phase1_text_cache_after_first.txt"

run_and_record phase1_production_smoke_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 1 \
  --batch-size 4 \
  --max-new-tokens 8 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --output-dir "$OUTPUT_ROOT/phase1_production_smoke_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase1_production_smoke_first/output" \
  "$OUTPUT_ROOT/phase1_production_smoke_replay/output" \
  1 \
  4 \
  off \
  off \
  none \
  >"$OUTPUT_ROOT/phase1_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase1_pair_validation.txt"
```

Require:

- `PRODUCTION_PAIR: PASS`;
- one result and one Markdown prediction;
- no PaddleX import;
- a non-empty recognition trace and timeline;
- decode and text configuration report TorchAir;
- vision configuration reports raw eager/manual;
- first process creates the expected B4/K4096 decode and compact text caches;
- replay creates no new graph shapes;
- no NPU `IndexPut` failure.

Length-cap stop reasons are expected because this is only an eight-token smoke.
Do not use its wall time as a throughput result.

The standard production preparation computes MRoPE `position_ids` and
`rope_deltas` on CPU before one transfer to NPU. Therefore it should not invoke
the failing NPU boolean `index_put_` used by the old single-crop example. If an
`IndexPutV2` traceback appears here, first confirm the stack trace really comes
from `run_omnidocbench.py` and `ContinuousRecognizer._prepare_cpu`; do not
switch to another runner or patch the work-server checkout.

## Phase 2: real production PromptFA correctness check

Run the same first page to natural EOS twice. Decode and text reuse the same
production caches; only vision attention changes.

Manual-vision reference:

```sh
run_and_record phase2_manual_vision_full_page \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 1 \
  --batch-size 4 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --output-dir "$OUTPUT_ROOT/phase2_manual_vision_full_page/output"
```

Aligned PromptFA:

```sh
run_and_record phase2_promptfa_vision_full_page \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 1 \
  --batch-size 4 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --vision-backend raw_eager \
  --vision-attention prompt_flash_attention \
  --vision-promptfa-align-128 \
  --vision-padding none \
  --output-dir "$OUTPUT_ROOT/phase2_promptfa_vision_full_page/output"

compare_production_outputs \
  "$OUTPUT_ROOT/phase2_manual_vision_full_page/output" \
  "$OUTPUT_ROOT/phase2_promptfa_vision_full_page/output" \
  >"$OUTPUT_ROOT/phase2_attention_comparison.txt" 2>&1

cat "$OUTPUT_ROOT/phase2_attention_comparison.txt"
```

The alignment flag is mandatory on 310P. It pads every physical PromptFA
vision sequence to a multiple of 128 and avoids:

```text
attention mask must be NULL, when Qs, Kvs is unAlign ...
```

Require both runs to emit one complete page with the same request count and no
missing crops. Exact token/text parity is expected from the old-server result.
If outputs differ, preserve the comparison and report the first differing
request; do not conceal the difference. Also report real and physical vision
tokens plus vision device time for manual versus PromptFA.

If PromptFA fails, preserve the exact native-op error and stop the PromptFA
dependency chain. Do not remove the alignment flag or guess operator arguments.

## Phase 3: compiled PromptFA and real eight-page baseline

Now compile the production vision stage with aligned PromptFA and run the first
eight pages to natural EOS. Text and decode use the already-created caches.

```sh
COMPILED_PFA_VISION=(
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-promptfa-align-128
  --vision-padding bucket
  --vision-buckets 640,768,1408,2944,4992
  --vision-torchair-cache-dir "$VISION_PFA_CACHE"
)

record_cache_inventory \
  "$VISION_PFA_CACHE" \
  "$OUTPUT_ROOT/phase3_vision_cache_before.txt"

run_and_record phase3_production_b4_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 4 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase3_production_b4_first/output"

record_cache_inventory \
  "$VISION_PFA_CACHE" \
  "$OUTPUT_ROOT/phase3_vision_cache_after_first.txt"

run_and_record phase3_production_b4_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 4 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B4_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase3_production_b4_replay/output"

record_cache_inventory \
  "$VISION_PFA_CACHE" \
  "$OUTPUT_ROOT/phase3_vision_cache_after_replay.txt"

validate_production_pair \
  "$OUTPUT_ROOT/phase3_production_b4_first/output" \
  "$OUTPUT_ROOT/phase3_production_b4_replay/output" \
  8 \
  4 \
  off \
  off \
  none \
  >"$OUTPUT_ROOT/phase3_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase3_pair_validation.txt"
```

Require:

- eight page results and eight Markdown predictions;
- exact first/replay ordered output parity;
- zero missing page/crop results;
- no unexpected graph creation during replay;
- no PaddleX modules loaded;
- timeline JSON/HTML, recognition trace, page-region manifest, route plan, and
  run summary;
- page results written incrementally;
- decode queue drains and private prefill-cache active slots return to zero.

Inspect `recognition_trace.jsonl` and report:

- vision execution counts and any eager-overflow lengths;
- text execution counts and any eager-overflow lengths;
- real and physical vision/text tokens;
- vision/text useful fractions and device tokens/s;
- decode graph calls, raw/effective decode tokens/s, active-slot fraction,
  admissions, and stop reasons.

Inspect the timeline and report whether these built-in production mechanisms
were actually exercised:

- background CPU recognition preparation;
- dedicated recognition-input H2D stream/event dependencies;
- prefill production and continuous decode;
- cross-page admissions;
- bounded page production and incremental page artifact writing.

This replay is the required 310P production baseline.

## Phase 4: larger continuous-decode arenas

Compare against the Phase 3 replay. Change only decode batch size/cache.

Approximate fp16 KV-only allocations at KV4096:

```text
B4:   1.69 GiB
B16:  6.75 GiB
B32: 13.50 GiB
```

These exclude weights, vision workspace, and compiler transients.

### B16

```sh
DECODE_B16_CACHE="$REPO/.runtime_cache/310p_decode_b16_k4096_$COMMIT_SHORT"
test ! -e "$DECODE_B16_CACHE"

record_cache_inventory \
  "$DECODE_B16_CACHE" \
  "$OUTPUT_ROOT/phase4_decode_b16_before.txt"

run_and_record phase4_production_b16_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 16 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B16_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase4_production_b16_first/output"

record_cache_inventory \
  "$DECODE_B16_CACHE" \
  "$OUTPUT_ROOT/phase4_decode_b16_after_first.txt"

run_and_record phase4_production_b16_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 16 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B16_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase4_production_b16_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase4_production_b16_first/output" \
  "$OUTPUT_ROOT/phase4_production_b16_replay/output" \
  8 \
  16 \
  off \
  off \
  none \
  >"$OUTPUT_ROOT/phase4_b16_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase4_b16_pair_validation.txt"
```

### Conditional B32

Attempt B32 only if:

1. B16 passes first/replay parity;
2. B16 replay improves `pipeline_e2e_s` over B4 replay;
3. peak HBM and free HBM leave room for another roughly 6.75 GiB plus compile
   transients.

Write the measured decision to
`$OUTPUT_ROOT/phase4_b32_decision.txt`. If it passes, run:

```sh
DECODE_B32_CACHE="$REPO/.runtime_cache/310p_decode_b32_k4096_$COMMIT_SHORT"
test ! -e "$DECODE_B32_CACHE"

run_and_record phase4_production_b32_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 32 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B32_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase4_production_b32_first/output"

record_cache_inventory \
  "$DECODE_B32_CACHE" \
  "$OUTPUT_ROOT/phase4_decode_b32_after_first.txt"

run_and_record phase4_production_b32_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${COMPILED_PFA_VISION[@]}" \
  "${PACKING_OFF[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size 32 \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$DECODE_B32_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase4_production_b32_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase4_production_b32_first/output" \
  "$OUTPUT_ROOT/phase4_production_b32_replay/output" \
  8 \
  32 \
  off \
  off \
  none \
  >"$OUTPUT_ROOT/phase4_b32_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase4_b32_pair_validation.txt"
```

Select the stable replay with the lowest measured `pipeline_e2e_s`. Preserve
all candidate results. Generate the exact variables mechanically:

```sh
SELECTION_ARGS=(
  4
  "$DECODE_B4_CACHE"
  "$OUTPUT_ROOT/phase3_production_b4_replay/output/run_summary.json"
  16
  "$DECODE_B16_CACHE"
  "$OUTPUT_ROOT/phase4_production_b16_replay/output/run_summary.json"
)
if test -f \
  "$OUTPUT_ROOT/phase4_production_b32_replay/output/run_summary.json"; then
  SELECTION_ARGS+=(
    32
    "$DECODE_B32_CACHE"
    "$OUTPUT_ROOT/phase4_production_b32_replay/output/run_summary.json"
  )
fi

"$PYTHON_BIN" - "${SELECTION_ARGS[@]}" \
  >"$OUTPUT_ROOT/phase4_selected_decode.env" <<'PY'
import json
import shlex
import sys
from pathlib import Path

arguments = sys.argv[1:]
assert len(arguments) % 3 == 0
candidates = []
for index in range(0, len(arguments), 3):
    batch_size = int(arguments[index])
    cache = Path(arguments[index + 1]).resolve()
    summary_path = Path(arguments[index + 2]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["configuration"]["batch_size"] == batch_size
    assert summary["result_count"] == 8
    assert summary["prediction_count"] == 8
    candidates.append(
        (
            float(summary["pipeline_e2e_s"]),
            batch_size,
            cache,
            summary_path,
        )
    )

candidates.sort()
for wall, batch_size, cache, summary_path in candidates:
    print(
        f"# candidate batch={batch_size} "
        f"pipeline_e2e_s={wall:.9f} summary={summary_path}"
    )
_, selected_batch, selected_cache, _ = candidates[0]
print(f"SELECTED_BATCH_SIZE={shlex.quote(str(selected_batch))}")
print(f"SELECTED_DECODE_CACHE={shlex.quote(str(selected_cache))}")
PY

. "$OUTPUT_ROOT/phase4_selected_decode.env"

case "$SELECTED_BATCH_SIZE" in
  4|16|32) ;;
  *) printf 'invalid selected batch size: %s\n' "$SELECTED_BATCH_SIZE"; exit 1 ;;
esac
test -d "$SELECTED_DECODE_CACHE"
cat "$OUTPUT_ROOT/phase4_selected_decode.env"
```

Do not assume the largest batch wins. Report raw/effective decode rate,
active-slot fraction, HBM, decode wall, E2E, and pages/s for every attempted
size.

## Phase 5: greedy compiled vision packing

Keep the selected decode size and default `min_pixels`. Add one 1,920-token
singleton vision graph and enable arrival-order greedy packing:

```sh
GREEDY_PFA_VISION=(
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-promptfa-align-128
  --vision-padding bucket
  --vision-buckets 640,768,1408,1920,2944,4992
  --vision-torchair-cache-dir "$VISION_PFA_CACHE"
  --vision-packing greedy
  --vision-pack-target 1920
)

run_and_record phase5_greedy_vision_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${GREEDY_PFA_VISION[@]}" \
  --text-packing off \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase5_greedy_vision_first/output"

record_cache_inventory \
  "$VISION_PFA_CACHE" \
  "$OUTPUT_ROOT/phase5_vision_cache_after_first.txt"

run_and_record phase5_greedy_vision_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${GREEDY_PFA_VISION[@]}" \
  --text-packing off \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase5_greedy_vision_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase5_greedy_vision_first/output" \
  "$OUTPUT_ROOT/phase5_greedy_vision_replay/output" \
  8 \
  "$SELECTED_BATCH_SIZE" \
  greedy \
  off \
  none \
  >"$OUTPUT_ROOT/phase5_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase5_pair_validation.txt"
```

Require structural success and first/replay parity. Token differences versus
packing-off are permitted but must be reported. Report groups, crops/group,
vision-transformer calls, real/physical tokens, fill fraction, vision device
time, E2E, and pages/s.

## Phase 6: packed text prefill

Text packing is meaningful only after vision groups exist. Create its separate
310P graph cache:

```sh
PACKED_TEXT_CACHE="$REPO/.runtime_cache/310p_text_packed_$COMMIT_SHORT"
test ! -e "$PACKED_TEXT_CACHE"

PACKED_TEXT=(
  --text-packing production_group
  --text-pack-buckets 128,256,512,1024
  --text-pack-max-members 32
  --text-packed-cache-dir "$PACKED_TEXT_CACHE"
)

run_and_record phase6_packed_text_first \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${GREEDY_PFA_VISION[@]}" \
  "${PACKED_TEXT[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase6_packed_text_first/output"

record_cache_inventory \
  "$PACKED_TEXT_CACHE" \
  "$OUTPUT_ROOT/phase6_packed_text_cache_after_first.txt"

run_and_record phase6_packed_text_replay \
  "${PRODUCTION_BASE[@]}" \
  "${DEFAULT_TEXT_BUCKETS[@]}" \
  "${GREEDY_PFA_VISION[@]}" \
  "${PACKED_TEXT[@]}" \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase6_packed_text_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase6_packed_text_first/output" \
  "$OUTPUT_ROOT/phase6_packed_text_replay/output" \
  8 \
  "$SELECTED_BATCH_SIZE" \
  greedy \
  production_group \
  none \
  >"$OUTPUT_ROOT/phase6_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase6_pair_validation.txt"
```

Report text groups, members/group, transformer calls, real/physical text
tokens, useful fraction/rate, KV redistribution time, E2E, and pages/s.
Compare against the Phase 5 replay.

## Phase 7: reduced-min-pixels cumulative candidate

Run only the established quarter-resolution setting:

```text
112,896 / 4 = 28,224
```

Add small singleton buckets so low-resolution crops are not padded directly to
640 vision or 176 text tokens:

```sh
REDUCED_PFA_VISION=(
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-promptfa-align-128
  --vision-padding bucket
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992
  --vision-torchair-cache-dir "$VISION_PFA_CACHE"
  --vision-packing greedy
  --vision-pack-target 1920
)

run_and_record phase7_min_pixels_28224_first \
  "${PRODUCTION_BASE[@]}" \
  "${REDUCED_PFA_VISION[@]}" \
  "${PACKED_TEXT[@]}" \
  --text-buckets 32,64,96,128,176,208,384,768,1280,1312 \
  --preprocessor-min-pixels 28224 \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase7_min_pixels_28224_first/output"

record_cache_inventory \
  "$VISION_PFA_CACHE" \
  "$OUTPUT_ROOT/phase7_vision_cache_after_first.txt"
record_cache_inventory \
  "$TEXT_CACHE" \
  "$OUTPUT_ROOT/phase7_text_cache_after_first.txt"

run_and_record phase7_min_pixels_28224_replay \
  "${PRODUCTION_BASE[@]}" \
  "${REDUCED_PFA_VISION[@]}" \
  "${PACKED_TEXT[@]}" \
  --text-buckets 32,64,96,128,176,208,384,768,1280,1312 \
  --preprocessor-min-pixels 28224 \
  --offset 0 \
  --limit 8 \
  --batch-size "$SELECTED_BATCH_SIZE" \
  --max-new-tokens 2808 \
  --torchair-cache-dir "$SELECTED_DECODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/phase7_min_pixels_28224_replay/output"

validate_production_pair \
  "$OUTPUT_ROOT/phase7_min_pixels_28224_first/output" \
  "$OUTPUT_ROOT/phase7_min_pixels_28224_replay/output" \
  8 \
  "$SELECTED_BATCH_SIZE" \
  greedy \
  production_group \
  28224 \
  >"$OUTPUT_ROOT/phase7_pair_validation.txt" 2>&1

tail -n 1 "$OUTPUT_ROOT/phase7_pair_validation.txt"
```

Output changes relative to default `min_pixels` are expected. Require
structural success and exact same-configuration first/replay parity. Report
real/physical vision and text token reductions, useful fractions/rates, every
major stage, E2E, and pages/s versus Phase 6.

Do not evaluate accuracy in this ladder.

## Profile-guided batched vision status

Do not enable production `--vision-packing profile_guided`. Its hard-coded
route timings are calibrated for Ascend 910B2, not 310P, and its production
policy expects B2x3072 and B4x1024 batched graph caches.

Record:

```text
batched graph mechanism: present in the repository
310P graph execution: not tested by this ladder
310P route timings: not measured
production profile-guided policy on 310P: unsupported until calibrated
```

Do not edit the profile on the work server.

## Phase 8: isolated owned-layout method check

This is the only 32-page task. It does not load the OCR recognizer.

One worker:

```sh
run_and_record phase8_layout_w1 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device cpu \
  --offset 0 \
  --limit 32 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/phase8_layout_w1/output"
```

Two workers:

```sh
run_and_record phase8_layout_w2 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device cpu \
  --offset 0 \
  --limit 32 \
  --workers 2 \
  --no-timeline \
  --reference-requests "$OUTPUT_ROOT/phase8_layout_w1/output/requests.jsonl" \
  --output-dir "$OUTPUT_ROOT/phase8_layout_w2/output"
```

W2 overlaps the next page's CPU image decode. It does not batch two pages
through one detector call and is not a production-runner `--workers` switch.

The W2 command exits 2 when request-manifest bytes differ. If so, preserve its
output and compare records field-by-field. Report pages/s, stage totals,
request counts, manifest hash/equality, and any tolerated one-pixel resize
difference.

## Artifact interpretation

For every production lane:

- `run_summary.json` contains configuration, setup, E2E, page, layout,
  recognition, packing, token, memory/accounting, and artifact paths;
- `recognition_trace.jsonl` contains one record per crop, including token
  IDs/text/stop reason, crop size, real/physical vision and text tokens, route,
  and per-stage timing;
- `page_regions.jsonl` is the compact page/region manifest;
- `timeline_trace.json` and `timeline.html` contain synchronization-neutral
  host/device spans and waits;
- `predictions/` contains one Markdown result per page;
- `vision_route_plan.json` records the route selected in that run.

Separate setup/compile/load time from `pipeline_e2e_s`. Use the replay process
for steady-state comparisons. A cache directory merely existing is not replay
proof; compare inventories and logs and confirm the replay did not create a
new graph shape.

## Stop condition and required report

Stop after Phase 8. Do not start a larger OCR workload.

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Use this exact report skeleton:

```text
310P EXP09 PRODUCTION LADDER: PASS | PARTIAL | FAIL

Git commit:
Host / exact NPU:
Python:
torch:
torch_npu:
TorchAir resolver route / module:
CANN / driver / firmware:
Model and dataset paths:

Phase 0 environment and TorchAir preflight:

Phase 1 real production smoke:
first / replay:
page / crop / prediction counts:
decode and text cache replay:
IndexPut status:

Phase 2 real production attention check:
manual vision:
aligned PromptFA:
output comparison:
real / physical vision tokens:
vision device time:

Phase 3 compiled production B4:
first / replay wall:
pages/s:
page / request / prediction counts:
vision/text routes and overflows:
vision useful / physical tokens/s:
text useful / physical tokens/s:
decode raw / effective tokens/s:
active-slot fraction:
peak HBM:
output parity:
timeline / trace / cache evidence:

Phase 4 decode arenas:
B4:
B16:
B32 or exact reason skipped:
selected batch and cache:

Phase 5 greedy vision packing:
groups / calls / fill:
vision tokens and device time:
wall / pages-s:
output parity:

Phase 6 packed text:
groups / calls:
KV redistribution:
text tokens and device time:
wall / pages-s:
output parity:

Phase 7 min_pixels 28224:
vision/text token reduction:
stage times:
wall / pages-s:
output parity:

Profile-guided status:

Phase 8 layout:
W1 wall / pages-s:
W2 wall / pages-s:
manifest comparison:

Best stable production configuration:
Best replay wall / pages-s:
Major stage breakdown:
Vision useful / physical tokens/s:
Text useful / physical tokens/s:
Decode raw / effective tokens/s:
Peak HBM:

First blocker or warning:
Exact command records:
Artifact paths:
```

Do not describe eight-page speed as full-corpus throughput. Do not report an
alternative runner as production validation.
