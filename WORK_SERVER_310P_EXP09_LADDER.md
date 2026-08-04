# Work-server 310P Experiment 09 validation ladder

This is the execution brief for the AI agent on Luka's replacement Atlas 310P
server. Read `CLAUDE.md` and `AGENTS.md` first.

## Goal

Prove that the real Experiment 09 pipeline runs on this 310P software stack,
identify which production optimizations work, and measure their approximate
effect on a small representative workload.

## Current requested task: run Phase 27 only

Phases 0-21 have already run, are in progress elsewhere, are superseded, or
retain historical instructions and evidence. Do not rerun the performance
ladder, vision matrices, native profiles, text prefill, decode, PromptFA
experiments, or the superseded standalone-layout Phase 19. Phase 21 proved the
page-index-8 failure occurs inside packed text graph call 7, before KV
redistribution. Phase 22 reconstructed the exact indices and proved both eager
Gather lanes, but its TorchAir lanes never executed because the old probe
passed a free function to `cache_compile`. Phase 23 did not reproduce the
failure in the isolated compiled Gather. Phase 24 moved the final gather out
of the graph, but the normal asynchronous run still failed with GatherV2;
Phase 25 proved the compiled graph completes and the external eager GatherV2
fails. The final-token selection now uses one-token slices plus concatenation,
with no GatherV2. Phase 26 proved that exact page 9 passes. Go directly to
**Phase 27: warm-cache 8-page reproduction and 32-page extension** at the end
of this document.

The reported production boundary is unusually sharp:

- `--offset 0 --limit 8` succeeds;
- `--offset 0 --limit 9` fails;
- `--offset 8 --limit 1` also fails.

This makes page index 8 (the ninth dataset page) the primary reproducer, but
the standalone layout frontend succeeds on that page. The current question is:

> What exact integrated layout-plus-recognition stage, operator, graph route,
> kernel/binary lookup, file, and software component fails, and why does that
> path differ from both a passing page and standalone page-9 layout?

This is an investigation phase, not a workaround phase. Do not modify
production model code, change operator expressions, install packages, delete
operator/compiler caches, retry downloads, or fall back to CPU as a proposed
solution. Preserve the first causal error and all relevant CANN logs.

## Current 310P layout route: eager NPU

The goal is NPU layout inference, not ACLGraph compatibility. Use:

```text
--layout-device npu
--no-layout-graph-capture
```

This keeps PP-DocLayoutV3 and the complete PaddleOCR-VL recognizer on logical
`npu:0`. It disables only ACLGraph capture for the layout detector. TorchAir
compilation for vision/text/decode is independent and remains enabled in the
later production phases.

Previous targeted results on this server established:

- eager scalar shape construction passed;
- eager top-k `gather` passed;
- ACLGraph capture failed with ACL error 107027 because capture rejects
  otherwise-supported dynamic metadata operations such as
  `unsqueeze(-1).repeat(...)`.

Do not run either capture probe again, do not enable layout graph capture, and
do not treat capture failure as an eager-NPU failure.

The production summary for every Phase 1-7 run must show:

```text
configuration.device = "npu:0"
configuration.layout_device = "npu:0"
configuration.layout_graph_capture = false
layout_frontend.device = "npu:0"
layout_frontend.graph_capture = false
layout_frontend.npu_indexput_compat = true
```

## Gate: one real page on eager NPU layout

Preserve all earlier CPU-layout and probe artifacts. Pull `main`, activate the
same NPU environment, and restore the exact `PYTHON_BIN`, `LAYOUT_MODEL`,
`DATASET_JSON`, and `IMAGES_DIR` values already selected in Phase 0.
If this is a fresh shell and those variables are absent, perform only the
Phase 0 path discovery and environment checks first, then return to this gate.

Use a new evidence root so no previous result is overwritten:

```sh
REPO="$(git rev-parse --show-toplevel)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
LAYOUT_GATE_ROOT="$REPO/tmp/09_persistent_page_engine/310p_layout_eager_npu_$COMMIT_SHORT"
mkdir -p "$LAYOUT_GATE_ROOT"

run_targeted_layout() {
  local run_name="$1"
  shift
  local evidence_dir="$LAYOUT_GATE_ROOT/$run_name"
  mkdir -p "$evidence_dir"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# git_commit=%s\n' "$(git rev-parse HEAD)"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local monitor_pid=
  if command -v npu-smi >/dev/null 2>&1; then
    (
      while true; do
        date --iso-8601=ns 2>/dev/null || date
        npu-smi info
        sleep 1
      done
    ) >"$evidence_dir/npu_smi_1s.log" 2>&1 &
    monitor_pid=$!
  fi

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$status" >"$evidence_dir/exit_code.txt"
  printf 'run=%s exit_code=%s log=%s\n' \
    "$run_name" "$status" "$evidence_dir/run.log"
  return "$status"
}
```

Run the first dataset page once on CPU as the same-machine semantic and timing
control:

```sh
run_targeted_layout layout_cpu_w1 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device cpu \
  --no-graph-capture \
  --offset 0 \
  --limit 1 \
  --workers 1 \
  --no-timeline \
  --output-dir "$LAYOUT_GATE_ROOT/layout_cpu_w1/output"
```

Then run exactly the same real page with eager NPU layout:

```sh
run_targeted_layout layout_npu_eager_w1 \
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
  --output-dir "$LAYOUT_GATE_ROOT/layout_npu_eager_w1/output"
```

The gate passes when the NPU process exits 0, emits a valid
`run_summary.json` and `requests.jsonl`, reports nonzero raw/filtered boxes and
requests, and has no NPU indexing or native-operator traceback. Compare CPU and
NPU request count, prompt sequence, crop sizes, and manifest SHA256. Exact
manifest equality is preferred but not mandatory: a one-pixel crop-size or
crop-boundary difference is acceptable if request count, order, prompts, and
layout structure remain coherent. Record every difference rather than hiding
it.

Report for CPU and NPU:

- setup time separately from measured frontend wall;
- page wall and pages/s;
- raw/filtered boxes, requests, and manifest SHA256;
- file read, image decode, preprocessing/H2D, detector submit/device time,
  postprocessing, and page-preparation stage totals;
- NPU/CPU speedup as `cpu_frontend_wall_s / npu_frontend_wall_s`.

If eager NPU fails, stop and preserve the first causal traceback. If it passes,
continue through Phases 0-8 below using eager NPU layout. ACLGraph capture is
out of scope.

## Immediate standalone IndexPut probe

Do not execute this historical diagnostic section during the eager-NPU layout
rerun. Skip directly to **Scope and stopping point** below. It remains here only
so the earlier environment comparison stays reproducible.

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

For the already-completed production-validation task, the stopping point was
the layout check and report. Phases 9-14 are retained historical tasks and are
not the current assignment. The current stopping point is stated at the top
of this file and in Phases 15-16.

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
| Layout | owned PaddleX-free eager NPU frontend | CPU/NPU control, summary, request manifest, no loaded PaddleX modules |
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
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_exp09_npu_layout_eager_$COMMIT_SHORT"
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
    printf '# git_commit=%s\n' "$(git rev-parse HEAD)"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local monitor_pid=
  if command -v npu-smi >/dev/null 2>&1; then
    (
      while true; do
        date --iso-8601=ns 2>/dev/null || date
        npu-smi info
        sleep 1
      done
    ) >"$evidence_dir/npu_smi_1s.log" 2>&1 &
    monitor_pid=$!
  fi

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
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
    assert configuration["device"] == "npu:0"
    assert configuration["layout_device"] == "npu:0"
    assert configuration["layout_graph_capture"] is False
    assert summary["layout_frontend"]["device"] == "npu:0"
    assert summary["layout_frontend"]["graph_capture"] is False
    assert summary["layout_frontend"]["npu_indexput_compat"] is True
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

Use this extractor on every successful production output. It reports
throughput from the actual measured device-stage and decode-control times; do
not infer token rates from E2E wall:

```sh
write_run_metrics() {
  local run_root="$1"
  local output_path="$2"
  "$PYTHON_BIN" - "$run_root" >"$output_path" <<'PY'
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
recognition = summary["recognition"]
stage = recognition["device_stage_s"]
trace = [
    json.loads(line)
    for line in (root / "recognition_trace.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

def rate(tokens, seconds):
    return float(tokens) / float(seconds) if seconds else None

def quantile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
    return values[max(index, 0)]

def distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }

vision_s = float(stage.get("vision_prefill", 0.0))
text_s = float(stage.get("text_prefill", 0.0))
decode_s = float(recognition["decode_wall_s"])
real_vision = int(recognition["real_vision_tokens"])
physical_vision = int(recognition["physical_vision_tokens"])
real_text = int(recognition["real_text_tokens"])
physical_text = int(recognition["physical_text_tokens"])
raw_decode = int(recognition["raw_decode_token_slots"])
effective_decode = int(recognition["effective_decode_tokens"])

result = {
    "run_root": str(root.resolve()),
    "configuration": {
        "layout_device": summary["configuration"]["layout_device"],
        "layout_graph_capture": summary["configuration"][
            "layout_graph_capture"
        ],
        "batch_size": summary["configuration"]["batch_size"],
        "cache_length": summary["configuration"]["cache_length"],
        "min_pixels": summary["configuration"][
            "effective_global_min_pixels"
        ],
        "vision_backend": summary["configuration"]["vision_backend"],
        "vision_attention": summary["configuration"]["vision_attention"],
        "vision_packing": summary["configuration"]["vision_packing"],
        "text_packing": summary["configuration"]["text_packing"],
    },
    "e2e": {
        "pages": summary["result_count"],
        "requests": recognition["requests"],
        "wall_s": summary["pipeline_e2e_s"],
        "pages_per_s": summary["pages_per_s"],
        "s_per_page": summary["s_per_page"],
        "setup_s": summary["setup_s"],
    },
    "layout": {
        "device": summary["layout_frontend"]["device"],
        "graph_capture": summary["layout_frontend"]["graph_capture"],
        "npu_indexput_compat": summary["layout_frontend"][
            "npu_indexput_compat"
        ],
        "stage_s": summary["layout_frontend"]["stage_s"],
        "statistics": summary["layout_frontend"]["statistics"],
    },
    "vision_prefill": {
        "real_tokens": real_vision,
        "physical_tokens": physical_vision,
        "padding_tokens": physical_vision - real_vision,
        "device_s": vision_s,
        "effective_real_tok_per_s": rate(real_vision, vision_s),
        "raw_physical_tok_per_s": rate(physical_vision, vision_s),
        "useful_fraction": recognition["vision_useful_token_fraction"],
        "execution_counts": dict(
            sorted(Counter(row["vision"]["execution"] for row in trace).items())
        ),
        "physical_tokens_per_request": distribution(
            row["vision"]["physical_vision_tokens"] for row in trace
        ),
    },
    "text_prefill": {
        "real_tokens": real_text,
        "physical_tokens": physical_text,
        "padding_tokens": physical_text - real_text,
        "device_s": text_s,
        "effective_real_tok_per_s": rate(real_text, text_s),
        "raw_physical_tok_per_s": rate(physical_text, text_s),
        "useful_fraction": recognition["text_useful_token_fraction"],
        "execution_counts": dict(
            sorted(
                Counter(
                    row["text_prefill"]["execution"] for row in trace
                ).items()
            )
        ),
        "physical_tokens_per_request": distribution(
            row["text_prefill"]["physical_text_tokens"] for row in trace
        ),
    },
    "decode": {
        "generated_tokens_including_eos": recognition[
            "generated_tokens_including_eos"
        ],
        "effective_tokens": effective_decode,
        "raw_physical_token_slots": raw_decode,
        "decode_wall_s": decode_s,
        "effective_tok_per_s": rate(effective_decode, decode_s),
        "raw_physical_tok_per_s": rate(raw_decode, decode_s),
        "useful_fraction": recognition["decode_useful_token_fraction"],
        "graph_calls": recognition["decode_graph_calls"],
        "active_slots": recognition["active_decode_token_slots"],
        "idle_slots": recognition["idle_decode_token_slots"],
        "lookahead_slots": recognition["lookahead_decode_token_slots"],
        "generated_tokens_per_request": distribution(
            row["generated_tokens_including_eos"] for row in trace
        ),
        "stop_reasons": recognition["stop_reason_counts"],
    },
    "packing": {
        "vision": recognition["vision_packing"],
        "text": recognition["text_packing"],
    },
    "all_device_stage_s": stage,
}
print(json.dumps(result, indent=2, sort_keys=True))
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
  --layout-device npu
  --no-layout-graph-capture
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
- layout configuration reports eager `npu:0`, graph capture false, and
  `npu_indexput_compat=true`.

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
route timings are calibrated for Ascend 910B2, not 310P.

Phase 9 below measures isolated 310P graph throughput. It does **not** validate
the production router, choose a new routing policy, or authorize editing the
pinned profile. Even after Phase 9 succeeds, production profile-guided routing
remains disabled until Luka explicitly asks for the measured 310P table to be
integrated and validated end to end.

## Phase 8: isolated CPU versus eager-NPU layout throughput

This is the only 32-page task. It does not load the OCR recognizer. Its primary
comparison is CPU W1 versus eager-NPU W1 on the same pages; setup time is
reported separately and excluded from steady frontend throughput.

CPU W1 control:

```sh
run_and_record phase8_layout_cpu_w1 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device cpu \
  --no-graph-capture \
  --offset 0 \
  --limit 32 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/phase8_layout_cpu_w1/output"
```

Eager-NPU W1:

```sh
run_and_record phase8_layout_npu_w1 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device npu \
  --layout-indexput-compat \
  --no-graph-capture \
  --offset 0 \
  --limit 32 \
  --workers 1 \
  --no-timeline \
  --output-dir "$OUTPUT_ROOT/phase8_layout_npu_w1/output"
```

Only if eager-NPU W1 succeeds, run eager-NPU W2:

```sh
run_and_record phase8_layout_npu_w2 \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --device npu \
  --layout-indexput-compat \
  --no-graph-capture \
  --offset 0 \
  --limit 32 \
  --workers 2 \
  --no-timeline \
  --reference-requests \
    "$OUTPUT_ROOT/phase8_layout_npu_w1/output/requests.jsonl" \
  --output-dir "$OUTPUT_ROOT/phase8_layout_npu_w2/output"
```

W2 overlaps next-page decode/preprocessing but never batches two pages through
one detector call; it is not a production-runner switch. Report for every
lane: setup time, frontend wall, pages/s, seconds/page, request count, manifest
hash, and all stage totals. For CPU W1 versus NPU W1, report exact manifest
equality plus prompt/order/crop-size differences, detector time, total
frontend speedup, and the percentage of frontend time remaining in image
decode, preprocessing/H2D, detector execution, postprocessing, and page
preparation.

## Phase 9: isolated 310P vision saturation matrix

### Purpose and strict measurement boundary

This retained historical phase answered whether the 310P vision transformer
was underfilled at small sequence lengths and whether true graph batching
closed part of the measured 910B2/310P gap. Do not execute it for the current
Phase 13 task.

Measure **raw physical vision-transformer tokens/s only**:

```text
physical tokens per graph call = batch size * static sequence length
raw physical tokens/s = physical tokens / NPU device-event graph time
```

The timed region is `VisionPrefillStage` only. It includes the compiled vision
transformer and excludes:

- model loading and graph compilation;
- cache loading and first-call setup;
- image loading, resizing, patch embedding, and input materialization;
- the projector;
- layout, text prefill, decode, and page assembly;
- packing usefulness and effective/real-token throughput.

The generic lab JSON also contains effective-token fields because other
experiments use them. Ignore those fields in this phase. Do not headline them,
average them into physical throughput, or use them to choose a winner.

The test has two views:

1. a production-B1 length sweep over already-compiled Phase 7 graphs;
2. a controlled compiled-PromptFA batch/context matrix:

   ```text
   fixed S=512:   B1x512,  B2x512,  B4x512
   fixed S=1024:  B1x1024, B2x1024, B4x1024
   fixed physical tokens/call=4096:
                  B1x4096, B2x2048, B4x1024
   ```

`B4x1024` belongs to both comparisons, so the complete matrix contains exactly
eight unique graph shapes. Do not add more shapes in this iteration.

All sequence lengths are multiples of 128. That is deliberate: Atlas 310P
PromptFA rejected a non-null attention mask for unaligned Q/K lengths during
the earlier ladder. The lab has no `--vision-promptfa-align-128` option because
the static `S` itself supplies the alignment. Do not remove the attention
mask, alter the operator call, or patch source if a graph fails.

### 9.0 Pull, reuse the proven environment, and discover retained paths

Read `CLAUDE.md`, `AGENTS.md`, and this Phase 9 section. Then:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE9_ROOT="$REPO/tmp/09_persistent_page_engine/310p_vision_saturation_$COMMIT_SHORT"
REFERENCE_ROOT="$REPO/tmp/09_persistent_page_engine/vision_lab/910b_saturation_4789067"
REFERENCE_B1="$REFERENCE_ROOT/b1_length_profile.json"
REFERENCE_MATRIX="$REFERENCE_ROOT/graph_saturation_matrix.json"
SYNTHETIC_CORPUS="$REFERENCE_ROOT/saturation_synthetic_corpus.json"

mkdir -p "$PHASE9_ROOT" "$REPO/.runtime_cache"
test -f "$REFERENCE_B1"
test -f "$REFERENCE_MATRIX"
test -f "$SYNTHETIC_CORPUS"
```

Do not rerun Phases 0-8. Recover the exact Python, recognizer model, and
production B1 vision-cache path from the latest successful Phase 7 command
record. This avoids guessing paths or accidentally selecting a different
environment:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
selected = max(matches, key=lambda path: path.stat().st_mtime)
print(selected)
PY
)"
test -f "$PHASE7_COMMAND"
printf 'phase7_command=%s\n' "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command line in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])


def option(name: str) -> str:
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc


print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
print(
    "PRODUCTION_VISION_CACHE="
    f"{shlex.quote(option('--vision-torchair-cache-dir'))}"
)
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
test -d "$PRODUCTION_VISION_CACHE"
printf 'python=%s\nmodel=%s\nproduction_b1_cache=%s\n' \
  "$PYTHON_BIN" "$RECOGNIZER_MODEL" "$PRODUCTION_VISION_CACHE"
```

Activate the same CANN/torch-npu environment used for the successful ladder
before running the next command. Keep the same free physical 310P exposed as
logical `npu:0`. Do not terminate another user's process. If no NPU is free,
stop and report that fact.

Run the exact resolver and device check:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" - \
  >"$PHASE9_ROOT/environment_preflight.txt" 2>&1 <<'PY'
import json
import platform
import sys

import torch
import torch_npu

from paddleocr_vl.model.compile_utils import import_torchair

torchair, CompilerConfig = import_torchair()
assert callable(torchair.inference.cache_compile)
assert torch.npu.is_available()
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(128, dtype=torch.float16, device="npu:0")
torch.npu.synchronize()

print("platform:", platform.platform())
print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torchair_module:", torchair.__name__)
print("torchair_file:", getattr(torchair, "__file__", "<namespace>"))
print("npu_name:", torch.npu.get_device_name(0))
print("tensor_sum:", float(x.float().sum().cpu().item()))
print("PHASE9_ENVIRONMENT: PASS")
PY

tail -n 1 "$PHASE9_ROOT/environment_preflight.txt"
test "$(tail -n 1 "$PHASE9_ROOT/environment_preflight.txt")" = \
  "PHASE9_ENVIRONMENT: PASS"
df -h "$REPO" "$REPO/.runtime_cache" >"$PHASE9_ROOT/disk_before.txt"
npu-smi info >"$PHASE9_ROOT/npu_before.txt" 2>&1
```

Record the provenance:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'production_b1_cache=%s\n' "$PRODUCTION_VISION_CACHE"
  printf 'reference_b1=%s\n' "$REFERENCE_B1"
  printf 'reference_matrix=%s\n' "$REFERENCE_MATRIX"
} >"$PHASE9_ROOT/provenance.txt"
```

### 9.1 Recording helper

Define this helper in the active shell. It records the expanded command,
complete log, exit code, and one-second NPU/HBM samples. The utilization log is
supporting evidence only: these graph calls are tens of milliseconds, so a
one-second sample can miss peaks. Batch/length throughput scaling is the
primary saturation evidence.

```sh
run_phase9() {
  local run_name="$1"
  shift
  local evidence_dir="$PHASE9_ROOT/$run_name"
  mkdir -p "$evidence_dir"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# git_commit=%s\n' "$(git rev-parse HEAD)"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local monitor_pid=
  if command -v npu-smi >/dev/null 2>&1; then
    (
      while true; do
        date --iso-8601=ns 2>/dev/null || date
        npu-smi info
        sleep 1
      done
    ) >"$evidence_dir/npu_smi_1s.log" 2>&1 &
    monitor_pid=$!
  fi

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$status" >"$evidence_dir/exit_code.txt"
  printf 'run=%s exit_code=%s log=%s\n' \
    "$run_name" "$status" "$evidence_dir/run.log"
  return "$status"
}

record_phase9_cache() {
  local cache_root="$1"
  local output="$2"
  {
    printf 'cache_root=%s\n' "$cache_root"
    if test -d "$cache_root"; then
      find "$cache_root" -type f -printf '%P\t%s\n' | sort
      printf 'file_count='
      find "$cache_root" -type f | wc -l
      du -sh "$cache_root"
    else
      printf 'missing\n'
    fi
  } >"$output"
}
```

### 9.2 Production-B1 length sweep: reuse only

This lane must reuse the successful Phase 7 singleton cache. It must not
compile new B1 graphs. The requested buckets exactly match the 910B2 reference:

```text
128,256,384,512,640,768,1408,1920,2944,4992
```

Use an intentionally empty batched-cache discovery directory so this command
profiles only production B1 graphs:

```sh
EMPTY_BATCHED_DISCOVERY="$PHASE9_ROOT/empty_batched_cache_discovery"
mkdir -p "$EMPTY_BATCHED_DISCOVERY"
test -z "$(find "$EMPTY_BATCHED_DISCOVERY" -mindepth 1 -print -quit)"

record_phase9_cache \
  "$PRODUCTION_VISION_CACHE" \
  "$PHASE9_ROOT/b1_cache_before.txt"

run_phase9 b1_length_profile \
  "$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/vision_lab_cached_profile.py" \
  --model "$RECOGNIZER_MODEL" \
  --b1-cache-dir "$PRODUCTION_VISION_CACHE" \
  --b1-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --batched-cache-dir "$EMPTY_BATCHED_DISCOVERY" \
  --warmup 3 \
  --repeats 20 \
  --output "$PHASE9_ROOT/b1_length_profile/result.json"

record_phase9_cache \
  "$PRODUCTION_VISION_CACHE" \
  "$PHASE9_ROOT/b1_cache_after.txt"
```

Validate:

```sh
"$PYTHON_BIN" - \
  "$PHASE9_ROOT/b1_length_profile/result.json" \
  >"$PHASE9_ROOT/b1_length_profile/validation.txt" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
expected = [128, 256, 384, 512, 640, 768, 1408, 1920, 2944, 4992]
graphs = payload["graphs"]
actual = [int(row["sequence_length"]) for row in graphs]
assert actual == expected, (actual, expected)
assert all(int(row["batch_size"]) == 1 for row in graphs)
assert all(int(row["physical_tokens"]) == int(row["sequence_length"]) for row in graphs)
assert all(float(row["raw_physical_tokens_per_s_median"]) > 0 for row in graphs)
assert not payload["skipped_b1"], payload["skipped_b1"]
print("PHASE9_B1_LENGTH_PROFILE: PASS")
PY

cat "$PHASE9_ROOT/b1_length_profile/validation.txt"
```

The cache inventories should have the same graph-directory set before and
after. Cache metadata files may update timestamps; do not use timestamps as
proof of compilation. The script's preflight must find all ten compatible
graphs.

If this lane fails only because no compatible retained B1 cache is found,
preserve the exact expected cache paths from `skipped_b1`, mark the B1 sweep
`PARTIAL`, and continue to the independent batch/context matrix. Do **not**
compile the ten production B1 buckets just for this phase. If it fails with an
NPU runtime/native-op error or leaves the device unhealthy, stop the phase.

### 9.3 Controlled compiled-PromptFA matrix

The tracked synthetic corpus contains four valid near-square crops at each of
512, 1024, 2048, and 4096 vision tokens. Its `default` and
`synthetic_repeat` corpora are intentionally identical. They provide two
independent timed passes through every graph; the difference between those
passes is the repeatability check.

Do not interpret the corpus's packing efficiency. Only read:

```text
results.<corpus>.<shape>.target_batch_metrics.raw_physical_tokens_per_s
```

Create one persistent 310P-only cache. Never copy the committed 910B2 cache:

```sh
MATRIX_CACHE="$REPO/.runtime_cache/310p_vision_saturation_matrix_$COMMIT_SHORT"

if test -e "$MATRIX_CACHE"; then
  if find "$PHASE9_ROOT" -path '*/matrix_*/exit_code.txt' -print -quit |
    grep -q .; then
    printf 'resuming existing Phase 9 cache: %s\n' "$MATRIX_CACHE"
  else
    printf 'unexpected pre-existing matrix cache: %s\n' "$MATRIX_CACHE"
    exit 1
  fi
else
  mkdir -p "$MATRIX_CACHE"
fi

record_phase9_cache \
  "$MATRIX_CACHE" \
  "$PHASE9_ROOT/matrix_cache_before.txt"

MATRIX_COMMON=(
  "$PYTHON_BIN"
  "$REPO/09_persistent_page_engine/scripts/vision_lab_batched_packed.py"
  --corpus "$SYNTHETIC_CORPUS"
  --model "$RECOGNIZER_MODEL"
  --variant synthetic_repeat
  --cache-dir "$MATRIX_CACHE"
  --warmup 3
  --repeats 10
)
```

Compile/measure three failure-isolated groups. Run them sequentially on the
same physical NPU.

#### Group A: fixed S=512

```sh
run_phase9 matrix_s512_first \
  "${MATRIX_COMMON[@]}" \
  --shape 1x512 \
  --shape 2x512 \
  --shape 4x512 \
  --output "$PHASE9_ROOT/matrix_s512_first/result.json"

record_phase9_cache \
  "$MATRIX_CACHE" \
  "$PHASE9_ROOT/matrix_cache_after_s512.txt"
```

#### Group B: fixed S=1024

```sh
run_phase9 matrix_s1024_first \
  "${MATRIX_COMMON[@]}" \
  --shape 1x1024 \
  --shape 2x1024 \
  --shape 4x1024 \
  --output "$PHASE9_ROOT/matrix_s1024_first/result.json"

record_phase9_cache \
  "$MATRIX_CACHE" \
  "$PHASE9_ROOT/matrix_cache_after_s1024.txt"
```

#### Group C: fixed 4096 physical tokens per graph call

`B4x1024` must reuse the graph created by Group B. The only new Group C shapes
should be `B1x4096` and `B2x2048`.

```sh
run_phase9 matrix_fixed4096_first \
  "${MATRIX_COMMON[@]}" \
  --shape 1x4096 \
  --shape 2x2048 \
  --shape 4x1024 \
  --output "$PHASE9_ROOT/matrix_fixed4096_first/result.json"

record_phase9_cache \
  "$MATRIX_CACHE" \
  "$PHASE9_ROOT/matrix_cache_after_fixed4096.txt"
```

For a failed group, preserve the log and first causal traceback. The last
`load_or_compile=b..._s...` line identifies the shape. Do not patch the model,
change attention, drop the mask, alter dtype, or substitute eager execution.
If the process exits cleanly and the NPU remains healthy, continue the other
independent groups so Luka gets the broadest compatibility map. If the device
runtime is unhealthy, stop immediately.

### 9.4 One all-shape warm-cache replay

Run this only if all three groups succeeded. It is the authoritative 310P
matrix used by the comparison script:

```sh
run_phase9 matrix_all_replay \
  "${MATRIX_COMMON[@]}" \
  --shape 1x512 \
  --shape 2x512 \
  --shape 4x512 \
  --shape 1x1024 \
  --shape 2x1024 \
  --shape 4x1024 \
  --shape 1x4096 \
  --shape 2x2048 \
  --output "$PHASE9_ROOT/matrix_all_replay/result.json"

record_phase9_cache \
  "$MATRIX_CACHE" \
  "$PHASE9_ROOT/matrix_cache_after_replay.txt"
```

Validate physical-token accounting, both identical corpus passes, all eight
shapes, and warm-cache discovery:

```sh
"$PYTHON_BIN" - \
  "$PHASE9_ROOT/matrix_all_replay/result.json" \
  >"$PHASE9_ROOT/matrix_all_replay/validation.txt" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "b1_s512": (1, 512),
    "b2_s512": (2, 512),
    "b4_s512": (4, 512),
    "b1_s1024": (1, 1024),
    "b2_s1024": (2, 1024),
    "b4_s1024": (4, 1024),
    "b1_s4096": (1, 4096),
    "b2_s2048": (2, 2048),
}
assert set(payload["results"]) == {"default", "synthetic_repeat"}
for corpus_name, rows in payload["results"].items():
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == set(expected), (corpus_name, sorted(by_name))
    for name, (batch, sequence) in expected.items():
        row = by_name[name]
        assert row["supported"] is True, (corpus_name, name, row.get("failure"))
        assert int(row["batch_size"]) == batch
        assert int(row["sequence_length"]) == sequence
        assert int(row["physical_tokens_per_full_call"]) == batch * sequence
        metrics = row["target_batch_metrics"]
        assert int(metrics["physical_tokens"]) % (batch * sequence) == 0
        assert float(metrics["raw_physical_tokens_per_s"]) > 0

assert payload["new_graphs_compiled"] == 0, payload["compile_metadata"]
assert all(
    row["cache_existed_before_run"]
    for row in payload["compile_metadata"].values()
)
print("PHASE9_MATRIX_REPLAY: PASS")
PY

cat "$PHASE9_ROOT/matrix_all_replay/validation.txt"
```

The `new_graphs_compiled == 0` check is based on compatible cache discovery.
Also compare the before/after cache inventories and inspect the log for
unexpected new graph directories. `compile_first_call_s` is cache
load/initialization evidence, not throughput.

### 9.5 Mechanical 910B2 versus 310P comparison

The committed 910B2 references were measured with:

```text
source commit: 47890673ee60090851d372329079355d9250f5c5
device: Ascend910B2, physical NPU 5
torch: 2.10.0+cpu
torch_npu: 2.10.0
attention: compiled PromptFA
timing: NPU device events around VisionPrefillStage only
B1 sweep: warmup 3, repeats 20
matrix: warmup 3, repeats 10, two identical corpus passes
```

The relevant vision-lab source files are unchanged between that source commit
and the commit that added these instructions. The JSON references, not this
rounded table, are authoritative:

| 910B2 shape | Raw physical tok/s |
|---|---:|
| B1x512 | 32,178 |
| B2x512 | 49,723 |
| B4x512 | 75,609 |
| B1x1024 | 47,092 |
| B2x1024 | 70,433 |
| B4x1024 | 87,674 |
| B1x4096 | 64,306 |
| B2x2048 | 79,095 |

Generate the exact comparison mechanically:

```sh
"$PYTHON_BIN" - \
  "$REFERENCE_B1" \
  "$REFERENCE_MATRIX" \
  "$PHASE9_ROOT/b1_length_profile/result.json" \
  "$PHASE9_ROOT/matrix_all_replay/result.json" \
  "$PHASE9_ROOT/phase9_comparison.json" \
  "$PHASE9_ROOT/phase9_comparison.md" <<'PY'
import json
import statistics
import sys
from pathlib import Path

(
    reference_b1_path,
    reference_matrix_path,
    target_b1_path,
    target_matrix_path,
    output_json_path,
    output_md_path,
) = map(Path, sys.argv[1:])


def load(path: Path):
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def b1_rows(payload):
    result = {}
    for row in payload["graphs"]:
        assert int(row["batch_size"]) == 1
        sequence = int(row["sequence_length"])
        result[sequence] = {
            "median_ms": float(row["median_ms"]),
            "raw_physical_tok_s": float(
                row["raw_physical_tokens_per_s_median"]
            ),
        }
    return result


def matrix_rows(payload):
    by_shape = {}
    for corpus_name, rows in payload["results"].items():
        for row in rows:
            assert row["supported"] is True
            batch = int(row["batch_size"])
            sequence = int(row["sequence_length"])
            shape = f"B{batch}x{sequence}"
            physical = int(row["physical_tokens_per_full_call"])
            assert physical == batch * sequence
            rate = float(
                row["target_batch_metrics"][
                    "raw_physical_tokens_per_s"
                ]
            )
            by_shape.setdefault(shape, []).append(rate)
    result = {}
    for shape, values in by_shape.items():
        assert len(values) == 2, (shape, values)
        mean = statistics.mean(values)
        spread_pct = 100.0 * (max(values) - min(values)) / mean
        result[shape] = {
            "per_pass_raw_physical_tok_s": values,
            "raw_physical_tok_s": mean,
            "duplicate_spread_pct": spread_pct,
        }
    return result


reference_b1 = b1_rows(load(reference_b1_path))
target_b1 = b1_rows(load(target_b1_path))
reference_matrix = matrix_rows(load(reference_matrix_path))
target_matrix = matrix_rows(load(target_matrix_path))

b1_order = [128, 256, 384, 512, 640, 768, 1408, 1920, 2944, 4992]
matrix_order = [
    "B1x512",
    "B2x512",
    "B4x512",
    "B1x1024",
    "B2x1024",
    "B4x1024",
    "B1x4096",
    "B2x2048",
]
assert set(b1_order) == set(reference_b1) == set(target_b1)
assert set(matrix_order) == set(reference_matrix) == set(target_matrix)

b1_comparison = []
for sequence in b1_order:
    r910 = reference_b1[sequence]["raw_physical_tok_s"]
    r310 = target_b1[sequence]["raw_physical_tok_s"]
    b1_comparison.append(
        {
            "shape": f"B1x{sequence}",
            "910b2_raw_physical_tok_s": r910,
            "310p_raw_physical_tok_s": r310,
            "ratio_910b2_over_310p": r910 / r310,
        }
    )

matrix_comparison = []
for shape in matrix_order:
    r910 = reference_matrix[shape]["raw_physical_tok_s"]
    r310 = target_matrix[shape]["raw_physical_tok_s"]
    matrix_comparison.append(
        {
            "shape": shape,
            "910b2_raw_physical_tok_s": r910,
            "310p_raw_physical_tok_s": r310,
            "ratio_910b2_over_310p": r910 / r310,
            "910b2_duplicate_spread_pct": reference_matrix[shape][
                "duplicate_spread_pct"
            ],
            "310p_duplicate_spread_pct": target_matrix[shape][
                "duplicate_spread_pct"
            ],
        }
    )


def scaling(rows, numerator, denominator):
    return (
        rows[numerator]["raw_physical_tok_s"]
        / rows[denominator]["raw_physical_tok_s"]
    )


scaling_summary = {
    "s512": {
        "910b2_b2_over_b1": scaling(reference_matrix, "B2x512", "B1x512"),
        "910b2_b4_over_b1": scaling(reference_matrix, "B4x512", "B1x512"),
        "310p_b2_over_b1": scaling(target_matrix, "B2x512", "B1x512"),
        "310p_b4_over_b1": scaling(target_matrix, "B4x512", "B1x512"),
    },
    "s1024": {
        "910b2_b2_over_b1": scaling(reference_matrix, "B2x1024", "B1x1024"),
        "910b2_b4_over_b1": scaling(reference_matrix, "B4x1024", "B1x1024"),
        "310p_b2_over_b1": scaling(target_matrix, "B2x1024", "B1x1024"),
        "310p_b4_over_b1": scaling(target_matrix, "B4x1024", "B1x1024"),
    },
    "fixed_4096_physical_tokens": {
        "910b2_b2x2048_over_b1x4096": scaling(
            reference_matrix, "B2x2048", "B1x4096"
        ),
        "910b2_b4x1024_over_b1x4096": scaling(
            reference_matrix, "B4x1024", "B1x4096"
        ),
        "310p_b2x2048_over_b1x4096": scaling(
            target_matrix, "B2x2048", "B1x4096"
        ),
        "310p_b4x1024_over_b1x4096": scaling(
            target_matrix, "B4x1024", "B1x4096"
        ),
    },
}

payload = {
    "metric": "raw physical VisionPrefillStage tokens/s",
    "b1_length_comparison": b1_comparison,
    "matrix_comparison": matrix_comparison,
    "batch_scaling": scaling_summary,
}
output_json_path.resolve().write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# Phase 9: 310P versus 910B2 raw physical vision throughput",
    "",
    "Only device-event time around VisionPrefillStage is compared.",
    "Compilation, embeddings, projector, and effective-token rates are excluded.",
    "",
    "## Production B1 length sweep",
    "",
    "| Shape | 910B2 physical tok/s | 310P physical tok/s | 910B2 / 310P |",
    "|---|---:|---:|---:|",
]
for row in b1_comparison:
    lines.append(
        f"| {row['shape']} | "
        f"{row['910b2_raw_physical_tok_s']:.1f} | "
        f"{row['310p_raw_physical_tok_s']:.1f} | "
        f"{row['ratio_910b2_over_310p']:.3f}x |"
    )

lines.extend(
    [
        "",
        "## Controlled batch/context matrix",
        "",
        "| Shape | 910B2 physical tok/s | 310P physical tok/s | "
        "910B2 / 310P | 310P duplicate spread |",
        "|---|---:|---:|---:|---:|",
    ]
)
for row in matrix_comparison:
    lines.append(
        f"| {row['shape']} | "
        f"{row['910b2_raw_physical_tok_s']:.1f} | "
        f"{row['310p_raw_physical_tok_s']:.1f} | "
        f"{row['ratio_910b2_over_310p']:.3f}x | "
        f"{row['310p_duplicate_spread_pct']:.3f}% |"
    )

lines.extend(["", "## Scaling ratios", ""])
for section, values in scaling_summary.items():
    lines.append(f"### {section}")
    lines.append("")
    for name, value in values.items():
        lines.append(f"- {name}: {value:.4f}x")
    lines.append("")

output_md_path.resolve().write_text(
    "\n".join(lines).rstrip() + "\n",
    encoding="utf-8",
)
print("PHASE9_COMPARISON: PASS")
PY

cat "$PHASE9_ROOT/phase9_comparison.md"
```

### 9.6 Interpretation rules

Answer from measured scaling, not from one utilization sample:

- `B4/B1` at fixed `S=512` quantifies small-context underfill.
- `B4/B1` at fixed `S=1024` shows whether batching still helps after each row
  contains more work.
- `B1x4096`, `B2x2048`, and `B4x1024` all execute 4096 physical tokens per
  graph call, but they do not perform equal attention FLOPs. The comparison
  deliberately shows whether one long packed row or several shorter
  independent rows produces higher physical token throughput.
- If 310P throughput rises strongly from B1 to B4, the production B1/small-row
  regime is underfilled.
- If the shape-for-shape 910B2/310P ratio shrinks materially at B4, batching
  closes part of the original platform gap.
- If the ratio remains roughly constant across B1/B2/B4, the absolute
  platform/kernel difference persists even after better filling the 310P.
- Do not call either chip “fully saturated” from this matrix alone. Say
  “throughput plateaued over the tested shapes” when that is what the numbers
  show.

The B1 length curve separately shows where increasing one row stops improving
physical tok/s. Report the peak tested B1 length and whether throughput falls
at 4992.

### 9.7 Phase 9 stop conditions and dedicated report

Stop after the comparison. Do not:

- run OmniDocBench pages;
- run the production recognizer pipeline;
- modify `PINNED_910B2_PROFILE`;
- enable profile-guided routing in production;
- compile additional shapes;
- turn an unsupported compiled graph into an eager test;
- patch PromptFA, model code, masks, or cache logic;
- install or replace packages.

Write:

```text
$PHASE9_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P VISION SATURATION MATRIX: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical 310P:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver route:
CANN / driver / firmware:
Recognizer model:

Measurement boundary:
attention / execution / dtype:
timing source:
included:
excluded:
metric reported: raw physical VisionPrefillStage tokens/s

Production B1 cache:
cache source Phase 7 command:
compatible B1 graphs found:
new B1 graphs compiled: MUST BE ZERO

B1 length sweep:
shape | median graph ms | 310P physical tok/s | 910B2 physical tok/s | ratio
peak B1 shape and physical tok/s:
does throughput fall at 4992:

Controlled matrix cache:
new shapes compiled:
warm replay cache status:

Fixed S=512:
B1x512:
B2x512:
B4x512:
B2/B1 scaling:
B4/B1 scaling:

Fixed S=1024:
B1x1024:
B2x1024:
B4x1024:
B2/B1 scaling:
B4/B1 scaling:

Fixed 4096 physical tokens/call:
B1x4096:
B2x2048:
B4x1024:
B2x2048 / B1x4096:
B4x1024 / B1x4096:

Shape-for-shape 910B2 / 310P ratios:
maximum duplicate-pass spread:
unsupported shapes or native errors:

Conclusion:
Is small-context B1 underfilled on 310P:
Does batching close part of the 910B2/310P gap:
Does the absolute gap persist at the best tested 310P shape:
What the matrix does NOT prove:

First causal warning or blocker:
Exact command records:
Cache inventories:
JSON and Markdown comparison paths:
All artifact paths:
```

Include the complete `phase9_comparison.md` table in the report. Report raw
physical tok/s to at least one decimal place and ratios to at least three
decimal places. Do not substitute effective tok/s.

## Phase 10: native B1 vision profiles at S512 and S2048

### 10.0 Purpose, boundary, and reference contract

This retained historical phase captures two already-warmed compiled PromptFA
graphs with the native `torch_npu` profiler. Do not execute it for the current
Phase 13 task:

```text
B1xS512
B1xS2048
```

The exact graph boundary is:

```text
VisionPrefillRuntime.compiled[sequence_length](
    prefix_hidden_states,
    rope_cos,
    rope_sin,
    attention_mask,
)
```

It includes all 27 vision encoder layers plus final post-LayerNorm. It excludes
patch embedding, image preprocessing, projector, text prefill, decode, layout,
page assembly, model loading, graph compilation, and cache first-call setup.
Inputs are synthetic device tensors with the exact production shapes and
dtypes. This phase measures raw physical transformer tokens only.

Every final profiler lane uses the same protocol:

```text
outside-profiler graph warmup: 3 replays
unprofiled control before capture: 20 NPU-event replays
profiler-scheduled warmup: 1 replay
profiler-active capture: 5 replays
unprofiled control after capture: 20 NPU-event replays
profiler: CPU + NPU, Level1, PipeUtilization
record_shapes: true
with_stack: true
with_modules: true
```

Use the mean of the before/after control medians as the physical-throughput
reference. The profiled replay is diagnostic only because collection perturbs
execution. Do not use profiler wall time or `profiler.step()` time as
throughput.

The committed 910B2 references are:

```text
S512:
  tmp/09_persistent_page_engine/vision_lab/
  vision_s512_npu_profile_910b_7d1f778/

S2048:
  tmp/09_persistent_page_engine/vision_lab/
  vision_s2048_npu_profile_910b_5ff4b6b/
```

Both used physical NPU 5 (`Ascend910B2`), `torch==2.10.0+cpu`,
`torch_npu==2.10.0`, FP16 compiled PromptFA, head dimension 72 padded to 80,
and the same `vision_prefill.py` source hash. The source commits differ only
because the profiler harness was generalized before the S2048 capture.

The authoritative JSON values are:

| Shape | control ms | physical tok/s | profiled ms | profiler overhead | kernel ms | kernels/replay | weighted cube |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1xS512 | 15.857095 | 32,288.386 | 16.774080 | 5.7828% | 15.583124 | 1,055 | 91.7560% |
| B1xS2048 | 31.648320 | 64,711.176 | 32.393841 | 2.3556% | 31.267644 | 1,055 | 94.5412% |

The 910B2 kernel mix, per replay, was:

| Kernel family | S512 ms / share | S2048 ms / share |
|---|---:|---:|
| all MatMul variants | 4.9613 / 31.84% | 7.9175 / 25.32% |
| PromptFlashAttention | 1.7098 / 10.97% | 7.6396 / 24.43% |
| StridedSliceD | 2.0222 / 12.98% | 3.7378 / 11.95% |
| Transpose | 2.0078 / 12.88% | 3.7261 / 11.92% |
| PadV3 | 0.9227 / 5.92% | 1.3291 / 4.25% |

Do not type these rounded numbers into the final comparison. Read the committed
JSON mechanically.

### 10.1 Pull and recover the proven 310P paths

Start in the repository and leave all earlier artifacts untouched:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE10_REL="tmp/09_persistent_page_engine/310p_vision_profiles_$COMMIT_SHORT"
PHASE10_ROOT="$REPO/$PHASE10_REL"
RAW_PROFILE_ROOT="$REPO/.runtime_cache/310p_vision_profiles_$COMMIT_SHORT"
PROFILER_SCRIPT="$REPO/09_persistent_page_engine/scripts/profile_vision_prefill_b1.py"
REFERENCE_S512="$REPO/tmp/09_persistent_page_engine/vision_lab/vision_s512_npu_profile_910b_7d1f778"
REFERENCE_S2048="$REPO/tmp/09_persistent_page_engine/vision_lab/vision_s2048_npu_profile_910b_5ff4b6b"

test -f "$PROFILER_SCRIPT"
test -f "$REFERENCE_S512/run_summary.json"
test -f "$REFERENCE_S512/parsed_profile_summary.json"
test -f "$REFERENCE_S2048/run_summary.json"
test -f "$REFERENCE_S2048/parsed_profile_summary.json"
test ! -e "$PHASE10_ROOT"
test ! -e "$RAW_PROFILE_ROOT"
mkdir -p "$PHASE10_ROOT/evidence" "$PHASE10_ROOT/results" "$RAW_PROFILE_ROOT"
```

Recover the exact Python, recognizer model, and production B1 cache from the
latest successful Phase 7 command. Do not guess paths:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
test -f "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])


def option(name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc


print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
print(
    "PRODUCTION_VISION_CACHE="
    f"{shlex.quote(option('--vision-torchair-cache-dir'))}"
)
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
test -d "$PRODUCTION_VISION_CACHE"
printf 'python=%s\nmodel=%s\ncache=%s\n' \
  "$PYTHON_BIN" "$RECOGNIZER_MODEL" "$PRODUCTION_VISION_CACHE"
```

Activate the same CANN/torch-npu environment used for the successful Phase 9
runs. Keep one free physical 310P exposed as logical `npu:0`. Do not kill or
reuse another user's process. If no NPU is free, stop.

Run this exact preflight:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" - >"$PHASE10_ROOT/environment_preflight.txt" 2>&1 <<'PY'
import platform
import sys

import torch
import torch_npu
import torch_npu.profiler as npu_prof

from paddleocr_vl.model.compile_utils import import_torchair

torchair, CompilerConfig = import_torchair()
assert callable(torchair.inference.cache_compile)
assert torch.npu.is_available()
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
assert npu_prof.ProfilerActivity.CPU is not None
assert npu_prof.ProfilerActivity.NPU is not None
assert npu_prof.AiCMetrics.PipeUtilization is not None
x = torch.arange(128, dtype=torch.float16, device="npu:0")
torch.npu.synchronize()

print("platform:", platform.platform())
print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torchair_module:", torchair.__name__)
print("torchair_file:", getattr(torchair, "__file__", "<namespace>"))
print("npu_name:", torch.npu.get_device_name(0))
print("tensor_sum:", float(x.float().sum().cpu().item()))
print("PHASE10_ENVIRONMENT: PASS")
PY

tail -n 1 "$PHASE10_ROOT/environment_preflight.txt"
test "$(tail -n 1 "$PHASE10_ROOT/environment_preflight.txt")" = \
  "PHASE10_ENVIRONMENT: PASS"
df -h "$REPO" "$REPO/.runtime_cache" >"$PHASE10_ROOT/disk_before.txt"
npu-smi info >"$PHASE10_ROOT/npu_before.txt" 2>&1
```

Record provenance:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'production_vision_cache=%s\n' "$PRODUCTION_VISION_CACHE"
  printf 'reference_s512=%s\n' "$REFERENCE_S512"
  printf 'reference_s2048=%s\n' "$REFERENCE_S2048"
} >"$PHASE10_ROOT/provenance.txt"
```

### 10.2 Recording helper

Define this helper in the same shell. Its one-second `npu-smi` samples are
supporting evidence only; a 16-32 ms replay can fall between samples.

```sh
run_phase10() {
  local run_name="$1"
  shift
  local evidence_dir="$PHASE10_ROOT/evidence/$run_name"
  mkdir -p "$evidence_dir"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# git_commit=%s\n' "$(git rev-parse HEAD)"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' "$@"
    printf '\n'
  } >"$evidence_dir/command.sh"
  chmod +x "$evidence_dir/command.sh"

  local monitor_pid=
  if command -v npu-smi >/dev/null 2>&1; then
    (
      while true; do
        date --iso-8601=ns 2>/dev/null || date
        npu-smi info
        sleep 1
      done
    ) >"$evidence_dir/npu_smi_1s.log" 2>&1 &
    monitor_pid=$!
  fi

  local status
  if "$@" >"$evidence_dir/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$status" >"$evidence_dir/exit_code.txt"
  printf 'run=%s exit_code=%s log=%s\n' \
    "$run_name" "$status" "$evidence_dir/run.log"
  return "$status"
}
```

### 10.3 Prepare the exact caches outside profiling

Phase 9's production B1 sweep already required a compatible S512 graph.
Verify it without permitting compilation:

```sh
run_phase10 cache_prepare_s512 \
  "$PYTHON_BIN" "$PROFILER_SCRIPT" \
  --sequence-length 512 \
  --model "$RECOGNIZER_MODEL" \
  --cache-dir "$PRODUCTION_VISION_CACHE" \
  --prepare-cache-only \
  --output-dir "$PHASE10_ROOT/results/cache_prepare_s512"
```

The S2048 profile needs a B1xS2048 graph. Phase 9's B2xS2048 matrix graph is a
different static shape and cache key; it cannot satisfy this test. Permit at
most this one new B1 graph:

```sh
run_phase10 cache_prepare_s2048 \
  "$PYTHON_BIN" "$PROFILER_SCRIPT" \
  --sequence-length 2048 \
  --model "$RECOGNIZER_MODEL" \
  --cache-dir "$PRODUCTION_VISION_CACHE" \
  --allow-compile-if-missing \
  --prepare-cache-only \
  --output-dir "$PHASE10_ROOT/results/cache_prepare_s2048"
```

Compilation or cache loading occurs before either profiler lane. Validate:

```sh
"$PYTHON_BIN" - \
  "$PHASE10_ROOT/results/cache_prepare_s512/cache_preparation.json" \
  "$PHASE10_ROOT/results/cache_prepare_s2048/cache_preparation.json" \
  >"$PHASE10_ROOT/cache_validation.txt" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

s512 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
s2048 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

assert s512["shape"]["batch_size"] == 1
assert s512["shape"]["sequence_length"] == 512
assert s512["cache_existed_before_run"] is True
assert s512["compile_was_permitted"] is False
assert s512["cache_populated_after_setup"] is True

assert s2048["shape"]["batch_size"] == 1
assert s2048["shape"]["sequence_length"] == 2048
assert s2048["compile_was_permitted"] is True
assert s2048["cache_populated_after_setup"] is True

assert s512["environment"]["commit"] == s2048["environment"]["commit"]
assert s512["environment"]["device_name"] == s2048["environment"]["device_name"]
assert s512["environment"]["torch"] == s2048["environment"]["torch"]
assert s512["environment"]["torch_npu"] == s2048["environment"]["torch_npu"]
print("s2048_cache_existed_before_run:", s2048["cache_existed_before_run"])
print("s2048_setup_s:", s2048["setup_s"])
print("PHASE10_CACHE_PREPARATION: PASS")
PY

cat "$PHASE10_ROOT/cache_validation.txt"
```

If S512 is missing, stop: that means the wrong Phase 7 cache or incompatible
source/environment was selected. If S2048 compilation fails, preserve the
first causal traceback and stop. Do not change attention, mask, dtype, source,
or static shape.

### 10.4 Run both final cache-only profiles

The following commands deliberately omit `--allow-compile-if-missing`. Both
must report:

```text
cache_only = true
cache_existed_before_run = true
compile_was_permitted = false
```

Run S512:

```sh
run_phase10 profile_s512 \
  "$PYTHON_BIN" "$PROFILER_SCRIPT" \
  --sequence-length 512 \
  --model "$RECOGNIZER_MODEL" \
  --cache-dir "$PRODUCTION_VISION_CACHE" \
  --warmup 3 \
  --control-repeats 20 \
  --profile-warmup-steps 1 \
  --profile-steps 5 \
  --parser-topn 30 \
  --output-dir "$PHASE10_ROOT/results/profile_s512" \
  --profile-dir "$RAW_PROFILE_ROOT/s512"
```

Then run S2048 on the same physical 310P:

```sh
run_phase10 profile_s2048 \
  "$PYTHON_BIN" "$PROFILER_SCRIPT" \
  --sequence-length 2048 \
  --model "$RECOGNIZER_MODEL" \
  --cache-dir "$PRODUCTION_VISION_CACHE" \
  --warmup 3 \
  --control-repeats 20 \
  --profile-warmup-steps 1 \
  --profile-steps 5 \
  --parser-topn 30 \
  --output-dir "$PHASE10_ROOT/results/profile_s2048" \
  --profile-dir "$RAW_PROFILE_ROOT/s2048"
```

Do not rerun a failed profile into the same directories. Preserve the raw
trace and complete log, diagnose the first causal error, and report it.

### 10.5 Validate the profiler contracts

```sh
"$PYTHON_BIN" - \
  "$PHASE10_ROOT/results/profile_s512" \
  "$PHASE10_ROOT/results/profile_s2048" \
  >"$PHASE10_ROOT/profile_validation.txt" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

roots = [Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()]
expected_lengths = [512, 2048]
summaries = []

for root, sequence_length in zip(roots, expected_lengths):
    summary = json.loads(
        (root / "run_summary.json").read_text(encoding="utf-8")
    )
    parsed = json.loads(
        (root / "parsed_profile_summary.json").read_text(encoding="utf-8")
    )
    assert summary["shape"]["batch_size"] == 1
    assert summary["shape"]["sequence_length"] == sequence_length
    assert summary["shape"]["physical_tokens_per_replay"] == sequence_length
    assert summary["attention"] == "prompt_flash_attention"
    assert summary["cache_only"] is True
    assert summary["cache_existed_before_run"] is True
    assert summary["compile_was_permitted"] is False
    assert summary["cache_populated_after_setup"] is True
    assert summary["profiler"]["level"] == "Level1"
    assert summary["profiler"]["aic_metric"] == "PipeUtilization"
    assert summary["profiler"]["scheduled_warmup_steps"] == 1
    assert summary["profile_active_steps"] == 5
    assert len(
        summary["measurements"]["pre_profile_control"]["device_event"][
            "samples_ms"
        ]
    ) == 20
    assert len(
        summary["measurements"]["profiled_diagnostic"]["device_event"][
            "samples_ms"
        ]
    ) == 5
    assert len(
        summary["measurements"]["post_profile_control"]["device_event"][
            "samples_ms"
        ]
    ) == 20
    assert summary["output"]["shape"][:2] == [1, sequence_length]
    assert summary["output"]["finite"] is True
    assert len(parsed["runs"]) == 1
    run = parsed["runs"][0]
    kernel = run["kernel_details"]
    assert kernel["row_count"] > 0
    names = {
        row["name"].strip()
        for row in kernel["top_kernel_types"]
    }
    assert "PromptFlashAttention" in names
    assert any(name.startswith("MatMul") for name in names)
    trace_path = Path(run["files"]["trace_view"])
    assert trace_path.is_file(), trace_path
    summaries.append(summary)

for field in ("commit", "device_name", "torch", "torch_npu"):
    assert (
        summaries[0]["environment"][field]
        == summaries[1]["environment"][field]
    ), field

print(
    "s512_control_tok_s:",
    summaries[0]["measurements"]["control_center_physical_tokens_per_s"],
)
print(
    "s2048_control_tok_s:",
    summaries[1]["measurements"]["control_center_physical_tokens_per_s"],
)
print("PHASE10_PROFILE_CONTRACTS: PASS")
PY

cat "$PHASE10_ROOT/profile_validation.txt"
```

### 10.6 Generate the four-way comparison mechanically

This comparison reads both committed 910B2 profiles and both new 310P
profiles. It aggregates every MatMul variant together because GE may select
`MatMulV2` at one sequence length and `MatMulV3` at another.

```sh
"$PYTHON_BIN" - \
  "$REFERENCE_S512" \
  "$REFERENCE_S2048" \
  "$PHASE10_ROOT/results/profile_s512" \
  "$PHASE10_ROOT/results/profile_s2048" \
  "$PHASE10_ROOT/profile_comparison.json" \
  "$PHASE10_ROOT/profile_comparison.md" <<'PY'
import json
import sys
from pathlib import Path

(
    reference_s512,
    reference_s2048,
    target_s512,
    target_s2048,
    output_json,
    output_md,
) = map(Path, sys.argv[1:])


def load(root):
    return (
        json.loads((root / "run_summary.json").read_text(encoding="utf-8")),
        json.loads(
            (root / "parsed_profile_summary.json").read_text(encoding="utf-8")
        ),
    )


def aggregate_profile(summary, parsed):
    run = parsed["runs"][0]
    kernel = run["kernel_details"]
    active_steps = int(summary["profile_active_steps"])
    groups = {}
    for row in kernel["top_kernel_types"]:
        name = row["name"].strip()
        bucket = groups.setdefault(
            name,
            {"count": 0, "duration_us": 0.0, "aicore_time_us": 0.0},
        )
        bucket["count"] += int(row["count"])
        bucket["duration_us"] += float(row["duration_us"])
        bucket["aicore_time_us"] += float(row["aicore_time_us"])

    def family(prefix):
        rows = [
            value
            for name, value in groups.items()
            if name.startswith(prefix)
        ]
        return {
            "count_per_replay": sum(row["count"] for row in rows)
            / active_steps,
            "ms_per_replay": sum(row["duration_us"] for row in rows)
            / active_steps
            / 1000.0,
        }

    def exact(*names):
        rows = [groups[name] for name in names if name in groups]
        return {
            "count_per_replay": sum(row["count"] for row in rows)
            / active_steps,
            "ms_per_replay": sum(row["duration_us"] for row in rows)
            / active_steps
            / 1000.0,
        }

    total_ms = float(kernel["total_duration_us"]) / active_steps / 1000.0
    categories = {
        "matmul_all": family("MatMul"),
        "prompt_flash_attention": exact("PromptFlashAttention"),
        "strided_slice": exact("StridedSliceD"),
        "transpose": exact("Transpose"),
        "pad": exact("PadV3"),
        "add_layernorm": exact("AddLayerNorm"),
        "concat": exact("ConcatV2D"),
        "gelu": exact("Gelu"),
        "rotary_elementwise": exact("Mul", "Add", "Cast", "Neg"),
    }
    for values in categories.values():
        values["share_pct"] = (
            100.0 * values["ms_per_replay"] / total_ms
            if total_ms
            else None
        )

    measurement = summary["measurements"]
    pre = measurement["pre_profile_control"]["device_event"]
    post = measurement["post_profile_control"]["device_event"]
    profiled = measurement["profiled_diagnostic"]["device_event"]
    pre_median = float(pre["median_ms"])
    post_median = float(post["median_ms"])
    return {
        "shape": summary["shape"],
        "environment": {
            key: summary["environment"][key]
            for key in ("commit", "device_name", "torch", "torch_npu")
        },
        "control_center_ms": float(
            measurement["control_center_device_median_ms"]
        ),
        "control_physical_tok_s": float(
            measurement["control_center_physical_tokens_per_s"]
        ),
        "pre_control_ms": pre_median,
        "post_control_ms": post_median,
        "pre_post_drift_pct": (
            100.0 * (post_median - pre_median)
            / ((pre_median + post_median) / 2.0)
        ),
        "profiled_ms": float(profiled["median_ms"]),
        "profiled_physical_tok_s": float(
            profiled["physical_tokens_per_s_median"]
        ),
        "profiler_slowdown_pct": float(
            measurement["profiled_device_slowdown_pct"]
        ),
        "kernel_ms_per_replay": total_ms,
        "kernels_per_replay": float(kernel["row_count"]) / active_steps,
        "aicore_ms_per_replay": float(
            kernel["total_aicore_time_us"]
        )
        / active_steps
        / 1000.0,
        "weighted_cube_utilization_pct": float(
            kernel["weighted_cube_utilization_pct"]
        ),
        "categories": categories,
    }


profiles = {
    "910b2": {
        "s512": aggregate_profile(*load(reference_s512)),
        "s2048": aggregate_profile(*load(reference_s2048)),
    },
    "310p": {
        "s512": aggregate_profile(*load(target_s512)),
        "s2048": aggregate_profile(*load(target_s2048)),
    },
}


def ratio(numerator, denominator):
    return numerator / denominator


comparison = {}
for device, rows in profiles.items():
    s512 = rows["s512"]
    s2048 = rows["s2048"]
    comparison[device] = {
        "s2048_over_s512_physical_throughput": ratio(
            s2048["control_physical_tok_s"],
            s512["control_physical_tok_s"],
        ),
        "s2048_over_s512_control_time": ratio(
            s2048["control_center_ms"],
            s512["control_center_ms"],
        ),
        "s2048_over_s512_kernel_time": ratio(
            s2048["kernel_ms_per_replay"],
            s512["kernel_ms_per_replay"],
        ),
        "category_time_ratios": {
            category: ratio(
                s2048["categories"][category]["ms_per_replay"],
                s512["categories"][category]["ms_per_replay"],
            )
            for category in s512["categories"]
            if s512["categories"][category]["ms_per_replay"] > 0
        },
    }

cross_device = {}
for length in ("s512", "s2048"):
    reference = profiles["910b2"][length]
    target = profiles["310p"][length]
    cross_device[length] = {
        "910b2_over_310p_physical_throughput": ratio(
            reference["control_physical_tok_s"],
            target["control_physical_tok_s"],
        ),
        "310p_over_910b2_control_time": ratio(
            target["control_center_ms"],
            reference["control_center_ms"],
        ),
        "310p_over_910b2_kernel_time": ratio(
            target["kernel_ms_per_replay"],
            reference["kernel_ms_per_replay"],
        ),
    }

payload = {
    "metric": "raw physical VisionPrefillStage tokens/s",
    "profiles": profiles,
    "within_device_scaling": comparison,
    "cross_device": cross_device,
}
output_json.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# B1 compiled PromptFA profiler comparison",
    "",
    "Throughput uses the center of pre/post unprofiled NPU-event medians.",
    "Profiled values are diagnostic and include native-profiler perturbation.",
    "",
    "## Throughput and profiler overhead",
    "",
    "| Device | Shape | control ms | physical tok/s | profiled ms | "
    "profiler overhead | pre/post drift |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for device in ("910b2", "310p"):
    for length in ("s512", "s2048"):
        row = profiles[device][length]
        lines.append(
            f"| {device} | {length} | {row['control_center_ms']:.6f} | "
            f"{row['control_physical_tok_s']:.1f} | "
            f"{row['profiled_ms']:.6f} | "
            f"{row['profiler_slowdown_pct']:.3f}% | "
            f"{row['pre_post_drift_pct']:.3f}% |"
        )

lines.extend(
    [
        "",
        "## Kernel totals",
        "",
        "| Device | Shape | kernel ms | kernels/replay | AI Core ms | "
        "weighted cube |",
        "|---|---|---:|---:|---:|---:|",
    ]
)
for device in ("910b2", "310p"):
    for length in ("s512", "s2048"):
        row = profiles[device][length]
        lines.append(
            f"| {device} | {length} | "
            f"{row['kernel_ms_per_replay']:.6f} | "
            f"{row['kernels_per_replay']:.1f} | "
            f"{row['aicore_ms_per_replay']:.6f} | "
            f"{row['weighted_cube_utilization_pct']:.3f}% |"
        )

for device in ("910b2", "310p"):
    lines.extend(
        [
            "",
            f"## {device} kernel-family mix",
            "",
            "| Family | S512 ms / share | S2048 ms / share | "
            "S2048/S512 time |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in profiles[device]["s512"]["categories"]:
        small = profiles[device]["s512"]["categories"][category]
        large = profiles[device]["s2048"]["categories"][category]
        time_ratio = (
            large["ms_per_replay"] / small["ms_per_replay"]
            if small["ms_per_replay"] > 0
            else float("nan")
        )
        lines.append(
            f"| {category} | {small['ms_per_replay']:.6f} / "
            f"{small['share_pct']:.3f}% | "
            f"{large['ms_per_replay']:.6f} / "
            f"{large['share_pct']:.3f}% | {time_ratio:.3f}x |"
        )

lines.extend(["", "## Scaling summary", ""])
for device, values in comparison.items():
    lines.append(
        f"- {device} S2048/S512 physical throughput: "
        f"{values['s2048_over_s512_physical_throughput']:.4f}x"
    )
    lines.append(
        f"- {device} S2048/S512 control time: "
        f"{values['s2048_over_s512_control_time']:.4f}x"
    )
for length, values in cross_device.items():
    lines.append(
        f"- {length} 910B2/310P physical throughput: "
        f"{values['910b2_over_310p_physical_throughput']:.4f}x"
    )

output_md.write_text(
    "\n".join(lines).rstrip() + "\n",
    encoding="utf-8",
)
print("PHASE10_COMPARISON: PASS")
PY

cat "$PHASE10_ROOT/profile_comparison.md"
```

### 10.7 Interpretation rules

Use these rules in the report:

- Compare devices with `control_physical_tok_s`, never profiled wall time.
- Report profiler slowdown separately for every shape and device. Do not
  assume the S512 overhead also applies to S2048.
- S2048 contains 4x as many physical tokens as S512. Report both the replay
  time ratio and throughput ratio.
- PromptFA's attention work grows faster than linearly with sequence length.
  Report the measured PromptFA time ratio and share shift; do not explain the
  whole graph using token count alone.
- Aggregate `MatMulV2`, `MatMulV3`, and any other `MatMul*` type. A GE kernel
  selection change is evidence, not a missing operation.
- Kernel count per replay indicates graph fragmentation. If it is unchanged
  while tokens grow 4x, say that fixed per-kernel overhead is better amortized;
  do not call that proof of launch-bound execution by itself.
- Weighted cube utilization applies only to cube-capable profiled kernels. It
  does not measure vector kernels, data movement, or whole-graph occupancy.
- `aicore_time` and vector-core time can overlap inside mixed kernels such as
  PromptFA; do not sum them as mutually exclusive wall time.
- One-second `npu-smi` utilization cannot resolve a 16-100 ms graph replay.
- If 310P uses a materially different kernel decomposition, report the exact
  operator types and shapes instead of forcing the 910B2 interpretation.
- Do not optimize or patch anything during this phase. The goal is diagnosis.

The central questions to answer are:

1. What is exact 310P physical tok/s at S512 and S2048?
2. Does S2048 approximately double throughput as on 910B2, or scale
   differently?
3. Which kernel families grow sublinearly, linearly, or superlinearly?
4. Does PromptFA become a much larger fraction at S2048?
5. Are 310P matmuls well utilized while layout/rotary/padding dominates, or is
   the bottleneck qualitatively different from 910B2?
6. Is the 910B2/310P gap smaller, equal, or larger at S2048 than S512?

### 10.8 Stop, report, and preserve evidence

Stop after the comparison. Do not run more sequence lengths, more profiler
metrics, model pages, eager attention, batch sizes above one, or optimization
experiments.

Write:

```text
$PHASE10_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P B1 VISION PROFILER: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical 310P:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver route:
CANN / driver / firmware:
Recognizer model:

Measurement boundary:
attention / execution / dtype:
outside warmup / control repeats / profiler warmup / active steps:
profiler level / metric:
included:
excluded:

Cache preparation:
S512 cache existed before:
S2048 cache existed before:
S2048 preparation setup seconds:
new graph count:
compilation during final profiles: MUST BE ZERO

S512:
pre-control median ms / physical tok/s:
profiled median ms / physical tok/s:
post-control median ms / physical tok/s:
control-center median ms / physical tok/s:
profiler overhead:
pre/post drift:
kernel ms / kernels per replay:
AI Core ms / weighted cube:
top kernel families with ms and share:

S2048:
pre-control median ms / physical tok/s:
profiled median ms / physical tok/s:
post-control median ms / physical tok/s:
control-center median ms / physical tok/s:
profiler overhead:
pre/post drift:
kernel ms / kernels per replay:
AI Core ms / weighted cube:
top kernel families with ms and share:

310P S2048/S512:
replay-time ratio:
physical-throughput ratio:
kernel-time ratio:
PromptFA time ratio / share change:
all-MatMul time ratio / share change:
slice / transpose / pad time ratios:
kernel-count change:

910B2 reference:
S512 control ms / physical tok/s:
S2048 control ms / physical tok/s:
S2048/S512 physical-throughput ratio:

Shape-for-shape:
S512 910B2/310P throughput ratio:
S2048 910B2/310P throughput ratio:
does the platform gap change with sequence length:

Conclusion:
dominant S512 bottleneck on 310P:
dominant S2048 bottleneck on 310P:
same or different from 910B2:
is short-sequence B1 underfilled:
single best next profiling/optimization direction, without implementing it:

First blocker or warning:
Exact command records:
Summary artifact paths:
Raw profiler paths:
```

Keep the raw native profiler trees under `.runtime_cache`; do not add them to
Git. Preserve and commit only the compact Phase 10 evidence:

```sh
git add -f -- \
  "$PHASE10_REL/environment_preflight.txt" \
  "$PHASE10_REL/disk_before.txt" \
  "$PHASE10_REL/npu_before.txt" \
  "$PHASE10_REL/provenance.txt" \
  "$PHASE10_REL/cache_validation.txt" \
  "$PHASE10_REL/profile_validation.txt" \
  "$PHASE10_REL/profile_comparison.json" \
  "$PHASE10_REL/profile_comparison.md" \
  "$PHASE10_REL/agent_report.md" \
  "$PHASE10_REL/results/cache_prepare_s512/cache_preparation.json" \
  "$PHASE10_REL/results/cache_prepare_s2048/cache_preparation.json" \
  "$PHASE10_REL/results/profile_s512/run_summary.json" \
  "$PHASE10_REL/results/profile_s512/parsed_profile_summary.json" \
  "$PHASE10_REL/results/profile_s512/parsed_profile_summary.md" \
  "$PHASE10_REL/results/profile_s2048/run_summary.json" \
  "$PHASE10_REL/results/profile_s2048/parsed_profile_summary.json" \
  "$PHASE10_REL/results/profile_s2048/parsed_profile_summary.md" \
  "$PHASE10_REL/evidence"

git diff --cached --check
git diff --cached --stat
```

Inspect the staged paths before committing. Ensure `.runtime_cache` and raw
`trace_view.json` data are not staged. Then commit and push the compact
evidence on `main`, and report the commit hash.

## Phase 11: PromptFA Linear-weight format and MLP-alignment matrix

### 11.0 Scope and exact experiment

This is a retained historical phase. Do not execute it for the current
Phase 14 request. Its original matrix used these three production
`VisionPrefillStage` shapes:

```text
B1xS512
B4xS512
B1xS2048
```

For each shape, run exactly four compiled variants:

```text
native 4304-wide MLP
FRACTAL_NZ 4304-wide MLP
native 4352-wide zero-extended MLP
FRACTAL_NZ 4352-wide zero-extended MLP
```

That is exactly 12 graph cases. Do not add sequence lengths, batch sizes,
attention implementations, dtypes, eager performance lanes, quantization, or
full-page OCR.

The timed boundary is the real production `VisionPrefillStage`:

```text
27 x (
  LayerNorm1 + Q/K/V + RoPE +
  npu_prompt_flash_attention + output projection + residual +
  LayerNorm2 + FC1/GELU/FC2 + residual
) + post-LayerNorm
```

It uses real PromptFA with the production head-dimension padding from 72 to 80.
It excludes patch embedding, projector, image/layout work, text prefill, and
decode. Inputs are synthetic but have exact production shapes and dtypes.

Physical throughput is:

```text
physical tokens per replay = batch_size * sequence_length
physical tok/s = physical tokens per replay / NPU-event replay time
```

Thus B4xS512 and B1xS2048 both process 2048 physical tokens per replay, but
their attention work is not equivalent: B4xS512 performs four independent
512-token attentions.

The 4352 variant does not change model semantics. Each layer's FC1 is extended
from `[4304, 1152]` to `[4352, 1152]` with 48 zero rows; FC2 is extended from
`[1152, 4304]` to `[1152, 4352]` with 48 zero columns. GELU(0)=0.

The FRACTAL_NZ variant must:

1. set `torch.npu.config.allow_internal_format = True` before the first NPU
   allocation;
2. load/pad the model;
3. explicitly cast all 162 vision-stage Linear weights to format code 29;
4. verify all 162 weights are actually format 29;
5. never silently time an ND fallback.

The committed 910B2 references below are comparison data only. Do not use
them as expected 310P performance:

| Shape | MLP | weights | 910B2 median ms | 910B2 physical tok/s | V2 / V3 |
|---|---:|---|---:|---:|---:|
| B1xS512 | 4304 | native ND | 15.7241 | 32,561.5 | 486 / 0 |
| B1xS512 | 4304 | FRACTAL_NZ | 14.1214 | 36,257.0 | 486 / 0 |
| B1xS512 | 4352 | native ND | 14.0940 | 36,327.6 | 486 / 0 |
| B1xS512 | 4352 | FRACTAL_NZ | 13.5761 | 37,713.2 | 486 / 0 |
| B4xS512 | 4304 | native ND | 27.3093 | 74,992.7 | 324 / 162 |
| B4xS512 | 4304 | FRACTAL_NZ | 27.1850 | 75,335.6 | 486 / 0 |
| B4xS512 | 4352 | native ND | 26.1087 | 78,441.4 | 324 / 162 |
| B4xS512 | 4352 | FRACTAL_NZ | 26.6486 | 76,852.0 | 486 / 0 |
| B1xS2048 | 4304 | native ND | 31.4757 | 65,066.0 | 324 / 162 |
| B1xS2048 | 4304 | FRACTAL_NZ | 31.2482 | 65,539.8 | 486 / 0 |
| B1xS2048 | 4352 | native ND | 30.1408 | 67,947.7 | 324 / 162 |
| B1xS2048 | 4352 | FRACTAL_NZ | 30.5553 | 67,026.1 | 486 / 0 |

Read the committed JSON mechanically for the final comparison; the rounded
table is only a human sanity check.

### 11.1 Pull and recover the proven environment

Start from the repository. Do not discard or overwrite any local work:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE11_REL="tmp/09_persistent_page_engine/310p_promptfa_formats_$COMMIT_SHORT"
PHASE11_ROOT="$REPO/$PHASE11_REL"
RAW_PROFILE_ROOT="$REPO/.runtime_cache/310p_promptfa_formats_$COMMIT_SHORT"
MATRIX_SCRIPT="$REPO/09_persistent_page_engine/scripts/run_vision_matmul_lab_matrix.py"
LAB_SCRIPT="$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py"
REFERENCE_B1="$REPO/tmp/09_persistent_page_engine/vision_matmul_lab/910b_promptfa_internal_formats_e447c8e"
REFERENCE_B4="$REPO/tmp/09_persistent_page_engine/vision_matmul_lab/910b_promptfa_b4s512_internal_formats_16dac71"

test -f "$MATRIX_SCRIPT"
test -f "$LAB_SCRIPT"
test -f "$REFERENCE_B1/matrix_summary.json"
test -f "$REFERENCE_B4/matrix_summary.json"
test ! -e "$PHASE11_ROOT"
test ! -e "$RAW_PROFILE_ROOT"
mkdir -p "$PHASE11_ROOT" "$RAW_PROFILE_ROOT"
```

Recover the exact Python and recognizer model from the latest retained
successful Phase 7 command. Do not guess a virtual environment or model path:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
test -f "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])

def option(name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc

print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
printf 'python=%s\nmodel=%s\n' "$PYTHON_BIN" "$RECOGNIZER_MODEL"
```

Activate the exact CANN/torch-npu environment used by the successful previous
phases. Keep one free physical 310P exposed as logical `npu:0`. Never terminate
another user's process. Stop if no NPU is free.

Record the environment before running:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'reference_b1=%s\n' "$REFERENCE_B1"
  printf 'reference_b4=%s\n' "$REFERENCE_B4"
} >"$PHASE11_ROOT/provenance.txt"

df -h "$REPO" "$REPO/.runtime_cache" >"$PHASE11_ROOT/disk_before.txt"
npu-smi info >"$PHASE11_ROOT/npu_before.txt" 2>&1
```

### 11.2 Required internal-format preflight

Run this exact probe before compiling graphs:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" - >"$PHASE11_ROOT/internal_format_preflight.txt" 2>&1 <<'PY'
import json
import platform
import sys

import torch
import torch_npu
import torch.nn.functional as F

from paddleocr_vl.model.compile_utils import import_torchair

torchair, CompilerConfig = import_torchair()
assert callable(torchair.inference.cache_compile)
assert torch.npu.is_available()

# This must happen before torch.npu.set_device or any NPU tensor allocation.
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)

x = torch.randn((512, 1152), dtype=torch.float16, device="npu:0")
w_nd = torch.randn((4304, 1152), dtype=torch.float16, device="npu:0")
before = int(torch_npu.get_npu_format(w_nd))
w_nz = torch_npu.npu_format_cast(w_nd, 29)
after = int(torch_npu.get_npu_format(w_nz))
y = F.linear(x, w_nz)
torch.npu.synchronize()

assert before == 2, before
assert after == 29, after
assert tuple(y.shape) == (512, 4304)
assert bool(torch.isfinite(y.float()).all().cpu().item())

print("platform:", platform.platform())
print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torchair_module:", torchair.__name__)
print("torchair_file:", getattr(torchair, "__file__", "<namespace>"))
print("npu_name:", torch.npu.get_device_name(0))
print("mm_bmm_format_nd:", torch.npu.get_mm_bmm_format_nd())
print("weight_format_before:", before)
print("weight_format_after:", after)
print("output_shape:", list(y.shape))
print("PHASE11_INTERNAL_FORMAT: PASS")
PY

cat "$PHASE11_ROOT/internal_format_preflight.txt"
test "$(tail -n 1 "$PHASE11_ROOT/internal_format_preflight.txt")" = \
  "PHASE11_INTERNAL_FORMAT: PASS"
```

If format code 29 is not observed, stop and report the exact warning and first
causal traceback. Do not remove the NZ lanes, set a private environment
variable, or relabel ND as NZ.

### 11.3 Run the twelve-case compiled PromptFA matrix

The runner creates a distinct persistent cache key for every batch, sequence,
MLP width, and weight format. Run all cases serially in one command:

```sh
{
  printf '#!/usr/bin/env bash\n'
  printf '# git_commit=%s\n' "$COMMIT"
  printf '# hostname=%s\n' "$(hostname)"
  printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf '%q ' \
    "$PYTHON_BIN" "$MATRIX_SCRIPT" \
    --name "310p_promptfa_internal_formats_$COMMIT_SHORT" \
    --model "$RECOGNIZER_MODEL" \
    --output-root "$PHASE11_ROOT/results" \
    --cache-dir "$RAW_PROFILE_ROOT/graphs" \
    --profile-root "$RAW_PROFILE_ROOT/profiles" \
    --execution torchair \
    --allow-compile-if-missing \
    --profile \
    --warmup 3 \
    --samples 10 \
    --calls-per-sample 5
  printf '\n'
} >"$PHASE11_ROOT/command.sh"
chmod +x "$PHASE11_ROOT/command.sh"

(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$PHASE11_ROOT/npu_smi_1s.log" 2>&1 &
MONITOR_PID=$!

set +e
"$PHASE11_ROOT/command.sh" >"$PHASE11_ROOT/run.log" 2>&1
STATUS=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$STATUS" >"$PHASE11_ROOT/exit_code.txt"
test "$STATUS" -eq 0
```

Do not run cases concurrently. Do not point this experiment at a production
vision cache. Do not rerun a failure into the same output/cache directories.
If one case fails, preserve everything and report the failed case plus the
first causal error.

The authoritative matrix should be:

```sh
MATRIX_ROOT="$PHASE11_ROOT/results/310p_promptfa_internal_formats_$COMMIT_SHORT"
MATRIX_JSON="$MATRIX_ROOT/matrix_summary.json"
test -f "$MATRIX_JSON"
```

### 11.4 Validate format, shape, PromptFA, and measurement contracts

```sh
"$PYTHON_BIN" - \
  "$MATRIX_JSON" \
  "$PHASE11_ROOT/matrix_validation.txt" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

matrix_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
payload = json.loads(matrix_path.read_text(encoding="utf-8"))

assert payload["status"] == "completed"
assert payload["execution"] == "torchair"
assert payload["profiled"] is True
assert payload["shapes"] == ["b1s512", "b4s512", "b1s2048"]
assert len(payload["cases"]) == 12

shape_counts = Counter()
rows = []
for case in payload["cases"]:
    assert case["exit_code"] == 0, case["case"]
    result = case["result"]
    assert result["status"] == "completed", case["case"]
    shape = result["shape"]
    key = (shape["batch_size"], shape["sequence_length"])
    shape_counts[key] += 1
    assert shape["physical_tokens_per_call"] == key[0] * key[1]
    assert shape["candidate_intermediate_size"] in (4304, 4352)
    assert result["numerics"]["measured_output_finite"] is True
    assert result["numerics"]["raw_candidate_vs_native_4304"][
        "left_finite"
    ]
    assert result["numerics"]["raw_candidate_vs_native_4304"][
        "right_finite"
    ]

    requested = result["requested"]["weight_format"]
    formats = result["weight_format"]["after_format_histogram"]
    if requested == "fractal_nz":
        assert formats == {"29": 162}, (case["case"], formats)
        assert result["weight_format"]["all_after_are_nz"] is True
        assert result["weight_format"]["runtime_gate"][
            "torch_npu_allow_internal_format_enabled_before_npu_allocation"
        ] is True
    else:
        assert formats == {"2": 162}, (case["case"], formats)

    dispatch = result["dispatch"]["counts"]
    rows.append(
        (
            case["case"],
            result["device_median_ms"],
            result["physical_tokens_per_s"],
            dispatch.get("MatMulV2", 0),
            dispatch.get("MatMulV3", 0),
            result["transdata"]["count"],
            result["numerics"]["measured_output_vs_raw_candidate"][
                "mean_abs"
            ],
            result["numerics"]["measured_output_vs_raw_candidate"][
                "max_abs"
            ],
        )
    )

assert shape_counts == {(1, 512): 4, (4, 512): 4, (1, 2048): 4}
lines = [
    "case | median_ms | physical_tok_s | MatMulV2 | MatMulV3 | "
    "TransData | compiled_mean_abs | compiled_max_abs"
]
lines.extend(" | ".join(map(str, row)) for row in rows)
lines.append("PHASE11_MATRIX_CONTRACTS: PASS")
output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output_path.read_text(encoding="utf-8"), end="")
PY
```

Do not impose a small max-absolute correctness gate. The vision output is an
intermediate decoder input, and compiled numerical noise may have isolated
larger maxima. Report mean/max absolute error, require finite values, and flag
only a qualitatively large mean error or non-finite output.

### 11.5 Generate the exact 910B2-versus-310P comparison

```sh
"$PYTHON_BIN" - \
  "$REFERENCE_B1/matrix_summary.json" \
  "$REFERENCE_B4/matrix_summary.json" \
  "$MATRIX_JSON" \
  "$PHASE11_ROOT/comparison.json" \
  "$PHASE11_ROOT/comparison.md" <<'PY'
import json
import sys
from pathlib import Path

ref_b1_path, ref_b4_path, target_path, out_json, out_md = map(
    Path, sys.argv[1:]
)

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def keyed(*matrices):
    rows = {}
    for matrix in matrices:
        for case in matrix["cases"]:
            result = case["result"]
            shape = result["shape"]
            key = (
                int(shape["batch_size"]),
                int(shape["sequence_length"]),
                int(shape["candidate_intermediate_size"]),
                str(result["requested"]["weight_format"]),
            )
            rows[key] = {
                "case": case["case"],
                "median_ms": float(result["device_median_ms"]),
                "physical_tok_s": float(result["physical_tokens_per_s"]),
                "dispatch": result["dispatch"]["counts"],
                "matmul_duration_us": result["dispatch"]["duration_us"],
                "transdata_count": int(result["transdata"]["count"]),
                "format_histogram": result["weight_format"][
                    "after_format_histogram"
                ],
            }
    return rows

reference = keyed(load(ref_b1_path), load(ref_b4_path))
target = keyed(load(target_path))
assert set(reference) == set(target), (
    sorted(set(reference) - set(target)),
    sorted(set(target) - set(reference)),
)

rows = []
for key in sorted(reference):
    ref = reference[key]
    current = target[key]
    rows.append(
        {
            "batch_size": key[0],
            "sequence_length": key[1],
            "intermediate_size": key[2],
            "weight_format": key[3],
            "910b2": ref,
            "310p": current,
            "910b2_over_310p_physical_tok_s": (
                ref["physical_tok_s"] / current["physical_tok_s"]
            ),
        }
    )

def lookup(device_rows, batch, seq, intermediate, weight_format):
    return device_rows[(batch, seq, intermediate, weight_format)]

effects = {}
for name, device_rows in (("910b2", reference), ("310p", target)):
    effects[name] = {}
    for batch, seq in ((1, 512), (4, 512), (1, 2048)):
        baseline = lookup(device_rows, batch, seq, 4304, "native")
        effects[name][f"b{batch}s{seq}"] = {
            "nz_4304_speedup": (
                lookup(
                    device_rows, batch, seq, 4304, "fractal_nz"
                )["physical_tok_s"]
                / baseline["physical_tok_s"]
            ),
            "aligned_4352_native_speedup": (
                lookup(device_rows, batch, seq, 4352, "native")[
                    "physical_tok_s"
                ]
                / baseline["physical_tok_s"]
            ),
            "aligned_4352_nz_speedup": (
                lookup(
                    device_rows, batch, seq, 4352, "fractal_nz"
                )["physical_tok_s"]
                / baseline["physical_tok_s"]
            ),
        }

payload = {"rows": rows, "within_device_effects": effects}
out_json.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# Production-PromptFA vision format/alignment comparison",
    "",
    "| Shape | MLP | weights | 910B2 tok/s | 310P tok/s | "
    "910B2/310P | 310P V2/V3 |",
    "|---|---:|---|---:|---:|---:|---:|",
]
for row in rows:
    shape = f"B{row['batch_size']}xS{row['sequence_length']}"
    dispatch = row["310p"]["dispatch"]
    lines.append(
        f"| {shape} | {row['intermediate_size']} | "
        f"{row['weight_format']} | "
        f"{row['910b2']['physical_tok_s']:.1f} | "
        f"{row['310p']['physical_tok_s']:.1f} | "
        f"{row['910b2_over_310p_physical_tok_s']:.3f}x | "
        f"{dispatch.get('MatMulV2', 0)}/"
        f"{dispatch.get('MatMulV3', 0)} |"
    )

lines.extend(["", "## Within-device speedups over 4304 native", ""])
for device in ("910b2", "310p"):
    lines.append(f"### {device}")
    for shape, values in effects[device].items():
        lines.append(
            f"- {shape}: 4304 NZ {values['nz_4304_speedup']:.4f}x; "
            f"4352 ND {values['aligned_4352_native_speedup']:.4f}x; "
            f"4352 NZ {values['aligned_4352_nz_speedup']:.4f}x"
        )
    lines.append("")

out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"), end="")
PY
```

### 11.6 Interpretation rules and required report

Use the following rules:

- Headline only unprofiled NPU-event physical tok/s.
- Profiler output diagnoses dispatch and formats; it is not the throughput
  timing.
- Confirm the profiler's MatMul signatures show the requested weight format.
- Aggregate all MatMul variants, but separately report V2/V3 counts and times.
- Report TransData even if it is zero.
- Do not assume that NZ must win. On 910B2, NZ helps B1xS512 but suppresses
  profitable MatMulV3 dispatch at the two larger flattened workloads.
- 310P was previously observed to remain on MatMulV2. If Phase 11 confirms
  that, explain whether NZ then helps consistently.
- Compare B4xS512 with both B1xS512 and B1xS2048. The former measures batching
  scaling; the latter holds physical tokens per call equal while changing
  attention decomposition.
- Do not modify production routing or integrate a winner during this phase.

Write:

```text
$PHASE11_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P PROMPTFA INTERNAL-FORMAT MATRIX: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical 310P:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver:
CANN / driver / firmware:
Recognizer model:

Boundary:
attention / layout / head-dim padding:
dtype:
warmup / samples / calls per sample:
physical-token definition:
included:
excluded:

Internal-format preflight:
ND format code:
NZ format code:
F.linear finite:
allow_internal_format set before first NPU allocation:

Matrix completion:
completed cases / expected cases:
new graph count:
failed or retried cases:
compilation during timing:

Results table:
shape | MLP width | weight format | median graph ms | physical tok/s |
MatMulV2 count/time | MatMulV3 count/time | TransData count/time |
mean/max compiled-vs-eager difference

Within 310P:
B1xS512 NZ speedup at 4304:
B1xS512 4352-ND speedup:
B1xS512 4352-NZ speedup:
B4xS512 NZ speedup at 4304:
B4xS512 4352-ND speedup:
B4xS512 4352-NZ speedup:
B1xS2048 NZ speedup at 4304:
B1xS2048 4352-ND speedup:
B1xS2048 4352-NZ speedup:
B4xS512 / B1xS512 physical-throughput scaling:
B4xS512 / B1xS2048 comparison at equal 2048 physical tokens:

Versus 910B2:
complete twelve-row comparison table:
shape-by-shape throughput ratios:
does 310P ever dispatch MatMulV3:
does NZ help 310P more consistently than 910B2:
does 4352 alignment help independently of NZ:

Conclusion:
best 310P configuration for each shape:
single global weight-format recommendation, if supported by all shapes:
whether a sequence-dependent policy would require duplicated weights:
single best next production experiment, without implementing it:

First blocker or warning:
Exact command:
Matrix JSON:
Comparison JSON / Markdown:
Raw profiler root:
All artifact paths:
```

The work server is pull-only. Do not commit or push from it. Keep the raw graph
caches and profiler trees under `.runtime_cache`, keep compact results under
`tmp/`, and send Luka the report plus exact artifact paths manually.

## Phase 12: full-stack joint-QK manual RoPE A/B

### 12.0 Scope, controlled variables, and reference

This is a retained historical phase. Do not execute it for the current
Phase 14 request. Its original comparison used exactly two compiled variants:

```text
control:   --rotary-implementation separate_manual
candidate: --rotary-implementation joint_manual
```

Every other argument must be identical:

```text
batch size:                  1
sequence length:             2048
vision layers:               all 27
vision hidden size:          1152
vision MLP width:            4352 (zero-extended from 4304)
attention head padding:      weights (D72 -> D80 once in weights)
Linear weight format:        FRACTAL_NZ
attention:                   real PromptFlashAttention
execution:                   TorchAir cache_compile
dtype:                       fp16
warmup:                      3 complete replays
measurement:                 10 samples x 5 complete replays
profile:                     1 warmup + 3 active complete replays
physical tokens per replay:  2048
```

The timed boundary is the complete compiled vision-transformer stack:

```text
27 x (
  LayerNorm1 + Q/K/V + FP32 RoPE +
  npu_prompt_flash_attention + output projection + residual +
  LayerNorm2 + FC1/GELU/FC2 + residual
) + post-LayerNorm
```

It excludes model loading, weight preparation, graph compilation, patch
embedding, projector, layout, text prefill, and decode. Do not report setup,
first-call, or profiler wall time as replay latency.

The candidate does not change the math. It concatenates Q and K, applies the
same existing FP32 half-RoPE formula once, and performs one combined Q/K
layout conversion. It should reduce RoPE-related slicing, arithmetic, concat,
transpose, and split work while leaving all PromptFA and Linear work
unchanged.

The committed 910B2 full-stack reference used the same B1xS2048, 27-layer,
4352-wide, D80 weight-padded PromptFA graph, but native ND Linear weights:

| 910B2 lane | median ms | physical tok/s |
|---|---:|---:|
| separate manual, warm control | 25.5523 | 80,149.2 |
| joint-QK manual | 24.3146 | 84,229.1 |

On 910B2, joint manual reduced latency by 4.84% and increased physical
throughput by 5.09%. Its raw D80 output was exactly equal to the separate
manual D80 reference. These numbers are context, not an expected 310P
result. The 310P A/B intentionally uses FRACTAL_NZ because that is the
production-relevant 310P weight representation; judge the optimization by
the within-310P A/B. FRACTAL_NZ may select a different eager MatMul kernel
than the native-D80 reference, so on 310P use the separate lane's
raw-versus-reference error as the calibrated numerical floor rather than
requiring zero error from either NZ lane.

### 12.1 Pull and recover the proven environment

Start from the repository. Do not discard, overwrite, stash, or clean any
local work:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE12_REL="tmp/09_persistent_page_engine/310p_joint_rope_$COMMIT_SHORT"
PHASE12_ROOT="$REPO/$PHASE12_REL"
CACHE_ROOT="$REPO/.runtime_cache/310p_joint_rope_$COMMIT_SHORT"
LAB_SCRIPT="$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py"

test -f "$LAB_SCRIPT"
test ! -e "$PHASE12_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$PHASE12_ROOT" "$CACHE_ROOT/graphs" "$CACHE_ROOT/profiles"
```

Recover the exact Python and recognizer model from the latest retained
successful Phase 7 command. Do not guess a venv or model path:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
test -f "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])

def option(name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc

print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
printf 'python=%s\nmodel=%s\n' "$PYTHON_BIN" "$RECOGNIZER_MODEL"
```

Activate the exact CANN/torch-npu environment used by the successful previous
phases. Keep one free physical 310P exposed as logical `npu:0`. Never
terminate another user's process. Stop if no NPU is free.

Record the environment and check disk before compiling:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'lab_script=%s\n' "$LAB_SCRIPT"
} >"$PHASE12_ROOT/provenance.txt"

df -h "$REPO" "$REPO/.runtime_cache" >"$PHASE12_ROOT/disk_before.txt"
npu-smi info >"$PHASE12_ROOT/npu_before.txt" 2>&1

CACHE_FREE_KIB="$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')"
test -n "$CACHE_FREE_KIB"
if test "$CACHE_FREE_KIB" -lt 20971520; then
  printf 'insufficient cache free space: %s KiB\n' "$CACHE_FREE_KIB" \
    >"$PHASE12_ROOT/disk_blocker.txt"
  exit 1
fi

"$PYTHON_BIN" "$LAB_SCRIPT" --help \
  >"$PHASE12_ROOT/lab_help.txt" 2>&1
grep -q 'joint_manual' "$PHASE12_ROOT/lab_help.txt"
grep -q 'attention-head-padding' "$PHASE12_ROOT/lab_help.txt"
```

If the cache filesystem has less than 20 GiB free, stop and report that
before compiling. Do not redirect compiler state to an unrecorded location.

### 12.2 Narrow TorchAir, PromptFA, and FRACTAL_NZ preflight

This phase relies on the already-proven Phase 11 mechanism, but verify that
the current shell exposes the required facilities before loading the full
model:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" - >"$PHASE12_ROOT/preflight.txt" 2>&1 <<'PY'
import platform
import sys

import torch
import torch_npu

from paddleocr_vl.model.compile_utils import import_torchair

torchair, CompilerConfig = import_torchair()
assert callable(torchair.inference.cache_compile)
assert callable(torch_npu.npu_prompt_flash_attention)
assert torch.npu.is_available()

# Must be set before the first NPU allocation.
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)

w_nd = torch.randn((1280, 1152), dtype=torch.float16, device="npu:0")
w_nz = torch_npu.npu_format_cast(w_nd, 29)
torch.npu.synchronize()
assert int(torch_npu.get_npu_format(w_nd)) == 2
assert int(torch_npu.get_npu_format(w_nz)) == 29

print("platform:", platform.platform())
print("python:", sys.version.replace("\n", " "))
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "<missing>"))
print("torchair_module:", torchair.__name__)
print("torchair_file:", getattr(torchair, "__file__", "<namespace>"))
print("npu_name:", torch.npu.get_device_name(0))
print("weight_format_nd:", int(torch_npu.get_npu_format(w_nd)))
print("weight_format_nz:", int(torch_npu.get_npu_format(w_nz)))
print("PHASE12_PREFLIGHT: PASS")
PY

cat "$PHASE12_ROOT/preflight.txt"
test "$(tail -n 1 "$PHASE12_ROOT/preflight.txt")" = \
  "PHASE12_PREFLIGHT: PASS"
```

If format code 29 is unavailable, PromptFA is unavailable, or TorchAir does
not resolve, stop and report the first causal traceback. Do not fall back to
native ND, manual attention, eager execution, or a different Python.

### 12.3 Run the two full-stage graphs serially

Write exact replayable commands:

```sh
write_case_command() {
  case_name="$1"
  rotary="$2"
  case_root="$PHASE12_ROOT/$case_name"
  mkdir -p "$case_root"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# git_commit=%s\n' "$COMMIT"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' \
      "$PYTHON_BIN" "$LAB_SCRIPT" \
      --batch-size 1 \
      --sequence-length 2048 \
      --intermediate-size 4352 \
      --weight-format fractal_nz \
      --attention-head-padding weights \
      --rotary-implementation "$rotary" \
      --execution torchair \
      --model "$RECOGNIZER_MODEL" \
      --cache-dir "$CACHE_ROOT/graphs" \
      --output-dir "$case_root/result" \
      --profile-dir "$CACHE_ROOT/profiles/$case_name" \
      --allow-compile-if-missing \
      --warmup 3 \
      --samples 10 \
      --calls-per-sample 5 \
      --profile \
      --profile-warmup-steps 1 \
      --profile-steps 3 \
      --parser-topn 200
    printf '\n'
  } >"$case_root/command.sh"
  chmod +x "$case_root/command.sh"
}

write_case_command separate_manual separate_manual
write_case_command joint_manual joint_manual
```

Run the control first and candidate second, never concurrently:

```sh
(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$PHASE12_ROOT/npu_smi_1s.log" 2>&1 &
MONITOR_PID=$!

STATUS=0
for case_name in separate_manual joint_manual; do
  case_root="$PHASE12_ROOT/$case_name"
  set +e
  "$case_root/command.sh" >"$case_root/run.log" 2>&1
  CASE_STATUS=$?
  set -e
  printf '%s\n' "$CASE_STATUS" >"$case_root/exit_code.txt"
  if test "$CASE_STATUS" -ne 0; then
    STATUS="$CASE_STATUS"
    break
  fi
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$STATUS" >"$PHASE12_ROOT/exit_code.txt"
test "$STATUS" -eq 0
```

Do not rerun a failed case into the same output, profile, or graph-cache
directory. Preserve it and report the first causal error. Do not run the
candidate if the control fails.

This phase may create at most two new graphs: one control graph and one joint
manual graph. The profiler replays the same graph and must not create another
shape.

### 12.4 Validate the experiment contract and derive the A/B

Run the exact validator/comparison below:

```sh
"$PYTHON_BIN" - \
  "$PHASE12_ROOT/separate_manual/result/run_summary.json" \
  "$PHASE12_ROOT/joint_manual/result/run_summary.json" \
  "$PHASE12_ROOT/comparison.json" \
  "$PHASE12_ROOT/comparison.md" <<'PY'
import json
import sys
from pathlib import Path

control_path, candidate_path, out_json, out_md = map(Path, sys.argv[1:])
control = json.loads(control_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

def validate(name, payload, summary_path):
    assert payload["status"] == "completed", (name, payload["status"])
    shape = payload["shape"]
    assert shape["batch_size"] == 1
    assert shape["sequence_length"] == 2048
    assert shape["physical_tokens_per_call"] == 2048
    assert shape["candidate_intermediate_size"] == 4352
    assert shape["layers"] == 27
    assert shape["linear_calls_per_full_stack"] == 162

    assert payload["requested"]["execution"] == "torchair"
    assert payload["requested"]["weight_format"] == "fractal_nz"
    assert payload["requested"]["attention_head_padding"] == "weights"
    assert payload["attention"]["implementation"] == \
        "prompt_flash_attention"
    assert payload["attention"]["promptfa_call_head_dim"] == 80
    assert payload["weight_format"]["after_format_histogram"] == {"29": 162}
    assert payload["weight_format"]["all_after_are_nz"] is True

    assert payload["measurements"]["samples"] == 10
    assert payload["measurements"]["calls_per_sample"] == 5
    assert payload["measurements"][
        "total_measured_full_stack_calls"
    ] == 50

    numeric = payload["numerics"]
    assert numeric["measured_output_finite"] is True
    raw = numeric["raw_candidate_vs_weight_padded_manual"]
    assert raw["left_finite"] and raw["right_finite"]
    assert raw["same_shape"]

    assert payload["compile"]["api"] == \
        "torchair.inference.cache_compile"
    assert "parsed_profile" in payload
    local_parsed = summary_path.parent / "parsed_profile_summary.json"
    assert local_parsed.is_file(), local_parsed
    return local_parsed

control_profile = validate("separate_manual", control, control_path)
candidate_profile = validate("joint_manual", candidate, candidate_path)
assert control["requested"]["rotary_implementation"] == "separate_manual"
assert candidate["requested"]["rotary_implementation"] == "joint_manual"
control_raw = control["numerics"][
    "raw_candidate_vs_weight_padded_manual"
]
candidate_raw = candidate["numerics"][
    "raw_candidate_vs_weight_padded_manual"
]
assert candidate_raw["mean_abs"] <= max(
    0.05,
    2.0 * control_raw["mean_abs"],
), (control_raw, candidate_raw)

def timing(payload):
    measurements = payload["measurements"]
    return {
        "median_ms": measurements[
            "device_event_per_call_ms"
        ]["median"],
        "mean_ms": measurements[
            "device_event_per_call_ms"
        ]["mean"],
        "p05_ms": measurements[
            "device_event_per_call_ms"
        ]["p05"],
        "p95_ms": measurements[
            "device_event_per_call_ms"
        ]["p95"],
        "physical_tok_s": measurements[
            "physical_tokens_per_s_device_median"
        ],
        "samples_ms": measurements[
            "device_event_per_call_ms"
        ]["samples"],
    }

def profile_types(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 1
    rows = payload["runs"][0]["kernel_details"]["top_kernel_types"]
    result = {}
    for row in rows:
        result[row["name"]] = {
            "count_total": int(row["count"]),
            "count_per_replay": float(row["count"]) / 3.0,
            "duration_us_total": float(row["duration_us"]),
            "duration_us_per_replay": float(row["duration_us"]) / 3.0,
        }
    return result

control_timing = timing(control)
candidate_timing = timing(candidate)
control_types = profile_types(control_profile)
candidate_types = profile_types(candidate_profile)

latency_ratio = candidate_timing["median_ms"] / control_timing["median_ms"]
throughput_ratio = (
    candidate_timing["physical_tok_s"]
    / control_timing["physical_tok_s"]
)

op_names = [
    "PromptFlashAttention",
    "MatMulV2",
    "MatMulV3",
    "StridedSliceD",
    "Transpose",
    "Mul",
    "ConcatV2D",
    "Add",
    "Cast",
    "Neg",
    "SplitVD",
]
op_comparison = {}
for op in op_names:
    empty = {
        "count_total": 0,
        "count_per_replay": 0.0,
        "duration_us_total": 0.0,
        "duration_us_per_replay": 0.0,
    }
    op_comparison[op] = {
        "control": control_types.get(op, empty),
        "joint": candidate_types.get(op, empty),
    }

result = {
    "control": control_timing,
    "joint": candidate_timing,
    "joint_latency_change_pct": (latency_ratio - 1.0) * 100.0,
    "joint_throughput_change_pct": (throughput_ratio - 1.0) * 100.0,
    "raw_vs_native_d80_control": control_raw,
    "raw_vs_native_d80_joint": candidate_raw,
    "numerics_verdict": "within calibrated separate-NZ floor",
    "operator_comparison": op_comparison,
}
out_json.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# 310P full-stack joint-QK manual RoPE A/B",
    "",
    "| lane | median ms | physical tok/s | raw-vs-D80 mean/max abs |",
    "|---|---:|---:|---|",
    (
        f"| separate manual | {control_timing['median_ms']:.4f} | "
        f"{control_timing['physical_tok_s']:.1f} | "
        f"{control_raw['mean_abs']:.6f} / "
        f"{control_raw['max_abs']:.6f} |"
    ),
    (
        f"| joint manual | {candidate_timing['median_ms']:.4f} | "
        f"{candidate_timing['physical_tok_s']:.1f} | "
        f"{candidate_raw['mean_abs']:.6f} / "
        f"{candidate_raw['max_abs']:.6f} |"
    ),
    "",
    (
        "Joint latency change: "
        f"{result['joint_latency_change_pct']:+.3f}%"
    ),
    (
        "Joint physical-throughput change: "
        f"{result['joint_throughput_change_pct']:+.3f}%"
    ),
    "",
    "| kernel type | control count/replay | joint count/replay | "
    "control us/replay | joint us/replay |",
    "|---|---:|---:|---:|---:|",
]
for op, values in op_comparison.items():
    before = values["control"]
    after = values["joint"]
    lines.append(
        f"| {op} | {before['count_per_replay']:.1f} | "
        f"{after['count_per_replay']:.1f} | "
        f"{before['duration_us_per_replay']:.1f} | "
        f"{after['duration_us_per_replay']:.1f} |"
    )
lines.append("")
lines.append("PHASE12_CONTRACTS: PASS")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"), end="")
PY

test "$(tail -n 1 "$PHASE12_ROOT/comparison.md")" = \
  "PHASE12_CONTRACTS: PASS"
```

If the operator names emitted by this 310P stack differ from the names in the
table, do not call the missing rows zero work. Inspect both
`parsed_profile_summary.json` files and report the actual corresponding
kernel names. The validator's correctness and timing contracts remain valid.

### 12.5 Interpretation and required report

Use these rules:

- Headline only unprofiled NPU-event replay time and physical tok/s from the
  ten-by-five measurement. Profiler timing is diagnostic only.
- Treat the within-310P separate-versus-joint ratio as the answer. Do not
  directly compare absolute 310P FRACTAL_NZ throughput with the 910B2 native
  ND throughput as if weight format were controlled.
- Use the separate NZ lane's raw-versus-native-D80 error as the calibrated
  floor. Require finite output and require the joint lane's mean absolute
  error to remain within the validator's generous floor. Do not impose a
  small max-absolute gate. Report both lanes' raw-versus-D80 and
  compiled-versus-raw mean/max errors.
- Confirm PromptFA count and all 162 Linear calls are unchanged. The intended
  win is fewer RoPE/layout operators, not different attention or MatMul work.
- Report all ten per-sample replay values. A median gain smaller than 2% is
  too small to justify production integration from this test; report it as
  inconclusive rather than adding more cases.
- Do not run more shapes. Do not run B4, S512, eager attention, manual
  attention, native ND weights, runtime head padding, or the native in-place
  RoPE operator.
- Do not integrate the candidate into production in this phase.

Write:

```text
$PHASE12_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P FULL-STACK JOINT-QK ROPE: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical 310P:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver:
CANN / driver / firmware:
Recognizer model:

Fixed boundary:
batch / sequence / physical tokens:
layers / hidden / MLP:
head padding / call head dim:
attention:
weight format:
dtype:
warmup / samples / calls per sample:
included:
excluded:

Control separate-manual:
median / mean / p05 / p95 replay ms:
all ten sample values:
physical tok/s:
compile cache / first-call evidence:
raw-vs-D80 mean / max abs:
compiled-vs-raw mean / max abs:

Candidate joint-manual:
median / mean / p05 / p95 replay ms:
all ten sample values:
physical tok/s:
compile cache / first-call evidence:
raw-vs-D80 mean / max abs:
compiled-vs-raw mean / max abs:

Within-310P A/B:
joint latency change:
joint physical-throughput change:

Full-stage profile, counts and duration per replay:
kernel | separate count/time | joint count/time
PromptFlashAttention:
MatMulV2:
MatMulV3:
StridedSlice:
Transpose:
Mul:
Concat:
Add:
Cast:
Neg:
Split:
other materially changed kernels:
total profiled kernel duration:

Structural check:
PromptFA unchanged:
all Linear work unchanged:
RoPE/layout work removed:
unexpected graph changes:

910B2 contextual result:
310P result:
Does portable joint-QK manual help 310P:
Should it proceed to real-crop/E2E production validation:

First blocker or warning:
Exact command records:
Comparison JSON / Markdown:
Parsed profile summaries:
Raw profile roots:
Graph cache:
All artifact paths:
```

## Phase 13: exact B1xS2048 MatMul-only throughput

### 13.0 Purpose and immutable experiment contract

This retained predecessor reproduced the historical 910B2 MatMul-only
calculation on 310P without changing the graph. Do not execute Phase 13;
Phase 14 superseded it with the paired six-lane matrix:

```text
batch:                       1
sequence length:             2048
physical tokens per replay:  2048
vision layers:               all 27
hidden size:                 1152
attention projections:       native 1152 outputs
MLP width:                   native 4304
attention head padding:      runtime D72 -> D80, then D80 -> D72
RoPE:                        separate_manual
Linear weights:              native ND, format code 2
attention:                   real PromptFlashAttention
execution:                   TorchAir cache_compile
dtype:                       fp16
warmup:                      3 complete full-stack replays
unprofiled measurement:      10 samples x 5 full-stack replays
profile:                     1 warmup + 3 active full-stack replays
Linear MatMuls per replay:   27 x 6 = 162
Linear FLOPs per replay:     1,683,744,620,544
```

The timed full-stack boundary is:

```text
27 x (
  LayerNorm1 + Q/K/V + FP32 RoPE +
  npu_prompt_flash_attention + output projection + residual +
  LayerNorm2 + FC1/GELU/FC2 + residual
) + post-LayerNorm
```

Report two different metrics and never conflate them:

```text
full-stage linear-equivalent TFLOP/s
  = 1,683,744,620,544 / unprofiled full-stage replay seconds / 1e12

MatMul-only TFLOP/s
  = 1,683,744,620,544 / summed MatMul kernel seconds per profile replay / 1e12
```

The first denominator includes the whole graph. The second denominator
contains only the 162 MatMul kernels. The profiler is diagnostic and may
perturb whole-graph time, so headline the unprofiled NPU-event measurement for
the full stage and use the profile only for MatMul-only time.

The exact matched 910B2 reference was freshly reproduced on commit `a082212`:

| Metric | 910B2 reference |
|---|---:|
| full-stage median | 31.535596 ms |
| physical throughput | 64,942.487 tok/s |
| full-stage linear-equivalent | 53.391876 TFLOP/s |
| MatMulV2 time per replay | 2.591413 ms |
| MatMulV3 time per replay | 5.352627 ms |
| total MatMul time per replay | 7.944040 ms |
| MatMul-only throughput | 211.950673 TFLOP/s |
| MatMul-only efficiency vs 280-TFLOP peak | 75.6967% |
| MatMul share of full-stage median | 25.1907% |

Those are comparison data, not expected 310P results. For a full physical
310P, use 70 FP16 TFLOP/s as the published peak denominator and explicitly
report if the server exposes a virtual slice instead of a full device.

Do not run B4, S512, 4352-wide MLP, FRACTAL_NZ, weight-padded attention,
joint RoPE, eager execution, manual attention, isolated single-Linears, or
page OCR. This phase is one compiled graph and one profiler capture.

### 13.1 Pull, recover the proven environment, and create evidence roots

Start from the work-server checkout. Do not discard, overwrite, stash, or
clean any local work:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE13_REL="tmp/09_persistent_page_engine/310p_matmul_only_$COMMIT_SHORT"
PHASE13_ROOT="$REPO/$PHASE13_REL"
CACHE_ROOT="$REPO/.runtime_cache/310p_matmul_only_$COMMIT_SHORT"
LAB_SCRIPT="$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py"

test -f "$LAB_SCRIPT"
test ! -e "$PHASE13_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$PHASE13_ROOT" "$CACHE_ROOT/graphs" "$CACHE_ROOT/profile"
```

Recover the exact interpreter and recognizer path from the retained successful
Phase 7 production command rather than guessing:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
test -f "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])

def option(name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc

print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
```

Activate the exact CANN/torch-npu shell environment used by the successful
previous phases. Expose one free full physical 310P as logical `npu:0`. Never
terminate another user's process; stop if no device is free.

Record the environment and require at least 20 GiB free for the one graph and
profile:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  "$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch
import torch_npu
print("platform=" + platform.platform())
print("python_version=" + sys.version.replace("\n", " "))
print("torch=" + torch.__version__)
print("torch_npu=" + getattr(torch_npu, "__version__", "<missing>"))
print("npu_available=" + str(torch.npu.is_available()))
if torch.npu.is_available():
    print("npu_name=" + torch.npu.get_device_name(0))
PY
} >"$PHASE13_ROOT/environment.txt" 2>&1

df -h "$REPO" "$REPO/.runtime_cache" >"$PHASE13_ROOT/disk_before.txt"
npu-smi info >"$PHASE13_ROOT/npu_before.txt" 2>&1
CACHE_FREE_KIB="$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')"
test -n "$CACHE_FREE_KIB"
if test "$CACHE_FREE_KIB" -lt 20971520; then
  printf 'insufficient cache free space: %s KiB\n' "$CACHE_FREE_KIB" \
    >"$PHASE13_ROOT/disk_blocker.txt"
  exit 1
fi
```

### 13.2 Run exactly one compiled full-stack graph

Write a replayable command:

```sh
{
  printf '#!/usr/bin/env bash\n'
  printf '# git_commit=%s\n' "$COMMIT"
  printf '# hostname=%s\n' "$(hostname)"
  printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf '%q ' \
    "$PYTHON_BIN" "$LAB_SCRIPT" \
    --batch-size 1 \
    --sequence-length 2048 \
    --intermediate-size 4304 \
    --weight-format native \
    --attention-head-padding runtime \
    --rotary-implementation separate_manual \
    --execution torchair \
    --model "$RECOGNIZER_MODEL" \
    --cache-dir "$CACHE_ROOT/graphs" \
    --output-dir "$PHASE13_ROOT/result" \
    --profile-dir "$CACHE_ROOT/profile" \
    --allow-compile-if-missing \
    --warmup 3 \
    --samples 10 \
    --calls-per-sample 5 \
    --profile \
    --profile-warmup-steps 1 \
    --profile-steps 3 \
    --parser-topn 200
  printf '\n'
} >"$PHASE13_ROOT/command.sh"
chmod +x "$PHASE13_ROOT/command.sh"
```

Run it once:

```sh
(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$PHASE13_ROOT/npu_smi_1s.log" 2>&1 &
MONITOR_PID=$!

set +e
"$PHASE13_ROOT/command.sh" >"$PHASE13_ROOT/run.log" 2>&1
STATUS=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$STATUS" >"$PHASE13_ROOT/exit_code.txt"
test "$STATUS" -eq 0
```

The command may create at most one graph. Do not rerun a failure into the same
output, profile, or cache directories. Preserve the first failure and report
its causal traceback.

### 13.3 Validate and produce the 910B2/310P comparison

The committed lab writes the MatMul-only calculation directly under
`parsed_profile.matmul_only`. Validate every accounting identity and create
the compact comparison:

```sh
"$PYTHON_BIN" - \
  "$PHASE13_ROOT/result/run_summary.json" \
  "$PHASE13_ROOT/comparison.json" \
  "$PHASE13_ROOT/comparison.md" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path, out_json, out_md = map(Path, sys.argv[1:])
payload = json.loads(summary_path.read_text(encoding="utf-8"))

assert payload["status"] == "completed"
shape = payload["shape"]
assert shape["batch_size"] == 1
assert shape["sequence_length"] == 2048
assert shape["physical_tokens_per_call"] == 2048
assert shape["hidden_size"] == 1152
assert shape["source_intermediate_size"] == 4304
assert shape["candidate_intermediate_size"] == 4304
assert shape["layers"] == 27
assert shape["linear_calls_per_full_stack"] == 162

requested = payload["requested"]
assert requested["execution"] == "torchair"
assert requested["weight_format"] == "native"
assert requested["attention_head_padding"] == "runtime"
assert requested["rotary_implementation"] == "separate_manual"
assert payload["attention"]["implementation"] == "prompt_flash_attention"
assert payload["weight_format"]["after_format_histogram"] == {"2": 162}

measurements = payload["measurements"]
assert measurements["samples"] == 10
assert measurements["calls_per_sample"] == 5
assert measurements["total_measured_full_stack_calls"] == 50
assert payload["linear_flops_per_full_stack_call"] == 1_683_744_620_544

profile = payload["parsed_profile"]
matmul = profile["matmul_only"]
assert matmul["active_profiled_full_stack_calls"] == 3
assert matmul["matmul_kernels_per_full_stack_call"] == 162
assert matmul["observed_matmul_kernel_count"] == 486
assert matmul["linear_flops_per_full_stack_call"] == 1_683_744_620_544

dispatch = profile["dispatch"]
assert sum(dispatch["counts"].values()) == 486
assert math.isclose(
    sum(dispatch["duration_us"].values()),
    matmul["total_matmul_kernel_duration_us"],
    rel_tol=0.0,
    abs_tol=1e-6,
)

full_ms = float(measurements["device_event_per_call_ms"]["median"])
physical_tok_s = float(
    measurements["physical_tokens_per_s_device_median"]
)
full_tflops = float(
    measurements["linear_tflop_per_s_device_median"]
)
matmul_ms = float(
    matmul["matmul_kernel_duration_per_full_stack_call_ms"]
)
matmul_tflops = float(matmul["matmul_only_linear_tflop_per_s"])
matmul_share_pct = matmul_ms / full_ms * 100.0

reference = {
    "device": "Ascend 910B2",
    "commit": "a082212",
    "published_fp16_peak_tflop_per_s": 280.0,
    "full_stage_median_ms": 31.535595703125,
    "physical_tokens_per_s": 64942.486556455144,
    "full_stage_linear_tflop_per_s": 53.391876164151554,
    "matmul_kernel_ms_per_replay": 7.94404,
    "matmul_only_tflop_per_s": 211.95067252229347,
}
target = {
    "device": "Ascend 310P",
    "published_fp16_peak_tflop_per_s": 70.0,
    "full_stage_median_ms": full_ms,
    "physical_tokens_per_s": physical_tok_s,
    "full_stage_linear_tflop_per_s": full_tflops,
    "matmul_kernel_ms_per_replay": matmul_ms,
    "matmul_only_tflop_per_s": matmul_tflops,
    "matmul_share_of_full_stage_pct": matmul_share_pct,
    "matmul_efficiency_pct": matmul_tflops / 70.0 * 100.0,
    "dispatch_counts_total": dispatch["counts"],
    "dispatch_duration_us_total": dispatch["duration_us"],
}
comparison = {
    "schema_version": 1,
    "fixed_linear_flops_per_replay": 1_683_744_620_544,
    "reference_910b2": reference,
    "target_310p": target,
    "ratios": {
        "910b2_over_310p_physical_throughput": (
            reference["physical_tokens_per_s"] / physical_tok_s
        ),
        "910b2_over_310p_matmul_only_throughput": (
            reference["matmul_only_tflop_per_s"] / matmul_tflops
        ),
        "310p_relative_efficiency_vs_910b2": (
            target["matmul_efficiency_pct"]
            / (
                reference["matmul_only_tflop_per_s"]
                / reference["published_fp16_peak_tflop_per_s"]
                * 100.0
            )
        ),
    },
}
out_json.write_text(
    json.dumps(comparison, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# B1xS2048 MatMul-only throughput: 910B2 vs 310P",
    "",
    "| device | full-stage ms | physical tok/s | "
    "full-stage linear TFLOP/s | MatMul ms | MatMul-only TFLOP/s | "
    "MatMul peak efficiency |",
    "|---|---:|---:|---:|---:|---:|---:|",
    (
        f"| 910B2 | {reference['full_stage_median_ms']:.6f} | "
        f"{reference['physical_tokens_per_s']:.3f} | "
        f"{reference['full_stage_linear_tflop_per_s']:.6f} | "
        f"{reference['matmul_kernel_ms_per_replay']:.6f} | "
        f"{reference['matmul_only_tflop_per_s']:.6f} | "
        f"{reference['matmul_only_tflop_per_s'] / 280.0 * 100.0:.4f}% |"
    ),
    (
        f"| 310P | {full_ms:.6f} | {physical_tok_s:.3f} | "
        f"{full_tflops:.6f} | {matmul_ms:.6f} | "
        f"{matmul_tflops:.6f} | "
        f"{target['matmul_efficiency_pct']:.4f}% |"
    ),
    "",
    (
        "910B2/310P MatMul-only throughput ratio: "
        f"{comparison['ratios']['910b2_over_310p_matmul_only_throughput']:.4f}x"
    ),
    (
        "310P MatMul share of full-stage median: "
        f"{matmul_share_pct:.4f}%"
    ),
    "",
    "PHASE13_CONTRACTS: PASS",
]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"), end="")
PY

test "$(tail -n 1 "$PHASE13_ROOT/comparison.md")" = \
  "PHASE13_CONTRACTS: PASS"
```

If the observed MatMul count is not exactly 486, stop. Do not divide by an
assumed three replays or silently ignore an unfamiliar MatMul kernel family.
Inspect `parsed_profile_summary.json`, identify whether the parser omitted a
real MatMul type, and report the mismatch.

### 13.4 Interpretation and required report

Write:

```text
$PHASE13_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P B1xS2048 MATMUL-ONLY PROFILE: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical NPU:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver:
CANN / driver / firmware:
Recognizer model:

Fixed graph contract:
batch / sequence / physical tokens:
layers / hidden / native MLP:
attention projection / head padding:
weight format:
attention / RoPE:
dtype:
warmup / unprofiled samples / calls per sample:
profile warmup / active replays:
linear FLOPs per replay:
MatMul kernels per replay:

Unprofiled full-stage result:
median / mean / p05 / p95 ms:
all ten per-sample replay values:
physical tok/s:
full-stage linear-equivalent TFLOP/s:
compile cache existed before / first-call evidence:

Profiled MatMul result:
MatMulV2 count / total us / per-replay ms:
MatMulV3 count / total us / per-replay ms:
other MatMul families:
all MatMul count:
all MatMul total us:
all MatMul ms per replay:
MatMul-only TFLOP/s:
MatMul-only percent of 70-TFLOP peak:
MatMul share of unprofiled full-stage median:
weighted cube utilization:
Block Dim / current and rated frequency if present:
MTE1 / MTE2 / MTE3 / Scalar utilization if present:

Matched 910B2 reference:
full-stage ms / physical tok/s / linear-equivalent TFLOP/s:
MatMul ms / MatMul-only TFLOP/s / percent of 280-TFLOP peak:

Head-to-head:
910B2/310P physical-throughput ratio:
910B2/310P MatMul-only throughput ratio:
published peak ratio:
310P relative efficiency versus 910B2:
Does the MatMul-only result reproduce the earlier ~16-TFLOP/s observation:
Does 310P remain in a low-efficiency MatMul regime:

First blocker or warning:
Exact command:
Run summary:
Parsed profile JSON / Markdown:
Comparison JSON / Markdown:
Raw profile root:
Graph cache:
All artifact paths:
```

Report the result even if it disproves the earlier approximately
16.2-TFLOP/s observation. Do not reinterpret high Cube utilization as percent
of peak FLOP/s; report it separately from the calculated MatMul-only
efficiency. Stop after this report and send it plus exact artifact paths back
to Luka manually.

The work server is pull-only. Do not edit tracked files, create a branch,
commit, or push. Keep graph caches and raw profiler trees under
`.runtime_cache`; keep the compact commands, logs, summaries, comparison, and
report under `tmp/`. Send Luka the report and exact artifact paths manually.

## Phase 14: six-lane native-ND versus padded-NZ MatMul comparison

### 14.0 Purpose, scope, and exact 910B2 references

This was the only phase in the earlier six-lane task. It is retained as
provenance for the graph/cache that Phases 15-16 reuse. Do not rerun these six
complete 27-layer vision-stage lanes:

| Label | Batch | Sequence | Physical tokens | MLP | Weight format |
|---|---:|---:|---:|---:|---|
| `b1_s512_native_nd` | 1 | 512 | 512 | 4304 | native ND |
| `b1_s512_padded_nz` | 1 | 512 | 512 | 4352 | FRACTAL_NZ |
| `b4_s512_native_nd` | 4 | 512 | 2048 | 4304 | native ND |
| `b4_s512_padded_nz` | 4 | 512 | 2048 | 4352 | FRACTAL_NZ |
| `b1_s2048_native_nd` | 1 | 2048 | 2048 | 4304 | native ND |
| `b1_s2048_padded_nz` | 1 | 2048 | 2048 | 4352 | FRACTAL_NZ |

Everything else is immutable:

```text
vision layers:               all 27
hidden size:                 1152
attention projections:       native 1152 outputs
attention head padding:      runtime D72 -> D80, then D80 -> D72
RoPE:                        separate_manual
attention:                   real PromptFlashAttention
execution:                   TorchAir cache_compile
dtype:                       fp16
warmup:                      3 complete full-stack replays
unprofiled measurement:      10 samples x 5 full-stack replays
profile:                     1 warmup + 3 active full-stack replays
Linear MatMuls per replay:   27 x 6 = 162
profiled MatMuls per lane:   3 x 162 = 486
```

The 4352 MLP is mathematically equivalent to the original 4304 MLP: FC1 adds
48 all-zero output rows and FC2 adds 48 all-zero input columns. `GELU(0)=0`.
It performs 0.726392% more Linear FLOPs. All 162 Linear weights must be
explicitly format code 29 in an NZ lane; a silent ND fallback invalidates the
lane.

This comparison intentionally changes MLP alignment and weight format
together. It answers whether the combined 4352+NZ configuration helps. It
does **not** attribute the gain separately to padding versus NZ, so do not
claim that either change alone caused the result.

Fresh matched 910B2 references from commit `fb9ad7b`:

| Shape/config | Full stage ms | Physical tok/s | Linear FLOPs | MatMul ms | MatMul-only TFLOP/s |
|---|---:|---:|---:|---:|---:|
| B1xS512 4304 ND | 15.757428 | 32,492.6 | 420,936,155,136 | 4.959353 | 84.877 |
| B1xS512 4352 NZ | 13.604366 | 37,635.0 | 423,993,802,752 | 2.858633 | 148.320 |
| B4xS512 4304 ND | 27.347794 | 74,887.2 | 1,683,744,620,544 | 8.004093 | 210.360 |
| B4xS512 4352 NZ | 26.655551 | 76,832.0 | 1,695,975,211,008 | 7.366140 | 230.239 |
| B1xS2048 4304 ND | 31.554370 | 64,903.8 | 1,683,744,620,544 | 7.983893 | 210.893 |
| B1xS2048 4352 NZ | 30.571022 | 66,991.5 | 1,695,975,211,008 | 7.339400 | 231.078 |

The 910B2 paired changes were:

```text
B1xS512:  full stage -13.664%; MatMul time -42.359%
B4xS512:  full stage  -2.531%; MatMul time  -7.970%
B1xS2048: full stage  -3.116%; MatMul time  -8.072%
```

These are references, not expected 310P results. For a full physical 310P,
use 70 FP16 TFLOP/s as the published peak denominator. Do not clamp a
calculated efficiency if it exceeds 100%; instead verify the device exposure,
clock, FLOP convention, profile duration, and MatMul count and report the
discrepancy.

Do not run 4304-NZ, 4352-ND, attention weight-padding, joint RoPE, manual
attention, eager execution, isolated Linear layers, other shapes, layout, OCR
pages, text prefill, or decode. At most six exact graphs may be created;
normally only five should be new if the Phase 13 native B1xS2048 cache is
reusable.

### 14.1 Pull and recover the proven environment

Do not discard, overwrite, stash, or clean any local work:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PHASE14_REL="tmp/09_persistent_page_engine/310p_matmul_matrix_$COMMIT_SHORT"
PHASE14_ROOT="$REPO/$PHASE14_REL"
NEW_CACHE_ROOT="$REPO/.runtime_cache/310p_matmul_matrix_$COMMIT_SHORT"
LAB_SCRIPT="$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py"

test -f "$LAB_SCRIPT"
test ! -e "$PHASE14_ROOT"
test ! -e "$NEW_CACHE_ROOT"
mkdir -p \
  "$PHASE14_ROOT/results" \
  "$PHASE14_ROOT/commands" \
  "$NEW_CACHE_ROOT/graphs" \
  "$NEW_CACHE_ROOT/profile"
```

Recover the exact interpreter and recognizer model from the retained
successful Phase 7 production command:

```sh
PHASE7_COMMAND="$(
  python3 - "$REPO" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
matches = list(
    repo.glob(
        "tmp/09_persistent_page_engine/"
        "310p_exp09_npu_layout_eager_*/"
        "phase7_min_pixels_28224_replay/command.sh"
    )
)
if not matches:
    raise SystemExit("no retained successful Phase 7 command.sh was found")
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
test -f "$PHASE7_COMMAND"

eval "$(
  python3 - "$PHASE7_COMMAND" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
lines = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit(f"expected one command in {path}, found {len(lines)}")
tokens = shlex.split(lines[0])

def option(name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} missing from {path}") from exc

print(f"PYTHON_BIN={shlex.quote(tokens[0])}")
print(f"RECOGNIZER_MODEL={shlex.quote(option('--recognizer-model'))}")
PY
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
```

Activate the exact CANN/torch-npu environment used by the successful previous
phases and expose one free full physical 310P as logical `npu:0`. Never
terminate another user's process. Stop if no device is free.

Reuse the latest Phase 13 graph root if it exists. The documentation-only
Phase 14 commit does not alter the graph source hash:

```sh
PRIOR_PHASE13_CACHE="$(
  find "$REPO/.runtime_cache" \
    -maxdepth 1 \
    -type d \
    -name '310p_matmul_only_*' \
    -print |
  sort |
  tail -n 1
)"

if test -n "$PRIOR_PHASE13_CACHE" &&
   test -d "$PRIOR_PHASE13_CACHE/graphs"; then
  GRAPH_CACHE_ROOT="$PRIOR_PHASE13_CACHE/graphs"
  GRAPH_CACHE_ROUTE="reused_phase13"
else
  GRAPH_CACHE_ROOT="$NEW_CACHE_ROOT/graphs"
  GRAPH_CACHE_ROUTE="new_phase14"
fi

PROFILE_ROOT="$NEW_CACHE_ROOT/profile"
mkdir -p "$GRAPH_CACHE_ROOT" "$PROFILE_ROOT"
```

Record the environment and verify free space. Six graphs can be large; require
at least 60 GiB free on the cache filesystem before starting. If the check
fails, stop and report rather than redirecting caches outside this repository:

```sh
{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_status_begin\n'
  git status --short --branch
  printf 'git_status_end\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'recognizer_model=%s\n' "$RECOGNIZER_MODEL"
  printf 'graph_cache_root=%s\n' "$GRAPH_CACHE_ROOT"
  printf 'graph_cache_route=%s\n' "$GRAPH_CACHE_ROUTE"
  "$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch
import torch_npu
print("platform=" + platform.platform())
print("python_version=" + sys.version.replace("\n", " "))
print("torch=" + torch.__version__)
print("torch_npu=" + getattr(torch_npu, "__version__", "<missing>"))
print("npu_available=" + str(torch.npu.is_available()))
if torch.npu.is_available():
    print("npu_name=" + torch.npu.get_device_name(0))
PY
} >"$PHASE14_ROOT/environment.txt" 2>&1

df -h "$REPO" "$GRAPH_CACHE_ROOT" >"$PHASE14_ROOT/disk_before.txt"
npu-smi info >"$PHASE14_ROOT/npu_before.txt" 2>&1
CACHE_FREE_KIB="$(df -Pk "$GRAPH_CACHE_ROOT" | awk 'NR == 2 {print $4}')"
test -n "$CACHE_FREE_KIB"
if test "$CACHE_FREE_KIB" -lt 62914560; then
  printf 'insufficient cache free space: %s KiB\n' "$CACHE_FREE_KIB" \
    >"$PHASE14_ROOT/disk_blocker.txt"
  exit 1
fi

find "$GRAPH_CACHE_ROOT" \
  -mindepth 1 -maxdepth 1 -type d -print |
sort >"$PHASE14_ROOT/graph_dirs_before.txt"
```

### 14.2 Create and run exactly six replayable commands

Create the immutable lane manifest:

```sh
cat >"$PHASE14_ROOT/lanes.tsv" <<'EOF'
b1_s512_native_nd 1 512 4304 native
b1_s512_padded_nz 1 512 4352 fractal_nz
b4_s512_native_nd 4 512 4304 native
b4_s512_padded_nz 4 512 4352 fractal_nz
b1_s2048_native_nd 1 2048 4304 native
b1_s2048_padded_nz 1 2048 4352 fractal_nz
EOF
```

For each lane, create one exact `command.sh`. Do not modify flags between
lanes:

```sh
while read -r LABEL BATCH SEQ WIDTH FORMAT; do
  RESULT_DIR="$PHASE14_ROOT/results/$LABEL"
  PROFILE_DIR="$PROFILE_ROOT/$LABEL"
  COMMAND="$PHASE14_ROOT/commands/$LABEL.sh"

  test ! -e "$RESULT_DIR"
  test ! -e "$PROFILE_DIR"

  {
    printf '#!/usr/bin/env bash\n'
    printf '# git_commit=%s\n' "$COMMIT"
    printf '# hostname=%s\n' "$(hostname)"
    printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
      "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf '%q ' \
      "$PYTHON_BIN" "$LAB_SCRIPT" \
      --batch-size "$BATCH" \
      --sequence-length "$SEQ" \
      --intermediate-size "$WIDTH" \
      --weight-format "$FORMAT" \
      --attention-head-padding runtime \
      --rotary-implementation separate_manual \
      --execution torchair \
      --model "$RECOGNIZER_MODEL" \
      --cache-dir "$GRAPH_CACHE_ROOT" \
      --output-dir "$RESULT_DIR" \
      --profile-dir "$PROFILE_DIR" \
      --allow-compile-if-missing \
      --warmup 3 \
      --samples 10 \
      --calls-per-sample 5 \
      --profile \
      --profile-warmup-steps 1 \
      --profile-steps 3 \
      --parser-topn 200
    printf '\n'
  } >"$COMMAND"
  chmod +x "$COMMAND"
done <"$PHASE14_ROOT/lanes.tsv"
```

Start one 1-second NPU monitor and run the six commands sequentially. The
matrix is sequential so lanes do not contend for the same NPU:

```sh
(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$PHASE14_ROOT/npu_smi_1s.log" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

for COMMAND in "$PHASE14_ROOT"/commands/*.sh; do
  LABEL="$(basename "$COMMAND" .sh)"
  printf 'START %s %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "$LABEL" |
  tee -a "$PHASE14_ROOT/progress.log"

  set +e
  "$COMMAND" >"$PHASE14_ROOT/results/$LABEL.log" 2>&1
  STATUS=$?
  set -e
  printf '%s\n' "$STATUS" >"$PHASE14_ROOT/results/$LABEL.exit_code.txt"

  printf 'END %s %s exit=%s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    "$LABEL" "$STATUS" |
  tee -a "$PHASE14_ROOT/progress.log"

  if test "$STATUS" -ne 0; then
    break
  fi
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
```

Require all six lanes:

```sh
while read -r LABEL _; do
  test "$(cat "$PHASE14_ROOT/results/$LABEL.exit_code.txt")" -eq 0
  test -f "$PHASE14_ROOT/results/$LABEL/run_summary.json"
done <"$PHASE14_ROOT/lanes.tsv"

find "$GRAPH_CACHE_ROOT" \
  -mindepth 1 -maxdepth 1 -type d -print |
sort >"$PHASE14_ROOT/graph_dirs_after.txt"

comm -13 \
  "$PHASE14_ROOT/graph_dirs_before.txt" \
  "$PHASE14_ROOT/graph_dirs_after.txt" \
  >"$PHASE14_ROOT/new_graph_dirs.txt"

NEW_GRAPH_COUNT="$(wc -l <"$PHASE14_ROOT/new_graph_dirs.txt")"
test "$NEW_GRAPH_COUNT" -le 6
```

Do not rerun a failed lane into the same output or profile directory. Preserve
the first failure, its causal traceback, completed earlier lanes, graph count,
and NPU monitor. Stop; do not invent a fallback.

### 14.3 Validate every lane and create the comparison

Run this validator exactly:

```sh
"$PYTHON_BIN" - "$PHASE14_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

expected = {
    "b1_s512_native_nd": {
        "batch": 1,
        "seq": 512,
        "width": 4304,
        "format": "native",
        "format_hist": {"2": 162},
        "flops": 420_936_155_136,
    },
    "b1_s512_padded_nz": {
        "batch": 1,
        "seq": 512,
        "width": 4352,
        "format": "fractal_nz",
        "format_hist": {"29": 162},
        "flops": 423_993_802_752,
    },
    "b4_s512_native_nd": {
        "batch": 4,
        "seq": 512,
        "width": 4304,
        "format": "native",
        "format_hist": {"2": 162},
        "flops": 1_683_744_620_544,
    },
    "b4_s512_padded_nz": {
        "batch": 4,
        "seq": 512,
        "width": 4352,
        "format": "fractal_nz",
        "format_hist": {"29": 162},
        "flops": 1_695_975_211_008,
    },
    "b1_s2048_native_nd": {
        "batch": 1,
        "seq": 2048,
        "width": 4304,
        "format": "native",
        "format_hist": {"2": 162},
        "flops": 1_683_744_620_544,
    },
    "b1_s2048_padded_nz": {
        "batch": 1,
        "seq": 2048,
        "width": 4352,
        "format": "fractal_nz",
        "format_hist": {"29": 162},
        "flops": 1_695_975_211_008,
    },
}

reference_910b2 = {
    "b1_s512_native_nd": {
        "full_stage_ms": 15.757427978515626,
        "physical_tok_s": 32492.612417336346,
        "matmul_ms": 4.959353333333338,
        "matmul_tflop_s": 84.87722629213748,
    },
    "b1_s512_padded_nz": {
        "full_stage_ms": 13.60436553955078,
        "physical_tok_s": 37634.97816281893,
        "matmul_ms": 2.8586333333333345,
        "matmul_tflop_s": 148.3204571247332,
    },
    "b4_s512_native_nd": {
        "full_stage_ms": 27.347793579101562,
        "physical_tok_s": 74887.21143357706,
        "matmul_ms": 8.00409333333333,
        "matmul_tflop_s": 210.36044314125945,
    },
    "b4_s512_padded_nz": {
        "full_stage_ms": 26.655551147460937,
        "physical_tok_s": 76832.02604479186,
        "matmul_ms": 7.366139999999987,
        "matmul_tflop_s": 230.23933987244376,
    },
    "b1_s2048_native_nd": {
        "full_stage_ms": 31.554370117187503,
        "physical_tok_s": 64903.846674615284,
        "matmul_ms": 7.9838933333333335,
        "matmul_tflop_s": 210.89267481997086,
    },
    "b1_s2048_padded_nz": {
        "full_stage_ms": 30.571022033691406,
        "physical_tok_s": 66991.54505671942,
        "matmul_ms": 7.339400000000004,
        "matmul_tflop_s": 231.07818227757025,
    },
}

rows = {}
for label, contract in expected.items():
    summary_path = root / "results" / label / "run_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    shape = payload["shape"]
    assert shape["batch_size"] == contract["batch"]
    assert shape["sequence_length"] == contract["seq"]
    assert shape["physical_tokens_per_call"] == (
        contract["batch"] * contract["seq"]
    )
    assert shape["hidden_size"] == 1152
    assert shape["source_intermediate_size"] == 4304
    assert shape["candidate_intermediate_size"] == contract["width"]
    assert shape["layers"] == 27
    assert shape["linear_calls_per_full_stack"] == 162

    requested = payload["requested"]
    assert requested["execution"] == "torchair"
    assert requested["weight_format"] == contract["format"]
    assert requested["attention_head_padding"] == "runtime"
    assert requested["rotary_implementation"] == "separate_manual"
    assert payload["attention"]["implementation"] == (
        "prompt_flash_attention"
    )
    assert payload["weight_format"]["after_format_histogram"] == (
        contract["format_hist"]
    )
    assert payload["weight_format"]["linear_weight_count"] == 162
    if contract["format"] == "fractal_nz":
        assert payload["weight_format"]["converted_count"] == 162
        assert payload["weight_format"]["all_after_are_nz"] is True
    else:
        assert payload["weight_format"]["converted_count"] == 0
        assert payload["weight_format"]["all_after_are_nz"] is False

    measurements = payload["measurements"]
    assert measurements["samples"] == 10
    assert measurements["calls_per_sample"] == 5
    assert measurements["total_measured_full_stack_calls"] == 50
    assert payload["linear_flops_per_full_stack_call"] == contract["flops"]

    profile = payload["parsed_profile"]
    matmul = profile["matmul_only"]
    assert matmul["active_profiled_full_stack_calls"] == 3
    assert matmul["matmul_kernels_per_full_stack_call"] == 162
    assert matmul["observed_matmul_kernel_count"] == 486
    assert matmul["linear_flops_per_full_stack_call"] == contract["flops"]
    dispatch = profile["dispatch"]
    assert sum(dispatch["counts"].values()) == 486
    assert math.isclose(
        sum(dispatch["duration_us"].values()),
        matmul["total_matmul_kernel_duration_us"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    full_ms = float(measurements["device_event_per_call_ms"]["median"])
    tok_s = float(measurements["physical_tokens_per_s_device_median"])
    full_tflops = float(
        measurements["linear_tflop_per_s_device_median"]
    )
    matmul_ms = float(
        matmul["matmul_kernel_duration_per_full_stack_call_ms"]
    )
    matmul_tflops = float(matmul["matmul_only_linear_tflop_per_s"])
    rows[label] = {
        "shape": {
            "batch": contract["batch"],
            "sequence": contract["seq"],
            "physical_tokens": contract["batch"] * contract["seq"],
            "mlp_width": contract["width"],
            "weight_format": contract["format"],
        },
        "linear_flops_per_replay": contract["flops"],
        "full_stage_ms": full_ms,
        "physical_tok_s": tok_s,
        "full_stage_linear_tflop_s": full_tflops,
        "matmul_ms": matmul_ms,
        "matmul_only_tflop_s": matmul_tflops,
        "matmul_peak_efficiency_pct_70t": matmul_tflops / 70.0 * 100.0,
        "matmul_share_of_full_stage_pct": matmul_ms / full_ms * 100.0,
        "dispatch_counts_three_replays": dispatch["counts"],
        "dispatch_duration_us_three_replays": dispatch["duration_us"],
        "weighted_cube_utilization_pct": profile.get(
            "weighted_cube_utilization_pct"
        ),
        "reference_910b2": reference_910b2[label],
        "ratios_910b2_over_310p": {
            "physical_tok_s": (
                reference_910b2[label]["physical_tok_s"] / tok_s
            ),
            "matmul_only_tflop_s": (
                reference_910b2[label]["matmul_tflop_s"]
                / matmul_tflops
            ),
        },
    }

pairs = {
    "b1_s512": (
        "b1_s512_native_nd",
        "b1_s512_padded_nz",
    ),
    "b4_s512": (
        "b4_s512_native_nd",
        "b4_s512_padded_nz",
    ),
    "b1_s2048": (
        "b1_s2048_native_nd",
        "b1_s2048_padded_nz",
    ),
}

deltas = {}
for pair, (native_label, padded_label) in pairs.items():
    native = rows[native_label]
    padded = rows[padded_label]
    deltas[pair] = {
        "linear_flops_pct": (
            padded["linear_flops_per_replay"]
            / native["linear_flops_per_replay"]
            - 1.0
        )
        * 100.0,
        "full_stage_ms_pct": (
            padded["full_stage_ms"] / native["full_stage_ms"] - 1.0
        )
        * 100.0,
        "physical_tok_s_pct": (
            padded["physical_tok_s"] / native["physical_tok_s"] - 1.0
        )
        * 100.0,
        "matmul_ms_pct": (
            padded["matmul_ms"] / native["matmul_ms"] - 1.0
        )
        * 100.0,
        "matmul_only_tflop_s_pct": (
            padded["matmul_only_tflop_s"]
            / native["matmul_only_tflop_s"]
            - 1.0
        )
        * 100.0,
    }
    assert math.isclose(
        deltas[pair]["linear_flops_pct"],
        0.7263922518159882,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

comparison = {
    "schema_version": 1,
    "device": "Ascend310P3",
    "published_fp16_peak_tflop_s": 70.0,
    "rows": rows,
    "padded_4352_nz_vs_native_4304_nd_delta_pct": deltas,
    "interpretation_limit": (
        "The paired lane changes MLP alignment and weight format together; "
        "it does not attribute the gain to either change independently."
    ),
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# 310P3 native-ND versus padded-NZ vision MatMul matrix",
    "",
    "| lane | full stage ms | physical tok/s | MatMul ms | "
    "MatMul-only TFLOP/s | peak efficiency | "
    "910B2/310P MatMul ratio |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for label, row in rows.items():
    lines.append(
        f"| {label} | {row['full_stage_ms']:.6f} | "
        f"{row['physical_tok_s']:.1f} | {row['matmul_ms']:.6f} | "
        f"{row['matmul_only_tflop_s']:.3f} | "
        f"{row['matmul_peak_efficiency_pct_70t']:.3f}% | "
        f"{row['ratios_910b2_over_310p']['matmul_only_tflop_s']:.3f}x |"
    )
lines.extend(
    [
        "",
        "## Padded 4352 NZ versus native 4304 ND",
        "",
    ]
)
for pair, delta in deltas.items():
    lines.append(
        f"- {pair}: full stage {delta['full_stage_ms_pct']:+.3f}%; "
        f"physical tok/s {delta['physical_tok_s_pct']:+.3f}%; "
        f"MatMul time {delta['matmul_ms_pct']:+.3f}%; "
        f"MatMul TFLOP/s {delta['matmul_only_tflop_s_pct']:+.3f}%"
    )
lines.extend(
    [
        "",
        "Combined-change caveat: 4352 alignment and FRACTAL_NZ were changed "
        "together; this matrix does not attribute their effects separately.",
        "",
        "PHASE14_CONTRACTS: PASS",
    ]
)
(root / "comparison.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
print((root / "comparison.md").read_text(encoding="utf-8"), end="")
PY

test "$(tail -n 1 "$PHASE14_ROOT/comparison.md")" = \
  "PHASE14_CONTRACTS: PASS"
```

If any lane has a MatMul count other than 486, stop. Do not divide by an
assumed replay count or silently omit an unfamiliar MatMul family. Inspect
that lane's parsed profile, identify whether the parser omitted a true MatMul
type, and report the mismatch.

### 14.4 Required report and interpretation questions

Write:

```text
$PHASE14_ROOT/agent_report.md
```

Use this exact skeleton:

```text
310P SIX-LANE VISION MATMUL MATRIX: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical NPU:
Logical NPU:
Python:
torch:
torch_npu:
TorchAir resolver:
CANN / driver / firmware:
Recognizer model:
Graph cache route:
Graph directories before / after / newly created:

Fixed graph contract:
layers / hidden:
attention projection / runtime head padding:
attention / RoPE:
execution / dtype:
warmup / samples / calls per sample:
profile warmup / active replays:
MatMul kernels per replay:

Six-lane table:
For each lane report:
- full-stage median / mean / p05 / p95 ms
- all ten per-sample replay values
- physical tok/s
- Linear FLOPs per replay
- MatMulV2 count / total us / per-replay ms
- MatMulV3 count / total us / per-replay ms
- any other MatMul families
- all-MatMul count / total us / per-replay ms
- MatMul-only TFLOP/s
- MatMul percent of 70-TFLOP peak
- MatMul share of full-stage median
- weighted Cube / MTE / Scalar utilization if present
- exact 910B2 reference and 910B2/310P ratio

Paired 4352-NZ versus 4304-ND changes:
B1xS512 full-stage / physical tok/s / MatMul-ms / MatMul-TFLOP deltas:
B4xS512 full-stage / physical tok/s / MatMul-ms / MatMul-TFLOP deltas:
B1xS2048 full-stage / physical tok/s / MatMul-ms / MatMul-TFLOP deltas:

Interpretation:
Does 4352+NZ rescue the underfilled B1xS512 MatMuls:
Does the gain persist, shrink, or reverse at B4xS512:
Does B4xS512 MatMul behavior match B1xS2048 at equal B*S=2048:
Does 310P reproduce the 910B2 pattern:
Is the earlier ~16.2-TFLOP/s B1xS2048 ND observation reproduced:
Can the gain be attributed separately to padding or NZ: NO

First blocker or warning:
Exact command paths:
Run-summary paths:
Parsed-profile paths:
Comparison JSON / Markdown:
Raw profile root:
Graph cache:
NPU monitor:
All artifact paths:
```

Do not use high Cube utilization as a substitute for calculated TFLOP/s.
Report both. Do not call the padded-NZ lane semantically different—the 4352
MLP is zero-extended and mathematically equivalent—but do explicitly state
that this experiment changes alignment and weight format together. Stop after
Phase 14.4 and send the report plus exact artifact paths back to Luka
manually.

The work server is pull-only. Do not edit tracked files, create a branch,
commit, or push. Keep graph caches and raw profiler trees under
`.runtime_cache`; keep commands, logs, summaries, comparison, and report under
`tmp/`.

## Phase 15: B1xS2048 compiled full-stack multi-metric profile

### 15.0 Scope and expected configuration

This is the first phase of the current task. It runs the real compiled
27-layer `VisionPrefillStage`, not a standalone Linear:

```text
batch x sequence:             B1 x S2048
physical tokens:              2048
hidden size:                  1152
MLP width:                    4352, zero-extended
Linear weights:               FRACTAL_NZ
attention head padding:       runtime D72 -> D80 -> D72
attention:                    PromptFlashAttention
RoPE:                         separate_manual
execution:                    TorchAir cache_compile
dtype:                        fp16
warmup:                       3 full-stage replays
measurement:                  10 samples x 5 replays
profile per metric:           1 warmup + 3 active replays
expected MatMuls per replay:  162
metrics:                      pipe, memory
```

Do not run `l2`, `memory_access`, `Occupancy`, or `MemoryDetail` on 310P.
Do not run `run_vision_msprof_op.py` during Phase 15: its directly observable
target is a diagnostic kernel replay, while this phase establishes the actual
compiled full-stack reference. Phase 16 deliberately runs that second tier
only after matching it back to this reference.

The matched 910B2 references for this exact configuration are:

```text
full-stage device median:       30.606900 ms
physical throughput:            66,913.016 tok/s
profile span per replay:         30.841417 ms
MatMul duration per replay:       7.348947 ms
MatMul-only throughput:          230.778 TFLOP/s
MatMul stage share:               23.83%
PromptFA stage share:             24.82%
StridedSliceD stage share:        12.39%
Transpose stage share:            11.81%
PadV3 stage share:                 4.71%
intra-replay device gaps:         ~0.30 ms
```

These are comparison values, not expected 310P results.

### 15.1 Pull `main` and recover the proven environment

The work server remains pull-only. Do not discard, stash, clean, or overwrite
local evidence:

```sh
test -z "$(git status --porcelain --untracked-files=no)" || {
  printf 'tracked worktree changes exist; stop without modifying them\n'
  git status --short --branch
  exit 1
}

git fetch origin main
git switch --detach origin/main

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"

SUITE_SCRIPT="$REPO/09_persistent_page_engine/scripts/run_vision_matmul_profile_suite.py"
test -f "$SUITE_SCRIPT"
"${PYTHON_BIN:-python3}" "$SUITE_SCRIPT" --help |
  grep -q -- '--progress-interval-s'
```

Recover the exact interpreter, recognizer model, and graph-cache root recorded
by the latest successful Phase 14 run:

```sh
PHASE14_ENV="$(
  find "$REPO/tmp/09_persistent_page_engine" \
    -path '*/310p_matmul_matrix_*/environment.txt' \
    -type f -print |
  sort |
  tail -n 1
)"
test -f "$PHASE14_ENV"

PYTHON_BIN="$(
  awk -F= '$1 == "python" {sub(/^python=/, ""); print; exit}' \
    "$PHASE14_ENV"
)"
RECOGNIZER_MODEL="$(
  awk -F= '$1 == "recognizer_model" {
    sub(/^recognizer_model=/, ""); print; exit
  }' "$PHASE14_ENV"
)"
GRAPH_CACHE_ROOT="$(
  awk -F= '$1 == "graph_cache_root" {
    sub(/^graph_cache_root=/, ""); print; exit
  }' "$PHASE14_ENV"
)"

test -x "$PYTHON_BIN"
test -f "$RECOGNIZER_MODEL/config.json"
test -d "$GRAPH_CACHE_ROOT"
```

Activate the same CANN/torch-npu environment that passed Phase 14 and expose
one free full physical 310P as logical `npu:0`. Never terminate another user's
process. Stop if no device is free. Then verify the active environment:

```sh
"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torch_npu.profiler as npu_prof

assert torch.npu.is_available()
print("torch=" + torch.__version__)
print("torch_npu=" + getattr(torch_npu, "__version__", "<missing>"))
print("device=" + torch.npu.get_device_name(0))
print(
    "supported_ai_core_metrics="
    + repr(npu_prof.supported_ai_core_metrics())
)
PY
```

The reported supported metrics must include PipeUtilization and Memory. If
either is absent, stop and report the exact capability list.

### 15.2 Prepare durable paths and monitoring

Use a fresh run name. The driver log deliberately lives outside the suite
output directory because the runner rejects an already-nonempty suite path:

```sh
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="310p_b1s2048_i4352_nz_multimetric_${COMMIT_SHORT}_${RUN_STAMP}"
DRIVER_ROOT="$REPO/tmp/09_persistent_page_engine/vision_profile_driver/$RUN_NAME"
DRIVER_LOG="$DRIVER_ROOT/driver.log"
EXIT_CODE_FILE="$DRIVER_ROOT/exit_code.txt"
COMMAND_FILE="$DRIVER_ROOT/command.sh"
NPU_LOG="$DRIVER_ROOT/npu_smi_1s.log"

SUITE_ROOT="$REPO/tmp/09_persistent_page_engine/vision_matmul_profile_suite/$RUN_NAME"
RAW_ROOT="$REPO/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/$RUN_NAME"

test ! -e "$DRIVER_ROOT"
test ! -e "$SUITE_ROOT"
test ! -e "$RAW_ROOT"
mkdir -p "$DRIVER_ROOT"

df -h "$REPO" "$GRAPH_CACHE_ROOT" >"$DRIVER_ROOT/disk_before.txt"
npu-smi info >"$DRIVER_ROOT/npu_before.txt" 2>&1

{
  printf '#!/usr/bin/env bash\n'
  printf '# git_commit=%s\n' "$COMMIT"
  printf '# hostname=%s\n' "$(hostname)"
  printf '# ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf '%q ' \
    "$PYTHON_BIN" "$SUITE_SCRIPT" \
    --name "$RUN_NAME" \
    --model "$RECOGNIZER_MODEL" \
    --cache-dir "$GRAPH_CACHE_ROOT" \
    --output-root \
      "$REPO/tmp/09_persistent_page_engine/vision_matmul_profile_suite" \
    --profile-root \
      "$REPO/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles" \
    --metrics pipe memory \
    --allow-compile-if-missing \
    --warmup 3 \
    --samples 10 \
    --calls-per-sample 5 \
    --profile-warmup-steps 1 \
    --profile-steps 3 \
    --progress-interval-s 15
  printf '\n'
} >"$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$NPU_LOG" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
```

### 15.3 Run the main command with a followable driver log

The main command must be redirected exactly as follows. `PYTHONUNBUFFERED=1`
and the runner's flushed progress messages make `driver.log` useful while the
job is still running:

```sh
set +e
PYTHONUNBUFFERED=1 "$COMMAND_FILE" >"$DRIVER_LOG" 2>&1
PROFILE_EXIT=$?
set -e
printf '%s\n' "$PROFILE_EXIT" >"$EXIT_CODE_FILE"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

test "$PROFILE_EXIT" -eq 0
```

Luka can follow progress from another shell:

```sh
tail -n 100 -f \
  "$REPO/tmp/09_persistent_page_engine/vision_profile_driver/$RUN_NAME/driver.log"
```

The driver log prints:

- metric preflight start/pass;
- suite output and raw roots;
- `lane 1/2 pipe` and `lane 2/2 memory` start/completion;
- a 15-second elapsed-time heartbeat and the current subprocess-log size;
- each completed lane's checkpoint path, device median, and physical tok/s;
- combined-analysis start/completion and final report path.

Each lane also writes its child output incrementally to
`$SUITE_ROOT/<metric>/run.log`. `suite_summary.json` is rewritten after every
successful lane, so a later failure does not erase earlier lane results.

If the command fails, do not delete anything and do not rerun with the same
name. Report:

```sh
tail -n 200 "$DRIVER_LOG"
find "$SUITE_ROOT" -maxdepth 3 -type f \
  \( -name 'run.log' -o -name 'run_summary.json' \
     -o -name 'suite_summary.json' \) -print
```

Then stop.

### 15.4 Validate and report

After a successful command, run this mechanical validation:

```sh
"$PYTHON_BIN" - \
  "$SUITE_ROOT/suite_summary.json" \
  "$SUITE_ROOT/combined_profile/profile_analysis.json" \
  "$DRIVER_ROOT/validation.json" <<'PY'
import json
import sys
from pathlib import Path

suite_path, analysis_path, output_path = map(Path, sys.argv[1:])
suite = json.loads(suite_path.read_text(encoding="utf-8"))
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

assert suite["metrics"] == ["pipe", "memory"]
assert set(suite["lanes"]) == {"pipe", "memory"}
for name in ("pipe", "memory"):
    lane = suite["lanes"][name]
    assert lane["device_median_ms"] > 0
    assert lane["physical_tokens_per_s"] > 0
    assert Path(lane["run_summary"]).is_file()

dims = analysis["contract_dims"]
assert dims["batch_size"] == 1
assert dims["sequence_length"] == 2048
assert dims["hidden_size"] == 1152
assert dims["intermediate_size"] == 4352
assert dims["layers"] == 27
assert dims["linear_calls_per_full_stack"] == 162
assert dims["head_padding_mode"] == "runtime"

families = {lane["metric_family"]: lane for lane in analysis["lanes"]}
assert set(families) == {"pipe", "memory"}
for name, lane in families.items():
    assert lane["mapping"]["status"] == "validated"
    assert lane["matmul_count"] == 486
    assert len(lane["replays"]) == 3
    assert all(replay["matmul_count"] == 162 for replay in lane["replays"])

result = {
    "status": "passed",
    "suite": str(suite_path),
    "analysis": str(analysis_path),
    "metrics": sorted(families),
    "device_median_ms": {
        name: suite["lanes"][name]["device_median_ms"]
        for name in ("pipe", "memory")
    },
    "physical_tokens_per_s": {
        name: suite["lanes"][name]["physical_tokens_per_s"]
        for name in ("pipe", "memory")
    },
}
output_path.write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2))
PY
```

Write the final report at:

```text
$DRIVER_ROOT/agent_report.md
```

Use this exact compact skeleton:

```text
310P PHASE 15 MULTI-METRIC VISION PROFILE: PASS | PARTIAL | FAIL

Git commit:
Host / exact physical NPU:
Python / torch / torch_npu:
CANN / driver / firmware:
ASCEND_RT_VISIBLE_DEVICES:
Model:
Graph cache and whether it replayed or compiled:
Supported profiler metrics:

Driver log / exit code:
Suite summary:
Combined JSON / Markdown:
Raw profiler root:
NPU monitor:

Pipe lane:
- device median / physical tok/s
- profile span / replay and intra-replay gap
- total kernels / MatMuls per replay
- MatMul duration / replay and MatMul-only TFLOP/s
- PromptFA / MatMul / StridedSliceD / Transpose / PadV3 ms and stage share
- remaining dominant operator families
- per-role q/k/v/out/fc1/fc2 duration and TFLOP/s
- MAC / MTE1 / MTE2 / FixPipe overlapping ratios
- Block Dim distribution

Memory lane:
- device median / physical tok/s
- main-memory and L1/L2 bandwidth fields exactly as exported
- fields that are zero, missing, or unavailable

Matched 910B2 comparison:
- full-stage 910B2/310P ratio
- physical-token-throughput ratio
- MatMul-only-TFLOP/s ratio
- whether 310P's extra deficit is MatMul, PromptFA, or vector/layout work

Validation:
- exact B1xS2048/4352-NZ contract
- 27 layers / 162 MatMuls per replay / 3 replays per lane
- mapping status

Interpretation:
- single best-supported bottleneck conclusion
- do not add overlapping MAC/MTE ratios
- do not describe bandwidth counters as unique tensor bytes

First blocker or warning:
All artifact paths:
```

After Phase 15 passes, continue directly to Phase 16. Do not edit tracked
files, create a branch, commit, push, or start another shape or page workload.

## Phase 16: production-matched square MatMulV2 deep profile

### 16.0 Purpose and evidence boundary

Phase 15 answers where the complete compiled 27-layer vision stage spends
time. Phase 16 asks why one representative production MatMul behaves as it
does. It directly replays the square attention/output projection:

```text
logical operation:   [2048, 1152] x [1152, 1152]^T + [1152]
activation:          fp16 ND
weight:              fp16 FRACTAL_NZ, materialized before the selected launch
bias:                fp16 ND
output:              fp16 ND
production roles:    q_proj, k_proj, v_proj, out_proj
selected op family:  MatMulV2
```

This direct target is deliberately limited to the square projection. Do not
pretend it represents FC1 or FC2: their eager dispatch can differ from the
compiled graph, so their production timings and TFLOP/s remain Phase 15
full-stack evidence.

The direct replay becomes interpretable only if
`analyze_vision_msprof_op.py` matches its shape, dtype, formats, MatMulV2
dispatch, and Block Dim back to Phase 15's compiled
`vision_linear_executions.csv`. If that validation fails, preserve the
artifacts and stop. Do not weaken the validator, change the kernel-name filter,
or substitute a different eager operator.

Run these six metrics sequentially:

```text
PipeUtilization
ArithmeticUtilization
Memory
MemoryL0
MemoryUB
ResourceConflictRatio
```

Do not request `Occupancy` or `MemoryDetail`: the 310P runtime does not support
those metric families. Consequently:

- Block Dim proves the number of configured logical task blocks, not physical
  core residency by itself.
- Count the nonzero per-block/per-core rows exported by the portable metrics,
  but label this as sampled block/core participation, not `Occupancy`.
- Do not report the 910B `Occupancy`-only composite cycle score or L2 detail as
  if it had been measured on 310P.

This is profiling and analysis only. Do not change tiling, source, formats,
padding, model code, or runtime settings based on the result.

### 16.1 Matched 910B2 reference

These values came from the same direct target on 910B2 at CANN 9.0.0,
torch 2.10.0+cpu, and torch_npu 2.10.0. They are comparison anchors, not 310P
pass thresholds:

```text
captured op:
  MatMulV2_NDNZ_ND_ND_FP16_FP16_FP16_false_true_all_197328
Block Dim:                         24
OpBasic task duration:            29.440 us
compiled q_proj mean:             23.504 us
compiled k_proj mean:             23.122 us
compiled v_proj mean:             23.266 us
compiled out_proj mean:           29.153 us

PipeUtilization, mean across 24 rows:
  Cube ratio:                      61.46%
  MTE1 ratio:                      41.05%
  MTE2 ratio:                      78.25%
  Scalar ratio:                    33.86%
  FixPipe ratio:                    9.17%
  Scalar stalled by MTE1:         18.20 us

ArithmeticUtilization:
  Cube FP16 ratio:                 61.38%
  recorded Cube FP instructions:  60 per row
  recorded Cube INT instructions:  0

Memory, mean across 24 rows:
  GM->L1 bandwidth usage:          37.41%
  core-visible read data:       2,881.25 KiB per row
  core-visible write data:        192.00 KiB per row
  L1 read bandwidth:               98.66 GB/s per row
  L1 write bandwidth:              98.76 GB/s per row
  main-memory read field:          98.60 GB/s per row
  main-memory write field:          6.57 GB/s per row

MemoryL0, mean across 24 rows:
  L0A read / write:               132.22 / 66.11 GB/s
  L0B read / write:               528.89 / 36.36 GB/s
  L0C Cube read / write:          251.22 / 264.44 GB/s

MemoryUB:
  Vector/Scalar UB read/write:      0 for this pure Cube MatMul

ResourceConflictRatio, mean across 24 rows:
  Cube wait ratio:                 84.71%
  MTE1 wait ratio:                 78.19%
  MTE2 wait ratio:                  9.99%
  Vector/UB conflict ratios:        0
  aic_time mean:                   28.041 us
  aic_time population CV:           1.87%
```

Ratios can overlap; never sum them. The bandwidth fields are profiler-local
path counters, not whole-card bandwidth. “Core-visible read data” counts
requests made from each execution block into GM-addressed space; it is not a
count of unique tensor bytes and is not automatically physical HBM traffic.

The 910B exact tiling key was `197328`, but `baseM/baseN/baseK`,
`singleCoreM/N/K`, and buffer depths were not present in the profiler CSVs or
ELF metadata. Do not infer those values from instruction counts. Record the
310P kernel suffix/tiling key, but report the exact tiling tuple as unavailable
unless a CANN artifact explicitly names the fields and values.

### 16.2 Recover and validate the Phase 15 compiled reference

Keep using the exact activated environment and logical `npu:0` from Phase 15.
Set `PHASE15_SUITE_SUMMARY` to the successful Phase 15
`suite_summary.json`. If this is a fresh shell, choose it from the Phase 15
report manually; do not blindly use the newest directory.

```sh
test -f "$PHASE15_SUITE_SUMMARY"

REFERENCE_DIR="$(
  "$PYTHON_BIN" - "$PHASE15_SUITE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["metrics"] == ["pipe", "memory"]
assert set(summary["lanes"]) == {"pipe", "memory"}
reference = Path(summary["combined_analysis_cache"]).resolve()
assert reference.is_dir()
assert (reference / "vision_linear_executions.csv").is_file()
print(reference)
PY
)"

test -f "$REFERENCE_DIR/vision_linear_executions.csv"
test -f "$REFERENCE_DIR/profile_analysis.json"
```

Verify the reference contract mechanically:

```sh
"$PYTHON_BIN" - "$REFERENCE_DIR/profile_analysis.json" <<'PY'
import json
import sys
from pathlib import Path

analysis = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dims = analysis["contract_dims"]
assert dims["batch_size"] == 1
assert dims["sequence_length"] == 2048
assert dims["hidden_size"] == 1152
assert dims["intermediate_size"] == 4352
assert dims["layers"] == 27
assert dims["linear_calls_per_full_stack"] == 162
assert dims["head_padding_mode"] == "runtime"

families = {lane["metric_family"]: lane for lane in analysis["lanes"]}
assert set(families) == {"pipe", "memory"}
for lane in families.values():
    assert lane["mapping"]["status"] == "validated"
    assert lane["matmul_count"] == 486
    assert len(lane["replays"]) == 3
    assert all(replay["matmul_count"] == 162 for replay in lane["replays"])

print("PHASE16_REFERENCE_CONTRACT: PASS")
PY
```

Then record exact capabilities and versions:

```sh
PHASE16_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PHASE16_BASE="310p_square_matmul_deep_${COMMIT_SHORT}_${PHASE16_STAMP}"
PHASE16_DRIVER="$REPO/tmp/09_persistent_page_engine/vision_msprof_driver/$PHASE16_BASE"
PHASE16_LOG="$PHASE16_DRIVER/driver.log"
PHASE16_NPU_LOG="$PHASE16_DRIVER/npu_smi_1s.log"
mkdir -p "$PHASE16_DRIVER"

{
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'reference_dir=%s\n' "$REFERENCE_DIR"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  "$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch
import torch_npu
import torch_npu.profiler as npu_prof

assert torch.npu.is_available()
supported = [str(item) for item in npu_prof.supported_ai_core_metrics()]
required = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "ResourceConflictRatio",
]
missing = [
    name
    for name in required
    if not any(item == name or item.endswith("." + name) for item in supported)
]
print("python_version=" + sys.version.replace("\n", " "))
print("platform=" + platform.platform())
print("torch=" + torch.__version__)
print("torch_npu=" + getattr(torch_npu, "__version__", "<missing>"))
print("device=" + torch.npu.get_device_name(0))
print("soc=" + str(torch_npu.npu.get_soc_version()))
print("supported_ai_core_metrics=" + repr(supported))
assert not missing, f"required profiler metrics are missing: {missing}"
PY
  command -v msprof
  msprof --version 2>&1 || true
  npu-smi info
  df -h "$REPO" "$REFERENCE_DIR"
} >"$PHASE16_DRIVER/environment.txt" 2>&1

cat "$PHASE16_DRIVER/environment.txt"
```

The capability list must contain all six requested metrics. If one is absent,
stop before any capture and report the exact list. Do not replace an
unsupported metric with `Occupancy`, `MemoryDetail`, or a similarly named
PyTorch profiler mode.

### 16.3 Run all six captures sequentially

The checked-in runner owns command records, stdout/stderr, target summaries,
and raw/evidence separation:

```sh
RUNNER="$REPO/09_persistent_page_engine/scripts/run_vision_msprof_op.py"
ANALYZER="$REPO/09_persistent_page_engine/scripts/analyze_vision_msprof_op.py"
test -f "$RUNNER"
test -f "$ANALYZER"

CAPTURE_EVIDENCE_ROOT="$REPO/tmp/09_persistent_page_engine/vision_msprof_op"
CAPTURE_RAW_ROOT="$REPO/.runtime_cache/09_persistent_page_engine_vision_msprof_op"

METRICS=(
  PipeUtilization
  ArithmeticUtilization
  Memory
  MemoryL0
  MemoryUB
  ResourceConflictRatio
)

for metric in "${METRICS[@]}"; do
  case "$metric" in
    PipeUtilization) suffix=pipe ;;
    ArithmeticUtilization) suffix=arithmetic ;;
    Memory) suffix=memory ;;
    MemoryL0) suffix=memoryl0 ;;
    MemoryUB) suffix=memoryub ;;
    ResourceConflictRatio) suffix=conflict ;;
    *) printf 'unhandled metric: %s\n' "$metric"; exit 1 ;;
  esac
  test ! -e "$CAPTURE_EVIDENCE_ROOT/${PHASE16_BASE}_${suffix}"
  test ! -e "$CAPTURE_RAW_ROOT/${PHASE16_BASE}_${suffix}"
done
```

Start a low-overhead NPU monitor and execute the captures. The outer driver
log is intentionally followable; each successful capture also writes its own
command, target summary, and msprof logs.

```sh
(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$PHASE16_NPU_LOG" 2>&1 &
PHASE16_MONITOR_PID=$!
trap 'kill "$PHASE16_MONITOR_PID" 2>/dev/null || true' EXIT

set +e
(
  set -euo pipefail
  for metric in "${METRICS[@]}"; do
    case "$metric" in
      PipeUtilization) suffix=pipe ;;
      ArithmeticUtilization) suffix=arithmetic ;;
      Memory) suffix=memory ;;
      MemoryL0) suffix=memoryl0 ;;
      MemoryUB) suffix=memoryub ;;
      ResourceConflictRatio) suffix=conflict ;;
    esac
    run_name="${PHASE16_BASE}_${suffix}"
    printf '\n[%s] capture start metric=%s run=%s\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" \
      "$metric" "$run_name"
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$RUNNER" \
      --run-name "$run_name" \
      --metric "$metric" \
      --msprof-warm-up 5 \
      --python "$PYTHON_BIN" \
      --evidence-root "$CAPTURE_EVIDENCE_ROOT" \
      --raw-root "$CAPTURE_RAW_ROOT"
    printf '[%s] capture completed metric=%s run=%s\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" \
      "$metric" "$run_name"
  done
) >"$PHASE16_LOG" 2>&1
PHASE16_CAPTURE_EXIT=$?
set -e
printf '%s\n' "$PHASE16_CAPTURE_EXIT" \
  >"$PHASE16_DRIVER/capture_exit_code.txt"

kill "$PHASE16_MONITOR_PID" 2>/dev/null || true
wait "$PHASE16_MONITOR_PID" 2>/dev/null || true
trap - EXIT

test "$PHASE16_CAPTURE_EXIT" -eq 0
```

Luka can follow this from another shell:

```sh
tail -n 100 -f \
  "$REPO/tmp/09_persistent_page_engine/vision_msprof_driver/$PHASE16_BASE/driver.log"
```

Expected time is roughly tens of seconds per metric, not a graph-compilation
run. If any capture fails, preserve all completed captures and report the
first failure plus:

```sh
tail -n 240 "$PHASE16_LOG"
find "$CAPTURE_EVIDENCE_ROOT" -maxdepth 4 \
  -path "*${PHASE16_BASE}*" -type f -print | sort
```

Do not delete or rerun a successful metric merely because a later one failed.
Use a new `PHASE16_BASE` for any approved retry.

### 16.4 Validate every capture against the compiled graph

```sh
for metric in "${METRICS[@]}"; do
  case "$metric" in
    PipeUtilization) suffix=pipe ;;
    ArithmeticUtilization) suffix=arithmetic ;;
    Memory) suffix=memory ;;
    MemoryL0) suffix=memoryl0 ;;
    MemoryUB) suffix=memoryub ;;
    ResourceConflictRatio) suffix=conflict ;;
  esac
  run_name="${PHASE16_BASE}_${suffix}"
  capture_dir="$CAPTURE_EVIDENCE_ROOT/$run_name"
  raw_dir="$CAPTURE_RAW_ROOT/$run_name"
  output_dir="$capture_dir/analysis"

  printf '[%s] validation start metric=%s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "$metric"
  "$PYTHON_BIN" "$ANALYZER" \
    --capture-dir "$capture_dir" \
    --raw-dir "$raw_dir" \
    --reference-dir "$REFERENCE_DIR" \
    --output-dir "$output_dir"
  printf '[%s] validation completed metric=%s report=%s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    "$metric" "$output_dir/report.md"
done | tee "$PHASE16_DRIVER/validation.log"
```

Run this cross-capture contract check:

```sh
"$PYTHON_BIN" - \
  "$CAPTURE_EVIDENCE_ROOT" \
  "$PHASE16_BASE" \
  "$PHASE16_DRIVER/validation.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
base = sys.argv[2]
output = Path(sys.argv[3]).resolve()
captures = {
    "PipeUtilization": "pipe",
    "ArithmeticUtilization": "arithmetic",
    "Memory": "memory",
    "MemoryL0": "memoryl0",
    "MemoryUB": "memoryub",
    "ResourceConflictRatio": "conflict",
}

dispatches = {}
for expected_metric, suffix in captures.items():
    path = root / f"{base}_{suffix}" / "analysis" / "analysis.json"
    assert path.is_file(), path
    analysis = json.loads(path.read_text(encoding="utf-8"))
    assert analysis["status"] == "passed", path
    assert analysis["capture_metric"] == expected_metric
    role = analysis["roles"]["square"]
    validation = role["validation"]
    assert validation["status"] == "passed"
    assert not validation["errors"]
    observed = validation["observed_dispatch"]
    reference = validation["reference_contract"]
    assert observed["operator_name"].startswith("MatMulV2_")
    assert observed["operator_core_type"] == "cube"
    assert observed["block_dim"] == reference["block_dim"]
    assert reference["input_shapes"] == [
        [2048, 1152],
        [72, 72, 16, 16],
        [1152],
    ]
    assert reference["input_formats"] == ["ND", "FRACTAL_NZ", "ND"]
    assert reference["output_shapes"] == [[2048, 1152]]
    assert reference["output_formats"] == ["ND"]
    assert reference["flops"] == 5435817984
    dispatches[expected_metric] = {
        "analysis": str(path),
        "operator_name": observed["operator_name"],
        "block_dim": observed["block_dim"],
        "device_id": observed["device_id"],
        "current_freq": observed["current_freq"],
        "metric_records": analysis["metric_records"],
    }

names = {item["operator_name"] for item in dispatches.values()}
block_dims = {item["block_dim"] for item in dispatches.values()}
assert len(names) == 1, names
assert len(block_dims) == 1, block_dims

result = {
    "status": "passed",
    "operator_name": next(iter(names)),
    "block_dim": next(iter(block_dims)),
    "captures": dispatches,
}
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
```

The validation must pass before interpreting timing or counters. A different
310P tiling-key suffix from 910B is expected and is not a failure. A different
operator family, tensor format, shape, or mismatch with the 310P compiled
reference is a failure.

### 16.5 Build a durable field-level summary

The analyzer preserves missing values and numeric zero separately. Generate
one field-level JSON across all six normalized CSVs:

```sh
"$PYTHON_BIN" - \
  "$CAPTURE_EVIDENCE_ROOT" \
  "$PHASE16_BASE" \
  "$PHASE16_DRIVER/metric_summary.json" <<'PY'
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1]).resolve()
base = sys.argv[2]
output = Path(sys.argv[3]).resolve()
captures = {
    "PipeUtilization": "pipe",
    "ArithmeticUtilization": "arithmetic",
    "Memory": "memory",
    "MemoryL0": "memoryl0",
    "MemoryUB": "memoryub",
    "ResourceConflictRatio": "conflict",
}

result = {"schema_version": 1, "captures": {}}
for capture_metric, suffix in captures.items():
    csv_path = (
        root / f"{base}_{suffix}" / "analysis" / "metric_records.csv"
    )
    assert csv_path.is_file(), csv_path
    fields = defaultdict(
        lambda: {
            "values": [],
            "record_count": 0,
            "missing_count": 0,
            "zero_count": 0,
            "core_ids": set(),
        }
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["category"], row["metric"], row["unit"])
            item = fields[key]
            item["record_count"] += 1
            if row["core_id"]:
                item["core_ids"].add(row["core_id"])
            if row["is_missing"].lower() == "true":
                item["missing_count"] += 1
                continue
            text = row["numeric_value"]
            if not text:
                item["missing_count"] += 1
                continue
            value = float(text)
            if not math.isfinite(value):
                item["missing_count"] += 1
                continue
            item["values"].append(value)
            if value == 0.0:
                item["zero_count"] += 1

    capture_out = []
    for (category, metric, unit), item in sorted(fields.items()):
        values = item.pop("values")
        core_ids = sorted(item.pop("core_ids"))
        entry = {
            **item,
            "category": category,
            "metric": metric,
            "unit": unit,
            "sampled_core_ids": core_ids,
            "sampled_core_count": len(core_ids),
        }
        if values:
            mean = statistics.fmean(values)
            stddev = statistics.pstdev(values)
            entry.update(
                {
                    "mean": mean,
                    "min": min(values),
                    "max": max(values),
                    "population_stddev": stddev,
                    "population_cv": (
                        stddev / abs(mean) if mean != 0 else None
                    ),
                }
            )
        capture_out.append(entry)
    result["captures"][capture_metric] = {
        "metric_records_csv": str(csv_path),
        "fields": capture_out,
    }

output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("PHASE16_METRIC_SUMMARY: PASS")
print(output)
PY
```

This generic summary intentionally retains the exact field names exported by
the 310P CANN version. Do not silently rename `mac` to `cube`, fill missing
fields with zero, or divide/add counters until the field semantics are clear.

### 16.6 Analysis questions the stronger agent must answer

The agent may inspect any generated CSV/JSON, `visualize_data.bin`, captured
ELF, installed CANN manifest, or operator-package metadata read-only. It may
write additional analysis scripts and notes under `$PHASE16_DRIVER`. It must
not edit tracked files or begin an optimization experiment.

Answer all of the following:

1. **Dispatch and format**
   - Exact 310P MatMulV2 kernel name and suffix/tiling key.
   - Block Dim and number of distinct nonzero sampled block/core rows.
   - ND/FRACTAL_NZ/ND contract and whether the compiled Phase 15 graph has any
     explicit `TransData` around its MatMuls.
   - Never attribute the direct target's one-time `npu_format_cast` to the
     selected MatMul; the runner materializes NZ before profiling it.

2. **Per-block balance**
   - Mean, min, max, population standard deviation, CV, and max/min for
     `aic_time` or the closest documented per-block duration.
   - Whether instruction counts and core-visible read/write quantities are
     equal across sampled rows.
   - Separate configured blocks, sampled rows, and physical-core occupancy.

3. **Arithmetic and pipeline**
   - Cube/MAC, MTE1, MTE2, Scalar, and FixPipe active ratios and times.
   - Cube FP16 versus integer instruction/FLOP counters.
   - Cube/MAC, MTE1, MTE2, and MTE3 wait ratios.
   - Explicitly state that active and wait ratios can overlap and are not a
     wall-time partition.

4. **Memory hierarchy**
   - GM->L1 usage, L1 read/write, main-memory fields, and core-visible
     read/write quantities.
   - L0A/L0B/L0C read/write bandwidth.
   - UB traffic/conflicts, with missing-versus-zero counts.
   - Do not multiply a per-row “main memory” field by Block Dim and call it HBM
     bandwidth. Do not call repeated per-core requests unique tensor bytes.

5. **310P versus 910B2**
   - Shape-for-shape ratios for OpBasic duration and each comparable field.
   - Whether 310P's much lower square-MatMul TFLOP/s is best explained by:
     fewer configured cores, lower Cube issue, MTE2/GM->L1 feeding, MTE1/L0
     feeding, wait/dependency structure, tail/imbalance, or a combination.
   - Compare directionally rather than treating different product-local
     bandwidth numbers as a common whole-card denominator.

6. **Tiling**
   - Search the captured and installed CANN metadata for explicit
     `baseM/baseN/baseK`, `singleCore*`, `step*`, and buffer-depth values.
   - If only the tiling key is observable, say exactly that. Do not reverse
     engineer a confident tiling tuple from counts alone.
   - Explain whether IBShare/pure-Cube optimization appears applicable as a
     hypothesis only. Do not enable it.

7. **Evidence quality**
   - Compare direct OpBasic duration with Phase 15 compiled q/k/v/out
     durations, but keep direct kernel replay and compiled graph timing
     separate.
   - Flag any implausible counter, including zero active bandwidth with
     nonzero activity or a bandwidth above a physically meaningful range.
   - Prefer “counter unavailable/inconsistent” over inventing a correction.

### 16.7 Required report and stop condition

Write:

```text
$PHASE16_DRIVER/agent_report.md
```

Use this structure:

```text
310P PHASE 16 SQUARE MATMUL DEEP PROFILE: PASS | PARTIAL | FAIL

Git commit:
Host / exact NPU:
Python / torch / torch_npu:
CANN / ops packages / driver / firmware / msprof:
ASCEND_RT_VISIBLE_DEVICES:
Supported metric families:
Phase 15 compiled reference:

Dispatch:
- exact operator name / tiling-key suffix
- Block Dim
- configured blocks / sampled rows / physical occupancy evidence
- tensor shapes, dtypes, and formats
- explicit TransData evidence

Direct duration versus compiled production:
- OpBasic and visualize duration
- Phase 15 q/k/v/out durations
- why direct and compiled timings differ

Per-block balance:
- aic_time mean/min/max/stddev/CV/max-min
- instruction/work-count balance

Pipe and arithmetic:
- Cube/MAC, MTE1, MTE2, Scalar, FixPipe active ratios
- FP16/INT instruction and FLOP counters
- wait ratios

Memory hierarchy:
- core-visible read/write quantities
- GM->L1 and L1/main-memory fields
- L0A/L0B/L0C fields
- UB traffic/conflicts
- missing, zero, and implausible fields

310P versus matched 910B2:
- duration ratio
- active/wait-ratio comparison
- GM/L1/L0 comparison with scope caveats
- best-supported explanation for the TFLOP/s gap

Tiling and advanced-template evidence:
- observable tiling key
- exact tiling fields found, or explicit unavailable verdict
- IBShare/pure-Cube applicability as hypothesis only

Full-stack connection:
- what Phase 16 explains about Phase 15
- what remains attributable to PromptFA/vector/layout work

First blocker or warning:
Validation JSON:
Metric summary JSON:
Driver/NPU logs:
Six evidence directories:
Six raw profiler directories:
All additional analysis artifacts:
```

The Phase 16 stop condition above is historical and is superseded by the
current Phase 17 instructions below.

## Phase 17: PromptFA sparse-mode comparison

### 17.0 Scope and 910B reference

This phase tests one public PromptFA control, not a new backend. Keep:

```text
27 real PaddleOCR-VL vision layers
FP16
PromptFA
BNSD
model head dimension 72, runtime padded to 80
manual separate RoPE
MLP intermediate size 4352
all 162 Linear weights in FRACTAL_NZ
TorchAir fullgraph cache_compile
```

Only `PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE` changes.

The matched 910B2 reference at commit `1fcb56c` is:

| Shape | Physical tokens | sparse 0 | sparse 1 | sparse-0 change |
|---|---:|---:|---:|---:|
| B1xS2048 | 2048 | 27.78 ms, 73.73k tok/s | 30.64 ms, 66.84k tok/s | -9.3% latency, +10.3% tok/s |
| B4xS512 | 2048 | 26.33 ms, 77.77k tok/s | 26.68 ms, 76.75k tok/s | -1.3% latency, +1.3% tok/s |

The B1xS2048 result was repeated twice from warm caches in interleaved order:
mode 1 measured 30.648 and 30.636 ms; mode 0 measured 27.778 and 27.776 ms.
Treat that as the cross-platform reference, not as an expected 310P result.

### 17.1 Environment and evidence root

Start from a clean checkout. Do not edit tracked files.

```sh
cd /path/to/paddle_ocr_vl_npu
git status --short
git pull --ff-only origin main
git status --short

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase17_promptfa_sparse_$COMMIT_SHORT"
mkdir -p "$OUTPUT_ROOT"
```

Recover the exact Python and model directory already used by successful
Phases 14-16. Do not guess a system Python or redownload the model.

```sh
: "${PYTHON_BIN:?restore the successful Phase 14 Python path}"
: "${MODEL_DIR:?restore the successful local PaddleOCR-VL model path}"
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/config.json"

"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", torchair.__file__)
print("device", torch_npu.npu.get_device_name())
PY
```

Use the same logical `npu:0` selection and CANN environment that passed Phase
14. Confirm no unrelated process owns the selected NPU before continuing.
Never kill a process not created by this phase.

Record the environment:

```sh
{
  printf 'commit=%s\n' "$COMMIT"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'model=%s\n' "$MODEL_DIR"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-}"
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair
print("torch=" + torch.__version__)
print("torch_npu=" + torch_npu.__version__)
print("torchair=" + str(torchair.__file__))
print("device=" + torch_npu.npu.get_device_name())
PY
  npu-smi info
} >"$OUTPUT_ROOT/environment.txt" 2>&1
```

### 17.2 Operator parity and timing probe

Run the committed public-operator probe first. This is fast and does not
compile graphs:

```sh
PROBE_DIR="$OUTPUT_ROOT/operator_probe"
mkdir -p "$PROBE_DIR"
{
  printf '%q ' \
    "$PYTHON_BIN" \
    "$REPO/09_persistent_page_engine/scripts/probes/probe_promptfa_sparse_modes.py" \
    --device npu:0 \
    --output "$PROBE_DIR/results.json"
  printf '\n'
} >"$PROBE_DIR/command.sh"

set -o pipefail
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/probes/probe_promptfa_sparse_modes.py" \
  --device npu:0 \
  --output "$PROBE_DIR/results.json" \
  2>&1 | tee "$PROBE_DIR/stdout.log"
```

Validate mechanically:

```sh
"$PYTHON_BIN" - "$PROBE_DIR/results.json" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
assert "310" in p["environment"]["device_name"], p["environment"]
assert p["environment"]["head_dim"] == 80
for name, result in p["comparisons"].items():
    if name.startswith("ragged/"):
        continue
    assert result["exact"], (name, result)
print("PHASE17_OPERATOR_PARITY: PASS")
PY
```

The no-mask lane is only a dense-attention diagnostic. The packed block-mask
comparison is the relevant correctness check for the production packing
semantics. The ragged `actual_seq_lengths` lane is diagnostic only: its public
interface is a Python list and therefore is not a proposed static-graph
production route.

If mode 0 versus mode 1 is not exact on the dense or packed comparisons, stop
and report the failing comparison. Do not continue to full-stack compiles.

### 17.3 Four compiled full-stack lanes

Use one shared cache root so an already-compatible mode-1 Phase 14 graph can
replay. Sparse mode is part of the cache key, so the two mode-0 shapes are
separate graphs. At most two newly compiled graphs are expected.

```sh
CACHE_ROOT="$REPO/.runtime_cache/09_persistent_page_engine_vision_matmul_lab"
mkdir -p "$CACHE_ROOT"
df -h "$CACHE_ROOT" | tee "$OUTPUT_ROOT/cache_df_before.txt"
```

Define one runner. It emits live progress and preserves the complete log.

```sh
run_sparse_lane() {
  mode="$1"
  batch="$2"
  seq="$3"
  lane="b${batch}_s${seq}_sparse${mode}"
  lane_root="$OUTPUT_ROOT/$lane"
  result_dir="$lane_root/result"
  mkdir -p "$lane_root"

  {
    printf '# commit=%s\n' "$COMMIT"
    printf '# sparse_mode=%s batch=%s sequence_length=%s\n' \
      "$mode" "$batch" "$seq"
    printf '%q ' env \
      "PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE=$mode" \
      "$PYTHON_BIN" \
      "$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py" \
      --model-dir "$MODEL_DIR" \
      --batch-size "$batch" \
      --sequence-length "$seq" \
      --intermediate-size 4352 \
      --weight-format fractal_nz \
      --execution torchair \
      --attention-head-padding runtime \
      --rotary-implementation separate_manual \
      --cache-dir "$CACHE_ROOT" \
      --output-dir "$result_dir" \
      --allow-compile-if-missing \
      --warmup 3 \
      --samples 10 \
      --calls-per-sample 5
    printf '\n'
  } >"$lane_root/command.sh"

  printf '\n[phase17] start %s\n' "$lane"
  set -o pipefail
  PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE="$mode" \
    "$PYTHON_BIN" \
    "$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py" \
    --model-dir "$MODEL_DIR" \
    --batch-size "$batch" \
    --sequence-length "$seq" \
    --intermediate-size 4352 \
    --weight-format fractal_nz \
    --execution torchair \
    --attention-head-padding runtime \
    --rotary-implementation separate_manual \
    --cache-dir "$CACHE_ROOT" \
    --output-dir "$result_dir" \
    --allow-compile-if-missing \
    --warmup 3 \
    --samples 10 \
    --calls-per-sample 5 \
    2>&1 | tee "$lane_root/stdout.log"
  printf '[phase17] done %s\n' "$lane"
}
```

Run in interleaved order to reduce thermal or clock drift:

```sh
run_sparse_lane 1 1 2048
run_sparse_lane 0 1 2048
run_sparse_lane 1 4 512
run_sparse_lane 0 4 512
```

Do not compare first-call/compile wall times. The reported performance field is
the median NPU-event time over 50 warm full-stack calls.

### 17.4 Validate and build the report

Generate a mechanical comparison:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$COMMIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
rows = []
commits = set()
for batch, seq in ((1, 2048), (4, 512)):
    by_mode = {}
    for mode in (0, 1):
        path = root / f"b{batch}_s{seq}_sparse{mode}" / "result" / "run_summary.json"
        data = json.load(open(path))
        assert data["status"] == "ready", path
        assert data["environment"]["commit"] == expected_commit
        commits.add(data["environment"]["commit"])
        assert data["shape"]["batch_size"] == batch
        assert data["shape"]["sequence_length"] == seq
        assert data["shape"]["physical_tokens_per_call"] == 2048
        assert data["shape"]["layers"] == 27
        assert data["shape"]["linear_calls_per_full_stack"] == 162
        assert data["shape"]["candidate_intermediate_size"] == 4352
        assert data["requested"]["execution"] == "torchair"
        assert data["attention"]["implementation"] == "prompt_flash_attention"
        assert data["attention"]["input_layout"] == "BNSD"
        assert data["attention"]["mask_sparse_mode"] == mode
        assert data["attention"]["promptfa_call_head_dim"] == 80
        assert data["attention"]["attention_mask_all_false"]
        assert data["weight_format"]["all_after_are_nz"]
        assert data["compile"]["fullgraph"]
        assert data["compile"]["ge_cache"]
        assert data["numerics"]["measured_output_finite"]
        ms = data["measurements"]["device_event_per_call_ms"]["median"]
        tps = data["measurements"]["physical_tokens_per_s_device_median"]
        by_mode[mode] = (ms, tps, path)

    ms0, tps0, path0 = by_mode[0]
    ms1, tps1, path1 = by_mode[1]
    rows.append({
        "batch": batch,
        "sequence_length": seq,
        "physical_tokens": 2048,
        "sparse0_device_median_ms": ms0,
        "sparse1_device_median_ms": ms1,
        "sparse0_physical_tokens_per_s": tps0,
        "sparse1_physical_tokens_per_s": tps1,
        "sparse0_latency_change_percent": (ms0 / ms1 - 1.0) * 100.0,
        "sparse0_throughput_change_percent": (tps0 / tps1 - 1.0) * 100.0,
        "sparse0_summary": str(path0),
        "sparse1_summary": str(path1),
    })

assert commits == {expected_commit}, commits
payload = {"status": "pass", "rows": rows}
(root / "comparison.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
for row in rows:
    print(json.dumps(row, sort_keys=True))
print("PHASE17_FULL_STACK_CONTRACTS: PASS")
PY
```

Also verify all four summaries report one identical:

- git commit;
- device name and software stack;
- model dimensions;
- execution, layout, D80 padding, RoPE, and NZ-weight contracts.

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Use this report skeleton:

```text
310P PHASE 17 PROMPTFA SPARSE MODE: PASS | PARTIAL | FAIL

Git commit:
Host / exact NPU:
Python / torch / torch_npu / TorchAir / CANN:
Cache root:
Newly compiled graph count:

Operator probe:
- dense B1xS2048 mode0 no-mask / mode0 mask / mode1 mask device us:
- packed B1xS2048 mode0 / mode1 device us:
- dense and packed exact-parity verdict:
- no-mask diagnostic:

Compiled 27-layer table:
| shape | physical tokens | sparse0 ms | sparse1 ms | sparse0 raw tok/s | sparse1 raw tok/s | latency delta | throughput delta |

Contract validation:
- 27 layers / 162 Linears:
- MLP4352 FRACTAL_NZ:
- BNSD PromptFA D80:
- TorchAir fullgraph:
- finite output:

Shape dependence:
- B1xS2048 mode0 effect:
- B4xS512 mode0 effect:
- whether the 310P effect agrees in direction with 910B2:

Interpretation boundary:
- mode0 is the documented 310P sparse mode;
- this phase does not establish production page-level speedup;
- no ATB, actual-sequence-list production route, alternate layout, or other
  attention backend was tested.

First blocker or warning:
Operator results JSON:
Comparison JSON:
Four run summaries:
Four stdout logs:
```

Stop after Phase 17. Do not start another shape, change mask construction,
test another attention backend, edit source, create a branch, commit, or push.
Send Luka the report and exact artifact paths manually.

## Artifact interpretation

For the current task, execute Phase 17 only. The remaining sections are
retained for earlier workflows and are not additional current work.

For every production lane:

- `run_summary.json` contains configuration, setup, E2E, page, layout,
  recognition, packing, token, memory/accounting, and artifact paths;
- `recognition_trace.jsonl` contains one record per crop, including token
  IDs/text/stop reason, crop size, real/physical vision and text tokens, route,
  and per-stage timing;
- `page_regions.jsonl` is the compact page/region manifest;
- `timeline_trace.json` and `timeline.html` contain synchronization-neutral
  host/device spans and waits;
- `npu_smi_1s.log` samples device utilization and HBM during the command;
- `predictions/` contains one Markdown result per page;
- `vision_route_plan.json` records the route selected in that run.

Separate setup/compile/load time from `pipeline_e2e_s`. Use the replay process
for steady-state comparisons. A cache directory merely existing is not replay
proof; compare inventories and logs and confirm the replay did not create a
new graph shape.

After all successful production runs, generate one mechanically derived metrics
file beside every production summary:

```sh
while IFS= read -r summary_path; do
  run_root="$(dirname "$summary_path")"
  test -f "$run_root/recognition_trace.jsonl" || continue
  write_run_metrics \
    "$run_root" \
    "$run_root/metrics_report.json"
done < <(
  find "$OUTPUT_ROOT" -path '*/output/run_summary.json' -print | sort
)
```

Use replay/warm-cache runs for performance comparisons. First/compile runs are
correctness and cache-creation evidence, not steady-state throughput. For each
phase report:

- pages, crops/requests, E2E wall, pages/s, and seconds/page;
- layout device, capture state, layout wall/stages, boxes, and request count;
- vision real/physical/padding tokens, useful fraction, transformer device
  time, effective real tok/s, raw physical tok/s, route counts, and overflows;
- text real/physical/padding tokens, useful fraction, transformer device time,
  effective real tok/s, raw physical tok/s, route counts, and overflows;
- decode generated/effective/raw token slots, decode-control wall, effective
  tok/s, raw physical tok/s, useful/active fraction, graph calls, admissions,
  stop reasons, and generated-length distribution;
- packing group/call counts, fill fractions, KV redistribution bytes/time;
- peak HBM and representative AI Core utilization from `npu_smi_1s.log`;
- absolute and percentage change in E2E and each affected major stage versus
  the immediately preceding replay.

The earlier successful Phase 1-7 CPU-layout outputs are the direct
same-hardware control. Preserve their paths and, where the phase configuration
matches, add CPU-layout versus eager-NPU-layout E2E, pages/s, layout-stage, and
OCR-stage deltas. Do not compare unlike batch sizes, packing modes, or
min-pixels settings.

## Stop condition and required report

For the earlier production-validation task, stop after Phase 8. For the
earlier isolated-vision saturation task, stop after Phase 9. The current task
is governed by Phases 15 and 16 above. Do not start any OCR page workload.

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

Eager-NPU layout gate:
CPU one-page wall / detector / requests / hash:
NPU one-page wall / detector / requests / hash:
CPU-to-NPU speedup:
manifest exact or precise tolerated differences:

Phase 0 environment and TorchAir preflight:

Phase 1 real production smoke:
first / replay:
page / crop / prediction counts:
layout device / graph capture / compatibility patch:
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
pages/s / seconds-page:
page / request / prediction counts:
vision/text routes and overflows:
vision real / physical / padding tokens:
vision effective / raw-physical tokens/s:
text real / physical / padding tokens:
text effective / raw-physical tokens/s:
decode generated / effective / raw slots:
decode effective / raw-physical tokens/s:
active-slot fraction:
layout and other major stage times:
peak HBM:
AI Core utilization:
output parity:
timeline / trace / cache evidence:
same-config CPU-layout replay and NPU-layout delta:

Phase 4 decode arenas:
B4:
B16:
B32 or exact reason skipped:
selected batch and cache:
per-candidate E2E / pages-s / decode rates / useful fraction / HBM:

Phase 5 greedy vision packing:
groups / calls / fill:
vision real / physical / padding tokens and device time:
vision effective / raw-physical tokens/s:
wall / pages-s:
change versus Phase 4:
output parity:

Phase 6 packed text:
groups / calls:
KV redistribution:
text real / physical / padding tokens and device time:
text effective / raw-physical tokens/s:
wall / pages-s:
change versus Phase 5:
output parity:

Phase 7 min_pixels 28224:
vision/text token reduction:
vision/text effective and raw-physical tokens/s:
all major stage times:
wall / pages-s:
change versus Phase 6:
output parity:

Profile-guided status:

Phase 8 layout:
CPU W1 setup / wall / pages-s / seconds-page:
NPU W1 setup / wall / pages-s / seconds-page:
NPU W2 setup / wall / pages-s / seconds-page:
CPU-to-NPU W1 speedup:
stage totals and percentages:
manifest comparison:

Phase 9 isolated vision saturation:
production B1 cache reuse / compatible graph count:
B1 length-curve peak:
fixed-S512 B1 / B2 / B4 raw physical tok/s:
fixed-S1024 B1 / B2 / B4 raw physical tok/s:
fixed-4096-token B1x4096 / B2x2048 / B4x1024:
310P B2/B1 and B4/B1 scaling:
shape-for-shape 910B2 / 310P ratios:
maximum duplicate-pass spread:
unsupported compiled shapes:
small-context underfill conclusion:
scope limitation:

Best stable production configuration:
Best replay wall / pages-s:
Major stage breakdown:
Vision real / physical tokens and effective / raw-physical tokens/s:
Text real / physical tokens and effective / raw-physical tokens/s:
Decode effective / raw slots and effective / raw-physical tokens/s:
Decode useful fraction / graph calls / length distribution:
Peak HBM:
AI Core utilization:

First blocker or warning:
Exact command records:
Artifact paths:
```

Do not describe eight-page speed as full-corpus throughput. Do not report an
alternative runner as production validation.

## Phase 18: 310P-only PromptFA approximate-softmax experiment

### 18.0 What is being tested

CANN 9.0 and 9.1 contain an undocumented x310 PromptFA path selected by
`innerPrecise=4`. On 310P it:

- keeps the existing PromptFA BMM1 and BMM2 kernels;
- keeps the explicit boolean attention mask;
- replaces the normal FP32 online-softmax max/sum state with FP16 state;
- selects the x310 `SoftmaxFlashV2Tmp<half>` implementation; and
- is explicitly rejected on 910B.

This directly targets the measured 310P vector-softmax bottleneck. It is not a
publicly supported precision mode, so both performance and accumulated
27-layer numerics are mandatory gates.

Current upstream CANN also has a host-tiling unit test stating that 310P
accepts `APPROXIMATE_COMPUTATION` and reaches the FP16-softmax adjustment
path. That confirms this is an intentional x310 tiling path, but the test does
not provide a numerical or performance guarantee.

The public torch_npu PromptFA schema does not expose `inner_precise`.
`vision_matmul_lab.py` now installs a process-local TorchAir converter only
when `--promptfa-inner-precise 4` is requested. Eager reference execution
remains ordinary mode-1 PromptFA; only the compiled candidate graph receives
GE `inner_precise=4`.

The converter was validated on 910B2 at commit `532331a`: CANN reached
`prompt_flash_attention_tiling.cpp` and rejected it with exactly:

```text
not support APPROXIMATE_COMPUTATION when curShortSocName is Atlas A2
```

Therefore, a different Python/converter error on 310P is a portability bug;
it is not an expected product rejection.

Keep every other property fixed:

```text
27 real PaddleOCR-VL vision layers
FP16
BNSD PromptFA
sparse_mode=0
full explicit boolean mask
pre_tokens=2147483647
next_tokens=2147483647
model head dimension 72, runtime padded to 80
separate manual RoPE
MLP intermediate size 4352
all 162 Linear weights in FRACTAL_NZ
TorchAir static cache_compile full graph
```

Do not substitute FIA or ATB. Do not change the layout, mask, packing, RoPE,
weights, or shapes.

### 18.1 Update and recover the proven environment

Start from the existing clean work-server checkout:

```sh
cd /path/to/paddle_ocr_vl_npu
git status --short
git pull --ff-only origin main
git status --short

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
REQUIRED_COMMIT="532331a24a562c7455644b21c9ccc130ecdecf52"
git merge-base --is-ancestor "$REQUIRED_COMMIT" "$COMMIT" || {
  printf 'Phase 18 requires commit 532331a or later; got %s\n' "$COMMIT"
  exit 1
}
```

Recover the same variables and NPU environment that passed Phase 14. Do not
guess a Python executable, create another environment, or redownload the
model:

```sh
: "${PYTHON_BIN:?restore the successful Phase 14 Python path}"
: "${MODEL_DIR:?restore the successful local PaddleOCR-VL model path}"
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/config.json"

"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
try:
    import torchair
except ImportError:
    from torch_npu.dynamo import torchair
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", torchair.__file__)
print("device", torch_npu.npu.get_device_name())
print("promptfa_schema", torch.ops.npu.npu_prompt_flash_attention.default._schema)
PY
```

The device name must contain `310`. Confirm that no unrelated process owns the
selected NPU. Never terminate a process that this phase did not start.

Create one new evidence root:

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase18_promptfa_inner4_$COMMIT_SHORT"
CACHE_ROOT="$REPO/.runtime_cache/09_persistent_page_engine_vision_matmul_lab"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"
df -h "$REPO" "$CACHE_ROOT" | tee "$OUTPUT_ROOT/disk_before.txt"

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'model=%s\n' "$MODEL_DIR"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
try:
    import torchair
except ImportError:
    from torch_npu.dynamo import torchair
print("torch=" + torch.__version__)
print("torch_npu=" + torch_npu.__version__)
print("torchair=" + str(torchair.__file__))
print("device=" + torch_npu.npu.get_device_name())
PY
  npu-smi info
} >"$OUTPUT_ROOT/environment.txt" 2>&1
```

Start one background utilization log for the entire phase:

```sh
(
  while true; do
    date --iso-8601=ns 2>/dev/null || date
    npu-smi info
    sleep 1
  done
) >"$OUTPUT_ROOT/npu_smi_1s.log" 2>&1 &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" >"$OUTPUT_ROOT/npu_monitor.pid"

stop_phase18_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
```

### 18.2 Matched lane runner

Define exactly one runner:

```sh
run_inner_lane() {
  inner="$1"
  batch="$2"
  seq="$3"
  lane="b${batch}_s${seq}_inner${inner}"
  lane_root="$OUTPUT_ROOT/$lane"
  result_dir="$lane_root/result"
  test ! -e "$lane_root"
  mkdir -p "$lane_root"

  {
    printf '# commit=%s\n' "$COMMIT"
    printf '# innerPrecise=%s batch=%s sequence_length=%s\n' \
      "$inner" "$batch" "$seq"
    printf '%q ' env \
      PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE=0 \
      PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" \
      "$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py" \
      --model "$MODEL_DIR" \
      --batch-size "$batch" \
      --sequence-length "$seq" \
      --intermediate-size 4352 \
      --weight-format fractal_nz \
      --execution torchair \
      --attention-head-padding runtime \
      --promptfa-inner-precise "$inner" \
      --rotary-implementation separate_manual \
      --cache-dir "$CACHE_ROOT" \
      --output-dir "$result_dir" \
      --allow-compile-if-missing \
      --warmup 3 \
      --samples 10 \
      --calls-per-sample 5
    printf '\n'
  } >"$lane_root/command.sh"
  chmod +x "$lane_root/command.sh"

  printf '\n[phase18] START %s %s\n' \
    "$lane" "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"
  set -o pipefail
  if PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE=0 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" \
    "$REPO/09_persistent_page_engine/scripts/vision_matmul_lab.py" \
    --model "$MODEL_DIR" \
    --batch-size "$batch" \
    --sequence-length "$seq" \
    --intermediate-size 4352 \
    --weight-format fractal_nz \
    --execution torchair \
    --attention-head-padding runtime \
    --promptfa-inner-precise "$inner" \
    --rotary-implementation separate_manual \
    --cache-dir "$CACHE_ROOT" \
    --output-dir "$result_dir" \
    --allow-compile-if-missing \
    --warmup 3 \
    --samples 10 \
    --calls-per-sample 5 \
    2>&1 | tee "$lane_root/stdout.log"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$lane_root/exit_code.txt"
  printf '[phase18] END %s exit=%s %s\n' \
    "$lane" "$status" \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"
  return "$status"
}
```

Each successful lane reports the median of 50 warm full-stack calls. Compile
and first-call wall time are diagnostics only.

### 18.3 First gate: B1xS512

Run the supported mode first, then mode 4:

```sh
run_inner_lane 1 1 512 || {
  stop_phase18_monitor
  exit 1
}
run_inner_lane 4 1 512 || {
  stop_phase18_monitor
  exit 1
}
```

If mode 4 fails, stop. Preserve its full log and report the first CANN/TorchAir
error. Do not try FIA or edit installed packages.

Validate the two summaries:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$COMMIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
rows = {}
for inner in (1, 4):
    path = root / f"b1_s512_inner{inner}" / "result" / "run_summary.json"
    p = json.load(open(path))
    assert p["status"] == "completed", (path, p["status"])
    assert p["environment"]["commit"] == commit
    assert p["environment"]["device_name"].find("310") >= 0
    assert p["shape"]["batch_size"] == 1
    assert p["shape"]["sequence_length"] == 512
    assert p["shape"]["layers"] == 27
    assert p["shape"]["linear_calls_per_full_stack"] == 162
    assert p["shape"]["candidate_intermediate_size"] == 4352
    assert p["attention"]["input_layout"] == "BNSD"
    assert p["attention"]["mask_sparse_mode"] == 0
    assert p["attention"]["pre_tokens"] == 2147483647
    assert p["attention"]["next_tokens"] == 2147483647
    assert p["attention"]["inner_precise"] == inner
    assert p["requested"]["attention_head_padding"] == "runtime"
    assert p["requested"]["rotary_implementation"] == "separate_manual"
    assert p["requested"]["execution"] == "torchair"
    assert p["weight_format"]["all_after_are_nz"]
    assert p["compile"]["fullgraph"]
    assert p["compile"]["ge_cache"]
    assert p["numerics"]["measured_output_finite"]
    rows[inner] = p

for inner, p in rows.items():
    diff = p["numerics"]["measured_output_vs_raw_candidate"]
    print(
        "inner", inner,
        "ms", p["measurements"]["device_event_per_call_ms"]["median"],
        "physical_tok_s",
        p["measurements"]["physical_tokens_per_s_device_median"],
        "max_abs", diff["max_abs"],
        "mean_abs", diff["mean_abs"],
        "finite", diff["left_finite"] and diff["right_finite"],
    )

ms1 = rows[1]["measurements"]["device_event_per_call_ms"]["median"]
ms4 = rows[4]["measurements"]["device_event_per_call_ms"]["median"]
print("inner4_stage_latency_change_pct", (ms4 / ms1 - 1.0) * 100.0)
print("inner4_stage_speedup_x", ms1 / ms4)
print("PHASE18_B1S512_GATE: PASS")
PY
```

Do not reject mode 4 solely for a large maximum difference. Report max and
mean absolute differences separately. Non-finite output is an immediate
failure; mean absolute difference at or above 1.0 is a prominent warning that
requires review, not something to hide.

Even if B1xS512 improves only modestly, continue: PromptFA was about 18% of
that stage, whereas it was about 50% of B1xS2048 on the previous 310P profile.

### 18.4 Long and batched gates

After the B1xS512 gate passes, run:

```sh
run_inner_lane 1 1 2048 || {
  stop_phase18_monitor
  exit 1
}
run_inner_lane 4 1 2048 || {
  stop_phase18_monitor
  exit 1
}
run_inner_lane 1 4 512 || {
  stop_phase18_monitor
  exit 1
}
run_inner_lane 4 4 512 || {
  stop_phase18_monitor
  exit 1
}
```

There are six total lanes in this phase and at most six exact graph
compilations. Do not add shapes or variants.

Stop the monitor after all planned lanes finish, including on a later failure:

```sh
stop_phase18_monitor
df -h "$REPO" "$CACHE_ROOT" | tee "$OUTPUT_ROOT/disk_after.txt"
```

### 18.5 Mechanical comparison

Create the final comparison:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$COMMIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
rows = []
for batch, seq in ((1, 512), (4, 512), (1, 2048)):
    by_inner = {}
    for inner in (1, 4):
        path = (
            root / f"b{batch}_s{seq}_inner{inner}"
            / "result" / "run_summary.json"
        )
        p = json.load(open(path))
        assert p["status"] == "completed", path
        assert p["environment"]["commit"] == commit
        assert "310" in p["environment"]["device_name"]
        assert p["shape"]["batch_size"] == batch
        assert p["shape"]["sequence_length"] == seq
        assert p["shape"]["physical_tokens_per_call"] == batch * seq
        assert p["shape"]["layers"] == 27
        assert p["shape"]["linear_calls_per_full_stack"] == 162
        assert p["shape"]["candidate_intermediate_size"] == 4352
        assert p["attention"]["implementation"] == "prompt_flash_attention"
        assert p["attention"]["input_layout"] == "BNSD"
        assert p["attention"]["mask_sparse_mode"] == 0
        assert p["attention"]["pre_tokens"] == 2147483647
        assert p["attention"]["next_tokens"] == 2147483647
        assert p["attention"]["inner_precise"] == inner
        assert p["weight_format"]["all_after_are_nz"]
        assert p["compile"]["fullgraph"]
        assert p["compile"]["ge_cache"]
        assert p["numerics"]["measured_output_finite"]
        by_inner[inner] = p

    p1 = by_inner[1]
    p4 = by_inner[4]
    ms1 = p1["measurements"]["device_event_per_call_ms"]["median"]
    ms4 = p4["measurements"]["device_event_per_call_ms"]["median"]
    tps1 = p1["measurements"]["physical_tokens_per_s_device_median"]
    tps4 = p4["measurements"]["physical_tokens_per_s_device_median"]
    diff1 = p1["numerics"]["measured_output_vs_raw_candidate"]
    diff4 = p4["numerics"]["measured_output_vs_raw_candidate"]
    rows.append({
        "batch": batch,
        "sequence_length": seq,
        "physical_tokens": batch * seq,
        "inner1_device_median_ms": ms1,
        "inner4_device_median_ms": ms4,
        "inner1_physical_tokens_per_s": tps1,
        "inner4_physical_tokens_per_s": tps4,
        "inner4_latency_change_percent": (ms4 / ms1 - 1.0) * 100.0,
        "inner4_throughput_change_percent": (tps4 / tps1 - 1.0) * 100.0,
        "inner4_stage_speedup_x": ms1 / ms4,
        "inner1_compiled_vs_eager_max_abs": diff1["max_abs"],
        "inner1_compiled_vs_eager_mean_abs": diff1["mean_abs"],
        "inner4_compiled_vs_eager_max_abs": diff4["max_abs"],
        "inner4_compiled_vs_eager_mean_abs": diff4["mean_abs"],
        "inner4_finite": (
            diff4["left_finite"] and diff4["right_finite"]
        ),
    })

payload = {
    "status": "pass",
    "commit": commit,
    "fixed_contract": {
        "sparse_mode": 0,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
        "layout": "BNSD",
        "intermediate_size": 4352,
        "weight_format": "FRACTAL_NZ",
        "head_padding": "runtime 72 to 80",
        "layers": 27,
    },
    "rows": rows,
}
(root / "comparison.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)

lines = [
    "# 310P PromptFA innerPrecise 1 versus 4",
    "",
    "| shape | inner1 ms | inner4 ms | stage speedup | "
    "inner4 physical tok/s | inner4 max abs | inner4 mean abs |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| B{row['batch']}xS{row['sequence_length']} "
        f"| {row['inner1_device_median_ms']:.4f} "
        f"| {row['inner4_device_median_ms']:.4f} "
        f"| {row['inner4_stage_speedup_x']:.4f}x "
        f"| {row['inner4_physical_tokens_per_s']:.1f} "
        f"| {row['inner4_compiled_vs_eager_max_abs']:.6g} "
        f"| {row['inner4_compiled_vs_eager_mean_abs']:.6g} |"
    )
lines.extend(["", "PHASE18_CONTRACTS: PASS", ""])
(root / "comparison.md").write_text("\n".join(lines))
print("\n".join(lines))
PY
```

### 18.6 Report

Write `$OUTPUT_ROOT/agent_report.md` using:

```text
310P PHASE 18 PROMPTFA APPROXIMATE SOFTMAX: PASS | PARTIAL | FAIL

Git commit:
Host / exact NPU:
Python / torch / torch_npu / TorchAir / CANN:
ASCEND_RT_VISIBLE_DEVICES:

Fixed graph contract:
- sparse mode / pre_tokens / next_tokens:
- layout / head dimension / runtime padding:
- MLP width / weight format:
- layers / Linear count:

B1xS512:
- inner1 ms / physical tok/s:
- inner4 ms / physical tok/s:
- stage speedup:
- compiled-vs-eager max abs / mean abs / finite:

B4xS512:
- same fields:

B1xS2048:
- same fields:

Did CANN accept innerPrecise=4 on 310P:
Largest performance gain:
Numerical warning, if any:
Does the result justify a production OCR/token-parity test:

First blocker or warning:
Exact command logs:
Environment:
Comparison JSON/Markdown:
NPU utilization log:
All result summaries:
```

Do not claim OCR accuracy from synthetic-shape encoder outputs. The next step
after a favorable Phase 18 result is one small real-crop/final-token parity
test, then production integration. Do not perform that next step in this
phase.

## Phase 19: page-nine layout binary-resolution failure

> **Superseded; do not run.** This phase was drafted before clarifying that
> page index 8 passes in `layout_owned_lab.py` and fails only when the complete
> OCR pipeline is constructed and run. Its standalone probe remains a useful
> control, but it is not the reproducer or primary investigation boundary.
> Follow Phase 20 instead.

### 19.0 Purpose and current evidence

Run this phase only. The known boundary is:

```text
offset 0, limit 8: PASS
offset 0, limit 9: FAIL
offset 8, limit 1: FAIL
```

Dataset indexing is zero-based, so "page 9" below means `page_index=8`.
The single-page reproducer is the primary case. Do not spend time running the
first eight pages or loading the PaddleOCR-VL recognizer.

The production layout path uses:

```text
OwnedLayoutFrontend
model_backend=transformers
device=npu:0
graph_capture=false
npu_indexput_compat=true
```

The detector input itself is always resized to fixed
`1 x 3 x 800 x 800`. Page content first creates variable NPU shapes in
postprocessing:

```text
sigmoid/topk
  -> threshold/nonzero
  -> reading-order gathers
  -> metadata gathers/index_select
  -> selected-mask index_select
  -> selected-mask sigmoid/threshold
  -> D2H
```

Therefore, do not accept "the layout model failed" as sufficient. Determine
whether the failure is:

1. image read/decode or CPU resize;
2. model/setup binary loading;
3. fixed-shape detector forward;
4. variable-shape metadata selection;
5. variable-shape mask selection/threshold;
6. D2H synchronization;
7. CPU polygon/structural postprocessing.

The phase is complete only when the first failing stage, Python call site,
CANN error code, CANN operator/kernel name, requested binary or library path
(if one exists), and installed-package/OPP evidence are all reported.

### 19.1 Restrictions

- Do not modify production source or installed packages.
- Do not add a model workaround.
- Do not enable ACLGraph or TorchAir.
- Do not load the OCR recognizer.
- Do not delete or invalidate any existing cache.
- Do not download a binary, model, wheel, CANN package, or operator package.
- Do not restart the server or NPU.
- Do not use `pkill` or `killall`.
- Do not repeatedly run the nine-page production pipeline.
- CPU is a diagnostic control only, never the proposed solution.
- A profiler is not the first tool here. CANN's operator profiler assumes a
  working application. First use synchronous execution, CANN plog, and file
  lookup tracing to localize the load failure.

### 19.2 Pull and preflight

Read `CLAUDE.md` and `AGENTS.md`, then:

```sh
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

test -x "$PYTHON_BIN"
test -f \
  "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_page_failure.py"
```

Activate exactly the same NPU environment used by the successful eight-page
run. Reuse its already-resolved paths:

```sh
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$LAYOUT_MODEL"
```

The recognizer model is deliberately irrelevant to this phase.

Create separate compact and heavyweight roots:

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase19_page9_$COMMIT_SHORT"
RAW_ROOT="$REPO/.runtime_cache/310p_phase19_page9_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RAW_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
```

`OUTPUT_ROOT` contains commands, ordinary logs, summaries, and compact error
extracts that may be committed. `RAW_ROOT` contains verbose CANN logs and
strace output and must remain ignored.

Capture the unmodified environment:

```sh
{
  printf 'commit=%s\n' "$COMMIT"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'date=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  env | sort
  "$PYTHON_BIN" - <<'PY'
import json
import os
import platform
import sys

import torch
import torch_npu
import torchvision
import transformers

print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_name": torch.npu.get_device_name(0),
    "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
    "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
}, indent=2, sort_keys=True))
PY
  command -v npu-smi >/dev/null 2>&1 && npu-smi info
  command -v msprof >/dev/null 2>&1 && msprof --version || true
  command -v strace >/dev/null 2>&1 && strace --version | head -n 1 || true
  df -h "$REPO" "$RAW_ROOT"
  if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W 2>/dev/null \
      | grep -Ei 'ascend|cann|torch|driver|firmware' || true
  fi
  if command -v rpm >/dev/null 2>&1; then
    rpm -qa 2>/dev/null \
      | grep -Ei 'ascend|cann|torch|driver|firmware' || true
  fi
} >"$OUTPUT_ROOT/environment.txt" 2>&1
```

Record the installed CANN/OPP roots without recursively copying them:

```sh
{
  for root in \
    "${ASCEND_HOME_PATH:-}" \
    "${ASCEND_OPP_PATH:-}" \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-9.0.0; do
    test -n "$root" || continue
    if test -e "$root"; then
      printf '\nroot=%s\n' "$root"
      readlink -f "$root" || true
      ls -ld "$root"
    fi
  done
  find /usr/local/Ascend -maxdepth 4 \
    \( -name 'version.info' -o -name 'ascend_toolkit_install.info' \) \
    -type f -print 2>/dev/null \
    | sort
} >"$OUTPUT_ROOT/cann_roots.txt" 2>&1
```

### 19.3 Identify and validate the exact page

Record the annotation, basename, file type, bytes, dimensions, and SHA256 for
page indices 0, 7, and 8:

```sh
"$PYTHON_BIN" - \
  "$DATASET_JSON" "$IMAGES_DIR" \
  >"$OUTPUT_ROOT/page_identity.json" <<'PY'
import hashlib
import json
import pathlib
import sys

dataset = pathlib.Path(sys.argv[1])
images = pathlib.Path(sys.argv[2])
pages = json.loads(dataset.read_text(encoding="utf-8"))
result = []
for index in (0, 7, 8):
    annotation = pages[index]
    source = pathlib.Path(annotation["page_info"]["image_path"])
    path = images / source.name
    data = path.read_bytes()
    result.append({
        "page_index": index,
        "annotation_image_path": str(source),
        "resolved_path": str(path.resolve()),
        "exists": path.is_file(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "magic_hex": data[:16].hex(),
        "page_info": annotation["page_info"],
    })
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
PY

while read -r image; do
  file "$image"
done < <(
  "$PYTHON_BIN" - "$OUTPUT_ROOT/page_identity.json" <<'PY'
import json
import sys
for row in json.load(open(sys.argv[1])):
    print(row["resolved_path"])
PY
) >"$OUTPUT_ROOT/page_file_types.txt"
```

Do not rename or "repair" page 9. If CPU decode fails, that is the root cause
branch and the NPU model should not be blamed.

### 19.4 Stage-isolation runner

Use the committed probe. It records and `fsync`s one JSON line before and
after every meaningful stage, then synchronizes the NPU after each operator
group. This changes timing but not the mathematical path; this phase is about
localization, not performance.

```sh
run_page_probe() {
  local name="$1"
  local page_index="$2"
  local device="$3"
  local backend="$4"
  local debug="$5"
  local lane="$OUTPUT_ROOT/$name"
  local raw="$RAW_ROOT/$name"
  local result="$lane/result"
  mkdir -p "$lane" "$raw/cann_logs" "$raw/ascend_work"
  test ! -e "$result"

  local -a debug_env=(
    PYTHONUNBUFFERED=1
    PYTHONFAULTHANDLER=1
    TORCH_NPU_COMPACT_ERROR_OUTPUT=0
    ASCEND_PROCESS_LOG_PATH="$raw/cann_logs"
    ASCEND_WORK_PATH="$raw/ascend_work"
    ASCEND_GLOBAL_EVENT_ENABLE=1
    ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000
  )
  if test "$debug" = 1; then
    debug_env+=(
      ASCEND_LAUNCH_BLOCKING=1
      ASCEND_GLOBAL_LOG_LEVEL=0
      ASCEND_MODULE_LOG_LEVEL=RUNTIME=0:ASCENDCL=0:OP=0:TBE=0
      TORCH_SHOW_CPP_STACKTRACES=1
    )
  fi

  local -a command=(
    "$PYTHON_BIN"
    "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_page_failure.py"
    --dataset-json "$DATASET_JSON"
    --images-dir "$IMAGES_DIR"
    --layout-model "$LAYOUT_MODEL"
    --page-index "$page_index"
    --device "$device"
    --model-backend "$backend"
    --layout-indexput-compat
    --output-dir "$result"
  )

  {
    printf '#!/usr/bin/env bash\n'
    printf '# commit=%s\n' "$COMMIT"
    printf '%q ' env "${debug_env[@]}" "${command[@]}"
    printf '\n'
  } >"$lane/command.sh"
  chmod +x "$lane/command.sh"

  printf '[phase19] START %s %s\n' \
    "$name" "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"
  set -o pipefail
  if env "${debug_env[@]}" "${command[@]}" \
    > >(tee "$lane/run.log") 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$lane/exit_code.txt"
  find "$raw" -type f -printf '%s %p\n' 2>/dev/null \
    | sort -n >"$lane/raw_file_manifest.txt"
  printf '[phase19] END %s exit=%s %s\n' \
    "$name" "$status" \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"
  return 0
}
```

Run fresh processes in this exact order:

```sh
run_page_probe cpu_page8_control 8 cpu transformers 0
run_page_probe npu_page0_control 0 npu transformers 0
run_page_probe npu_page7_control 7 npu transformers 0
run_page_probe npu_page8_normal 8 npu transformers 0
run_page_probe npu_page8_sync_debug 8 npu transformers 1
run_page_probe npu_page8_owned_sync_debug 8 npu owned 1
```

Interpretation:

- CPU page 8 must prove the file, processor, and model are structurally usable.
- NPU pages 0 and 7 prove the same process and fixed model work on controls.
- Normal NPU page 8 preserves the original behavior.
- Synchronous-debug NPU page 8 pins an asynchronous device error to the
  correct Python/operator call and captures debug plog.
- The owned-backend lane uses the same weights and postprocessing but removes
  the Transformers model implementation. It is diagnostic only:
  - Transformers fails, owned passes: inspect Transformers-only model math or
    compatibility patch.
  - both fail at the same postprocess stage: inspect the shared
    variable-shape postprocess/operator package.
  - both fail in fixed forward but at different operators: report both;
    do not infer one common cause.

Do not reject the investigation if the owned backend has small numerical or
box-count differences. It is a boundary probe, not the production oracle.

If `npu_page8_normal` unexpectedly passes, run that same lane in three new
fresh processes (`normal_repeat_1..3`). Do not loop indefinitely. Report the
failure frequency and whether process state or a prior page is required.

Run the exact committed layout frontend once as a non-instrumented
reproduction. This must remain layout-only; it must not load the recognizer:

```sh
EXACT_LANE="$OUTPUT_ROOT/exact_layout_frontend_page8"
EXACT_RAW="$RAW_ROOT/exact_layout_frontend_page8"
mkdir -p \
  "$EXACT_LANE" \
  "$EXACT_RAW/cann_logs" \
  "$EXACT_RAW/ascend_work"

EXACT_COMMAND=(
  "$PYTHON_BIN"
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL"
  --device npu
  --model-backend transformers
  --layout-indexput-compat
  --no-graph-capture
  --offset 8
  --limit 1
  --workers 1
  --no-timeline
  --output-dir "$EXACT_LANE/output"
)

{
  printf '#!/usr/bin/env bash\n'
  printf '# commit=%s\n' "$COMMIT"
  printf '%q ' env \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TORCH_NPU_COMPACT_ERROR_OUTPUT=0 \
    ASCEND_LAUNCH_BLOCKING=1 \
    ASCEND_GLOBAL_LOG_LEVEL=0 \
    ASCEND_MODULE_LOG_LEVEL=RUNTIME=0:ASCENDCL=0:OP=0:TBE=0 \
    ASCEND_PROCESS_LOG_PATH="$EXACT_RAW/cann_logs" \
    ASCEND_WORK_PATH="$EXACT_RAW/ascend_work" \
    ASCEND_GLOBAL_EVENT_ENABLE=1 \
    ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000 \
    "${EXACT_COMMAND[@]}"
  printf '\n'
} >"$EXACT_LANE/command.sh"
chmod +x "$EXACT_LANE/command.sh"

set -o pipefail
if env \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  TORCH_NPU_COMPACT_ERROR_OUTPUT=0 \
  ASCEND_LAUNCH_BLOCKING=1 \
  ASCEND_GLOBAL_LOG_LEVEL=0 \
  ASCEND_MODULE_LOG_LEVEL=RUNTIME=0:ASCENDCL=0:OP=0:TBE=0 \
  ASCEND_PROCESS_LOG_PATH="$EXACT_RAW/cann_logs" \
  ASCEND_WORK_PATH="$EXACT_RAW/ascend_work" \
  ASCEND_GLOBAL_EVENT_ENABLE=1 \
  ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000 \
  "${EXACT_COMMAND[@]}" \
  > >(tee "$EXACT_LANE/run.log") 2>&1; then
  status=0
else
  status=$?
fi
printf '%s\n' "$status" >"$EXACT_LANE/exit_code.txt"
```

The exact lane and staged probe must fail compatibly. If they fail at
apparently different boundaries, explain whether ordinary asynchronous
reporting accounts for it. Do not proceed to a fix until that discrepancy is
understood.

### 19.5 Mechanical stage table and error extraction

Generate a table from whatever each process managed to flush:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" >"$OUTPUT_ROOT/stage_matrix.md" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = (
    "cpu_page8_control",
    "npu_page0_control",
    "npu_page7_control",
    "npu_page8_normal",
    "npu_page8_sync_debug",
    "npu_page8_owned_sync_debug",
)
print("| lane | exit | summary status | last stage | exception |")
print("|---|---:|---|---|---|")
for name in names:
    lane = root / name
    exit_path = lane / "exit_code.txt"
    summary_path = lane / "result" / "summary.json"
    exit_code = exit_path.read_text().strip() if exit_path.exists() else "?"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        failure = summary.get("failure") or {}
        exception = (
            f"{failure.get('exception_type', '')}: "
            f"{failure.get('exception', '')}"
        ).replace("|", "\\|").replace("\n", " ")
        print(
            f"| {name} | {exit_code} | {summary['status']} "
            f"| {summary['last_stage']} | {exception} |"
        )
    else:
        print(f"| {name} | {exit_code} | no summary | process-level | |")
PY
cat "$OUTPUT_ROOT/stage_matrix.md"
```

Extract error context from Python logs and raw CANN logs without discarding the
full raw evidence:

```sh
{
  rg -n -i -C 8 \
    'error|failed|failure|exception|binary|kernel|tiling|op type|optype|soc|not support|not found|dlopen|load.*so|ACL_ERROR|EZ[0-9]{5}|507[0-9]{3}|107[0-9]{3}' \
    "$OUTPUT_ROOT/npu_page8_normal/run.log" \
    "$OUTPUT_ROOT/npu_page8_sync_debug/run.log" \
    "$OUTPUT_ROOT/npu_page8_owned_sync_debug/run.log" \
    "$RAW_ROOT/npu_page8_sync_debug" \
    "$RAW_ROOT/npu_page8_owned_sync_debug" \
    2>/dev/null || true
} >"$OUTPUT_ROOT/page8_error_extract.txt"
```

Read the full synchronized Python traceback and enough surrounding plog to
answer all of the following:

```text
first failing probe stage:
exact Python source file and line:
PyTorch op or torch_npu API:
CANN op type / kernel name:
ACL / PTA / runtime error code:
SoC version requested:
input shapes, dtypes, formats, and important attrs:
binary/library path requested:
does that path exist:
was selection missing, loading rejected, or execution rejected:
```

Do not report only the last `torch.npu.synchronize()` frame. The reason for
`ASCEND_LAUNCH_BLOCKING=1` is to reveal the enqueue call that caused it.

### 19.6 File-resolution trace, conditional on a binary/load error

Run this section only when the traceback or plog actually refers to pulling,
selecting, opening, or loading a binary/kernel/library. If `strace` is absent,
record that fact and continue with plog/OPP inspection; do not install it.

Trace a passing and failing process so ordinary missing locale/Python files can
be subtracted:

```sh
if command -v strace >/dev/null 2>&1; then
  run_strace() {
    local name="$1"
    local page_index="$2"
    local lane="$OUTPUT_ROOT/$name"
    local raw="$RAW_ROOT/$name"
    mkdir -p "$lane" "$raw/cann_logs" "$raw/ascend_work" "$raw/strace"
    test ! -e "$lane/result"

    set -o pipefail
    if strace -ff -s 1024 -e trace=file \
      -o "$raw/strace/file" \
      env \
        PYTHONUNBUFFERED=1 \
        PYTHONFAULTHANDLER=1 \
        TORCH_NPU_COMPACT_ERROR_OUTPUT=0 \
        ASCEND_LAUNCH_BLOCKING=1 \
        ASCEND_GLOBAL_LOG_LEVEL=3 \
        ASCEND_PROCESS_LOG_PATH="$raw/cann_logs" \
        ASCEND_WORK_PATH="$raw/ascend_work" \
        "$PYTHON_BIN" \
        "$REPO/09_persistent_page_engine/scripts/probes/probe_layout_page_failure.py" \
        --dataset-json "$DATASET_JSON" \
        --images-dir "$IMAGES_DIR" \
        --layout-model "$LAYOUT_MODEL" \
        --page-index "$page_index" \
        --device npu \
        --model-backend transformers \
        --layout-indexput-compat \
        --output-dir "$lane/result" \
        > >(tee "$lane/run.log") 2>&1; then
      status=0
    else
      status=$?
    fi
    printf '%s\n' "$status" >"$lane/exit_code.txt"
    rg -n -i \
      'ENOENT|EACCES|\.so([.0-9]*)?[\" ]|\.o[\" ]|kernel|opp|binary|tiling|op_impl|op_proto' \
      "$raw/strace" \
      >"$lane/file_trace_interesting.txt" 2>/dev/null || true
  }

  run_strace strace_page0_control 0
  run_strace strace_page8_failure 8
else
  printf 'strace unavailable; not installed\n' \
    >"$OUTPUT_ROOT/strace_unavailable.txt"
fi
```

Compare the two filtered traces. An `ENOENT` is causal only if:

1. it is unique to, or immediately precedes the page-8 failure;
2. plog or the runtime error references the same operator/path; and
3. it is not followed by a successful fallback open.

Do not call every Python import probe or locale lookup a missing NPU binary.

If the error names an ordinary shared library, and only then, inspect the
actual implicated file/plugin:

```sh
file /exact/path/to/implicated.so
sha256sum /exact/path/to/implicated.so
ldd /exact/path/to/implicated.so
readelf -d /exact/path/to/implicated.so
```

Save this output under `$OUTPUT_ROOT/implicated_library.txt`. Never run `ldd`
over every shared library in CANN.

### 19.7 Installed OPP/kernel audit

After obtaining the exact operator/kernel name from plog, search the installed
operator package narrowly. Substitute the exact value; do not guess:

```sh
OP_NAME='REPLACE_WITH_EXACT_CANN_OP_OR_KERNEL_NAME'
test "$OP_NAME" != REPLACE_WITH_EXACT_CANN_OP_OR_KERNEL_NAME

{
  printf 'operator=%s\n' "$OP_NAME"
  printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-}"
  for root in \
    "${ASCEND_OPP_PATH:-}" \
    "${ASCEND_HOME_PATH:-}/opp" \
    /usr/local/Ascend/ascend-toolkit/latest/opp \
    /usr/local/Ascend/cann-9.0.0/opp; do
    test -d "$root" || continue
    printf '\nroot=%s\n' "$root"
    find "$root" -type f \
      \( -iname "*${OP_NAME}*" -o -iname '*ops-info*.json' \) \
      -print 2>/dev/null | sort
  done
} >"$OUTPUT_ROOT/implicated_op_files.txt" 2>&1
```

Then use `rg -n -i -C 4 "$OP_NAME"` only inside the relevant metadata/config
directories found above. Record:

- whether the op is registered for the exact 310P SoC;
- supported dtype, format, shape, and attribute constraints;
- selector/tiling implementation path;
- selected `.o`/kernel binary path, if plog exposes one;
- whether the binary exists and is readable;
- file architecture and SHA256;
- owning CANN/operator package if package-manager metadata can identify it.

If the error is numeric, search the installed runtime headers for the exact
code and save the matching enum/description:

```sh
ERROR_CODE='REPLACE_WITH_EXACT_ERROR_CODE'
rg -n -C 3 "$ERROR_CODE" \
  /usr/local/Ascend/*/include \
  /usr/local/Ascend/*/runtime/include \
  2>/dev/null \
  >"$OUTPUT_ROOT/error_code_header_matches.txt" || true
```

Distinguish these failure classes explicitly:

```text
operator not registered
binary selector not registered
kernel not registered
no binary variant for SoC/shape/dtype/format/attrs
binary file absent
binary file present but unreadable
plugin/shared-library dependency missing
binary incompatible with runtime/driver/SoC
tiling failure
kernel launch/execution failure mislabeled as load failure
```

### 19.8 One additional synchronization fallback

Only if `ASCEND_LAUNCH_BLOCKING=1` still produces an unrelated asynchronous
stack, repeat the single page-8 probe once with:

```text
TASK_QUEUE_ENABLE=0
ASCEND_LAUNCH_BLOCKING=1
```

Record this as a diagnostic-only change. Do not use it for performance or call
it a solution.

### 19.9 Source mapping

Map the observed stage back to the committed code:

```text
page decode / CPU resize / H2D:
  pipeline/layout_frontend.py

fixed detector forward and 310P IndexPut compatibility:
  pipeline/layout_frontend.py
  pipeline/layout_model_runtime.py

project-owned comparison model:
  pipeline/owned_layout_model/

variable score/box/mask selection:
  pipeline/layout_model_runtime.py
  _post_process_selected_masks_only

stage-isolation probe:
  scripts/probes/probe_layout_page_failure.py
```

Quote exact file/line locations from the checked-out commit in the report.

### 19.10 Stopping condition and report

Stop after root-cause evidence is collected. Do not implement a fix in this
phase, even if it looks easy. Luka wants to discuss the diagnosis first.

Write `$OUTPUT_ROOT/agent_report.md`:

```text
310P PHASE 19 PAGE-9 LAYOUT FAILURE: ROOT CAUSE FOUND | BOUNDED | UNRESOLVED

Git commit:
Host / exact NPU:
Python / torch / torch_npu / torchvision / Transformers / CANN:

Reproducer:
- page index / annotation path / resolved path / SHA256:
- offset 0 limit 8 status:
- offset 8 limit 1 status:
- deterministic across fresh processes:

Control matrix:
| lane | exit | last passed stage | first failing stage | key result |

Exact failure:
- Python source and line:
- torch/torch_npu operation:
- CANN op and kernel:
- error code and official/local meaning:
- input shapes / dtype / format / attrs:
- requested SoC and binary/library path:

Binary/package audit:
- file exists:
- readable:
- architecture/hash:
- owning package and version:
- plugin dependencies:
- OPP registration/selector support for 310P:

Passing-page versus page-9 difference:
- original image size/format:
- fixed model input shape:
- selected detection count:
- first NPU shape that differs:
- file lookup unique to failing process:

Transformers versus owned backend:
- result:
- boundary this proves:

Root-cause statement:
- confirmed facts:
- strongest inference:
- remaining uncertainty:
- confidence:

Potential fixes to discuss (do not implement):
1.
2.

Evidence:
- stage_matrix.md
- page8_error_extract.txt
- synchronized traceback:
- compact CANN log excerpt:
- strace comparison, if available:
- OPP/kernel audit:
- raw verbose evidence root:
```

The root-cause statement must be specific. Examples of insufficient reports:

```text
"page 9 is incompatible"
"the layout model cannot pull a binary"
"CANN has an issue"
"an NPU op failed"
```

If no exact binary path is ever requested, say that plainly. The original
wording may describe an operator-selector or kernel-registration failure
rather than a filesystem download/load attempt.

Do not edit tracked files, create a branch, commit, or push from the 310P work
server. Keep both evidence roots local to that server. Paste back
`agent_report.md`, `stage_matrix.md`, the compact error extract, and the exact
paths to the full raw evidence. Inspect file sizes and redact only
credentials/tokens—not technical paths, operator names, shapes, or error
codes.

## Phase 20: integrated OCR page-nine failure

### 20.0 Correction and exact investigation boundary

Run this phase only. Phase 19 made the wrong assumption that page index 8
failed in the standalone layout frontend. The corrected evidence is:

```text
standalone layout, offset 8, limit 1: PASS
complete OCR pipeline, offset 0, limit 8: PASS
complete OCR pipeline, offset 0, limit 9: FAIL
complete OCR pipeline, offset 8, limit 1: FAIL
```

Do not investigate `layout_owned_lab.py` as though it were the failing
application. Its page-index-8 output is a control and a request manifest.

The production ordering also matters. `run_omnidocbench.py` constructs
`OwnedLayoutFrontend`, then the complete `ContinuousRecognizer`, then
`OwnedPageEngine`. In `OwnedPageEngine.run`, page preparation runs on a
dedicated layout stream, and `layout_stream.synchronize()` completes before
the prepared page enters the OCR request source. Therefore, for
`--offset 8 --limit 1`, there is no next-page layout inference executing
concurrently with OCR. The one-page failure must be separated into:

1. layout fails only when the already-constructed recognizer and its device
   state are resident;
2. layout succeeds, then one of page index 8's crop requests fails in
   recognition;
3. a delayed asynchronous error is attributed to the wrong Python call.

Cross-page layout/OCR overlap remains a possible additional problem for the
nine-page streaming run, but it cannot be the necessary cause if the
single-page integrated reproducer fails in the same way.

### 20.1 Restrictions

- Do not edit tracked source, create a branch, commit, or push.
- Do not reinstall Python/CANN packages or download another model/operator
  package.
- Do not delete or replace TorchAir, ACLGraph, OPP, or model caches.
- Do not apply a model workaround or CPU fallback.
- Do not run a performance sweep or compile new shapes deliberately.
- Do not call the problem a layout failure unless the synchronized traceback
  and partial artifacts show that `prepare_page` did not finish.
- Preserve the first causal error. A later shutdown/accounting exception is
  not the root cause.

Use the exact environment and Python established by the preceding phases.
Confirm `git status --short` before starting, but do not clean unrelated
server-local evidence.

### 20.2 Recover the existing evidence before rerunning

First locate the exact successful and failing production runs already made.
Do not reconstruct their commands from memory:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
REPO=$PWD
RAW_ROOT="$REPO/.runtime_cache/phase20_integrated_page9"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/phase20_integrated_page9"
mkdir -p "$RAW_ROOT" "$OUTPUT_ROOT"

find "$REPO/tmp" -type f \
  \( -name 'command.sh' -o -name 'command.txt' -o -name 'run.log' \
     -o -name 'manifest.json' -o -name 'run_summary.json' \) \
  -print > "$OUTPUT_ROOT/candidate_artifacts.txt"

rg -n --fixed-strings -- '--offset 8' "$REPO/tmp" \
  > "$OUTPUT_ROOT/offset8_hits.txt" || true
rg -n --fixed-strings -- '--limit 8' "$REPO/tmp" \
  > "$OUTPUT_ROOT/limit8_hits.txt" || true
rg -n --fixed-strings -- '--limit 9' "$REPO/tmp" \
  > "$OUTPUT_ROOT/limit9_hits.txt" || true
```

Identify these three exact run roots:

```text
A: passing complete pipeline, offset 0 / limit 8
B: failing complete pipeline, offset 0 / limit 9
C: failing complete pipeline, offset 8 / limit 1
```

For each, copy its exact command text, commit, device, environment versions,
cache paths, layout flags, vision route, packing settings, text route, decode
batch/cache length, and output path into:

```text
$OUTPUT_ROOT/existing_run_matrix.md
```

Diff the commands argument by argument. Any difference beyond offset, limit,
and output directory must be reported before treating the runs as a valid
boundary.

Read the complete failing log from its first error onward. Also inspect every
partial artifact that exists:

```text
manifest.json
timeline_trace.json
timeline.html
recognition_trace.jsonl
layout_mask_guard.json
page_regions.jsonl
predictions/
```

`run_omnidocbench.py` writes the mask-guard snapshot and timeline from a
`finally` block, even when the integrated run aborts. The recognition trace is
flushed after every completed crop. A missing final `run_summary.json` is
expected on failure.

### 20.3 Establish the request manifest and last completed work

Use the already-passing standalone layout result for page index 8, or rerun
only that existing layout command if its artifacts are missing. Its
`requests.jsonl` is the expected ordered crop manifest. Do not run the
Phase-19 staged probe unless standalone layout unexpectedly stops passing.

Create `$OUTPUT_ROOT/existing_failure_analysis.json` and
`existing_failure_analysis.md` by comparing:

- all expected page-index-8 requests in standalone `requests.jsonl`;
- all completed requests in the failing integrated
  `recognition_trace.jsonl`;
- the last 100 timeline events ordered by host start/end time;
- the last event for each `flow_id`;
- the producer thread and main-thread traceback.

The first request without a final recognition trace is only a candidate.
Prefill, decode, and result completion can be interleaved. Use timeline
`flow_id`, stage name, track, lane, and thread together to identify the last
submitted and last completed stage.

The analysis must answer, before any new run:

```text
Did page-index-8 layout finish in the integrated run?
How many crops did layout produce?
How many crops entered OCR?
How many crops completed OCR?
What was the last completed OCR stage and request?
Was the first error raised by owned-page-producer or the recognizer/main thread?
Does the error precede or follow "Prepare owned page" completion?
```

### 20.4 One exact synchronized production rerun

Use run C's exact saved production command. Change only its output directory
and add diagnostic environment variables. Do not guess or rewrite its
performance/graph flags. Reuse its exact warm compiler-cache paths.

Before launching, record:

```sh
git rev-parse HEAD > "$OUTPUT_ROOT/git_commit.txt"
git status --short > "$OUTPUT_ROOT/git_status.txt"
df -h "$REPO" > "$OUTPUT_ROOT/df_before.txt"
npu-smi info > "$OUTPUT_ROOT/npu_smi_before.txt" 2>&1
find <each-exact-cache-root-from-command> -maxdepth 3 -type f -printf \
  '%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' \
  > "$OUTPUT_ROOT/cache_inventory_before.txt"
```

Use a fresh raw CANN-log directory:

```sh
RUN_RAW="$RAW_ROOT/synchronized_offset8_limit1"
RUN_OUT="$OUTPUT_ROOT/synchronized_offset8_limit1"
mkdir -p "$RUN_RAW/cann" "$RUN_OUT"

export PYTHONFAULTHANDLER=1
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_NPU_COMPACT_ERROR_OUTPUT=0
export ASCEND_PROCESS_LOG_PATH="$RUN_RAW/cann"
export ASCEND_WORK_PATH="$RUN_RAW"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_MODULE_LOG_LEVEL='RUNTIME=0:ASCENDCL=0:OP=0:TBE=0'
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000
```

Run the exact C command with `--output-dir "$RUN_OUT"` and preserve live
progress:

```sh
set -o pipefail
<EXACT SAVED OFFSET-8 LIMIT-1 COMMAND, OUTPUT DIR CHANGED ONLY> \
  2>&1 | tee "$RUN_OUT/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$RUN_OUT/exit_code.txt"
```

In another shell, sample NPU state until the process exits:

```sh
while kill -0 <EXACT_RUN_PID> 2>/dev/null; do
  printf '\n===== %s =====\n' "$(date -Ins)"
  npu-smi info
  sleep 0.5
done > "$RUN_OUT/npu_smi_monitor.log" 2>&1
```

Record the run PID rather than matching and killing unrelated Python
processes. Do not use `pkill` or `killall`.

If launch blocking still reports an obviously delayed unrelated call, repeat
this one single-page run once with:

```text
TASK_QUEUE_ENABLE=0
ASCEND_LAUNCH_BLOCKING=1
```

Label it diagnostic-only. Do not use it for performance.

### 20.5 Classify the failure before choosing another experiment

#### Branch A: layout does not finish, but standalone layout passes

This means "recognizer already resident" is part of the reproducer; it still
does not prove simultaneous layout/OCR execution. Record:

- NPU allocated/reserved memory after layout frontend construction;
- NPU allocated/reserved memory after recognizer construction;
- NPU memory immediately before page preparation and at failure;
- exact layout op, input shape, dtype, tensor format, workspace request,
  error code, and kernel/binary path;
- whether the same op succeeds in the standalone frontend with the same
  input and shape.

Then run one boundary control with the exact production command plus
`--layout-device cpu`. This is diagnostic only. If CPU layout reaches OCR and
fails at the same OCR stage, layout was not the causal component. If CPU
layout completes the page, report that recognizer residency changes the NPU
layout route or resource state; do not present CPU layout as the fix.

#### Branch B: layout finishes and OCR fails

Map the exact failing crop/group to the standalone request manifest and
timeline. Identify:

```text
request_id and crop/block index
crop dimensions and pixel profile
real and physical vision-token lengths
vision route: eager/compiled, single/packed/batched, B/S bucket, graph key
real and physical text-prefill lengths
text route and graph key
decode batch/cache shape and graph key
last successful stage
first failing torch / torch_npu operation
CANN op/kernel, shapes, dtypes, formats, attrs, and error code
```

Compare that route with all routes exercised by the passing first eight
pages. State whether page index 8 is the first user of a new vision bucket, packed
group, text bucket, decoder shape, graph artifact, or native operator shape.
Check cache mtimes before and after: the failing run must not silently compile
a new graph while being described as warm-cache replay.

If the error mentions loading, pulling, resolving, registering, or selecting
a binary, do the conditional binary audit in section 20.6. Do not assume the
word "binary" means a network download.

#### Branch C: single-page integrated run passes under exact synchronization

Only in this case test cross-page behavior. Run the exact nine-page production
command twice:

```text
streaming default
same command plus --preprocess-all-pages-first
```

Everything else, including caches and route flags, must remain identical.

- layout-first passes, streaming fails: cross-page concurrency or stream/
  resource interaction is implicated;
- both fail at the same OCR operation: concurrency is not the root cause;
- both pass only with launch blocking: timing changes are masking a race, not
  fixing it.

Do not run this pair if the synchronized single-page reproducer already fails.

### 20.6 Conditional operator/binary audit

Run this section only after an exact failing operation has been identified.
Extract from CANN logs:

```text
first error code and its decoded meaning
operator and kernel name
SoC/version selected
input and output shapes
dtypes and storage formats
attributes/tiling key/workspace
requested file or shared library, if an actual path exists
selector/registration failure text, if no file path exists
```

Distinguish these cases explicitly:

```text
no registered operator/kernel variant for the SoC or attributes
registered variant exists but a binary file is absent
file exists but is unreadable or has a missing shared-library dependency
binary is incompatible with driver/runtime/SoC
tiling/workspace generation failed before launch
kernel launched and failed during execution
an earlier asynchronous failure surfaced at a later load/launch call
```

If and only if an actual file lookup remains ambiguous, run the exact failing
single-page command once under:

```sh
strace -ff -tt -e trace=openat,access,statx,readlink \
  -o "$RUN_RAW/strace" \
  <EXACT COMMAND>
```

Compare with a passing integrated page/crop route, filtering only the exact
operator/kernel/library names. Audit the current `ASCEND_OPP_PATH` and owning
package; do not copy files into it or alter registration.

Do not start `msprof` while the application still fails. First make the
application boundary and first causal error deterministic.

### 20.7 Stop and report

Stop after root-cause evidence is collected. Do not implement a fix. Write:

```text
$OUTPUT_ROOT/agent_report.md
$OUTPUT_ROOT/existing_run_matrix.md
$OUTPUT_ROOT/existing_failure_analysis.md
$OUTPUT_ROOT/error_extract.txt
```

The report must use this form:

```text
310P PHASE 20 INTEGRATED PAGE-9 FAILURE: ROOT CAUSE FOUND | BOUNDED | UNRESOLVED

Git commit:
Host / exact NPU:
Python / torch / torch_npu / torchvision / Transformers / CANN:

Verified boundary:
- standalone layout offset 8 / limit 1:
- full pipeline offset 0 / limit 8:
- full pipeline offset 0 / limit 9:
- full pipeline offset 8 / limit 1:
- commands equivalent except offset/limit/output:

Integrated ordering:
- recognizer constructed before layout preparation:
- page-index-8 layout completed before OCR consumed its requests:
- cross-page overlap required to reproduce: yes/no/unknown

Failure location:
- traceback thread:
- page preparation completed:
- expected crops:
- crops entering OCR:
- crops completed:
- failing request/group:
- last successful stage:
- first failing stage:
- torch / torch_npu call:
- CANN op/kernel:
- shape/dtype/format/attrs:
- error code:
- graph route/cache key:
- requested binary/library path, or explicitly none:

Passing-first-8 versus page-index-8 difference:
- first new crop/shape/bucket/group/operator:
- warm-cache status:

Recognizer-residency/resource evidence:
- allocated/reserved memory after frontend:
- allocated/reserved memory after recognizer:
- memory/workspace at failure:
- implicated: yes/no/unknown

Root-cause statement:
- confirmed facts:
- strongest inference:
- remaining uncertainty:
- confidence:

Potential fixes to discuss (do not implement):
1.
2.

Evidence:
- existing run matrix:
- existing partial-artifact analysis:
- synchronized traceback:
- compact CANN error extract:
- full raw CANN root:
- cache inventories:
- NPU monitor:
- conditional strace/OPP audit:
```

An acceptable bounded result identifies whether the failure is before or
after the layout-to-recognition handoff, plus the exact first failing
request/stage/operator and evidence path. These are not acceptable:

```text
"page 9 layout is broken"
"the model cannot pull a binary"
"CANN failed somewhere in OCR"
"limit 9 uses too much memory"
```

If no exact binary path is requested, say so. Paste back the four compact
report files and the paths to the raw evidence. Redact credentials only; keep
operator names, shapes, graph/cache paths, software versions, and error codes.

## Phase 21: packed-text graph versus exact KV-copy isolation

### 21.0 Purpose and corrected Phase-20 conclusion

Run this phase only. Phase 20 proved:

```text
standalone page-index-8 layout: PASS
integrated page-index-8 run: FAIL
cross-page layout/OCR overlap required: NO
failure surfaces at or immediately before packed-text KV redistribution
AICore error: MTE DDR address out of range
missing binary/library: NO
```

Phase 20 did **not** yet prove:

- that the compiled packed-text graph completed successfully;
- that `GatherV2_216` was lowered from the Python slice-copy rather than being
  an internal node of the preceding graph;
- that all packed offsets, lengths, views, storage bounds, formats, and cache
  lifetimes were valid;
- that the problem is a CANN implementation defect rather than our runtime
  metadata or ordering.

This phase resolves those questions with one graph-boundary run, at most one
per-copy run, and at most three tiny fresh-process replay lanes. It does not
benchmark performance or implement a fix.

### 21.1 Diagnostic implementation

The committed probe is:

```text
09_persistent_page_engine/scripts/probes/
  probe_text_kv_redistribution_failure.py
```

It executes the real `run_omnidocbench.py` entrypoint with the original
production arguments. It monkeypatches only diagnostic boundaries before that
entrypoint is executed. Normal production source and cache identities remain
unchanged.

Every event is appended and `fsync`ed to `events.jsonl`. This is deliberate:
the last `copy_before` record must survive even if an AICore exception poisons
or terminates the process.

The two integrated barrier strategies are:

```text
graph_only
  device-wide synchronization immediately after packed graph execution;
  production redistribute_cache remains byte-for-byte unchanged

full
  same graph barrier, plus metadata validation, a durable pre-copy record,
  and device-wide synchronization after every individual KV copy
```

Replay lanes reconstruct the recorded source/destination shapes, strides,
storage offsets, dtype, and NPU format with fresh allocations. Each lane must
run in a fresh process because an AICore exception can poison the runtime.

### 21.2 Pull and preflight

Read `CLAUDE.md` and `AGENTS.md`, then:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PROBE="$REPO/09_persistent_page_engine/scripts/probes/probe_text_kv_redistribution_failure.py"

test -x "$PYTHON_BIN"
test -f "$PROBE"
"$PYTHON_BIN" "$PROBE" --help

OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase21_kv_copy_$COMMIT_SHORT"
RAW_ROOT="$REPO/.runtime_cache/310p_phase21_kv_copy_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RAW_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
```

Activate exactly the environment used by the Phase-20 synchronized
page-index-8 run. Record:

```sh
{
  printf 'commit=%s\n' "$COMMIT"
  printf 'host=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'date=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  "$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
import torch_npu

print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_name": torch.npu.get_device_name(0),
}, indent=2))
PY
  npu-smi info
} 2>&1 | tee "$OUTPUT_ROOT/preflight.log"
```

Do not continue if the intended NPU is occupied by an unrelated process.
Never use `pkill` or `killall`.

### 21.3 Recover the exact Phase-20 production arguments

Use the exact successful command construction from Phase 20's
`synchronized_offset8_limit1` run. Do not reconstruct cache roots or
optimization flags from defaults.

Create a Bash array named `PRODUCTION_ARGS` containing every argument after
`run_omnidocbench.py`, except remove the old `--output-dir <path>` pair. It
must still contain:

```text
--offset 8
--limit 1
--layout-device npu
--no-layout-graph-capture
the exact Phase-20 dataset, image, and model paths
the exact vision/text/decode configuration
the exact four warm cache roots
```

Record the resulting array unambiguously:

```sh
{
  printf '%q ' "$PYTHON_BIN" \
    "$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py" \
    "${PRODUCTION_ARGS[@]}"
  printf '\n'
} > "$OUTPUT_ROOT/original_production_command_without_output_dir.sh"
```

Compare it to the Phase-20 saved command before running. The only differences
may be the wrapper script, diagnostics, and new output directory. Do not
change packing, buckets, cache length, batch size, min-pixels, or page
selection.

### 21.4 Run A: graph boundary only

This run proves whether the packed graph itself completes before the original
redistribution begins.

```sh
RUN_NAME=graph_only
RUN_OUT="$OUTPUT_ROOT/$RUN_NAME"
RUN_RAW="$RAW_ROOT/$RUN_NAME"
mkdir -p "$RUN_OUT/diagnostic" "$RUN_RAW/cann"

export PYTHONFAULTHANDLER=1
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_NPU_COMPACT_ERROR_OUTPUT=0
export ASCEND_PROCESS_LOG_PATH="$RUN_RAW/cann"
export ASCEND_WORK_PATH="$RUN_RAW"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_MODULE_LOG_LEVEL='RUNTIME=0:ASCENDCL=0:OP=0:TBE=0'
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000

{
  printf '%q ' "$PYTHON_BIN" "$PROBE" \
    --mode integrated \
    --integrated-barriers graph_only \
    --diagnostic-dir "$RUN_OUT/diagnostic" \
    -- "${PRODUCTION_ARGS[@]}" \
    --output-dir "$RUN_OUT/pipeline"
  printf '\n'
} > "$RUN_OUT/command.sh"

set +e
set -o pipefail
"$PYTHON_BIN" "$PROBE" \
  --mode integrated \
  --integrated-barriers graph_only \
  --diagnostic-dir "$RUN_OUT/diagnostic" \
  -- "${PRODUCTION_ARGS[@]}" \
  --output-dir "$RUN_OUT/pipeline" \
  2>&1 | tee "$RUN_OUT/run.log"
GRAPH_ONLY_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$GRAPH_ONLY_EXIT" > "$RUN_OUT/exit_code.txt"
```

Summarize the durable event tail:

```sh
"$PYTHON_BIN" - "$RUN_OUT/diagnostic/events.jsonl" \
  > "$RUN_OUT/event_summary.txt" <<'PY'
import collections
import json
import sys

path = sys.argv[1]
events = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
counts = collections.Counter(event["event"] for event in events)
print("records:", len(events))
print("counts:", json.dumps(counts, indent=2, sort_keys=True))
print("tail:")
for event in events[-20:]:
    print(json.dumps({
        "sequence": event["sequence"],
        "event": event["event"],
        "graph_call": event.get("graph_call"),
        "exception": event.get("exception"),
    }, ensure_ascii=False))
PY
cat "$RUN_OUT/event_summary.txt"
```

Interpret and stop/continue exactly as follows:

```text
packed_graph_sync_failed
  STOP. The failure is inside the compiled packed graph, before redistribution.
  Do not run the full-copy probe or replay lanes.

packed_graph_sync_passed, then production fails at original redistribute_cache
  CONTINUE to Run B. The graph is exonerated and the copy path remains causal.

complete page passes
  STOP. The explicit graph-to-copy barrier changes the result. This is an
  ordering/dependency problem, not a demonstrated invalid GatherV2 shape.
```

### 21.5 Run B: identify the exact member/layer/KV copy

Run this only if Run A records `packed_graph_sync_passed` and the original
redistribution still fails.

Use a fresh process and fresh CANN-log directory:

```sh
RUN_NAME=full_per_copy
RUN_OUT="$OUTPUT_ROOT/$RUN_NAME"
RUN_RAW="$RAW_ROOT/$RUN_NAME"
mkdir -p "$RUN_OUT/diagnostic" "$RUN_RAW/cann"

export ASCEND_PROCESS_LOG_PATH="$RUN_RAW/cann"
export ASCEND_WORK_PATH="$RUN_RAW"

{
  printf '%q ' "$PYTHON_BIN" "$PROBE" \
    --mode integrated \
    --integrated-barriers full \
    --diagnostic-dir "$RUN_OUT/diagnostic" \
    -- "${PRODUCTION_ARGS[@]}" \
    --output-dir "$RUN_OUT/pipeline"
  printf '\n'
} > "$RUN_OUT/command.sh"

set +e
set -o pipefail
"$PYTHON_BIN" "$PROBE" \
  --mode integrated \
  --integrated-barriers full \
  --diagnostic-dir "$RUN_OUT/diagnostic" \
  -- "${PRODUCTION_ARGS[@]}" \
  --output-dir "$RUN_OUT/pipeline" \
  2>&1 | tee "$RUN_OUT/run.log"
FULL_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$FULL_EXIT" > "$RUN_OUT/exit_code.txt"
```

Each copy record contains:

```text
graph call and copy ordinal
member index
key/value and transformer layer
segment offset and length
scratch and destination cache lengths
base and view shapes/strides/storage offsets
underlying storage size and calculated highest accessed element
data/storage pointers and alias status
contiguity and NPU format
```

The probe asserts all application-level bounds and cache-storage uniqueness
before launching the copy.

Summarize it:

```sh
"$PYTHON_BIN" - "$RUN_OUT/diagnostic/events.jsonl" \
  > "$RUN_OUT/copy_summary.json" <<'PY'
import json
import sys

events = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
before = {
    event["copy"]["copy_id"]: event
    for event in events
    if event["event"] == "copy_before"
}
passed = [
    event["copy"]["copy_id"]
    for event in events
    if event["event"] == "copy_sync_passed"
]
failed = [
    event
    for event in events
    if event["event"] in {
        "copy_validation_failed",
        "copy_enqueue_failed",
        "copy_sync_failed",
    }
]
candidate = failed[-1] if failed else next(
    (
        event
        for event in reversed(events)
        if event["event"] == "copy_before"
        and event["copy"]["copy_id"] not in set(passed)
    ),
    None,
)
payload = {
    "event_counts": {
        name: sum(event["event"] == name for event in events)
        for name in sorted({event["event"] for event in events})
    },
    "passed_copy_count": len(passed),
    "candidate": candidate,
    "candidate_before": (
        before.get(candidate["copy"]["copy_id"])
        if candidate is not None
        else None
    ),
    "previous_passing_copy": (
        before.get(passed[-1]) if passed else None
    ),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
cat "$RUN_OUT/copy_summary.json"
```

Interpret:

```text
copy_validation_failed
  STOP. This is our bounds, alias, or tensor-metadata bug, not a CANN bug.

copy_sync_failed with all validations true
  CONTINUE to the three fresh-process replay lanes.

complete page passes only with per-copy barriers
  STOP. Queueing/ordering between copies is causal. There is no isolated
  failing shape to replay; do not pretend the final copy is the culprit.
```

### 21.6 Fresh-process replay lanes

Run only when Run B identifies one failed or incomplete validated copy.
All three commands read Run B's durable `events.jsonl`.

The lanes are:

```text
candidate_current
  current sliced copy with the exact failing shape/stride/offset

neighbor_current
  the immediately preceding copy that synchronized successfully

candidate_per_head
  same candidate, copied one KV head at a time so each view is contiguous
```

Run each lane in a fresh Python process with a distinct CANN-log directory.
Do not combine them in one invocation:

```sh
TRACE="$OUTPUT_ROOT/full_per_copy/diagnostic/events.jsonl"

for LANE in candidate_current neighbor_current candidate_per_head; do
  LANE_OUT="$OUTPUT_ROOT/replay_$LANE"
  LANE_RAW="$RAW_ROOT/replay_$LANE"
  mkdir -p "$LANE_OUT" "$LANE_RAW/cann"
  export ASCEND_PROCESS_LOG_PATH="$LANE_RAW/cann"
  export ASCEND_WORK_PATH="$LANE_RAW"

  set +e
  set -o pipefail
  "$PYTHON_BIN" "$PROBE" \
    --mode replay \
    --diagnostic-dir "$OUTPUT_ROOT/full_per_copy/diagnostic" \
    --trace "$TRACE" \
    --replay-lane "$LANE" \
    --output "$LANE_OUT/summary.json" \
    2>&1 | tee "$LANE_OUT/run.log"
  LANE_EXIT="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$LANE_EXIT" > "$LANE_OUT/exit_code.txt"
done
```

Do not reuse an interpreter or interactive Python session between lanes.

### 21.7 Decision table

Use this table exactly:

| Observation | Conclusion |
| --- | --- |
| graph sync fails | failure is inside compiled packed text prefill |
| graph-only page passes | graph-to-redistribution dependency/order bug |
| full probe catches invalid bound/storage/alias | application/runtime bookkeeping bug |
| graph-only fails, full passes | asynchronous multi-copy queue/order interaction |
| candidate fresh copy fails; neighbor passes | shape/stride-specific Torch-NPU/CANN copy defect |
| candidate and neighbor fresh copies both fail | broader replay/operator limitation; control is not discriminating |
| candidate fresh copy passes | integrated graph output format/lifetime/allocator/stream state is required |
| candidate current fails; per-head passes | strided cross-head copy path is causal; per-head copy is a viable fix candidate |
| candidate per-head also fails | problem is not only the non-contiguous head stride |

The phrase "CANN GatherV2 bug" is allowed only when:

1. Run A proves the packed graph completed;
2. Run B proves all Python bounds/storage invariants;
3. `candidate_current` fails from fresh allocations;
4. the adjacent passing control does not fail.

### 21.8 Report and stopping point

Do not implement a fix. Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Use:

```text
310P PHASE 21 PACKED-TEXT/KV-COPY ISOLATION: ROOT CAUSE FOUND | BOUNDED | INCONCLUSIVE

Commit / host / exact NPU / software:
Exact production command equivalence:

Run A, graph-only:
- exit:
- last graph event:
- packed graph independently synchronized:
- original redistribution status:
- conclusion:

Run B, full per-copy (if run):
- exit:
- graph call:
- pack physical/real length:
- segment lengths/offsets:
- passed copy count:
- failing copy ordinal:
- member / K-or-V / layer:
- offset / length:
- source base/view shape and strides:
- destination base/view shape and strides:
- storage offsets and maximum accessed elements:
- storage sizes:
- formats and contiguity:
- all bounds/alias validations:
- enqueue versus synchronization failure:
- exact CANN op/error:

Fresh replay:
| lane | exit | exact result | CANN op/error |

Classification using the Phase-21 decision table:

What is proven:
What is not proven:
One proposed next code change to discuss (do not implement):

Evidence:
- graph-only events/log/CANN root:
- full-copy events/summary/log/CANN root:
- each replay summary/log/CANN root:
```

Paste back `agent_report.md`, Run A's `event_summary.txt`, Run B's
`copy_summary.json` if produced, and all replay `summary.json` files. Include
paths to raw CANN evidence. Do not paste enormous plogs unless a compact
operator/error excerpt is needed.

## Phase 22: isolate the packed final-token GatherV2

### 22.0 Purpose and exact hypothesis

Run this phase only. Phase 21 proved that packed graph calls 1-6 synchronize
successfully and graph call 7 fails inside `GatherV2_216`, before
`redistribute_cache` begins.

The packed graph contains one explicit gather-like operation:

```python
return torch.index_select(hidden_states, 1, last_token_indices)
```

Source:

```text
09_persistent_page_engine/paddleocr_vl/model/text_packed_prefill.py
PackedTextPrefillStage.forward
```

The ordinary non-packed graph also selects its final token, but uses one index.
The packed graph supplies a fixed-width vector containing each member's last
token position. This phase determines whether `GatherV2_216` is that final
selection and whether the failure belongs to:

```text
invalid packed indices generated by our code
eager Torch-NPU/CANN GatherV2
TorchAir/GE compiled GatherV2 only
the full packed graph's memory planning or earlier tensor corruption
```

Do not disable text packing yet. That broad control cannot distinguish these
possibilities.

### 22.1 Probe behavior

The committed probe is:

```text
09_persistent_page_engine/scripts/probes/
  probe_packed_last_token_gather.py
```

Analyze mode reads the existing Phase-21 trace and reconstructs, for every
packed graph call:

```text
physical and real sequence length
padding
segment lengths and offsets
max packed members and hidden size
last_token_indices = cumulative_sum(segment_lengths) - 1
zero padding of the fixed-width index vector
whether every index is in bounds
whether the final active index equals physical_seq_len - 1
graph pass/fail status
closest earlier passing control, preferring the same static shape
```

Run mode constructs only:

```python
torch.index_select(hidden_states, 1, last_token_indices)
```

TorchAir receives this operation through
`GatherLastTokensStage.forward`, a bound `torch.nn.Module` method as required
by `torchair.inference.cache_compile`. Do not replace it with a free function.

It first runs the same graph shape with all-zero indices. It then runs the
recorded indices and synchronizes. This separates static-shape support from
value-specific failure.

Every NPU lane is a separate process. Do not combine lanes in an interactive
Python session after an AICore exception.

### 22.2 Pull and locate Phase-21 evidence

Read `CLAUDE.md` and `AGENTS.md`, then:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PROBE="$REPO/09_persistent_page_engine/scripts/probes/probe_packed_last_token_gather.py"
TRACE="$REPO/tmp/09_persistent_page_engine/310p_phase21_kv_copy_62db002/graph_only/diagnostic/events.jsonl"
PHASE21_RAW="$REPO/.runtime_cache/310p_phase21_kv_copy_62db002/graph_only/cann"

test -x "$PYTHON_BIN"
test -f "$PROBE"
test -s "$TRACE"
test -d "$PHASE21_RAW"
"$PYTHON_BIN" "$PROBE" --help

OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase22_gather_$COMMIT_SHORT"
RAW_ROOT="$REPO/.runtime_cache/310p_phase22_gather_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RAW_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
```

Record the environment and verify the intended NPU is not occupied by an
unrelated process:

```sh
{
  printf 'commit=%s\n' "$COMMIT"
  printf 'host=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
import torch_npu

print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_name": torch.npu.get_device_name(0),
}, indent=2))
PY
  npu-smi info
} 2>&1 | tee "$OUTPUT_ROOT/preflight.log"
```

Never use `pkill` or `killall`.

### 22.3 Analyze the existing graph calls

This step does not execute an NPU operation:

```sh
"$PYTHON_BIN" "$PROBE" \
  --mode analyze \
  --trace "$TRACE" \
  --output "$OUTPUT_ROOT/call_analysis.json" \
  > "$OUTPUT_ROOT/call_analysis_stdout.json"
```

Create a readable matrix:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT/call_analysis.json" \
  > "$OUTPUT_ROOT/call_matrix.txt" <<'PY'
import json
import sys

analysis = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "call status physical real padding segments "
    "last_active boundary in_bounds offsets_valid indices"
)
for call in analysis["calls"]:
    print(
        call["graph_call"],
        call["status"],
        call.get("physical_seq_len"),
        call.get("real_seq_len"),
        call.get("padding_tokens"),
        call.get("segment_count"),
        call.get("last_active_index"),
        call.get("last_index_hits_physical_boundary"),
        call.get("indices_in_bounds"),
        call.get("offsets_match_lengths"),
        call.get("last_token_indices"),
    )
print("failed_call:", analysis["failed_call"])
print("control_call:", analysis["control_call"])
print(
    "control_same_static_shape:",
    analysis["control_same_static_shape"],
)
print(
    "failed_vs_control:",
    json.dumps(analysis["failed_vs_control"], indent=2),
)
PY
cat "$OUTPUT_ROOT/call_matrix.txt"
```

Stopping rule:

```text
failed call indices_in_bounds=false or offsets_match_lengths=false
  STOP. This is our packed metadata bug; do not run GatherV2 probes.

all validations true
  Continue.
```

The report must include all calls 1-7, not only the failed call.

### 22.4 Map the Phase-21 operator evidence

Extract compact references to the faulting operator and its input descriptors:

```sh
rg -a -n -C 8 \
  'GatherV2_216|te_gatherv2_7faad72e|MTE instruction|0x800000|507011' \
  "$PHASE21_RAW" \
  > "$OUTPUT_ROOT/phase21_gatherv2_extract.txt" || true
```

Search the exact packed cache from the Phase-20/21 production command for the
node name without modifying the cache:

```sh
PACKED_CACHE=<EXACT_PHASE20_TEXT_PACKED_CACHE_ROOT>
test -d "$PACKED_CACHE"
rg -a -n -C 4 'GatherV2_216|GatherV2' "$PACKED_CACHE" \
  > "$OUTPUT_ROOT/packed_cache_gather_hits.txt" || true
```

If the binary cache does not expose readable node metadata, record that
plainly. Do not recompile the full packed transformer merely to obtain an
unpacked dump in this phase.

### 22.5 Configure isolated NPU lanes

Use the same diagnostic environment as Phase 21:

```sh
export PYTHONFAULTHANDLER=1
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_NPU_COMPACT_ERROR_OUTPUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_MODULE_LOG_LEVEL='RUNTIME=0:ASCENDCL=0:OP=0:TBE=0'
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000

GATHER_CACHE="$RAW_ROOT/torchair_cache"
mkdir -p "$GATHER_CACHE"
```

The control and failed TorchAir lanes deliberately share `GATHER_CACHE`. If
the analysis reports `control_same_static_shape=true`, both processes must
load the same static compiled gather executable while changing only index
values.

Define this shell helper:

```sh
run_gather_lane() {
  LANE_NAME="$1"
  SELECTION="$2"
  BACKEND="$3"
  INDEX_VARIANT="$4"

  LANE_OUT="$OUTPUT_ROOT/$LANE_NAME"
  LANE_RAW="$RAW_ROOT/$LANE_NAME"
  mkdir -p "$LANE_OUT" "$LANE_RAW/cann"
  export ASCEND_PROCESS_LOG_PATH="$LANE_RAW/cann"
  export ASCEND_WORK_PATH="$LANE_RAW"

  COMMAND=(
    "$PYTHON_BIN" "$PROBE"
    --mode run
    --trace "$TRACE"
    --output "$LANE_OUT/summary.json"
    --selection "$SELECTION"
    --backend "$BACKEND"
    --index-variant "$INDEX_VARIANT"
    --device npu:0
  )
  if test "$BACKEND" = torchair; then
    COMMAND+=(--cache-dir "$GATHER_CACHE")
  fi
  {
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
  } > "$LANE_OUT/command.sh"

  set +e
  set -o pipefail
  "${COMMAND[@]}" 2>&1 | tee "$LANE_OUT/run.log"
  LANE_EXIT="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$LANE_EXIT" > "$LANE_OUT/exit_code.txt"
}
```

Each helper call launches a new Python process. Do not turn it into one Python
loop.

### 22.6 Run the four primary lanes

Run controls first:

```sh
run_gather_lane control_eager     control eager    recorded
run_gather_lane control_torchair  control torchair recorded
run_gather_lane failed_eager      failed  eager    recorded
run_gather_lane failed_torchair   failed  torchair recorded
```

Every lane:

1. allocates a deterministic hidden tensor with the recorded static shape;
2. executes all-zero indices and synchronizes;
3. executes the selected recorded index vector and synchronizes;
4. checks exact output values if execution succeeds.

The `summary.json` is written before the target call. If the process is killed,
its last `status` shows whether failure occurred during warm zeros or the
recorded target indices.

Collect a compact matrix:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" > "$OUTPUT_ROOT/lane_matrix.txt" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "control_eager",
    "control_torchair",
    "failed_eager",
    "failed_torchair",
):
    summary_path = root / name / "summary.json"
    exit_path = root / name / "exit_code.txt"
    summary = (
        json.load(open(summary_path, encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    print(
        name,
        "exit=" + (
            exit_path.read_text().strip()
            if exit_path.exists()
            else "missing"
        ),
        "status=" + str(summary.get("status")),
        "warm_s=" + str(summary.get("warm_call_s")),
        "target_s=" + str(summary.get("target_call_s")),
        "exact=" + str(summary.get("exact")),
        "exception=" + str(summary.get("exception")),
    )
PY
cat "$OUTPUT_ROOT/lane_matrix.txt"
```

### 22.7 One conditional boundary-index pair

Run this only if:

```text
failed call last_index_hits_physical_boundary=true
and either failed_eager or failed_torchair fails on recorded indices
after its all-zero warm call passed
```

`boundary_minus_one` changes only the final active index from
`physical_seq_len - 1` to `physical_seq_len - 2`. Static shapes, compiled
cache, all other indices, and the hidden tensor remain unchanged.

Run it for each backend whose recorded failed lane failed:

```sh
run_gather_lane failed_eager_boundary_minus_one \
  failed eager boundary_minus_one

run_gather_lane failed_torchair_boundary_minus_one \
  failed torchair boundary_minus_one
```

Do not run other index sweeps.

### 22.8 Decision table

| Observation | Conclusion |
| --- | --- |
| reconstructed index or offset is invalid | application packing bug |
| control and failed warm-zero calls fail | static GatherV2 shape/kernel unsupported or probe environment invalid |
| failed eager fails, control eager passes | eager Torch-NPU/CANN GatherV2 is value-pattern-specific |
| failed eager passes, failed TorchAir fails | TorchAir/GE compilation or compiled GatherV2 tiling issue |
| failed eager and TorchAir both pass | failure requires the full packed graph's memory plan, aliasing, or earlier corruption |
| control TorchAir and failed TorchAir fail despite different values | not specific to call-7 indices; inspect static-shape/control mismatch |
| recorded boundary index fails, minus-one passes | physical end-boundary index is causal |
| recorded and minus-one both fail | not explained solely by the final boundary index |
| isolated failure uses same fault kernel hash as Phase 21 | direct evidence mapping `GatherV2_216` to final `index_select` |

Do not call this a CANN bug merely because the full graph failed. A
standalone eager or compiled gather must reproduce it first.

### 22.9 Report and stopping point

Stop after the primary lanes and conditional boundary pair. Do not modify the
packed graph or disable text packing.

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Use:

```text
310P PHASE 22 PACKED FINAL-TOKEN GATHER: ROOT CAUSE FOUND | BOUNDED | NOT REPRODUCED

Commit / host / exact NPU / software:
Phase-21 trace:

Call matrix:
| call | status | physical | real | segment lengths | last indices | boundary |

Failed call:
- call:
- static hidden shape/dtype:
- segment lengths/offsets:
- reconstructed last_token_indices:
- indices in bounds:
- final active index at physical boundary:

Control call:
- call:
- same static shape:
- differing values:

Phase-21 node mapping:
- source operation:
- GatherV2 input descriptors from CANN:
- cache-readable node evidence:
- fault kernel/hash:

Isolated lanes:
| lane | warm status | target status | exact | CANN op/error/kernel |

Boundary-minus-one lanes, if applicable:

Classification from decision table:

What is proven:
What remains unknown:
Smallest next code change to discuss (do not implement):

Evidence:
- call_analysis.json / call_matrix.txt:
- Phase-21 Gather extract:
- cache search:
- each lane summary/log/CANN root:
- lane_matrix.txt:
```

Paste back `agent_report.md`, `call_matrix.txt`, `lane_matrix.txt`, each
`summary.json`, and the compact Phase-21 Gather extract. Include raw CANN log
paths but do not paste enormous plogs.

## Phase 23: execute the missing compiled GatherV2 lanes

### 23.0 Purpose and stopping boundary

Run this phase only. Do not rerun page 9, layout, vision prefill, packed text
prefill, decode, or the Phase-22 eager lanes.

Phase 22 already proved:

```text
control call 6:
  hidden shape = (1, 512, 1024)
  indices = [302, 375, 428, 0, ...]
  eager = exact pass

failed call 7:
  hidden shape = (1, 512, 1024)
  segment lengths = [133, 109, 69, 58, 58, 51]
  indices = [132, 241, 310, 368, 426, 477, 0, ...]
  all indices are in bounds
  eager = exact pass
```

The old TorchAir lanes did not test anything: `cache_compile` rejected the
probe's free function with `Only method can be cached now`. Commit `66e61ab`
replaced it with the bound method `GatherLastTokensStage.forward`.

This phase answers only:

> Does the exact call-7 index vector fail in a standalone compiled GatherV2
> graph on 310P, after the same-shape call-6 control compiled and passed?

Stop after that answer. Do not modify the packed transformer or introduce a
workaround in this phase. Do not edit tracked source, commit, or push from the
work server.

For comparison, the committed 910B evidence at `5af97ef` shows:

```text
control call 6 compiled: exact pass, max_abs_error=0
exact 310P call-7 vector compiled: exact pass, max_abs_error=0
```

Those are hardware controls, not proof of 310P behavior.

### 23.1 Pull and preflight

Use the same activated Python/CANN environment and the same physical 310P used
for Phases 21 and 22.

```sh
cd "$(git rev-parse --show-toplevel)"
REPO="$(git rev-parse --show-toplevel)"

git status --short --branch
git pull --ff-only origin main

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PROBE="$REPO/09_persistent_page_engine/scripts/probes/probe_packed_last_token_gather.py"
TRACE="$REPO/tmp/09_persistent_page_engine/310p_phase21_kv_copy_62db002/graph_only/diagnostic/events.jsonl"

test -x "$PYTHON_BIN"
test -f "$PROBE"
test -f "$TRACE"
git merge-base --is-ancestor 66e61ab "$COMMIT"
grep -q 'class GatherLastTokensStage' "$PROBE"
grep -q 'cache_compile(' "$PROBE"
grep -q 'stage.forward' "$PROBE"
"$PYTHON_BIN" "$PROBE" --help
```

Do not continue if the Phase-21 trace is missing. Do not reconstruct it from
the prose report; the exact JSONL is the input authority.

Confirm the device is free of unrelated work. Never use `pkill` or `killall`.
Record:

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase23_compiled_gather_$COMMIT_SHORT"
RAW_ROOT="$REPO/.runtime_cache/310p_phase23_compiled_gather_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RAW_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'host=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'date=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  "$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
import torch_npu

print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": torch.npu.is_available(),
    "npu_name": torch.npu.get_device_name(0),
}, indent=2))
PY
  npu-smi info
  df -h "$REPO"
} 2>&1 | tee "$OUTPUT_ROOT/preflight.log"
```

### 23.2 Re-analyze the unchanged Phase-21 trace

Run analyze mode once with the fixed probe:

```sh
"$PYTHON_BIN" "$PROBE" \
  --mode analyze \
  --trace "$TRACE" \
  --output "$OUTPUT_ROOT/call_analysis.json" \
  2>&1 | tee "$OUTPUT_ROOT/analyze.log"
```

Assert the exact decision inputs before touching the NPU:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT/call_analysis.json" <<'PY' \
  | tee "$OUTPUT_ROOT/analysis_assertions.txt"
import json
import sys

analysis = json.load(open(sys.argv[1], encoding="utf-8"))
calls = {int(call["graph_call"]): call for call in analysis["calls"]}
control = calls[6]
failed = calls[7]

assert analysis["failed_call"] == 7, analysis["failed_call"]
assert analysis["control_call"] == 6, analysis["control_call"]
assert analysis["control_same_static_shape"] is True

assert control["physical_seq_len"] == 512
assert control["hidden_size"] == 1024
assert control["max_members"] == 32
assert control["dtype"] == "torch.float16"
assert control["segment_lengths"] == [303, 73, 53]
assert control["active_last_token_indices"] == [302, 375, 428]

assert failed["physical_seq_len"] == 512
assert failed["hidden_size"] == 1024
assert failed["max_members"] == 32
assert failed["dtype"] == "torch.float16"
assert failed["segment_lengths"] == [133, 109, 69, 58, 58, 51]
assert failed["segment_offsets"] == [0, 133, 242, 311, 369, 427]
assert failed["active_last_token_indices"] == [
    132, 241, 310, 368, 426, 477
]
assert failed["indices_in_bounds"] is True
assert failed["offsets_match_lengths"] is True
assert failed["last_index_hits_physical_boundary"] is False

print("PHASE23_ANALYSIS_CONTRACT: PASS")
PY
```

Proceed only if the final line is:

```text
PHASE23_ANALYSIS_CONTRACT: PASS
```

### 23.3 Configure the two compiled lanes

Use one fresh shared cache. The control process compiles the graph; the failed
process must load the same graph with different index values.

```sh
GATHER_CACHE="$RAW_ROOT/shared_torchair_cache"
test ! -e "$GATHER_CACHE"
mkdir -p "$GATHER_CACHE"

export PYTHONFAULTHANDLER=1
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_NPU_COMPACT_ERROR_OUTPUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_MODULE_LOG_LEVEL='RUNTIME=0:ASCENDCL=0:OP=0:TBE=0'
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=10000

printf '[phase23] ready %s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" \
  | tee "$OUTPUT_ROOT/progress.log"
```

Define:

```sh
run_compiled_gather() {
  lane_name="$1"
  selection="$2"

  lane_out="$OUTPUT_ROOT/$lane_name"
  lane_raw="$RAW_ROOT/$lane_name"
  test ! -e "$lane_out"
  test ! -e "$lane_raw"
  mkdir -p "$lane_out" "$lane_raw/cann"

  export ASCEND_PROCESS_LOG_PATH="$lane_raw/cann"
  export ASCEND_WORK_PATH="$lane_raw"

  command=(
    "$PYTHON_BIN" "$PROBE"
    --mode run
    --trace "$TRACE"
    --output "$lane_out/summary.json"
    --selection "$selection"
    --backend torchair
    --cache-dir "$GATHER_CACHE"
    --index-variant recorded
    --device npu:0
  )

  {
    printf '#!/usr/bin/env bash\n'
    printf '# commit=%s\n' "$COMMIT"
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$lane_out/command.sh"

  printf '[phase23] START %s %s\n' \
    "$lane_name" \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"

  set +e
  set -o pipefail
  "${command[@]}" 2>&1 | tee "$lane_out/run.log"
  lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" > "$lane_out/exit_code.txt"

  printf '[phase23] END %s exit=%s %s\n' \
    "$lane_name" \
    "$lane_exit" \
    "$(date --iso-8601=seconds 2>/dev/null || date)" \
    | tee -a "$OUTPUT_ROOT/progress.log"

  find "$lane_raw" -type f -printf '%s %p\n' 2>/dev/null \
    | sort -n > "$lane_out/raw_file_manifest.txt"
  return 0
}
```

Progress is visible while it runs:

```sh
tail -f "$OUTPUT_ROOT/progress.log"
tail -f "$OUTPUT_ROOT/control_torchair/run.log"
tail -f "$OUTPUT_ROOT/failed_torchair/run.log"
```

### 23.4 Run control, then exact failed indices

Run the same-shape control first:

```sh
run_compiled_gather control_torchair control
cat "$OUTPUT_ROOT/control_torchair/summary.json"

"$PYTHON_BIN" - \
  "$OUTPUT_ROOT/control_torchair/summary.json" \
  "$OUTPUT_ROOT/control_torchair/exit_code.txt" <<'PY' \
  | tee "$OUTPUT_ROOT/control_assertions.txt"
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
exit_code = int(open(sys.argv[2], encoding="utf-8").read().strip())
assert exit_code == 0, exit_code
assert summary["status"] == "passed", summary["status"]
assert summary["selection"] == "control"
assert summary["graph_call"] == 6
assert summary["cache_was_warm"] is False
assert summary["exact"] is True
assert summary["max_abs_error"] == 0.0
print("PHASE23_CONTROL_CONTRACT: PASS")
PY
```

The control must have:

```text
exit_code = 0
status = passed
exact = true
max_abs_error = 0
cache_was_warm = false
```

If the control fails, stop. Do not run call 7 against an unvalidated or
partially written cache. Report whether failure occurred during the all-zero
warm call or the target call.

If the control passes, inventory the compiled cache:

```sh
find "$GATHER_CACHE" -type f -printf \
  '%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' \
  | sort > "$OUTPUT_ROOT/cache_after_control.txt"
```

Then run the exact call-7 vector in a fresh process:

```sh
run_compiled_gather failed_torchair failed
cat "$OUTPUT_ROOT/failed_torchair/summary.json"

"$PYTHON_BIN" - "$OUTPUT_ROOT/failed_torchair/summary.json" <<'PY' \
  | tee "$OUTPUT_ROOT/failed_metadata_assertions.txt"
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
call = summary["call"]
assert summary["selection"] == "failed"
assert summary["graph_call"] == 7
assert summary["cache_was_warm"] is True
assert call["physical_seq_len"] == 512
assert call["segment_lengths"] == [133, 109, 69, 58, 58, 51]
assert summary["effective_last_token_indices"][:6] == [
    132, 241, 310, 368, 426, 477
]
assert all(
    index == 0
    for index in summary["effective_last_token_indices"][6:]
)
print("PHASE23_FAILED_METADATA_CONTRACT: PASS")
PY
```

Its `summary.json` must show:

```text
graph_call = 7
physical_seq_len = 512
segment_lengths = [133, 109, 69, 58, 58, 51]
effective_last_token_indices = [132, 241, 310, 368, 426, 477, 0, ...]
cache_was_warm = true
```

If and only if `failed_torchair` exited zero and passed exactly, repeat only
that lane once in one more fresh process:

```sh
if "$PYTHON_BIN" - \
  "$OUTPUT_ROOT/failed_torchair/summary.json" \
  "$OUTPUT_ROOT/failed_torchair/exit_code.txt" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
exit_code = int(open(sys.argv[2], encoding="utf-8").read().strip())
raise SystemExit(
    0
    if (
        exit_code == 0
        and summary.get("status") == "passed"
        and summary.get("exact") is True
        and summary.get("max_abs_error") == 0.0
    )
    else 1
)
PY
then
  run_compiled_gather failed_torchair_repeat failed
  cat "$OUTPUT_ROOT/failed_torchair_repeat/summary.json"
fi
```

Do not repeat a failing lane. Do not run a value sweep, boundary-minus-one,
other shapes, eager, or the complete OCR pipeline.

### 23.5 Compact report and decision

Create:

```sh
"$PYTHON_BIN" - "$OUTPUT_ROOT" > "$OUTPUT_ROOT/lane_matrix.txt" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "control_torchair",
    "failed_torchair",
    "failed_torchair_repeat",
):
    lane = root / name
    if not lane.exists():
        continue
    summary_path = lane / "summary.json"
    exit_path = lane / "exit_code.txt"
    summary = (
        json.load(open(summary_path, encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    print(
        name,
        "exit=" + (
            exit_path.read_text().strip()
            if exit_path.exists()
            else "missing"
        ),
        "status=" + str(summary.get("status")),
        "cache_warm=" + str(summary.get("cache_was_warm")),
        "warm_s=" + str(summary.get("warm_call_s")),
        "target_s=" + str(summary.get("target_call_s")),
        "exact=" + str(summary.get("exact")),
        "max_abs=" + str(summary.get("max_abs_error")),
        "exception=" + str(summary.get("exception")),
    )
PY
cat "$OUTPUT_ROOT/lane_matrix.txt"
```

Use this decision table:

| Observation | Conclusion and next step |
| --- | --- |
| control compiled Gather fails | standalone compiled shape is unsupported or the probe/cache environment is invalid; inspect that failure before call 7 |
| control passes, call 7 fails with the Phase-21 GatherV2 kernel/error | exact index values trigger a 310P compiled GatherV2 bug; the smallest fix is to move final-token selection outside the graph |
| control passes, call 7 fails with a different operator | do not attribute Phase 21 to GatherV2 yet; report the new causal operator |
| control and call 7 pass exactly, including repeat | standalone compiled Gather is exonerated; the failure requires the full packed transformer's memory plan, aliasing, scratch reuse, or earlier corruption |
| all-zero warm call passes but target call fails | index-value-dependent compiled behavior |
| all-zero warm call itself fails | static compiled graph/kernel or cache problem, not call-7 values |

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Use:

```text
310P PHASE 23 FIXED COMPILED GATHER: REPRODUCED | NOT REPRODUCED | CONTROL FAILED

Commit / host / exact NPU / software:
Phase-21 trace:
Probe contract:
- bound method present:
- analysis assertions:

Control compiled lane:
- exit/status:
- warm versus target status:
- cache_was_warm:
- exact/max_abs:
- CANN operator/error/kernel, if any:

Exact call-7 compiled lane:
- exit/status:
- exact segment lengths:
- exact active indices:
- cache_was_warm:
- warm versus target status:
- exact/max_abs:
- CANN operator/error/kernel, if any:

Repeat lane, if run:

Classification:
What is proven:
What remains unknown:
Next experiment:

Evidence:
- preflight.log:
- call_analysis.json:
- analysis_assertions.txt:
- control_assertions.txt:
- failed_metadata_assertions.txt:
- progress.log:
- lane_matrix.txt:
- each command.sh / run.log / summary.json / exit_code.txt:
- cache_after_control.txt:
- raw CANN roots:
```

Paste back `agent_report.md`, `lane_matrix.txt`, both primary
`summary.json` files, and the relevant compact CANN error extract if a lane
fails. Do not paste enormous plogs.

## Phase 24: quick page-nine E2E validation

Run only the real page at `--offset 8 --limit 1`. This is a production-path
smoke test of the narrow final-gather change, not another diagnostic phase.
Do not rerun the eight-page ladder, profiler, Gather probe, or any matrix.

### 24.1 Pull and verify the source boundary

Use the same environment, physical 310P, model, dataset, cache roots, and
production arguments as the prior Phase-20/21 page-nine reproducer.

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SOURCE="$REPO/09_persistent_page_engine/paddleocr_vl/model/text_packed_prefill.py"

test -x "$PYTHON_BIN"
test -f "$SOURCE"
grep -A55 'class PackedTextPrefillStage' "$SOURCE" \
  | grep -q 'return self.text_model.norm(hidden_states)'
grep -A25 'def run_prepared' "$SOURCE" \
  | grep -q 'torch.index_select'
```

The packed-text cache key includes this file's source hash, so the modified
graph gets its own cache directory automatically. Do not delete or rename any
existing caches.

### 24.2 Run one normal E2E page

Recover the exact normal production command used for the Phase-20/21
`--offset 8 --limit 1` reproducer. Keep every model, dataset, packing, bucket,
PromptFA, decode, layout, min-pixels, cache-length, and cache-root argument
unchanged. Change only the output directory. Do not use the Phase-21 probe,
`ASCEND_LAUNCH_BLOCKING`, per-graph synchronization, profiling, or graph
barriers.

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase24_eager_final_gather_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'host=%s\n' "$(hostname)"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'date=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  "$PYTHON_BIN" - <<'PY'
import json
import torch
import torch_npu

print(json.dumps({
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu": torch.npu.get_device_name(0),
}, indent=2))
PY
} 2>&1 | tee "$OUTPUT_ROOT/preflight.log"
```

Before running, save the exact reconstructed command as
`$OUTPUT_ROOT/command.sh`. It must invoke:

```text
09_persistent_page_engine/scripts/run_omnidocbench.py
--offset 8
--limit 1
--output-dir <the Phase-24 output directory>
```

Run it normally with progress visible:

```sh
set -o pipefail
bash "$OUTPUT_ROOT/command.sh" 2>&1 | tee "$OUTPUT_ROOT/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$OUTPUT_ROOT/exit_code.txt"
```

### 24.3 Report and stop

Success means:

- process exit code is zero;
- exactly one page result is emitted;
- normal end-of-run accounting passes;
- no AICore exception or `GatherV2_216` DDR out-of-range error appears;
- the page reaches recognition completion.

Report only:

```text
310P PHASE 24 EAGER FINAL GATHER E2E: PASS | FAIL

commit / host / exact NPU / software:
exact command:
exit code:
page results:
wall time:
packed-text graph compile/cache status:
final-token gather path verified:
first causal error, if failed:
evidence paths:
```

Paste back this compact report and, on failure, the smallest relevant error
extract. Do not implement another workaround or run more pages.

## Phase 25: synchronize around the eager final gather

This is the same one-page Phase-24 E2E command with one diagnostic environment
variable. It answers whether the AICore exception is already pending when the
packed transformer graph finishes, or is triggered by the eager
`torch.index_select` that follows it.

Do not run any other phase or page.

### 25.1 Pull and verify

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
SOURCE="$REPO/09_persistent_page_engine/paddleocr_vl/model/text_packed_prefill.py"

grep -q 'PADDLE_OCR_VL_PACKED_GATHER_SYNC_DIAGNOSTIC' "$SOURCE"
grep -q 'compiled graph synchronization failed before eager final-token gather' \
  "$SOURCE"
grep -q 'eager final-token gather synchronization failed after compiled graph completed' \
  "$SOURCE"
```

Use the exact Phase-24 command and environment. Change only its output
directory and set:

```sh
export PADDLE_OCR_VL_PACKED_GATHER_SYNC_DIAGNOSTIC=1
```

Do not add `ASCEND_LAUNCH_BLOCKING`, a profiler, graph probe, or other
synchronization.

### 25.2 Run the same page

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase25_gather_boundary_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
```

Copy the exact Phase-24 `command.sh`, changing only its output directory.
Preserve `--offset 8 --limit 1` and every production option. Save the new exact
command as `$OUTPUT_ROOT/command.sh`, then:

```sh
set -o pipefail
bash "$OUTPUT_ROOT/command.sh" 2>&1 | tee "$OUTPUT_ROOT/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$OUTPUT_ROOT/exit_code.txt"

grep -nE \
  'packed-gather-sync|compiled graph synchronization failed|eager final-token gather synchronization failed|AICore|GatherV2|DDR address' \
  "$OUTPUT_ROOT/run.log" \
  > "$OUTPUT_ROOT/boundary_extract.txt" || true
cat "$OUTPUT_ROOT/boundary_extract.txt"
```

### 25.3 Interpret and stop

The markers are emitted once per packed-text graph call:

```text
[packed-gather-sync] compiled_graph_enqueued
[packed-gather-sync] compiled_graph_sync_passed
[packed-gather-sync] eager_gather_enqueued
[packed-gather-sync] eager_gather_sync_passed
```

Use only this decision:

| Last successful marker / exception | Conclusion |
| --- | --- |
| exception says `compiled graph synchronization failed before eager final-token gather` | failure is already inside the full compiled packed transformer graph |
| `compiled_graph_sync_passed`, then exception says `eager final-token gather synchronization failed after compiled graph completed` | compiled graph completed; the external eager GatherV2 is causal |
| both sync markers pass for every call and page succeeds | added synchronization avoids the asynchronous/lifetime failure |

Write and paste back:

```text
310P PHASE 25 PACKED GATHER BOUNDARY: GRAPH | EAGER GATHER | SYNC-SENSITIVE PASS

commit / host / exact NPU / software:
exit code:
number of compiled_graph_sync_passed markers:
number of eager_gather_sync_passed markers:
last successful marker:
wrapped RuntimeError:
underlying AICore operator / kernel / error:
wall time:
evidence:
```

Do not implement a workaround afterward.

## Phase 26: quick slice-and-concat page-nine E2E

The final-token selection no longer uses `torch.index_select`. Run the exact
normal Phase-24 page-nine command once to test this narrow workaround.

### 26.1 Pull and verify

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
SOURCE="$REPO/09_persistent_page_engine/paddleocr_vl/model/text_packed_prefill.py"

grep -A35 'def run_prepared' "$SOURCE" | grep -q 'torch.cat'
if grep -A35 'def run_prepared' "$SOURCE" | grep -q 'torch.index_select'; then
  echo "run_prepared still contains GatherV2-producing index_select" >&2
  exit 1
fi
```

Unset the Phase-25 diagnostic variable:

```sh
unset PADDLE_OCR_VL_PACKED_GATHER_SYNC_DIAGNOSTIC
```

### 26.2 Run one page and stop

Use the exact normal Phase-24 command, including all production arguments and
`--offset 8 --limit 1`. Change only:

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase26_slice_concat_$COMMIT_SHORT"
```

Save the exact command as `$OUTPUT_ROOT/command.sh`, then:

```sh
set -o pipefail
bash "$OUTPUT_ROOT/command.sh" 2>&1 | tee "$OUTPUT_ROOT/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$OUTPUT_ROOT/exit_code.txt"

grep -nE 'AICore|GatherV2|DDR address|completed=|summary=' \
  "$OUTPUT_ROOT/run.log" \
  > "$OUTPUT_ROOT/compact_extract.txt" || true
cat "$OUTPUT_ROOT/compact_extract.txt"
```

Report:

```text
310P PHASE 26 SLICE-CONCAT FINAL TOKEN E2E: PASS | FAIL

commit / host / NPU / software:
exit code:
page results:
wall time:
packed-text graph cache/compile status:
stop-reason counts:
GatherV2 or AICore error:
first causal error, if failed:
evidence:
```

Success requires one completed page, normal accounting, and no GatherV2 DDR
out-of-range error. Do not run more pages or implement another change.

## Phase 27: warm-cache 8-page reproduction and 32-page extension

Run the current best production configuration first on pages 0-7, then on
pages 0-31. Finish layout and crop preparation for every selected page before
starting OCR by adding `--preprocess-all-pages-first`. This removes concurrent
layout work from the OCR interval. Do not compile experimental graphs, change
routing, profile, or introduce diagnostics.

The historical eight-page Phase-7 anchor at commit `4789067` was:

```text
pipeline E2E:             25.69 s
throughput:               0.3114 pages/s
recognition requests:     122
vision real / physical:   58,368 / 68,864 tokens
vision device time:       10.338 s
text real / physical:     16,178 / 22,528 tokens
text device time:         1.445 s
decode wall:              4.66 s
decode effective / raw:   7,330 / 14,144 tokens
layout:                   1.42 s
```

The historical anchor used the streaming frontend, so its total E2E and
pages/s are context rather than a direct layout-first comparison. Report
meaningful differences rather than forcing agreement. For the new runs,
report OCR-only wall and pages/s from
`recognition.run_scoped_scheduler_wall_s`; `pipeline_e2e_s` still includes
the layout-first phase.

### 27.1 Pull and establish the exact command

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PHASE26_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase26_slice_concat_33a0407"
PHASE26_COMMAND="$PHASE26_ROOT/command.sh"

test -x "$PYTHON_BIN"
test -f "$PHASE26_COMMAND"
grep -q -- '--batch-size 32' "$PHASE26_COMMAND"
grep -q -- '--preprocessor-min-pixels 28224' "$PHASE26_COMMAND"
grep -q -- '--text-packing production_group' "$PHASE26_COMMAND"
grep -q -- '--vision-packing greedy' "$PHASE26_COMMAND"
grep -q -- '--vision-promptfa-align-128' "$PHASE26_COMMAND"
grep -q -- '--layout-device npu' "$PHASE26_COMMAND"
grep -q -- '--no-layout-graph-capture' "$PHASE26_COMMAND"
```

The Phase-26 command is the argument authority. Preserve every model, dataset,
cache, bucket, packing, PromptFA, decode, layout, and min-pixels option.
Change only `--offset`, `--limit`, and `--output-dir`, and add exactly one
option:

```text
--preprocess-all-pages-first
```

Unset the earlier diagnostic variable:

```sh
unset PADDLE_OCR_VL_PACKED_GATHER_SYNC_DIAGNOSTIC
```

Confirm no unrelated process occupies the selected NPU. Never use `pkill` or
`killall`.

### 27.2 Run pages 0-7

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase27_scale_$COMMIT_SHORT"
RUN8="$OUTPUT_ROOT/pages_8"
RUN32="$OUTPUT_ROOT/pages_32"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$RUN8" "$RUN32"
```

Create `$RUN8/command.sh` from the exact Phase-26 command with:

```text
--offset 0
--limit 8
--preprocess-all-pages-first
--output-dir <RUN8>/output
```

Before running, verify:

```sh
grep -q -- '--offset 0' "$RUN8/command.sh"
grep -q -- '--limit 8' "$RUN8/command.sh"
grep -q -- '--batch-size 32' "$RUN8/command.sh"
grep -q -- '--preprocess-all-pages-first' "$RUN8/command.sh"
```

Run:

```sh
set -o pipefail
bash "$RUN8/command.sh" 2>&1 | tee "$RUN8/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$RUN8/exit_code.txt"
```

Require exit zero, 8 results, 8 predictions, normal accounting, and no AICore
exception. Require
`page_preprocessing_mode == "all_before_recognition"` in `run_summary.json`.
Compare its predictions or recognition token IDs against the historical
Phase-7 replay if that artifact is still present. The final-token selection is
mathematically identical, so same-environment parity is expected.

If the eight-page run fails, stop and report. Do not start 32 pages.

### 27.3 Run pages 0-31

Create `$RUN32/command.sh` from the same Phase-26 command with:

```text
--offset 0
--limit 32
--preprocess-all-pages-first
--output-dir <RUN32>/output
```

Verify:

```sh
grep -q -- '--offset 0' "$RUN32/command.sh"
grep -q -- '--limit 32' "$RUN32/command.sh"
grep -q -- '--batch-size 32' "$RUN32/command.sh"
grep -q -- '--preprocess-all-pages-first' "$RUN32/command.sh"
```

Then:

```sh
set -o pipefail
bash "$RUN32/command.sh" 2>&1 | tee "$RUN32/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$RUN32/exit_code.txt"
```

Require exit zero, 32 results, 32 predictions, normal accounting, all
recognition requests accounted for, no AICore exception, and
`page_preprocessing_mode == "all_before_recognition"`.

### 27.4 Compact comparison

Read both `run_summary.json` files and report:

| Metric | Historical 8 pages | Current 8 pages | Current 32 pages |
| --- | ---: | ---: | ---: |
| setup seconds | — | | |
| pipeline E2E seconds | 25.69 | | |
| total-pipeline pages/s | 0.3114 | | |
| OCR scheduler wall seconds | — | | |
| OCR-only pages/s | — | | |
| pages / recognition requests | 8 / 122 | | |
| layout seconds | 1.42 | | |
| vision real / physical tokens | 58,368 / 68,864 | | |
| vision device seconds | 10.338 | | |
| effective / physical vision tokens/s | 5,646 / 6,661 | | |
| text real / physical tokens | 16,178 / 22,528 | | |
| text device seconds | 1.445 | | |
| effective / physical text tokens/s | 11,198 / 15,593 | | |
| decode effective / raw slots | 7,330 / 14,144 | | |
| decode wall seconds | 4.66 | | |
| effective / raw decode tokens/s | 1,574 / 3,036 | | |
| vision packing groups / fill | | | |
| text packing calls / fill | | | |
| stop reasons | | | |

Use stage values from the summaries, not inferred wall-clock subtraction.
Calculate OCR-only pages/s as:

```text
page_count / recognition.run_scoped_scheduler_wall_s
```

Do not use `recognition.wall_s` as OCR-only time: in layout-first mode that
field spans both page preparation and recognition. The scheduler wall begins
when the recognizer is actually invoked.

Also state whether the current eight-page prediction/token comparison with the
historical replay is exact, unavailable, or different.

Write:

```text
$OUTPUT_ROOT/agent_report.md
```

Paste back `agent_report.md`, the compact table, both exit codes, and both
summary paths. Do not paste full logs unless there is a failure.

## Phase 28: identify the exact scheduler blocking boundary

Run this phase only on the 16-page configuration that has been observed to
stop making progress around page 12. This is a diagnostic replay, not a
performance benchmark. Do not change the model, caches, packing, layout-first
frontend, min-pixels, or graph configuration.

Commit `03cfa99` adds structured progress events without adding NPU
synchronization. Every event is emitted to stderr, flushed immediately, and
begins with:

```text
EXP09_SCHEDULER
```

The relevant boundaries are deliberately paired:

```text
decode_step_begin                 -> decode_step_end
token_copy_schedule_begin         -> token_copy_schedule_end
pending_token_wait_begin          -> pending_token_wait_end
ready_source_next_begin           -> ready_source_next_end
prefill_enqueue_begin             -> prefill_enqueue_end
prefill_finalize_begin            -> prefill_finalize_end
prefill_h2d_executor_shutdown_begin -> prefill_h2d_executor_shutdown_end
```

A successful 910B control using the same diagnostics completed 244/244 crops.
Its iteration 525 emitted the full sequence through `iteration_end`. It later
entered a normal four-request tail at iteration 855 and drained the last
request at iteration 1026. Therefore `iteration == 525` or
`active_count == 4` is not itself a scheduler terminal condition.

### 28.1 Pull and construct the diagnostic command

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
git merge-base --is-ancestor 03cfa99 "$COMMIT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import kornia_rs
import torch
import torch_npu
print("python_dependencies: PASS")
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY
```

Use the exact command that reproduced the 16-page stop. Preserve all existing
arguments, including `--preprocess-all-pages-first`. Confirm that it contains:

```text
--offset 0
--limit 16
--batch-size 16
--preprocessor-min-pixels 28224
--text-packing production_group
--vision-packing greedy
--vision-promptfa-align-128
--layout-device npu
--no-layout-graph-capture
--preprocess-all-pages-first
```

Add only:

```text
--scheduler-progress
```

and change `--output-dir` to the new run directory:

```sh
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase28_scheduler_$COMMIT_SHORT"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
```

Save the fully expanded command as `$OUTPUT_ROOT/command.sh`. Do not use a
different runner or a reduced isolated model.

### 28.2 Run with a durable live log

Run unbuffered and preserve the actual Python exit code:

```sh
set -o pipefail
PYTHONUNBUFFERED=1 bash "$OUTPUT_ROOT/command.sh" 2>&1 \
  | tee "$OUTPUT_ROOT/run.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$OUTPUT_ROOT/exit_code.txt"
```

The log is intentionally live. In a second shell, progress can be inspected
without touching the process:

```sh
tail -f "$OUTPUT_ROOT/run.log"
```

or, for scheduler boundaries only:

```sh
grep --line-buffered '^EXP09_SCHEDULER ' "$OUTPUT_ROOT/run.log" | tail -n 80
```

If no new `EXP09_SCHEDULER` event appears for 90 seconds after the scheduler
has started, treat the run as hung. Do not wait five minutes. Before stopping
the process, capture:

```sh
grep '^EXP09_SCHEDULER ' "$OUTPUT_ROOT/run.log" \
  | tail -n 120 > "$OUTPUT_ROOT/last_scheduler_events.log"
npu-smi info > "$OUTPUT_ROOT/npu_smi_at_stall.txt" 2>&1 || true
```

Then terminate only the exact process started by this run. Never use `pkill`
or `killall`.

### 28.3 Mechanical classification

Do not infer the blocking point from the last progress counter. Use the last
unmatched paired event:

| Last unmatched event | Classification |
| --- | --- |
| `decode_step_begin` | compiled decode invocation/replay did not return |
| `token_copy_schedule_begin` | sampled-token copy submission blocked |
| `pending_token_wait_begin` | prior decode/token-copy event never completed |
| `ready_source_next_begin` | synchronous refill is blocked inside the ready-source generator |
| `prefill_enqueue_begin` | staged prefill enqueue/H2D path blocked |
| `prefill_finalize_begin` | prefill device completion or first-token D2H blocked |
| `prefill_h2d_executor_shutdown_begin` | ready-source exhaustion is blocked draining the H2D executor |
| `iteration_end` with no following `iteration_begin` | Python control left the iteration loop unexpectedly; inspect the following non-scheduler traceback/log |

If the last unmatched event is `ready_source_next_begin`, inspect the nested
engine events after it. `refill_ready_queue()` is then the synchronous caller,
but the nested unmatched event identifies the actual operation that did not
return.

Repeated adjacent `refill_begin` and `refill_end` events with
`source_exhausted=true` and `pulled=0` are known harmless zero-work calls.
They are inefficient but are not a hang.

### 28.4 Compact report

Create `$OUTPUT_ROOT/agent_report.md` containing:

```text
310P PHASE 28 SCHEDULER BOUNDARY: DECODE | TOKEN_D2H | REFILL_PREFILL | OTHER | PASS

commit / host / exact NPU / software:
exact command:
exit code:
last completed page/result count:
last iteration:
active_count / active request IDs and token counts:
ready_depth / source_exhausted / submitted / completed:
last 20 EXP09_SCHEDULER events:
last unmatched begin event:
mechanical classification:
NPU utilization/state at stall:
first causal error, if any:
evidence paths:
```

Paste back the report and the literal last 20 `EXP09_SCHEDULER` JSON lines.
Do not summarize those lines away: their exact event names and snapshots are
the evidence needed for the next code decision.

## Phase 29: isolate the 1280-token decode boundary

Run this phase after pulling a commit that contains
`--diagnostic-decode-effective-length`. The earlier offset-0 and offset-4
replays stopped at different global scheduler iterations, but both stopped
when this same request reached the same local state:

```text
request_id:       page_000011_block_000003
prompt_length:    1021
generated tokens: 259 at the stalled wait
```

The pending graph was launched while the CPU request state still contained
258 tokens. Its expected cache write position was therefore 1279 and its
effective valid KV length was 1280. The following graph was submitted at
position 1280, but the host stopped while waiting for the previous graph's
sampled-token transfer.

This phase records one extra compute event only when that exact request reaches
effective length 1280. It then synchronizes that compute event before
synchronizing the existing sampled-token D2H event. It does not alter the
compiled decode function, cache tensors, graph inputs, packing, or normal
iterations.

### 29.1 Preflight

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "$PYTHON_BIN"

"$PYTHON_BIN" 09_persistent_page_engine/scripts/run_omnidocbench.py \
  --help | grep -q -- '--diagnostic-decode-effective-length'
"$PYTHON_BIN" 09_persistent_page_engine/scripts/run_omnidocbench.py \
  --help | grep -q -- '--diagnostic-decode-request-id'
```

Start from the exact successful-to-reproduce Phase-28 command. Do not change
its model paths, batch size, caches, min-pixels, text packing, vision packing,
PromptFA alignment, layout device, layout graph-capture setting, cache length,
or max-new-tokens. Keep:

```text
--preprocess-all-pages-first
--scheduler-progress
```

Add exactly:

```text
--diagnostic-decode-effective-length 1280
--diagnostic-decode-request-id page_000011_block_000003
```

These scheduler-only source changes must not require a new TorchAir decode
graph. Record whether the existing decode cache was reused; do not delete or
replace any cache directory.

### 29.2 Lane A: isolate the source page

Use the Phase-28 command with only these selection/output changes:

```text
--offset 11
--limit 1
--output-dir <LANE_A>/output
```

This preserves the original page numbering, so the target remains
`page_000011_block_000003`. The lane may contain the target page's other
layout regions, but removes all preceding-page history.

```sh
ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase29_decode1280_$COMMIT_SHORT"
LANE_A="$ROOT/page11_only"
test ! -e "$ROOT"
mkdir -p "$LANE_A/cann"
```

Create `$LANE_A/command.sh` with the fully expanded production command and
these environment variables before its final `exec`:

```sh
export ASCEND_PROCESS_LOG_PATH="<absolute LANE_A path>/cann"
export TORCH_NPU_COMPACT_ERROR_OUTPUT=0
export PYTHONUNBUFFERED=1
```

Run it in the background so the exact process is known and the log is durable:

```sh
bash "$LANE_A/command.sh" >"$LANE_A/run.log" 2>&1 &
RUN_PID=$!
printf '%s\n' "$RUN_PID" >"$LANE_A/pid.txt"
tail --pid="$RUN_PID" -f "$LANE_A/run.log"
wait "$RUN_PID"
printf '%s\n' "$?" >"$LANE_A/exit_code.txt"
```

If no new scheduler record appears for 90 seconds, capture evidence before
terminating only `$RUN_PID`:

```sh
grep '^EXP09_SCHEDULER ' "$LANE_A/run.log" \
  | tail -n 120 >"$LANE_A/last_scheduler_events.log"
npu-smi info >"$LANE_A/npu_smi_at_stall.txt" 2>&1 || true
find "$LANE_A/cann" -type f -print \
  | sort >"$LANE_A/cann_files.txt"
grep -R -n -E 'ERROR|AICore|aicore|5070|0x[0-9A-Fa-f]+' \
  "$LANE_A/cann" >"$LANE_A/cann_errors.txt" 2>&1 || true
kill -TERM "$RUN_PID"
wait "$RUN_PID" || true
```

Do not use `pkill` or `killall`.

### 29.3 Mechanical interpretation

At the target, the log must contain `diagnostic_pending_state` with:

```text
request_id=page_000011_block_000003
cache_position=1279
effective_length=1280
generated_tokens=258
```

Classify using the last unmatched diagnostic boundary:

| Observed sequence | Conclusion |
| --- | --- |
| `diagnostic_compute_sync_begin` with no end/error | the compiled decode graph at effective length 1280 did not complete |
| `diagnostic_compute_sync_error` | the compiled graph failed; report the exact Python/CANN error and first matching plog error |
| compute sync ends, then `diagnostic_d2h_sync_begin` has no end/error | compute completed; the sampled-token copy stream/D2H event chain is blocked |
| `diagnostic_d2h_sync_error` | D2H/event synchronization returned a runtime error; report it verbatim |
| both diagnostic syncs end | the target boundary completed in this reduced history; continue to Lane B |

The ordinary `pending_token_wait_begin` line is not the classification once
these more precise diagnostic events exist.

If Lane A blocks or errors at the target, stop after collecting its evidence.
The page-level lane has isolated the failure sufficiently for the next
operator-level experiment.

### 29.4 Lane B: changed history, original multi-page interaction

Run Lane B only if Lane A completes. Use the exact same command and caches,
changing only:

```text
--offset 4
--limit 16
--output-dir <LANE_B>/output
```

Set:

```sh
LANE_B="$ROOT/offset4_limit16"
mkdir -p "$LANE_B/cann"
```

Create `$LANE_B/command.sh` with its own absolute
`ASCEND_PROCESS_LOG_PATH`, run it with the same PID/log procedure, and apply
the same 90-second evidence capture. This lane tests whether the exact
1280-token boundary requires the larger active-slot/cache-state composition.

### 29.5 Report

Write `$ROOT/agent_report.md`:

```text
310P PHASE 29 DECODE LENGTH 1280: COMPUTE_BLOCK | COMPUTE_ERROR |
TOKEN_D2H_BLOCK | TOKEN_D2H_ERROR | HISTORY_DEPENDENT | PASS

commit / host / exact NPU / software:
exact Lane-A command:
Lane-A exit / result count:
target diagnostic_pending_state JSON:
last matched diagnostic boundary:
compute sync outcome and wait:
D2H sync outcome and wait:
first Python/CANN error:
first matching plog error:
decode graph cache reused or recompiled:
Lane-B command/outcome, if run:
mechanical conclusion:
evidence paths:
```

Paste back the report plus every `EXP09_SCHEDULER` line whose event begins
with `diagnostic_`. Do not summarize those lines away.

## Phase 30: dense-decode position boundary and physical-KV control

### 30.0 What Phase 29 established

Phase 29 reproduced the failure with one source page and one remaining active
decode slot. The fixed B16/KV4096 compiled graph was enqueued for:

```text
target row:       slot 3
cache_position:   1279
effective length: 1280
other rows:       inactive, cache_position 0
```

The separate compute event never completed. D2H was not reached. This removes
page history, ready-queue refill, slot swapping, and sampled-token transfer as
the primary cause.

The matching 910B page control also establishes that the target crop did not
use a 1280 prefill bucket:

```text
vision route:      compiled B1/S4096 (4032 real, 4096 physical)
text-prefill route: compiled packed 1024 (1021 real, 1024 physical)
decode boundary:   cache_position 1279 / effective length 1280
```

On 910B2 the full page crossed the diagnostic boundary, with both compute and
D2H syncs completing. The standalone mixed-position probe also passed target
positions 1277 through 1281 for both physical KV4096 and KV3584. These are
controls only; they do not predict 310P behavior.

### 30.1 Constraints and preflight

- Use the same NPU, Python environment, model, and software stack as Phase 29.
- Do not edit tracked files, create branches, commit, or push.
- Run each lane in its own process. A hung NPU process must be terminated by
  its exact PID only; never use `pkill` or `killall`.
- Do not use AICore utilization percentage as a causal diagnostic.
- Preserve the Phase-29 artifacts and compiler caches.

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git pull --ff-only origin main
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "$PYTHON_BIN"
PROBE=09_persistent_page_engine/scripts/probes/probe_dense_decode_position_boundary.py
"$PYTHON_BIN" "$PROBE" --help | grep -q -- '--inactive-position'
ROOT="$WORK_SERVER_REPO/tmp/09_persistent_page_engine/310p_phase30_decode_boundary_$COMMIT_SHORT"
test ! -e "$ROOT"
mkdir -p "$ROOT"
```

Use the same TorchAir decode cache root used by Phase 29 for the KV4096 lane.
The runtime creates a shape-specific child directory and must report
`batch_size=16`, `cache_length=4096`, `decode_attention=increfa`,
`decode_cache_update=npu_scatter`, and `decode_optimization=combined_apply`.
Do not trust a parent directory's historical `b32` name; inspect the emitted
runtime metadata.

### 30.2 Standalone mixed-position probe, KV4096

The probe builds the production dense `TextDecodeRuntime` with random model
weights and a zero-initialized KV cache. It is deliberately a shape/value
control, not real-input parity. Fifteen rows stay at position zero while slot 3
is tested at the boundary.

```sh
LANE="$ROOT/probe_k4096"
mkdir -p "$LANE"

timeout 600 "$PYTHON_BIN" "$PROBE" \
  --batch-size 16 \
  --cache-length 4096 \
  --target-row 3 \
  --inactive-position 0 \
  --positions 1277,1278,1279,1280,1281 \
  --cache-dir "$WORK_SERVER_REPO/.runtime_cache/310p_decode_b32_k4096_4789067" \
  --output "$LANE/result.json" \
  >"$LANE/run.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$LANE/pid.txt"
wait "$PID"
printf '%s\n' "$?" >"$LANE/exit_code.txt"
```

The probe flushes every event to both stdout and
`result.progress.jsonl`. If the timeout fires, report the last progress row.
The decisive boundary is:

```text
step_sync_begin target_position=1279 effective_length=1280
```

If this row has no matching `step_sync_end`, the synthetic production-shaped
graph reproduces the 310P hang without real images, prefill, scheduling, or
D2H.

### 30.3 Standalone mixed-position probe, KV3584

Run this lane regardless of whether KV4096 passes or times out. Use a new cache
root because KV3584 is a new compiled graph shape.

```sh
LANE="$ROOT/probe_k3584"
mkdir -p "$LANE"

timeout 900 "$PYTHON_BIN" "$PROBE" \
  --batch-size 16 \
  --cache-length 3584 \
  --target-row 3 \
  --inactive-position 0 \
  --positions 1277,1278,1279,1280,1281 \
  --cache-dir "$WORK_SERVER_REPO/.runtime_cache/310p_phase30_dense_k3584" \
  --output "$LANE/result.json" \
  >"$LANE/run.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$LANE/pid.txt"
wait "$PID"
printf '%s\n' "$?" >"$LANE/exit_code.txt"
```

Compilation may dominate setup. Distinguish compile/setup time from the five
step synchronizations.

### 30.4 Real page, physical KV3584

Copy the exact successful-to-reproduce Phase-29 Lane-A command. Change only:

```text
--cache-length 3584
--max-new-tokens 2048
--torchair-cache-dir .runtime_cache/310p_phase30_e2e_k3584
--output-dir <Phase-30 E2E output>
```

Keep:

```text
--offset 11 --limit 1 --batch-size 16
--diagnostic-decode-effective-length 1280
--diagnostic-decode-request-id page_000000_block_000003
--preprocess-all-pages-first --scheduler-progress
```

The smaller generation cap does not affect the request before generated token
259. For the 1021-token target prompt, the maximum required cache under this
lane is `1021 + 2048 - 1 = 3068`, which fits KV3584. This lane answers whether
the real-input failure depends on the physical KV4096 graph shape.

Use the same 90-second no-progress evidence procedure as Phase 29. Record all
`diagnostic_` events and whether a new KV3584 decode graph compiled.

### 30.5 Interpretation and report

| KV4096 probe | KV3584 probe/E2E | Interpretation |
| --- | --- | --- |
| hangs at position 1279 | also hangs there | position/prefix-dependent compiled-op failure, not specific to physical KV4096 |
| hangs at position 1279 | passes | interaction between effective length 1280 and the physical KV4096 graph/tiling |
| passes | real KV4096 E2E hangs | trigger depends on real token, RoPE delta, or accumulated KV contents rather than position shape alone |
| both probes pass, KV3584 E2E passes | physical KV shape or real KV4096 state remains the discriminator |

Write `$ROOT/agent_report.md`:

```text
310P PHASE 30 DENSE DECODE BOUNDARY: POSITION_ONLY | KV4096_INTERACTION |
REAL_STATE_DEPENDENT | UNRESOLVED

commit / host / exact NPU / software:
Phase-29 decode cache root:
KV4096 probe exact command / exit:
KV4096 runtime cache-key metadata:
KV4096 last progress row / per-position outcomes:
KV3584 probe exact command / exit:
KV3584 runtime cache-key metadata:
KV3584 last progress row / per-position outcomes:
KV3584 E2E exact changed arguments / exit:
KV3584 E2E target diagnostic events:
new compile versus cache replay for each lane:
mechanical classification from the table:
evidence paths:
```

Paste back the report, both probes' complete `DENSE_DECODE_BOUNDARY` progress
records, and every KV3584 E2E `EXP09_SCHEDULER` line beginning with
`diagnostic_`.

## Phase 31: minimal IncreFlashAttention boundary reproduction

### 31.0 Question and constraints

Phase 30 proved that the complete production-shaped compiled decoder hangs at
`cache_position=1279` (`effective_length=1280`) with both physical KV4096 and
KV3584, while the adjacent positions pass. This phase asks whether one
`npu_incre_flash_attention` call reproduces the boundary without the model,
decoder layers, KV scatter, scheduler, or page pipeline.

Do not use event polling, `ASCEND_LAUNCH_BLOCKING`, a profiler, or an in-process
timeout. Every measured process performs one ordinary event synchronization.
Use the shell `timeout` command as the external watchdog. Run every lane in its
own process and terminate only its exact PID.

### 31.1 Preflight

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PROBE=09_persistent_page_engine/scripts/probes/probe_incre_flash_attention_boundary.py
"$PYTHON_BIN" "$PROBE" --help | grep -q -- '--mask'

ROOT="$WORK_SERVER_REPO/tmp/09_persistent_page_engine/310p_phase31_minimal_increfa_$COMMIT_SHORT"
test ! -e "$ROOT"
mkdir -p "$ROOT"
```

The probe uses synthetic Q/K/V tensors at the production B16/KV4096/head
shapes. It issues only one `npu_incre_flash_attention` operation per tested
position. Its `step_sync_begin`/`step_sync_end` records are flushed before and
after the sole synchronization boundary.

For compiled lanes, first compile/replay the identical graph on safe positions
1278 and 1280. Only after that process passes, start a fresh process using the
same cache and test position 1279 alone. This prevents compile time from being
confused with the hang timeout.

### 31.2 Lane A: compiled, production position mask

```sh
LANE="$ROOT/A_compiled_masked"
CACHE="$WORK_SERVER_REPO/.runtime_cache/310p_phase31_increfa_masked"
mkdir -p "$LANE"

timeout 900 "$PYTHON_BIN" "$PROBE" \
  --backend torchair --mask position --cache-init zeros \
  --batch-size 16 --cache-length 4096 --target-row 3 --inactive-position 0 \
  --positions 1278,1280 --cache-dir "$CACHE" \
  --output "$LANE/safe.json" >"$LANE/safe.log" 2>&1
printf '%s\n' "$?" >"$LANE/safe_exit_code.txt"

timeout --signal=TERM --kill-after=5s 60 "$PYTHON_BIN" "$PROBE" \
  --backend torchair --mask position --cache-init zeros \
  --batch-size 16 --cache-length 4096 --target-row 3 --inactive-position 0 \
  --positions 1279 --cache-dir "$CACHE" \
  --output "$LANE/boundary.json" >"$LANE/boundary.log" 2>&1
printf '%s\n' "$?" >"$LANE/boundary_exit_code.txt"
```

Do not run the boundary command unless the safe command exits zero. Confirm
that the boundary process reused the safe command's cache rather than compiling
a new shape.

### 31.3 Lane B: eager, production position mask

```sh
LANE="$ROOT/B_eager_masked"
mkdir -p "$LANE"

timeout --signal=TERM --kill-after=5s 60 "$PYTHON_BIN" "$PROBE" \
  --backend eager --mask position --cache-init zeros \
  --batch-size 16 --cache-length 4096 --target-row 3 --inactive-position 0 \
  --positions 1278,1280,1279 \
  --output "$LANE/result.json" >"$LANE/run.log" 2>&1
printf '%s\n' "$?" >"$LANE/exit_code.txt"
```

Position 1279 is deliberately last, so both safe controls remain recorded if
the final synchronization hangs.

### 31.4 Lane C: compiled, no attention mask

```sh
LANE="$ROOT/C_compiled_nomask"
CACHE="$WORK_SERVER_REPO/.runtime_cache/310p_phase31_increfa_nomask"
mkdir -p "$LANE"

timeout 900 "$PYTHON_BIN" "$PROBE" \
  --backend torchair --mask none --cache-init zeros \
  --batch-size 16 --cache-length 4096 --target-row 3 --inactive-position 0 \
  --positions 1278,1280 --cache-dir "$CACHE" \
  --output "$LANE/safe.json" >"$LANE/safe.log" 2>&1
printf '%s\n' "$?" >"$LANE/safe_exit_code.txt"

timeout --signal=TERM --kill-after=5s 60 "$PYTHON_BIN" "$PROBE" \
  --backend torchair --mask none --cache-init zeros \
  --batch-size 16 --cache-length 4096 --target-row 3 --inactive-position 0 \
  --positions 1279 --cache-dir "$CACHE" \
  --output "$LANE/boundary.json" >"$LANE/boundary.log" 2>&1
printf '%s\n' "$?" >"$LANE/boundary_exit_code.txt"
```

### 31.5 Mechanical interpretation and report

| Lane A | Lane B | Lane C | Conclusion |
| --- | --- | --- | --- |
| hangs | passes | passes | compiled masked IncreFA lowering/tiling bug |
| hangs | hangs | passes | eager and compiled masked IncreFA kernel/mask bug |
| hangs | hangs | hangs | IncreFA problem not specific to compilation or the mask |
| passes | passes | passes | the full decoder graph interaction is required; one IncreFA call is insufficient |
| passes | hangs | any | unexpected; report without interpreting |

For a hang, the decisive evidence is a final flushed
`step_sync_begin target_position=1279 effective_length=1280` with no matching
`step_sync_end`, followed by shell exit 124. Record the wall-time lower bound as
`>60 s`; do not report an NPU kernel duration because the completion event was
never reached.

Write `$ROOT/agent_report.md` containing:

```text
310P PHASE 31 MINIMAL INCREFA BOUNDARY: COMPILED_MASK | KERNEL_MASK |
GENERAL_INCREFA | FULL_GRAPH_INTERACTION | UNRESOLVED

commit / host / exact NPU / software:
Lane A safe and boundary exact commands / exits / progress records:
Lane B exact command / exit / progress record:
Lane C safe and boundary exact commands / exits / progress records:
cache compile versus replay evidence for A and C:
first Python/CANN/plog error, if any:
mechanical classification from the table:
evidence paths:
```

Paste back the complete report and every `INCRE_FA_BOUNDARY` line. Do not
summarize the progress records away.

## Phase 32: standalone masked-GQA IncreFA discriminator

### 32.0 Why this phase exists

Phase 31 established the following 310P result at cache position 1279
(`effective_length=1280`):

- compiled IncreFA with the production bool position mask hung;
- eager IncreFA with the same mask hung;
- compiled IncreFA with `atten_mask=None` passed;
- an additional eager `atten_mask=None` lane also passed.

TorchAir is therefore no longer the discriminator. This phase removes the
model config and every repository runtime helper as well. It asks a narrower
question: is the nonterminating path specifically masked GQA on 310P?

The low-level CANN contract is important here. PaddleOCR-VL uses 16 query heads
and 2 KV heads, passing `numKeyValueHeads=2`. The CANN
`aclnnIncreFlashAttention` documentation says Atlas inference-series
accelerator cards support only `numKeyValueHeads=0`; zero denotes MHA with the
same number of query and KV heads. Nonzero GQA is documented for Atlas A2.

References:

- TorchNPU 26.0 API:
  <https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_incre_flash_attention.md>
- low-level `aclnnIncreFlashAttention` contract:
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/context/ops-transformer/aclnnIncreFlashAttention.md>
- current TorchNPU/CANN compatibility table:
  <https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md>

Do not describe the expected hang as acceptable behavior merely because the
GQA configuration is outside the documented 310P contract. A submitted kernel
that never completes, returns no error, and requires killing the process is
still a vendor-quality failure mode. This phase identifies the smallest exact
trigger; it does not choose a production workaround.

### 32.1 Constraints and preflight

The probe is one standalone Python file. It imports no project module, reads no
model file, performs no TorchAir compile, and runs exactly one eager
`npu_incre_flash_attention` call followed by one ordinary device
synchronization. Do not add polling, a profiler, `ASCEND_LAUNCH_BLOCKING`, an
in-process alarm, or other pipeline code.

Run every lane in a fresh process under the shell timeout. Run passing controls
before a potentially hanging lane. Do not continue using a device if killing a
hung process leaves it unhealthy.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PROBE=09_persistent_page_engine/scripts/probes/probe_increfa_masked_gqa.py
"$PYTHON_BIN" "$PROBE" --help | grep -q -- '--lane'

ROOT="$WORK_SERVER_REPO/tmp/09_persistent_page_engine/310p_phase32_increfa_gqa_$COMMIT_SHORT"
test ! -e "$ROOT"
mkdir -p "$ROOT"

{
  git rev-parse HEAD
  hostname
  npu-smi info
  "$PYTHON_BIN" -m pip show torch torch-npu
} >"$ROOT/preflight.log" 2>&1
```

The probe itself records the Python, torch, torch_npu, and
`torch_npu.version.git_version` values in each completed result and in the
flushed `setup_begin` progress row. Preserve that full row even when a lane
hangs. This matters because `torch_npu.__version__ == 2.10.0` alone does not
distinguish the CANN-9.0 `v2.10.0-26.0.0` build from the CANN-9.1
`v2.10.0-26.1.0` build.

Use this shell helper for each lane. It records the exact command, streams the
probe output into a log, and preserves timeout exit 124:

```sh
run_lane() {
  lane_name="$1"
  shift
  lane_dir="$ROOT/$lane_name"
  mkdir -p "$lane_dir"
  printf '%q ' timeout --signal=TERM --kill-after=5s 60 \
    "$PYTHON_BIN" "$PROBE" "$@" --output "$lane_dir/result.json" \
    >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"
  timeout --signal=TERM --kill-after=5s 60 \
    "$PYTHON_BIN" "$PROBE" "$@" --output "$lane_dir/result.json" \
    >"$lane_dir/run.log" 2>&1
  lane_exit="$?"
  printf '%s\n' "$lane_exit" >"$lane_dir/exit_code.txt"
  return "$lane_exit"
}
```

Because an expected timeout is nonzero, invoke each helper call with
`set +e` or as the condition of an `if`; do not let `set -e` abort before the
exit code is saved.

### 32.2 Passing controls first

Run GQA without a mask. This exactly retains the production 16:2 head ratio but
avoids the mask-selected kernel route:

```sh
set +e
run_lane A_b1_gqa_nomask \
  --lane gqa_nomask --batch-size 1 --batch-pattern uniform \
  --cache-length 4096 --effective-length 1280
A_EXIT="$?"
set -e
test "$A_EXIT" -eq 0
```

Next run masked MHA using 16 stored KV heads and
`num_key_value_heads=0`. This is the documented Atlas inference-series form and
uses the same bool mask contents as the failing GQA lane:

```sh
set +e
run_lane B_b1_mha_masked \
  --lane mha_masked --batch-size 1 --batch-pattern uniform \
  --cache-length 4096 --effective-length 1280
B_EXIT="$?"
set -e
test "$B_EXIT" -eq 0
```

Stop and report `CONTROL_FAILURE` if either control fails or hangs. Do not run
the trigger lane after a failed control.

### 32.3 Minimal trigger: B1 masked GQA

Only after both controls pass, run the smallest suspected trigger:

```sh
set +e
run_lane C_b1_gqa_masked \
  --lane gqa_masked --batch-size 1 --batch-pattern uniform \
  --cache-length 4096 --effective-length 1280
C_EXIT="$?"
set -e
```

If this exits 124 with a final `sync_begin` and no `sync_end`, stop. The issue
has been reduced to one B1 eager masked-GQA IncreFA call.

### 32.4 Conditional batch-shape escalation

Run this section only if Lane C exits zero. It determines whether the trigger
requires B16 or mixed per-row prefix lengths.

First test B16 with every row at effective length 1280:

```sh
set +e
run_lane D_b16_gqa_masked_uniform \
  --lane gqa_masked --batch-size 16 --batch-pattern uniform \
  --cache-length 4096 --effective-length 1280
D_EXIT="$?"
set -e
```

If D passes, reproduce the Phase-31 mask geometry: row 3 has effective length
1280 while all other rows have effective length 1.

```sh
set +e
run_lane E_b16_gqa_masked_mixed \
  --lane gqa_masked --batch-size 16 --batch-pattern mixed \
  --target-row 3 --inactive-effective-length 1 \
  --cache-length 4096 --effective-length 1280
E_EXIT="$?"
set -e
```

Stop after the first hang. Do not rerun a hanging lane merely to collect more
wall time.

### 32.5 Classification and report

| A GQA no mask | B MHA mask | C B1 GQA mask | D/E if needed | Classification |
| --- | --- | --- | --- | --- |
| pass | pass | hangs | not run | `B1_MASKED_GQA` |
| pass | pass | pass | B16 uniform hangs | `BATCHED_MASKED_GQA` |
| pass | pass | pass | uniform passes, mixed hangs | `MIXED_PREFIX_MASKED_GQA` |
| pass | pass | pass | both pass | `PHASE31_CONTEXT_REQUIRED` |
| any control fails | any | not run | not run | `CONTROL_FAILURE` |

For a hang, require all three pieces of evidence:

1. shell exit 124;
2. the last flushed `INCREFA_GQA` event is `sync_begin`;
3. no `sync_end`, Python exception, CANN error, or AICore exception appears.

Report the duration only as `>60 s`. Do not call it a 60-second kernel time.

Write `$ROOT/agent_report.md`:

```text
310P PHASE 32 STANDALONE MASKED GQA: B1_MASKED_GQA |
BATCHED_MASKED_GQA | MIXED_PREFIX_MASKED_GQA |
PHASE31_CONTEXT_REQUIRED | CONTROL_FAILURE

commit / host / exact NPU / software:
torch_npu version and git_version:
CANN / driver / firmware:
Lane A exact command / exit / complete progress:
Lane B exact command / exit / complete progress:
Lane C exact command / exit / complete progress:
conditional Lane D/E exact commands / exits / complete progress:
last event for every timed-out lane:
first Python/CANN/plog error, if any:
mechanical classification from the table:
what is proven:
what is not proven:
evidence paths:
```

Paste back the report and every `INCREFA_GQA` progress line. Do not summarize
the version metadata or omit passing-control progress.

## Phase 33: full-decoder masked-GQA boundary in the text-decode lab

### 33.0 What this phase asks

Phase 32 produced this exact 310P discriminator:

- B1 GQA without a mask passed;
- B1 masked MHA passed;
- B1 masked GQA passed;
- B16 masked GQA with every row at effective length 1280 hung;
- an additional B16 masked-MHA control passed.

The trigger is therefore **batched masked GQA**, not masked GQA in general.
Phase 33 moves that result into the production-faithful text-decode lab. It
runs the real 18-layer PaddleOCR-VL text decoder, production decode arena,
production bool position mask, cache update, LM head, and argmax at B16,
physical KV4096, and `cache_position=1279`.

There are two decoder implementations in this phase:

- `combined_apply` is the unchanged production GQA path: 16 query heads, two
  stored KV heads, and `num_key_value_heads=2` in IncreFA.
- `combined_apply_mha_repeat` is a lab-only discriminator. The persistent
  cache remains `[B, 2, 4096, 128]`. Within each decoder layer, immediately
  before IncreFA, that layer's K and V are expanded 2 to 16 heads and the op is
  called with `num_key_value_heads=0`. Do not promote this preset into the E2E
  runner in this phase.

This distinction matters for memory interpretation. At B16/KV4096/fp16, the
ordinary persistent K+V allocation is 64 MiB per layer, or 1.125 GiB across 18
layers. The MHA lane does **not** make that persistent allocation eight times
larger. It materializes up to 512 MiB of K+V for the current layer, an
additional 448 MiB transient tensor footprint before compiler workspaces and
allocator reuse. This phase only establishes termination and numerical sanity;
it does not yet decide whether that overhead is acceptable.

The matching 910B controls are already recorded under:

```text
tmp/09_persistent_page_engine/text_decode_boundary_910b_5f1b27a/
```

Both raw-eager and TorchAir GQA/MHA lanes passed at position 1279. The compiled
single-step times were 12.57 ms for GQA and 17.96 ms for repeated-KV MHA. A
four-step, 16-request compiled-MHA correctness control had mean logit error
0.01149, max 0.27539, and 64/64 argmax matches against baseline eager. These
are 910B controls only and do not predict 310P termination or speed.

### 33.1 Constraints and preflight

- Pull `main` and record the exact commit. The required source commit is
  `5f1b27a` or a descendant containing it.
- Do not edit tracked files, create a branch, commit, or push.
- Use the same working Python/NPU environment and model directory as the
  successful Phase 32 and Experiment 09 runs.
- Use one NPU. Run every lane in a fresh process.
- Passing lanes must be run before the known hanging trigger.
- A hanging process must be terminated by its exact PID only. Never use
  `pkill` or `killall`.
- Do not interpret AICore utilization as proof of progress.
- Do not run an E2E page workload in this phase.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LAB=09_persistent_page_engine/scripts/text_decode_lab.py
MODEL_DIR="${MODEL_DIR:-/workspace/models/PaddleOCR-VL-1.6}"
CACHE_ROOT="$WORK_SERVER_REPO/.runtime_cache/310p_phase33_text_decode"
ROOT="$WORK_SERVER_REPO/tmp/09_persistent_page_engine/310p_phase33_text_decode_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -d "$MODEL_DIR"
"$PYTHON_BIN" "$LAB" --help | grep -q -- 'boundary'
"$PYTHON_BIN" "$LAB" --help | grep -q -- 'combined_apply_mha_repeat'
test ! -e "$ROOT"
mkdir -p "$ROOT"

{
  printf 'commit=%s\n' "$COMMIT"
  hostname
  npu-smi info
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
import torch_npu
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
PY
} >"$ROOT/preflight.log" 2>&1
```

Use this helper. It records the expanded command, streams flushed lab markers
to a log, records the exit code, and bounds setup/compile plus a possible
silent device hang. The compile lanes receive 1,200 seconds because a new
TorchAir cache key is expected. The final trigger reuses the GQA cache and
receives 300 seconds; if setup has not reached `step_begin` in that time,
classify it as `SETUP_TIMEOUT`, not the expected boundary hang.

```sh
run_lane() {
  lane_name="$1"
  timeout_s="$2"
  shift 2
  lane_dir="$ROOT/$lane_name"
  mkdir -p "$lane_dir"
  printf '%q ' timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$LAB" "$@" --output "$lane_dir/result.json" \
    >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"
  set +e
  timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$LAB" "$@" --output "$lane_dir/result.json" \
    >"$lane_dir/run.log" 2>&1
  lane_exit="$?"
  printf '%s\n' "$lane_exit" >"$lane_dir/exit_code.txt"
  grep 'TEXT_DECODE_BOUNDARY' "$lane_dir/run.log" \
    >"$lane_dir/boundary_events.log" || true
  return "$lane_exit"
}

common_args="--model $MODEL_DIR --cache-dir $CACHE_ROOT --batch-size 16 --active-slots 16 --cache-length 4096 --profile-position 1279"
```

When reporting a timeout, inspect the last flushed marker:

- no `step_begin`: setup or compile timeout; not a boundary reproduction;
- `step_begin` without `step_returned`: the Python graph call itself did not
  return;
- `sync_begin` without `sync_end` or `sync_error`: device computation did not
  complete;
- `sync_error`: report the full Python/CANN error verbatim.

### 33.2 Lane A: raw-eager full-decoder MHA control

This confirms that the 18-layer implementation and transient per-layer repeat
are valid before involving TorchAir.

```sh
set +e
run_lane A_mha_raw_pos1279 600 \
  --mode boundary --backend raw_eager $common_args \
  --decode-optimization combined_apply_mha_repeat
A_EXIT="$?"
set -e
test "$A_EXIT" -eq 0
```

Require `step_begin`, `step_returned`, `sync_begin`, and `sync_end`, in that
order. Stop as `MHA_RAW_FAILURE` if this lane errors or times out.

### 33.3 Lane B: compiled full-decoder MHA at the trigger position

This is the direct candidate-control lane. It compiles one B16/KV4096 graph,
then executes it at position 1279.

```sh
set +e
run_lane B_mha_torchair_pos1279 1200 \
  --mode boundary --backend torchair --allow-compile $common_args \
  --decode-optimization combined_apply_mha_repeat
B_EXIT="$?"
set -e
test "$B_EXIT" -eq 0
```

Require all four completion markers. Also require runtime metadata to report
`backend=torchair`, `batch_size=16`, `cache_length=4096`,
`decode_optimization=combined_apply_mha_repeat`, and `attention=mha_repeat`.
Stop as `MHA_COMPILED_FAILURE` if this lane fails.

### 33.4 Lane C: compile/reuse the production GQA graph at position 1278

The graph shape is identical to the trigger. Only the runtime cache position is
one token earlier. This lane creates the GQA cache without asking the compiler
to run first at the known bad boundary.

```sh
set +e
run_lane C_gqa_torchair_pos1278 1200 \
  --mode boundary --backend torchair --allow-compile \
  --model "$MODEL_DIR" --cache-dir "$CACHE_ROOT" \
  --batch-size 16 --active-slots 16 --cache-length 4096 \
  --profile-position 1278 --decode-optimization combined_apply
C_EXIT="$?"
set -e
test "$C_EXIT" -eq 0
```

Require all four completion markers and metadata with `attention=gqa`. Record
the exact shape-cache directory and whether it was created or reused. Stop as
`GQA_SAFE_CONTROL_FAILURE` if this lane fails.

### 33.5 Lane D: cached production GQA at position 1279

Run the expected trigger only after A-C pass. Deliberately omit
`--allow-compile`. The exact B16/KV4096/source/optimization cache created in
Lane C must be reused, so a timeout after `sync_begin` cannot be confused with
compilation.

```sh
set +e
run_lane D_gqa_torchair_pos1279 300 \
  --mode boundary --backend torchair $common_args \
  --decode-optimization combined_apply
D_EXIT="$?"
set -e
```

Expected 310P result: exit 124, with `step_begin`, `step_returned`, and
`sync_begin`, but no `sync_end`, `sync_error`, Python traceback, CANN error, or
AICore exception. Report the duration only as `>boundary wait`; do not call the
300-second process timeout a kernel duration because model setup occurs first.

If Lane D exits zero, classify `TEXT_LAB_GQA_PASSED`; the standalone trigger
does not reproduce in the full compiled decoder. If it errors, classify
`TEXT_LAB_GQA_ERROR` and preserve the full error. If it times out before
`step_begin`, classify `SETUP_TIMEOUT` and do not claim reproduction.

### 33.6 Optional numerical check after A-D

Run only if Lane B passed and the device remains healthy after Lane D is
terminated. This reuses the compiled MHA graph and checks four decoder steps on
16 recorded requests. It is not an accuracy certification.

```sh
set +e
run_lane E_mha_correctness 600 \
  --mode correctness --backend torchair \
  --model "$MODEL_DIR" --cache-dir "$CACHE_ROOT" \
  --batch-size 16 --cache-length 4096 \
  --correctness-items 16 --correctness-steps 4 \
  --decode-optimization combined_apply_mha_repeat
E_EXIT="$?"
set -e
```

This mode has no boundary markers. Require exit zero and report mean/max logit
error, argmax matches, and written-KV mean/max error. Do not reject solely on a
moderate maximum-logit difference; mean error and token decisions are more
informative here.

### 33.7 Report

Write `$ROOT/agent_report.md`:

```text
310P PHASE 33 TEXT-DECODE LAB BOUNDARY: TEXT_LAB_REPRODUCED_MHA_PASSES |
TEXT_LAB_GQA_PASSED | TEXT_LAB_GQA_ERROR | MHA_RAW_FAILURE |
MHA_COMPILED_FAILURE | GQA_SAFE_CONTROL_FAILURE | SETUP_TIMEOUT

commit / host / exact NPU / software:
torch_npu version and git_version:
CANN / driver / firmware:
model path:
Lane A exact command / exit / complete boundary markers / elapsed:
Lane B exact command / exit / complete boundary markers / elapsed:
Lane B runtime metadata and shape-cache path:
Lane C exact command / exit / complete boundary markers / elapsed:
Lane C runtime metadata and shape-cache path / created or reused:
Lane D exact command / exit / complete boundary markers:
Lane D last marker and first Python/CANN/plog error, if any:
Lane E correctness metrics, if run:
peak NPU memory reported for passing GQA and MHA lanes, if available:
mechanical classification:
what is proven:
what is not proven:
evidence paths:
```

Paste back the report and every `TEXT_DECODE_BOUNDARY` line from A-D. Do not
summarize the event sequence away. In particular, the result is only
`TEXT_LAB_REPRODUCED_MHA_PASSES` if B completes and D reaches `sync_begin` but
never reaches `sync_end` or `sync_error`.

## Phase 34: real B16 OCR generation through effective length 1280

### 34.0 Question and fixed workload

Phase 33 proved that the complete compiled MHA decoder can execute one
synthetic step at the failing 310P boundary. Phase 34 tests the part that the
one-step and four-step probes cannot establish: real autoregressive generation
with real vision/text prefills, production KV admission, 16 active slots, 374
decode iterations, and the natural transition through cache position 1279.

Use the committed 910B GQA output as the semantic reference:

```text
tmp/09_persistent_page_engine/real_decode_generation_910b_e257add/gqa_reference.json
```

The fixed source is OmniDocBench page
`page-573c437e-c309-4483-a038-ef2f440b104a.png`, owned-layout block 3. The
script asserts a 1022-by-772 crop, prompt `Table Recognition:`, and exactly
1,021 input tokens. It duplicates that same genuine crop into 16 independent
requests. Do not replace it with synthetic KV, shorten generation, change the
crop, or lower the target. All 16 requests must naturally cross effective
length 1280 and terminate by EOS.

The candidate remains lab-only. Do not change the Experiment 09 production
default from `combined_apply`.

### 34.1 Preflight

Pull `main` and require commit `9862b09` or a descendant. Use the same working
environment, model paths, and NPU-selection procedure as Phase 33. Do not edit
tracked files, create a branch, commit, or push. Use one fresh process and an
external timeout. Never terminate unrelated processes.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LAB="$REPO/09_persistent_page_engine/scripts/text_decode_real_generation.py"
MODEL_DIR="${MODEL_DIR:-/workspace/models/PaddleOCR-VL-1.6}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV3_safetensors}"
PAGE_IMAGE="${PAGE_IMAGE:-/workspace/datasets/OmniDocBench/images/page-573c437e-c309-4483-a038-ef2f440b104a.png}"
CACHE_ROOT="$REPO/.runtime_cache/310p_phase33_text_decode"
REFERENCE="$REPO/tmp/09_persistent_page_engine/real_decode_generation_910b_e257add/gqa_reference.json"
ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase34_real_generation_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -f "$LAB"
test -d "$MODEL_DIR"
test -d "$LAYOUT_MODEL"
test -f "$PAGE_IMAGE"
test -s "$REFERENCE"
"$PYTHON_BIN" "$LAB" --help | grep -q -- 'combined_apply_mha_repeat'
test ! -e "$ROOT"
mkdir -p "$ROOT/mha"

{
  printf 'commit=%s\n' "$COMMIT"
  hostname
  npu-smi info
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
import torch_npu
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
PY
} >"$ROOT/preflight.log" 2>&1
```

If Phase 33 used a different cache root, set `CACHE_ROOT` to that exact root.
The MHA B16/KV4096 graph should normally be reused. If no matching graph is
present and setup compiles one, report the compile explicitly and do not count
setup time as generation time.

### 34.2 Run the real MHA lane

Record the exact expanded command, keep live output in a log, preserve the
exit code, and allow setup/cache loading plus generation enough time. The lab
prints only the boundary synchronization events, not every scheduler event, so
the log itself does not dominate host time.

```sh
printf '%q ' timeout --signal=TERM --kill-after=10s 1200 \
  "$PYTHON_BIN" "$LAB" \
  --page-image "$PAGE_IMAGE" \
  --layout-model "$LAYOUT_MODEL" \
  --layout-device cpu \
  --layout-model-backend owned \
  --recognizer-model "$MODEL_DIR" \
  --decode-cache-dir "$CACHE_ROOT" \
  --decode-backend torchair \
  --decode-optimization combined_apply_mha_repeat \
  --batch-size 16 --replicas 16 \
  --cache-length 4096 --max-new-tokens 512 \
  --target-effective-length 1280 \
  --min-pixels 28224 \
  --vision-backend raw_eager \
  --vision-attention prompt_flash_attention \
  --vision-promptfa-align-128 \
  --text-backend raw_eager \
  --reference "$REFERENCE" \
  --output "$ROOT/mha/result.json" \
  >"$ROOT/mha/command.sh"
printf '\n' >>"$ROOT/mha/command.sh"

set +e
timeout --signal=TERM --kill-after=10s 1200 \
  "$PYTHON_BIN" "$LAB" \
  --page-image "$PAGE_IMAGE" \
  --layout-model "$LAYOUT_MODEL" \
  --layout-device cpu \
  --layout-model-backend owned \
  --recognizer-model "$MODEL_DIR" \
  --decode-cache-dir "$CACHE_ROOT" \
  --decode-backend torchair \
  --decode-optimization combined_apply_mha_repeat \
  --batch-size 16 --replicas 16 \
  --cache-length 4096 --max-new-tokens 512 \
  --target-effective-length 1280 \
  --min-pixels 28224 \
  --vision-backend raw_eager \
  --vision-attention prompt_flash_attention \
  --vision-promptfa-align-128 \
  --text-backend raw_eager \
  --reference "$REFERENCE" \
  --output "$ROOT/mha/result.json" \
  2>&1 | tee "$ROOT/mha/run.log"
MHA_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$MHA_EXIT" >"$ROOT/mha/exit_code.txt"
grep 'EXP09_SCHEDULER' "$ROOT/mha/run.log" \
  >"$ROOT/mha/boundary_events.log" || true
```

Do not run the production GQA generation lane first: its 310P failure at this
boundary is already established and can leave the device unhealthy. If the
MHA lane passes and Luka later asks for a same-machine GQA confirmation, run it
last in a new process with a strict timeout.

### 34.3 Mechanical checks

Require exit zero and parse `result.json`; do not infer success from the final
console line alone. Require all of the following:

- `configuration.decode_optimization == "combined_apply_mha_repeat"`;
- 16 results and `cohort.input_tokens == [1021]`;
- every request stops by `eos`, all 16 requests have the same generated
  length, and every request crosses effective length 1280;
- `cohort.all_crossed_target`, `all_token_ids_identical`, and
  `all_text_identical` are true;
- report the 910B reference comparison, but do not require cross-platform
  token identity; require the 16 requests within the 310P cohort to agree;
- the log has one `diagnostic_pending_state` at cache position 1279/effective
  length 1280, followed by matching compute-sync and D2H-sync end events;
- no Python traceback, CANN error, AICore error, or timeout.

Report these performance numbers from the JSON rather than estimating them
from shell wall time:

- `run_wall_s` and `schedule.timing_s.continuous_decode_wall`;
- raw, effective, and effective-device decode tok/s;
- `schedule.timing_s.decode_model_and_argmax_device`;
- `schedule.timing_s.d2h_wait_wall` and `retire_and_refill_host_wall`;
- graph calls, raw/effective token totals, and effective fraction;
- memory before, peak, and peak delta;
- setup time and whether the graph was reused or compiled.

For context only, the corresponding 910B MHA result is 916.1 effective tok/s,
985.2 effective-device tok/s, 6.057 s model-plus-argmax device time, and a
263,521,792-byte peak delta. The 910B production GQA reference is 4,104.4
effective tok/s and 0.940 s model-plus-argmax device time. Do not treat those
as 310P pass thresholds.

### 34.4 Report

Write `$ROOT/agent_report.md`:

```text
310P PHASE 34 REAL B16 GENERATION: MHA_REAL_GENERATION_PASS |
MHA_BOUNDARY_HANG | MHA_RUNTIME_ERROR |
SETUP_TIMEOUT

commit / host / exact NPU / software:
torch_npu version and git_version:
CANN / driver / firmware:
exact command / exit code:
model, layout model, page image, cache and reference paths:
graph reused or compiled:
source crop size / prompt / input-token count:
request count / generated lengths / stop reasons:
all 16 crossed effective length 1280:
boundary compute and D2H event sequence with wait times:
reference token / text / stop-reason exactness:
run wall / continuous-decode wall:
raw / effective / effective-device decode tok/s:
model-plus-argmax device / D2H wait / retire-refill wall:
graph calls / raw and effective token totals / effective fraction:
memory before / peak / peak delta:
first Python/CANN/plog error, if any:
mechanical classification:
what is proven:
what is not proven:
evidence paths:
```

Paste back the report and the complete `boundary_events.log`. A passing Phase
34 proves that the MHA workaround survives real B16 OCR generation across the
faulting boundary with a stable, internally identical 310P cohort. It does not
make MHA a production choice; its 310P sustained cost still has to be judged
against alternatives. The completed 310P Phase 34 generated 366 tokens per
request, stopped all 16 by EOS, and measured 128.6 effective tok/s with no
runtime error. It diverged from the 910B reference at the first token and used
different table markup while preserving the table content; this
cross-platform formatting trajectory is recorded, not treated as a boundary
failure.

## Phase 35: B4 MHA boundary and first complete 310P E2E result

### 35.0 Goal and stopping order

Phase 34 proved that repeated-KV MHA avoids the 310P masked-GQA hang during
real B16 generation, but measured only 128.6 effective tok/s. Phase 35 lowers
decode batch size to four and connects the same workaround to the real
OmniDocBench runner. Execute exactly in this order:

1. compile and complete one B4/KV4096 text-decode boundary step at position
   1279;
2. run the one full page that contains the 1,021-token table crop and require
   every crop to finish;
3. only after both pass, run the first eight OmniDocBench pages for an E2E
   throughput result.

Do not begin with 32 pages. Do not enable scheduler-progress logging or the
timeline during the timed E2E lanes. Use layout-first mode so layout and OCR
do not contend for the NPU and the recognition timing remains interpretable.
The only intended behavior change from the existing optimized pipeline is
`batch_size=4` plus `combined_apply_mha_repeat`.

The matching 910B controls completed at commit `c5c3a6e`: the B4 boundary
step synchronized at effective length 1280, and the target page completed nine
of nine crops by EOS in 3.02 s, with 1.90 s decode wall. These numbers are not
310P thresholds.

### 35.1 Preflight

Pull `main` and require commit `389d4e0` or a descendant. Do not edit tracked
files, create branches, commit, or push. Use the same working NPU environment
and model/dataset paths as Phases 33-34. Use one NPU and terminate only a PID
created by these commands.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/PaddleOCR-VL-1.6}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV3_safetensors}"
DATASET_JSON="${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/datasets/OmniDocBench/images}"
TEXT_LAB="$REPO/09_persistent_page_engine/scripts/text_decode_lab.py"
E2E="$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py"
DECODE_CACHE="$REPO/.runtime_cache/310p_phase35_mha_b4"
ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase35_mha_b4_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -d "$MODEL_DIR"
test -d "$LAYOUT_MODEL"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -f "$TEXT_LAB"
test -f "$E2E"
"$PYTHON_BIN" "$TEXT_LAB" --help | grep -q -- 'combined_apply_mha_repeat'
"$PYTHON_BIN" "$E2E" --help | grep -q -- '--decode-optimization'
test ! -e "$ROOT"
mkdir -p "$ROOT/text_boundary" "$ROOT/target_page" "$ROOT/eight_pages"

{
  printf 'commit=%s\n' "$COMMIT"
  hostname
  npu-smi info
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
import torch_npu
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
PY
} >"$ROOT/preflight.log" 2>&1
```

Use this helper for the two E2E lanes. It records the exact command, streams
page progress into `run.log`, preserves the real pipeline exit code, and
bounds a possible silent device hang.

```sh
run_e2e() {
  lane="$1"
  timeout_s="$2"
  offset="$3"
  limit="$4"
  lane_dir="$ROOT/$lane"
  output_dir="$lane_dir/output"
  shift 4

  printf '%q ' timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset "$offset" --limit "$limit" \
    --batch-size 4 --cache-length 4096 --max-new-tokens 2808 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_mha_repeat \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-promptfa-align-128 \
    --vision-packing greedy --vision-pack-target 1920 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output_dir" "$@" >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"

  set +e
  timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset "$offset" --limit "$limit" \
    --batch-size 4 --cache-length 4096 --max-new-tokens 2808 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_mha_repeat \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-promptfa-align-128 \
    --vision-packing greedy --vision-pack-target 1920 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output_dir" "$@" 2>&1 | tee "$lane_dir/run.log"
  lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" >"$lane_dir/exit_code.txt"
  return "$lane_exit"
}
```

### 35.2 Gate A: B4 compiled text-decode boundary

This creates the exact B4/KV4096/repeated-MHA graph cache that the E2E runner
must reuse. It is a full 18-layer decoder step, not a standalone IncreFA call.

```sh
printf '%q ' timeout --signal=TERM --kill-after=10s 1200 \
  "$PYTHON_BIN" "$TEXT_LAB" \
  --mode boundary --backend torchair --allow-compile \
  --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
  --batch-size 4 --active-slots 4 --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_mha_repeat \
  --output "$ROOT/text_boundary/result.json" \
  >"$ROOT/text_boundary/command.sh"
printf '\n' >>"$ROOT/text_boundary/command.sh"

set +e
timeout --signal=TERM --kill-after=10s 1200 \
  "$PYTHON_BIN" "$TEXT_LAB" \
  --mode boundary --backend torchair --allow-compile \
  --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
  --batch-size 4 --active-slots 4 --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_mha_repeat \
  --output "$ROOT/text_boundary/result.json" \
  2>&1 | tee "$ROOT/text_boundary/run.log"
BOUNDARY_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$BOUNDARY_EXIT" >"$ROOT/text_boundary/exit_code.txt"
grep 'TEXT_DECODE_BOUNDARY' "$ROOT/text_boundary/run.log" \
  >"$ROOT/text_boundary/events.log" || true
test "$BOUNDARY_EXIT" -eq 0
grep -q '"event": "sync_end"' "$ROOT/text_boundary/events.log"
```

Stop if the graph call or synchronization hangs/errors. Require metadata with
B4, KV4096, position 1279, effective length 1280, TorchAir, and
`attention=mha_repeat`.

### 35.3 Gate B: boundary-containing full page

Dataset offset 11 selects the page used in Phases 29 and 34. It contains the
table crop whose prompt length is 1,021 and whose real generation crosses
effective length 1280.

```sh
run_e2e target_page 1800 11 1
```

Require exit zero, `result_count=1`, `prediction_count=1`, nine recognition
requests, and nine EOS stops. Require the summary configuration to report B4,
KV4096, TorchAir, and `decode_optimization=combined_apply_mha_repeat`. Confirm
that setup metadata points to the B4 cache under `$DECODE_CACHE` and did not
silently select production GQA. Formatting need not be token-identical to the
910B output; the termination and coherent page content are the gate.

### 35.4 Lane C: eight-page E2E performance run

Run only after Gates A-B pass:

```sh
run_e2e eight_pages 3600 0 8
```

Require eight results/predictions and no timeout, Python traceback, CANN error,
AICore error, or non-EOS stop. Report setup separately from measured pipeline
E2E. From `run_summary.json`, report:

- pipeline E2E seconds, pages/s, and seconds/page;
- layout total and detailed stage times;
- recognition requests, input tokens, real/physical vision and text tokens;
- generated tokens, decode graph calls, raw/effective/idle/look-ahead slots;
- decode wall, run-scoped scheduler wall, and effective decode tok/s;
- every device-stage total, especially vision prefill, text prefill, KV
  redistribution, and recognition H2D;
- vision/text packing groups, fill fractions, and bucket histograms;
- stop-reason counts and page completion times;
- graph cache reuse/compile state and setup-time decomposition.

Do not calculate performance from shell timeout duration or include model/setup
time in pages/s. Do not compare the eight-page number directly with a
different page subset.

### 35.5 Report

Write `$ROOT/agent_report.md`:

```text
310P PHASE 35 B4 MHA E2E: EIGHT_PAGE_PASS | TARGET_PAGE_PASS_ONLY |
B4_BOUNDARY_FAILURE | TARGET_PAGE_HANG | EIGHT_PAGE_HANG | RUNTIME_ERROR

commit / host / exact NPU / software:
CANN / driver / firmware:
Gate A command / graph compiled or reused / exit:
Gate A complete boundary event sequence and synchronized elapsed:
Gate B command / exit / result and prediction counts:
Gate B requests / EOS stops / prompt-1021 crop completion:
Gate B E2E / pages-s / decode wall / decode effective tok-s:
Lane C command / exit / page and prediction counts:
Lane C E2E / pages-s / seconds-page:
Lane C layout stage breakdown:
Lane C recognition token totals and decode accounting:
Lane C raw / effective decode tok-s and decode wall:
Lane C vision/text/H2D/KV stage times:
Lane C packing statistics and bucket histograms:
Lane C stop reasons and completion times:
setup-time decomposition and cache compile/reuse:
first Python/CANN/plog error, if any:
mechanical classification:
what is proven:
what is not proven:
evidence paths:
```

Paste back the report and Gate A's complete `events.log`. If the eight-page
lane passes, stop; do not automatically start 32 pages. We will use the
measured eight-page decode rate and page time to decide whether a longer run is
worthwhile.

## Phase 36: static actual-sequence-length GQA, lab throughput and 32-page E2E

### Goal and decision boundary

Phase 30-33 isolated the 310P hang to masked GQA IncreFA at
`cache_position=1279` / effective length 1280. Phase 36 tests the least
invasive workaround: keep the existing BNSD GQA cache and boolean tail mask,
but always pass the compile-time constant
`actual_seq_lengths=[4096] * batch_size` to every IncreFA layer. The constant
is present on every decode step, so this remains one static TorchAir graph and
does not switch graphs at the boundary.

Execute in this order:

1. compare the optimized masked-GQA control and static-actual GQA at safe
   non-1280 positions in the full B32/KV4096 text-decode lab;
2. synchronize one static-actual full-decoder step at position 1279;
3. run the boundary-containing page at dataset offset 11;
4. if all gates pass, run the first 32 pages with all layout preprocessing
   completed before OCR.

Do not run the normal masked-GQA control at position 1279. Do not test PSE or
MHA in this phase. Do not change batch size, KV length, packing, min_pixels,
or frontend scheduling between the two lab profiles and the E2E lane.

The matching 910B checks at commits `4b4eb55` and `05c1441` established:

- B16/KV4096 static-actual synchronized at position 1279;
- at safe positions, full-decoder control was 6,391.6 physical tok/s and
  static-actual was 6,170.3 physical tok/s (3.46% lower);
- the 32-page layout-first static-actual E2E run finished 510/510 recognition
  requests in 20.599 s, with zero token or text mismatches against the normal
  GQA control.

Those are functional controls, not 310P performance thresholds.

### 36.1 Preflight

Pull `main` and require a descendant of `05c1441`. Use the already-established
310P Experiment 09 environment from Phases 33-35. The selected Python must
import `torch_npu` and `kornia_rs`; do not fall back to system Python. Use one
free 310P and terminate only PIDs started by this phase.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/PaddleOCR-VL-1.6}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV3_safetensors}"
DATASET_JSON="${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/datasets/OmniDocBench/images}"
TEXT_LAB="$REPO/09_persistent_page_engine/scripts/text_decode_lab.py"
E2E="$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py"
DECODE_CACHE="$REPO/.runtime_cache/310p_phase36_static_actual_b32"
PACKED_TEXT_CACHE="$REPO/.runtime_cache/310p_text_packed_4789067"
ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase36_static_actual_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -d "$MODEL_DIR"
test -d "$LAYOUT_MODEL"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -f "$TEXT_LAB"
test -f "$E2E"
test ! -e "$ROOT"
mkdir -p "$ROOT/control_profile" "$ROOT/static_profile" \
  "$ROOT/static_boundary" "$ROOT/target_page" "$ROOT/pages32"

"$PYTHON_BIN" - <<'PY'
import kornia_rs
import torch
import torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available())
PY

"$PYTHON_BIN" "$TEXT_LAB" --help \
  | grep -q -- 'combined_apply_static_actual'
"$PYTHON_BIN" "$E2E" --help \
  | grep -q -- 'combined_apply_static_actual'
df -h "$REPO" "$REPO/.runtime_cache"

{
  printf 'commit=%s\n' "$COMMIT"
  hostname
  npu-smi info
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
import torch_npu
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
PY
} >"$ROOT/preflight.log" 2>&1
```

If imports, paths, free space, or NPU availability fail, stop and report. Do
not silently change Python, model, dataset, or device.

### 36.2 Full-decoder safe-position throughput comparison

Both lanes are the complete optimized text decoder plus LM head at B32 and
KV4096. Position 1024 and the 30 timed steps stay far from the failing 1279
boundary. The two optimization names create separate cache entries under one
cache root. Compilation time is setup and must not enter tok/s.

Use this helper so progress is visible and the true exit status is preserved:

```sh
run_profile() {
  lane="$1"
  optimization="$2"
  lane_dir="$ROOT/$lane"

  printf '%q ' timeout --signal=TERM --kill-after=10s 2400 \
    "$PYTHON_BIN" "$TEXT_LAB" \
    --mode profile --backend torchair --allow-compile \
    --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
    --batch-size 32 --active-slots 32 --cache-length 4096 \
    --profile-position 1024 --warmup 3 --repeats 30 \
    --decode-optimization "$optimization" \
    --output "$lane_dir/result.json" >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"

  set +e
  timeout --signal=TERM --kill-after=10s 2400 \
    "$PYTHON_BIN" "$TEXT_LAB" \
    --mode profile --backend torchair --allow-compile \
    --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
    --batch-size 32 --active-slots 32 --cache-length 4096 \
    --profile-position 1024 --warmup 3 --repeats 30 \
    --decode-optimization "$optimization" \
    --output "$lane_dir/result.json" \
    2>&1 | tee "$lane_dir/run.log"
  lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" >"$lane_dir/exit_code.txt"
  return "$lane_exit"
}

run_profile static_profile combined_apply_static_actual
run_profile control_profile combined_apply
```

Require both exits to be zero. Extract from each result:

- mean, median, p95, min, and max full-step latency;
- physical and active tok/s (they are equal with 32 active slots);
- summed device time and model/argmax device time;
- peak allocated-memory delta;
- compile-wrapper and first-call setup time;
- exact resolved TorchAir cache directory and optimization metadata.

Compute `static/control` throughput and latency ratios. Do not use host wall
time as the throughput denominator. Record whether each graph compiled or
reused; do not describe compilation time as inference latency.

### 36.3 Static-actual boundary gate

This must reuse the B32 static-actual graph from 36.2. Omit `--allow-compile`:

```sh
printf '%q ' timeout --signal=TERM --kill-after=10s 300 \
  "$PYTHON_BIN" "$TEXT_LAB" \
  --mode boundary --backend torchair \
  --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
  --batch-size 32 --active-slots 32 --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_static_actual \
  --output "$ROOT/static_boundary/result.json" \
  >"$ROOT/static_boundary/command.sh"
printf '\n' >>"$ROOT/static_boundary/command.sh"

set +e
timeout --signal=TERM --kill-after=10s 300 \
  "$PYTHON_BIN" "$TEXT_LAB" \
  --mode boundary --backend torchair \
  --model "$MODEL_DIR" --cache-dir "$DECODE_CACHE" \
  --batch-size 32 --active-slots 32 --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_static_actual \
  --output "$ROOT/static_boundary/result.json" \
  2>&1 | tee "$ROOT/static_boundary/run.log"
BOUNDARY_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$BOUNDARY_EXIT" >"$ROOT/static_boundary/exit_code.txt"
grep 'TEXT_DECODE_BOUNDARY' "$ROOT/static_boundary/run.log" \
  >"$ROOT/static_boundary/events.log" || true
```

Require exit zero and the complete sequence `step_begin`, `step_returned`,
`sync_begin`, `sync_end`. The result must say B32, KV4096, position 1279,
effective length 1280, `attention=gqa`, TorchAir, and
`combined_apply_static_actual`. If it hangs, times out, or reports a CANN/AICore
error, stop. Do not run E2E.

### 36.4 E2E helper and boundary-containing page

The only behavior change from the prior optimized GQA pipeline is
`combined_apply_static_actual`. Layout is fully completed before recognition.
The B32 decode cache from the lab must be reused.

```sh
run_e2e() {
  lane="$1"
  timeout_s="$2"
  offset="$3"
  limit="$4"
  lane_dir="$ROOT/$lane"
  output_dir="$lane_dir/output"

  printf '%q ' timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset "$offset" --limit "$limit" \
    --batch-size 32 --cache-length 4096 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_static_actual \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-promptfa-align-128 --vision-padding bucket \
    --vision-packing greedy --vision-pack-target 1920 \
    --vision-router-lookahead 32 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --text-packed-cache-dir "$PACKED_TEXT_CACHE" \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output_dir" >"$lane_dir/command.sh"
  printf '\n' >>"$lane_dir/command.sh"

  set +e
  timeout --signal=TERM --kill-after=10s "$timeout_s" \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset "$offset" --limit "$limit" \
    --batch-size 32 --cache-length 4096 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_static_actual \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-promptfa-align-128 --vision-padding bucket \
    --vision-packing greedy --vision-pack-target 1920 \
    --vision-router-lookahead 32 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --text-packed-cache-dir "$PACKED_TEXT_CACHE" \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output_dir" \
    2>&1 | tee "$lane_dir/run.log"
  lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" >"$lane_dir/exit_code.txt"
  return "$lane_exit"
}

run_e2e target_page 1800 11 1
```

Require one result and prediction, all recognition requests EOS-terminated,
and no Python, CANN, AICore, or timeout error. Confirm the summary says B32,
KV4096, TorchAir, static-actual optimization, and all-before-recognition.
Confirm at least one request crosses effective length 1280 by calculating
`input_tokens + decode_tokens_after_prefill_including_eos` from the trace.

### 36.5 First 32 pages, layout first

Run only after 36.2-36.4 pass:

```sh
run_e2e pages32 7200 0 32
```

Require 32 results and predictions, no timeout/error, and EOS for every
recognition request. This subset includes the crop that deterministically hung
normal masked GQA at position 1279. Report progress timestamps from `run.log`
and explicitly state whether the run passed page 12 and completed page 32.

From `run_summary.json`, report:

- measured pipeline E2E, pages/s, and seconds/page (setup excluded);
- complete setup decomposition, especially decode first-call/cache time;
- layout total and detailed stage times;
- recognition requests and all real/physical vision/text/decode token totals;
- decode graph calls, raw/effective/idle/look-ahead token slots;
- decode wall and raw/effective decode tok/s;
- run-scoped scheduler wall;
- every device-stage total, especially vision prefill, text prefill, H2D, KV
  redistribution, and decode;
- vision/text packing groups, fill fractions, and bucket histograms;
- stop-reason counts and page completion times;
- number of requests whose generated path crossed effective lengths 1280,
  2560, or 3840.

Do not include model/setup/compile time in pages/s. Do not infer decode tok/s
from E2E; use the recorded decode wall and token counters. Do not attribute
speed differences to static actual length unless the safe-position lab profile
supports them.

### 36.6 Report

Write `$ROOT/agent_report.md`:

```text
310P PHASE 36 STATIC ACTUAL GQA: FULL_32_PASS | TARGET_PAGE_PASS_ONLY |
BOUNDARY_FAILURE | PROFILE_FAILURE | TARGET_PAGE_FAILURE | FULL_32_HANG |
RUNTIME_ERROR

commit / host / exact NPU / software:
CANN / driver / firmware:
control profile command / compiled or reused / exit:
static profile command / compiled or reused / exit:
control B32 safe-position latency distribution / physical tok-s / memory:
static B32 safe-position latency distribution / physical tok-s / memory:
static-vs-control latency and throughput ratios:
static boundary command / cache reuse / exit:
static boundary complete event sequence and synchronized elapsed:
target-page command / exit / results / requests / EOS:
target-page 1280-crossing evidence / E2E / decode wall / tok-s:
32-page command / exit / results / requests / EOS:
32-page E2E / pages-s / seconds-page:
32-page layout stage breakdown:
32-page token totals and decode accounting:
32-page raw / effective decode tok-s and decode wall:
32-page vision/text/H2D/KV device-stage times:
32-page packing statistics and bucket histograms:
32-page boundary-crossing request counts:
setup decomposition and graph cache evidence:
first Python/CANN/plog error, if any:
mechanical classification:
what is proven:
what is not proven:
evidence paths:
```

Paste back the complete report, both profile result summaries, the boundary
`events.log`, and the last 80 lines of the 32-page log. Stop after Phase 36;
do not promote static actual length to the default production preset yet.

## Phase 37: checkpointed 310P OmniDocBench prefix reference

### Goal and execution rule

Measure the optimized 310P pipeline and official OmniDocBench metrics on the
same cumulative prefixes that were measured on one Ascend 910B2:
`0:32`, `0:64`, `0:128`, and `0:256`.

This phase is deliberately checkpointed. Execute **one prefix only**, run its
evaluation, write and paste its report, then stop. Do not start the next prefix
until Luka explicitly asks you to continue. The order is 32, 64, 128, 256.
Each checkpoint has one measured E2E run and one evaluation. Do not queue all
four runs in one shell loop and do not defer reporting until the end.

The committed 910B reference is:

```text
tmp/09_persistent_page_engine/
  910b_static_actual_reference_prefixes_3a9244b/
    REFERENCE.md
    reference_summary.json
```

`reference_summary.json` is the comparison authority. It contains both 910B
timing repeats, exact commands/configuration, layout and recognition stage
times, token accounting, packing statistics, output hashes, and evaluation
metrics. The short headline is:

| pages | 910B E2E mean s (range) | 910B pages/s mean (range) |
|---:|---:|---:|
| 32 | 19.481 (19.388-19.573) | 1.6427 (1.6349-1.6505) |
| 64 | 45.747 (45.141-46.352) | 1.3993 (1.3807-1.4178) |
| 128 | 71.987 (71.911-72.063) | 1.7781 (1.7762-1.7800) |
| 256 | 125.214 (124.579-125.849) | 2.0446 (2.0342-2.0549) |

The 910B runs used physical NPU 5, CANN 9.0.0, project commit `3a9244b`,
layout-first execution, B32/KV4096 static-actual GQA, `min_pixels=28224`,
compiled PromptFA vision, greedy vision packing with target 1920/lookahead 32,
and production-group text packing. Both timing repeats had exact recognition
semantics and byte-identical prediction Markdown at every prefix.

The evaluator is `opendatalab/OmniDocBench` commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`, using `quick_match` with 12
workers. CDM is intentionally skipped. Therefore there is no leaderboard
Overall score; compare text/formula/table/reading-order metrics individually.
Lower Edit distance is better and higher TEDS is better.

### 37.1 Preconditions and preflight

Require Phase 36 to have passed the B32 static-actual boundary, target page,
and first-32-pages E2E gates. If Phase 36 did not reach `FULL_32_PASS`, stop and
report; do not hide that failure by starting this phase.

Pull `main` and require commit `153866a` or a descendant. Continue using the
same verified Python, model, layout model, dataset, NPU selection, and warmed
TorchAir caches from Phase 36. Do not install or upgrade torch, torch_npu,
TorchAir, Transformers, OpenCV, or model dependencies for this phase.

```sh
cd "$(git rev-parse --show-toplevel)"
git status --short --branch
git pull --ff-only origin main

REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/PaddleOCR-VL-1.6}"
LAYOUT_MODEL="${LAYOUT_MODEL:-/workspace/models/PP-DocLayoutV3_safetensors}"
DATASET_JSON="${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/datasets/OmniDocBench/images}"
E2E="$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py"
REFERENCE="$REPO/tmp/09_persistent_page_engine/910b_static_actual_reference_prefixes_3a9244b/reference_summary.json"
DECODE_CACHE="$REPO/.runtime_cache/310p_phase36_static_actual_b32"
VISION_CACHE="$REPO/.runtime_cache/09_persistent_page_engine_vision_torchair"
VISION_BATCHED_CACHE="$REPO/.runtime_cache/09_vision_router_batched"
TEXT_CACHE="$REPO/.runtime_cache/09_persistent_page_engine_text_torchair"
PACKED_TEXT_CACHE="$REPO/.runtime_cache/310p_text_packed_4789067"
ROOT="$REPO/tmp/09_persistent_page_engine/310p_phase37_prefixes_$COMMIT_SHORT"
CACHE_ROOTS=(
  "$DECODE_CACHE"
  "$VISION_CACHE"
  "$VISION_BATCHED_CACHE"
  "$TEXT_CACHE"
  "$PACKED_TEXT_CACHE"
)

test -x "$PYTHON_BIN"
test -d "$MODEL_DIR"
test -d "$LAYOUT_MODEL"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -f "$E2E"
test -f "$REFERENCE"
test -d "$DECODE_CACHE"
test -d "$VISION_CACHE"
test -d "$VISION_BATCHED_CACHE"
test -d "$TEXT_CACHE"
test -d "$PACKED_TEXT_CACHE"
mkdir -p "$ROOT"

git merge-base --is-ancestor 153866a HEAD
"$PYTHON_BIN" - <<'PY'
import json
import kornia_rs
import torch
import torch_npu
from pathlib import Path

reference = Path(
    "tmp/09_persistent_page_engine/"
    "910b_static_actual_reference_prefixes_3a9244b/"
    "reference_summary.json"
)
data = json.loads(reference.read_text())
assert data["project_commit"] == "3a9244b"
assert set(data["prefixes"]) == {"32", "64", "128", "256"}
assert data["evaluator"]["cdm"] == "skipped_by_request"
assert torch.npu.is_available()
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("reference", reference, "OK")
PY
```

Resolve an existing official evaluator checkout. Do not substitute the old
lightweight experiment-06 metrics. Do not mutate the main project checkout to
install evaluation dependencies.

```sh
if test -n "${OMNIDOCBENCH_EVAL_REPO:-}"; then
  EVAL_REPO="$OMNIDOCBENCH_EVAL_REPO"
else
  EVAL_REPO=""
  for candidate in \
    "$REPO/../OmniDocBench_eval" \
    "$REPO/../OmniDocBench" \
    /workspace/repos/OmniDocBench_eval \
    /workspace/repos/OmniDocBench
  do
    if test -f "$candidate/pdf_validation.py"; then
      EVAL_REPO="$candidate"
      break
    fi
  done
fi

test -n "$EVAL_REPO"
test -f "$EVAL_REPO/pdf_validation.py"
test "$(git -C "$EVAL_REPO" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

if test -n "${OMNIDOCBENCH_EVAL_PYTHON:-}"; then
  EVAL_PY="$OMNIDOCBENCH_EVAL_PYTHON"
elif test -x /workspace/venvs/omnidocbench_py310/bin/python; then
  EVAL_PY=/workspace/venvs/omnidocbench_py310/bin/python
elif test -x "$EVAL_REPO/.venv/bin/python"; then
  EVAL_PY="$EVAL_REPO/.venv/bin/python"
else
  printf '%s\n' "No verified OmniDocBench evaluator Python found."
  exit 1
fi

"$EVAL_PY" - <<'PY'
import Levenshtein
import apted
import bs4
import lxml
import numpy
import pandas
import yaml
print("official evaluator imports: PASS")
PY
```

If the evaluator checkout, exact commit, or evaluator Python is missing, stop
and report `EVALUATOR_PREFLIGHT_MISSING` with the paths/checks that failed. Do
not improvise a different evaluator during a measured checkpoint. The dataset
directory is not expected to contain `pdf_validation.py`; the evaluator is a
separate checkout of `opendatalab/OmniDocBench`.

#### 37.1A Missing evaluator acquisition checkpoint

Luka has now authorized this checkpoint when the evaluator code is absent.
Run it by itself, report the result, and stop before any measured prefix run.
Do not place evaluator source or its environment inside the PaddleOCR project.

Fetch only the exact evaluator revision used by the 910B references:

```sh
EVAL_REPO=/workspace/repos/OmniDocBench_eval
EVAL_COMMIT=2b161d010d2e3aff77a0edef359ea3a6411d23cd

if test -e "$EVAL_REPO"; then
  test -d "$EVAL_REPO/.git"
else
  mkdir -p "$(dirname "$EVAL_REPO")"
  git init "$EVAL_REPO"
  git -C "$EVAL_REPO" remote add origin \
    https://github.com/opendatalab/OmniDocBench.git
fi

git -C "$EVAL_REPO" fetch --depth=1 origin "$EVAL_COMMIT"
git -C "$EVAL_REPO" checkout --detach FETCH_HEAD
test "$(git -C "$EVAL_REPO" rev-parse HEAD)" = "$EVAL_COMMIT"
test -f "$EVAL_REPO/pdf_validation.py"
test -f "$EVAL_REPO/pyproject.toml"
du -sh "$EVAL_REPO"
```

The official evaluator declares Python `>=3.10,<3.12`; do not use the
Experiment-09 Python 3.12 environment for it. First look for an already usable
evaluator interpreter:

```sh
EVAL_PY=""
for candidate in \
  /workspace/venvs/omnidocbench_py310/bin/python \
  /workspace/venvs/omnidocbench/bin/python \
  "$EVAL_REPO/.venv/bin/python"
do
  if test -x "$candidate"; then
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
assert (3, 10) <= sys.version_info[:2] < (3, 12)
import Levenshtein, apted, bs4, lxml, numpy, pandas, yaml
PY
    then
      EVAL_PY="$candidate"
      break
    fi
  fi
done
```

If `EVAL_PY` is still empty, locate a base Python 3.10 or 3.11 interpreter.
Do not guess that `python` is suitable, and do not bypass the evaluator's
Python constraint:

```sh
if test -z "$EVAL_PY"; then
  BASE_PY=""
  for candidate in \
    /usr/local/python3.10.13/bin/python3 \
    /usr/local/bin/python3.10 \
    /usr/bin/python3.10 \
    /usr/local/bin/python3.11 \
    /usr/bin/python3.11
  do
    if test -x "$candidate"; then
      BASE_PY="$candidate"
      break
    fi
  done
  if test -z "$BASE_PY"; then
    printf '%s\n' "EVALUATOR_PYTHON_310_OR_311_MISSING"
    exit 1
  fi

  EVAL_VENV=/workspace/venvs/omnidocbench_py310
  "$BASE_PY" -m venv "$EVAL_VENV"
  "$EVAL_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$EVAL_VENV/bin/python" -m pip install "$EVAL_REPO"
  EVAL_PY="$EVAL_VENV/bin/python"
fi

"$EVAL_PY" - <<'PY'
import sys
import Levenshtein
import apted
import bs4
import lxml
import numpy
import pandas
import yaml
assert (3, 10) <= sys.version_info[:2] < (3, 12)
print("evaluator_python", sys.executable)
print("evaluator_version", sys.version)
print("official evaluator imports: PASS")
PY
```

Report:

- `git -C "$EVAL_REPO" rev-parse HEAD`;
- the absolute `pdf_validation.py` path;
- `EVAL_PY` and its Python version;
- whether an existing environment was reused or a new one was created;
- any clone, package-download, proxy, or wheel-build failure verbatim.

Success marker: `PHASE37_EVALUATOR_SETUP: PASS`. After reporting it, wait for
Luka before running the first prefix checkpoint.

Record the environment once:

```sh
{
  printf 'project_commit=%s\n' "$COMMIT"
  printf 'evaluator_commit=%s\n' "$(git -C "$EVAL_REPO" rev-parse HEAD)"
  printf 'project_python=%s\n' "$PYTHON_BIN"
  printf 'evaluator_python=%s\n' "$EVAL_PY"
  hostname
  npu-smi info
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
import torch_npu
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
PY
} >"$ROOT/preflight.log" 2>&1
```

### 37.2 One-prefix E2E helper

The helper below runs exactly one prefix. Do not wrap it in a loop. It keeps
layout fully ahead of OCR and matches the committed 910B configuration. Model
setup and cache loading are excluded by `pipeline_e2e_s`; on-demand graph
compilation during the pipeline is not excluded and must be reported.

```sh
run_prefix() {
  pages="$1"
  lane="$ROOT/n${pages}"
  output="$lane/output"
  test ! -e "$lane"
  mkdir -p "$lane"

  du -sb "${CACHE_ROOTS[@]}" \
    >"$lane/cache_sizes_before.txt" 2>&1 || true
  find "${CACHE_ROOTS[@]}" -type f -printf '%T@ %s %p\n' \
    | sort >"$lane/cache_files_before.txt" 2>/dev/null || true

  printf '%q ' timeout --signal=TERM --kill-after=15s 7200 \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset 0 --limit "$pages" \
    --batch-size 32 --cache-length 4096 --max-new-tokens 2808 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_static_actual \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-torchair-cache-dir "$VISION_CACHE" \
    --vision-batched-cache-dir "$VISION_BATCHED_CACHE" \
    --vision-promptfa-align-128 --vision-padding bucket \
    --vision-packing greedy --vision-pack-target 1920 \
    --vision-router-lookahead 32 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --text-torchair-cache-dir "$TEXT_CACHE" \
    --text-packed-cache-dir "$PACKED_TEXT_CACHE" \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output" >"$lane/command.sh"
  printf '\n' >>"$lane/command.sh"

  set +e
  timeout --signal=TERM --kill-after=15s 7200 \
    "$PYTHON_BIN" "$E2E" \
    --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR" \
    --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL_DIR" \
    --offset 0 --limit "$pages" \
    --batch-size 32 --cache-length 4096 --max-new-tokens 2808 \
    --preprocessor-min-pixels 28224 \
    --decode-backend torchair \
    --decode-optimization combined_apply_static_actual \
    --torchair-cache-dir "$DECODE_CACHE" \
    --vision-backend torchair \
    --vision-attention prompt_flash_attention \
    --vision-torchair-cache-dir "$VISION_CACHE" \
    --vision-batched-cache-dir "$VISION_BATCHED_CACHE" \
    --vision-promptfa-align-128 --vision-padding bucket \
    --vision-packing greedy --vision-pack-target 1920 \
    --vision-router-lookahead 32 \
    --text-packing production_group \
    --text-pack-buckets 128,256,512,1024 \
    --text-pack-max-members 32 \
    --text-torchair-cache-dir "$TEXT_CACHE" \
    --text-packed-cache-dir "$PACKED_TEXT_CACHE" \
    --layout-device npu --no-layout-graph-capture \
    --preprocess-all-pages-first --no-timeline \
    --output-dir "$output" 2>&1 | tee "$lane/run.log"
  lane_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$lane_exit" >"$lane/exit_code.txt"

  du -sb "${CACHE_ROOTS[@]}" \
    >"$lane/cache_sizes_after.txt" 2>&1 || true
  find "${CACHE_ROOTS[@]}" -type f -printf '%T@ %s %p\n' \
    | sort >"$lane/cache_files_after.txt" 2>/dev/null || true
  diff -u "$lane/cache_files_before.txt" "$lane/cache_files_after.txt" \
    >"$lane/cache_files.diff" || true
  return "$lane_exit"
}
```

For the 32-page checkpoint only, a successful Phase-36 `pages32/output` may be
used as its one measured run instead of rerunning, but only if its
`run_summary.json` says all of the following: offset 0, count 32, B32, KV4096,
`combined_apply_static_actual`, min_pixels 28224, PromptFA with align-128,
vision target 1920/lookahead 32, production-group text packing, layout NPU
eager, and `all_before_recognition`. Copy the complete Phase-36 `pages32`
artifact directory to `$ROOT/n32` and record the source path in
`$ROOT/n32/reused_phase36_path.txt`. Otherwise run:

```sh
run_prefix 32
```

For later checkpoints, execute exactly one of these only after Luka asks:

```sh
run_prefix 64
# OR, in a later turn:
run_prefix 128
# OR, in a later turn:
run_prefix 256
```

Require exit zero, exactly `pages` results and predictions, no `length` stops,
stop reasons limited to `eos` and `kv_cache_full`, no Python/CANN/AICore error,
and summary configuration matching the command. Omitting `--max-new-tokens`
is intentional: the committed default equals KV4096, and the scheduler admits
every fitting prompt before stopping that request at EOS or its exact KV
boundary. If cache files changed or logs show compilation during the measured
pipeline, label timing `COMPILE_CONTAMINATED`; still evaluate the completed
outputs, report the contamination, and stop for a decision rather than
silently rerunning.

### 37.3 Official evaluation for that prefix, without CDM

Run this only after the current prefix E2E succeeds. It writes evaluator
results inside the prefix artifact directory, not into the evaluator checkout.

```sh
evaluate_prefix() {
  pages="$1"
  lane="$ROOT/n${pages}"
  output="$lane/output"
  eval_root="$lane/evaluation"
  work="$eval_root/work"
  test -f "$output/run_summary.json"
  test -f "$output/OmniDocBench_subset.json"
  test -d "$output/predictions"
  mkdir -p "$work"

  "$EVAL_PY" - "$output" "$work" <<'PY'
from pathlib import Path
import sys

output = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
config = f"""end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 12
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {output / 'OmniDocBench_subset.json'}
    prediction:
      data_path: {output / 'predictions'}
    match_method: quick_match
    match_workers: 12
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
(work / "config.yaml").write_text(config)
PY

  printf '%q ' timeout --signal=TERM --kill-after=15s 3600 \
    "$EVAL_PY" "$EVAL_REPO/pdf_validation.py" --config config.yaml \
    >"$eval_root/command.sh"
  printf '\n' >>"$eval_root/command.sh"

  set +e
  (
    cd "$work"
    PYTHONPATH="$EVAL_REPO" \
      timeout --signal=TERM --kill-after=15s 3600 \
      "$EVAL_PY" "$EVAL_REPO/pdf_validation.py" --config config.yaml
  ) 2>&1 | tee "$eval_root/evaluation.log"
  eval_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$eval_exit" >"$eval_root/exit_code.txt"
  return "$eval_exit"
}
```

For the current checkpoint only:

```sh
evaluate_prefix 32   # replace 32 with the one prefix Luka authorized
```

Require exit zero. Read:

```text
$ROOT/n<PAGES>/evaluation/work/result/
  predictions_quick_match_metric_result.json
  predictions_quick_match_run_summary.json
```

Require page-match `page_timeout=0`, `quick_match_timeout=0`, and TEDS error,
exception, and timeout counts all zero. CDM must remain absent/null. Do not
invent an Overall score from the remaining metrics.

### 37.4 Per-checkpoint analysis and report

For the current prefix, compare the 310P run to
`reference_summary.json["prefixes"][str(PAGES)]`. Use the 910B timing mean and
range for E2E comparison; use its repeat-1 evaluation metrics because those are
the committed official evaluation outputs.

Write both `$ROOT/n<PAGES>/checkpoint_report.json` and
`$ROOT/n<PAGES>/agent_report.md`. Report:

- exact project/evaluator commits, host, NPU, CANN/driver/firmware, Python,
  torch, and torch_npu;
- exact E2E and evaluation commands and all artifact paths;
- pipeline E2E, pages/s, seconds/page, and the ratio to 910B mean pages/s;
- setup separately, including compile first-call/cache evidence;
- complete layout total and stage breakdown, layout pages/s, and ratios to the
  corresponding 910B values;
- requests, input tokens, projected image tokens, real/physical vision tokens,
  real/physical text tokens, generated/effective/raw decode tokens;
- vision prefill seconds and real/physical tok/s; text prefill seconds and
  real/physical tok/s; decode wall and raw/effective tok/s;
- every device-stage total, packing group counts/fill fractions/histograms,
  decode graph calls, active/idle/lookahead slots, stop reasons, and output
  counts;
- official text-block Edit distance, display-formula Edit distance, table
  TEDS/TEDS-structure/table Edit distance, and reading-order Edit distance;
- signed metric deltas versus 910B (310P minus 910B), with the reminder that
  lower Edit distance and higher TEDS are better;
- evaluator fallback/error/timeout counts;
- whether timing was cache-warm or compile-contaminated;
- what is proven, what is not proven, and the first error if anything failed.

Use this first line:

```text
310P PHASE 37 N<PAGES>: PASS | COMPILE_CONTAMINATED | E2E_FAILURE |
EVALUATOR_FAILURE | EVALUATOR_PREFLIGHT_MISSING | RUNTIME_ERROR
```

After writing the report, paste it back immediately together with:

1. the current prefix's `checkpoint_report.json`;
2. the final 80 lines of its E2E log;
3. the evaluator headline metrics and fallback/timeout counts;
4. the cache diff or a statement that it was empty.

Then **stop**. Do not start the next prefix until Luka explicitly says to
continue.

---

## Phase 38: exact 32-page 310P versus 910B accuracy localization

### Goal and decision boundary

The Phase-37 headline metrics only say that the final pages differ. This phase
must locate the first boundary at which they differ:

1. layout geometry;
2. recorded crop/request metadata;
3. the prefill-produced first generated token;
4. a later token after an initially shared generation prefix;
5. final assembled page Markdown.

Do not run another model variant, attention lane, eager lane, evaluator, or
larger prefix in this phase. First produce the exact comparison. The result of
that comparison decides the next experiment.

The committed 910B reference is the ten-vision-bucket B32/KV4096
`combined_apply_static_actual` run. On 910B, this reference was already proven
token-identical across all 510 crops to both:

- the same ten-bucket run with normal `combined_apply` GQA; and
- the evaluated forty-bucket static-actual n32 reference.

Therefore the ten-bucket restriction and static actual are already excluded as
causes of any 310P token difference.

### 38.1 Pull, preflight, and paths

```sh
cd /workspace/repos/paddle_ocr_vl_npu
git pull --ff-only origin main
git status --short
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
COMPARE=09_persistent_page_engine/scripts/compare_e2e_outputs.py
REFERENCE=tmp/09_persistent_page_engine/910b_accuracy_reference_n32_05c1441/output
PHASE37=tmp/09_persistent_page_engine/310p_phase37_prefixes_47b5d56/n32/output
ROOT=tmp/09_persistent_page_engine/310p_phase38_accuracy_1a3d8b7

test -f "$REFERENCE/run_summary.json"
test -f "$REFERENCE/recognition_trace.jsonl"
test -f "$REFERENCE/page_regions.jsonl"
test -d "$REFERENCE/predictions"
test -f "$PHASE37/run_summary.json"
test -f "$PHASE37/recognition_trace.jsonl"
test -f "$PHASE37/page_regions.jsonl"
test -d "$PHASE37/predictions"
test ! -e "$ROOT"
mkdir -p "$ROOT"
```

Record the exact project commit and confirm the reference summary says:

- offset 0, count 32, 510 recognition requests;
- B32, KV4096, min_pixels 28224;
- `combined_apply_static_actual`;
- PromptFA, greedy vision packing, target 1920, lookahead 32;
- vision buckets exactly
  `128,256,384,512,640,768,1408,1920,2944,4992`;
- production-group text packing with buckets `128,256,512,1024`;
- all-page preprocessing before recognition.

The historical reference has `max_new_tokens=2808`; that is acceptable only
because all 510 reference crops stopped at EOS. The fresh 310P run below must
use the new EOS-or-KV-full policy and report `max_new_tokens=4096`.

### 38.2 Immediately compare the existing Phase-37 n32 output

This is CPU-only and should finish quickly. It establishes the differences
before spending time on another E2E run.

```sh
"$PYTHON_BIN" "$COMPARE" \
  --reference-output "$REFERENCE" \
  --candidate-output "$PHASE37" \
  --output-dir "$ROOT/existing_phase37_comparison" \
  --worst-limit 30 \
  2>&1 | tee "$ROOT/existing_phase37_compare.log"
```

Do not summarize this as merely pass/fail. Preserve `comparison.json` and
`comparison.md`. Read and report at least:

- whether all 32 layout geometries are exact;
- whether request order and all 510 recorded request-metadata contracts are
  exact;
- exact-token crop count and fraction;
- first-generated-token divergence count;
- divergence-after-shared-prefix count;
- length-only divergence count;
- candidate-minus-reference output tokens;
- all label rows;
- all pages with any divergent crops, sorted by divergent-crop count;
- the worst 30 crop rows and their token/text similarity;
- every stop-reason transition.

### 38.3 Fresh 32-page run with EOS-or-KV-full generation

Use only the already-compiled ten vision buckets. Omitting
`--max-new-tokens` is deliberate. Do not add `2808` back.

```sh
LANE="$ROOT/fresh_n32"
OUTPUT="$LANE/output"
mkdir -p "$LANE"

printf '%q ' timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 32 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --output-dir "$OUTPUT" >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

set +e
timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 32 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --output-dir "$OUTPUT" 2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
test "$run_exit" -eq 0
```

Require `run_summary.json` to report `max_new_tokens=4096`, 32/32 results,
510 requests, no `length` stops, and stop reasons limited to `eos` and
`kv_cache_full`. Cache loading may take time, but no new graph compilation or
cache regeneration is allowed in the measured run.

### 38.4 Compare fresh 310P to 910B and to old 310P

```sh
"$PYTHON_BIN" "$COMPARE" \
  --reference-output "$REFERENCE" \
  --candidate-output "$OUTPUT" \
  --output-dir "$ROOT/fresh_vs_910b" \
  --worst-limit 30 \
  2>&1 | tee "$ROOT/fresh_vs_910b.log"

"$PYTHON_BIN" "$COMPARE" \
  --reference-output "$PHASE37" \
  --candidate-output "$OUTPUT" \
  --output-dir "$ROOT/fresh_vs_old_310p" \
  --worst-limit 30 \
  2>&1 | tee "$ROOT/fresh_vs_old_310p.log"
```

The fresh-versus-old comparison determines whether the generation-policy
change affected the 32-page output. Because all historical n32 requests ended
at EOS, the expected result is 510/510 exact token streams. If it is not exact,
do not attribute that automatically to `max_new_tokens`; report the first
divergences and stop.

### 38.5 Report and stop

Write `$ROOT/agent_report.md` with these sections:

1. exact commits, software, host, NPU, and commands;
2. reference contract verification;
3. existing Phase-37 versus 910B boundary summary;
4. fresh-run configuration and stop-reason verification;
5. fresh 310P versus 910B boundary summary;
6. fresh versus old 310P determinism summary;
7. differences by label and page;
8. worst 30 crop divergences;
9. what is proven about layout, recorded request metadata, token zero, later
   generation, and assembled pages; explicitly note that the historical trace
   has no crop-pixel hash, so geometry parity does not prove byte-identical
   crop pixels;
10. what remains unresolved and the single narrow next experiment suggested by
    the observed first-divergence distribution.

Paste the full report, all three generated `comparison.md` files, and the
headline `recognition` and `layout` objects from each `comparison.json`. Then
**stop**. Do not start that next experiment yet.

---

## Phase 39: fixed seven-page input-identity and execution localization

### Goal and hard boundary

Phase 38 proved a deterministic cross-device output difference but did not
record crop bytes or final CPU model inputs.  Phase 39 reruns only original
OmniDocBench pages 8 through 14 (106 historical recognition requests), with a
fixed ten-case diagnostic manifest and exact pre-H2D fingerprints.

This phase answers one question:

> For divergent crops, are the raw crop pixels, prepared CPU tensors, and
> vision/text execution routes actually identical between 310P and 910B?

Do not run the evaluator, a larger prefix, eager variants, new graph shapes,
or per-layer tensor dumps.  The fixed-corpus classification decides whether
the next phase belongs in the frontend or inside model execution.

### 39.1 Pull and verify the committed 910B reference

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
LAB=09_persistent_page_engine/scripts/accuracy_lab.py
CASES=09_persistent_page_engine/accuracy_lab/cases.json
REFERENCE=tmp/09_persistent_page_engine/910b_accuracy_lab_7p_8e19fdc/output
ROOT=tmp/09_persistent_page_engine/310p_phase39_accuracy_lab_8e19fdc

test -f "$CASES"
test -f "$REFERENCE/run_summary.json"
test -f "$REFERENCE/recognition_trace.jsonl"
test -f "$REFERENCE/page_regions.jsonl"
test -d "$REFERENCE/predictions"
test ! -e "$ROOT"
mkdir -p "$ROOT"
```

If the committed reference does not exist, stop immediately and report
`REFERENCE_NOT_COMMITTED`.  Do not substitute the old 32-page Phase-38
reference: it lacks input fingerprints.

Before running, inspect the reference and require:

- offset 8, count 7, and the exact seven image names in `cases.json`;
- B32, KV4096, min_pixels 28224, static-actual GQA;
- the exact ten vision buckets from Phase 38;
- the explicit 21-value text bucket ladder in the command below;
- production-group text packing at 128/256/512/1024;
- `page_preprocessing_mode=all_before_recognition`;
- `recognition_input_fingerprints=true`;
- only EOS or KV-full stop reasons.

### 39.2 Run the matching 310P corpus

CPU hashing does not change any compiled graph.  Reuse the already validated
310P graph caches; no new compilation or regeneration is allowed.

```sh
LANE="$ROOT/310p_e2e"
OUTPUT="$LANE/output"
mkdir -p "$LANE"

printf '%q ' timeout --signal=TERM --kill-after=15s 900 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 8 --limit 7 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --recognition-input-fingerprints \
  --output-dir "$OUTPUT" >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

set +e
timeout --signal=TERM --kill-after=15s 900 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 8 --limit 7 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --recognition-input-fingerprints \
  --output-dir "$OUTPUT" 2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
test "$run_exit" -eq 0
```

Path adaptations are allowed only for dataset/model roots.  Record them.  Do
not alter any model, bucket, packing, frontend, generation, or fingerprint
setting.

### 39.3 Run the strict fixed-corpus comparison

```sh
"$PYTHON_BIN" "$LAB" \
  --cases "$CASES" \
  --reference-output "$REFERENCE" \
  --candidate-output "$OUTPUT" \
  --output-dir "$ROOT/comparison" \
  2>&1 | tee "$ROOT/comparison.log"
```

Do not pass `--allow-missing-fingerprints`.  A missing crop or prepared-input
hash is a failed instrumentation contract, not an inconclusive result.

### 39.4 Report and stop

Write `$ROOT/agent_report.md`.  Begin with exactly one classification:

```text
310P PHASE 39: MODEL_EXECUTION_DIFFERENCE_PROVEN |
ALL_DIVERGENCES_HAVE_DIFFERENT_PREPARED_INPUTS |
MIXED_INPUT_AND_ROUTE_EVIDENCE | FIXED_CORPUS_TOKEN_EXACT |
INPUT_IDENTITY_UNRESOLVED | E2E_FAILURE | REFERENCE_NOT_COMMITTED
```

Then include:

1. exact commit, host/NPU/software, expanded E2E command, and cache-hit evidence;
2. reference and candidate request/page counts and stop reasons;
3. complete fixed-corpus boundary summary;
4. exact/different/unavailable cross-tabs for crop hashes, prepared-input
   hashes, recorded request metadata, vision routes, and text-prefill routes;
5. the complete ten-case table from `report.md`;
6. for every divergent selected case, both original request IDs, first
   divergence index, full route signatures, crop hash, combined prepared hash,
   and all six individual tensor-hash statuses;
7. whether the two control candidates are actually cross-device token exact;
8. contract warnings, what is proven, and what remains unresolved.

Paste the complete `agent_report.md`, `comparison/report.md`, the top-level
`classification` object, all evidence cross-tabs, and the ten selected-case
objects from `comparison/report.json`.  Then **stop**.  Do not start per-layer
profiling or modify any source.

---

## Phase 40: 128-page manual degeneration incidence survey

### Goal and interpretation boundary

Run the first 128 OmniDocBench pages on 310P with the exact current production
configuration, compare all recognition crops against the committed matched
910B reference, and build a high-recall manual-review set.

This phase is not an accuracy evaluation and token non-parity is not an error.
LaTeX spelling, harmless markup, and semantically equivalent output can differ
between devices.  The goal is to count genuinely degenerate generations such
as runaway repetition, script corruption, gross omissions, or unrelated text,
then describe where those confirmed cases concentrate.  Do not change model
code, graph code, cache policy, or generation policy in this phase.

The 910B reference was produced by behavior commit `491de50` and committed as
artifacts in `96654bb`.  Its fixed contract is:

- offset 0, count 128, 128 results, 2,082 recognition requests;
- B32, KV4096, min_pixels 28224, static-actual GQA decode;
- layout completed before recognition;
- ten explicit vision buckets and the 21-value text bucket ladder below;
- PromptFA, greedy vision packing at 1920, production-group text packing;
- input fingerprints plus private-cache and decode-slot reuse metadata;
- stop reasons `{eos: 2081, kv_cache_full: 1}`;
- pipeline E2E 78.651 s on 910B2 (performance context only).

### 40.1 Pull, preflight, and verify the reference

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
SURVEY=09_persistent_page_engine/scripts/degeneration_survey.py
REFERENCE=tmp/09_persistent_page_engine/910b_degeneration_survey_n128_491de50/output
ROOT=tmp/09_persistent_page_engine/310p_phase40_degeneration_n128_$(git rev-parse --short HEAD)

test -f "$REFERENCE/run_summary.json"
test -f "$REFERENCE/recognition_trace.jsonl"
test -f "$SURVEY"
test ! -e "$ROOT"
mkdir -p "$ROOT"

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path("tmp/09_persistent_page_engine/910b_degeneration_survey_n128_491de50/output")
summary = json.loads((root / "run_summary.json").read_text())
rows = [json.loads(line) for line in (root / "recognition_trace.jsonl").read_text().splitlines() if line.strip()]
assert summary["offset"] == 0 and summary["count"] == 128
assert summary["result_count"] == 128 and summary["prediction_count"] == 128
assert summary["recognition"]["requests"] == len(rows) == 2082
assert summary["configuration"]["recognition_input_fingerprints"] is True
assert summary["configuration"]["page_preprocessing_mode"] == "all_before_recognition"
assert summary["recognition"]["stop_reason_counts"] == {"eos": 2081, "kv_cache_full": 1}
for row in rows:
    fp = row.get("input_fingerprints") or {}
    assert (fp.get("crop") or {}).get("sha256")
    assert fp.get("prepared_inputs_sha256")
    assert isinstance(row.get("decode_slot_index"), int)
    assert isinstance(row.get("decode_slot_epoch"), int)
    text = row.get("text_prefill") or {}
    assert isinstance(text.get("private_cache_slot_index"), int)
    assert isinstance(text.get("private_cache_generation"), int)
print("PHASE40_REFERENCE_CONTRACT: PASS")
PY
```

If any assertion fails, report `REFERENCE_CONTRACT_FAILURE` and stop.  Do not
substitute a Phase-37 or Phase-38 reference; they lack this full metadata
contract.

Record exact software/NPU state and warm-cache state before the run.  These
cache roots must already exist and be nonempty; a missing cache is not
permission to compile a new experiment silently.

```sh
{
  date -Is
  hostname
  git rev-parse HEAD
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
} 2>&1 | tee "$ROOT/preflight.log"

for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_vision_router_batched \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_before.txt"
```

### 40.2 Run the matched 128-page 310P lane

Path adaptations are allowed only for dataset/model roots.  Record them.
Do not add `--max-new-tokens`: current production policy generates until EOS
or KV capacity.  Do not run the evaluator in this phase.

```sh
LANE="$ROOT/310p_e2e"
OUTPUT="$LANE/output"
mkdir -p "$LANE"

printf '%q ' timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 128 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --recognition-input-fingerprints \
  --output-dir "$OUTPUT" >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

set +e
timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 128 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --recognition-input-fingerprints \
  --output-dir "$OUTPUT" 2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
test "$run_exit" -eq 0
```

Validate the completed lane mechanically before surveying it:

```sh
"$PYTHON_BIN" - "$REFERENCE" "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

reference = Path(sys.argv[1])
candidate = Path(sys.argv[2])
ref_rows = [json.loads(line) for line in (reference / "recognition_trace.jsonl").read_text().splitlines() if line.strip()]
summary = json.loads((candidate / "run_summary.json").read_text())
rows = [json.loads(line) for line in (candidate / "recognition_trace.jsonl").read_text().splitlines() if line.strip()]
assert summary["offset"] == 0 and summary["count"] == 128
assert summary["result_count"] == 128 and summary["prediction_count"] == 128
assert summary["recognition"]["requests"] == len(rows) == 2082
assert set(row["request_id"] for row in rows) == set(row["request_id"] for row in ref_rows)
assert set(summary["recognition"]["stop_reason_counts"]) <= {"eos", "kv_cache_full"}
for row in rows:
    fp = row.get("input_fingerprints") or {}
    assert (fp.get("crop") or {}).get("sha256")
    assert fp.get("prepared_inputs_sha256")
    assert isinstance(row.get("decode_slot_index"), int)
    assert isinstance(row.get("decode_slot_epoch"), int)
    text = row.get("text_prefill") or {}
    assert isinstance(text.get("private_cache_slot_index"), int)
    assert isinstance(text.get("private_cache_generation"), int)
print("PHASE40_310P_CONTRACT: PASS")
print(json.dumps({
    "pipeline_e2e_s": summary["pipeline_e2e_s"],
    "pages_per_s": summary["pages_per_s"],
    "requests": summary["recognition"]["requests"],
    "stop_reasons": summary["recognition"]["stop_reason_counts"],
}, indent=2))
PY

for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_vision_router_batched \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_after.txt"
diff -u "$ROOT/cache_before.txt" "$ROOT/cache_after.txt" \
  | tee "$ROOT/cache_diff.txt" || true
```

If request IDs do not exactly match the reference, report the missing/extra
sets and stop before surveying; do not force an approximate comparison.
Report any cache size/file-count change and whether setup or first-call timings
show compilation.

### 40.3 Generate the high-recall review set

```sh
"$PYTHON_BIN" "$SURVEY" \
  --reference-output "$REFERENCE" \
  --candidate-output "$OUTPUT" \
  --output-dir "$ROOT/survey" \
  --review-limit 150 \
  2>&1 | tee "$ROOT/survey.log"
```

Open `$ROOT/survey/review.html`.  The flags are triage signals only.  Manually
inspect every row carrying `candidate_runaway_length`, `candidate_repetition`,
`candidate_added_script`, or `candidate_possible_early_eos`, plus the 30 most
severe remaining low-similarity/length-delta rows.  Compare the crop image,
the IoU-matched GT candidate, the 910B output, and the 310P output.

For each inspected row, assign exactly one provisional disposition:

```text
EQUIVALENT_SYNTAX | 310P_BETTER | 910B_BETTER | BOTH_WRONG |
310P_DEGENERATION | 910B_DEGENERATION | INPUT_MISMATCH | UNCERTAIN
```

Call a case `310P_DEGENERATION` only for a material failure such as runaway
repetition, unrelated multilingual/script corruption, gross content loss, or
content unrelated to the visible crop.  A different but valid LaTeX spelling
is not degeneration.  Preserve the full 910B and 310P text for every confirmed
degeneration.

### 40.4 Incidence and category report, then stop

Write `$ROOT/manual_review.md` and `$ROOT/agent_report.md`.  Include:

1. exact commit, host/NPU/software, expanded command, cache-hit/compile evidence;
2. run timing, pages/s, stage times/tok/s, request count, and stop reasons;
3. exact/different/unavailable crop and prepared-input fingerprint counts;
4. total token-exact, token-different, triage-candidate, manually inspected,
   confirmed 310P-degeneration, confirmed 910B-degeneration, and uncertain
   counts;
5. confirmed 310P-degeneration incidence over all 2,082 requests and over only
   input-exact requests;
6. all triage flag counts, while stating explicitly that these are not error
   counts;
7. confirmed-degeneration cross-tabs by OCR label, input-fingerprint status,
   first-use versus reused private-cache generation, first-use versus reused
   decode-slot epoch, vision bucket, text bucket, and output-length band;
8. a case table for every confirmed degeneration with request ID, source page,
   label, crop size, GT candidate, full 910B text, full 310P text, both token
   counts, first divergence, flags, cache slot/generation, decode slot/epoch,
   and route metadata;
9. whether `page_000014_block_000006` recurs as the known multilingual runaway;
10. a short pattern assessment: isolated stochastic-looking cases versus a
    concentration by content class, length, cache reuse, decode-slot reuse, or
    route shape.

Paste the complete `agent_report.md`, `manual_review.md`, and the top-level
summary from `survey/survey.json`.  Keep all run and survey artifacts local on
the work server: this agent is pull-only and must not commit or push them.
Then **stop**.  Do not begin a debugging experiment until the confirmed
incidence and category table have been discussed.

### 40.5 Local-only follow-up: correct the denominator and inspect cache history

Run this only after Phase 40 has been discussed and the user explicitly asks
for the follow-up.  It performs no NPU work and modifies no source.  It reads
the agent's local Phase-39 and Phase-40 traces, writes one local JSON evidence
file, and prints a compact result.

The purpose is to answer exactly three questions:

1. What is the real number of input-exact requests across all 2,082 crops?
2. For each of the seven manually confirmed cases, had its reused private-cache
   slot previously held a longer prompt, leaving a possible stale KV tail?
3. For the known page-14 runaway, were the crop, prepared tensors, and actual
   vision/text pack companions identical between the non-runaway Phase-39 run
   and runaway Phase-40 run?

```sh
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only origin main
git status --short

PYTHON_BIN=/usr/local/python3.12.13/bin/python
P39=tmp/09_persistent_page_engine/310p_phase39_accuracy_lab_8e19fdc/310p_e2e/output
P40_ROOT=tmp/09_persistent_page_engine/310p_phase40_degeneration_n128_4817613
P40="$P40_ROOT/310p_e2e/output"
REF=tmp/09_persistent_page_engine/910b_degeneration_survey_n128_491de50/output

test -f "$P39/run_summary.json"
test -f "$P39/recognition_trace.jsonl"
test -f "$P40/run_summary.json"
test -f "$P40/recognition_trace.jsonl"
test -f "$REF/recognition_trace.jsonl"

"$PYTHON_BIN" - "$P39" "$P40" "$REF" "$P40_ROOT/followup_analysis.json" <<'PY'
import json
import sys
from pathlib import Path

p39_root, p40_root, ref_root, output_path = map(Path, sys.argv[1:])

def load_rows(root):
    return [
        json.loads(line)
        for line in (root / "recognition_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]

def load_summary(root):
    return json.loads((root / "run_summary.json").read_text())

def fingerprint(row):
    value = row.get("input_fingerprints") or {}
    return (
        (value.get("crop") or {}).get("sha256"),
        value.get("prepared_inputs_sha256"),
    )

def visible_tokens(row):
    values = list(row.get("token_ids") or [])
    if values and values[-1] == 2:
        values.pop()
    return values

def pack_members(rows, target, stage):
    route = target[stage]
    group_id = route["pack_group_id"]
    pack_index = route.get("text_pack_index")
    members = []
    for row in rows:
        candidate_route = row.get(stage) or {}
        if candidate_route.get("pack_group_id") != group_id:
            continue
        if stage == "text_prefill" and candidate_route.get("text_pack_index") != pack_index:
            continue
        members.append({
            "block_index": row["block_index"],
            "crop_sha256": fingerprint(row)[0],
            "prepared_inputs_sha256": fingerprint(row)[1],
            "input_tokens": row["input_tokens"],
            "projected_image_tokens": row["projected_image_tokens"],
        })
    return members

p39_rows = load_rows(p39_root)
p40_rows = load_rows(p40_root)
ref_rows = load_rows(ref_root)
p39_summary = load_summary(p39_root)
p40_summary = load_summary(p40_root)

p40_by_id = {row["request_id"]: row for row in p40_rows}
ref_by_id = {row["request_id"]: row for row in ref_rows}
assert set(p40_by_id) == set(ref_by_id)
assert len(p40_by_id) == 2082

input_status = {"exact": 0, "different": 0, "unavailable": 0}
for request_id, candidate in p40_by_id.items():
    left = fingerprint(ref_by_id[request_id])
    right = fingerprint(candidate)
    if not all(left + right):
        input_status["unavailable"] += 1
    elif left == right:
        input_status["exact"] += 1
    else:
        input_status["different"] += 1

degeneration_ids = [
    "page_000014_block_000006",
    "page_000064_block_000005",
    "page_000086_block_000020",
    "page_000090_block_000003",
    "page_000064_block_000007",
    "page_000063_block_000003",
    "page_000111_block_000002",
]

slot_histories = []
for request_id in degeneration_ids:
    row = p40_by_id[request_id]
    route = row["text_prefill"]
    slot = int(route["private_cache_slot_index"])
    generation = int(route["private_cache_generation"])
    history = [
        other
        for other in p40_rows
        if int(other["text_prefill"]["private_cache_slot_index"]) == slot
        and int(other["text_prefill"]["private_cache_generation"]) < generation
    ]
    prior_lengths = [int(other["input_tokens"]) for other in history]
    max_prior = max(prior_lengths, default=0)
    current = int(row["input_tokens"])
    slot_histories.append({
        "request_id": request_id,
        "slot": slot,
        "generation": generation,
        "current_prompt_tokens": current,
        "prior_prompt_tokens": prior_lengths,
        "max_prior_prompt_tokens": max_prior,
        "possible_stale_tail_tokens": max(0, max_prior - current),
        "had_longer_prior_prompt": max_prior > current,
    })

p39_by_id = {row["request_id"]: row for row in p39_rows}
old = p39_by_id["page_000006_block_000006"]
new = p40_by_id["page_000014_block_000006"]
old_pool = p39_summary["recognition"]["text_packing"]["private_cache_pool"]
new_pool = p40_summary["recognition"]["text_packing"]["private_cache_pool"]
known_case = {
    "phase39_tokens": len(visible_tokens(old)),
    "phase40_tokens": len(visible_tokens(new)),
    "crop_and_prepared_exact": fingerprint(old) == fingerprint(new),
    "vision_pack_members_exact": pack_members(p39_rows, old, "vision")
    == pack_members(p40_rows, new, "vision"),
    "text_pack_members_exact": pack_members(p39_rows, old, "text_prefill")
    == pack_members(p40_rows, new, "text_prefill"),
    "phase39_private_pool": old_pool,
    "phase40_private_pool": new_pool,
    "phase40_private_slot": new["text_prefill"]["private_cache_slot_index"],
    "phase40_private_generation": new["text_prefill"]["private_cache_generation"],
}

result = {
    "input_fingerprint_counts_all_requests": input_status,
    "confirmed_degenerations": len(degeneration_ids),
    "incidence_all_requests": len(degeneration_ids) / len(p40_rows),
    "incidence_input_exact": len(degeneration_ids) / input_status["exact"],
    "slot_histories": slot_histories,
    "known_case_phase39_vs_phase40": known_case,
}
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
PY
```

Do not push or commit the JSON.  Report back in **at most three sentences**:

1. the exact all-request fingerprint counts and corrected degeneration rates;
2. how many of seven slots had a longer prior prompt, plus the stale-tail range;
3. whether the known case had exact pack members, its Phase-39/40 token counts
   and reuse states, and whether that supports or contradicts stale KV state.

Do not add general theory, repeat the Phase-40 report, modify code, or begin a
new NPU experiment.

---

## Phase 41: full OmniDocBench v1.6 production run and official evaluation

### Goal and fixed comparison boundary

Run the complete 1,651-page OmniDocBench v1.6 dataset on one 310P3 with the
same production configuration as the completed 910B2 reference, evaluate all
1,651 predictions with the guarded evaluator committed in this repository,
and report a direct performance and quality comparison.

This is a measurement phase, not an optimization or debugging phase.  Do not
change model code, generation policy, packing, graph shapes, evaluator matching
semantics, or the dataset.  Do not add `--max-new-tokens`: production policy is
to generate until EOS or until the 4,096-token KV capacity is exhausted.  The
only allowed path adaptations are the already-established local model,
dataset, evaluator, Python, and cache roots.  Record every adaptation.

The fixed 910B2 reference was run on commit `8634d3a` with warm compiled
caches, physical NPU 7, CANN 9.0.0, torch 2.10.0+cpu, and torch_npu 2.10.0.
Its exact production result is:

| Metric | 910B2 full reference |
|---|---:|
| pages / results / predictions | 1,651 / 1,651 / 1,651 |
| setup | 46.126 s |
| pipeline E2E | 1,055.523 s |
| throughput | 1.56415 pages/s |
| latency-equivalent | 0.63932 s/page |
| all-pages-first layout | 202.385 s, 8.158 pages/s |
| OCR scheduler wall after layout | 849.540 s |
| recognition requests | 30,557 |
| stop reasons | 30,534 EOS; 23 KV-cache-full |
| real / physical vision tokens | 18,805,052 / 21,310,208 |
| vision prefill | 368.851 s; 50,983 real and 57,775 physical tok/s |
| real / physical text-prefill tokens | 5,098,504 / 6,792,832 |
| text prefill | 103.343 s; 49,336 real and 65,731 physical tok/s |
| effective / raw decode tokens | 1,656,185 / 1,727,488 |
| decode | 225.393 s; 7,348 effective and 7,664 raw tok/s |

The 910B evaluator completed with exit 0.  Process-isolated page matching
finished all 1,651 pages, using bounded fallback for three pathological pages.
TEDS evaluated 665 table pairs; one 2,054,838-character malformed prediction
hit the explicit 120-second TEDS timeout, was recorded, and did not hang or
crash the run.  Use the official evaluator page-level metrics below as the
quality comparison values (these are also the values selected by the notebook
summary where that notebook exposes the metric):

| Official evaluator metric | 910B2 full reference |
|---|---:|
| text-block Edit distance | 0.0408832043 |
| display-formula Edit distance | 0.0868281401 |
| table Edit distance | 0.0569611203 |
| table TEDS | 0.9434504390 |
| table TEDS structure-only | 0.9676980846 |
| reading-order Edit distance | 0.1380544862 |

The page denominators are 1,557 text-block pages, 313 display-formula pages,
458 table pages, and 1,638 reading-order pages.  Evaluation wall time was
approximately 358 seconds, including the three overlapping 120-second page
timeouts and one 120-second TEDS timeout.

For clarity, lower is better for all Edit-distance metrics and higher is better
for TEDS.  CDM was not run and must remain omitted.  The table-sample aggregate
printed elsewhere by the evaluator (`TEDS all = 0.929508`) is not the notebook
summary above; do not mix the two aggregation levels.

### 41.1 Pull, verify commits, choose a free NPU, and preflight

Start clean and preserve all evidence locally.  This agent is pull-only: it
must not edit tracked files, create a branch, commit, or push.

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
git rev-parse HEAD
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
ROOT=tmp/09_persistent_page_engine/310p_phase41_full_eval_$(git rev-parse --short HEAD)
LANE="$ROOT/e2e"
OUTPUT="$LANE/output"
EVAL="$ROOT/evaluation"

test -f "$E2E"
test -f "$EVAL_WRAPPER"
test ! -e "$ROOT"
mkdir -p "$LANE" "$EVAL/work"
```

`git status --short` must be empty before starting.  If it is dirty, inventory
the paths and stop; do not discard or hide anything.  `npu-setup` must select a
genuinely free physical 310P3 with enough HBM for the known production caches.
Never kill another user's process.

Locate the evaluator already used in Phase 37.  Prefer the first candidate
whose root contains `pdf_validation.py`; do not clone or update it silently.

```sh
EVALUATOR_ROOT=
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  "$HOME/OmniDocBench_eval" \
  "$HOME/OmniDocBench"
do
  if test -f "$candidate/pdf_validation.py"; then
    EVALUATOR_ROOT="$candidate"
    break
  fi
done

if test -z "$EVALUATOR_ROOT"; then
  find /workspace "$HOME" -maxdepth 5 -type f -name pdf_validation.py \
    -print 2>/dev/null | tee "$ROOT/evaluator_candidates.txt"
  echo "EVALUATOR_ROOT_NOT_FOUND"
  exit 1
fi

EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
test -x "$EVAL_PYTHON"
test -f "$EVALUATOR_ROOT/pdf_validation.py"
```

Record exact state, including the evaluator commit and the physical NPU:

```sh
{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
  printf 'evaluator_root=%s\n' "$EVALUATOR_ROOT"
  git -C "$EVALUATOR_ROOT" rev-parse HEAD
  "$EVAL_PYTHON" -V
} 2>&1 | tee "$ROOT/preflight.log"
```

The evaluator wrapper was validated against evaluator commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`.  If the local evaluator differs,
report the exact commit and stop.  Do not change evaluator revisions without
Luka's approval.

```sh
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

All five production cache roots below must already exist and be nonempty.  A
missing cache is not permission for an unreported fresh compile.  Record cache
file counts and bytes before and after the run.

```sh
for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_vision_router_batched \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_before.txt"
```

### 41.2 Run all 1,651 pages with the matched production configuration

The command below deliberately uses the ten 310P-compatible vision buckets
from Phase 37, B32 static-actual GQA decode, KV4096, PromptFA with 128 alignment,
greedy vision packing, production-group text packing, and all-pages-first
layout.  It omits fingerprints and timeline tracing to match the 910B full
performance reference.  Do not change any of those choices.

```sh
printf '%q ' timeout --signal=TERM --kill-after=30s 10800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 1651 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --output-dir "$OUTPUT" >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 10800 \
  "$PYTHON_BIN" "$E2E" \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --offset 0 --limit 1651 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --vision-batched-cache-dir .runtime_cache/09_vision_router_batched \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --output-dir "$OUTPUT" 2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
run_wall_s="$SECONDS"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
printf '%s\n' "$run_wall_s" >"$LANE/launcher_wall_s.txt"
test "$run_exit" -eq 0
```

The foreground `tee` is the authoritative progress log.  In another terminal,
low-cost monitoring is allowed with:

```sh
tail -f "$LANE/run.log"
watch -n 30 'npu-smi info; wc -l '"$OUTPUT"'/page_regions.jsonl '"$OUTPUT"'/recognition_trace.jsonl 2>/dev/null'
```

Do not enable extra scheduler tracing merely to obtain progress; that would no
longer be the matched performance lane.

Validate the completed output before evaluation:

```sh
"$PYTHON_BIN" - "$OUTPUT" <<'PY' | tee "$LANE/compact_summary.json"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "run_summary.json").read_text())
r = summary["recognition"]
s = r["device_stage_s"]
assert summary["offset"] == 0 and summary["count"] == 1651
assert summary["result_count"] == 1651
assert summary["prediction_count"] == 1651
assert summary["configuration"]["page_preprocessing_mode"] == "all_before_recognition"
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full"}
assert r["requests"] == sum(1 for line in (root / "recognition_trace.jsonl").open() if line.strip())

result = {
    "setup_s": summary["setup_s"],
    "pipeline_e2e_s": summary["pipeline_e2e_s"],
    "pages_per_s": summary["pages_per_s"],
    "s_per_page": summary["s_per_page"],
    "layout_s": summary["layout_frontend"]["stage_s"]["page_total_s"],
    "ocr_scheduler_wall_s": r["run_scoped_scheduler_wall_s"],
    "requests": r["requests"],
    "stop_reasons": r["stop_reason_counts"],
    "vision": {
        "real_tokens": r["real_vision_tokens"],
        "physical_tokens": r["physical_vision_tokens"],
        "seconds": s["vision_prefill"],
        "real_tps": r["real_vision_tokens"] / s["vision_prefill"],
        "physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
    },
    "text": {
        "real_tokens": r["real_text_tokens"],
        "physical_tokens": r["physical_text_tokens"],
        "seconds": s["text_prefill"],
        "real_tps": r["real_text_tokens"] / s["text_prefill"],
        "physical_tps": r["physical_text_tokens"] / s["text_prefill"],
    },
    "decode": {
        "generated_including_eos": r["generated_tokens_including_eos"],
        "effective_tokens": r["effective_decode_tokens"],
        "raw_slots": r["raw_decode_token_slots"],
        "seconds": r["decode_wall_s"],
        "effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
        "raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
    },
}
print(json.dumps(result, indent=2))
PY
```

Print this compact summary immediately when E2E completes so Luka can see the
performance result before evaluation starts.  Then continue to 41.3 unless an
assertion failed.

Record cache state again and report any differences:

```sh
for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_vision_router_batched \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_after.txt"
diff -u "$ROOT/cache_before.txt" "$ROOT/cache_after.txt" \
  | tee "$ROOT/cache_diff.txt" || true
```

### 41.3 Run the guarded full evaluator

Create a fresh evaluation config pointing at the exact Phase-41 subset and
prediction directory.  This runtime config is an artifact under `tmp/`, not a
tracked source edit.

```sh
cat >"$EVAL/work/config.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 12
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $WORK_SERVER_REPO/$OUTPUT/OmniDocBench_subset.json
    prediction:
      data_path: $WORK_SERVER_REPO/$OUTPUT/predictions
    match_method: quick_match
    match_workers: 12
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

cd "$EVAL/work"
ulimit -n 65536
SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 "$EVAL_PYTHON" \
  "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --config config.yaml \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 12 \
  --teds-workers 12 \
  --page-timeout-sec 120 \
  --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30 \
  2>&1 | tee evaluation.log
eval_exit="${PIPESTATUS[0]}"
eval_wall_s="$SECONDS"
set -e
printf '%s\n' "$eval_exit" >../exit_code.txt
printf '%s\n' "$eval_wall_s" >../wall_s.txt
test "$eval_exit" -eq 0
cd "$WORK_SERVER_REPO"
```

The wrapper is intentionally different from invoking `pdf_validation.py`
directly:

- each page match runs in a killable child process;
- a 120-second primary timeout is recorded and retried with bounded fallback;
- fallback LaTeX conversion has its own hard limit;
- exact table strings bypass unnecessary TEDS work;
- every TEDS timeout/error is recorded instead of hanging or corrupting the
  whole evaluation.

Do not exclude difficult pages from ground truth.  Do not rerun only a filtered
499-page set.  A guarded timeout is valid evidence and must remain in the full
1,651-page denominator/report.

Validate the evaluator outputs:

```sh
RESULT="$EVAL/work/result"
test -f "$RESULT/predictions_quick_match_metric_result.json"
test -f "$RESULT/predictions_quick_match_run_summary.json"
test -f "$RESULT/predictions_quick_match_stage_execution.json"

"$EVAL_PYTHON" - "$RESULT" <<'PY' | tee "$EVAL/compact_eval_summary.json"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "predictions_quick_match_run_summary.json").read_text())
metric_result = json.loads(
    (root / "predictions_quick_match_metric_result.json").read_text()
)
stage = report["stage_execution"]
assert stage["page_match"]["page_count"] == 1651

out = {
    "text_block_Edit_dist": metric_result["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
    "display_formula_Edit_dist": metric_result["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_Edit_dist": metric_result["table"]["all"]["Edit_dist"]["ALL_page_avg"],
    "reading_order_Edit_dist": metric_result["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_TEDS": metric_result["table"]["page"]["TEDS"]["ALL"],
    "table_TEDS_structure_only": metric_result["table"]["page"]["TEDS_structure_only"]["ALL"],
    "table_TEDS_sample_aggregate": metric_result["table"]["all"]["TEDS"]["all"],
    "page_denominators": report["page_denominators"],
    "page_match": stage["page_match"],
    "table_TEDS_execution": stage["metrics"]["table"]["TEDS"],
}
print(json.dumps(out, indent=2))
PY
```

The evaluator's `display_formula_CDM` notebook field is expected to remain null
because CDM is disabled; it is not the display-formula Edit distance.  The
compact script deliberately reads the actual display-formula Edit-distance
`ALL_page_avg` directly from `predictions_quick_match_metric_result.json`.

### 41.4 Produce the head-to-head report, then stop

Write `$ROOT/agent_report.md`.  Begin with exactly one classification:

```text
310P PHASE 41 FULL: PASS | E2E_FAILURE | EVALUATOR_FAILURE |
COMPILE_CONTAMINATED | DATASET_MISMATCH
```

Include all of the following:

1. exact project/evaluator commits, host, physical NPU, CANN, driver, firmware,
   Python, torch, and torch_npu;
2. exact expanded E2E and evaluation commands and all artifact paths;
3. setup, pipeline E2E, pages/s, seconds/page, layout time/pages-s, and OCR
   scheduler wall;
4. request count; stop reasons; real/physical vision tokens; real/physical
   text-prefill tokens; generated/effective/raw decode tokens;
5. vision/text/decode seconds and real/physical/effective/raw tok/s;
6. every device-stage total, packing group counts/fill fractions/histograms,
   decode graph calls, idle/lookahead slots, KV bytes copied, and private-cache
   high-water mark;
7. cache before/after diff and explicit warm-cache versus compile-contaminated
   classification;
8. official evaluator page-level quality metrics, all denominators, page-match
   fallback cases, TEDS timeout/error/exception cases, and evaluator wall time;
9. signed quality deltas against 910B2: `310P - 910B` for Edit distances and
   `310P - 910B` for TEDS, while marking the desired direction correctly;
10. performance ratios: 310P/910B pages/s and each 310P/910B stage tok/s, plus
    the reciprocal slowdown;
11. any long/degenerate prediction responsible for evaluator fallback or TEDS
    timeout, without manually excluding it from the score;
12. concise `what is proven`, `what is not proven`, and the first causal error
    if any stage failed.

Paste the complete `agent_report.md`, `e2e/compact_summary.json`,
`evaluation/compact_eval_summary.json`, and both evaluator stage/run summaries.
Keep all artifacts local on the work server.  Do not modify source, commit,
push, or begin another optimization.  Then **stop**.

---

## Phase 42: recover Phase-41 TEDS without rerunning inference or page matching

### Goal and diagnosis

Do **not** rerun the 1,651-page pipeline.  Do **not** rerun page matching.  The
Phase-41 evaluator already saved the 665 matched table samples; recompute TEDS
and TEDS-structure directly from that frozen input.

The Phase-41 TEDS result is invalid because evaluator commit `2b161d0` uses a
12-thread `ThreadPoolExecutor`, and each thread starts another
`multiprocessing.Process`.  If `process.start()` fails, the evaluator's
unconditional `process.join()` in `finally` masks the original exception with
`AssertionError: can only join a started process`.  That happened for 134
table pairs, which were incorrectly recorded as zero scores.

The repository wrapper now replaces that nested thread/process design with a
bounded parent scheduler.  The main thread starts at most 12 direct TEDS child
processes; the parent owns each hard timeout and cleanup.  A child metric error
is still scored as zero under the evaluator's established semantics, but a
worker-start or worker-lifecycle failure raises with its real cause instead of
silently becoming a score.

This phase also audits the independently computed table Edit-distance result.
TEDS worker failures cannot alter table Edit distance: Edit distance is computed
later from the frozen normalized `gt` and `pred` strings.  Do not call table
Edit distance "contaminated" by TEDS.  The two 310P-only page-match fallbacks
also cannot by themselves explain the `+0.5513` page-average gap: with 458 table
pages, even changing two pages from perfect to maximally wrong moves the mean by
at most `2 / 458 = 0.00437`.

This is a CPU-only evaluator recovery.  Do not source `npu-setup`, reserve an
NPU, load either model, or touch compiler caches.

### 42.1 Pull and verify the frozen Phase-41 inputs

The work-server agent remains pull-only.  It must not edit tracked files,
create a branch, commit, or push.

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"

EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
PHASE41=tmp/09_persistent_page_engine/310p_phase41_full_eval_7bda07e
OLD_RESULT="$PHASE41/evaluation/work/result"
TABLE_INPUT="$OLD_RESULT/predictions_quick_match_table_result.json"
OLD_METRIC="$OLD_RESULT/predictions_quick_match_metric_result.json"
RERUN="$PHASE41/evaluation_teds_process_$(git rev-parse --short HEAD)"
WORK="$RERUN/work"

test -x "$EVAL_PYTHON"
test -f "$EVAL_WRAPPER"
test -f "$TABLE_INPUT"
test -f "$OLD_METRIC"
test ! -e "$RERUN"
mkdir -p "$RERUN"
rg -n 'process_isolated_parent_timeout|teds-only-input' "$EVAL_WRAPPER" \
  | tee "$RERUN/wrapper_markers.txt"
```

Locate and pin the same evaluator checkout used in Phase 41:

```sh
EVALUATOR_ROOT=
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  "$HOME/OmniDocBench_eval" \
  "$HOME/OmniDocBench"
do
  if test -f "$candidate/pdf_validation.py"; then
    EVALUATOR_ROOT="$candidate"
    break
  fi
done
test -n "$EVALUATOR_ROOT"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'evaluator_root=%s\n' "$EVALUATOR_ROOT"
  git -C "$EVALUATOR_ROOT" rev-parse HEAD
  "$EVAL_PYTHON" -V
  sha256sum "$TABLE_INPUT" "$OLD_METRIC"
} | tee "$RERUN/preflight.log"
```

Confirm that the frozen input contains exactly 665 matched table pairs over
458 table pages before starting any metric work:

```sh
"$EVAL_PYTHON" - "$TABLE_INPUT" <<'PY' | tee "$RERUN/input_contract.json"
import json
import sys
from pathlib import Path

samples = json.loads(Path(sys.argv[1]).read_text())
pages = set()
for sample in samples:
    img_id = sample["img_id"]
    if img_id.endswith((".jpg", ".png")):
        page = img_id
    else:
        page = "_".join(img_id.split("_")[:-1])
    pages.add(page)
out = {"sample_count": len(samples), "page_count": len(pages)}
assert out == {"sample_count": 665, "page_count": 458}, out
print(json.dumps(out, indent=2))
PY
```

If any contract fails, report it and stop.  Do not search for a different
input or regenerate matching.

### 42.2 Recompute only TEDS from the frozen matched tables

`--teds-only-output-dir` must name a nonexistent directory; the wrapper creates
it.  The outer `RERUN` directory already exists so the foreground log remains
visible while the metric runs.

```sh
printf '%q ' timeout --signal=TERM --kill-after=30s 1800 \
  "$EVAL_PYTHON" "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --teds-workers 12 \
  --teds-timeout-sec 120 \
  --teds-expected-samples 665 \
  --teds-expected-pages 458 \
  --teds-only-input "$WORK_SERVER_REPO/$TABLE_INPUT" \
  --teds-only-output-dir "$WORK_SERVER_REPO/$WORK" \
  >"$RERUN/command.sh"
printf '\n' >>"$RERUN/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 1800 \
  "$EVAL_PYTHON" "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --teds-workers 12 \
  --teds-timeout-sec 120 \
  --teds-expected-samples 665 \
  --teds-expected-pages 458 \
  --teds-only-input "$WORK_SERVER_REPO/$TABLE_INPUT" \
  --teds-only-output-dir "$WORK_SERVER_REPO/$WORK" \
  2>&1 | tee "$RERUN/run.log"
run_exit="${PIPESTATUS[0]}"
run_wall_s="$SECONDS"
set -e
printf '%s\n' "$run_exit" >"$RERUN/exit_code.txt"
printf '%s\n' "$run_wall_s" >"$RERUN/wall_s.txt"
test "$run_exit" -eq 0
```

Progress must visibly use the `TEDS (process-isolated)` bar.  The log must not
contain `can only join a started process`.  A timeout is an explicit, valid
zero-score result.  A worker start/lifecycle failure must terminate the command
with its real error and is an evaluator failure.

Validate and print the corrected headline values immediately:

```sh
SUMMARY="$WORK/teds_only_summary.json"
test -f "$SUMMARY"
"$EVAL_PYTHON" - "$SUMMARY" <<'PY' | tee "$RERUN/compact_summary.json"
import json
import sys
from pathlib import Path

s = json.loads(Path(sys.argv[1]).read_text())
e = s["execution"]
assert s["sample_count"] == 665, s["sample_count"]
assert s["page_count"] == 458, s["page_count"]
assert e["scheduler"] == "process_isolated_parent_timeout", e
assert e["sample_count"] == 665, e
assert e["error_case_count"] == 0, e["error_cases"]
out = {
    "elapsed_s": s["elapsed_s"],
    "sample_TEDS": s["sample_aggregate"]["TEDS"]["all"],
    "sample_TEDS_structure_only": s["sample_aggregate"]["TEDS_structure_only"]["all"],
    "page_TEDS": s["page_aggregate"]["TEDS"]["ALL"],
    "page_TEDS_structure_only": s["page_aggregate"]["TEDS_structure_only"]["ALL"],
    "timeouts": e["timeout_case_count"],
    "timeout_cases": e["timeout_cases"],
    "errors": e["error_case_count"],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

If there are genuine child metric errors, the wrapper records them as zero in
accordance with the evaluator, but the strict compact check above intentionally
fails.  Report the cases and stop rather than presenting a corrected score as
clean.

### 42.3 Audit the separate table Edit-distance regression

Use the frozen table sample file.  This does not execute TEDS and does not
change any score.  It verifies the official page-average Edit distance and
shows whether the regression is broad or concentrated.

```sh
"$EVAL_PYTHON" - "$TABLE_INPUT" "$OLD_METRIC" <<'PY' \
  >"$RERUN/table_edit_diagnostics.json"
import collections
import json
import math
import sys
from pathlib import Path

samples = json.loads(Path(sys.argv[1]).read_text())
metric = json.loads(Path(sys.argv[2]).read_text())
page_acc = collections.defaultdict(lambda: [0.0, 0])
sample_rows = []
raw_formats = collections.Counter()
normalized_formats = collections.Counter()

def page_name(img_id):
    if img_id.endswith((".jpg", ".png")):
        return img_id
    return "_".join(img_id.split("_")[:-1])

def classify(text):
    text = str(text or "").lstrip().lower()
    if not text:
        return "empty"
    if text.startswith("<fcel") or "<fcel" in text[:256]:
        return "fcel"
    if text.startswith("<table") or "<table" in text[:256]:
        return "html_table"
    return "other"

for sample in samples:
    gt = sample.get("norm_gt") or sample.get("gt") or ""
    pred = sample.get("norm_pred") or sample.get("pred") or ""
    upper = max(len(gt), len(pred))
    edit = float(sample["metric"]["Edit_dist"])
    name = page_name(sample["img_id"])
    page_acc[name][0] += edit * upper
    page_acc[name][1] += upper
    raw_formats[classify(sample.get("pred"))] += 1
    normalized_formats[classify(sample.get("norm_pred"))] += 1
    sample_rows.append({
        "img_id": sample["img_id"],
        "gt_idx": sample.get("gt_idx"),
        "pred_idx": sample.get("pred_idx"),
        "edit": edit,
        "gt_len": len(gt),
        "pred_len": len(pred),
        "raw_format": classify(sample.get("pred")),
        "normalized_format": classify(sample.get("norm_pred")),
    })

page_scores = {
    name: edits / upper if upper else 0.0
    for name, (edits, upper) in page_acc.items()
}
values = sorted(page_scores.values())
def quantile(q):
    if not values:
        return None
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)

official = metric["table"]["all"]["Edit_dist"]["ALL_page_avg"]
recomputed = sum(values) / len(values)
assert len(samples) == 665 and len(values) == 458
assert abs(official - recomputed) < 1e-12, (official, recomputed)
out = {
    "sample_count": len(samples),
    "page_count": len(values),
    "official_page_Edit_dist": official,
    "recomputed_page_Edit_dist": recomputed,
    "page_quantiles": {
        "p0": quantile(0), "p25": quantile(.25), "p50": quantile(.5),
        "p75": quantile(.75), "p90": quantile(.9), "p95": quantile(.95),
        "p99": quantile(.99), "p100": quantile(1),
    },
    "page_threshold_counts": {
        str(t): sum(value >= t for value in values)
        for t in (0.1, 0.25, 0.5, 0.75, 0.9)
    },
    "sample_threshold_counts": {
        str(t): sum(row["edit"] >= t for row in sample_rows)
        for t in (0.1, 0.25, 0.5, 0.75, 0.9)
    },
    "raw_prediction_formats": dict(raw_formats),
    "normalized_prediction_formats": dict(normalized_formats),
    "worst_20_pages": sorted(
        ({"img_id": name, "Edit_dist": score} for name, score in page_scores.items()),
        key=lambda row: (-row["Edit_dist"], row["img_id"]),
    )[:20],
    "worst_20_samples": sorted(
        sample_rows,
        key=lambda row: (-row["edit"], -row["pred_len"], row["img_id"]),
    )[:20],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

cat "$RERUN/table_edit_diagnostics.json"
```

### 42.4 Report, then stop

Write `$RERUN/agent_report.md` and paste it to Luka.  Begin with exactly one:

```text
310P PHASE 42 TEDS RECOVERY: PASS | INPUT_MISMATCH | TEDS_TIMEOUTS |
TEDS_METRIC_ERRORS | EVALUATOR_INFRASTRUCTURE_FAILURE
```

Include:

1. project and evaluator commits, host, Python, frozen input SHA-256, exact
   command, exit code, and wall time;
2. corrected sample TEDS/TEDS-structure and corrected page TEDS/TEDS-structure;
3. timeout and error counts with every case;
4. deltas of corrected page scores against the 910B page references
   `0.9434504389897741` and `0.9676980845673955`;
5. the complete table Edit diagnostic summary, especially median/p90/p95,
   threshold counts, format counts, and worst 20 pages;
6. an explicit statement that table Edit `0.608267...` was reproduced from the
   frozen samples and is independent of the Phase-41 TEDS process failures;
7. concise `what is proven` and `what remains unresolved`.

Paste `agent_report.md`, `compact_summary.json`, and
`table_edit_diagnostics.json`.  Do not rerun inference, matching, or another
metric.  Do not edit source, commit, push, or begin a new experiment.  Then
**stop**.

---

## Phase 43: localize the 396 missing tables across the frozen pipeline

### Goal

Phase 42 proved that 396 of 665 310P GT-table samples reached the evaluator
with an empty prediction.  An empty evaluator prediction proves only that the
GT table was left unmatched; it does not identify which upstream boundary lost
it.  Trace the same frozen Phase-41 artifacts through:

```text
GT table bbox
  -> best-overlap final page block and label
  -> recognition request label/prompt
  -> raw recognition text
  -> final page-block content
  -> evaluator matched prediction
```

This is a CPU-only artifact audit.  Do not rerun inference, page matching, or
TEDS.  Do not source `npu-setup` or reserve an NPU.

The identical audit already passed on the frozen 910B full run.  Its committed
reference is:

```text
tmp/09_persistent_page_engine/910b_table_pipeline_audit_fad036f/report.json
```

The critical 910B reference counts are:

| Boundary | 910B count out of 665 GT tables |
|---|---:|
| best-overlap block IoU >= 0.5 | 657 |
| best-overlap block labeled table | 657 |
| Table Recognition request | 641 |
| any non-empty recognition text | 662 |
| non-empty evaluator prediction | 657 |
| empty evaluator prediction | 8 |

Across the whole 910B run there were 751 final blocks labeled `table`, 751
table recognition requests, and 751 `Table Recognition:` prompts.

### 43.1 Pull and verify frozen inputs

The work-server agent remains pull-only.  Do not edit tracked files, create a
branch, commit, or push.

```sh
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"

AUDIT=09_persistent_page_engine/scripts/audit_table_pipeline.py
PHASE41=tmp/09_persistent_page_engine/310p_phase41_full_eval_7bda07e
E2E_OUTPUT="$PHASE41/e2e/output"
TABLE_RESULT="$PHASE41/evaluation/work/result/predictions_quick_match_table_result.json"
REFERENCE=tmp/09_persistent_page_engine/910b_table_pipeline_audit_fad036f/report.json
ROOT="$PHASE41/table_pipeline_audit_$(git rev-parse --short HEAD)"
REPORT="$ROOT/report.json"

test -f "$AUDIT"
test -f "$E2E_OUTPUT/OmniDocBench_subset.json"
test -f "$E2E_OUTPUT/page_regions.jsonl"
test -f "$E2E_OUTPUT/recognition_trace.jsonl"
test -f "$TABLE_RESULT"
test -f "$REFERENCE"
test ! -e "$ROOT"
mkdir -p "$ROOT"

{
  date -Is
  hostname
  git rev-parse HEAD
  sha256sum \
    "$E2E_OUTPUT/OmniDocBench_subset.json" \
    "$E2E_OUTPUT/page_regions.jsonl" \
    "$E2E_OUTPUT/recognition_trace.jsonl" \
    "$TABLE_RESULT" \
    "$REFERENCE"
} | tee "$ROOT/preflight.log"
```

### 43.2 Run the frozen-artifact audit

```sh
PYTHONUNBUFFERED=1 /usr/local/python3.12.13/bin/python \
  "$AUDIT" \
  --e2e-output "$E2E_OUTPUT" \
  --table-result "$TABLE_RESULT" \
  --report "$REPORT" \
  --expected-pages 1651 \
  --expected-tables 665 \
  2>&1 | tee "$ROOT/run.log"
test "${PIPESTATUS[0]}" -eq 0
```

The audit uses no model code.  It matches each GT table to the final page block
with maximum bbox IoU, then follows that block ID into the already recorded
recognition trace and finally the frozen evaluator sample.

Create a compact direct comparison against 910B:

```sh
/usr/local/python3.12.13/bin/python - "$REPORT" "$REFERENCE" <<'PY' \
  | tee "$ROOT/comparison.json"
import json
import sys
from pathlib import Path

test = json.loads(Path(sys.argv[1]).read_text())
ref = json.loads(Path(sys.argv[2]).read_text())

def value(report, *path):
    node = report
    for key in path[:-1]:
        node = node[key]
    return node.get(path[-1], 0)

paths = {
    "whole_run_layout_table": ("whole_run", "layout_label_histogram", "table"),
    "whole_run_recognition_table": ("whole_run", "recognition_label_histogram", "table"),
    "gt_good_iou": ("gt_table_path", "iou_band", "good_ge_0.5"),
    "gt_best_block_table": ("gt_table_path", "best_block_label", "table"),
    "gt_table_recognition": ("gt_table_path", "recognition_label", "table"),
    "gt_recognition_text_nonempty": ("gt_table_path", "recognition_text_nonempty", "True"),
    "gt_evaluator_pred_nonempty": ("gt_table_path", "evaluator_pred_nonempty", "True"),
    "gt_evaluator_pred_empty": ("gt_table_path", "evaluator_pred_nonempty", "False"),
}

out = {}
for name, path in paths.items():
    test_value = value(test, *path)
    ref_value = value(ref, *path)
    out[name] = {
        "310P": test_value,
        "910B": ref_value,
        "delta_310P_minus_910B": test_value - ref_value,
    }
out["310P_failure_stage"] = test["gt_table_path"]["failure_stage"]
out["910B_failure_stage"] = ref["gt_table_path"]["failure_stage"]
out["310P_empty_by_failure_stage"] = test["empty_evaluator_predictions"]["failure_stage"]
out["310P_empty_by_best_block_label"] = test["empty_evaluator_predictions"]["best_block_label"]
out["310P_empty_by_recognition_label"] = test["empty_evaluator_predictions"]["recognition_label"]
out["310P_stage_paths"] = test["gt_table_path"]["stage_paths"]
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

### 43.3 Interpret mechanically, report, and stop

Use the first boundary whose count collapses relative to 910B:

- Low GT-table IoU means layout localization/geometry.
- Good IoU but a non-table best-block label means layout classification.
- Table block but non-table/missing recognition request means routing.
- Table request but empty recognition text means recognition generation.
- Non-empty table recognition and final content but empty evaluator prediction
  means page assembly/parser/matching.

Write `$ROOT/agent_report.md`, beginning with:

```text
310P PHASE 43 TABLE LOSS LOCALIZATION: PASS | INPUT_MISMATCH | AUDIT_FAILURE
```

Report the complete `comparison.json`, the 310P whole-run label/prompt counts,
all 310P GT-table boundary counts, the empty-prediction failure-stage split,
and the top five `stage_paths`.  End with one sentence naming the first proven
dominant loss boundary.  Do not speculate about a model numerical cause yet.

Paste `agent_report.md` and `comparison.json`.  Do not run another experiment,
modify source, commit, push, or use an NPU.  Then **stop**.

---

## Phase 44: exact full-run generation and OmniDocBench difference atlas

### Goal and boundary

Use only the already-frozen Phase-41 inference/evaluator output and the
corrected Phase-42 TEDS sidecars.  Compare them against the committed full
910B reference at three separate levels:

1. raw per-crop recognizer generations;
2. exact page-level contribution to every official OmniDocBench metric;
3. evaluator sample rows used only to localize the page-level changes.

This phase is CPU-only.  Do not source `npu-setup`, reserve an NPU, load the
model, rerun inference, rerun matching, or rerun TEDS.  Do not modify source,
commit, or push.  The work-server agent remains pull-only.

Important interpretation boundaries:

- Page-level contribution is exact and additive.  Sample-level contribution is
  denominator-coupled localization evidence, not proof that one block caused a
  page score.
- Reading order is content-match-coupled.  It is not a pure layout metric.
- A `(page, block_index)` table transition without exact input fingerprints is
  a stable-key observation, not proof of a hardware-only numerical change.
- Heuristic runaway/repetition flags are review candidates, not confirmed
  degeneration labels.
- Do not convert pipe/plain tables or otherwise change predictions.

### 44.1 Pull and preflight the frozen evidence

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"

PHASE41=tmp/09_persistent_page_engine/310p_phase41_full_eval_7bda07e
E2E_OUTPUT="$PHASE41/e2e/output"
EVAL_DIR="$PHASE41/evaluation/work/result"
PHASE42="$PHASE41/evaluation_teds_process_3c24d44/work"
TABLE_SCORES="$PHASE42/result/teds_recomputed_per_table_TEDS.json"
TEDS_SUMMARY="$PHASE42/teds_only_summary.json"

REFERENCE=tmp/09_persistent_page_engine/910b_generation_difference_reference_ab00d1f/omnidocbench_v1_6_910b2_full_8634d3a.gdatlas.zip
EXPORTER=09_persistent_page_engine/scripts/export_generation_difference_bundle.py
ATLAS_SCRIPT=09_persistent_page_engine/scripts/generation_difference_atlas.py
ROOT="$PHASE41/generation_difference_atlas_$(git rev-parse --short HEAD)"
CANDIDATE="$ROOT/omnidocbench_v1_6_310p3_full_phase41_phase42.gdatlas.zip"
ATLAS="$ROOT/atlas"

test -f "$E2E_OUTPUT/recognition_trace.jsonl"
test -f "$E2E_OUTPUT/run_summary.json"
test -f "$EVAL_DIR/predictions_quick_match_text_block_result.json"
test -f "$EVAL_DIR/predictions_quick_match_display_formula_result.json"
test -f "$EVAL_DIR/predictions_quick_match_table_result.json"
test -f "$EVAL_DIR/predictions_quick_match_reading_order_result.json"
test -f "$EVAL_DIR/predictions_quick_match_metric_result.json"
test -f "$TABLE_SCORES"
test -f "$TEDS_SUMMARY"
test -f "$REFERENCE"
test -f "$EXPORTER"
test -f "$ATLAS_SCRIPT"
test ! -e "$ROOT"
mkdir -p "$ROOT"

REFERENCE_SHA=$(sha256sum "$REFERENCE" | awk '{print $1}')
test "$REFERENCE_SHA" = a1c2ec99b8aa2b0a18f26cedc9fa7383aa42c78620224aed497035b46bb1ba84

{
  date -Is
  hostname
  git rev-parse HEAD
  sha256sum \
    "$E2E_OUTPUT/recognition_trace.jsonl" \
    "$E2E_OUTPUT/run_summary.json" \
    "$EVAL_DIR/predictions_quick_match_text_block_result.json" \
    "$EVAL_DIR/predictions_quick_match_display_formula_result.json" \
    "$EVAL_DIR/predictions_quick_match_table_result.json" \
    "$EVAL_DIR/predictions_quick_match_reading_order_result.json" \
    "$EVAL_DIR/predictions_quick_match_metric_result.json" \
    "$TABLE_SCORES" \
    "$TEDS_SUMMARY" \
    "$REFERENCE"
} 2>&1 | tee "$ROOT/preflight.log"
```

Hard contracts for this comparison:

```text
pages                         1651 on each side
910B recognition requests    30557
310P recognition requests    30568
raw table requests            751 on each side
910B text evaluator rows    19689
310P text evaluator rows    19728
formula evaluator rows       2352 on each side
table evaluator rows          665 on each side, across 458 pages
reading-order rows           1638 on each side
evaluator commit             2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

The exporter validates that all evaluator pages belong to the exact ordered
1,651-image run, that corrected TEDS has complete 665-key coverage, and that
the corrected score summary names the exact frozen table-result file.  It must
stop rather than fall back to the Phase-41 TEDS values containing 134 worker
errors.

The asymmetric text-row counts are deliberate frozen contracts.  The initial
Phase-44 attempt proved that the 310P evaluator contains 19,728 rows across the
same 1,557 text-scored pages, while the 910B bundle contains 19,689.  Evaluator
rows are prediction-dependent match groups, so this `+39` difference is not by
itself proof of 39 extra layout detections.  The atlas must preserve it and use
the concrete GT-atom membership audit plus exact page-level recomposition to
distinguish grouping changes, missing matches, and meaningful score changes.
GT atoms present on only one side are evidence to report, not a reason to drop
the affected page or abort the page-level comparison.

### 44.2 Export the frozen 310P bundle

```sh
PYTHONUNBUFFERED=1 /usr/local/python3.12.13/bin/python \
  "$EXPORTER" \
  --e2e-output "$E2E_OUTPUT" \
  --eval-dir "$EVAL_DIR" \
  --output "$CANDIDATE" \
  --label 310P3 \
  --project-commit 7bda07e662f855d5988552e9fb6bce81a11a330f \
  --evaluator-commit 2b161d010d2e3aff77a0edef359ea3a6411d23cd \
  --table-scores "$TABLE_SCORES" \
  --teds-summary "$TEDS_SUMMARY" \
  --require-corrected-teds \
  --expected-pages 1651 \
  --expected-requests 30568 \
  --expected-table-requests 751 \
  --expected-text-rows 19728 \
  --expected-formula-rows 2352 \
  --expected-table-rows 665 \
  --expected-reading-order-rows 1638 \
  --expected-table-pages 458 \
  2>&1 | tee "$ROOT/export.log"
test "${PIPESTATUS[0]}" -eq 0

sha256sum "$CANDIDATE" | tee "$ROOT/candidate_bundle.sha256"
unzip -p "$CANDIDATE" manifest.json \
  | /usr/local/python3.12.13/bin/python -m json.tool \
  | tee "$ROOT/candidate_manifest.json"
```

Report this checkpoint immediately in the live progress stream:

```text
310P PHASE 44 EXPORT: PASS | EXPORT_CONTRACT_FAILURE
```

Include the bundle SHA-256 and byte size.  If export fails, report the first
failed contract and stop; do not weaken a count, provenance, or corrected-TEDS
check.

### 44.3 Run the bundle-to-bundle atlas

```sh
PYTHONUNBUFFERED=1 /usr/local/python3.12.13/bin/python \
  "$ATLAS_SCRIPT" \
  --reference-bundle "$REFERENCE" \
  --candidate-bundle "$CANDIDATE" \
  --reference-label 910B2 \
  --candidate-label 310P3 \
  --output-dir "$ATLAS" \
  --expected-pages 1651 \
  --expected-reference-table-requests 751 \
  --expected-candidate-table-requests 751 \
  --review-limit 200 \
  2>&1 | tee "$ROOT/atlas.log"
test "${PIPESTATUS[0]}" -eq 0

test -f "$ATLAS/report.json"
test -f "$ATLAS/generation_records.jsonl"
test -f "$ATLAS/metric_records.jsonl"
test -f "$ATLAS/page_metric_records.jsonl"
test -f "$ATLAS/evaluator_gt_universe_differences.jsonl"
test -f "$ATLAS/table_relevance_pages.jsonl"
test -f "$ATLAS/table_logit_candidates.json"
test -f "$ATLAS/review.html"

jq -e '.teds_authority_audit.reference.authority == "corrected_process_isolated"' \
  "$ATLAS/report.json"
jq -e '.teds_authority_audit.candidate.authority == "corrected_process_isolated"' \
  "$ATLAS/report.json"
jq -e '.generation.reference_requests == 30557' "$ATLAS/report.json"
jq -e '.generation.candidate_requests == 30568' "$ATLAS/report.json"
```

The command prints progress after generation pairing, each evaluator family,
each TEDS view, and artifact writing.  It must not be silent for the whole run.

### 44.4 Produce the compact evidence summary

```sh
/usr/local/python3.12.13/bin/python - "$ATLAS/report.json" <<'PY' \
  | tee "$ROOT/headline.json"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
metrics = {}
for name, value in report["metrics"].items():
    metrics[name] = {
        "reference": value.get("official_reference"),
        "candidate": value.get("official_candidate"),
        "candidate_minus_reference": value.get("official_candidate_minus_reference"),
        "page_concentration": value.get("page_concentration"),
        "pair_status": value.get("pair_status"),
        "raw_difference_class": value.get("raw_difference_class"),
        "normalized_difference_class": value.get("normalized_difference_class"),
    }

out = {
    "generation": report["generation"],
    "metrics": metrics,
    "evaluator_gt_universe_audit": report["evaluator_gt_universe_audit"],
    "teds_authority_audit": report["teds_authority_audit"],
    "table_format_to_omnidocbench": report["table_format_to_omnidocbench"],
    "reading_order_evaluator_pred_idx_zero_audit": report["reading_order_evaluator_pred_idx_zero_audit"],
    "table_logit_candidates": report["table_logit_candidates"],
    "top_harmful_pages": {
        name: rows[:20] for name, rows in report["top_harmful_pages"].items()
    },
    "top_harmful_samples": {
        name: rows[:20] for name, rows in report["top_harmful_samples"].items()
    },
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

### 44.5 Manual review questions

Read `headline.json`, `report.json`, and the relevant full JSONL rows.  Review
the largest positive page/sample contributors instead of treating all 30,000+
string differences alike.

Answer these questions with concrete page/block examples:

1. How many raw generations are exact, whitespace/NFKC/wrapper-only, small
   content changes, substantial content changes, and heuristic runaway or
   repetition candidates?  Split by text/formula/table label.
   Also report every evaluator GT-result membership difference.  These are
   reference-only/candidate-only localization records; do not force-pair them
   or remove their pages from the exact page-level comparison.
2. For text and formulas, do a few pages dominate the score delta?  Manually
   classify the ten largest positive-loss samples as one of:
   `syntax/spacing only`, `310P clearly better`, `910B clearly better`,
   `both wrong`, `real 310P degeneration`, or `evaluator/matching confound`.
   Specifically search for the observed `minus -> \\quad` substitutions.
3. For tables, report the independent raw format counts on each device, the
   stable-key transition matrix, how many transitions have exact input proof,
   and the table Edit/TEDS loss grouped by transition signature.  Do not call
   unverified stable-key rows hardware-caused.
4. For reading order, report how much of the +delta is concentrated in the top
   10/25/50 pages.  Separate missing members from actual inversions and report
   the `pred_idx=0` evaluator-defect counts on both sides.
5. Identify exact-target-input `910B fcel -> 310P pipe/plain` cases whose first
   token diverges.  These are candidate leads only; a later replay must first
   reconstruct the complete vision/text pack companions, order, and offsets.

Do not infer semantic dominance from Edit distance alone.  Include examples
where the 310P syntax differs but is equally correct or closer to GT.

### 44.6 Report and stop

Write `$ROOT/agent_report.md`, beginning with exactly one classification:

```text
310P PHASE 44 DIFFERENCE ATLAS: PASS | EXPORT_CONTRACT_FAILURE |
REFERENCE_BUNDLE_MISMATCH | ATLAS_RECONCILIATION_FAILURE
```

For PASS, report:

- project/evaluator commits and both bundle SHA-256 values;
- paired/reference-only/candidate-only generation counts;
- generation difference/triage counts by label;
- every official 910B/310P score and signed delta;
- page-level concentration for text, formula, table Edit, corrected page TEDS,
  and reading order;
- the evaluator GT-result universe audit, including every one-sided atom or
  reading-order GT-sequence difference;
- the table-format transition/relevance results;
- the manual ten-example disposition for text/formula and the true degeneration
  count found in this review;
- reading-order missing/inversion split and evaluator-defect audit;
- the top exact-target table leads for a later token-zero logit replay;
- `what is proven` and `what remains unresolved`.

Paste `agent_report.md` plus `headline.json`.  Full bundle, JSONL, and HTML
artifacts remain local on the work server.  Do not use an NPU or begin the logit
replay.  Then **stop**.

### 44.7 Compact manual-transfer addendum

If `headline.json` is too large to transfer through the coordinating chat, do
not rerun the atlas, inference, matching, or TEDS.  Read the already-generated
`headline.json`, `report.json`, and JSONL artifacts, then append a section named
`Compact transfer summary` to the existing `$ROOT/agent_report.md`.  Keep the
complete Markdown report below 100 KB.  Do not paste raw JSON, full token-ID
sequences, complete generations, or large arrays.

The compact section must include:

1. **Generation comparison:** paired/reference-only/candidate-only request
   counts; every difference-class count split by text/formula/table; triage
   flag counts; and every manually confirmed real degeneration with
   page/block, token counts, and a short description.
2. **Official metrics:** one table containing every 910B score, 310P score,
   and signed candidate-minus-reference delta for text Edit, formula Edit,
   table Edit, corrected page TEDS, corrected page TEDS-structure, and reading
   order Edit.
3. **Metric concentration:** for each metric, top-10/top-25/top-50 page
   contribution to net and gross regression.  List only the ten most harmful
   pages with reference score, candidate score, and signed contribution.
4. **Evaluator GT-result universe audit:** exact/nonexact status,
   differing-page count, and reference-only/candidate-only atom counts for
   every evaluator kind.  Include every difference when there are at most 20;
   otherwise include the first 20 and the total counts.
5. **Manual text/formula review:** the ten reviewed examples and their
   disposition (`syntax/spacing only`, `310P clearly better`, `910B clearly
   better`, `both wrong`, `real 310P degeneration`, or
   `evaluator/matching confound`), with short GT/910B/310P excerpts.  Report
   every observed minus-to-`\\quad` case.
6. **Tables:** independent raw format counts for both devices; the complete
   compact transition matrix; exact-input-proven versus stable-key-only
   transition counts; table Edit/TEDS loss grouped by transition signature;
   and the five strongest exact-target token-zero replay candidates.
7. **Reading order:** worse/better/unchanged page counts; missing-member versus
   inversion counts; top-10/top-25/top-50 concentration; and `pred_idx=0`
   evaluator-defect counts for both devices.
8. Finish with explicit `What is proven` and `What remains unresolved`
   sections.

Preserve the existing report content.  After appending the compact section,
print only the updated `agent_report.md` for manual transfer.  Do not print or
attempt to transfer `headline.json`.

---

## Phase 45: exact table token-zero boundary replay

### Goal and hard boundary

Replay one already-proven exact-input table crop through the production
recognition-prefill chain and compare its intermediate tensors directly with a
committed 910B2 reference.  The selected crop is:

```text
case                 table_token0_11_3
source page          OmniDocBench index 11
source image         page-573c437e-c309-4483-a038-ef2f440b104a.png
layout block         3
prompt               Table Recognition:
input tokens         1021
projected tokens     1008
vision route         singleton 4032 -> compiled S4992
text route           singleton 1021 -> compiled packed S1024
```

Phase 39 proved that this crop has the same raw crop hash, all six prepared CPU
tensor hashes, and both execution routes on 910B2 and 310P3, yet diverges at
the first generated token.  Because it is a singleton in both prefill stages,
there are no vision- or text-pack companion/order confounds.

The committed 910B2 reference was generated by behavior commit `228a10a` and
recorded by artifact commit `5e6b758`.  A second warm 910B2 replay was
byte-exact at all eight captured boundaries.  Its first token is ID `101309`,
`<fcel>`, with logit `28.5`; the second token candidate is `<ecel>` at
`17.953125`.

This is a one-crop prefill replay, not an E2E benchmark.  It uses the real
owned layout frontend and the real production CPU preparation, H2D, vision
embedding, compiled vision prefill, projector, multimodal scatter, compiled
packed text prefill, LM head, and fp32 argmax path.  It does **not** launch
decode.  Do not run another crop, full page generation, the evaluator, a
different attention path, or a new graph shape in this phase.  Do not edit
source, commit, push, or create a branch.

### 45.1 Pull and verify the committed reference

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
REPLAY=09_persistent_page_engine/scripts/table_token0_replay.py
REFERENCE_ROOT=tmp/09_persistent_page_engine/910b_phase45_table_token0_228a10a/output
REFERENCE="$REFERENCE_ROOT/tensor_bundle.pt"
REFERENCE_REPORT="$REFERENCE_ROOT/report.json"
ROOT="tmp/09_persistent_page_engine/310p_phase45_table_token0_$(git rev-parse --short HEAD)"

test -f "$REPLAY"
test -f "$REFERENCE"
test -f "$REFERENCE_REPORT"
test ! -e "$ROOT"
mkdir -p "$ROOT"

REFERENCE_SHA="$(sha256sum "$REFERENCE" | awk '{print $1}')"
test "$REFERENCE_SHA" = \
  1152bfbc0cebc97fcc961076d1f5783cfbb022471fe77de26e4e1d2056f42a00
test "$(stat -c %s "$REFERENCE")" -eq 36551201

"$PYTHON_BIN" - "$REFERENCE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
assert d["contract"]["status"] == "PASS"
assert all(d["contract"]["checks"].values())
assert d["first_token"] == 101309
assert d["first_token_decoded"] == "<fcel>"
assert d["vision_route"]["real_vision_tokens"] == 4032
assert d["vision_route"]["physical_vision_tokens"] == 4992
assert d["vision_route"]["packing"] == "single"
assert d["text_prefill_route"]["real_text_tokens"] == 1021
assert d["text_prefill_route"]["physical_text_tokens"] == 1024
assert d["text_prefill_route"]["pack_members"] == 1
print("PHASE45_REFERENCE_CONTRACT: PASS")
PY
```

If any reference check fails, report
`310P PHASE 45 TABLE TOKEN0: REFERENCE_BUNDLE_MISMATCH` and stop.  Do not
substitute another 910B run or regenerate the reference on 310P.

### 45.2 Record environment and warm-cache evidence

Use the previously validated production caches.  This phase adds no compiled
shape.  A missing cache is not permission to silently compile a replacement.

```sh
{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
  sha256sum "$REFERENCE" "$REFERENCE_REPORT"
} 2>&1 | tee "$ROOT/preflight.log"

for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_before.txt"
```

Report this checkpoint immediately in the live progress stream:

```text
310P PHASE 45 PREFLIGHT: PASS
```

Include the exact NPU, software versions, project commit, reference SHA, and
cache file/byte counts.

### 45.3 Run the single production-prefill replay

Path adaptations are allowed only for the dataset image/model roots.  Record
any adaptation.  Do not change the case, dtype, bucket ladders, min-pixels,
packing, PromptFA, decode optimization, or cache length.

```sh
OUTPUT="$ROOT/output"

printf '%q ' timeout --signal=TERM --kill-after=15s 1200 \
  "$PYTHON_BIN" "$REPLAY" \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --reference-bundle "$REFERENCE" \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$OUTPUT" > "$ROOT/command.sh"
printf '\n' >> "$ROOT/command.sh"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1200 \
  "$PYTHON_BIN" "$REPLAY" \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --reference-bundle "$REFERENCE" \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$OUTPUT" 2>&1 | tee "$ROOT/run.log"
run_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$run_exit" > "$ROOT/exit_code.txt"
test "$run_exit" -eq 0
test -f "$OUTPUT/report.json"
test -f "$OUTPUT/tensor_bundle.pt"
```

The script emits flushed progress at layout setup, page preparation,
recognizer setup, and prefill replay.  Long silence before
`prefill_replay_begin` is model/cache setup, not the measured replay.  The
terminal success line must begin `PHASE45 status=PASS`.

The script itself hard-fails unless all Phase-39 input contracts hold,
including these two exact hashes:

```text
crop                    62359561cc5557d9a1972c23f0915ab37c77feb7b6c1aac4f86d320bfde1af2f
prepared CPU inputs     ad925263b4f156d3d11b3367f7fc2b09c77abdbac53cb5ec9bf37007e38c291e
```

### 45.4 Produce the compact comparison

```sh
"$PYTHON_BIN" - "$OUTPUT/report.json" <<'PY' \
  | tee "$ROOT/headline.json"
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
assert d["contract"]["status"] == "PASS"
assert all(d["contract"]["checks"].values())
assert d["comparisons"] is not None
assert d["graph_input_comparisons"] is not None
assert d["route_comparisons"]["vision"]["exact"]
assert d["route_comparisons"]["text_prefill"]["exact"]

boundaries = {}
for name, row in d["comparisons"].items():
    boundaries[name] = {
        "shape_exact": row["shape_exact"],
        "dtype_exact": row.get("dtype_exact"),
        "byte_exact": row.get("byte_exact"),
        "mean_abs": row.get("mean_abs"),
        "max_abs": row.get("max_abs"),
        "rms_abs": row.get("rms_abs"),
        "p95_abs": row.get("p95_abs"),
        "p99_abs": row.get("p99_abs"),
        "relative_l2": row.get("relative_l2"),
        "cosine_similarity": row.get("cosine_similarity"),
    }

out = {
    "contract": d["contract"],
    "input_fingerprints": d["input_fingerprints"],
    "vision_route_comparison": d["route_comparisons"]["vision"],
    "text_route_comparison": d["route_comparisons"]["text_prefill"],
    "graph_input_comparisons": d["graph_input_comparisons"],
    "candidate_first_token": {
        "id": d["decision"]["top1_id"],
        "token": d["decision"]["top1_token"],
        "logit": d["decision"]["top1_logit"],
        "margin": d["decision"]["top1_margin"],
    },
    "same_top1_as_910B": d["decision"]["same_top1"],
    "reference_token_candidate_rank": d["decision"]["reference_token_candidate_rank"],
    "candidate_logit_for_reference_token": d["decision"]["candidate_logit_for_reference_token"],
    "candidate_top1_minus_reference_token": d["decision"]["candidate_top1_minus_reference_token"],
    "candidate_top20": d["decision"]["top20"],
    "boundary_comparisons": boundaries,
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Do not call the first non-byte-exact boundary the cause.  Cross-device
floating-point kernels can differ slightly from the first projection onward.
Instead, report the complete numerical curve and identify where mean/RMS/L2
error materially amplifies and where the top-token ordering changes.

Pay special attention to:

1. CPU crop/prepared hashes and route signatures (must be exact);
2. exact-hash status for static masks, positions, segment IDs, local positions,
   and last-token indices in `graph_input_comparisons`;
3. `vision_embeddings`, compiled `vision_prefill_output`, and
   `projector_output` error growth;
4. multimodal and packed-text graph inputs;
5. the final text hidden-state difference;
6. full logit difference, 310P top-20, the 910B `<fcel>` rank/logit on 310P,
   and the margin by which the 310P winner beats it.

### 45.5 Report and stop

Write `$ROOT/agent_report.md`.  Begin with exactly one classification:

```text
310P PHASE 45 TABLE TOKEN0: PASS_TOKEN0_DIVERGENCE | PASS_TOKEN0_MATCH |
INPUT_CONTRACT_FAILURE | ROUTE_MISMATCH | REFERENCE_BUNDLE_MISMATCH |
RUNTIME_FAILURE
```

For either PASS classification, include:

- exact commit, host/NPU/software, command, cache-hit/compile evidence, wall
  time, and artifact paths;
- every input-contract check and both exact Phase-39 hashes;
- complete vision/text route signatures and equality result;
- every graph-input hash equality result;
- one table, in captured execution order, with shape, dtype, byte-exact,
  mean/max/RMS/p95/p99 absolute error, relative L2, and cosine similarity for
  all eight boundaries;
- the candidate first-token ID/text/logit/margin, the 910B `<fcel>` rank and
  logit on 310P, candidate-winner-minus-`<fcel>` margin, and candidate top 20;
- a short evidence-based statement about where the numerical difference first
  becomes materially larger, without inventing a universal pass threshold;
- `What is proven` and `What remains unresolved`.

Paste `agent_report.md` and `headline.json`.  Do not paste or attempt to push
the 35 MB candidate tensor bundle.  Do not run decode, another crop, or a model
variant.  Then **stop**.

---

## Phase 46: pre-transformer vision divergence localization and corpus correlation

### Goal and interpretation boundary

Phase 45 showed that the selected table crop already differs at the vision
embedding boundary, before the compiled 27-layer vision transformer is called.
It did **not** isolate whether that difference comes from the patch Conv2d, the
interpolated learned position embedding, their fp16 addition, or the separately
constructed vision RoPE tensors.  It also did not establish whether the size of
that early difference predicts which crops later generate different tokens.

Phase 46 answers both questions without running a vision-transformer layer:

1. `exact` mode captures full tensors for the same Phase-45 table crop at every
   pre-transformer boundary;
2. `corpus` mode captures deterministic 8,192-element samples at those
   boundaries for all 106 crops in the fixed seven-page Phase-39 corpus and
   compares the numerical errors with the already-recorded 910B/310P token
   divergence.

The captured controls and outputs are:

```text
prepared pixel_values
  -> reshape                       conv_input
  -> patch Conv2d                  patch_embeddings

learned position_embedding.weight
  -> bilinear interpolate          position_embeddings

patch_embeddings + position_embeddings
  ->                              summed_embeddings

rotary_inv_freq
  -> base angle table              rotary_base_angles
  -> grid-index selection/repeat   rotary_selected_angles
  -> cos / sin                     rope_cos / rope_sin
```

The script uses the real owned layout frontend and production recognizer CPU
preparation, but stops before transformer layer 0.  It does not execute the
vision transformer, projector, text prefill, LM head, or decode.  Do not edit
model code, change dtype, change resize/min-pixels policy, change a route, add a
graph shape, or regenerate the Phase-39 generations.  Do not call the first
non-byte-exact floating-point result a bug merely because it is nonexact;
report its magnitude and how it changes at the next boundary.

The two committed 910B2 reference artifacts were produced by behavior commit
`edc0e49` and recorded by artifact commits `8e5c16d` and `244e1b5`.  Repeated
910B2 validation was byte-exact for every full-tensor exact boundary and for
all seven sampled corpus boundaries on all 106 crops.

### 46.1 Pull and verify both 910B references

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
PROBE=09_persistent_page_engine/scripts/vision_embedding_debug.py
EXACT_REFERENCE=tmp/09_persistent_page_engine/910b_phase46_vision_embedding_exact_edc0e49/output/tensor_bundle.pt
CORPUS_REFERENCE=tmp/09_persistent_page_engine/910b_phase46_vision_embedding_corpus_8e5c16d/output/corpus_bundle.pt
GENERATION_OUTPUT=tmp/09_persistent_page_engine/310p_phase39_accuracy_lab_8e19fdc/310p_e2e/output
ROOT="tmp/09_persistent_page_engine/310p_phase46_vision_embedding_$(git rev-parse --short HEAD)"

test -f "$PROBE"
test -f "$EXACT_REFERENCE"
test -f "$CORPUS_REFERENCE"
test -f "$GENERATION_OUTPUT/recognition_trace.jsonl"
test ! -e "$ROOT"
mkdir -p "$ROOT"

test "$(sha256sum "$EXACT_REFERENCE" | awk '{print $1}')" = \
  2eac0b9f4995b4c24b92d15631f36901a5cec6c0c6f7923875cfbd5d83e1324a
test "$(stat -c %s "$EXACT_REFERENCE")" -eq 39143083
test "$(sha256sum "$CORPUS_REFERENCE" | awk '{print $1}')" = \
  59cb70fd4eb71e9cabce7c750eb06bb3ba72ea2414c0e8afa95cedc9c24936b0
test "$(stat -c %s "$CORPUS_REFERENCE")" -eq 16566651

"$PYTHON_BIN" - "$EXACT_REFERENCE" "$CORPUS_REFERENCE" <<'PY'
import sys
import torch

exact = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
corpus = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert exact["mode"] == "exact"
assert exact["case_id"] == "table_token0_11_3"
assert exact["metadata"]["real_vision_tokens"] == 4032
assert exact["metadata"]["physical_vision_tokens"] == 4992
assert exact["metadata"]["sum_reconstruction_exact"] is True
assert corpus["mode"] == "corpus"
assert corpus["sample_elements"] == 8192
assert len(corpus["records"]) == 106
print("PHASE46_REFERENCE_CONTRACT: PASS")
PY
```

If a reference, Phase-39 generation output, hash, size, or contract is missing,
report `310P PHASE 46: REFERENCE_OR_INPUT_MISMATCH` and stop.  Do not
substitute a different run or regenerate anything.

### 46.2 Record environment and warm-cache evidence

Use the same four already-validated production cache roots as Phase 45.  This
probe does not invoke a compiled graph, but recognizer construction still loads
the production runtime and its cached artifacts.

```sh
{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
  sha256sum "$EXACT_REFERENCE" "$CORPUS_REFERENCE"
} 2>&1 | tee "$ROOT/preflight.log"

for cache in \
  .runtime_cache/310p_phase36_static_actual_b32 \
  .runtime_cache/09_persistent_page_engine_vision_torchair \
  .runtime_cache/09_persistent_page_engine_text_torchair \
  .runtime_cache/310p_text_packed_4789067
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_before.txt"
```

Report `310P PHASE 46 PREFLIGHT: PASS` immediately, with the exact commit,
physical NPU, software versions, both reference hashes, and cache counts.

### 46.3 Exact full-tensor decomposition for the table crop

Run this lane first and report it before starting the corpus lane.  Path
adaptations are allowed only for dataset/model roots and must be recorded.

```sh
EXACT_OUTPUT="$ROOT/exact/output"
mkdir -p "$ROOT/exact"

printf '%q ' timeout --signal=TERM --kill-after=15s 1200 \
  "$PYTHON_BIN" "$PROBE" \
  --mode exact \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --reference-bundle "$EXACT_REFERENCE" \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$EXACT_OUTPUT" > "$ROOT/exact/command.sh"
printf '\n' >> "$ROOT/exact/command.sh"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1200 \
  "$PYTHON_BIN" "$PROBE" \
  --mode exact \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --reference-bundle "$EXACT_REFERENCE" \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$EXACT_OUTPUT" 2>&1 | tee "$ROOT/exact/run.log"
exact_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$exact_exit" > "$ROOT/exact/exit_code.txt"
test "$exact_exit" -eq 0
test -f "$EXACT_OUTPUT/report.json"
test -f "$EXACT_OUTPUT/tensor_bundle.pt"
```

The terminal success line begins `VISION_EMBED status=PASS mode=exact`.  The
script hard-fails unless the Phase-39 crop hash, prepared-input hash, token
counts, 4032-to-4992 route, and exact reconstruction of the embedding sum all
hold.

Create a compact exact headline:

```sh
"$PYTHON_BIN" - "$EXACT_OUTPUT/report.json" <<'PY' \
  | tee "$ROOT/exact/headline.json"
import json
import sys

d = json.load(open(sys.argv[1]))
assert all(d["contract"].values())
assert d["comparisons"] is not None
ordered = [
    "conv_input",
    "patch_embedding_weight",
    "patch_embedding_bias",
    "position_embedding_weight",
    "rotary_inv_freq",
    "patch_embeddings",
    "position_embeddings",
    "summed_embeddings",
    "rotary_base_angles",
    "rotary_selected_angles",
    "rope_cos",
    "rope_sin",
]
print(json.dumps({
    "contract": d["contract"],
    "input_hashes": d["input_hashes"],
    "metadata": d["metadata"],
    "boundaries": {name: d["comparisons"][name] for name in ordered},
}, indent=2, ensure_ascii=False))
PY
```

Before continuing, report one compact table in that order containing shape,
dtype, byte exactness, mean/max/RMS/p95/p99 absolute error, relative L2, and
cosine.  Apply this decision tree mechanically:

1. `conv_input` or any model control weight/frequency is non-byte-exact:
   classify `CONTROL_OR_INPUT_DIFFERENCE`, list it, and stop before corpus;
2. controls are exact and `patch_embeddings` first differs:
   classify the first arithmetic divergence as `PATCH_CONV2D`;
3. patch output is exact and `position_embeddings` first differs:
   classify it as `POSITION_INTERPOLATION`;
4. both components are exact and only `summed_embeddings` differs:
   classify it as `FP16_EMBEDDING_ADD`;
5. embedding sum is exact but the RoPE chain differs:
   name the first of base-angle construction, grid selection/repeat, or
   cos/sin where it differs;
6. more than one independent branch differs, report every branch.  Do not
   force a single-cause label.

This label identifies the earliest observed arithmetic divergence for this
crop, not yet a universal model bug.

### 46.4 Fixed 106-crop correlation lane

Run only after the exact lane has valid exact controls and inputs.  Use the
already-completed Phase-39 310P generation trace.  Do not generate tokens in
this phase.

```sh
CORPUS_OUTPUT="$ROOT/corpus/output"
mkdir -p "$ROOT/corpus"

printf '%q ' timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$PROBE" \
  --mode corpus \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --generation-output "$GENERATION_OUTPUT" \
  --reference-bundle "$CORPUS_REFERENCE" \
  --sample-elements 8192 \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$CORPUS_OUTPUT" > "$ROOT/corpus/command.sh"
printf '\n' >> "$ROOT/corpus/command.sh"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$PROBE" \
  --mode corpus \
  --images-dir /workspace/datasets/OmniDocBench/images \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --generation-output "$GENERATION_OUTPUT" \
  --reference-bundle "$CORPUS_REFERENCE" \
  --sample-elements 8192 \
  --torchair-cache-dir .runtime_cache/310p_phase36_static_actual_b32 \
  --vision-torchair-cache-dir .runtime_cache/09_persistent_page_engine_vision_torchair \
  --text-torchair-cache-dir .runtime_cache/09_persistent_page_engine_text_torchair \
  --text-packed-cache-dir .runtime_cache/310p_text_packed_4789067 \
  --output-dir "$CORPUS_OUTPUT" 2>&1 | tee "$ROOT/corpus/run.log"
corpus_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$corpus_exit" > "$ROOT/corpus/exit_code.txt"
test "$corpus_exit" -eq 0
test -f "$CORPUS_OUTPUT/report.json"
test -f "$CORPUS_OUTPUT/corpus_bundle.pt"
```

The script prints one flushed `capture_crop position=N/106` line per crop and
finishes with `VISION_EMBED status=PASS mode=corpus`.  Create the compact
headline below; it deliberately excludes the large per-crop arrays.

```sh
"$PYTHON_BIN" - "$CORPUS_OUTPUT/report.json" <<'PY' \
  | tee "$ROOT/corpus/headline.json"
import json
import sys

d = json.load(open(sys.argv[1]))
c = d["comparison"]
assert d["requests"] == 106
assert c is not None
print(json.dumps({
    "requests": c["requests"],
    "crop_hash_exact": c["crop_hash_exact"],
    "prepared_inputs_exact": c["prepared_inputs_exact"],
    "route_exact": c["route_exact"],
    "boundary_shapes_exact": c["boundary_shapes_exact"],
    "model_execution_eligible": c["model_execution_eligible"],
    "eligible_token_exact": c["eligible_token_exact"],
    "eligible_token_divergent": c["eligible_token_divergent"],
    "eligible_first_token_divergent": c["eligible_first_token_divergent"],
    "correlation": c["correlation"],
}, indent=2, ensure_ascii=False))
PY
```

Only `model_execution_eligible` crops enter the correlation.  Eligibility
requires exact crop hash, prepared-input hash, vision route, and every boundary
shape.  If fewer than 106 are eligible, identify every excluded page/block and
which contract failed; do not mix those crops into a model-execution claim.

For every one of the seven output boundaries, report:

- Spearman correlation of relative L2 with token-sequence generation distance;
- Spearman correlation of cosine with generation distance;
- ROC-AUC of relative L2 for any token divergence and for first-token
  divergence;
- mean relative L2 split by token-exact/divergent and by first-token
  exact/divergent;
- the ten crops with highest relative L2, including page/block/label,
  generation lengths, first divergence, token-sequence ratio, and numerical
  metrics.

Interpret correlations directionally.  AUC near 0.5 is nonpredictive; above
0.5 means larger early error ranks divergent crops higher.  Do not invent a
universal numeric-correctness threshold, and do not infer causation solely from
correlation.  The important question is whether the same earliest branch found
in the exact crop also varies across the corpus and strongly separates
token-exact from token-divergent generations.

### 46.5 Final report and stop

Write `$ROOT/agent_report.md`.  Begin with one overall classification:

```text
310P PHASE 46: PATCH_CONV2D | POSITION_INTERPOLATION |
FP16_EMBEDDING_ADD | VISION_ROPE | MULTIPLE_PRETRANSFORMER_BRANCHES |
NO_PRETRANSFORMER_DIFFERENCE | CONTROL_OR_INPUT_DIFFERENCE | RUNTIME_FAILURE
```

Include:

- exact commit, host/NPU/software, both exact commands, both reference hashes,
  cache evidence, wall times, and artifact paths;
- the exact-lane contract and complete 12-row numerical table;
- the first differing arithmetic operation and why the preceding boundary is
  ruled out;
- all 106-crop eligibility and generation counts;
- one compact seven-row correlation table with both Spearman values, both AUC
  values, and all four group means;
- the top ten crops for the earliest differing branch and for
  `summed_embeddings`, with generation outcomes;
- an explicit answer to: `Does pre-transformer error magnitude predict later
  generation divergence in this fixed corpus?`;
- `What is proven` and `What remains unresolved`.

Paste `agent_report.md`, `exact/headline.json`, and `corpus/headline.json`.
Do not paste, commit, or push either candidate tensor bundle.  Do not run a
transformer variant, change a model operation, or start a corrective experiment
in this phase.  Then **stop**.

---

## Phase 47: checkpoint provenance, dtype conversion, and embedding-op numerics

### Goal and interpretation boundary

Phase 46 found large differences in the three resident vision-embedding
parameters before any arithmetic was run.  That means `PATCH_CONV2D` was not a
valid root-cause classification yet: different Conv2d inputs (the weights and
bias) naturally produce different outputs.  This phase identifies the earliest
real boundary among:

```text
model/config file bytes
  -> safetensors source tensors (all 620 tensors)
  -> deterministic CPU BF16/FP16/FP32 values
  -> CPU-to-NPU transfer / direct safetensors-to-NPU load
  -> production model loader's resident parameters
  -> Conv2d / bilinear interpolation / add / cos / sin
```

The operation matrix uses the exact fixed Phase-46 crop input, but always loads
the weights from the candidate `--model-dir`.  It runs FP16, BF16, default FP32,
and FP32 with Conv HF32 explicitly disabled.  For each lane it compares the
candidate NPU output both with a candidate-weight CPU-FP32 reference and with
the committed 910B output.  Therefore:

- candidate-versus-CPU metrics isolate local operator numerical quality even
  when the checkpoint differs;
- candidate-versus-910B output metrics are interpretable as an operator
  comparison only after all source/cast/resident inputs are proven exact;
- a non-byte-exact floating-point output is not automatically an accuracy bug.

This probe executes no transformer layer, graph compile/replay, projector, text
prefill, LM head, decode, layout model, or dataset corpus.  It should take only
a few minutes and does not require any compile cache.  Do not download or
replace a model file during this phase even if a difference is found.

The definitive 910B reference was produced by behavior commit `2b26e0a` on an
Ascend 910B2 with Python 3.12.13, torch 2.10.0+cpu, torch_npu 2.10.0.  Its
`model.safetensors` is 1,917,255,968 bytes with SHA-256
`85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db`.
That is also the SHA-256 currently published by the official
`PaddlePaddle/PaddleOCR-VL-1.6` Hugging Face repository.  The local HF metadata
records source revision `66317acc4c9fc17bd154591ce650735cd2855f3e`; the
safetensors metadata itself contains only `format=pt`.  All 620 tensors in the
reference checkpoint are BF16.

### 47.1 Pull and verify the 910B reference

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
PROBE=09_persistent_page_engine/scripts/checkpoint_embedding_ops_probe.py
MODEL_DIR=/workspace/models/PaddleOCR-VL-1.6
PHASE46_EXACT=tmp/09_persistent_page_engine/910b_phase46_vision_embedding_exact_edc0e49/output/tensor_bundle.pt
REFERENCE=tmp/09_persistent_page_engine/910b_phase47_checkpoint_embedding_2b26e0a/output/probe_bundle.pt
ROOT="tmp/09_persistent_page_engine/310p_phase47_checkpoint_embedding_$(git rev-parse --short HEAD)"

test -f "$PROBE"
test -f "$MODEL_DIR/model.safetensors"
test -f "$PHASE46_EXACT"
test -f "$REFERENCE"
test ! -e "$ROOT"
mkdir -p "$ROOT"

test "$(sha256sum "$REFERENCE" | awk '{print $1}')" = \
  9add33b8244ba7377be67151a99a8a34f43396d9a169f4d9f5bc97d4c1a08d0f
test "$(stat -c %s "$REFERENCE")" -eq 4150889

"$PYTHON_BIN" - "$REFERENCE" <<'PY'
import sys
import torch

d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert d["schema_version"] == 2
assert d["kind"] == "checkpoint_embedding_ops_probe"
assert len(d["tensor_manifest"]) == 620
assert set(d["operation_matrix"]) == {
    "fp16", "bf16", "fp32", "fp32_hf32_off"
}
assert all(row["dtype"] == "bfloat16" for row in d["tensor_manifest"].values())
assert d["files"]["model.safetensors"]["sha256"] == \
    "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
print("PHASE47_REFERENCE_CONTRACT: PASS")
PY
```

If the repository is dirty, a required path/hash/size/contract is wrong, or no
NPU is available, report `310P PHASE 47: PREFLIGHT_FAILURE` and stop.  Do not
substitute an earlier `776722d` reference; it lacks the strict-HF32 lane and
full-output comparison.

### 47.2 Record environment and model-source evidence

```sh
{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
print("aclnn_allow_hf32", torch_npu.npu.aclnn.allow_hf32)
print("conv_allow_hf32", torch_npu.npu.conv.allow_hf32)
print("matmul_allow_hf32", torch_npu.npu.matmul.allow_hf32)
PY
  npu-smi info
  ls -ld "$MODEL_DIR"
  ls -l "$MODEL_DIR/model.safetensors"
  readlink -f "$MODEL_DIR" "$MODEL_DIR/model.safetensors"
  stat -c '%n size=%s mtime=%y' "$MODEL_DIR/model.safetensors"
  sha256sum "$MODEL_DIR/model.safetensors" "$REFERENCE" "$PHASE46_EXACT"
  find "$MODEL_DIR/.cache/huggingface/download" -maxdepth 1 \
    -name 'model.safetensors.metadata' -type f -print -exec sed -n '1,5p' {} \;
} 2>&1 | tee "$ROOT/preflight.log"
```

Report `310P PHASE 47 PREFLIGHT: PASS` immediately with the exact commit,
physical NPU, CANN/driver/firmware, Python/torch/torch_npu, model path/resolved
path, model size/hash, and the three lines of HF model metadata if present.  If
that metadata is absent, say provenance metadata is unavailable; do not infer a
source repository from the directory name.

### 47.3 Run the single comparison probe

```sh
OUTPUT="$ROOT/output"

printf '%q ' timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$PROBE" \
  --model-dir "$MODEL_DIR" \
  --phase46-exact-bundle "$PHASE46_EXACT" \
  --reference-bundle "$REFERENCE" \
  --output-dir "$OUTPUT" > "$ROOT/command.sh"
printf '\n' >> "$ROOT/command.sh"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$PROBE" \
  --model-dir "$MODEL_DIR" \
  --phase46-exact-bundle "$PHASE46_EXACT" \
  --reference-bundle "$REFERENCE" \
  --output-dir "$OUTPUT" 2>&1 | tee "$ROOT/run.log"
probe_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$probe_exit" > "$ROOT/exit_code.txt"
test "$probe_exit" -eq 0
test -f "$OUTPUT/report.json"
test -f "$OUTPUT/probe_bundle.pt"
```

Progress is explicit: 8 file hashes, 620 tensors in 50-tensor increments, 9
selected tensor roundtrips, 3 production model loads, and 4 operation lanes.
The terminal success line is `CHECKPOINT_PROBE status=PASS`.

Create a compact headline without copying tensor arrays:

```sh
"$PYTHON_BIN" - "$OUTPUT/report.json" <<'PY' \
  | tee "$ROOT/headline.json"
import json
import sys

d = json.load(open(sys.argv[1]))
c = d["reference_comparison"]
assert c is not None
operations = {}
for lane, rows in c["operation_samples"].items():
    operations[lane] = {}
    for name, row in rows.items():
        operations[lane][name] = {
            "full_output_hash_exact": row["full_output_hash_exact"],
            "candidate_vs_910b": row["sample_comparison"],
            "candidate_vs_cpu_fp32": row["candidate_vs_cpu_fp32"],
            "reference_vs_cpu_fp32": row["reference_vs_cpu_fp32"],
        }
print(json.dumps({
    "classification": d["reference_classification"],
    "candidate_environment": c["candidate_environment"],
    "reference_environment": c["reference_environment"],
    "candidate_checkpoint_provenance": c["candidate_checkpoint_provenance"],
    "reference_checkpoint_provenance": c["reference_checkpoint_provenance"],
    "file_comparisons": c["file_comparisons"],
    "tensor_manifest": c["tensor_manifest"],
    "selected_casts_exact": c["selected_casts_exact"],
    "npu_roundtrips": c["npu_roundtrips"],
    "direct_npu_safetensors": c["direct_npu_safetensors"],
    "production_model_loads": c["production_model_loads"],
    "operations": operations,
}, indent=2, ensure_ascii=False))
PY
```

### 47.4 Mechanical interpretation

Use the probe's `reference_classification.classification` as the headline; it
already applies this first-difference order:

1. `CHECKPOINT_FILE_DIFFERENCE`: model.safetensors size/hash differs;
2. `SOURCE_TENSOR_DIFFERENCE`: file matches but one of the 620 source tensor
   shape/dtype/value hashes differs;
3. `CPU_CAST_DIFFERENCE`: source tensors match but a selected deterministic
   FP16/BF16/FP32 CPU cast differs;
4. `NPU_TRANSFER_DIFFERENCE`: a CPU-to-NPU-to-CPU roundtrip is not byte-exact,
   differs from 910B, or direct safetensors-to-NPU loading differs;
5. `PRODUCTION_LOAD_DIFFERENCE`: source/casts/transfers match but one of nine
   production resident parameters differs in FP16/BF16/FP32;
6. `OPERATOR_OUTPUT_DIFFERENCE`: all inputs match but at least one full output
   hash differs;
7. `EXACT_MATCH`: every boundary including all operation outputs is byte exact.

Independently of that headline, report the complete 4-by-5 operation matrix:

| lane | op boundary | full hash exact vs 910B | candidate-vs-910B max/mean/relative-L2/cosine | candidate-vs-CPU max/mean/relative-L2 | 910B-vs-CPU relative-L2 |
|---|---|---:|---|---|---:|

The five boundaries are patch Conv2d, bilinear position interpolation, their
addition, `cos`, and `sin`.  Explicitly compare default FP32 with
`fp32_hf32_off` on the candidate.  If they are byte exact, HF32 is not affecting
this concrete calculation despite the property value.  If strict FP32 is near
CPU (~1e-7 relative L2) while FP16 is near its normal quantization floor
(roughly 1e-4 to 1e-3), the Phase-46 ~0.1 weight/output divergence cannot be
explained as ordinary FP16-versus-FP32 arithmetic.

If the checkpoint differs, the candidate-versus-910B operator output comparison
is confounded by different weights.  State that plainly and use only each
machine's candidate-versus-CPU metrics to judge its operators.  Do not call
Conv2d inaccurate merely because different checkpoint weights yield different
outputs.

### 47.5 Final report and stop

Write `$ROOT/agent_report.md` containing:

- one headline classification from the list above;
- exact command, commit, host/NPU/software, wall time, and artifact paths;
- 910B and 310P model file size/hash, resolved path, safetensors metadata, and
  HF download revision/hash metadata when present;
- dtype counts for all 620 tensors and source/FP16 manifest exact counts;
- any differing tensor names (all of them if at most 30, otherwise counts by
  top-level subsystem plus the first 30);
- the 9-by-3 CPU cast, NPU roundtrip, direct-NPU, and production-load verdicts;
- the complete 20-row operation table described above;
- an explicit answer to each question:
  1. Are both machines using byte-identical official model weights?
  2. Is there any FP16/BF16/FP32 cast or transfer difference?
  3. Does the production loader alter resident weights on either platform?
  4. Is default-HF32 versus strict-FP32 relevant for this exact Conv2d?
  5. After controlling the weights, is any 310P operator error large enough to
     plausibly explain Phase 46?
- `What is proven` and `What remains unresolved`.

Paste `agent_report.md` and `headline.json`.  Do not paste or commit the
candidate `probe_bundle.pt`, do not replace/download the checkpoint, and do not
run any E2E/compile/corpus experiment.  Then **stop**.

---

## Phase 48: install the correct PaddleOCR-VL-1.6 snapshot and retest one crop

### Root cause and goal

Phase 47 proved that the work server was not running a damaged copy of
PaddleOCR-VL-1.6.  It was running the separate, older official
`PaddlePaddle/PaddleOCR-VL` v1 checkpoint:

```text
old v1 repo       PaddlePaddle/PaddleOCR-VL
old v1 weight SHA 3085f1042e184f68f8a412aa0f64f2c4b8562989598bbfba326aaa11fc685de8

required repo     PaddlePaddle/PaddleOCR-VL-1.6
required revision 66317acc4c9fc17bd154591ce650735cd2855f3e
required SHA      85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db
```

The v1 and v1.6 files have the same byte size and tensor structure, but 614 of
620 tensor values differ.  The tokenizer/processor assets also differ.  This
made every previous 310P-versus-910B accuracy comparison a model-version
comparison, not a hardware comparison.

This phase:

1. preserves `/home/lukaiv/models/PaddleOCR-VL` as the v1 checkpoint;
2. resumably downloads the complete, pinned v1.6 snapshot into a new directory;
3. verifies exact file hashes before loading the model;
4. reruns the Phase-47 source/cast/transfer/operator comparison;
5. only if every pre-operator boundary is exact, recompiles the three prefill
   shapes needed by the single Phase-45 table crop into fresh v1.6 cache roots
   and compares its first token with 910B.

Do not overwrite, rename, or delete the existing v1 model.  Do not use `main`
without a revision pin.  Do not run a page prefix, evaluator, or full E2E test
in this phase.

### 48.1 Pull, establish paths, and check disk

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
OLD_MODEL=/home/lukaiv/models/PaddleOCR-VL
MODEL_V16=/home/lukaiv/models/PaddleOCR-VL-1.6
DOWNLOAD_DIR=/home/lukaiv/models/PaddleOCR-VL-1.6.download
HF_REPO=PaddlePaddle/PaddleOCR-VL-1.6
HF_REVISION=66317acc4c9fc17bd154591ce650735cd2855f3e
ROOT="tmp/09_persistent_page_engine/310p_phase48_v16_$(git rev-parse --short HEAD)"

PROBE=09_persistent_page_engine/scripts/checkpoint_embedding_ops_probe.py
REPLAY=09_persistent_page_engine/scripts/table_token0_replay.py
PHASE46_EXACT=tmp/09_persistent_page_engine/910b_phase46_vision_embedding_exact_edc0e49/output/tensor_bundle.pt
PHASE47_REFERENCE=tmp/09_persistent_page_engine/910b_phase47_checkpoint_embedding_2b26e0a/output/probe_bundle.pt
PHASE45_REFERENCE=tmp/09_persistent_page_engine/910b_phase45_table_token0_228a10a/output/tensor_bundle.pt

test -f "$OLD_MODEL/model.safetensors"
test "$(sha256sum "$OLD_MODEL/model.safetensors" | awk '{print $1}')" = \
  3085f1042e184f68f8a412aa0f64f2c4b8562989598bbfba326aaa11fc685de8
test -f "$PROBE"
test -f "$REPLAY"
test -f "$PHASE46_EXACT"
test -f "$PHASE47_REFERENCE"
test -f "$PHASE45_REFERENCE"
test ! -e "$ROOT"
mkdir -p "$ROOT"

{
  date -Is
  hostname
  git rev-parse HEAD
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import huggingface_hub
import torch, torch_npu
print("huggingface_hub", huggingface_hub.__version__)
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
  df -h /home/lukaiv/models "$WORK_SERVER_REPO"
  du -sh "$OLD_MODEL"
  sha256sum "$OLD_MODEL/model.safetensors" "$PHASE47_REFERENCE"
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk /home/lukaiv/models | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 6291456
```

Require at least 6 GiB free because the download/cache may temporarily hold
more than one copy of the 1.92 GB weights.  If `huggingface_hub` is missing, the
old-model hash is unexpected, the repository is dirty, no NPU is free, or disk
is insufficient, report `310P PHASE 48: PREFLIGHT_FAILURE` and stop.  Do not
install packages or delete files without Luka's approval.

Report `310P PHASE 48 PREFLIGHT: PASS` immediately with commit, NPU/software,
old-model hash, free disk, and whether `$MODEL_V16` or `$DOWNLOAD_DIR` already
exists.

### 48.2 Download the pinned snapshot without touching v1

If `$MODEL_V16` already exists, do not download into or overwrite it.  Go
straight to the hash verification in 48.3.  A nonexact existing directory is a
conflict to report, not permission to delete it.

Otherwise, resume into `$DOWNLOAD_DIR`.  Re-running this same command after a
network interruption resumes completed files:

```sh
mkdir -p "$DOWNLOAD_DIR"

set +e
HF_HUB_DOWNLOAD_TIMEOUT=600 \
HF_HUB_ETAG_TIMEOUT=60 \
PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" - "$HF_REPO" "$HF_REVISION" "$DOWNLOAD_DIR" <<'PY' \
  2>&1 | tee "$ROOT/download.log"
import sys
from huggingface_hub import snapshot_download

repo_id, revision, local_dir = sys.argv[1:]
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    max_workers=2,
)
print("PHASE48_DOWNLOAD_COMPLETE", path, flush=True)
PY
download_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$download_exit" > "$ROOT/download_exit_code.txt"
```

If the default Hugging Face endpoint fails because of the work-server proxy,
rerun the exact same command once with this additional environment variable:

```sh
HF_ENDPOINT=https://hf-mirror.com
```

Do not change repo/revision or mix files manually.  The mirror is acceptable
only because 48.3 verifies every behavior-relevant file by SHA-256.  If both
attempts fail, preserve `$DOWNLOAD_DIR` for resumability, report the last real
network error and downloaded byte count, then stop.

### 48.3 Verify the complete v1.6 snapshot and promote it atomically

Set the candidate directory without modifying either model:

```sh
if test -d "$MODEL_V16"; then
  CANDIDATE_MODEL="$MODEL_V16"
else
  test "$download_exit" -eq 0
  CANDIDATE_MODEL="$DOWNLOAD_DIR"
fi

"$PYTHON_BIN" - "$CANDIDATE_MODEL" <<'PY' \
  | tee "$ROOT/model_hash_verification.json"
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = {
    "model.safetensors": "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
    "config.json": "ce7f4565f8b1db78532ad5d1b9ebe55c2139d49bd4cb04778b580a08a598f171",
    "preprocessor_config.json": "111872ab1e8bb7fd040ac5087bfced7ab8f011f02139b088cba294964c3b1d0e",
    "processor_config.json": "1568858960a9760c54431dae693a6152e601ff55cdf6d2eab97a4a99958faea0",
    "tokenizer.json": "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d",
    "tokenizer.model": "34ef7db83df785924fb83d7b887b6e822a031c56e15cff40aaf9b982988180df",
    "tokenizer_config.json": "1f979337347cc0cb72a6282d8a23ed183539aa81a87a906f022aee2bab83c7c5",
    "generation_config.json": "a6701d78ab3b4d972307cdec3b69d4c13f46e0d5140514f50ab7d84259324b94",
}

rows = {}
for name, wanted in expected.items():
    path = root / name
    if not path.is_file():
        rows[name] = {"exists": False, "expected": wanted}
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    rows[name] = {
        "exists": True,
        "size": path.stat().st_size,
        "expected": wanted,
        "actual": actual,
        "exact": actual == wanted,
    }
print(json.dumps({"root": str(root), "files": rows}, indent=2))
assert all(row.get("exact") for row in rows.values()), rows
print("PHASE48_MODEL_FILES: EXACT")
PY
```

These are the hashes measured from the committed 910B v1.6 reference.  All
eight must match; a correct weights file combined with stale v1 tokenizer or
processor assets is not a valid installation.

If `$MODEL_V16` did not exist and verification passed, promote the complete
directory with one rename:

```sh
if test "$CANDIDATE_MODEL" = "$DOWNLOAD_DIR"; then
  test ! -e "$MODEL_V16"
  mv "$DOWNLOAD_DIR" "$MODEL_V16"
fi
test -f "$MODEL_V16/model.safetensors"
test "$(sha256sum "$MODEL_V16/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db
```

Report `310P PHASE 48 MODEL INSTALL: PASS` with the final path, eight hashes,
HF metadata revision/hash when present, download wall time, and disk usage.

### 48.4 Rerun Phase 47 against byte-identical v1.6 inputs

```sh
CHECK_ROOT="$ROOT/checkpoint"
mkdir -p "$CHECK_ROOT"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$PROBE" \
  --model-dir "$MODEL_V16" \
  --phase46-exact-bundle "$PHASE46_EXACT" \
  --reference-bundle "$PHASE47_REFERENCE" \
  --output-dir "$CHECK_ROOT/output" 2>&1 | tee "$CHECK_ROOT/run.log"
check_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$check_exit" > "$CHECK_ROOT/exit_code.txt"
test "$check_exit" -eq 0

"$PYTHON_BIN" - "$CHECK_ROOT/output/report.json" <<'PY' \
  | tee "$CHECK_ROOT/gate.json"
import json
import sys

d = json.load(open(sys.argv[1]))
c = d["reference_comparison"]
k = d["reference_classification"]
assert c is not None and k is not None
assert k["model_file_exact"]
assert not k["config_file_differences"]
assert k["source_manifest_exact"]
assert k["selected_cpu_casts_exact"]
assert k["npu_roundtrips_exact"]
assert k["direct_npu_safetensors_exact"]
assert k["production_model_loads_exact"]
assert c["tensor_manifest"]["source_exact"] == 620
assert c["tensor_manifest"]["fp16_exact"] == 620
assert not c["tensor_manifest"]["different"]
assert all(
    exact
    for row in c["selected_casts_exact"].values()
    for exact in row.values()
)
print(json.dumps({
    "classification": k,
    "tensor_manifest": c["tensor_manifest"],
    "file_comparisons": c["file_comparisons"],
    "operation_samples": c["operation_samples"],
}, indent=2))
print("PHASE48_CHECKPOINT_GATE: PASS")
PY
```

The likely overall classification is `OPERATOR_OUTPUT_DIFFERENCE`, because
cross-hardware floating-point outputs need not be byte-exact.  That is fine.
The hard gate is that every checkpoint/source/cast/transfer/production-load
field above is exact.  If any hard-gate assertion fails, report the first field
and stop before compilation.

Report the complete 20-row FP16/BF16/FP32/strict-FP32 operation matrix.  Do not
repeat Phase 47's incorrect claim that a candidate-versus-its-own-CPU Conv2d
error is caused by cross-machine weight differences.  Those weights cancel in
that comparison.  The already-observed 310P FP32 Conv2d relative-L2 around
`2.9e-4` is a real reduced-precision kernel characteristic; the production
FP16 result around `2.9e-4` remains the relevant path.

### 48.5 Fresh-cache one-crop token-zero correctness replay

Only run this section after `PHASE48_CHECKPOINT_GATE: PASS`.

The v1 and v1.6 `config.json` files are byte-identical, and the existing
TorchAir cache keys include the config hash rather than the checkpoint hash.
Therefore do not reuse the v1 cache roots for this correctness test.  Use these
new roots:

```sh
DECODE_CACHE=.runtime_cache/310p_phase48_v16_decode
VISION_CACHE=.runtime_cache/310p_phase48_v16_vision
TEXT_CACHE=.runtime_cache/310p_phase48_v16_text
PACKED_CACHE=.runtime_cache/310p_phase48_v16_text_packed

for cache in "$DECODE_CACHE" "$VISION_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"; do
  test ! -e "$cache"
done

REPLAY_ROOT="$ROOT/token0"
mkdir -p "$REPLAY_ROOT"

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=15s 1800 \
  "$PYTHON_BIN" "$REPLAY" \
  --images-dir /home/lukaiv/datasets/OmniDocBench/images \
  --layout-model /home/lukaiv/models/PP-DocLayoutV3_safetensors \
  --recognizer-model "$MODEL_V16" \
  --reference-bundle "$PHASE45_REFERENCE" \
  --torchair-cache-dir "$DECODE_CACHE" \
  --vision-torchair-cache-dir "$VISION_CACHE" \
  --text-torchair-cache-dir "$TEXT_CACHE" \
  --text-packed-cache-dir "$PACKED_CACHE" \
  --vision-buckets 4992 \
  --text-buckets 1024 \
  --text-pack-buckets 1024 \
  --output-dir "$REPLAY_ROOT/output" 2>&1 | tee "$REPLAY_ROOT/run.log"
replay_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$replay_exit" > "$REPLAY_ROOT/exit_code.txt"
test "$replay_exit" -eq 0
test -f "$REPLAY_ROOT/output/report.json"
```

The three bucket overrides are diagnostic-only and preserve the exact
production operations for this crop; they merely avoid compiling 36 unused
vision/text buckets.  The crop still executes singleton vision S4992 and
singleton packed-text S1024.  Fresh cache roots prevent any v1 graph artifact
from entering the result.

Extract the compact correctness result:

```sh
"$PYTHON_BIN" - "$REPLAY_ROOT/output/report.json" <<'PY' \
  | tee "$REPLAY_ROOT/headline.json"
import json
import sys

d = json.load(open(sys.argv[1]))
assert d["contract"]["status"] == "PASS"
assert all(d["contract"]["checks"].values())
assert d["route_comparisons"]["vision"]["exact"]
assert d["route_comparisons"]["text_prefill"]["exact"]
print(json.dumps({
    "contract": d["contract"],
    "input_fingerprints": d["input_fingerprints"],
    "route_comparisons": d["route_comparisons"],
    "graph_input_comparisons": d["graph_input_comparisons"],
    "decision": d["decision"],
    "comparisons": d["comparisons"],
}, indent=2, ensure_ascii=False))
PY
```

Report whether the candidate first token is the 910B token ID `101309`
(`<fcel>`), its logit/margin, the 910B token's rank if it is not top-1, and the
complete numerical curve for all eight boundaries.  Do not demand byte-exact
NPU activations; identify error growth and the final top-token decision.

### 48.6 Final report and stop

Write `$ROOT/agent_report.md` with one headline:

```text
310P PHASE 48: V16_TOKEN0_MATCH | V16_TOKEN0_DIVERGENCE |
V16_CHECKPOINT_GATE_FAILURE | DOWNLOAD_FAILURE | RUNTIME_FAILURE
```

Include:

- exact pinned HF repo/revision, final path, all eight file hashes, and proof
  that the old v1 directory remains unchanged;
- download method, attempts, wall time, byte count, and whether a mirror was
  required;
- Phase-47 gate results: 620/620 source and FP16 tensors exact, all selected
  casts/transfers/direct loads/production loads, and all 20 operator rows;
- fresh-cache compile evidence and cache sizes;
- the exact one-crop input/route contracts, eight-boundary comparison, and
  first-token decision;
- `What is proven` and `What remains unresolved`.

Paste `agent_report.md`, `checkpoint/gate.json`, and `token0/headline.json`.
Do not commit/push large model or runtime artifacts.  Do not start a 32-page or
larger run.  Then **stop**.

---

## Phase 49: full OmniDocBench v1.6 rerun with the corrected v1.6 checkpoint

### Goal and fixed comparison boundary

Phase 48 proved that the old work-server run used the official v1 checkpoint,
not the byte-identical v1.6 checkpoint used by the 910B reference.  It then
proved that this checkpoint mismatch was the sole cause of the Phase-45
token-zero divergence.  Now repeat the complete 1,651-page production run with
the verified v1.6 snapshot, run the guarded official OmniDocBench evaluator,
and compare performance, generation streams, and quality directly with the
committed 910B2 full reference.

This is the corrected equivalent of Phase 41.  Do not optimize or change the
pipeline.  In particular:

- use `/home/lukaiv/models/PaddleOCR-VL-1.6`, whose weight SHA is fixed below;
- do not use `/home/lukaiv/models/PaddleOCR-VL`;
- do not pass the old `--max-new-tokens 2808`; the current default is the
  4,096-token secondary safety ceiling, while each request stops at EOS or its
  own remaining KV capacity;
- use all-pages-first layout, B32 static-actual GQA decode, KV4096, the fixed
  ten-bucket 310P vision ladder, production-group text packing, and no timeline;
- use fresh v1.6 cache roots.  The v1 and v1.6 `config.json` files are identical,
  so reusing the old cache roots risks silently loading graphs compiled with v1
  weight constants;
- evaluate all 1,651 pages.  Do not exclude hard pages or substitute a filtered
  ground truth.

The committed 910B reference is:

```text
run summary
  tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/output/run_summary.json
evaluation
  tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/evaluation/work/result/
generation bundle
  tmp/09_persistent_page_engine/910b_generation_difference_reference_ab00d1f/
  omnidocbench_v1_6_910b2_full_8634d3a.gdatlas.zip
```

The 910B headline is 1,651 pages in 1,055.523 s, or 1.56415 pages/s.  Its
official metrics are text Edit `0.0408832`, formula Edit `0.0868281`, table Edit
`0.0569611`, page TEDS `0.9434504`, structure TEDS `0.9676981`, and reading-order
Edit `0.1380545`.  The scripts below read the committed files rather than
trusting these rounded values.

### 49.1 Pull and prove the corrected checkpoint

The work-server agent remains pull-only.  Do not edit tracked files, create a
branch, commit, or push.

```sh
set -o pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
BATCHED_VISION=09_persistent_page_engine/scripts/vision_lab_batched_packed.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
OLD_MODEL=/home/lukaiv/models/PaddleOCR-VL
DATASET_JSON=/home/lukaiv/datasets/OmniDocBench/OmniDocBench.json
IMAGES_DIR=/home/lukaiv/datasets/OmniDocBench/images
LAYOUT_MODEL=/home/lukaiv/models/PP-DocLayoutV3_safetensors
ROOT="tmp/09_persistent_page_engine/310p_phase49_v16_full_$(git rev-parse --short HEAD)"
PREP="$ROOT/cache_prepare"
LANE="$ROOT/e2e"
OUTPUT="$LANE/output"
EVAL="$ROOT/evaluation"

REFERENCE_RUN=tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/output/run_summary.json
REFERENCE_RESULT=tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/evaluation/work/result
REFERENCE_METRIC="$REFERENCE_RESULT/predictions_quick_match_metric_result.json"
REFERENCE_BUNDLE=tmp/09_persistent_page_engine/910b_generation_difference_reference_ab00d1f/omnidocbench_v1_6_910b2_full_8634d3a.gdatlas.zip

DECODE_CACHE=.runtime_cache/310p_phase49_v16_decode_b32_k4096
VISION_CACHE=.runtime_cache/310p_phase49_v16_vision_b1
BATCHED_CACHE=.runtime_cache/310p_phase49_v16_vision_batched
TEXT_CACHE=.runtime_cache/310p_phase49_v16_text
PACKED_CACHE=.runtime_cache/310p_phase49_v16_text_packed

test ! -e "$ROOT"
mkdir -p "$PREP" "$LANE" "$EVAL/work"
test -x "$PYTHON_BIN"
test -x "$EVAL_PYTHON"
test -f "$E2E"
test -f "$EVAL_WRAPPER"
test -f "$BATCHED_VISION"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -f "$MODEL/model.safetensors"
test -f "$OLD_MODEL/model.safetensors"
test -f "$REFERENCE_RUN"
test -f "$REFERENCE_METRIC"
test -f "$REFERENCE_BUNDLE"

test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db
test "$(sha256sum "$OLD_MODEL/model.safetensors" | awk '{print $1}')" = \
  3085f1042e184f68f8a412aa0f64f2c4b8562989598bbfba326aaa11fc685de8
test "$(sha256sum "$REFERENCE_BUNDLE" | awk '{print $1}')" = \
  a1c2ec99b8aa2b0a18f26cedc9fa7383aa42c78620224aed497035b46bb1ba84

PHASE48_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase48_v16_*' | sort | tail -n 1)"
test -n "$PHASE48_ROOT"
test -f "$PHASE48_ROOT/agent_report.md"
rg -n '310P PHASE 48: V16_TOKEN0_MATCH' "$PHASE48_ROOT/agent_report.md"

for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  test ! -e "$cache"
done

{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
  npu-smi info
  sha256sum \
    "$MODEL/model.safetensors" \
    "$OLD_MODEL/model.safetensors" \
    "$REFERENCE_RUN" "$REFERENCE_METRIC" "$REFERENCE_BUNDLE"
  df -h "$WORK_SERVER_REPO" /home/lukaiv/models
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 10485760
```

If the v1.6 hash, Phase-48 token match, clean checkout, reference artifact, free
NPU, or 10-GiB disk-space check fails, report
`310P PHASE 49 PREFLIGHT: FAILURE` and
stop.  Never fall back to the old checkpoint.  Report
`310P PHASE 49 PREFLIGHT: PASS` immediately with the exact commit, device,
software, v1.6 hash, and free disk.

### 49.2 Prepare fresh v1.6 graph caches outside the measured run

First compile the two profile-guided batched-vision graphs.  The production
runtime requires these caches to be warm rather than compiling them itself:

```sh
SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 7200 \
  "$PYTHON_BIN" "$BATCHED_VISION" \
  --corpus tmp/09_persistent_page_engine/vision_lab/corpus_recognition_trace_variants.json \
  --model "$MODEL" \
  --variant min_pixels_28224 \
  --cache-dir "$BATCHED_CACHE" \
  --shape 2x3072 --shape 4x1024 \
  --warmup 0 --repeats 1 \
  --output "$PREP/batched_vision.json" \
  2>&1 | tee "$PREP/batched_vision.log"
batched_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$batched_exit" >"$PREP/batched_vision_exit_code.txt"
test "$batched_exit" -eq 0
```

Define the exact production argument vector once.  Both the warmup and full
run must use this unchanged vector:

```sh
PRODUCTION_ARGS=(
  "$E2E"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL"
  --recognizer-model "$MODEL"
  --batch-size 32 --cache-length 4096
  --preprocessor-min-pixels 28224
  --decode-backend torchair
  --decode-optimization combined_apply_static_actual
  --torchair-cache-dir "$DECODE_CACHE"
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992
  --vision-torchair-cache-dir "$VISION_CACHE"
  --vision-batched-cache-dir "$BATCHED_CACHE"
  --vision-promptfa-align-128 --vision-padding bucket
  --vision-packing greedy --vision-pack-target 1920
  --vision-router-lookahead 32
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312
  --text-packing production_group
  --text-pack-buckets 128,256,512,1024
  --text-pack-max-members 32
  --text-torchair-cache-dir "$TEXT_CACHE"
  --text-packed-cache-dir "$PACKED_CACHE"
  --layout-device npu --no-layout-graph-capture
  --preprocess-all-pages-first --no-timeline
)
```

Run one page to make every configured singleton-vision, text-prefill,
packed-text, and decode graph warm.  Compilation belongs to this preparation
lane, not to the full-run result:

```sh
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1 --output-dir "$PREP/one_page_output" \
  >"$PREP/one_page_command.sh"
printf '\n' >>"$PREP/one_page_command.sh"

SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 7200 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1 --output-dir "$PREP/one_page_output" \
  2>&1 | tee "$PREP/one_page.log"
prep_exit="${PIPESTATUS[0]}"
prep_wall_s="$SECONDS"
set -e
printf '%s\n' "$prep_exit" >"$PREP/one_page_exit_code.txt"
printf '%s\n' "$prep_wall_s" >"$PREP/one_page_wall_s.txt"
test "$prep_exit" -eq 0

for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_before_full.txt"
```

Report `310P PHASE 49 CACHE PREP: PASS` with compile wall time, each cache's
file count/bytes, and the compile/cache metadata from the one-page
`run_summary.json`.  If preparation fails or any cache is empty, report the
first causal error and stop.  Do not start the 1,651-page run with a partial
cache set.

### 49.3 Run all 1,651 pages and compare exact crop generations

```sh
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 14400 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
run_wall_s="$SECONDS"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
printf '%s\n' "$run_wall_s" >"$LANE/launcher_wall_s.txt"
test "$run_exit" -eq 0
```

The foreground log is the authoritative progress stream.  In a second shell,
the user may inspect it with:

```sh
tail -f "$LANE/run.log"
```

Validate and print the performance headline immediately, before evaluation:

```sh
"$PYTHON_BIN" - "$OUTPUT/run_summary.json" <<'PY' \
  | tee "$LANE/compact_summary.json"
import json, sys
d = json.load(open(sys.argv[1]))
r = d["recognition"]
s = r["device_stage_s"]
assert d["offset"] == 0 and d["count"] == 1651
assert d["result_count"] == 1651 and d["prediction_count"] == 1651
assert d["configuration"]["page_preprocessing_mode"] == "all_before_recognition"
assert d["configuration"]["cache_length"] == 4096
assert d["configuration"]["max_new_tokens"] == 4096
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full"}
out = {
    "setup_s": d["setup_s"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "pages_per_s": d["pages_per_s"],
    "s_per_page": d["s_per_page"],
    "layout_s": d["layout_frontend"]["stage_s"]["page_total_s"],
    "ocr_scheduler_wall_s": r["run_scoped_scheduler_wall_s"],
    "requests": r["requests"],
    "stop_reasons": r["stop_reason_counts"],
    "vision": {"real": r["real_vision_tokens"], "physical": r["physical_vision_tokens"], "s": s["vision_prefill"]},
    "text": {"real": r["real_text_tokens"], "physical": r["physical_text_tokens"], "s": s["text_prefill"]},
    "decode": {"effective": r["effective_decode_tokens"], "raw": r["raw_decode_token_slots"], "s": r["decode_wall_s"]},
}
for key in ("vision", "text"):
    out[key]["real_tps"] = out[key]["real"] / out[key]["s"]
    out[key]["physical_tps"] = out[key]["physical"] / out[key]["s"]
out["decode"]["effective_tps"] = out["decode"]["effective"] / out["decode"]["s"]
out["decode"]["raw_tps"] = out["decode"]["raw"] / out["decode"]["s"]
print(json.dumps(out, indent=2))
PY
```

Now compare every shared request's exact token stream with the committed 910B
trace.  This is deliberately independent of OmniDocBench normalization:

```sh
unzip -p "$REFERENCE_BUNDLE" recognition_trace.jsonl \
  >"$ROOT/910b_recognition_trace.jsonl"

"$PYTHON_BIN" - \
  "$ROOT/910b_recognition_trace.jsonl" \
  "$OUTPUT/recognition_trace.jsonl" <<'PY' \
  | tee "$LANE/generation_comparison.json"
import json, sys

def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    result = {str(row["request_id"]): row for row in rows}
    assert len(result) == len(rows), "duplicate request_id"
    return result

ref = load(sys.argv[1])
candidate = load(sys.argv[2])
shared = sorted(set(ref) & set(candidate))
token_exact = []
text_exact = []
compact_exact = []
length_deltas = []
for key in shared:
    left, right = ref[key], candidate[key]
    lt = [int(x) for x in left.get("token_ids", [])]
    rt = [int(x) for x in right.get("token_ids", [])]
    token_exact.append(lt == rt)
    text_exact.append(left.get("text") == right.get("text"))
    compact_exact.append(
        "".join(str(left.get("text", "")).split()) ==
        "".join(str(right.get("text", "")).split())
    )
    if lt != rt:
        length_deltas.append({
            "request_id": key,
            "reference_tokens": len(lt),
            "candidate_tokens": len(rt),
            "absolute_delta": abs(len(rt) - len(lt)),
        })
out = {
    "reference_requests": len(ref),
    "candidate_requests": len(candidate),
    "shared_requests": len(shared),
    "reference_only": sorted(set(ref) - set(candidate)),
    "candidate_only": sorted(set(candidate) - set(ref)),
    "token_exact": sum(token_exact),
    "token_different": len(shared) - sum(token_exact),
    "text_exact": sum(text_exact),
    "whitespace_compact_text_exact": sum(compact_exact),
    "worst_30_token_length_deltas": sorted(
        length_deltas,
        key=lambda row: (-row["absolute_delta"], row["request_id"]),
    )[:30],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Report `310P PHASE 49 E2E: PASS` immediately with the compact timing/tok-s
summary and exact generation counts.  Do not interpret a token difference as
wrong until the official evaluation finishes.

Record cache state again.  Any new graph directory during the full run makes
the timing compile-contaminated and must be called out:

```sh
for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  printf '%s\tfiles=%s\tbytes=%s\n' \
    "$cache" \
    "$(find "$cache" -type f | wc -l)" \
    "$(du -sb "$cache" | cut -f1)"
done | tee "$ROOT/cache_after_full.txt"
diff -u "$ROOT/cache_before_full.txt" "$ROOT/cache_after_full.txt" \
  | tee "$ROOT/cache_diff.txt" || true
```

### 49.4 Run the guarded official evaluator

Locate the evaluator checkout already established in Phases 41/42.  Do not
clone or change it:

```sh
EVALUATOR_ROOT=
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  /home/lukaiv/repos/OmniDocBench_eval \
  "$HOME/repos/OmniDocBench_eval" \
  "$HOME/OmniDocBench_eval" \
  "$HOME/OmniDocBench"
do
  if test -f "$candidate/pdf_validation.py"; then
    EVALUATOR_ROOT="$candidate"
    break
  fi
done
test -n "$EVALUATOR_ROOT"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

cat >"$EVAL/work/config.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 12
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $WORK_SERVER_REPO/$OUTPUT/OmniDocBench_subset.json
    prediction:
      data_path: $WORK_SERVER_REPO/$OUTPUT/predictions
    match_method: quick_match
    match_workers: 12
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

cd "$EVAL/work"
ulimit -n 65536
SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
  "$EVAL_PYTHON" "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --config config.yaml \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 12 --teds-workers 12 \
  --page-timeout-sec 120 \
  --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30 \
  --teds-timeout-sec 120 \
  2>&1 | tee evaluation.log
eval_exit="${PIPESTATUS[0]}"
eval_wall_s="$SECONDS"
set -e
printf '%s\n' "$eval_exit" >../exit_code.txt
printf '%s\n' "$eval_wall_s" >../wall_s.txt
cd "$WORK_SERVER_REPO"
test "$eval_exit" -eq 0
```

This wrapper uses process-isolated page matching and the corrected
parent-owned TEDS process scheduler.  It should not reproduce the old nested
thread/process `can only join a started process` failure.  A bounded timeout is
valid evidence; a worker lifecycle error is not.  Do not exclude pages or run
the old direct `pdf_validation.py` command if this fails.

Validate the complete evaluator result:

```sh
RESULT="$EVAL/work/result"
METRIC="$RESULT/predictions_quick_match_metric_result.json"
EVAL_SUMMARY="$RESULT/predictions_quick_match_run_summary.json"
STAGE="$RESULT/predictions_quick_match_stage_execution.json"
test -f "$METRIC"
test -f "$EVAL_SUMMARY"
test -f "$STAGE"

"$EVAL_PYTHON" - "$METRIC" "$EVAL_SUMMARY" <<'PY' \
  | tee "$EVAL/compact_eval_summary.json"
import json, sys
m = json.load(open(sys.argv[1]))
s = json.load(open(sys.argv[2]))
stage = s["stage_execution"]
assert stage["page_match"]["page_count"] == 1651
out = {
    "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
    "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
    "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
    "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    "page_denominators": s["page_denominators"],
    "page_match": stage["page_match"],
    "table_TEDS_execution": stage["metrics"]["table"]["TEDS"],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Report `310P PHASE 49 EVALUATION: PASS` with all six official metrics,
denominators, fallback/timeouts/errors, and evaluation wall time.

### 49.5 Produce the direct 310P-versus-910B comparison and stop

Use the exact committed 910B JSON as the comparison source:

```sh
"$PYTHON_BIN" - \
  "$REFERENCE_RUN" "$REFERENCE_METRIC" \
  "$OUTPUT/run_summary.json" "$METRIC" <<'PY' \
  | tee "$ROOT/head_to_head.json"
import json, sys

ref_run, ref_metric, run, metric = [json.load(open(path)) for path in sys.argv[1:]]

def perf(d):
    r = d["recognition"]
    s = r["device_stage_s"]
    return {
        "pages_per_s": d["pages_per_s"],
        "pipeline_e2e_s": d["pipeline_e2e_s"],
        "layout_pages_per_s": 1651 / d["layout_frontend"]["stage_s"]["page_total_s"],
        "vision_physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
        "text_physical_tps": r["physical_text_tokens"] / s["text_prefill"],
        "decode_raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
        "requests": r["requests"],
        "generated_including_eos": r["generated_tokens_including_eos"],
        "stop_reasons": r["stop_reason_counts"],
    }

def quality(m):
    return {
        "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
        "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
        "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
        "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    }

rp, cp = perf(ref_run), perf(run)
rq, cq = quality(ref_metric), quality(metric)
out = {
    "performance": {},
    "quality": {},
}
for name in rp:
    row = {"910B2": rp[name], "310P3": cp[name]}
    if isinstance(rp[name], (int, float)) and isinstance(cp[name], (int, float)):
        row["310P_minus_910B"] = cp[name] - rp[name]
        row["310P_over_910B"] = cp[name] / rp[name] if rp[name] else None
    out["performance"][name] = row
for name in rq:
    delta = cq[name] - rq[name]
    higher_is_better = "TEDS" in name
    out["quality"][name] = {
        "910B2": rq[name],
        "310P3": cq[name],
        "310P_minus_910B": delta,
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "310P_better": delta > 0 if higher_is_better else delta < 0,
    }
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 49 FULL V1.6: PASS | PREFLIGHT_FAILURE | CACHE_PREP_FAILURE |
E2E_FAILURE | COMPILE_CONTAMINATED | EVALUATOR_FAILURE | DATASET_MISMATCH
```

Include:

1. project/evaluator commits, host, exact NPU and software, dataset paths, and
   the v1.6 weights hash;
2. proof the old v1 checkpoint was not loaded and all five cache roots were
   fresh and v1.6-specific;
3. cache-preparation wall time and cache before/after evidence;
4. complete E2E stage/tok-s metrics, packing histograms, stop reasons, requests,
   and setup versus measured pipeline time;
5. exact shared token-stream/text comparison against 910B, including unmatched
   request IDs and the worst length deltas;
6. all six official metrics, denominators, page fallbacks, TEDS timeouts/errors,
   and evaluator wall time;
7. the full `head_to_head.json`, with desired metric direction stated correctly;
8. concise `What is proven`, `What remains unresolved`, and the first causal
   error if anything failed.

Paste `agent_report.md`, `e2e/compact_summary.json`,
`e2e/generation_comparison.json`, `evaluation/compact_eval_summary.json`, and
`head_to_head.json`.  Keep large outputs/caches local; do not commit or push
them.  Do not begin another accuracy-debugging or optimization phase.  Then
**stop**.

## Phase 50: optimized B1 vision-transformer sequence sweep on 310P

### Goal and fixed experiment boundary

Reproduce the committed 910B2 B1 sequence sweep on one Atlas 310P3.  This is a
vision-transformer throughput experiment only.  Do not run layout, crop
preprocessing, the projector, text prefill, decode, or OmniDocBench evaluation.

The measured boundary is the exact compiled `VisionPrefillStage`: all 27 vision
encoder layers, production PromptFA with runtime D72-to-D80 head padding, RoPE,
LayerNorms and residuals, Q/K/V/output projections, FC1/GELU/FC2, and the final
post-LayerNorm.  Every lane is fixed to:

```text
batch size                 1
sequence lengths           128,256,384,512,640,768,1408,1920,2048,
                           2944,4096,4992,5120
source intermediate width  4304
candidate width            4352, zero-extended exactly
Linear weight format       FRACTAL_NZ, all 162 weights verified as format 29
execution                  TorchAir fullgraph, static, inference cache
attention                  prompt_flash_attention
attention head padding     runtime
PromptFA inner precise     1
RoPE                       separate_manual
dtype                      fp16
measurement                3 warmups, 10 samples, 5 complete stage calls/sample
metric of interest         physical tokens/s from median NPU-event time
```

Run the 13 shapes **strictly sequentially in the listed order**.  Do not use
`&`, `xargs -P`, GNU Parallel, multiple tmux panes, or multiple NPU processes.
One shape must exit and release its process before the next process starts.

The committed 910B2 comparison source is:

```text
tmp/09_persistent_page_engine/vision_matmul_lab/
  910b_b1_4352_nz_sequence_sweep_e1ffd91_r2/sweep_summary.json
```

That run peaked at 68.48k physical tokens/s at S2944 and was effectively
saturated from roughly S1920 onward.  Read the committed JSON for exact values;
do not transcribe rounded numbers into the comparison logic.

### 50.1 Pull, verify the checkpoint, and establish fresh caches

The work-server agent remains pull-only.  Do not edit tracked files, create a
branch, commit, or push.

Run Sections 50.1 through 50.3 in one persistent shell so the selected NPU and
the variables below remain fixed.  If the agent's terminal tool starts a fresh
shell for every command, open one interactive shell first and execute all three
sections inside it.

```sh
set -euo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short
test -z "$(git status --porcelain)"
source npu-setup

PYTHON=/usr/local/python3.12.13/bin/python
LAB=09_persistent_page_engine/scripts/vision_matmul_lab.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
REFERENCE=tmp/09_persistent_page_engine/vision_matmul_lab/910b_b1_4352_nz_sequence_sweep_e1ffd91_r2/sweep_summary.json
ROOT="tmp/09_persistent_page_engine/310p_phase50_b1_4352_nz_$(git rev-parse --short HEAD)"
CACHE=.runtime_cache/310p_phase50_v16_b1_4352_nz
SHAPES="128 256 384 512 640 768 1408 1920 2048 2944 4096 4992 5120"

test -x "$PYTHON"
test -f "$LAB"
test -f "$MODEL/model.safetensors"
test -f "$MODEL/config.json"
test -f "$REFERENCE"
test ! -e "$ROOT"
test ! -e "$CACHE"
test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db

mkdir -p "$ROOT"

{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON" -V
  "$PYTHON" - <<'PY'
import torch
import torch_npu
import torchair

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", getattr(torchair, "__version__", "unknown"))
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
assert "310P" in torch.npu.get_device_name(torch.npu.current_device())
PY
  npu-smi info
  sha256sum "$MODEL/model.safetensors" "$MODEL/config.json" "$REFERENCE"
  df -h "$WORK_SERVER_REPO"
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 8388608

printf '%s\n' \
  "commit=$(git rev-parse HEAD)" \
  "device=$ASCEND_RT_VISIBLE_DEVICES" \
  "model=$MODEL" \
  "model_sha256=85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db" \
  "shapes=$SHAPES" \
  "batch_size=1" \
  "intermediate_size=4352" \
  "weight_format=fractal_nz" \
  "execution=torchair" \
  "attention=prompt_flash_attention" \
  "head_padding=runtime" \
  "warmup=3 samples=10 calls_per_sample=5" \
  "cache=$CACHE" \
  >"$ROOT/command.txt"
```

If the checkout, checkpoint hash, 310P device assertion, TorchAir import, free
NPU, reference JSON, clean worktree, fresh root/cache, or 8-GiB disk check
fails, report `310P PHASE 50 PREFLIGHT: FAILURE` with the first causal error and
stop.  Never fall back to the old v1 checkpoint, native weights, width 4304,
raw eager, manual attention, or an existing cache root.

Report `310P PHASE 50 PREFLIGHT: PASS` immediately with commit, exact device,
software versions, checkpoint hash, free disk, and the two fresh paths.

### 50.2 Run the thirteen shapes serially

The result directory for a shape must not exist before invoking the lab.  Keep
the wrapper log beside the lab-owned directory; creating `s${S}/run.log` before
launch would correctly trigger the lab's non-empty-output safety check.

```sh
cd "$WORK_SERVER_REPO"
set -o pipefail

for S in $SHAPES; do
  OUT="$ROOT/s${S}"
  test ! -e "$OUT"

  printf '[%s] BEGIN B1xS%s\n' "$(date -Is)" "$S" \
    | tee -a "$ROOT/progress.log"

  set +e
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
    "$PYTHON" "$LAB" \
      --batch-size 1 \
      --sequence-length "$S" \
      --intermediate-size 4352 \
      --weight-format fractal_nz \
      --execution torchair \
      --attention-implementation prompt_flash_attention \
      --attention-head-padding runtime \
      --promptfa-inner-precise 1 \
      --rotary-implementation separate_manual \
      --model "$MODEL" \
      --cache-dir "$CACHE" \
      --allow-compile-if-missing \
      --warmup 3 --samples 10 --calls-per-sample 5 \
      --output-dir "$OUT" \
      2>&1 | tee "$ROOT/s${S}.log"
  code="${PIPESTATUS[0]}"
  set -e

  printf '%s\n' "$code" >"$ROOT/s${S}.exit_code.txt"
  printf '[%s] END B1xS%s exit=%s\n' "$(date -Is)" "$S" "$code" \
    | tee -a "$ROOT/progress.log"

  if [ "$code" -ne 0 ]; then
    printf 'PHASE50_STOP first_failed_shape=%s exit=%s\n' "$S" "$code" \
      | tee -a "$ROOT/progress.log"
    exit "$code"
  fi

  "$PYTHON" - "$OUT/run_summary.json" "$S" <<'PY' \
    | tee -a "$ROOT/progress.log"
import json, sys

path, expected_s = sys.argv[1], int(sys.argv[2])
d = json.load(open(path))
shape = d["shape"]
weights = d["weight_format"]
compile_meta = d["compile"]
measurements = d["measurements"]

assert shape["batch_size"] == 1
assert shape["sequence_length"] == expected_s
assert shape["candidate_intermediate_size"] == 4352
assert shape["linear_calls_per_full_stack"] == 162
assert weights["requested"] == "fractal_nz"
assert weights["converted_count"] == 162
assert weights["all_after_are_nz"] is True
assert weights["after_format_histogram"] == {"29": 162}
assert compile_meta["cache_existed_before"] is False

print(
    "PHASE50_PROGRESS",
    f"S={expected_s}",
    f"median_ms={measurements['device_event_per_call_ms']['median']:.6f}",
    f"physical_tps={measurements['physical_tokens_per_s_device_median']:.3f}",
    f"compile_first_call_s={compile_meta['first_call_s']:.3f}",
    "weights_nz=162/162",
)
PY
done

printf '[%s] SWEEP COMPLETE\n' "$(date -Is)" \
  | tee -a "$ROOT/progress.log"
```

The process loop itself is the serialization guarantee.  The `BEGIN` for shape
N+1 must appear only after the `END ... exit=0` and `PHASE50_PROGRESS` lines for
shape N.  If a shape fails, times out, reports any native-format Linear, or says
its cache existed before, stop at that shape.  Do not skip it and do not continue
with later shapes.

For live progress, use:

```sh
tail -n 40 -f "$ROOT/progress.log"
```

### 50.3 Aggregate and compare directly with 910B2

Run this only after all 13 exit files contain zero:

```sh
cd "$WORK_SERVER_REPO"

"$PYTHON" - "$ROOT" "$REFERENCE" <<'PY' \
  | tee "$ROOT/comparison.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reference_path = pathlib.Path(sys.argv[2])
expected = [128, 256, 384, 512, 640, 768, 1408, 1920, 2048, 2944, 4096, 4992, 5120]

reference = json.load(reference_path.open())
reference_by_s = {int(p["sequence_length"]): p for p in reference["points"]}
assert sorted(reference_by_s) == sorted(expected)

points = []
for sequence_length in expected:
    exit_path = root / f"s{sequence_length}.exit_code.txt"
    assert exit_path.read_text().strip() == "0"
    path = root / f"s{sequence_length}" / "run_summary.json"
    d = json.load(path.open())
    shape = d["shape"]
    weights = d["weight_format"]
    compile_meta = d["compile"]
    measurements = d["measurements"]

    assert shape["batch_size"] == 1
    assert shape["sequence_length"] == sequence_length
    assert shape["candidate_intermediate_size"] == 4352
    assert weights["all_after_are_nz"] is True
    assert weights["after_format_histogram"] == {"29": 162}
    assert compile_meta["cache_existed_before"] is False

    tps_310 = float(measurements["physical_tokens_per_s_device_median"])
    tps_910 = float(reference_by_s[sequence_length]["physical_tokens_per_s"])
    points.append({
        "sequence_length": sequence_length,
        "310P_device_median_ms": float(measurements["device_event_per_call_ms"]["median"]),
        "310P_physical_tokens_per_s": tps_310,
        "910B2_device_median_ms": float(reference_by_s[sequence_length]["device_median_ms"]),
        "910B2_physical_tokens_per_s": tps_910,
        "310P_over_910B_tps": tps_310 / tps_910,
        "910B_over_310P_slowdown": tps_910 / tps_310,
        "310P_linear_tflop_per_s": float(measurements["linear_tflop_per_s_device_median"]),
        "compile_first_call_s": float(compile_meta["first_call_s"]),
        "all_162_linear_weights_nz": True,
    })

peak_310 = max(points, key=lambda p: p["310P_physical_tokens_per_s"])
peak_910 = max(points, key=lambda p: p["910B2_physical_tokens_per_s"])
out = {
    "schema_version": 1,
    "classification": "310P_PHASE50_B1_4352_NZ_SWEEP_PASS",
    "fixed_configuration": {
        "batch_size": 1,
        "intermediate_size": 4352,
        "weight_format": "fractal_nz",
        "attention": "prompt_flash_attention",
        "attention_head_padding": "runtime",
        "execution": "torchair",
        "warmup": 3,
        "samples": 10,
        "calls_per_sample": 5,
    },
    "points": points,
    "310P_peak": peak_310,
    "910B2_peak": peak_910,
}
print(json.dumps(out, indent=2))
PY

test "$(find "$ROOT" -name '*.exit_code.txt' -exec cat {} \; | sort -u)" = 0
test "$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name 's*' | wc -l)" -eq 13
```

### 50.4 Report and stop

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 50 B1 4352/NZ SWEEP: PASS | PREFLIGHT_FAILURE |
SHAPE_FAILURE | FORMAT_FAILURE | COMPARISON_FAILURE
```

For a pass, include:

1. commit, host, exact physical/logical NPU, CANN/driver/firmware, Python,
   torch, torch_npu, TorchAir, and the verified v1.6 model hash;
2. the exact fixed configuration and proof all 13 runs were serial;
3. one table with sequence length, 310P median milliseconds, 310P physical
   tokens/s, 910B2 physical tokens/s, 310P/910B ratio, and slowdown;
4. proof every lane converted and retained 162/162 FRACTAL_NZ weights;
5. proof every cache was fresh and a count/size summary of `$CACHE`;
6. the 310P peak shape/tokens/s, where the curve approximately saturates, and
   whether long sequences improve, plateau, or regress;
7. concise `What is proven`, `What remains unresolved`, and the first causal
   error if anything failed.

Paste `agent_report.md`, `comparison.json`, and `progress.log` back to Luka.
Keep caches and full per-shape logs on the work server.  Do not profile, alter
the model, change PromptFA settings, run B2/B4, integrate into E2E, or start a
new optimization experiment.  Then **stop**.

## Phase 50.5: project the measured B1 curve over the full crop corpus

### Goal and interpretation

Run this only after both Phase 49 and Phase 50 report `PASS`.  This is an
analysis-only follow-up: it must not select an NPU, import torch, load a model,
compile a graph, or execute a vision layer.

Use the corrected Phase-49 full-run trace to assign each of the 30,557 real
OmniDocBench crops to the smallest Phase-50 B1 bucket that fits it.  Then sum
`crop_count[bucket] * measured_median_ms[bucket]` to obtain the literal
theoretical time for executing every crop separately with the optimized
4352/FRACTAL_NZ implementation.  Report both:

- effective throughput: real crop tokens divided by projected time; and
- raw throughput: padded bucket tokens divided by projected time.

Also apply the measured curve to the frozen 910B reference's B1 greedy-packing
group histogram.  This second number answers a different question: what the
same recorded B1 packing plan would cost if every packed graph ran at the
isolated optimized median.  It is an estimate, because the isolated lab uses a
single synthetic grid and an all-false attention mask while production packs
use block-structured masks.  Do not present it as a measured E2E result.

The analysis must independently reproduce these frozen corpus facts before it
is allowed to emit a projection:

```text
pages                         1,651
crops                         30,557
real vision tokens            18,805,052
per-crop B1 physical tokens   22,503,424
reference B1 packed groups    9,665
reference packed physical     21,310,208
910B optimized B1 projection  528.074783506 s
910B packed-plan estimate     334.186300341 s
```

### 50.5.1 Locate the completed inputs

The work-server agent remains pull-only.  Do not edit tracked files, create a
branch, commit, or push.

```sh
set -euo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"

PYTHON=/usr/local/python3.12.13/bin/python
PHASE49="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase49_v16_full_*' | sort | tail -n 1)"
PHASE50="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase50_b1_4352_nz_*' | sort | tail -n 1)"
REFERENCE_SWEEP=tmp/09_persistent_page_engine/vision_matmul_lab/910b_b1_4352_nz_sequence_sweep_e1ffd91_r2/sweep_summary.json
REFERENCE_RUN=tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/output/run_summary.json
TRACE="$PHASE49/e2e/output/recognition_trace.jsonl"
CANDIDATE_RUN="$PHASE49/e2e/output/run_summary.json"
CANDIDATE_SWEEP="$PHASE50/comparison.json"
OUT="$PHASE50/corpus_projection"

test -x "$PYTHON"
test -n "$PHASE49" && test -n "$PHASE50"
test -f "$PHASE49/agent_report.md"
test -f "$PHASE50/agent_report.md"
rg -n '^310P PHASE 49 FULL V1.6: PASS' "$PHASE49/agent_report.md"
rg -n '^310P PHASE 50 B1 4352/NZ SWEEP: PASS' "$PHASE50/agent_report.md"
test -f "$TRACE"
test -f "$CANDIDATE_RUN"
test -f "$CANDIDATE_SWEEP"
test -f "$REFERENCE_SWEEP"
test -f "$REFERENCE_RUN"
test ! -e "$OUT"
mkdir -p "$OUT"

printf '%s\n' \
  "phase49=$PHASE49" \
  "phase50=$PHASE50" \
  "trace=$TRACE" \
  "candidate_run=$CANDIDATE_RUN" \
  "candidate_sweep=$CANDIDATE_SWEEP" \
  "reference_sweep=$REFERENCE_SWEEP" \
  "reference_run=$REFERENCE_RUN" \
  | tee "$OUT/inputs.txt"
```

If either phase is not a pass, or any exact input is missing, report
`310P PHASE 50.5 FULL-CORPUS PROJECTION: INPUT_FAILURE` with the first missing
contract and stop.  Do not substitute an older v1 trace or a rounded table from
an agent report.

### 50.5.2 Calculate both projections

```sh
"$PYTHON" - \
  "$TRACE" "$CANDIDATE_RUN" "$CANDIDATE_SWEEP" \
  "$REFERENCE_RUN" "$REFERENCE_SWEEP" <<'PY' \
  | tee "$OUT/projection.json"
import collections
import json
import math
import sys

(
    trace_path,
    candidate_run_path,
    candidate_sweep_path,
    reference_run_path,
    reference_sweep_path,
) = sys.argv[1:]

candidate_run = json.load(open(candidate_run_path))
candidate_sweep = json.load(open(candidate_sweep_path))
reference_run = json.load(open(reference_run_path))
reference_sweep = json.load(open(reference_sweep_path))

candidate_ms = {
    int(point["sequence_length"]): float(point["310P_device_median_ms"])
    for point in candidate_sweep["points"]
}
reference_ms = {
    int(point["sequence_length"]): float(point["device_median_ms"])
    for point in reference_sweep["points"]
}
buckets = sorted(reference_ms)
assert buckets == [
    128, 256, 384, 512, 640, 768, 1408,
    1920, 2048, 2944, 4096, 4992, 5120,
]
assert sorted(candidate_ms) == buckets

crop_counts = collections.Counter()
real_tokens = 0
crops = 0
for line in open(trace_path):
    if not line.strip():
        continue
    row = json.loads(line)
    tokens = int(row["vision"]["real_vision_tokens"])
    bucket = next((value for value in buckets if value >= tokens), None)
    assert bucket is not None, (row["request_id"], tokens)
    crop_counts[bucket] += 1
    real_tokens += tokens
    crops += 1

expected_crop_counts = {
    256: 16812,
    384: 3001,
    512: 1992,
    640: 1589,
    768: 1303,
    1408: 2927,
    1920: 951,
    2048: 129,
    2944: 585,
    4096: 321,
    4992: 683,
    5120: 264,
}
assert dict(crop_counts) == expected_crop_counts
assert crops == 30557
assert real_tokens == 18805052
crop_physical_tokens = sum(bucket * count for bucket, count in crop_counts.items())
assert crop_physical_tokens == 22503424

reference_recognition = reference_run["recognition"]
reference_histogram = reference_recognition["vision_packing"]["graph_shape_histogram"]
packed_counts = collections.Counter()
for name, count in reference_histogram.items():
    if name == "eager_overflow":
        # The full trace proves all 264 old overflow calls are exactly S5120.
        packed_counts[5120] += int(count)
    else:
        prefix = "b1_s"
        assert name.startswith(prefix), name
        packed_counts[int(name[len(prefix):])] += int(count)
assert sum(packed_counts.values()) == 9665
packed_physical_tokens = sum(bucket * count for bucket, count in packed_counts.items())
assert packed_physical_tokens == 21310208

def projection(counts, medians_ms, physical_tokens):
    rows = []
    total_s = 0.0
    for bucket in buckets:
        count = int(counts.get(bucket, 0))
        time_s = count * medians_ms[bucket] / 1000.0
        total_s += time_s
        rows.append({
            "bucket": bucket,
            "calls": count,
            "median_ms": medians_ms[bucket],
            "time_s": time_s,
            "physical_tokens": count * bucket,
        })
    return {
        "calls": sum(counts.values()),
        "time_s": total_s,
        "seconds_per_page": total_s / 1651,
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "padding_tokens": physical_tokens - real_tokens,
        "useful_token_fraction": real_tokens / physical_tokens,
        "effective_real_tokens_per_s": real_tokens / total_s,
        "raw_physical_tokens_per_s": physical_tokens / total_s,
        "per_bucket": rows,
    }

direct_310 = projection(crop_counts, candidate_ms, crop_physical_tokens)
direct_910 = projection(crop_counts, reference_ms, crop_physical_tokens)
packed_310 = projection(packed_counts, candidate_ms, packed_physical_tokens)
packed_910 = projection(packed_counts, reference_ms, packed_physical_tokens)

assert math.isclose(direct_910["time_s"], 528.0747835060118, abs_tol=1e-9)
assert math.isclose(packed_910["time_s"], 334.1863003410339, abs_tol=1e-9)

def actual(summary):
    recognition = summary["recognition"]
    seconds = float(recognition["device_stage_s"]["vision_prefill"])
    real = int(recognition["real_vision_tokens"])
    physical = int(recognition["physical_vision_tokens"])
    return {
        "time_s": seconds,
        "seconds_per_page": seconds / 1651,
        "real_tokens": real,
        "physical_tokens": physical,
        "effective_real_tokens_per_s": real / seconds,
        "raw_physical_tokens_per_s": physical / seconds,
        "vision_packing": recognition["vision_packing"],
    }

actual_310 = actual(candidate_run)
actual_910 = actual(reference_run)
assert actual_310["real_tokens"] == real_tokens
assert actual_910["real_tokens"] == real_tokens

out = {
    "schema_version": 1,
    "classification": "310P_PHASE50_5_FULL_CORPUS_PROJECTION_PASS",
    "semantics": {
        "direct_b1": "Each real crop separately, assigned to the smallest fitting Phase-50 bucket.",
        "reference_b1_packed_plan": "Frozen 910B B1 greedy-group histogram replayed with isolated sweep medians; estimate, not measured production.",
        "actual": "Measured device-event total from each full production run.",
    },
    "corpus": {
        "pages": 1651,
        "crops": crops,
        "real_vision_tokens": real_tokens,
        "direct_b1_bucket_counts": dict(sorted(crop_counts.items())),
        "reference_b1_packed_group_counts": dict(sorted(packed_counts.items())),
    },
    "projection": {
        "310P_direct_b1": direct_310,
        "910B2_direct_b1": direct_910,
        "310P_reference_b1_packed_plan": packed_310,
        "910B2_reference_b1_packed_plan": packed_910,
    },
    "actual_full_run": {
        "310P3_phase49": actual_310,
        "910B2_8634d3a": actual_910,
    },
    "comparison": {
        "direct_b1_310P_over_910B_time": direct_310["time_s"] / direct_910["time_s"],
        "direct_b1_310P_over_910B_effective_tps": direct_310["effective_real_tokens_per_s"] / direct_910["effective_real_tokens_per_s"],
        "packed_plan_310P_over_910B_time": packed_310["time_s"] / packed_910["time_s"],
        "packed_plan_310P_over_910B_effective_tps": packed_310["effective_real_tokens_per_s"] / packed_910["effective_real_tokens_per_s"],
        "310P_direct_b1_over_310P_actual_time": direct_310["time_s"] / actual_310["time_s"],
        "310P_packed_plan_over_310P_actual_time": packed_310["time_s"] / actual_310["time_s"],
        "910B_direct_b1_over_910B_actual_time": direct_910["time_s"] / actual_910["time_s"],
        "910B_packed_plan_over_910B_actual_time": packed_910["time_s"] / actual_910["time_s"],
    },
}
print(json.dumps(out, indent=2))
PY
```

This command should finish in seconds and produce incremental terminal output
only at completion because it reads and aggregates JSONL without any NPU work.
If desired, observe the output file from another shell with:

```sh
ls -lh "$OUT/projection.json"
```

### 50.5.3 Report and stop

Write `$OUT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 50.5 FULL-CORPUS PROJECTION: PASS | INPUT_FAILURE |
CORPUS_MISMATCH | CALCULATION_FAILURE
```

For a pass, paste two compact tables:

1. 310P optimized direct B1, 910B optimized direct B1, measured 310P
   production, and measured 910B production: calls/groups where meaningful,
   total vision seconds, seconds/page, effective real tokens/s, and raw
   physical tokens/s;
2. the 310P and 910B estimates under the frozen reference B1 packing plan,
   including seconds and both throughput definitions.

State explicitly that the direct-B1 number is a literal sum of measured B1
medians, while the packed-plan number is only a shape-histogram estimate.  Give
the 310P/910B slowdown for both projections and explain how much of the gap
between direct B1 and actual production comes from reducing 30,557 crop calls
to 9,665 packed calls.  Include concise `What is proven` and `What remains an
estimate` sections.

Paste `agent_report.md` and `projection.json` back to Luka.  Do not run another
NPU experiment, recompile anything, or modify the pipeline.  Then **stop**.

## Phase 51: native-4304 B1 vision baseline and optimized ablation on 310P

### Goal and fixed comparison boundary

Run the exact Phase-50 B1 sequence sweep again on the same Atlas 310P3, changing
only the two coupled vision-Linear choices:

```text
Phase 51 baseline       intermediate_size=4304, weight_format=native/ND
Phase 50 optimized      intermediate_size=4352, weight_format=FRACTAL_NZ
```

Keep runtime D72-to-D80 attention-head padding enabled in both lanes.  It is a
PromptFA correctness requirement and is not the MLP weight-padding ablation
being tested here.  Everything else must remain identical: B1, the same 13
sequence lengths, all 27 vision layers, PromptFA, separate-manual RoPE,
TorchAir fullgraph, fp16, three warmups, ten samples, and five complete stage
calls per sample.

The committed 910B2 evidence for the same pair is:

```text
baseline native 4304
  tmp/09_persistent_page_engine/vision_matmul_lab/
  910b_b1_4304_native_sequence_sweep_33213bd_r1/sweep_summary.json
optimized 4352 + NZ
  tmp/09_persistent_page_engine/vision_matmul_lab/
  910b_b1_4352_nz_sequence_sweep_e1ffd91_r2/sweep_summary.json
direct comparison
  tmp/09_persistent_page_engine/vision_matmul_lab/
  910b_b1_4304_native_sequence_sweep_33213bd_r1/
  comparison_to_4352_nz.json
```

Phase 51 must produce both the per-shape 310P comparison and the same
full-OmniDocBench theoretical projections used in Phase 50.5.  It does not run
layout, OCR E2E, evaluation, B2/B4, profiling, or any additional ablation.

### 51.1 Pull, locate the passes, and establish a fresh native cache

The work-server agent remains pull-only.  Do not edit tracked files, create a
branch, commit, or push.  Run Sections 51.1 through 51.3 in one persistent
shell so the selected NPU and variables remain fixed.

```sh
set -uo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"
source npu-setup
set -e
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

PYTHON=/usr/local/python3.12.13/bin/python
LAB=09_persistent_page_engine/scripts/vision_matmul_lab.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
PHASE49="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase49_v16_full_*' | sort | tail -n 1)"
PHASE50="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase50_b1_4352_nz_*' | sort | tail -n 1)"
ROOT="tmp/09_persistent_page_engine/310p_phase51_b1_4304_native_$(git rev-parse --short HEAD)"
CACHE=.runtime_cache/310p_phase51_v16_b1_4304_native
SHAPES="128 256 384 512 640 768 1408 1920 2048 2944 4096 4992 5120"

REFERENCE_BASELINE=tmp/09_persistent_page_engine/vision_matmul_lab/910b_b1_4304_native_sequence_sweep_33213bd_r1/sweep_summary.json
REFERENCE_OPTIMIZED=tmp/09_persistent_page_engine/vision_matmul_lab/910b_b1_4352_nz_sequence_sweep_e1ffd91_r2/sweep_summary.json
REFERENCE_RUN=tmp/09_persistent_page_engine/910b_full_e2e_eval_8634d3a_r1/output/run_summary.json
PHASE50_COMPARISON="$PHASE50/comparison.json"
PHASE49_RUN="$PHASE49/e2e/output/run_summary.json"
PHASE49_TRACE="$PHASE49/e2e/output/recognition_trace.jsonl"

test -x "$PYTHON"
test -f "$LAB"
test -f "$MODEL/model.safetensors"
test -n "$PHASE49" && test -n "$PHASE50"
test -f "$PHASE49/agent_report.md"
test -f "$PHASE50/agent_report.md"
rg -n '^310P PHASE 49 FULL V1.6: PASS' "$PHASE49/agent_report.md"
rg -n '^310P PHASE 50 B1 4352/NZ SWEEP: PASS' "$PHASE50/agent_report.md"
test -f "$PHASE50_COMPARISON"
test -f "$PHASE49_RUN"
test -f "$PHASE49_TRACE"
test -f "$REFERENCE_BASELINE"
test -f "$REFERENCE_OPTIMIZED"
test -f "$REFERENCE_RUN"
test ! -e "$ROOT"
test ! -e "$CACHE"
test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db

mkdir -p "$ROOT"

{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON" -V
  "$PYTHON" - <<'PY'
import torch
import torch_npu
import torchair

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", getattr(torchair, "__version__", "unknown"))
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
assert "310P" in torch.npu.get_device_name(torch.npu.current_device())
PY
  npu-smi info
  sha256sum \
    "$MODEL/model.safetensors" \
    "$REFERENCE_BASELINE" "$REFERENCE_OPTIMIZED" \
    "$PHASE50_COMPARISON" "$PHASE49_RUN"
  df -h "$WORK_SERVER_REPO"
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 8388608

printf '%s\n' \
  "commit=$(git rev-parse HEAD)" \
  "device=$ASCEND_RT_VISIBLE_DEVICES" \
  "model=$MODEL" \
  "model_sha256=85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db" \
  "shapes=$SHAPES" \
  "batch_size=1 intermediate_size=4304 weight_format=native" \
  "execution=torchair attention=prompt_flash_attention head_padding=runtime" \
  "warmup=3 samples=10 calls_per_sample=5" \
  "cache=$CACHE" \
  >"$ROOT/command.txt"
```

If the clean checkout, v1.6 hash, Phase-49 pass, Phase-50 pass, 310P device,
TorchAir import, reference files, fresh root/cache, or 8-GiB disk check fails,
report `310P PHASE 51 PREFLIGHT: FAILURE` with the first causal error and stop.
Never reuse the Phase-50 optimized cache, an old v1 cache, or any existing
native cache.  Report `310P PHASE 51 PREFLIGHT: PASS` immediately with the
exact commit, NPU, software, checkpoint hash, and paths.

### 51.2 Run all thirteen native-4304 shapes serially

The output directory for each shape must not exist before the lab starts.  Keep
the wrapper log beside it, not inside it.

```sh
cd "$WORK_SERVER_REPO"
set -o pipefail

for S in $SHAPES; do
  OUT="$ROOT/s${S}"
  test ! -e "$OUT"
  printf '[%s] BEGIN BASELINE B1xS%s\n' "$(date -Is)" "$S" \
    | tee -a "$ROOT/progress.log"

  set +e
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
    "$PYTHON" "$LAB" \
      --batch-size 1 \
      --sequence-length "$S" \
      --intermediate-size 4304 \
      --weight-format native \
      --execution torchair \
      --attention-implementation prompt_flash_attention \
      --attention-head-padding runtime \
      --promptfa-inner-precise 1 \
      --rotary-implementation separate_manual \
      --model "$MODEL" \
      --cache-dir "$CACHE" \
      --allow-compile-if-missing \
      --warmup 3 --samples 10 --calls-per-sample 5 \
      --output-dir "$OUT" \
      2>&1 | tee "$ROOT/s${S}.log"
  code="${PIPESTATUS[0]}"
  set -e

  printf '%s\n' "$code" >"$ROOT/s${S}.exit_code.txt"
  printf '[%s] END BASELINE B1xS%s exit=%s\n' \
    "$(date -Is)" "$S" "$code" | tee -a "$ROOT/progress.log"
  if [ "$code" -ne 0 ]; then
    printf 'PHASE51_STOP first_failed_shape=%s exit=%s\n' "$S" "$code" \
      | tee -a "$ROOT/progress.log"
    exit "$code"
  fi

  "$PYTHON" - "$OUT/run_summary.json" "$S" <<'PY' \
    | tee -a "$ROOT/progress.log"
import json, sys

d = json.load(open(sys.argv[1]))
expected_s = int(sys.argv[2])
shape = d["shape"]
weights = d["weight_format"]
compile_meta = d["compile"]
m = d["measurements"]
assert shape["batch_size"] == 1
assert shape["sequence_length"] == expected_s
assert shape["candidate_intermediate_size"] == 4304
assert shape["linear_calls_per_full_stack"] == 162
assert weights["requested"] == "native"
assert weights["converted_count"] == 0
assert weights["after_format_histogram"] == {"2": 162}
assert weights["all_after_are_nz"] is False
assert compile_meta["cache_existed_before"] is False
print(
    "PHASE51_PROGRESS",
    f"S={expected_s}",
    f"median_ms={m['device_event_per_call_ms']['median']:.6f}",
    f"physical_tps={m['physical_tokens_per_s_device_median']:.3f}",
    f"compile_first_call_s={compile_meta['first_call_s']:.3f}",
    "native_weights=162/162",
)
PY
done

printf '[%s] BASELINE SWEEP COMPLETE\n' "$(date -Is)" \
  | tee -a "$ROOT/progress.log"
```

The loop itself is the serialization guarantee.  Stop on the first nonzero
exit, timeout, non-4304 shape, converted weight, non-native format, or stale
cache.  Do not skip a shape or continue after a failed contract.

For live progress:

```sh
tail -n 40 -f "$ROOT/progress.log"
```

### 51.3 Compare four curves and project the full corpus

Run this only after all 13 shapes pass:

```sh
"$PYTHON" - \
  "$ROOT" "$PHASE50_COMPARISON" \
  "$REFERENCE_BASELINE" "$REFERENCE_OPTIMIZED" \
  "$PHASE49_RUN" "$REFERENCE_RUN" "$PHASE49_TRACE" <<'PY' \
  | tee "$ROOT/comparison.json"
import collections
import json
import math
import pathlib
import sys

(
    root_path,
    phase50_path,
    reference_baseline_path,
    reference_optimized_path,
    phase49_run_path,
    reference_run_path,
    trace_path,
) = sys.argv[1:]
root = pathlib.Path(root_path)
phase50 = json.load(open(phase50_path))
reference_baseline = json.load(open(reference_baseline_path))
reference_optimized = json.load(open(reference_optimized_path))
phase49_run = json.load(open(phase49_run_path))
reference_run = json.load(open(reference_run_path))

shapes = [128, 256, 384, 512, 640, 768, 1408, 1920, 2048, 2944, 4096, 4992, 5120]

baseline_310 = {}
for sequence_length in shapes:
    assert (root / f"s{sequence_length}.exit_code.txt").read_text().strip() == "0"
    d = json.load((root / f"s{sequence_length}" / "run_summary.json").open())
    assert d["shape"]["candidate_intermediate_size"] == 4304
    assert d["weight_format"]["requested"] == "native"
    assert d["weight_format"]["converted_count"] == 0
    assert d["weight_format"]["after_format_histogram"] == {"2": 162}
    baseline_310[sequence_length] = {
        "median_ms": float(d["measurements"]["device_event_per_call_ms"]["median"]),
        "physical_tps": float(d["measurements"]["physical_tokens_per_s_device_median"]),
        "linear_tflop_per_s": float(d["measurements"]["linear_tflop_per_s_device_median"]),
    }

optimized_310 = {
    int(p["sequence_length"]): {
        "median_ms": float(p["310P_device_median_ms"]),
        "physical_tps": float(p["310P_physical_tokens_per_s"]),
        "linear_tflop_per_s": float(p["310P_linear_tflop_per_s"]),
    }
    for p in phase50["points"]
}
baseline_910 = {
    int(p["sequence_length"]): {
        "median_ms": float(p["device_median_ms"]),
        "physical_tps": float(p["physical_tokens_per_s"]),
    }
    for p in reference_baseline["points"]
}
optimized_910 = {
    int(p["sequence_length"]): {
        "median_ms": float(p["device_median_ms"]),
        "physical_tps": float(p["physical_tokens_per_s"]),
    }
    for p in reference_optimized["points"]
}
assert sorted(baseline_310) == sorted(optimized_310) == shapes
assert sorted(baseline_910) == sorted(optimized_910) == shapes

rows = []
for sequence_length in shapes:
    b310 = baseline_310[sequence_length]
    o310 = optimized_310[sequence_length]
    b910 = baseline_910[sequence_length]
    o910 = optimized_910[sequence_length]
    rows.append({
        "sequence_length": sequence_length,
        "310P_baseline_native_4304_ms": b310["median_ms"],
        "310P_optimized_4352_nz_ms": o310["median_ms"],
        "310P_baseline_native_4304_physical_tps": b310["physical_tps"],
        "310P_optimized_4352_nz_physical_tps": o310["physical_tps"],
        "310P_optimized_over_baseline_tps": o310["physical_tps"] / b310["physical_tps"],
        "310P_optimized_time_reduction_fraction": 1.0 - o310["median_ms"] / b310["median_ms"],
        "910B2_baseline_native_4304_physical_tps": b910["physical_tps"],
        "910B2_optimized_4352_nz_physical_tps": o910["physical_tps"],
        "910B2_optimized_over_baseline_tps": o910["physical_tps"] / b910["physical_tps"],
        "310P_over_910B_baseline_tps": b310["physical_tps"] / b910["physical_tps"],
        "310P_over_910B_optimized_tps": o310["physical_tps"] / o910["physical_tps"],
    })

crop_counts = collections.Counter()
real_tokens = 0
for line in open(trace_path):
    if not line.strip():
        continue
    row = json.loads(line)
    tokens = int(row["vision"]["real_vision_tokens"])
    bucket = next((value for value in shapes if value >= tokens), None)
    assert bucket is not None, (row["request_id"], tokens)
    crop_counts[bucket] += 1
    real_tokens += tokens
expected_crop_counts = {
    256: 16812, 384: 3001, 512: 1992, 640: 1589, 768: 1303,
    1408: 2927, 1920: 951, 2048: 129, 2944: 585, 4096: 321,
    4992: 683, 5120: 264,
}
assert dict(crop_counts) == expected_crop_counts
assert sum(crop_counts.values()) == 30557
assert real_tokens == 18805052

reference_histogram = reference_run["recognition"]["vision_packing"]["graph_shape_histogram"]
packed_counts = collections.Counter()
for name, count in reference_histogram.items():
    if name == "eager_overflow":
        packed_counts[5120] += int(count)
    else:
        assert name.startswith("b1_s"), name
        packed_counts[int(name[4:])] += int(count)
assert sum(packed_counts.values()) == 9665

def project(counts, curve):
    physical = sum(bucket * count for bucket, count in counts.items())
    seconds = sum(count * curve[bucket]["median_ms"] / 1000.0 for bucket, count in counts.items())
    return {
        "calls": sum(counts.values()),
        "time_s": seconds,
        "seconds_per_page": seconds / 1651,
        "real_tokens": real_tokens,
        "physical_tokens": physical,
        "effective_real_tokens_per_s": real_tokens / seconds,
        "raw_physical_tokens_per_s": physical / seconds,
    }

projections = {
    "310P_baseline_direct_b1": project(crop_counts, baseline_310),
    "310P_optimized_direct_b1": project(crop_counts, optimized_310),
    "910B2_baseline_direct_b1": project(crop_counts, baseline_910),
    "910B2_optimized_direct_b1": project(crop_counts, optimized_910),
    "310P_baseline_reference_packed_plan": project(packed_counts, baseline_310),
    "310P_optimized_reference_packed_plan": project(packed_counts, optimized_310),
    "910B2_baseline_reference_packed_plan": project(packed_counts, baseline_910),
    "910B2_optimized_reference_packed_plan": project(packed_counts, optimized_910),
}
assert math.isclose(projections["910B2_baseline_direct_b1"]["time_s"], 579.2756568290711, abs_tol=1e-9)
assert math.isclose(projections["910B2_optimized_direct_b1"]["time_s"], 528.074783506012, abs_tol=1e-9)
assert math.isclose(projections["910B2_baseline_reference_packed_plan"]["time_s"], 353.54757228775026, abs_tol=1e-9)
assert math.isclose(projections["910B2_optimized_reference_packed_plan"]["time_s"], 334.1863003410339, abs_tol=1e-9)

def actual(summary):
    recognition = summary["recognition"]
    seconds = float(recognition["device_stage_s"]["vision_prefill"])
    real = int(recognition["real_vision_tokens"])
    physical = int(recognition["physical_vision_tokens"])
    assert real == real_tokens
    return {
        "time_s": seconds,
        "effective_real_tokens_per_s": real / seconds,
        "raw_physical_tokens_per_s": physical / seconds,
        "physical_tokens": physical,
        "vision_packing": recognition["vision_packing"],
    }

out = {
    "schema_version": 1,
    "classification": "310P_PHASE51_NATIVE_4304_BASELINE_PASS",
    "fixed_boundary": {
        "shared": "B1, full 27-layer VisionPrefillStage, PromptFA, runtime D80 head padding, TorchAir, fp16, 3x10x5 timing",
        "baseline": "intermediate_size=4304, 162 native/ND Linear weights",
        "optimized": "intermediate_size=4352 zero-extension, 162 FRACTAL_NZ Linear weights",
    },
    "per_shape": rows,
    "full_corpus_projection": projections,
    "actual_full_run": {
        "310P3_phase49": actual(phase49_run),
        "910B2_8634d3a": actual(reference_run),
    },
}
print(json.dumps(out, indent=2))
PY
```

### 51.4 Report and stop

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 51 NATIVE-4304 BASELINE: PASS | PREFLIGHT_FAILURE |
SHAPE_FAILURE | FORMAT_FAILURE | COMPARISON_FAILURE | CORPUS_MISMATCH
```

For a pass, include:

1. commit, host, exact NPU, software, checkpoint hash, fresh cache proof, and
   proof all 13 processes ran strictly serially;
2. one per-shape table containing both 310P configurations, optimized/baseline
   throughput gain, both 910B configurations, and the 310P/910B ratio before
   and after optimization;
3. proof all baseline lanes stayed 4304 with 162/162 native format-2 Linear
   weights and zero conversions;
4. direct-B1 full-corpus seconds and effective/raw throughput for all four
   hardware/configuration pairs;
5. frozen-reference-packing-plan seconds and throughput for all four pairs;
6. the measured Phase-49 310P and measured full-run 910B vision totals, kept
   explicitly separate from projections;
7. where the optimization helps most, where it is negligible, the
   corpus-weighted time reduction on 310P versus 910B, and whether 4352+NZ
   closes or widens the hardware gap;
8. concise `What is proven`, `What remains an estimate`, and the first causal
   error if anything failed.

Paste `agent_report.md`, `comparison.json`, and `progress.log` back to Luka.
Do not attribute the combined gain separately to width padding or NZ; that
would require two additional ablation lanes.  Do not run those lanes, profile,
or integrate the optimization into production.  Then **stop**.

## Phase 52: optimized KV2048 full E2E and official evaluation on 310P

### Goal and fixed comparison boundary

Replicate the newly completed 910B2 full OmniDocBench experiment on one
Atlas 310P3, using the same model checkpoint, page set, preprocessing policy,
graph shapes, packing policy, KV capacity, optimized vision weights, decode
batch size, and evaluator.  This is a direct hardware comparison and a
production-integration check for the Phase-50 4352/FRACTAL_NZ vision result.

The fixed production configuration is:

```text
pages                         1,651, offset 0
layout                        NPU eager, owned frontend at current source defaults,
                              all 1,651 pages completed before OCR starts
decode                        B32, KV2048, max_new_tokens=2048,
                              TorchAir combined_apply_static_actual
global crop preprocessing     min_pixels=28224, max_pixels=401408
text crops                     additional linear scale=0.5
vision                         TorchAir PromptFA, 128 alignment,
                              buckets 128,256,384,512,640,768,1024,
                              1408,1920,2048
vision Linear weights          MLP intermediate 4304 -> 4352 by zero extension,
                              all 162 Linear weights in FRACTAL_NZ
vision packing                 greedy, target 1024, lookahead 32
text prefill                   production-group packing,
                              pack buckets 128,256,512,1024
timeline/fingerprints          disabled
```

Do not change any of these choices.  In particular:

- `combined_apply_static_actual` is required on 310P.  Do not substitute
  `combined_apply`; the static-actual path is the established workaround for
  the silent IncreFA hang at effective length 1280.
- `--vision-promptfa-align-128` is required on 310P.
- `--max-new-tokens 2048` is not the old artificial 2808-token cap.  With a
  2048-token KV cache it is the absolute safety ceiling; each crop still stops
  at EOS or when its own remaining KV capacity is exhausted.
- `--preprocessor-max-pixels 401408` is the global 2048-vision-token cap.
  `--text-crop-scale 0.5` is additionally applied only to text crops.
- Keep the ten exact vision buckets.  Do not restore the default large bucket
  ladder; it is both a different experiment and unsafe for 21-GB HBM.
- Vision packing remains B1 packed-sequence execution.  Do not introduce B2 or
  B4 batched-vision graphs.
- Do not evaluate a filtered page set and do not exclude evaluator fallbacks.

The committed 910B2 reference is the authority:

```text
run summary
  tmp/09_persistent_page_engine/
  910b_opt_kv2048_pack1024_full_7b9d419/output/run_summary.json
exact command and progress log
  tmp/09_persistent_page_engine/
  910b_opt_kv2048_pack1024_full_7b9d419/run.log
official evaluator result
  tmp/09_persistent_page_engine/
  910b_opt_kv2048_pack1024_full_7b9d419/evaluation/work/result/
```

Its measured headline is:

| Metric | 910B2 reference |
|---|---:|
| setup | 42.311 s |
| pipeline E2E | 878.290 s |
| pages/s | 1.87979 |
| seconds/page | 0.53197 |
| all-pages-first layout | 281.558 s |
| requests | 30,557 |
| vision prefill | 178.561 s; 48,450 real / 53,275 physical tok/s |
| text prefill | 70.587 s; 36,268 real / 56,626 physical tok/s |
| decode | 177.393 s; 9,450 effective / 9,828 raw tok/s |
| stop reasons | 30,470 EOS; 87 KV-cache-full |

The official 910B2 metrics are text Edit `0.0504264899`, display-formula Edit
`0.0912943862`, table Edit `0.0763942410`, page TEDS `0.9217778901`, structure
TEDS `0.9486102386`, and reading-order Edit `0.1401727153`.  Lower is better for
Edit distances; higher is better for TEDS.  The scripts below read the committed
JSON rather than relying on these rounded values.

This phase is checkpointed.  Report back immediately after preflight, after
cache preparation, after the full E2E run, and after evaluation.  Do not wait
until everything finishes to provide the first update.

### 52.1 Pull, prove provenance, and establish fresh caches

The work-server agent is pull-only.  Do not edit tracked files, create a branch,
commit, or push.  Run the phase in one persistent shell so the selected NPU and
all variables remain fixed.

```sh
set -uo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"
source npu-setup
set -e
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

PYTHON_BIN=/usr/local/python3.12.13/bin/python
EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
DATASET_JSON=/home/lukaiv/datasets/OmniDocBench/OmniDocBench.json
IMAGES_DIR=/home/lukaiv/datasets/OmniDocBench/images
LAYOUT_MODEL=/home/lukaiv/models/PP-DocLayoutV3_safetensors

ROOT="tmp/09_persistent_page_engine/310p_phase52_opt_kv2048_$(git rev-parse --short HEAD)"
PREP="$ROOT/cache_prepare"
LANE="$ROOT/e2e"
OUTPUT="$LANE/output"
EVAL="$ROOT/evaluation"

REFERENCE_ROOT=tmp/09_persistent_page_engine/910b_opt_kv2048_pack1024_full_7b9d419
REFERENCE_RUN="$REFERENCE_ROOT/output/run_summary.json"
REFERENCE_RESULT="$REFERENCE_ROOT/evaluation/work/result"
REFERENCE_METRIC="$REFERENCE_RESULT/predictions_quick_match_metric_result.json"
REFERENCE_EVAL_SUMMARY="$REFERENCE_RESULT/predictions_quick_match_run_summary.json"

DECODE_CACHE=.runtime_cache/310p_phase52_v16_decode_b32_k2048
VISION_CACHE=.runtime_cache/310p_phase52_v16_vision_4352_nz
BATCHED_CACHE=.runtime_cache/310p_phase52_v16_vision_batched_unused
TEXT_CACHE=.runtime_cache/310p_phase52_v16_text
PACKED_CACHE=.runtime_cache/310p_phase52_v16_text_packed

test ! -e "$ROOT"
for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  test ! -e "$cache"
done

mkdir -p "$PREP" "$LANE" "$EVAL/work"
test -x "$PYTHON_BIN"
test -x "$EVAL_PYTHON"
test -f "$E2E"
test -f "$EVAL_WRAPPER"
test -f "$MODEL/model.safetensors"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -d "$LAYOUT_MODEL"
test -f "$REFERENCE_RUN"
test -f "$REFERENCE_METRIC"
test -f "$REFERENCE_EVAL_SUMMARY"

test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db

PHASE49="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase49_v16_full_*' | sort | tail -n 1)"
test -n "$PHASE49"
test -f "$PHASE49/agent_report.md"
rg -n '^310P PHASE 49 FULL V1.6: PASS' "$PHASE49/agent_report.md"

{
  date -Is
  hostname
  git rev-parse HEAD
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torchair", getattr(torchair, "__version__", "unknown"))
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
assert "310P" in torch.npu.get_device_name(torch.npu.current_device())
PY
  npu-smi info
  sha256sum \
    "$MODEL/model.safetensors" \
    "$REFERENCE_RUN" "$REFERENCE_METRIC" "$REFERENCE_EVAL_SUMMARY"
  df -h "$WORK_SERVER_REPO" /home/lukaiv/models
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 10485760
```

If the checkout is dirty, the v1.6 weight hash differs, Phase 49 is not a pass,
the committed reference is absent, the device is not a 310P, TorchAir cannot be
imported, any phase cache already exists, or less than 10 GiB is free, report
`310P PHASE 52 PREFLIGHT: FAILURE` with the first causal error and stop.  Do not
delete an old cache or run to make room without Luka's approval.

Otherwise report `310P PHASE 52 PREFLIGHT: PASS` immediately, including the
project commit, host, selected physical/logical NPU, CANN/driver/firmware,
Python/torch/torch_npu/TorchAir, checkpoint SHA, free disk, and all five cache
paths.

### 52.2 Compile once, then prove the exact lane is warm

Define the production arguments once.  The compile smoke, warm smoke, and full
run must all reuse this array unchanged.

```sh
PRODUCTION_ARGS=(
  "$E2E"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL"
  --recognizer-model "$MODEL"
  --batch-size 32 --cache-length 2048 --max-new-tokens 2048
  --preprocessor-min-pixels 28224
  --preprocessor-max-pixels 401408
  --text-crop-scale 0.5
  --decode-backend torchair
  --decode-optimization combined_apply_static_actual
  --torchair-cache-dir "$DECODE_CACHE"
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-buckets 128,256,384,512,640,768,1024,1408,1920,2048
  --vision-torchair-cache-dir "$VISION_CACHE"
  --vision-batched-cache-dir "$BATCHED_CACHE"
  --vision-promptfa-align-128
  --vision-mlp-intermediate-size 4352
  --vision-linear-weight-format fractal_nz
  --vision-padding bucket
  --vision-packing greedy --vision-pack-target 1024
  --vision-router-lookahead 32
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312
  --text-packing production_group
  --text-pack-buckets 128,256,512,1024
  --text-pack-max-members 32
  --text-torchair-cache-dir "$TEXT_CACHE"
  --text-packed-cache-dir "$PACKED_CACHE"
  --layout-device npu --no-layout-graph-capture
  --preprocess-all-pages-first --no-timeline
)

printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1 --output-dir "$PREP/compile_output" \
  >"$PREP/compile_command.sh"
printf '\n' >>"$PREP/compile_command.sh"

SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 10800 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1 --output-dir "$PREP/compile_output" \
  2>&1 | tee "$PREP/compile.log"
compile_exit="${PIPESTATUS[0]}"
compile_wall_s="$SECONDS"
set -e
printf '%s\n' "$compile_exit" >"$PREP/compile_exit_code.txt"
printf '%s\n' "$compile_wall_s" >"$PREP/compile_wall_s.txt"
test "$compile_exit" -eq 0
```

The one-page startup is expected to be slow because it prepares every configured
singleton-vision, text-prefill, packed-text, and decode graph.  Do not classify
that compile time as production E2E.  Verify the smoke output and weight format:

```sh
"$PYTHON_BIN" - "$PREP/compile_output/run_summary.json" <<'PY' \
  | tee "$PREP/compile_contract.json"
import json, sys
d = json.load(open(sys.argv[1]))
c = d["configuration"]
r = d["recognition"]
mlp = c["vision_mlp"]
fmt = c["vision_linear_weight_format"]
assert d["result_count"] == d["prediction_count"] == 1
assert c["batch_size"] == 32 and c["cache_length"] == 2048
assert c["max_new_tokens"] == 2048
assert c["decode_optimization"] == "combined_apply_static_actual"
assert c["preprocessor_min_pixels"] == 28224
assert c["preprocessor_max_pixels"] == 401408
assert c["text_crop_scale"] == 0.5
assert c["page_preprocessing_mode"] == "all_before_recognition"
assert c["vision_buckets"] == [128,256,384,512,640,768,1024,1408,1920,2048]
assert c["vision_pack_target"] == 1024
assert mlp == {
    "source_intermediate_size": 4304,
    "target_intermediate_size": 4352,
    "layer_count": 27,
    "zero_extended": True,
}
assert fmt["requested"] == "fractal_nz"
assert fmt["target_format_code"] == 29
assert fmt["linear_weight_count"] == 162
assert fmt["converted_count"] == 162
assert fmt["after_format_histogram"] == {"29": 162}
assert fmt["all_after_are_nz"] is True
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full"}
print(json.dumps({
    "setup_s": d["setup_s"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "result_count": d["result_count"],
    "requests": r["requests"],
    "stop_reasons": r["stop_reason_counts"],
    "vision_mlp": mlp,
    "vision_linear_weight_format": fmt,
}, indent=2))
PY

for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
done

for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  if test -d "$cache"; then
    printf '%s\tfiles=%s\tbytes=%s\n' \
      "$cache" "$(find "$cache" -type f | wc -l)" \
      "$(du -sb "$cache" | cut -f1)"
  else
    printf '%s\tfiles=0\tbytes=0\n' "$cache"
  fi
done | tee "$PREP/cache_after_compile.txt"
```

`BATCHED_CACHE` is allowed to remain empty: this exact lane uses B1 packed
sequences, not B2/B4 batched-vision graphs.

Now rerun the same first page into a different output directory.  This is the
warm-cache gate, not another experiment:

```sh
SECONDS=0
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1 --output-dir "$PREP/warm_output" \
  2>&1 | tee "$PREP/warm.log"
warm_exit="${PIPESTATUS[0]}"
warm_wall_s="$SECONDS"
set -e
printf '%s\n' "$warm_exit" >"$PREP/warm_exit_code.txt"
printf '%s\n' "$warm_wall_s" >"$PREP/warm_wall_s.txt"
test "$warm_exit" -eq 0

"$PYTHON_BIN" - \
  "$PREP/compile_output/run_summary.json" \
  "$PREP/warm_output/run_summary.json" <<'PY' \
  | tee "$PREP/warm_contract.json"
import json, sys

cold, warm = [json.load(open(path)) for path in sys.argv[1:]]
for d in (cold, warm):
    c = d["configuration"]
    fmt = c["vision_linear_weight_format"]
    assert d["result_count"] == d["prediction_count"] == 1
    assert c["batch_size"] == 32 and c["cache_length"] == 2048
    assert c["max_new_tokens"] == 2048
    assert c["decode_optimization"] == "combined_apply_static_actual"
    assert c["preprocessor_max_pixels"] == 401408
    assert c["text_crop_scale"] == 0.5
    assert c["vision_pack_target"] == 1024
    assert c["vision_mlp"]["target_intermediate_size"] == 4352
    assert c["vision_mlp"]["zero_extended"] is True
    assert fmt["converted_count"] == 162
    assert fmt["after_format_histogram"] == {"29": 162}
    assert fmt["all_after_are_nz"] is True
print(json.dumps({
    "compile_setup_s": cold["setup_s"],
    "warm_setup_s": warm["setup_s"],
    "compile_pipeline_e2e_s": cold["pipeline_e2e_s"],
    "warm_pipeline_e2e_s": warm["pipeline_e2e_s"],
    "configuration_exact": cold["configuration"] == warm["configuration"],
}, indent=2))
PY

for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  if test -d "$cache"; then
    printf '%s\tfiles=%s\tbytes=%s\n' \
      "$cache" "$(find "$cache" -type f | wc -l)" \
      "$(du -sb "$cache" | cut -f1)"
  else
    printf '%s\tfiles=0\tbytes=0\n' "$cache"
  fi
done | tee "$ROOT/cache_before_full.txt"

diff -u "$PREP/cache_after_compile.txt" "$ROOT/cache_before_full.txt" \
  | tee "$PREP/cache_compile_to_warm_diff.txt" || true

"$PYTHON_BIN" - \
  "$PREP/compile_output/recognition_trace.jsonl" \
  "$PREP/warm_output/recognition_trace.jsonl" <<'PY' \
  | tee "$PREP/first_page_parity.json"
import json, sys

def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {
        str(row["request_id"]): {
            "token_ids": [int(x) for x in row["token_ids"]],
            "text": row["text"],
        }
        for row in rows
    }

compile_rows = load(sys.argv[1])
warm_rows = load(sys.argv[2])
assert compile_rows == warm_rows
print(json.dumps({
    "request_count": len(compile_rows),
    "request_ids_exact": True,
    "token_ids_exact": True,
    "text_exact": True,
}, indent=2))
PY
```

Validate the warm summary with the same Python contract above.  The parity
script must report identical request IDs, token IDs, and text; compilation/cache
reuse must not change the result.  Review the cache diff too: updated metadata
bytes are acceptable, but a new graph directory or a large file-count increase
means cache preparation was incomplete.

Report `310P PHASE 52 CACHE PREP: PASS` immediately with compile and warm wall
times, both setup times, first-page token parity, the 162/162 format-29 proof,
and cache file counts/bytes.  If either smoke fails, first-page parity fails, a
required cache is empty, or the warm run compiles a missing graph, report the
first causal error and stop before the full run.

### 52.3 Run all 1,651 pages with the frozen configuration

```sh
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 21600 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
run_wall_s="$SECONDS"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
printf '%s\n' "$run_wall_s" >"$LANE/launcher_wall_s.txt"
test "$run_exit" -eq 0
```

The foreground `tee` is the authoritative progress stream.  In another shell:

```sh
tail -n 40 -f "$LANE/run.log"
```

The runner prints `completed=N/1651`; do not enable scheduler tracing or a
timeline merely to get more progress.  If there is no new completion line for
five minutes, record the last 100 log lines and `npu-smi info`, but do not kill
the process unless it reaches the explicit timeout or Luka asks.

Validate the finished run and print the performance result before evaluation:

```sh
"$PYTHON_BIN" - "$OUTPUT/run_summary.json" "$REFERENCE_RUN" <<'PY' \
  | tee "$LANE/compact_summary.json"
import json, sys
d = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
c = d["configuration"]
r = d["recognition"]
s = r["device_stage_s"]
fmt = c["vision_linear_weight_format"]

assert d["offset"] == 0 and d["count"] == 1651
assert d["result_count"] == 1651 and d["prediction_count"] == 1651
assert c["page_preprocessing_mode"] == "all_before_recognition"
assert c["batch_size"] == 32 and c["cache_length"] == 2048
assert c["max_new_tokens"] == 2048
assert c["decode_optimization"] == "combined_apply_static_actual"
assert c["preprocessor_min_pixels"] == 28224
assert c["preprocessor_max_pixels"] == 401408
assert c["text_crop_scale"] == 0.5
assert c["vision_pack_target"] == 1024
assert c["vision_mlp"]["target_intermediate_size"] == 4352
assert c["vision_mlp"]["zero_extended"] is True
assert fmt["converted_count"] == 162
assert fmt["after_format_histogram"] == {"29": 162}
assert fmt["all_after_are_nz"] is True
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full"}

# These are preprocessing/routing contracts, not hardware throughput results.
expected = {
    "requests": 30557,
    "input_tokens": 2560072,
    "projected_image_tokens": 2162831,
    "real_vision_tokens": 8651324,
    "physical_vision_tokens": 9512832,
    "real_text_tokens": 2560072,
    "physical_text_tokens": 3997056,
}
contract_deltas = {k: int(r[k]) - v for k, v in expected.items()}
ref_r = ref["recognition"]
packing_contract = {
    "vision_groups_exact": r["vision_packing"]["groups"] == ref_r["vision_packing"]["groups"],
    "vision_graph_histogram_exact": (
        r["vision_packing"]["graph_shape_histogram"]
        == ref_r["vision_packing"]["graph_shape_histogram"]
    ),
    "vision_no_eager_overflow": r["vision_packing"]["eager_overflow_groups"] == 0,
    "text_groups_exact": r["text_packing"]["groups"] == ref_r["text_packing"]["groups"],
    "text_bucket_histogram_exact": (
        r["text_packing"]["bucket_histogram"]
        == ref_r["text_packing"]["bucket_histogram"]
    ),
    "text_no_fallback": r["text_packing"]["fallback_crops"] == 0,
}

out = {
    "setup_s": d["setup_s"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "pages_per_s": d["pages_per_s"],
    "s_per_page": d["s_per_page"],
    "layout_s": d["layout_frontend"]["stage_s"]["page_total_s"],
    "layout_pages_per_s": 1651 / d["layout_frontend"]["stage_s"]["page_total_s"],
    "ocr_scheduler_wall_s": r["run_scoped_scheduler_wall_s"],
    "requests": r["requests"],
    "contract_deltas_vs_910B": contract_deltas,
    "packing_contract_vs_910B": packing_contract,
    "stop_reasons": r["stop_reason_counts"],
    "vision": {
        "real_tokens": r["real_vision_tokens"],
        "physical_tokens": r["physical_vision_tokens"],
        "seconds": s["vision_prefill"],
        "real_tps": r["real_vision_tokens"] / s["vision_prefill"],
        "physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
    },
    "text": {
        "real_tokens": r["real_text_tokens"],
        "physical_tokens": r["physical_text_tokens"],
        "seconds": s["text_prefill"],
        "real_tps": r["real_text_tokens"] / s["text_prefill"],
        "physical_tps": r["physical_text_tokens"] / s["text_prefill"],
    },
    "decode": {
        "generated_including_eos": r["generated_tokens_including_eos"],
        "effective_tokens": r["effective_decode_tokens"],
        "raw_slots": r["raw_decode_token_slots"],
        "seconds": r["decode_wall_s"],
        "effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
        "raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
    },
    "vision_packing": r["vision_packing"],
    "text_packing": r["text_packing"],
    "device_stage_s": s,
}
print(json.dumps(out, indent=2))
PY
```

Every value in `contract_deltas_vs_910B` should be zero.  A nonzero value means
the page/crop/preprocessing work was not actually identical; report it as a
structural mismatch before interpreting stage throughput.  Still retain the
completed predictions and continue to official evaluation if all 1,651 pages
completed, unless the mismatch is explained by a wrong flag, model, or dataset.

Record cache state and look for compile contamination:

```sh
for cache in \
  "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
  "$TEXT_CACHE" "$PACKED_CACHE"
do
  if test -d "$cache"; then
    printf '%s\tfiles=%s\tbytes=%s\n' \
      "$cache" "$(find "$cache" -type f | wc -l)" \
      "$(du -sb "$cache" | cut -f1)"
  else
    printf '%s\tfiles=0\tbytes=0\n' "$cache"
  fi
done | tee "$ROOT/cache_after_full.txt"
diff -u "$ROOT/cache_before_full.txt" "$ROOT/cache_after_full.txt" \
  | tee "$ROOT/cache_diff.txt" || true
```

Report `310P PHASE 52 E2E: PASS` immediately with setup versus pipeline time,
pages/s, layout time, OCR scheduler wall, vision/text/decode stage seconds and
tok/s, all token totals, packing histograms/fill fractions, stop reasons, cache
diff, and whether the timed full run was cache-warm or compile-contaminated.

### 52.4 Run the guarded official evaluator

Locate the evaluator checkout already used by Phase 49.  Do not clone, update,
patch, or invoke `pdf_validation.py` directly.

```sh
EVALUATOR_ROOT=
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  /home/lukaiv/repos/OmniDocBench_eval \
  "$HOME/repos/OmniDocBench_eval" \
  "$HOME/OmniDocBench_eval" \
  "$HOME/OmniDocBench"
do
  if test -f "$candidate/pdf_validation.py"; then
    EVALUATOR_ROOT="$candidate"
    break
  fi
done
test -n "$EVALUATOR_ROOT"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

cat >"$EVAL/work/config.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 12
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $WORK_SERVER_REPO/$OUTPUT/OmniDocBench_subset.json
    prediction:
      data_path: $WORK_SERVER_REPO/$OUTPUT/predictions
    match_method: quick_match
    match_workers: 24
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

cd "$EVAL/work"
ulimit -n 65536
SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
  "$EVAL_PYTHON" "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --config config.yaml \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 24 --teds-workers 12 \
  --page-timeout-sec 120 \
  --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30 \
  --teds-timeout-sec 120 \
  2>&1 | tee evaluation.log
eval_exit="${PIPESTATUS[0]}"
eval_wall_s="$SECONDS"
set -e
printf '%s\n' "$eval_exit" >../exit_code.txt
printf '%s\n' "$eval_wall_s" >../wall_s.txt
cd "$WORK_SERVER_REPO"
test "$eval_exit" -eq 0
```

The wrapper isolates page matching and TEDS in killable child processes.  Keep
all 1,651 pages.  A bounded fallback/timeout is valid evidence and must be
reported; an unbounded hang, nested-process lifecycle error, or missing result
is an evaluator failure.

```sh
RESULT="$EVAL/work/result"
METRIC="$RESULT/predictions_quick_match_metric_result.json"
EVAL_SUMMARY="$RESULT/predictions_quick_match_run_summary.json"
test -f "$METRIC"
test -f "$EVAL_SUMMARY"

"$EVAL_PYTHON" - "$METRIC" "$EVAL_SUMMARY" <<'PY' \
  | tee "$EVAL/compact_eval_summary.json"
import json, sys
m = json.load(open(sys.argv[1]))
e = json.load(open(sys.argv[2]))
stage = e["stage_execution"]
assert stage["page_match"]["page_count"] == 1651
out = {
    "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
    "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
    "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
    "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    "page_denominators": e["page_denominators"],
    "page_match": stage["page_match"],
    "table_TEDS_execution": stage["metrics"]["table"]["TEDS"],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Report `310P PHASE 52 EVALUATION: PASS` immediately with all six metrics, page
denominators, match fallbacks/timeouts, TEDS samples/timeouts/errors, and wall
time.

### 52.5 Produce the direct 310P-versus-910B report and stop

```sh
"$PYTHON_BIN" - \
  "$REFERENCE_RUN" "$REFERENCE_METRIC" \
  "$OUTPUT/run_summary.json" "$METRIC" <<'PY' \
  | tee "$ROOT/head_to_head.json"
import json, sys
ref_run, ref_metric, run, metric = [json.load(open(p)) for p in sys.argv[1:]]

def perf(d):
    r = d["recognition"]
    s = r["device_stage_s"]
    return {
        "pipeline_e2e_s": d["pipeline_e2e_s"],
        "pages_per_s": d["pages_per_s"],
        "seconds_per_page": d["s_per_page"],
        "layout_s": d["layout_frontend"]["stage_s"]["page_total_s"],
        "layout_pages_per_s": 1651 / d["layout_frontend"]["stage_s"]["page_total_s"],
        "vision_s": s["vision_prefill"],
        "vision_real_tps": r["real_vision_tokens"] / s["vision_prefill"],
        "vision_physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
        "text_s": s["text_prefill"],
        "text_real_tps": r["real_text_tokens"] / s["text_prefill"],
        "text_physical_tps": r["physical_text_tokens"] / s["text_prefill"],
        "decode_s": r["decode_wall_s"],
        "decode_effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
        "decode_raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
        "requests": r["requests"],
        "real_vision_tokens": r["real_vision_tokens"],
        "physical_vision_tokens": r["physical_vision_tokens"],
        "real_text_tokens": r["real_text_tokens"],
        "physical_text_tokens": r["physical_text_tokens"],
        "generated_including_eos": r["generated_tokens_including_eos"],
        "stop_reasons": r["stop_reason_counts"],
    }

def quality(m):
    return {
        "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
        "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
        "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
        "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    }

rp, cp = perf(ref_run), perf(run)
rq, cq = quality(ref_metric), quality(metric)
out = {"performance": {}, "quality": {}}
for name in rp:
    row = {"910B2": rp[name], "310P3": cp[name]}
    if isinstance(rp[name], (int, float)) and isinstance(cp[name], (int, float)):
        row["310P_minus_910B"] = cp[name] - rp[name]
        row["310P_over_910B"] = cp[name] / rp[name] if rp[name] else None
        row["910B_over_310P"] = rp[name] / cp[name] if cp[name] else None
    out["performance"][name] = row
for name in rq:
    delta = cq[name] - rq[name]
    higher = "TEDS" in name
    out["quality"][name] = {
        "910B2": rq[name],
        "310P3": cq[name],
        "310P_minus_910B": delta,
        "direction": "higher_is_better" if higher else "lower_is_better",
        "absolute_delta": abs(delta),
    }
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 52 OPTIMIZED KV2048 FULL: PASS | PREFLIGHT_FAILURE |
CACHE_PREP_FAILURE | E2E_FAILURE | COMPILE_CONTAMINATED |
STRUCTURAL_MISMATCH | EVALUATOR_FAILURE | DATASET_MISMATCH
```

For a pass, include:

1. project/evaluator commits, host, exact NPU, CANN/driver/firmware,
   Python/torch/torch_npu/TorchAir, dataset/model paths, and model SHA;
2. exact compile-smoke, warm-smoke, full-E2E, and evaluator commands;
3. compile versus warm setup/wall times, first-page token parity, cache
   counts/bytes, and proof the timed run introduced no new graphs;
4. proof of B32/KV2048/static-actual decode, max-pixels 401408,
   text-scale 0.5, target-1024 packing, 4304-to-4352 zero extension, and all
   162 weights in format 29;
5. setup, pipeline E2E, pages/s, seconds/page, complete layout breakdown and
   layout pages/s, and OCR scheduler wall;
6. all request/token totals, all device-stage totals, vision/text packing
   histograms and fill fractions, decode slots/calls/useful fraction, private
   cache high-water mark, KV bytes copied, and stop reasons by crop label;
7. exact zero/nonzero preprocessing-contract deltas against 910B;
8. the direct 310P/910B ratio and reciprocal slowdown for pages/s, layout,
   vision real/physical tok/s, text real/physical tok/s, and decode
   effective/raw tok/s;
9. all six official metrics, signed deltas with the correct desired direction,
   denominators, page fallbacks, TEDS samples/timeouts/errors, and evaluator
   wall time;
10. concise `What is proven`, `What differs`, `What remains unresolved`, and
    the first causal error if any stage failed.

Paste `agent_report.md`, `cache_prepare/compile_contract.json`,
`e2e/compact_summary.json`, `evaluation/compact_eval_summary.json`,
`head_to_head.json`, and `cache_diff.txt` back to Luka.  Keep the large
predictions, traces, logs, and compiler caches local on the work server.  Do not
edit source, commit, push, start a different bucket sweep, or begin another
optimization phase.  Then **stop**.

## Phase 53: 310P compiled decode length-mode throughput matrix

### 53.0 Goal, exact scope, and interpretation

Measure the cost of the two compiled masked-GQA workarounds on one Atlas
310P3, against the normal masked-GQA control, at the same static text-decode
shape matrix already measured on one Ascend 910B2:

```text
batch sizes        16, 32, 64, 128
KV/cache length    2048
dtype              fp16
backend            TorchAir, full static graph
active slots       equal to batch size
profile position   1024
warmup             3 complete decode steps
timed steps        30 complete decode steps
```

The three implementation lanes are:

```text
combined_apply
  Normal optimized masked GQA.  No PSE and no actual_seq_lengths.

combined_apply_static_actual
  The same optimized GQA, always passing the compile-time constant
  actual_seq_lengths=[2048] * batch_size.  The boolean mask still carries
  each row's logical prefix.

combined_apply_pse_sentinel
  The same optimized GQA with one always-present PSE graph.  Away from the
  1280 boundary the PSE is all zero.  At effective length 1280, the otherwise
  masked position 1280 is exposed in the boolean mask and suppressed with
  -inf in PSE, preserving the logical attention result while avoiding the
  310P masked-GQA kernel boundary.
```

This phase has two distinct questions:

1. At safe positions, what physical tok/s and full-step latency does each
   length-mode graph achieve?
2. Do static-actual and PSE both synchronize at the exact failing
   `cache_position=1279` / effective-length-1280 boundary for every B?

The normal control must **never** be executed at position 1279 on 310P.  Its
throughput profile begins at position 1024 and the measured steps remain far
below 1279.  Do not infer that normal masked GQA is safe at 1279 from a safe
profile result.

This is a synthetic full-decoder lab, not an E2E OCR run.  The measured call is
the complete optimized production decode step: token embedding, all 18 text
decoder layers, LM head, argmax, KV update, and the decode-arena step.  With
every slot active, physical tok/s and active tok/s are equal.  There is no
scheduler effective-tok/s metric in this phase.

The committed 910B2 reference is:

```text
tmp/09_persistent_page_engine/
  910b_decode_length_modes_b16_128_k2048_24acb27/
    summary.json
    REPORT.md
```

Its measured physical tok/s values are:

| B | normal | static actual | static vs normal | PSE | PSE vs normal |
|---:|---:|---:|---:|---:|---:|
| 16 | 8035.3 | 7552.9 | -6.00% | 7797.0 | -2.97% |
| 32 | 10721.4 | 10514.5 | -1.93% | 10467.3 | -2.37% |
| 64 | 15401.2 | 14904.1 | -3.23% | 15045.1 | -2.31% |
| 128 | 20334.3 | 20853.2 | +2.55% | 20391.7 | +0.28% |

All eight 910B2 workaround boundary gates passed.  The 910B2 runtime reported
`decode_native_fallback`: it did not materialize FRACTAL_NZ decoder weights.
The 310P is expected to report `decode_nz` and all selected decoder weights in
format 29.  Therefore:

- within-310P comparisons between normal/static/PSE are the primary result;
- absolute 310P/910B tok/s ratios are still useful device-level observations,
  but the report must state the different effective weight formats instead of
  pretending the two runtimes used the same internal format.

Execute batch sizes strictly in this order: B16, B32, B64, B128.  Finish all
three safe profiles and both workaround boundary gates for one B, write and
print that batch's compact report, and only then begin the next B.  Do not run
lanes concurrently.  Do not hide progress until the end.

### 53.1 Persistent shell and preflight

The work-server agent is pull-only.  Do not edit tracked files, create a
branch, commit, or push.  Run the whole phase from one persistent shell so
`ASCEND_RT_VISIBLE_DEVICES` cannot change between lanes.

```sh
set -eo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git status --short --branch
test -z "$(git status --porcelain)"
git merge-base --is-ancestor \
  4905f75cf2520549f640ec0c29c8f3846d51b1a1 HEAD

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

PYTHON_BIN=/usr/local/python3.12.13/bin/python
LAB=09_persistent_page_engine/scripts/text_decode_lab.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
REFERENCE=tmp/09_persistent_page_engine/910b_decode_length_modes_b16_128_k2048_24acb27/summary.json
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
ROOT="tmp/09_persistent_page_engine/310p_phase53_decode_length_modes_${COMMIT_SHORT}"
CACHE=".runtime_cache/310p_phase53_decode_length_modes_k2048_${COMMIT_SHORT}"
BATCHES="16 32 64 128"
OPTIMIZATIONS="combined_apply combined_apply_static_actual combined_apply_pse_sentinel"

test -x "$PYTHON_BIN"
test -f "$LAB"
test -f "$MODEL/config.json"
test -f "$MODEL/model.safetensors"
test -f "$REFERENCE"
test ! -e "$ROOT"
test ! -e "$CACHE"
mkdir -p "$ROOT" "$CACHE"

{
  date -Is
  hostname
  printf 'commit=%s\n' "$COMMIT"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
print("torchair", getattr(torchair, "__version__", "unknown"))
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
assert "310P" in torch.npu.get_device_name(torch.npu.current_device())
PY
  npu-smi info
  sha256sum "$MODEL/config.json" "$MODEL/model.safetensors" "$REFERENCE"
  df -h "$WORK_SERVER_REPO"
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 8388608

printf '%s\n' \
  "commit=$COMMIT" \
  "device=$ASCEND_RT_VISIBLE_DEVICES" \
  "model=$MODEL" \
  "reference=$REFERENCE" \
  "batch_sizes=$BATCHES" \
  "cache_length=2048" \
  "profile_position=1024" \
  "warmup=3" \
  "repeats=30" \
  "optimizations=$OPTIMIZATIONS" \
  "cache=$CACHE" \
  >"$ROOT/contract.txt"
```

If checkout cleanliness, required commit ancestry, imports, exact 310P device,
model/reference paths, fresh artifact/cache roots, or the 8-GiB disk check
fails, report `310P PHASE 53 PREFLIGHT: FAILURE` with the first causal error
and stop.  Do not switch Python, model, device, cache length, or reference.

Report `310P PHASE 53 PREFLIGHT: PASS` immediately with commit, host, exact
physical NPU, software versions, checkpoint hash, free disk, and artifact/cache
paths.

### 53.2 Define the profile and boundary helpers

Keep these definitions in the same persistent shell.  Each command has a hard
timeout, a timestamped progress record, its exact shell command, an exit-code
file, and a live `tee` log.  Compilation/first-call time is setup metadata in
`result.json`; it is never the throughput denominator.

```sh
run_profile() {
  B="$1"
  OPT="$2"
  LANE="b${B}_${OPT}"
  DIR="$ROOT/$LANE"
  test ! -e "$DIR"
  mkdir -p "$DIR"

  printf '[%s] PROFILE_BEGIN B=%s optimization=%s\n' \
    "$(date -Is)" "$B" "$OPT" | tee -a "$ROOT/progress.log"
  npu-smi info >"$DIR/npu_before.log" 2>&1

  printf '%q ' \
    timeout --signal=TERM --kill-after=30s 3600 \
    "$PYTHON_BIN" "$LAB" \
    --mode profile --backend torchair --allow-compile \
    --model "$MODEL" --cache-dir "$CACHE" \
    --batch-size "$B" --active-slots "$B" --cache-length 2048 \
    --profile-position 1024 --warmup 3 --repeats 30 \
    --decode-optimization "$OPT" \
    --output "$DIR/result.json" \
    >"$DIR/command.sh"
  printf '\n' >>"$DIR/command.sh"

  set +e
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
    "$PYTHON_BIN" "$LAB" \
    --mode profile --backend torchair --allow-compile \
    --model "$MODEL" --cache-dir "$CACHE" \
    --batch-size "$B" --active-slots "$B" --cache-length 2048 \
    --profile-position 1024 --warmup 3 --repeats 30 \
    --decode-optimization "$OPT" \
    --output "$DIR/result.json" \
    2>&1 | tee "$DIR/run.log"
  CODE="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$CODE" >"$DIR/exit_code.txt"
  npu-smi info >"$DIR/npu_after.log" 2>&1 || true
  printf '[%s] PROFILE_END B=%s optimization=%s exit=%s\n' \
    "$(date -Is)" "$B" "$OPT" "$CODE" | tee -a "$ROOT/progress.log"
  test "$CODE" -eq 0
  test -f "$DIR/result.json"
}

run_boundary() {
  B="$1"
  OPT="$2"
  LANE="boundary_b${B}_${OPT}"
  DIR="$ROOT/$LANE"
  test ! -e "$DIR"
  mkdir -p "$DIR"

  printf '[%s] BOUNDARY_BEGIN B=%s optimization=%s position=1279\n' \
    "$(date -Is)" "$B" "$OPT" | tee -a "$ROOT/progress.log"

  printf '%q ' \
    timeout --signal=TERM --kill-after=30s 300 \
    "$PYTHON_BIN" "$LAB" \
    --mode boundary --backend torchair \
    --model "$MODEL" --cache-dir "$CACHE" \
    --batch-size "$B" --active-slots "$B" --cache-length 2048 \
    --profile-position 1279 \
    --decode-optimization "$OPT" \
    --output "$DIR/result.json" \
    >"$DIR/command.sh"
  printf '\n' >>"$DIR/command.sh"

  set +e
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 300 \
    "$PYTHON_BIN" "$LAB" \
    --mode boundary --backend torchair \
    --model "$MODEL" --cache-dir "$CACHE" \
    --batch-size "$B" --active-slots "$B" --cache-length 2048 \
    --profile-position 1279 \
    --decode-optimization "$OPT" \
    --output "$DIR/result.json" \
    2>&1 | tee "$DIR/run.log"
  CODE="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$CODE" >"$DIR/exit_code.txt"
  printf '[%s] BOUNDARY_END B=%s optimization=%s exit=%s\n' \
    "$(date -Is)" "$B" "$OPT" "$CODE" | tee -a "$ROOT/progress.log"
  test "$CODE" -eq 0
  test -f "$DIR/result.json"
  grep -q '"event": "step_begin"' "$DIR/run.log"
  grep -q '"event": "step_returned"' "$DIR/run.log"
  grep -q '"event": "sync_begin"' "$DIR/run.log"
  grep -q '"event": "sync_end"' "$DIR/run.log"
  if grep -qiE 'sync_error|AICore|5070|RuntimeError|Traceback' "$DIR/run.log"; then
    echo "boundary log contains a runtime error" >&2
    return 1
  fi
}
```

Do not add `--allow-compile` to `run_boundary`: every boundary call must reuse
the graph created by that mode's safe profile.  If it tries to compile, the
cache contract is broken.

### 53.3 Execute and report one complete batch at a time

Run this block once for B16, inspect and report it, then repeat for B32, B64,
and B128.  Do not put the four B values in a background or parallel loop.

```sh
run_one_batch() {
  B="$1"
  run_profile "$B" combined_apply
  run_profile "$B" combined_apply_static_actual
  run_profile "$B" combined_apply_pse_sentinel

  # Never call normal combined_apply at position 1279 on 310P.
  run_boundary "$B" combined_apply_static_actual
  run_boundary "$B" combined_apply_pse_sentinel

  "$PYTHON_BIN" - "$ROOT" "$REFERENCE" "$B" <<'PY' \
    | tee "$ROOT/b${B}_compact_report.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reference = json.load(open(sys.argv[2]))
batch = int(sys.argv[3])
optimizations = (
    "combined_apply",
    "combined_apply_static_actual",
    "combined_apply_pse_sentinel",
)
reference_rows = {
    (int(row["batch_size"]), row["optimization"]): row
    for row in reference["rows"]
}
rows = []
for optimization in optimizations:
    path = root / f"b{batch}_{optimization}" / "result.json"
    payload = json.load(open(path))
    result = payload["result"]
    setup = payload["setup"]
    metadata = setup["runtime_metadata"]
    weight = setup["weight_format"]
    row = {
        "batch_size": batch,
        "optimization": optimization,
        "mean_ms": result["latency_ms"]["mean"],
        "median_ms": result["latency_ms"]["median"],
        "p95_ms": result["latency_ms"]["p95"],
        "min_ms": result["latency_ms"]["min"],
        "max_ms": result["latency_ms"]["max"],
        "physical_tok_per_s": result["throughput"]["raw_physical_tok_per_s"],
        "model_and_argmax_s": result["device_s"]["model_and_argmax"],
        "full_step_s": result["device_s"]["full_production_step"],
        "peak_delta_mib": result["memory_bytes"]["peak_delta"] / 2**20,
        "kv_cache_mib": metadata["cache_allocated_bytes"] / 2**20,
        "compile_first_call_s": setup["runtime_setup_detail_s"]["compile_first_call"],
        "weight_format": metadata["linear_weight_format"],
        "weight_target_format_code": weight["target_format_code"],
        "weight_target_count": weight["target_count"],
        "weight_converted_count": weight["converted_count"],
        "weight_already_nz_count": weight["already_nz_count"],
        "all_after_are_nz": weight["all_after_are_nz"],
        "after_formats_sample": weight["after_formats_sample"],
    }
    ref = reference_rows[(batch, optimization)]
    row["reference_910b_tok_per_s"] = ref["raw_physical_tok_per_s"]
    row["310p_over_910b"] = row["physical_tok_per_s"] / ref["raw_physical_tok_per_s"]
    row["910b_over_310p"] = ref["raw_physical_tok_per_s"] / row["physical_tok_per_s"]
    rows.append(row)

control = rows[0]["physical_tok_per_s"]
for row in rows:
    row["throughput_delta_vs_310p_control_percent"] = (
        row["physical_tok_per_s"] / control - 1.0
    ) * 100.0

boundaries = []
for optimization in optimizations[1:]:
    path = root / f"boundary_b{batch}_{optimization}" / "result.json"
    payload = json.load(open(path))
    result = payload["result"]
    assert result["shape"]["cache_position"] == 1279
    assert result["shape"]["effective_length"] == 1280
    boundaries.append({
        "optimization": optimization,
        "passed": True,
        "elapsed_s": result["elapsed_s"],
    })

assert all(row["weight_format"] == rows[0]["weight_format"] for row in rows)
assert all(row["weight_format"] == "decode_nz" for row in rows)
assert all(row["weight_target_format_code"] == 29 for row in rows)
assert all(row["all_after_are_nz"] for row in rows)
out = {
    "batch_size": batch,
    "cache_length": 2048,
    "safe_profile_position": 1024,
    "rows": rows,
    "boundary_gates": boundaries,
}
print(json.dumps(out, indent=2))
PY
}
```

Execute and report in this exact order:

```sh
run_one_batch 16
```

Immediately report `310P PHASE 53 B16: PASS` with the three latency
distributions, physical tok/s, within-310P deltas, 910B ratios, memory, weight
format, compile/cache state, and both boundary results.  If it passes and no
other process has taken the selected NPU, continue:

If the B16 compact-report assertions show native fallback, any non-29 selected
weight, or `all_after_are_nz=false`, report
`310P PHASE 53 B16: WEIGHT_FORMAT_MISMATCH` and stop.  Do not benchmark later
batch sizes under a different weight-format contract.

```sh
run_one_batch 32
```

Report `310P PHASE 53 B32: PASS` in the same format, then continue:

```sh
run_one_batch 64
```

Report `310P PHASE 53 B64: PASS`.  Before B128, explicitly state that its
GQA KV cache is expected to be 4,608 MiB and show `npu-smi info`.  If another
process has appeared on the selected NPU, stop rather than competing with it.
Otherwise run:

```sh
run_one_batch 128
```

Report `310P PHASE 53 B128: PASS` in the same format.  If B128 OOMs, times out,
or fails compilation, retain B16-B64 as valid results, classify B128 precisely,
and do not reduce B, KV length, model, or repeats to manufacture a pass.

For any failed lane, report its exact command, exit code, last 100 log lines,
first Python/CANN/plog error, NPU state, and whether it failed during model
load, NZ formatting, graph compilation/first call, warmup, timed steps, or
boundary synchronization.  Then stop; do not skip to later batches.

### 53.4 Final cross-platform summary and required report

After all four batches pass, combine the four compact reports:

```sh
"$PYTHON_BIN" - "$ROOT" <<'PY' | tee "$ROOT/summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
batches = [16, 32, 64, 128]
reports = [json.load(open(root / f"b{batch}_compact_report.json")) for batch in batches]
out = {
    "schema_version": 1,
    "kind": "310p_text_decode_length_mode_matrix",
    "batch_reports": reports,
    "all_profile_lanes_passed": True,
    "all_boundary_gates_passed": all(
        boundary["passed"]
        for report in reports
        for boundary in report["boundary_gates"]
    ),
}
print(json.dumps(out, indent=2))
PY

grep -RniE 'Traceback|AICore|5070|RuntimeError|out of memory|timed out' \
  "$ROOT" --include='*.log' >"$ROOT/error_scan.txt" || true
```

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 53 DECODE LENGTH MODES: PASS | PREFLIGHT_FAILURE |
PROFILE_FAILURE | BOUNDARY_FAILURE | WEIGHT_FORMAT_MISMATCH |
B128_RESOURCE_FAILURE | RUNTIME_ERROR
```

For a pass, include:

1. project commit, host, exact physical/logical NPU, CANN/driver/firmware,
   Python, torch, torch_npu, and TorchAir;
2. model/config hashes, committed 910B reference hash, cache root, and proof
   all work was sequential on one NPU;
3. one table containing B, mode, mean/median/p95/min/max latency, physical
   tok/s, percentage versus the same-B 310P normal control, 910B tok/s,
   310P/910B ratio, and reciprocal 910B slowdown;
4. per-lane model-and-argmax/full-step device totals, peak memory delta, exact
   KV-cache bytes, compile-first-call time, and cache directory;
5. effective decoder weight format for every lane, format-29 proof, selected/
   converted/already-NZ counts, and an explicit note that the 910B reference
   used native fallback;
6. all eight boundary results with complete `step_begin`, `step_returned`,
   `sync_begin`, `sync_end` evidence and synchronized elapsed time;
7. a direct answer to which workaround is faster at B16/B32/B64/B128, whether
   its cost changes with B, and whether either workaround ever beats the
   normal safe-position control beyond plausible run noise;
8. `What is proven`, `What is not proven`, and the first causal error if any
   lane failed.

Paste back `agent_report.md`, `summary.json`, the four
`b<B>_compact_report.json` files, and the complete `progress.log`.  Keep raw
logs and compiler caches on the work server.  Do not run E2E OCR, do not test
normal masked GQA at position 1279, do not promote PSE to production, and do
not begin another batch/KV sweep.  Then **stop**.

## Phase 54: B64 PSE plus staged-W8 full production run on 310P

### 54.0 Goal and frozen comparison

Run the exact production configuration that has now completed on one 910B2,
but on one 310P3:

```text
pages                         1,651, offset 0
page frontend                 all pages prepared before OCR begins
layout                        owned NPU frontend, eager model execution,
                              staged CPU pipeline with 4 input workers and
                              8 page-finalization workers
decode                        B64, KV2048, TorchAir,
                              combined_apply_pse_sentinel
global crop preprocessing     min_pixels=28224, max_pixels=401408
text crops                    additional linear scale=0.5
vision                        TorchAir PromptFA, align-128 on 310P,
                              buckets 128,256,384,512,640,768,1024,
                              1408,1920,2048
vision Linear weights         4304 -> 4352 zero extension and FRACTAL_NZ
vision packing                greedy, target 1024, lookahead 32
text prefill                  production-group packing,
                              buckets 128,256,512,1024
timeline/scheduler tracing    disabled
```

This phase answers four questions:

1. Can the Phase-53 B64 PSE cache be promoted into the real production runner
   with at most one decode-only materialization gate, then replayed without
   further cache growth, including the formerly failing effective-length-1280
   boundary?
2. Does B64 improve 310P decode and full-pipeline throughput over the completed
   Phase-52 B32 static-actual run?
3. Does the staged W8 layout frontend transfer to 310P without changing the
   30,557-request page contract?
4. Does official OmniDocBench quality remain consistent with the corrected
   checkpoint and the 910B2 B64-PSE reference?

The committed 910B2 authority is:

```text
tmp/09_persistent_page_engine/
  910b_full_b64_pse_layout8_def2260/full/
```

Its clean, warm-cache result is:

| Metric | 910B2 B64 PSE + W8 |
|---|---:|
| setup | 42.489 s |
| pipeline E2E | 717.705 s |
| pages/s | 2.30039 |
| vision prefill | 178.088 s |
| text prefill | 74.936 s |
| decode wall | 124.087 s |
| raw decode tok/s | 13,372.8 |
| effective decode tok/s | 12,668.8 |
| useful decode slots | 94.735% |
| stop reasons | 30,470 EOS; 29 KV-full; 58 repetition |

The committed official metrics are read from the reference JSON during the
phase.  Do not transcribe or round them by hand.

The prior 310P authority is the completed Phase-52 run.  Phase 54 must compare
against both it and the 910B2 reference.  This is a production validation, not
a new sweep: do not try B32/B128, a different KV length, another layout-worker
count, different pixels, or a different bucket list.

This phase is checkpointed.  Report immediately after preflight, after the
8-page replay gate, after the full E2E run, and after evaluation.  Do not wait
until the end to provide the first update.

### 54.1 Pull, resolve proven caches, and preflight

The work-server agent is pull-only.  Do not edit tracked files, commit, push,
or create branches.  Use one persistent shell for the whole phase.

```sh
set -uo pipefail

WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"
git merge-base --is-ancestor \
  78b3dbf86a3c8e515f2c51775a0598c66b01b725 HEAD

source npu-setup
set -e
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

PYTHON_BIN=/usr/local/python3.12.13/bin/python
EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
LAYOUT_MODEL=/home/lukaiv/models/PP-DocLayoutV3_safetensors
DATASET_JSON=/home/lukaiv/datasets/OmniDocBench/OmniDocBench.json
IMAGES_DIR=/home/lukaiv/datasets/OmniDocBench/images

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
ROOT="tmp/09_persistent_page_engine/310p_phase54_b64_pse_layout8_${COMMIT_SHORT}"
GATE="$ROOT/gate8"
LANE="$ROOT/full"
OUTPUT="$LANE/output"
EVAL="$LANE/evaluation"

REFERENCE_ROOT=tmp/09_persistent_page_engine/910b_full_b64_pse_layout8_def2260/full
REFERENCE_RUN="$REFERENCE_ROOT/output/run_summary.json"
REFERENCE_METRIC="$REFERENCE_ROOT/evaluation/work/result/predictions_quick_match_metric_result.json"
REFERENCE_EVAL_SUMMARY="$REFERENCE_ROOT/evaluation/work/result/predictions_quick_match_run_summary.json"

PHASE52_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase52_opt_kv2048_*' | sort | tail -n 1)"
test -n "$PHASE52_ROOT"
test -f "$PHASE52_ROOT/agent_report.md"
rg -ni 'phase 52.*pass|classification.*pass' "$PHASE52_ROOT/agent_report.md"
PHASE52_RUN="$("$PYTHON_BIN" - "$PHASE52_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
matches = []
for path in root.rglob("run_summary.json"):
    try:
        payload = json.load(open(path))
    except Exception:
        continue
    config = payload.get("configuration", {})
    if (
        payload.get("count") == 1651
        and payload.get("result_count") == 1651
        and config.get("batch_size") == 32
        and config.get("decode_optimization") == "combined_apply_static_actual"
    ):
        matches.append(path)
assert len(matches) == 1, matches
print(matches[0])
PY
)"
PHASE52_METRIC="$(find "$PHASE52_ROOT" -type f \
  -name 'predictions_quick_match_metric_result.json' | sort | tail -n 1)"
test -f "$PHASE52_RUN"
test -f "$PHASE52_METRIC"

PHASE53_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase53_decode_length_modes_*' | sort | tail -n 1)"
test -n "$PHASE53_ROOT"
test -f "$PHASE53_ROOT/agent_report.md"
test -f "$PHASE53_ROOT/contract.txt"
test -f "$PHASE53_ROOT/b64_combined_apply_pse_sentinel/result.json"
test -f "$PHASE53_ROOT/boundary_b64_combined_apply_pse_sentinel/result.json"
rg -ni 'phase 53.*pass|classification.*pass' "$PHASE53_ROOT/agent_report.md"

DECODE_CACHE="$(awk -F= '$1 == "cache" {print substr($0, index($0,"=")+1)}' \
  "$PHASE53_ROOT/contract.txt")"
VISION_CACHE=.runtime_cache/310p_phase52_v16_vision_4352_nz
BATCHED_CACHE=.runtime_cache/310p_phase52_v16_vision_batched_unused
TEXT_CACHE=.runtime_cache/310p_phase52_v16_text
PACKED_CACHE=.runtime_cache/310p_phase52_v16_text_packed

test -n "$DECODE_CACHE"
for cache in "$DECODE_CACHE" "$VISION_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
done

test ! -e "$ROOT"
mkdir -p "$GATE" "$LANE" "$EVAL/work"
test -x "$PYTHON_BIN"
test -x "$EVAL_PYTHON"
test -f "$E2E"
test -f "$EVAL_WRAPPER"
test -f "$MODEL/model.safetensors"
test -d "$LAYOUT_MODEL"
test -f "$DATASET_JSON"
test -d "$IMAGES_DIR"
test -f "$REFERENCE_RUN"
test -f "$REFERENCE_METRIC"
test -f "$REFERENCE_EVAL_SUMMARY"

test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db

{
  date -Is
  hostname
  printf 'commit=%s\n' "$COMMIT"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'phase52_root=%s\n' "$PHASE52_ROOT"
  printf 'phase53_root=%s\n' "$PHASE53_ROOT"
  printf 'decode_cache=%s\n' "$DECODE_CACHE"
  printf 'vision_cache=%s\n' "$VISION_CACHE"
  printf 'text_cache=%s\n' "$TEXT_CACHE"
  printf 'packed_cache=%s\n' "$PACKED_CACHE"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import torchair

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("torch_npu_git", getattr(torch_npu.version, "git_version", None))
print("torchair", getattr(torchair, "__version__", "unknown"))
print("logical_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
assert "310P" in torch.npu.get_device_name(torch.npu.current_device())
PY
  npu-smi info
  sha256sum \
    "$MODEL/model.safetensors" \
    "$PHASE52_RUN" "$PHASE52_METRIC" \
    "$PHASE53_ROOT/b64_combined_apply_pse_sentinel/result.json" \
    "$PHASE53_ROOT/boundary_b64_combined_apply_pse_sentinel/result.json" \
    "$REFERENCE_RUN" "$REFERENCE_METRIC" "$REFERENCE_EVAL_SUMMARY"
  df -h "$WORK_SERVER_REPO" /home/lukaiv/models
} 2>&1 | tee "$ROOT/preflight.log"

available_kb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {print $4}')"
test "$available_kb" -ge 8388608

"$PYTHON_BIN" - \
  "$PHASE53_ROOT/b64_combined_apply_pse_sentinel/result.json" \
  "$PHASE53_ROOT/boundary_b64_combined_apply_pse_sentinel/result.json" <<'PY' \
  | tee "$ROOT/phase53_b64_contract.json"
import json, sys
profile, boundary = [json.load(open(path)) for path in sys.argv[1:]]
pr = profile["result"]
br = boundary["result"]
assert pr["shape"]["batch_size"] == 64
assert pr["shape"]["cache_length"] == 2048
assert profile["setup"]["runtime_metadata"]["decode_optimization"] == "combined_apply_pse_sentinel"
assert br["shape"]["cache_position"] == 1279
assert br["shape"]["effective_length"] == 1280
print(json.dumps({
    "profile_physical_tok_per_s": pr["throughput"]["raw_physical_tok_per_s"],
    "boundary_cache_position": br["shape"]["cache_position"],
    "boundary_effective_length": br["shape"]["effective_length"],
    "boundary_elapsed_s": br["elapsed_s"],
}, indent=2))
PY
```

If the checkout is dirty, required commit ancestry fails, the v1.6 hash is
wrong, Phase 52 or Phase 53 is not a pass, the Phase-53 B64 PSE boundary result
is absent, any proven cache is empty, the selected device is not a 310P3, or
less than 8 GiB is free, report `310P PHASE 54 PREFLIGHT: FAILURE` with the
first causal error and stop.  Do not delete caches or switch devices silently.

Otherwise report `310P PHASE 54 PREFLIGHT: PASS` immediately with commit,
host, exact NPU, software versions, checkpoint hash, free disk, Phase-52 and
Phase-53 roots, all cache paths, the Phase-53 B64 PSE tok/s, and its successful
1279/1280 boundary gate.

### 54.2 Freeze one production argument array and run two 8-page cache gates

The gate and full run must use this same array.  Only `--limit` and
`--output-dir` may differ.

```sh
PRODUCTION_ARGS=(
  "$E2E"
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL"
  --recognizer-model "$MODEL"
  --batch-size 64 --cache-length 2048 --max-new-tokens 2048
  --preprocessor-min-pixels 28224
  --preprocessor-max-pixels 401408
  --text-crop-scale 0.5
  --decode-backend torchair
  --decode-optimization combined_apply_pse_sentinel
  --torchair-cache-dir "$DECODE_CACHE"
  --vision-backend torchair
  --vision-attention prompt_flash_attention
  --vision-buckets 128,256,384,512,640,768,1024,1408,1920,2048
  --vision-torchair-cache-dir "$VISION_CACHE"
  --vision-batched-cache-dir "$BATCHED_CACHE"
  --vision-promptfa-align-128
  --vision-mlp-intermediate-size 4352
  --vision-linear-weight-format fractal_nz
  --vision-padding bucket
  --vision-packing greedy --vision-pack-target 1024
  --vision-router-lookahead 32
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312
  --text-packing production_group
  --text-pack-buckets 128,256,512,1024
  --text-pack-max-members 32
  --text-torchair-cache-dir "$TEXT_CACHE"
  --text-packed-cache-dir "$PACKED_CACHE"
  --layout-device npu --no-layout-graph-capture
  --preprocess-all-pages-first --layout-workers 8
  --no-timeline
)

record_caches() {
  OUT="$1"
  : >"$OUT"
  for cache in \
    "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
    "$TEXT_CACHE" "$PACKED_CACHE"
  do
    if test -d "$cache"; then
      printf '%s\tfiles=%s\tbytes=%s\n' \
        "$cache" "$(find "$cache" -type f | wc -l)" \
        "$(du -sb "$cache" | cut -f1)" >>"$OUT"
    else
      printf '%s\tfiles=0\tbytes=0\n' "$cache" >>"$OUT"
    fi
  done
}

record_caches "$GATE/cache_before.txt"
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 8 --output-dir "$GATE/output" \
  >"$GATE/command.sh"
printf '\n' >>"$GATE/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 1800 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 8 --output-dir "$GATE/output" \
  2>&1 | tee "$GATE/run.log"
gate_exit="${PIPESTATUS[0]}"
gate_wall_s="$SECONDS"
set -e
printf '%s\n' "$gate_exit" >"$GATE/exit_code.txt"
printf '%s\n' "$gate_wall_s" >"$GATE/wall_s.txt"
record_caches "$GATE/cache_after.txt"
test "$gate_exit" -eq 0

"$PYTHON_BIN" - "$GATE/output/run_summary.json" <<'PY' \
  | tee "$GATE/contract.json"
import json, sys
d = json.load(open(sys.argv[1]))
c = d["configuration"]
r = d["recognition"]
w = d["layout_frontend"]["worker_pipeline"]
fmt = c["vision_linear_weight_format"]
assert d["offset"] == 0 and d["count"] == 8
assert d["result_count"] == d["prediction_count"] == 8
assert d["page_preprocessing_mode"] == "all_before_recognition"
assert d["layout_workers"] == 8
assert w == {
    "strategy": "bounded_staged_cpu_pipeline",
    "input_workers": 4,
    "page_prepare_workers": 8,
    "max_inflight_pages": 8,
}
assert c["batch_size"] == 64 and c["cache_length"] == 2048
assert c["max_new_tokens"] == 2048
assert c["decode_optimization"] == "combined_apply_pse_sentinel"
assert c["preprocessor_min_pixels"] == 28224
assert c["preprocessor_max_pixels"] == 401408
assert c["text_crop_scale"] == 0.5
assert c["vision_buckets"] == [128,256,384,512,640,768,1024,1408,1920,2048]
assert c["vision_pack_target"] == 1024
assert c["vision_mlp"] == {
    "source_intermediate_size": 4304,
    "target_intermediate_size": 4352,
    "layer_count": 27,
    "zero_extended": True,
}
assert fmt["target_format_code"] == 29
assert fmt["linear_weight_count"] == 162
assert fmt["after_format_histogram"] == {"29": 162}
assert fmt["all_after_are_nz"] is True
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full", "repetition"}
compile_s = d["recognizer_setup_timing_s"]["compile_first_call"]
print(json.dumps({
    "setup_s": d["setup_s"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "pages_per_s": d["pages_per_s"],
    "compile_first_call_s": compile_s,
    "result_count": d["result_count"],
    "requests": r["requests"],
    "worker_pipeline": w,
    "stop_reasons": r["stop_reason_counts"],
    "decode_raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
    "decode_effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
}, indent=2))
PY

"$PYTHON_BIN" - \
  "$GATE/cache_before.txt" "$GATE/cache_after.txt" "$DECODE_CACHE" <<'PY' \
  | tee "$GATE/cache_contract.json"
import json, sys

def load(path):
    rows = {}
    for line in open(path):
        cache, files, size = line.rstrip().split("\t")
        rows[cache] = {
            "files": int(files.split("=", 1)[1]),
            "bytes": int(size.split("=", 1)[1]),
        }
    return rows

before, after = map(load, sys.argv[1:3])
decode_cache = sys.argv[3]
assert before.keys() == after.keys()
deltas = {
    cache: {
        "file_delta": after[cache]["files"] - before[cache]["files"],
        "byte_delta": after[cache]["bytes"] - before[cache]["bytes"],
    }
    for cache in before
}
prefill_growth = {
    cache: row for cache, row in deltas.items()
    if cache != decode_cache and row["file_delta"] != 0
}
assert not prefill_growth, prefill_growth
print(json.dumps({
    "cache_deltas": deltas,
    "decode_only_materialization_allowed": True,
    "prefill_caches_added_no_files": True,
}, indent=2))
PY

# A slow first call is not by itself a cache miss.  On the 910B2 authority,
# the first production gate spent 19.882 s in compile_first_call while the
# following process spent 0.228 s.  The first gate may materialize files only
# under DECODE_CACHE.  Now run the identical gate again and require every
# cache's file count to remain fixed.
WARM_GATE="$ROOT/gate8_warm"
mkdir -p "$WARM_GATE"
record_caches "$WARM_GATE/cache_before.txt"
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 8 --output-dir "$WARM_GATE/output" \
  >"$WARM_GATE/command.sh"
printf '\n' >>"$WARM_GATE/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 1800 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 8 --output-dir "$WARM_GATE/output" \
  2>&1 | tee "$WARM_GATE/run.log"
warm_gate_exit="${PIPESTATUS[0]}"
warm_gate_wall_s="$SECONDS"
set -e
printf '%s\n' "$warm_gate_exit" >"$WARM_GATE/exit_code.txt"
printf '%s\n' "$warm_gate_wall_s" >"$WARM_GATE/wall_s.txt"
record_caches "$WARM_GATE/cache_after.txt"
test "$warm_gate_exit" -eq 0

"$PYTHON_BIN" - "$WARM_GATE/output/run_summary.json" <<'PY' \
  | tee "$WARM_GATE/contract.json"
import json, sys
d = json.load(open(sys.argv[1]))
c = d["configuration"]
w = d["layout_frontend"]["worker_pipeline"]
assert d["offset"] == 0 and d["count"] == 8
assert d["result_count"] == d["prediction_count"] == 8
assert d["page_preprocessing_mode"] == "all_before_recognition"
assert d["layout_workers"] == 8
assert w["input_workers"] == 4 and w["page_prepare_workers"] == 8
assert c["batch_size"] == 64 and c["cache_length"] == 2048
assert c["decode_optimization"] == "combined_apply_pse_sentinel"
assert c["vision_linear_weight_format"]["after_format_histogram"] == {"29": 162}
print(json.dumps({
    "setup_s": d["setup_s"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "pages_per_s": d["pages_per_s"],
    "compile_first_call_s": d["recognizer_setup_timing_s"]["compile_first_call"],
    "result_count": d["result_count"],
    "worker_pipeline": w,
}, indent=2))
PY

"$PYTHON_BIN" - \
  "$WARM_GATE/cache_before.txt" "$WARM_GATE/cache_after.txt" <<'PY' \
  | tee "$WARM_GATE/cache_contract.json"
import json, sys

def load(path):
    rows = {}
    for line in open(path):
        cache, files, size = line.rstrip().split("\t")
        rows[cache] = {
            "files": int(files.split("=", 1)[1]),
            "bytes": int(size.split("=", 1)[1]),
        }
    return rows

before, after = map(load, sys.argv[1:])
assert before.keys() == after.keys()
deltas = {
    cache: {
        "file_delta": after[cache]["files"] - before[cache]["files"],
        "byte_delta": after[cache]["bytes"] - before[cache]["bytes"],
    }
    for cache in before
}
assert all(row["file_delta"] == 0 for row in deltas.values()), deltas
print(json.dumps({"cache_deltas": deltas, "no_new_files": True}, indent=2))
PY
```

Report `310P PHASE 54 GATE8: PASS` immediately with both gates' wall/setup/E2E,
requests, worker-pipeline proof, both compile-first-call times, decode tok/s,
stop reasons, and both cache-delta reports.  The first gate may add files only
inside `DECODE_CACHE`; that is the bounded production materialization this gate
exists to perform.  The second gate must add zero files everywhere.  A slow
`compile_first_call` without file growth is cache loading and must be reported,
not mislabeled as compilation.  Any prefill-cache growth in gate one or any
file growth in gate two is `310P PHASE 54 GATE8: CACHE_MISS`; preserve the
outputs and stop before the full run.  Do not create a different cache root.

### 54.3 Full 1,651-page production run

Before launching, show `npu-smi info` and verify no other process appeared on
the selected NPU.  Then:

```sh
record_caches "$LANE/cache_before.txt"
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  >"$LANE/command.sh"
printf '\n' >>"$LANE/command.sh"

SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 10800 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$OUTPUT" \
  2>&1 | tee "$LANE/run.log"
run_exit="${PIPESTATUS[0]}"
run_wall_s="$SECONDS"
set -e
printf '%s\n' "$run_exit" >"$LANE/exit_code.txt"
printf '%s\n' "$run_wall_s" >"$LANE/launcher_wall_s.txt"
record_caches "$LANE/cache_after.txt"
test "$run_exit" -eq 0
```

The foreground log is the progress source.  From another shell:

```sh
tail -n 60 -f "$LANE/run.log"
```

The runner prints completed-page progress.  Do not enable the timeline or
scheduler diagnostics.  If no page completes for five minutes, capture the
last 120 log lines and `npu-smi info`; do not broadly kill Python processes.

Validate and summarize the completed run:

```sh
"$PYTHON_BIN" - "$OUTPUT/run_summary.json" "$REFERENCE_RUN" <<'PY' \
  | tee "$LANE/compact_summary.json"
import json, sys
d, ref = [json.load(open(path)) for path in sys.argv[1:]]
c = d["configuration"]
r = d["recognition"]
s = r["device_stage_s"]
w = d["layout_frontend"]["worker_pipeline"]
fmt = c["vision_linear_weight_format"]

assert d["offset"] == 0 and d["count"] == 1651
assert d["result_count"] == d["prediction_count"] == 1651
assert d["page_preprocessing_mode"] == "all_before_recognition"
assert d["layout_workers"] == 8
assert w["input_workers"] == 4 and w["page_prepare_workers"] == 8
assert c["batch_size"] == 64 and c["cache_length"] == 2048
assert c["max_new_tokens"] == 2048
assert c["decode_optimization"] == "combined_apply_pse_sentinel"
assert c["preprocessor_min_pixels"] == 28224
assert c["preprocessor_max_pixels"] == 401408
assert c["text_crop_scale"] == 0.5
assert c["vision_pack_target"] == 1024
assert c["vision_mlp"]["target_intermediate_size"] == 4352
assert c["vision_mlp"]["zero_extended"] is True
assert fmt["after_format_histogram"] == {"29": 162}
assert fmt["all_after_are_nz"] is True
assert set(r["stop_reason_counts"]) <= {"eos", "kv_cache_full", "repetition"}

expected = {
    "requests": 30557,
    "input_tokens": 2560072,
    "projected_image_tokens": 2162831,
    "real_vision_tokens": 8651324,
    "physical_vision_tokens": 9512832,
    "real_text_tokens": 2560072,
    "physical_text_tokens": 3997056,
}
contract_deltas = {key: int(r[key]) - value for key, value in expected.items()}
assert all(value == 0 for value in contract_deltas.values()), contract_deltas

ref_r = ref["recognition"]
packing = {
    "vision_groups_exact": r["vision_packing"]["groups"] == ref_r["vision_packing"]["groups"],
    "vision_histogram_exact": r["vision_packing"]["graph_shape_histogram"] == ref_r["vision_packing"]["graph_shape_histogram"],
    "vision_no_eager_overflow": r["vision_packing"]["eager_overflow_groups"] == 0,
    "text_groups_exact": r["text_packing"]["groups"] == ref_r["text_packing"]["groups"],
    "text_histogram_exact": r["text_packing"]["bucket_histogram"] == ref_r["text_packing"]["bucket_histogram"],
    "text_no_fallback": r["text_packing"]["fallback_crops"] == 0,
}
assert all(packing.values()), packing

out = {
    "setup_s": d["setup_s"],
    "compile_first_call_s": d["recognizer_setup_timing_s"]["compile_first_call"],
    "pipeline_e2e_s": d["pipeline_e2e_s"],
    "pages_per_s": d["pages_per_s"],
    "s_per_page": d["s_per_page"],
    "layout_workers": d["layout_workers"],
    "layout_worker_pipeline": w,
    "layout_stage_s": d["layout_frontend"]["stage_s"],
    "ocr_scheduler_wall_s": r["run_scoped_scheduler_wall_s"],
    "requests": r["requests"],
    "contract_deltas_vs_910b": contract_deltas,
    "packing_contract_vs_910b": packing,
    "stop_reasons": r["stop_reason_counts"],
    "vision": {
        "real_tokens": r["real_vision_tokens"],
        "physical_tokens": r["physical_vision_tokens"],
        "seconds": s["vision_prefill"],
        "real_tps": r["real_vision_tokens"] / s["vision_prefill"],
        "physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
    },
    "text": {
        "real_tokens": r["real_text_tokens"],
        "physical_tokens": r["physical_text_tokens"],
        "seconds": s["text_prefill"],
        "real_tps": r["real_text_tokens"] / s["text_prefill"],
        "physical_tps": r["physical_text_tokens"] / s["text_prefill"],
    },
    "decode": {
        "generated_including_eos": r["generated_tokens_including_eos"],
        "effective_tokens": r["effective_decode_tokens"],
        "raw_slots": r["raw_decode_token_slots"],
        "active_slots": r["active_decode_token_slots"],
        "idle_slots": r["idle_decode_token_slots"],
        "lookahead_slots": r["lookahead_decode_token_slots"],
        "graph_calls": r["decode_graph_calls"],
        "seconds": r["decode_wall_s"],
        "effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
        "raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
        "useful_fraction": r["decode_useful_token_fraction"],
    },
    "device_stage_s": s,
    "vision_packing": r["vision_packing"],
    "text_packing": r["text_packing"],
}
print(json.dumps(out, indent=2))
PY

"$PYTHON_BIN" - "$LANE/cache_before.txt" "$LANE/cache_after.txt" <<'PY' \
  | tee "$LANE/cache_contract.json"
import json, sys

def load(path):
    rows = {}
    for line in open(path):
        cache, files, size = line.rstrip().split("\t")
        rows[cache] = {
            "files": int(files.split("=", 1)[1]),
            "bytes": int(size.split("=", 1)[1]),
        }
    return rows

before, after = map(load, sys.argv[1:])
deltas = {
    cache: {
        "file_delta": after[cache]["files"] - before[cache]["files"],
        "byte_delta": after[cache]["bytes"] - before[cache]["bytes"],
    }
    for cache in before
}
assert all(row["file_delta"] == 0 for row in deltas.values()), deltas
print(json.dumps({"cache_deltas": deltas, "no_new_files": True}, indent=2))
PY
```

Report `310P PHASE 54 E2E: PASS` immediately with setup and launcher wall,
pipeline E2E, pages/s, seconds/page, complete layout stage totals, OCR scheduler
wall, every device-stage total, vision/text real and physical tok/s, decode raw
and effective tok/s, useful fraction, graph calls and slot counts, packing
histograms/fill fractions, stop reasons, private-cache high-water/bytes, and
cache deltas.  State explicitly whether the run was warm or compile
contaminated.

### 54.4 Guarded official evaluation

Use the permanent evaluator wrapper.  Do not invoke `pdf_validation.py`
directly and do not exclude any page.

```sh
EVALUATOR_ROOT=
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  /home/lukaiv/repos/OmniDocBench_eval \
  "$HOME/repos/OmniDocBench_eval" \
  "$HOME/OmniDocBench_eval" \
  "$HOME/OmniDocBench"
do
  if test -f "$candidate/pdf_validation.py"; then
    EVALUATOR_ROOT="$candidate"
    break
  fi
done
test -n "$EVALUATOR_ROOT"
test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = \
  2b161d010d2e3aff77a0edef359ea3a6411d23cd

cat >"$EVAL/work/config.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 12
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $WORK_SERVER_REPO/$OUTPUT/OmniDocBench_subset.json
    prediction:
      data_path: $WORK_SERVER_REPO/$OUTPUT/predictions
    match_method: quick_match
    match_workers: 24
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

cd "$EVAL/work"
ulimit -n 65536
SECONDS=0
set +e
set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
  "$EVAL_PYTHON" "$WORK_SERVER_REPO/$EVAL_WRAPPER" \
  --config config.yaml \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 24 --teds-workers 12 \
  --page-timeout-sec 120 \
  --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30 \
  --teds-timeout-sec 120 \
  2>&1 | tee evaluation.log
eval_exit="${PIPESTATUS[0]}"
eval_wall_s="$SECONDS"
set -e
printf '%s\n' "$eval_exit" >../exit_code.txt
printf '%s\n' "$eval_wall_s" >../wall_s.txt
cd "$WORK_SERVER_REPO"
test "$eval_exit" -eq 0

RESULT="$EVAL/work/result"
METRIC="$RESULT/predictions_quick_match_metric_result.json"
EVAL_SUMMARY="$RESULT/predictions_quick_match_run_summary.json"
test -f "$METRIC"
test -f "$EVAL_SUMMARY"

"$EVAL_PYTHON" - "$METRIC" "$EVAL_SUMMARY" <<'PY' \
  | tee "$EVAL/compact_eval_summary.json"
import json, sys
m, e = [json.load(open(path)) for path in sys.argv[1:]]
stage = e["stage_execution"]
assert stage["page_match"]["page_count"] == 1651
print(json.dumps({
    "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
    "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
    "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
    "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
    "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    "page_denominators": e["page_denominators"],
    "page_match": stage["page_match"],
    "table_TEDS_execution": stage["metrics"]["table"]["TEDS"],
}, indent=2, ensure_ascii=False))
PY
```

Report `310P PHASE 54 EVALUATION: PASS` immediately with wall time, all six
metrics, page denominators, match fallbacks/timeouts, and TEDS
samples/timeouts/errors.  Timeouts handled by the wrapper are evidence and must
be reported; an unbounded hang or missing final result is a failure.

### 54.5 Three-way report: 310P before, 310P after, and 910B2 reference

```sh
"$PYTHON_BIN" - \
  "$PHASE52_RUN" "$PHASE52_METRIC" \
  "$OUTPUT/run_summary.json" "$METRIC" \
  "$REFERENCE_RUN" "$REFERENCE_METRIC" <<'PY' \
  | tee "$ROOT/head_to_head.json"
import json, sys
old_run, old_metric, new_run, new_metric, ref_run, ref_metric = [
    json.load(open(path)) for path in sys.argv[1:]
]

def performance(d):
    r = d["recognition"]
    s = r["device_stage_s"]
    return {
        "pipeline_e2e_s": d["pipeline_e2e_s"],
        "pages_per_s": d["pages_per_s"],
        "seconds_per_page": d["s_per_page"],
        "layout_workers": d["layout_workers"],
        "layout_summed_page_stage_s": d["layout_frontend"]["stage_s"]["page_total_s"],
        "vision_s": s["vision_prefill"],
        "vision_real_tps": r["real_vision_tokens"] / s["vision_prefill"],
        "vision_physical_tps": r["physical_vision_tokens"] / s["vision_prefill"],
        "text_s": s["text_prefill"],
        "text_real_tps": r["real_text_tokens"] / s["text_prefill"],
        "text_physical_tps": r["physical_text_tokens"] / s["text_prefill"],
        "decode_s": r["decode_wall_s"],
        "decode_effective_tps": r["effective_decode_tokens"] / r["decode_wall_s"],
        "decode_raw_tps": r["raw_decode_token_slots"] / r["decode_wall_s"],
        "decode_useful_fraction": r["decode_useful_token_fraction"],
        "generated_including_eos": r["generated_tokens_including_eos"],
        "stop_reasons": r["stop_reason_counts"],
    }

def quality(m):
    return {
        "text_block_Edit_dist": m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
        "display_formula_Edit_dist": m["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_Edit_dist": m["table"]["all"]["Edit_dist"]["ALL_page_avg"],
        "table_TEDS": m["table"]["page"]["TEDS"]["ALL"],
        "table_TEDS_structure_only": m["table"]["page"]["TEDS_structure_only"]["ALL"],
        "reading_order_Edit_dist": m["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    }

perf = {
    "310P_phase52_B32_static_actual": performance(old_run),
    "310P_phase54_B64_PSE_W8": performance(new_run),
    "910B2_B64_PSE_W8": performance(ref_run),
}
qual = {
    "310P_phase52_B32_static_actual": quality(old_metric),
    "310P_phase54_B64_PSE_W8": quality(new_metric),
    "910B2_B64_PSE_W8": quality(ref_metric),
}

def deltas(a, b):
    out = {}
    for key, av in a.items():
        bv = b[key]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = {
                "from": av,
                "to": bv,
                "absolute": bv - av,
                "ratio": bv / av if av else None,
                "percent": (bv / av - 1.0) * 100.0 if av else None,
            }
    return out

out = {
    "performance": perf,
    "quality": qual,
    "phase54_vs_phase52_performance": deltas(
        perf["310P_phase52_B32_static_actual"],
        perf["310P_phase54_B64_PSE_W8"],
    ),
    "310P_vs_910B2_performance": deltas(
        perf["910B2_B64_PSE_W8"],
        perf["310P_phase54_B64_PSE_W8"],
    ),
    "phase54_vs_phase52_quality": deltas(
        qual["310P_phase52_B32_static_actual"],
        qual["310P_phase54_B64_PSE_W8"],
    ),
    "310P_vs_910B2_quality": deltas(
        qual["910B2_B64_PSE_W8"],
        qual["310P_phase54_B64_PSE_W8"],
    ),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

grep -RniE 'Traceback|AICore|5070|RuntimeError|out of memory|timed out' \
  "$ROOT" --include='*.log' >"$ROOT/error_scan.txt" || true
```

Write `$ROOT/agent_report.md` beginning with exactly one classification:

```text
310P PHASE 54 B64 PSE LAYOUT8 FULL: PASS | PREFLIGHT_FAILURE |
GATE_CACHE_MISS | GATE_RUNTIME_FAILURE | E2E_FAILURE | OOM_FAILURE |
COMPILE_CONTAMINATED | STRUCTURAL_MISMATCH | EVALUATOR_FAILURE
```

For a pass, include:

1. project/evaluator commits, host, exact physical/logical NPU,
   CANN/driver/firmware, Python/torch/torch_npu/TorchAir, dataset/model paths,
   and model SHA;
2. exact Phase-52, Phase-53, decode-cache, prefill-cache, and committed 910B2
   reference paths;
3. exact gate, full-E2E, and evaluator commands;
4. gate wall/setup/E2E, worker-pipeline proof, compile-first-call time, cache
   deltas, and proof the Phase-53 B64 PSE graph replayed without compilation;
5. setup, pipeline E2E, pages/s, seconds/page, launcher wall, all layout stage
   totals, and OCR scheduler wall;
6. every request/token total, every device-stage total, vision/text packing
   histograms and fill fractions, private-cache capacity/high-water/bytes,
   decode calls/slots/useful fraction, KV bytes copied, and stop reasons by
   crop label;
7. exact preprocessing and packing contract comparison against 910B2;
8. Phase-54 versus Phase-52 310P deltas for E2E, pages/s, vision/text times and
   tok/s, decode time/raw/effective tok/s/useful fraction, and layout;
9. Phase-54 310P versus 910B2 ratios and reciprocal slowdowns for those same
   performance metrics;
10. all six official metrics for all three runs, signed deltas with the
    correct better direction, denominators, page fallbacks, TEDS
    samples/timeouts/errors, and evaluator wall;
11. `What is proven`, `What differs`, `What remains unresolved`, and the first
    causal error if anything failed.

Paste back `agent_report.md`, `gate8/contract.json`,
`gate8/cache_contract.json`, `full/compact_summary.json`,
`full/cache_contract.json`, `full/evaluation/compact_eval_summary.json`, and
`head_to_head.json`.  Keep large predictions, logs, and compiler caches on the
work server.  Do not edit source, create another cache, change B/KV/pixels,
start another sweep, or rerun a failed full job under altered settings.  Then
**stop**.

## Phase 55: dense vision buckets and 768-token packing target

### 55.0 Goal and fixed comparison

Repeat the Phase-54 production run with exactly two vision-routing changes:

```text
vision pack target      1024 -> 768
vision bucket ladder    128,256,384,512,640,768,896,1024,
                        1152,1280,1408,1536,1664,1792,1920,2048
```

Everything else remains fixed: all 1,651 pages are prepared before OCR; staged
layout uses four input workers and eight page-finalization workers; decode is
B64/KV2048 with the PSE sentinel workaround; global pixels are
28224..401408; text crops use scale 0.5; vision uses compiled PromptFA, the
4352-wide zero-extended MLP, and FRACTAL_NZ weights; text prefill uses
production-group packing; timeline tracing stays off.

The denser ladder does not resize, clip, or discard any crop.  It only reduces
padding for a singleton crop or packed group whose real length lies between
two old graph shapes.  Crops larger than the 768-token packing target remain
singleton groups and route to the smallest fitting dense bucket.

This phase permits one deliberate vision-cache materialization gate because
896, 1152, 1280, 1536, 1664, and 1792 are new graph shapes.  It then requires
an identical warm gate with zero cache growth before the full run.  Do not call
the first gate a failure merely because those six graphs are compiled.  Decode,
text, and packed-text caches must not grow.

The 910B2 control for this exact target and ladder completed all 1,651 pages on
commit `c06a9cf` under:

```text
tmp/09_persistent_page_engine/
  910b_full_b64_pse_layout8_target768_dense_c06a9cf/
```

| Metric | old target-1024 | dense target-768 | signed delta |
|---|---:|---:|---:|
| pipeline E2E | 717.705 s | 727.988 s | +10.283 s |
| pages/s | 2.30039 | 2.26790 | -0.03249 |
| vision prefill | 178.088 s | 203.540 s | +25.452 s |
| physical vision tokens | 9,512,832 | 9,274,368 | -238,464 |
| vision fill | 90.416% | 92.352% | +1.936 pp |
| vision groups | 8,781 | 11,230 | +2,449 |
| text prefill | 74.936 s | 76.407 s | +1.471 s |
| physical text tokens | 3,997,056 | 3,214,080 | -782,976 |
| text fill | 64.049% | 79.652% | +15.603 pp |
| decode | 124.087 s | 126.486 s | +2.399 s |

Thus dense-768 is a measured regression on 910B2: lower padding did not repay
the 27.9% increase in graph calls.  This does **not** settle the 310P question,
because the measured optimized 310P physical-tok/s curve peaks at different
shapes and degrades much more strongly above 768.  Phase 55 is the direct 310P
test of that hypothesis, not an assumption that the 910B result transfers.
Read the exact values from the committed compact evidence; do not substitute
the older target-1024 reference or claim that padding reduction alone implies a
speedup.

### 55.1 Preflight and cache resolution

Use one persistent shell.  The work-server checkout is pull-only.

```sh
set -uo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"
source npu-setup
set -e

PYTHON_BIN=/usr/local/python3.12.13/bin/python
EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
E2E=09_persistent_page_engine/scripts/run_omnidocbench.py
EVAL_WRAPPER=09_persistent_page_engine/scripts/run_omnidocbench_eval.py
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
LAYOUT_MODEL=/home/lukaiv/models/PP-DocLayoutV3_safetensors
DATASET_JSON=/home/lukaiv/datasets/OmniDocBench/OmniDocBench.json
IMAGES_DIR=/home/lukaiv/datasets/OmniDocBench/images
EVALUATOR_ROOT=/workspace/repos/OmniDocBench_eval

COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
ROOT="tmp/09_persistent_page_engine/310p_phase55_dense768_${SHORT}"
MAT="$ROOT/materialize_gate"
WARM="$ROOT/warm_gate"
FULL="$ROOT/full"
test ! -e "$ROOT"
mkdir -p "$MAT" "$WARM" "$FULL/evaluation/work"

PHASE52_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase52_opt_kv2048_*' | sort | tail -n 1)"
PHASE53_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase53_decode_length_modes_*' | sort | tail -n 1)"
PHASE54_ROOT="$(find tmp/09_persistent_page_engine -maxdepth 1 -type d \
  -name '310p_phase54_b64_pse_layout8_*' | sort | tail -n 1)"
test -n "$PHASE52_ROOT" && test -n "$PHASE53_ROOT"

DECODE_CACHE="$(awk -F= '$1 == "cache" {print substr($0,index($0,"=")+1)}' \
  "$PHASE53_ROOT/contract.txt")"
VISION_CACHE=.runtime_cache/310p_phase52_v16_vision_4352_nz
BATCHED_CACHE=.runtime_cache/310p_phase52_v16_vision_batched_unused
TEXT_CACHE=.runtime_cache/310p_phase52_v16_text
PACKED_CACHE=.runtime_cache/310p_phase52_v16_text_packed

for cache in "$DECODE_CACHE" "$VISION_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"
do
  test -d "$cache"
  test -n "$(find "$cache" -type f -print -quit)"
done
test "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" = \
  85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db
df -h . "$VISION_CACHE" | tee "$ROOT/df_preflight.txt"
npu-smi info | tee "$ROOT/npu_preflight.txt"
```

Define the production command once:

```sh
PRODUCTION_ARGS=(
  "$E2E"
  --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL"
  --batch-size 64 --cache-length 2048 --max-new-tokens 2048
  --preprocessor-min-pixels 28224 --preprocessor-max-pixels 401408
  --text-crop-scale 0.5
  --decode-backend torchair
  --decode-optimization combined_apply_pse_sentinel
  --torchair-cache-dir "$DECODE_CACHE"
  --vision-backend torchair --vision-attention prompt_flash_attention
  --vision-buckets 128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048
  --vision-torchair-cache-dir "$VISION_CACHE"
  --vision-batched-cache-dir "$BATCHED_CACHE"
  --vision-promptfa-align-128
  --vision-mlp-intermediate-size 4352
  --vision-linear-weight-format fractal_nz
  --vision-padding bucket --vision-packing greedy
  --vision-pack-target 768 --vision-router-lookahead 32
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312
  --text-packing production_group
  --text-pack-buckets 128,256,512,1024 --text-pack-max-members 32
  --text-torchair-cache-dir "$TEXT_CACHE"
  --text-packed-cache-dir "$PACKED_CACHE"
  --layout-device npu --no-layout-graph-capture
  --preprocess-all-pages-first --layout-workers 8 --no-timeline
)

record_caches() {
  out="$1"; : >"$out"
  for cache in "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" \
               "$TEXT_CACHE" "$PACKED_CACHE"; do
    if test -d "$cache"; then
      printf '%s\tfiles=%s\tbytes=%s\n' "$cache" \
        "$(find "$cache" -type f | wc -l)" "$(du -sb "$cache" | cut -f1)"
    else
      printf '%s\tfiles=0\tbytes=0\n' "$cache"
    fi
  done >"$out"
}
```

### 55.2 Materialize once, then prove warm replay

Run both gates on the first eight pages.  Capture the exact command, log, exit
code, wall, and cache inventory before and after each gate.

```sh
run_gate() {
  lane="$1"
  record_caches "$lane/cache_before.txt"
  printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
    --offset 0 --limit 8 --output-dir "$lane/output" >"$lane/command.sh"
  printf '\n' >>"$lane/command.sh"
  SECONDS=0; set +e; set -o pipefail
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 3600 \
    "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
    --offset 0 --limit 8 --output-dir "$lane/output" \
    2>&1 | tee "$lane/run.log"
  ec="${PIPESTATUS[0]}"; set -e
  printf '%s\n' "$ec" >"$lane/exit_code.txt"
  printf '%s\n' "$SECONDS" >"$lane/wall_s.txt"
  record_caches "$lane/cache_after.txt"
  test "$ec" -eq 0
}

run_gate "$MAT"
run_gate "$WARM"
```

Validate mechanically:

```sh
"$PYTHON_BIN" - "$MAT" "$WARM" "$VISION_CACHE" <<'PY' \
  | tee "$ROOT/gate_contract.json"
import json, pathlib, sys

def inventory(path):
    out = {}
    for line in open(path):
        cache, files, size = line.rstrip().split("\t")
        out[cache] = (int(files.split("=",1)[1]), int(size.split("=",1)[1]))
    return out

lanes = [pathlib.Path(x) for x in sys.argv[1:3]]
vision_cache = sys.argv[3]
summaries = [json.load(open(x / "output/run_summary.json")) for x in lanes]
for d in summaries:
    c, r = d["configuration"], d["recognition"]
    assert d["count"] == d["result_count"] == d["prediction_count"] == 8
    assert c["batch_size"] == 64 and c["cache_length"] == 2048
    assert c["decode_optimization"] == "combined_apply_pse_sentinel"
    assert c["vision_pack_target"] == 768
    assert c["vision_buckets"] == [128,256,384,512,640,768,896,1024,
                                    1152,1280,1408,1536,1664,1792,1920,2048]
    assert c["vision_linear_weight_format"]["after_format_histogram"] == {"29":162}
    assert set(r["stop_reason_counts"]) <= {"eos","kv_cache_full","repetition"}

before0, after0 = inventory(lanes[0]/"cache_before.txt"), inventory(lanes[0]/"cache_after.txt")
before1, after1 = inventory(lanes[1]/"cache_before.txt"), inventory(lanes[1]/"cache_after.txt")
growth0 = {k: after0[k][0]-before0[k][0] for k in before0}
growth1 = {k: after1[k][0]-before1[k][0] for k in before1}
assert all(v == 0 for k,v in growth0.items() if k != vision_cache), growth0
assert growth0[vision_cache] > 0, growth0
assert all(v == 0 for v in growth1.values()), growth1
print(json.dumps({"materialization_file_deltas":growth0,
                  "warm_file_deltas":growth1,
                  "materialization_setup_s":summaries[0]["setup_s"],
                  "warm_setup_s":summaries[1]["setup_s"],
                  "warm_e2e_s":summaries[1]["pipeline_e2e_s"]}, indent=2))
PY
```

If the materialization gate grows anything except the vision cache, or the
warm gate grows any cache, report `GATE_CACHE_MISS` and stop.  Otherwise report
`310P PHASE 55 WARM GATE: PASS` immediately and continue.

### 55.3 Full E2E and official evaluation

```sh
record_caches "$FULL/cache_before.txt"
printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$FULL/output" >"$FULL/command.sh"
printf '\n' >>"$FULL/command.sh"
SECONDS=0; set +e; set -o pipefail
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 7200 \
  "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
  --offset 0 --limit 1651 --output-dir "$FULL/output" \
  2>&1 | tee "$FULL/run.log"
ec="${PIPESTATUS[0]}"; set -e
printf '%s\n' "$ec" >"$FULL/exit_code.txt"
printf '%s\n' "$SECONDS" >"$FULL/wall_s.txt"
record_caches "$FULL/cache_after.txt"
test "$ec" -eq 0

"$EVAL_PYTHON" "$EVAL_WRAPPER" \
  --config "$FULL/evaluation/config.yaml" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --match-workers 24 --teds-workers 12 \
  --page-timeout-sec 120 --fallback-timeout-sec 180 \
  --fallback-latex-timeout-sec 30 --teds-timeout-sec 120 \
  2>&1 | tee "$FULL/evaluation/run.log"
```

Create the evaluator config exactly as in Phase 54, changing only its
prediction/result paths to `$FULL`.  Do not change matching semantics or omit
problem pages.  The full run must add zero cache files.

### 55.4 Required report

Write `$ROOT/agent_report.md` and report one of:

```text
310P PHASE 55 DENSE768 FULL: PASS | PREFLIGHT_FAILURE |
GATE_CACHE_MISS | GATE_RUNTIME_FAILURE | E2E_FAILURE |
COMPILE_CONTAMINATED | STRUCTURAL_MISMATCH | EVALUATOR_FAILURE
```

For a pass, include:

1. exact commit, host/NPU/software/model SHA, all commands, cache paths, and
   materialization/warm/full cache deltas;
2. setup, pipeline E2E, pages/s, layout wall and every layout stage total;
3. vision/text/decode wall, real/physical/effective tok/s, all token totals,
   decode useful fraction, calls, and stop reasons;
4. vision group count, crops/group, real/physical tokens, fill fraction, and
   complete dense-bucket histogram; text groups/tokens/fill/histogram;
5. signed deltas against Phase 52 and Phase 54 for layout, vision, text,
   decode, E2E, and pages/s;
6. all six official metrics and signed deltas against Phase 52/54 and the
   committed 910B2 dense-768 authority;
7. an explicit decomposition of savings from B64-PSE decode, staged-W8
   layout, dense vision buckets, and target 768.  Label measured deltas versus
   projections; do not add independently timed overlapping stages;
8. `What is proven`, `What differs`, `What remains unresolved`, and the first
   causal error if anything failed.

Keep large predictions, compiler caches, and raw logs on the work server.
Paste back `agent_report.md`, `gate_contract.json`, the full compact summary,
evaluation compact summary, and the head-to-head JSON.  Do not alter another
parameter or start a follow-up sweep.  Then **stop**.

## Phase 56A: CDM runtime preflight only

This is the first, deliberately short feedback cycle for installing the native
OmniDocBench Formula CDM runtime in the existing work-server container.  Do not
run OCR, do not run evaluation, do not install packages, and do not download
anything in this phase.  Its only purpose is to discover the exact environment
so the next pushed revision can choose the fastest valid installation path.

Pull `main`, require a clean checkout, and run the following from the repository
root.  It is CPU-only and must finish in well under one minute:

```sh
set -eu
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"

ROOT=tmp/09_persistent_page_engine/310p_phase56_cdm
mkdir -p "$ROOT"

arch="$(uname -m)"
os="$(. /etc/os-release 2>/dev/null && printf '%s-%s' "${ID:-unknown}" "${VERSION_ID:-unknown}" || printf unknown)"
uid="$(id -u)"
apt=no; command -v apt-get >/dev/null 2>&1 && apt=yes
py310="$(command -v python3.10 2>/dev/null || true)"
eval_py=NONE
for candidate in \
  /workspace/venvs/omnidocbench_py310/bin/python \
  "$WORK_SERVER_REPO"/../venvs/omnidocbench_py310/bin/python \
  "$WORK_SERVER_REPO"/.venv_eval/bin/python; do
  if test -x "$candidate"; then eval_py="$candidate"; break; fi
done
evaluator=NONE
for candidate in \
  /workspace/repos/OmniDocBench_eval \
  "$WORK_SERVER_REPO"/../OmniDocBench_eval \
  /home/lukaiv/OmniDocBench_eval; do
  if test -f "$candidate/pdf_validation.py"; then evaluator="$candidate/pdf_validation.py"; break; fi
done
matched="$(find "$WORK_SERVER_REPO/tmp/09_persistent_page_engine" \
  -type f -name 'predictions_quick_match_display_formula_result.json' \
  -print -quit 2>/dev/null || true)"
pdflatex_v="$(pdflatex --version 2>/dev/null | head -n 1 || true)"
kpsewhich_v="$(kpsewhich --version 2>/dev/null | head -n 1 || true)"
magick_v="$(magick --version 2>/dev/null | head -n 1 || true)"
convert_v="$(convert --version 2>/dev/null | head -n 1 || true)"
gs_v="$(gs --version 2>/dev/null | head -n 1 || true)"
free_gb="$(df -Pk "$WORK_SERVER_REPO" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')"

probe() {
  url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -LIsS --connect-timeout 5 --max-time 12 "$url" >/dev/null 2>&1
  elif command -v wget >/dev/null 2>&1; then
    wget --spider -T 12 "$url" >/dev/null 2>&1
  else
    return 1
  fi
}
github=no; probe https://github.com/ImageMagick/ImageMagick && github=yes
texmirror=no; probe https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2025/tlnet-final/install-tl-unx.tar.gz && texmirror=yes
apt_net=no; probe http://ports.ubuntu.com/ubuntu-ports/ && apt_net=yes

printf '%s\n' \
  "CDM_PREFLIGHT arch=$arch os=$os uid=$uid apt=$apt py310=${py310:-NONE} eval_py=$eval_py evaluator=$([ "$evaluator" != NONE ] && echo yes || echo no) matched=$([ -n "$matched" ] && echo yes || echo no) pdflatex=$([ -n "$pdflatex_v" ] && echo yes || echo no) kpsewhich=$([ -n "$kpsewhich_v" ] && echo yes || echo no) magick=$([ -n "$magick_v" ] && echo yes || echo no) convert=$([ -n "$convert_v" ] && echo yes || echo no) gs=$([ -n "$gs_v" ] && echo yes || echo no) github=$github texmirror=$texmirror apt_net=$apt_net free_gb=$free_gb evaluator_path=$evaluator matched_path=${matched:-NONE}" \
  | tee "$ROOT/preflight_sentence.txt"
```

Return **exactly the one sentence printed by the command**, with no report,
explanation, Markdown fence, or additional investigation.  Then stop.  In
particular, do not run `setup_omnidocbench_eval_runtime.sh` yet: its current
defaults target the 910B `/workspace` layout and the next revision will adapt
them from this preflight rather than guessing.
