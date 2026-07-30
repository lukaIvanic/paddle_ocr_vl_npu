# Work-server 310P Experiment 09 validation ladder

This is the execution brief for the AI agent on Luka's replacement Atlas 310P
server. Read `CLAUDE.md` and `AGENTS.md` first.

## Goal

Prove that the real Experiment 09 pipeline runs on this 310P software stack,
identify which production optimizations work, and measure their approximate
effect on a small representative workload.

## Current requested task: run Phase 13 only

Phases 0-12 have already run or have retained instructions and evidence. For
the current task, do not rerun the production pipeline, layout lab, dataset
validation, attention correctness check, decode ladder, packing ladder,
saturation matrices, native B1 profiler, Phase 11 format/alignment matrix, or
Phase 12 RoPE comparison. Read their retained instructions and artifacts only
as context.

Pull current `main`, reuse the exact Python/model/NPU environment that passed
the earlier ladder, and execute only **Phase 13: B1xS2048 MatMul-only
throughput** below. Phase 13 loads only the PaddleOCR-VL recognizer and does
not need OmniDocBench images, the layout model, text prefill, or decode.

The current question is deliberately narrow:

> For the exact original B1xS2048 27-layer compiled-PromptFA vision graph, how
> many FP16-equivalent TFLOP/s do the MatMul kernels themselves achieve on
> 310P, and how does that compare with the matched 910B2 result?

Hold everything else fixed: B1xS2048, the native 4304-wide MLP, production
runtime D72-to-D80 PromptFA padding, native ND Linear weights, separate manual
RoPE, fp16, real PromptFA, and TorchAir `cache_compile`. Time the complete
27-layer stage with NPU events, but calculate MatMul-only throughput from the
sum of all profiled MatMul kernel durations.

Do not run additional shapes, weight formats, MLP widths, RoPE variants, eager
lanes, page workloads, or routing experiments. Do not optimize source code,
change production routing, or update pinned routing tables during this task.

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
the layout check and report. Phases 9-12 are also retained historical tasks.
For the current task, skip Phases 0-12 and stop after Phase 13.

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

This is the only phase to execute for the current request. Run exactly these
three production `VisionPrefillStage` shapes:

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

This is the only phase to execute for the current request. Run exactly two
compiled variants:

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

This is the only phase to execute for the current task. It reproduces the
historical 910B2 MatMul-only calculation on 310P without changing the graph:

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

## Artifact interpretation

For the current Phase 13-only task, stop after Phase 13.4 and report. The
remaining sections are retained for earlier workflows and are not additional
current work.

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
is governed by Phase 13.4 above. Do not start any OCR page workload.

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
