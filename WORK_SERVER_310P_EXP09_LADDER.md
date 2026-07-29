# Work-server 310P Experiment 09 validation ladder

This is the execution brief for the AI agent on Luka's replacement Atlas 310P
server. Read `CLAUDE.md` and `AGENTS.md` first.

## Goal

Prove that the real Experiment 09 pipeline runs on this 310P software stack,
identify which production optimizations work, and measure their approximate
effect on a small representative workload.

## Current requested task: run Phase 9 only

Phases 0-8 have already passed on this server. For the current task, do not
rerun the production pipeline, layout lab, dataset validation, attention
correctness check, decode ladder, or packing ladder. Read their retained
instructions and artifacts only as context.

Pull current `main`, reuse the exact Python/model/NPU environment that passed
the earlier ladder, and execute only **Phase 9: isolated 310P vision
saturation matrix** below. Phase 9 loads only the PaddleOCR-VL recognizer and
does not need OmniDocBench images, the layout model, text prefill, or decode.

The current question is deliberately narrow:

> How does raw physical vision-transformer throughput on 310P change with
> packed row length and true graph batch size, and how does that scaling
> compare shape-for-shape with the measured 910B2 matrix?

Do not optimize source code, change production routing, or update the pinned
910B2 routing table during this task.

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
the layout check and report. For the current task, skip Phases 0-8 and stop
after Phase 9 and its dedicated report.

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

This is the only phase to run for the current request. It answers whether the
310P vision transformer is underfilled at small sequence lengths and whether
true graph batching closes part of the measured 910B2/310P gap.

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
current isolated-vision task, skip Phases 0-8 and stop after Phase 9. Do not
start any OCR page workload.

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
