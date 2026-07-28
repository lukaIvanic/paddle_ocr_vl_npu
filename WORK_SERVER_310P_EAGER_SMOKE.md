# Work-server 310P eager smoke handoff

> **Replacement-server note:** do not start from this historical brief. Use
> `WORK_SERVER_310P_EXP09_LADDER.md`, which contains the canonical
> bootstrap-to-production ladder and validates the real Experiment 09
> CPU-MRoPE serving path. This file is retained only for its focused diagnostic
> procedures, including Phase 4A IndexPut triage when explicitly referenced.

This historical file is an execution brief for focused diagnostics on Luka's
work server. Read `AGENTS.md` first and follow only a phase explicitly
referenced by the canonical validation ladder. The work-server checkout is a
pull-only NPU validation lane: do not edit tracked files, commit, push, or
create branches.

## Objective

Establish whether Experiment 09 can perform a small, real PaddleOCR-VL run on
the server's Atlas 310P environment. This is a compatibility smoke test, not a
performance benchmark.

Validate, in order:

1. The actual server exposes an Ascend NPU and a usable `torch_npu` Python
   environment.
2. The PaddleOCR-VL recognizer weights can be found and loaded.
3. The PP-DocLayoutV3 weights and at least one full OmniDocBench page can be
   found.
4. One repository crop can run through eager recognizer inference.
5. One full page can run through layout, crop preparation, vision prefill,
   text prefill, and eager static-KV decoding.
6. The native NPU operations used by that path are present and actually
   callable in the installed environment.

Do **not** test TorchAir, `torch.compile`, cached graphs, performance settings,
PromptFlashAttention, multiple workers, or a full benchmark in this pass.

## Rules

- Discover the actual machine state. Do not assume the checkout is under
  `/workspace`, that `npu-setup` exists, or that a particular Python path is
  installed.
- Work from the current clone. Resolve it with
  `git rev-parse --show-toplevel`.
- Run `git pull --ff-only origin main` before testing. If tracked local changes
  prevent the pull, stop and report them; do not discard them.
- Do not install or replace PyTorch, torch-npu, CANN, or system packages.
- Do not silently fall back to CPU or CUDA.
- Do not download large model or dataset artifacts during this first pass.
  Search sensible mounted storage roots and existing Hugging Face caches. If
  something is absent, report precisely what is missing and where you looked.
- Put all logs and small outputs below the clone's
  `tmp/09_persistent_page_engine/310p_eager_smoke/`.
- Preserve the exact commands, absolute paths, Python executable, package
  versions, Git commit, hardware identity, stdout, stderr, and exit codes.

## Phase 1: discover the environment

Start inside the cloned repository. Record:

```sh
git status --short --branch
git rev-parse HEAD
hostname
uname -a
command -v npu-smi
command -v python
command -v python3
command -v npu-setup
```

If `npu-setup` exists, inspect what it is before sourcing it. Source it only if
it is clearly the work server's intended NPU environment setup. Otherwise use
the server's existing documented activation mechanism or current environment.

Use `npu-smi info` to confirm the physical product is an Atlas 310P-class
device and to select a free device. Do not terminate other users' processes.
If no device is free, stop and report that as the blocker.

Select a Python executable by testing the current environment and plausible
existing virtual environments. The chosen interpreter must successfully run:

```python
import torch
import torch_npu

assert torch.npu.is_available()
assert torch.npu.device_count() >= 1
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(8, dtype=torch.float16, device="npu:0")
y = (x + 1).cpu()
print(y)
```

Record:

- the absolute Python executable;
- `torch.__version__`;
- `torch_npu.__version__`;
- device name and device count;
- relevant CANN/runtime version information exposed by the environment.

Also import the non-system dependencies needed by Experiment 09. At minimum
check Transformers, tokenizers, safetensors, torchvision, Pillow, NumPy,
OpenCV, Shapely, YAML, and `kornia_rs`. If an import is missing, report the
exact module and stop rather than changing the base NPU stack.

## Phase 2: locate artifacts

Resolve and record these three inputs without assuming their locations:

- `RECOGNIZER_MODEL`: local PaddleOCR-VL 1.6 model directory;
- `LAYOUT_MODEL`: local `PP-DocLayoutV3_safetensors` directory;
- `FULL_PAGE_IMAGE`: one real full-page image from OmniDocBench.

Prefer explicit environment variables or paths already documented on the
machine. Otherwise search only sensible mounted storage roots, home
directories, and existing Hugging Face cache roots. Do not scan virtual
filesystems or the entire `/` tree.

The recognizer may exist either as a plainly named directory or as a Hugging
Face snapshot beneath a path resembling:

```text
models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/<revision>
```

For OmniDocBench, locate both `OmniDocBench.json` and its corresponding images
directory, then select a referenced full-page image that exists. Do not use one
of the repository's region crops as the full-page test.

Check that model configuration, tokenizer/processor assets, and weight shards
are present before attempting inference. Record directory sizes and the
resolved absolute paths.

If the recognizer model is absent, the recognizer smoke cannot proceed. If only
layout or OmniDocBench is absent, still run the repository-crop recognizer
smoke and report the overall result as partial.

## Phase 3: inspect native-operation registration

With the chosen Python, import `torch_npu` and report whether these Experiment
09 operation entrypoints exist and are callable:

```text
torch_npu.npu_add_rms_norm
torch_npu.npu_apply_rotary_pos_emb
torch_npu.npu_incre_flash_attention
torch_npu.npu_rms_norm
torch_npu.npu_rotary_mul
torch_npu.npu_scatter_nd_update_
torch_npu.npu_swiglu
torch_npu.scatter_update_
```

This is only a registration preflight. Do not claim an operation works merely
because `hasattr` succeeds. The inference runs below are the callable proof.
Do not probe PromptFlashAttention in this pass.

## Phase 4: eager recognizer smoke on one repository crop

From the repository root, run the equivalent of the following with the
discovered Python and model path:

```sh
PYTHONPATH="$REPO/09_persistent_page_engine" \
"$PYTHON_BIN" -m paddleocr_vl.model.example \
  --model "$RECOGNIZER_MODEL" \
  --crop "$REPO/crops/crop_01_text_block_en.png" \
  --prompt "OCR:" \
  --max-new-tokens 8
```

Do not pass `--static` for this first recognizer check. Capture the complete
log and exit code. Success means the model loads on NPU, preprocesses the real
crop, produces token IDs, and decodes text without CPU/CUDA fallback.

If this fails, stop before full-page inference. Report the first causal
traceback, not only the last wrapper exception.

### Phase 4A: IndexPut regression triage

Run this subsection only if Phase 4 fails at the MRoPE assignment resembling:

```python
position_ids[:, batch_idx, attention_mask[batch_idx] == 1] = (
    llm_positions.to(position_ids.device)
)
```

That expression is boolean advanced-index assignment. On NPU it dispatches
through PyTorch's `aten::index_put_` and the torch-npu/CANN IndexPut
implementation. The same Experiment 09 source previously completed on another
310P server, and the relevant source lines have not recently changed, so treat
this first as an environment/operator regression rather than changing model
code.

Do not install packages, switch CANN installations, or edit tracked files.
Collect the evidence below, write it under:

```text
$OUTPUT_ROOT/indexput_triage/
```

Then stop. Do not continue to Phase 5.

First preserve the exact source and environment identity:

```sh
mkdir -p "$OUTPUT_ROOT/indexput_triage"

{
  git status --short --branch
  git rev-parse HEAD
  git blame -L 60,61 \
    09_persistent_page_engine/paddleocr_vl/model/example.py
  git blame -L 194,194 \
    09_persistent_page_engine/paddleocr_vl/model/modeling.py
  hostname
  uname -a
  npu-smi info
  "$PYTHON_BIN" -m pip show torch torch-npu
  printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-}"
  printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-}"
  printf 'ASCEND_AICPU_PATH=%s\n' "${ASCEND_AICPU_PATH:-}"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-}"
  readlink -f "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
  if [ -n "${ASCEND_OPP_PATH:-}" ]; then
    readlink -f "$ASCEND_OPP_PATH"
  fi
} >"$OUTPUT_ROOT/indexput_triage/environment.txt" 2>&1
```

Also run this exact Python fingerprint with the chosen interpreter:

```sh
"$PYTHON_BIN" - \
  >"$OUTPUT_ROOT/indexput_triage/python_environment.txt" 2>&1 <<'PY'
import importlib.metadata
import os
import platform
import sys

import torch
import torch_npu

print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch_npu:", torch_npu.__version__)
try:
    distribution_version = importlib.metadata.version("torch-npu")
except importlib.metadata.PackageNotFoundError:
    distribution_version = "<distribution metadata not found>"
print("torch_npu distribution:", distribution_version)
print("torch_npu file:", torch_npu.__file__)
print("device count:", torch.npu.device_count())
print("device 0:", torch.npu.get_device_name(0))
for name in (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
):
    print(f"{name}={os.environ.get(name, '')}")
PY
```

Save the complete original Phase 4 log. In the report, quote the first causal
IndexPut error, including any ACLNN operator name, missing `IndexPutV2` binary,
tensor dtype/shape, or dispatch message. Do not report only the final wrapper
exception.

Finally, run this isolated CPU-control/NPU reproducer. It deliberately removes
the model, processor, and image from the experiment while retaining the
advanced-index assignment pattern:

```sh
if "$PYTHON_BIN" - \
  >"$OUTPUT_ROOT/indexput_triage/minimal_reproducer.txt" 2>&1 <<'PY'
import traceback

import torch
import torch_npu


def assignment(device: str):
    seq_len = 32
    position_ids = torch.ones(
        (3, 1, seq_len), dtype=torch.int64, device=device
    )
    attention_mask = torch.ones(
        (1, seq_len), dtype=torch.int64, device=device
    )
    llm_positions = (
        torch.arange(seq_len, dtype=torch.int64, device=device)
        .view(1, -1)
        .expand(3, -1)
        .contiguous()
    )
    position_ids[:, 0, attention_mask[0] == 1] = llm_positions
    if device.startswith("npu"):
        torch.npu.synchronize()
    return position_ids.cpu()


cpu_result = assignment("cpu")
print("CPU_CONTROL: PASS", tuple(cpu_result.shape), cpu_result.dtype)

torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
try:
    npu_result = assignment("npu:0")
    torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    print("NPU_INDEXPUT: PASS")
except Exception:
    print("NPU_INDEXPUT: FAIL")
    traceback.print_exc()
    raise
PY
then
  INDEXPUT_REPRO_EXIT_CODE=0
else
  INDEXPUT_REPRO_EXIT_CODE=$?
fi
printf '%s\n' "$INDEXPUT_REPRO_EXIT_CODE" \
  >"$OUTPUT_ROOT/indexput_triage/minimal_reproducer_exit_code.txt"
```

Interpret the result narrowly:

- CPU passes and NPU fails: the boundary is NPU IndexPut dispatch/runtime, not
  MRoPE mathematics.
- An `aclnnIndexPutImpl` or `IndexPutV2` message points at the torch-npu,
  installed OPP package, and exact 310P SoC support boundary.
- A PyTorch dispatcher/schema error points first at the exact
  PyTorch/torch-npu build pairing.
- If the reproducer passes but Phase 4 fails, retain the complete Phase 4
  traceback and report the real tensor shapes/dtypes; the failure is more
  specific than basic IndexPut availability.

Append this block to `agent_report.md`:

```text
310P INDEXPUT TRIAGE: REPRODUCED | NOT REPRODUCED | NOT RUN

Git commit:
Host / exact NPU product:
Python executable:
torch version / path:
torch_npu version / path:
CANN home / resolved target:
OPP path / resolved target:
Driver / firmware:

Phase 4 first causal error:
CPU control: PASS | FAIL
Standalone NPU IndexPut: PASS | FAIL
Standalone reproducer exit code:
Suspected compatibility boundary:
Artifact paths:
```

This phase is diagnostic only. A likely permanent code-side workaround is to
compute MRoPE position IDs on CPU and transfer the completed tensor once, which
is already how the normal serving preparation path is structured. Do not make
that change on the work server; return the evidence so it can be authored and
reviewed in the local lane if needed.

## Phase 5: tiny full-page Experiment 09 eager smoke

Only after Phase 4 succeeds and the layout model plus full-page image exist,
run `09_persistent_page_engine/scripts/run_offline_e2e.py` with all three
recognizer execution stages explicitly eager:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_offline_e2e.py" \
  --image "$FULL_PAGE_IMAGE" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --dtype fp16 \
  --decode-backend raw_eager \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --text-backend raw_eager \
  --text-padding none \
  --batch-size 1 \
  --cache-length 2048 \
  --max-new-tokens 8 \
  --max-regions 2 \
  --no-save-annotated \
  --output-dir "$OUTPUT_ROOT/full_page"
```

This intentionally keeps the smoke small while exercising the real
full-page-to-recognition flow. The recognizer must remain on NPU. The layout
frontend should use its committed Experiment 09 implementation; do not replace
it with PaddleX or another pipeline.

If and only if the run explicitly reports that the 2048 cache capacity is too
small for the selected regions, rerun once with `--cache-length 4096`. Do not
make any other performance-oriented changes.

Success requires:

- exit code zero;
- one page processed;
- at least one layout region and one recognized region;
- an output `run.json`;
- generated token IDs/text for the recognized regions;
- configuration in `run.json` showing `raw_eager` for decode, vision, and text
  execution;
- no CPU/CUDA fallback;
- no TorchAir compilation or cache creation.

The eight-token cap makes this a plumbing and operator test. Truncated OCR text
is expected and is not a quality failure.

## Failure localization

Do not begin open-ended optimization or source modification. Classify a
failure into one of these buckets:

1. NPU/runtime unavailable.
2. No compatible Python environment.
3. Missing Python dependency.
4. Recognizer model absent or incomplete.
5. Layout model absent or incomplete.
6. OmniDocBench annotation/image absent.
7. Model/config incompatibility.
8. Native torch-npu operation missing or rejected at call time.
9. NPU out-of-memory.
10. Experiment 09 code failure after all prerequisites pass.

For a call-time native-op failure, include the operation name, installed
torch/torch-npu/CANN versions, tensor shapes/dtypes if shown, and the complete
first relevant traceback. Do not patch around it on the work server.

## Required report

Write `agent_report.md` under the output root and end your response to Luka
with the same concise block:

```text
310P EAGER SMOKE: PASS | PARTIAL | FAIL

Git commit:
Host / NPU:
Python:
torch:
torch_npu:
CANN/runtime:

Recognizer model:
Layout model:
OmniDocBench JSON:
Full-page image:

Generic NPU tensor test: PASS | FAIL
Dependency import check: PASS | FAIL
Native-op registration preflight: PASS | PARTIAL | FAIL
One-crop eager recognizer: PASS | FAIL
Full-page eager Experiment 09: PASS | SKIPPED | FAIL

Recognized regions:
Output run.json:
First generated token IDs/text:

First blocker or important warning:
Exact failing command:
Log paths:
```

The report must distinguish:

- discovered facts from assumptions;
- operation registration from actual execution;
- a missing artifact from a code/runtime failure;
- a smoke-test pass from any claim about performance or OCR accuracy.
