# MinerU FP16 decode-attention KV-cache probe: 310P handoff

**Historical compatibility probe only.** For performance work use
[`../FULL_MODEL_KV_310P_HANDOFF.md`](../FULL_MODEL_KV_310P_HANDOFF.md),
which measures real crop generation inside the complete compiled decoder.
Do not interpret this isolated eager probe as full-model speed.

This is the complete task brief for an agent with no conversation history.
Use this file and the adjacent code. You do not need access to the authoring
machine, Mac sessions, the 910B host, model weights, crops, or datasets.

## Objective and established context

On an actual Ascend 310P, establish correctness first, then measure attention
latency for MinerU-shaped one-token decode: FP16, 14 query heads, 2 KV heads,
head dimension 64, cache capacity 4096, block size 128. This is a synthetic
single-attention-operation probe, not a full MinerU or page-throughput run.

The audited vLLM-Ascend revision is
`80610e4438dba05011b05f89fc45d91e96992671`. Its 310P DecodeOnly path allocates
each K/V cache as `[num_blocks, 8, 128, 16]` with actual FRACTAL_NZ storage
descriptor 29, writes with `torch_npu._npu_reshape_and_cache`, and reads with
`torch_npu._npu_paged_attention`. Our custom MinerU currently uses native dense
BNSD K/V with IncreFA. NZ decoder weights are unrelated to NZ cache storage.

Prior 910B2 tests on CANN/ATB 9.0 and torch-npu 2.10 passed native controls.
The genuine blocked-NZ writer preserved KV values, but the 910B paged reader
rejected that shape. Ordinary-shaped NZ returned wrong attention values.
Neither failure establishes 310P support or speed. Huawei documents ND
`[blocks, block, KV_heads, D]` on A2/910B and blocked NZ on Atlas inference
products/310P. Published ATB tables require CPU int32 context lengths; the
audited vLLM revision passes NPU lengths. Test both explicitly and separately.

## Constraints

- This server may have different CANN, ATB, Python, torch-npu and vLLM versions.
  Discover and record them; do not reproduce the 910B environment by upgrading.
- Source is pull-only. Do not edit tracked files, commit, push, create branches,
  install packages, change shared environment configuration, or modify vLLM.
  Runtime evidence may be written under the repo's ignored `tmp/` directory.
- Use the server's existing working environment and device-selection procedure.
  Select one healthy, unoccupied 310P. Sharing NPU 7 was authorized only on the
  separate 910B server; that authorization does not select or reserve a card here.
- Do not kill services or reset devices. Stop after a timeout or device error;
  a terminated worker can leave device-side work running.
- No BF16 and no CPU attention fallback. CPU computes the independent FP32
  reference and may hold small length metadata; attention still runs on NPU.
- Workers import only torch and torch-npu; the controller uses stdlib. vLLM and
  TorchAir are not required. A missing operator is a recorded compatibility
  result, not permission to install a replacement or alter the call contract.

## 1. Obtain source without disturbing existing work

Start inside this repository's existing checkout. Resolve its path locally:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
```

If tracked files are changed, stop and report them. Do not stash, overwrite or
discard them. Otherwise fetch the published branch and use a detached checkout:

```bash
git fetch origin codex/mineru-kv-cache-layout-probe
git checkout --detach FETCH_HEAD
git rev-parse HEAD
git merge-base --is-ancestor 353bb3583faf99ed9de7edea13c25e060441f329 HEAD
```

Require the ancestor check to exit 0 and this handoff file to exist. Record the
full fetched commit. If GitHub access fails, report it without changing remotes.

## 2. Discover the environment before inference

Use existing local runbooks or inspect installed setup scripts to activate the
server's working CANN/ATB environment. Do not copy `/usr/local/...` paths,
`npu-setup`, interpreter names, SSH routes, or device IDs from the 910B notes.
Set and export `PYTHON` to the absolute path of the working torch-npu interpreter.
Set `ASCEND_RT_VISIBLE_DEVICES` to exactly one available physical 310P using
this server's established procedure: workers address its logical `npu:0`.
Do not continue if you cannot identify a healthy, unoccupied device.

Create a fresh directory for preflight evidence:

```bash
PREFLIGHT_DIR="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/preflight_310P_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PREFLIGHT_DIR"
git rev-parse HEAD > "$PREFLIGHT_DIR/commit.txt"
command -v "$PYTHON" > "$PREFLIGHT_DIR/interpreter.txt"
npu-smi info > "$PREFLIGHT_DIR/occupancy_before.txt" 2>&1
```

If `npu-smi` is unavailable or fails, preserve its output and use an existing
server inventory tool. Report how health and occupancy were checked.
Run this preflight with the selected interpreter. It imports torch-npu but
does not allocate attention fixtures or run kernels:

```bash
"$PYTHON" - <<'PY' > "$PREFLIGHT_DIR/environment.json" 2> "$PREFLIGHT_DIR/environment.log"
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys

report = {
    "python": sys.version, "executable": sys.executable,
    "hostname": platform.node(), "platform": platform.platform(),
    "environment": {k: os.environ.get(k) for k in (
        "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
        "LD_LIBRARY_PATH", "LD_PRELOAD", "TORCH_DEVICE_BACKEND_AUTOLOAD")},
    "distributions": {},
}
for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
    try:
        dist = metadata.distribution(name)
        report["distributions"][name] = {
            "version": dist.version, "root": str(dist.locate_file("")),
            "direct_url": dist.read_text("direct_url.json"),
        }
    except metadata.PackageNotFoundError:
        report["distributions"][name] = {"installed": False}
try:
    import torch
    import torch_npu
    report["torch_npu_path"] = str(Path(torch_npu.__file__).resolve())
    report["torch_git_version"] = getattr(torch.version, "git_version", None)
    version_file = Path(torch_npu.__file__).parent / "version.py"
    report["torch_npu_version_source"] = version_file.read_text() if version_file.exists() else None
    report["npu_available"] = bool(torch.npu.is_available())
    report["device_name"] = torch.npu.get_device_name(0) if report["npu_available"] else None
    report["allow_internal_format_api"] = hasattr(torch.npu.config, "allow_internal_format")
    report["operators"] = {}
    for name in ("empty_with_format", "get_npu_format", "npu_format_cast",
                 "npu_incre_flash_attention", "npu_fused_infer_attention_score",
                 "npu_fused_infer_attention_score_v2", "_npu_paged_attention",
                 "_npu_reshape_and_cache"):
        fn = getattr(torch_npu, name, None)
        try:
            signature = str(inspect.signature(fn)) if fn is not None else None
        except (TypeError, ValueError):
            signature = "native signature unavailable"
        report["operators"][name] = {"available": fn is not None, "signature": signature}
except Exception as exc:
    report["import_or_device_error"] = repr(exc)
print(json.dumps(report, indent=2))
PY
```

Read the report before continuing. Require an available observed **310P**, one
selected visible physical device, and `allow_internal_format`,
`empty_with_format`, `get_npu_format`, `npu_format_cast`. If an import, library
load, or prerequisite API fails, report that blocker. Do not repeatedly submit
identical failing NPU cases or replace the interpreter with a CPU-only one.
Record actual CANN and ATB versions from installed version files/package
metadata located using this server's setup. Report unavailable metadata honestly.

If vLLM-Ascend is installed, inspect its 310P cache allocator, writer dispatch,
and paged-attention call site read-only; save relevant excerpts with file paths,
hashes and revision metadata. Check cache shape, storage descriptor and length
device against the audited revision above. Do not import/start the vLLM engine.
If absent, continue operator probes but label the tested contract as the pinned
reference, not a verified installed vLLM production path.

## 3. Run the small matrices and inspect every result

From `WORK_SERVER_REPO`, first run portable IncreFA/FIA with B1/B16 and S768:

```bash
CHIP=310P OPERATORS=increfa,fia RUN_NAME=portable BATCHES=1,16 CONTEXTS=768 \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

If one API is absent in preflight, remove only that operator from `OPERATORS`,
record why, and run the available one. If both are absent, record this and
proceed to the private matrix only if its APIs exist and the device is healthy.
FIA v2 is optional; do not require it on an older runtime.

If both private APIs exist, run both metadata variants separately:

```bash
CHIP=310P OPERATORS=paged PAGED_LENGTH_DEVICE=npu RUN_NAME=paged_npu_lengths \
  BATCHES=1,16 CONTEXTS=768 \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

Inspect the completed run and current device health before the next command:

```bash
CHIP=310P OPERATORS=paged PAGED_LENGTH_DEVICE=cpu RUN_NAME=paged_cpu_lengths \
  BATCHES=1,16 CONTEXTS=768 \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

`run_probe.sh` prints its unique `RUN_DIR`, saves the exact command, exit code,
logs, summary and per-case JSON/logs. Exit 2 means at least one wrong output;
exit 1 means timeout/incomplete matrix or no passing case. Exit 0 can still
contain rejected cases. Read statuses, not just the exit code. A completed
contract rejection does not itself prevent testing another supported operation;
a timeout, hardware/driver error or unhealthy device does. Do not run these
commands as an unconditional shell chain or under an outer `set -e` that hides
the report after a nonzero exit.

The essential 310P case is `operator=paged`, `layout=blocked4`, `format=29`,
`fill=writer`. Require actual format 29 at attention and correct finite output
against the independent CPU reference. Record the writer logical roundtrip
separately. Synthetic packing and descriptor-2 variants are controls, and can
legitimately reject on this chip. Failed or incorrect cases have no valid timing.

## 4. Expand only a validated, supported path

For operators with correct S768 controls, expand the relevant operator separately
using the same wrapper, `CONTEXTS=768,1408,2816,4096`, `PATTERNS=uniform,ragged`,
and `BATCHES=1,16`. Keep the validated length-device variant explicit for paged
attention. Record the exact command. S1408/2816 exercise the existing IncreFA
PSE-sentinel workaround. If earlier errors identify unsupported contracts,
avoid repeating them purely to obtain timings; preserve their initial evidence.

Compare ND/NZ only when both cases pass with identical operator, layout, fill
method, length-device variant and fixture hashes. The runner prints such pairs
as `FORMAT_COMPARISONS`. If ND is unsupported by the 310P paged operation, a
format-only speed ratio is unavailable. A correct NZ paged result versus correct
native IncreFA is an **operator-and-cache-contract comparison**. Do not attribute
that difference solely to NZ. Timings include eager dispatch, exclude cache
writing/packing, and establish neither compiled-decoder speed nor pages/s.

## 5. Return a complete report to Luka

Return the preflight directory and every run directory intact, including failed
cases and logs, plus before/after occupancy snapshots. State:

- Full source commit, exact commands and exit codes; observed chip and physical
  device; interpreter, torch/torch-npu, CANN/ATB and optional vLLM revisions.
- APIs present/missing and differences from the pinned vLLM contract; which
  length-device variant matches the installation, or that this is unverified.
- Every case status; actual K/V formats at allocation, after writing and at
  attention; logical roundtrip results and correctness errors/tolerances.
- Passing-case wall/device medians, fixture hashes, `FORMAT_COMPARISONS`, and
  whether timings were isolated from other workloads.
- Which contracts are validated, rejected, wrong, timed out or unavailable.
  Do not turn an import/setup failure into a hardware-support conclusion.

If blocked, return the failing command, relevant JSON/logs and the minimal
proposed source change; do not apply it. Luka will relay the report to the
authoring agent. You do not need to send messages or push artifacts yourself.
