# BGE-M3 AddLayerNormQuantV2: self-contained 310P handoff

## Objective and scope

Rebuild the exact upstream operator used by the 910B experiment for **ascend310p**,
validate it directly and through TorchAir, then compare regular full-W8A8 BGE-M3
with the V2-integrated model. Prioritize device kernel counts, durations, shapes,
MTE2/MTE3, vector/scalar/MAC counters and layout transfers. End-to-end time is
secondary. Slower execution is still useful evidence; do not suppress it.

Everything authored for this experiment is in this Git repository. No Blue Zone
SSH connection, private installed operator package, graph cache, local chat,
WhatsApp attachment, or model file from the authoring Mac is needed. The target
still needs its **own** CANN/compiler/torch-npu environment, the pinned upstream
source below, and the pinned public BGE-M3 checkpoint. Public dependency downloads
are explicit steps; Git alone does not contain those large third-party artifacts.

**Validated:** full model on Ascend910B2, CANN 9.0.1, torch 2.10.0+cpu,
torch-npu 2.10.0.post2, Transformers 5.5.4. Benchmark source `1031242c`;
committed evidence in `tmp/26_bge_m3_inference/v2_model_1031242c/`.

**Not validated here:** the 310P binary, its CANN-version compatibility, the 310P
model graph, and 310P performance. Portability entry points begin at `1835824d`.
Do not describe the 910B result or upstream support declaration as a 310P test.

The new composed-op probe was regression-tested on 910B at `1835824d`: all four
cases pass, with every returned tensor bit-for-bit equal between eager and
compiled execution. Evidence: `tmp/26_bge_m3_inference/v2_handoff_probe_1835824d/`.

The upstream v9.1.0 source explicitly registers `AddConfig("ascend310p", ...)` in
`norm/add_layer_norm_quant_v2/op_host/add_layer_norm_quant_v2_def.cpp`.
Its ACLNN documentation permits this test's FP16 static/per-tensor contract on
Atlas inference-series products: static mode, no additional sum output,
multiplication scale, no second quantizer, norm axis >=32 bytes, no BF16.
Thus try the existing implementation first; do not start by rewriting a kernel.

## What was actually implemented

1. Built **unchanged** CANN `ops-nn` v9.1.0 V2 kernel, tiler and ACLNN API. This is
   an upstream operator we integrated, not a newly optimized kernel we wrote.
2. `26_bge_m3_inference/test_add_layer_norm_quant_v2.py` uses ctypes to wrap
   PyTorch NPU buffers in ACL tensor descriptors, obtains workspace, and calls
   `aclnnAddLayerNormQuantV2`. It returns FP16 normalized output and INT8 output.
   It synchronizes to retain temporary buffers safely; its Python wall time is
   not a production-performance measurement.
3. `fused_norm_v2.py` registers `bge_m3::norm_quant_v2`, supplies fake outputs,
   and lowers it to a GE `AddLayerNormQuantV2` node. Compiled execution does not
   call the Python/ctypes bridge for each norm.
4. `prepare_v2_graph_package.py` copies the installed vendor package, retaining
   only V2's kernel dispatch in its operator-info JSON files. Otherwise upstream
   dependencies also override the original LayerNorm kernels and contaminate
   the baseline. Shared dependency libraries/headers remain present.
5. `v2_graph_infer.cpp` supplies the missing static, single-scale graph
   shape/datatype metadata. It changes no kernel/tiler/API. On CANN 9.0.1 we also
   preload stock and package ES libraries before importing torch_npu.
6. `FusedBGEM3` wires 48 boundaries: embedding norm +24 attention-output norms
   +23 FFN-output norms. FP16 output carries the residual; INT8 output goes
   directly to the next W8A8 matmul. The final FFN norm remains ordinary.
   Projection bias is passed as `[1,1024]`, matching gamma/beta. The existing
   calibration/dequantization scales remain fixed; V2 receives their FP16
   reciprocal. This introduces rounding differences, not recalibration.

For 310P the ordinary W8A8 weight convention is stored `[N,K]` in FRACTAL_NZ,
then transposed in the graph. Both ordinary matmul and the fused model's raw
projection now honor that convention. It still requires a real 310P test.

## Working rules

- Work in the existing research checkout, pull-only. Do not edit tracked source,
  create branches, commit, or push. A needed source change should be returned
  as a proposed patch with the failing command and relevant logs.
- Preserve other work and existing evidence. Use fresh build/install/run/cache
  directories. Do not copy 910B `.so`, kernel binaries, or GE caches to 310P.
- Use an idle 310P through the server's normal selection procedure; do not
  terminate another user's process. Keep the same device for both model lanes.
- Use the established NPU environment. Do not replace a global driver, CANN or
  Torch installation to make a test pass. Missing build dependencies may be
  supplied in a project environment if permitted by server policy; otherwise
  report the exact missing dependencies.
- Continue through successful gates automatically. Stop on a real failing gate,
  preserve evidence, and report it. No acknowledgment is needed between passes.
- CPU is allowed for mathematical references, not as an inference fallback.

## 1. Bootstrap and record the local environment

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git rev-parse HEAD
```

If local edits prevent the pull, report them without discarding them. Read this
brief and the relevant experiment README after pulling. Activate the target's
normal CANN/torch-npu environment. `source npu-setup` is only appropriate if that
helper exists on **this** server; it is not a dependency supplied by this repo.
Set these values to actual paths, not Blue Zone paths:

```bash
export PYTHON_BIN="$(command -v python3)"  # after activating the NPU environment
export BGE_CANN_ROOT="${ASCEND_HOME_PATH:?set to the actual CANN toolkit root}"
export BGE_WORK_ROOT="$REPO/tmp/26_bge_m3_inference/310p_$(git rev-parse --short=12 HEAD)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BGE_WORK_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
set -eo pipefail
{
  git rev-parse HEAD
  uname -m
  npu-smi info
  "$PYTHON_BIN" -c 'import sys,importlib.metadata as m; print(sys.executable); print(sys.version); print({n:m.version(n) for n in ("torch","torch-npu","transformers")})'
  command -v cmake g++ git dos2unix pigz
  cmake --version
  readlink -f "$BGE_CANN_ROOT"
} 2>&1 | tee "$BGE_WORK_ROOT/environment.log"
```

Also record the installed CANN and driver versions, device-selection environment
and the physical device chosen. Check that this CANN root contains `include/`
and `lib64/`. Do not blindly install the 910B versions: report your actual stack.
The scripts require torch.library.custom_op/fake registration, TorchAir's
cache_compile, and torch_npu quantize/quant_matmul/format_cast/trans_quant_param.

## 2. Obtain and build pinned upstream source locally

Use the complete source tree, not just the V2 folder: its implementation depends
on shared AddLayerNormQuant/AddLayerNorm files and build infrastructure.

```bash
export BGE_OPS_SRC="$BGE_WORK_ROOT/ops-nn"
git clone --depth 1 --branch v9.1.0 https://gitcode.com/cann/ops-nn.git "$BGE_OPS_SRC"
test "$(git -C "$BGE_OPS_SRC" rev-parse HEAD)" = ceb4536a2bd6fc99b85aec9d0fdcc0f470376292
(
  cd "$BGE_OPS_SRC"
  bash build.sh --pkg --soc=ascend310p --ops=add_layer_norm_quant_v2 \
    --no_force --vendor_name=bge_v2 -j8
) 2>&1 | tee "$BGE_WORK_ROOT/build.log"
git -C "$BGE_OPS_SRC" diff --exit-code
```

`--no_force` skips rebuilding dependency kernels supplied by CANN; it does not
specialize V2. Read upstream build prerequisites before starting. The 910B host
needed dos2unix and pigz in addition to its existing compiler tools. Upstream
builds can download build dependencies; capture any network/dependency failure.
If GitCode is inaccessible, report that precise blocker or use an available
mirror **only after verifying the same commit**. Do not silently use newer main.

Find the single generated `cann-ops-nn-bge_v2_linux-*.run` under `build_out`.
Do not assume aarch64: build for the work server's actual host architecture.

```bash
mapfile -t BGE_PACKAGES < <(find "$BGE_OPS_SRC/build_out" -maxdepth 1 -type f -name 'cann-ops-nn-bge_v2_linux-*.run')
test "${#BGE_PACKAGES[@]}" -eq 1
export BGE_PACKAGE="${BGE_PACKAGES[0]}"
sha256sum "$BGE_PACKAGE" | tee "$BGE_WORK_ROOT/package.sha256"
bash "$BGE_PACKAGE" --quiet --install-path="$BGE_WORK_ROOT/install" \
  2>&1 | tee "$BGE_WORK_ROOT/install.log"
export BGE_VENDOR="$BGE_WORK_ROOT/install/vendors/bge_v2_nn"
test -f "$BGE_VENDOR/op_api/lib/libcust_opapi.so"
```

Verify the generated 310P operator-info JSON contains `AddLayerNormQuantV2` and
that `nm -D` on `libcust_opapi.so` exports both ACLNN workspace and launch symbols.
If stock CANN already has V2, record that fact, but use our isolated built package
for replication; do not accidentally mix two implementations.

## 3. Direct Python/ACLNN gate, without loading the model

```bash
export ASCEND_CUSTOM_OPP_PATH="$BGE_VENDOR"
export LD_LIBRARY_PATH="$BGE_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
"$PYTHON_BIN" 26_bge_m3_inference/test_add_layer_norm_quant_v2.py \
  --expected-chip 310P --op-api "$BGE_VENDOR/op_api/lib/libcust_opapi.so" \
  --rows 256 --width 1024 --output "$BGE_WORK_ROOT/direct_256x1024" \
  2>&1 | tee "$BGE_WORK_ROOT/direct.log"
```

This checks with/without bias against CPU FP32 math and captures real NPU kernels.
Require finite outputs, FP16 atol .004/rtol .002, and INT8 maximum difference <=1.
Expect one V2 device kernel per call; inspect extra kernels if the runtime adds
any rather than treating a Python function name as proof of fusion. Keep all
raw profile files. The script disables NPU JIT compile for 310P, matching the
existing Qwen 310P experiments. Do not bypass a chip guard by editing source.

## 4. Private graph package and composed operator gate

```bash
export BGE_GRAPH_VENDOR="$BGE_WORK_ROOT/graph/vendors/bge_v2_nn"
"$PYTHON_BIN" 26_bge_m3_inference/prepare_v2_graph_package.py \
  --vendor "$BGE_VENDOR" --output "$BGE_GRAPH_VENDOR" --cann "$BGE_CANN_ROOT" \
  2>&1 | tee "$BGE_WORK_ROOT/graph_package.log"
export ASCEND_CUSTOM_OPP_PATH="$BGE_GRAPH_VENDOR"
export LD_LIBRARY_PATH="$BGE_GRAPH_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
"$PYTHON_BIN" 26_bge_m3_inference/probe_norm_v2_graph.py \
  --expected-chip 310P --op-api "$BGE_GRAPH_VENDOR/op_api/lib/libcust_opapi.so" \
  --rows 256 --output "$BGE_WORK_ROOT/graph_probe" \
  2>&1 | tee "$BGE_WORK_ROOT/graph_probe.log"
```

Keep basename **bge_v2_nn**: GE uses it to resolve generated Python compiler
modules. Renaming that directory caused “no supported Ops kernel and engine”.
Do not replace CANN's own files. Do not set `IGNORE_INFER_ERROR=1`.

The metadata shim links `exe_graph`, `register`, and `opp_registry`. An older
CANN may lack these interfaces; report that as compatibility work, not a kernel
math failure. The loader imports ES dependencies before torch_npu. Preserve
CANN's existing PYTHONPATH if adding any paths: replacing it can hide `tbe`.

The probe exercises bias absent/present, width 1024, both 1024- and 4096-output
INT8 linears, and the fused projection's weight orientation. It checks eager
against math/layout controls, compiled against eager, then profiles. Its four
returned tensors are FP16 norm, INT8 norm, already-INT8 matmul output, and raw
projection output. These are composed-op diagnostics, not model timings.
On 310P `graph_transpose` must report true. Only continue when all cases pass.

## 5. Obtain/verify BGE-M3 and run the paired model experiment

Use an existing local checkpoint if it matches `26_bge_m3_inference/release.json`:
BAAI/bge-m3 revision `5617a9f61b028005a4858fdac845db406aefb181`.
Otherwise download into a model-cache directory with sufficient space:

```bash
export BGE_MODEL_DIR=/absolute/path/to/local/bge-m3
"$PYTHON_BIN" 26_bge_m3_inference/download_model.py \
  --output "$BGE_MODEL_DIR" --endpoint https://hf-mirror.com \
  2>&1 | tee "$BGE_WORK_ROOT/model_download.log"
```

The downloader reuses files only when hashes match; Hugging Face's official
endpoint can also be passed. Do not commit weights. Offline servers need those
pinned files staged by their normal model-provisioning mechanism, not Blue Zone.
The benchmark verifies the full manifest again before inference.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" 26_bge_m3_inference/benchmark_norm_v2.py \
  --expected-chip 310P --model-dir "$BGE_MODEL_DIR" \
  --op-api "$BGE_GRAPH_VENDOR/op_api/lib/libcust_opapi.so" \
  --output "$BGE_WORK_ROOT/model_comparison" \
  2>&1 | tee "$BGE_WORK_ROOT/model_comparison.log"
"$PYTHON_BIN" 26_bge_m3_inference/summarize_norm_v2.py \
  --run-root "$BGE_WORK_ROOT/model_comparison" \
  --output "$BGE_WORK_ROOT/kernel_comparison.json"
```

The harness calibrates once using the committed 12 texts, shares identical INT8
weights/scales, checks captured real norm inputs, tests 8 held-out texts and two
semantic examples, and compares compiled B2/S128, B1/S512, B4/S512. Each model
lane receives two opposite-order profile captures with 10 active forwards each.
The output directory contains fresh graph caches; never share them across SoCs.

Check for 48 V2 calls, 48 remaining Quantize calls and 144 projection matmuls.
The 910B baseline had 49 AddLayerNorm +96 Quantize, but 310P may choose different
fusions/names: inspect actual rows before claiming equal counts. Keep device
kernel timings primary even if the candidate is slower.

The counter summarizer retains native AIV, AIC, and unprefixed pipeline fields.
Missing/N/A counters are explicit nulls with valid-sample counts, not zeros.
Do not equate a combined-core 310P field blindly with a separate AIV 910B field.
PipeUtilization measures busy-cycle ratios, not actual HBM GB/s or bandwidth
saturation. Keep any unsupported-metric warnings and report availability.

If B4/S512 fails due to capacity, retain all completed cases and report OOM at
that shape. Do not relabel a smaller input as the requested B4/S512 run.

## 6. Return a compact, independently interpretable report

Report:

- Research commit, upstream commit, package/API hashes, exact chip, physical
  device, architecture, Python/Torch/torch-npu/TorchAir/CANN/driver versions.
- Stage status: build, direct op, composed compiled probe, model cases, profiles.
- Direct and compiled norm/INT8 errors; final-embedding cosine vs regular W8A8
  and FP16; both semantic score matrices. Small static-W8A8 drift is acceptable;
  report it rather than claiming bitwise model parity.
- Per shape/lane: active forwards, kernel counts and summed duration, V2/norm/
  quant/matmul/softmax/Add/layout subtotals, internal device gaps, input shapes.
- MTE2/MTE3/vector/scalar/MAC busy times and ratios, blocks/core types, reported
  cube utilization where available. Distinguish unavailable counters from zero.
- Exact failing command and relevant traceback/compiler/plog if anything fails;
  proposed minimal source patch if needed. No silent fallback or global upgrade.
- Absolute output root and a retained archive excluding the large `cache/`
  directories. Include logs, result JSON, audited summaries, package manifest,
  and raw profiler CSV/trace/CANN files. The agent cannot push these; Luka relays
  the report/artifact through the usual work-server channel.

Useful reference: the Git-committed 910B model evidence contains all numerical
and compact counter results needed for comparison. Absolute cross-chip timings
are context; first compare regular versus V2 on the **same 310P**.
