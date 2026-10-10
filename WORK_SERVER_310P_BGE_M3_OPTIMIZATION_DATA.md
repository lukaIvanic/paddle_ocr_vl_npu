# 310P BGE M3 optimization diagnostics handoff

Collect the minimum evidence needed to optimize the existing AddLayerNormQuantV2
kernel for this 310P. Reuse the successful run and its working package. This is
**data acquisition and source inspection**, not a new kernel implementation,
rebuild, full model benchmark campaign, or environment upgrade. Return evidence
for the authoring agent to implement the first controlled change.

Device kernel duration, scalar/vector/MTE activity, tiling and instruction work
are primary. End-to-end timing is secondary. Continue through available steps
without waiting for confirmation, and clearly mark unavailable information.

## Existing successful run

The report screenshot identifies:

- Research checkout `/home/lukaiv/paddle_ocr_vl_npu_bge_v2`, source `7f75575c`.
- Run directory ending `310p_7f75575cc33c_20261010_175904`.
- `REPORT.md` and `bge_m3_v2_310p_evidence.tar.gz` (about 4.9 MB).
- CANN 9.1.0-beta.1, NPU 3 at the time of that run.
- Upstream ops-nn v9.1.0, commit
  `ceb4536a2bd6fc99b85aec9d0fdcc0f470376292`.
- All direct/composed/model gates passed. 839 -> 696 kernels/forward;
  48 V2, 48 Quantize and 144 quantized matmuls; summed kernel time +1.7–2.7%;
  V2 scalar ratio 0.88–0.95. These are reported results, not the missing raw data.

The old run already established correctness. Do not repeat all fourteen captures
merely because the authoring agent has only seen the screenshot. First locate
and return the old report, summary JSON and raw CSV/trace files if retained.
The old archive reportedly excluded traces; check the original run directories.

Use the existing working checkout. Read `AGENTS.md`, `CLAUDE.md`, and
`WORK_SERVER_310P_BGE_M3_V2.md`. Pull main only if it can fast-forward without
losing local changes. The 910B gamma-cache result is recorded at `a3bb1403`;
**do not install its 910B binary or apply its patch to this baseline**.

Keep the work server pull-only: no tracked-source edits, commits or pushes.
Do not terminate other jobs, replace global CANN, or disable certificate checks.
Use an idle device through the server's established selection procedure. Record
physical card/chip, container-visible logical index, and any vNPU partition.
NPU 3 is historical context, not an instruction to occupy a busy device.

## 1 Return existing evidence first

Find the named run/archive within the known research checkout and its research
output roots. Do not broadly search unrelated home folders or chat transcripts.
Copy these into a fresh diagnostics directory, preserving the originals:

- `REPORT.md`, environment and exact command records, `result.json`, and any
  kernel comparison/counter summary JSON.
- Existing `kernel_details.csv` for regular W8A8 and V2 for B2/S128, B1/S512,
  B4/S512. Include both repeated captures when they exist.
- Corresponding `trace_view.json` and `summary.json`, if still available.
- Direct V2 and composed-op result JSON, and package/build/source manifests.
- The exact local compatibility patch/build overlay and launch environment
  recipe that made this run succeed.

The screenshot reports two compatibility fixes: package `libes_nn.so` was
preloaded to avoid a duplicate-ES crash, and `v2_graph_infer.cpp` was included
as `op_host/bge_v2_infershape.cpp` in the package build because early shim
registration was displaced. Preserve the actual proven arrangement. Report
these changes separately from the kernel/tiler source; do not rediscover the
failure by dropping the working preload or by rebuilding an unpatched package.
Do not include credentials, full environment dumps, model weights or caches.

If the report plus old CSVs already answer a requested measurement, use them.

## 2 Identify the actual hardware and runtime

Record the exact SoC/SKU, board model, driver/CANN/torch/torch-npu versions,
virtualization/partition limits, total and currently available device memory,
and selected physical device. Use the existing environment, `npu-smi info`, and
board information supported by the installed `npu-smi --help`.

Retain the matching installed platform INI, with its path and SHA256. Common
location: `$ASCEND_HOME_PATH/<host-architecture>-linux/data/platform_config/`.
Report fields for AI cores, independent Vector Cores, UB, L1/L0 buffers, L2,
and memory size. These INI values are compiler definitions; actual installed
memory and vNPU resources must be checked on the machine.

Do not interpret opaque fields such as `ddr_rate` as measured GB/s. Do not add
AI Core count and independent Vector Core count and label that the V2 launch.
The V2 tiler calls `GetCoreNumAiv()`: on coupled Atlas inference hardware that
API returns AI Core count, not the separate Vector Core pool.

## 3 Recover kernel level timing and counters

For each shape and capture, return a table with:

- Package/lane, actual kernel/node name and type, input/output shapes and dtypes.
- Active-forward count, kernel count per forward, Block Num and core/task type.
- Mean and median duration in microseconds, with per-capture values retained.
- Absolute `scalar_time`, `vec_time`, `mte2_time`, `mte3_time`, `mac_time` and
  their ratios, using the native CSV column names. Include `aicore_time`, total
  cycles and I-cache counters if already captured.
- Missing/N/A fields as null, never zero; valid sample counts alongside means.

For the model, isolate the replaced norm/quant boundaries from all other work.
Match actual 310P node names/dataflow; do not hard-code the 910B naming scheme.
Separate the embedding site from residual sites where possible. If the regular
310P path uses multiple Add/Cast/LayerNorm kernels, include the complete replaced
sequence, not just a conveniently named LayerNorm. Explain the 143-kernel count
reduction with a type/count breakdown. Keep the remaining 48 quantizers separate.

Preserve absolute device gap time as well as its reported percentage reduction.
Scalar/vector/MTE pipelines overlap, so their times must not be summed into a
kernel duration. Neither scalar ratio nor MTE ratio alone proves a bottleneck.

The repository's `26_bge_m3_inference/summarize_norm_v2.py` understands plain
310P counters and aiv_/aic_ counters and audits profiler-step coverage. Use it
on an intact original run if its expected result/trace layout exists. Preserve
raw CSVs even if the existing parser cannot consume that run's schema; do not
edit tracked scripts or fabricate missing trace coverage.

## 4 Collect only missing direct profiles and tiling logs

Use the proven baseline V2 package and selected idle 310P. Resolve these variables
to the existing local paths (there are no Blue Zone dependencies):

```bash
export BGE_DIAG_ROOT="$PWD/tmp/26_bge_m3_inference/310p_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BGE_DIAG_ROOT"
export PYTHON_BIN="$(command -v python3)"  # after activating the established environment
export BGE_OP_API=/actual/working/vendor/op_api/lib/libcust_opapi.so
# Set ASCEND_RT_VISIBLE_DEVICES to the selected idle device using the server's
# proven mapping. The script calls set_device(0): logical 0 must be that device.
# Retain the working ASCEND_CUSTOM_OPP_PATH, LD_LIBRARY_PATH and LD_PRELOAD.
test -f "$BGE_OP_API"
```

If isolated profiles for the three shapes are missing, use the existing portable
script; no checkpoint or rebuild is needed:

```bash
for rows in 256 512 2048; do
  "$PYTHON_BIN" 26_bge_m3_inference/test_add_layer_norm_quant_v2.py \
    --expected-chip 310P --op-api "$BGE_OP_API" --rows "$rows" --width 1024 \
    --output "$BGE_DIAG_ROOT/direct_$rows" \
    > "$BGE_DIAG_ROOT/direct_$rows.log" 2>&1 || break
done
```

This script checks no-bias and broadcast-bias correctness; its three active
profile calls use **broadcast bias**. These short captures are diagnostics, not
an optimization speedup trial. Return raw CSV/trace files, not only result.json
(which omits most hardware counters). Stop on a failure and preserve its log.
Do not use `gamma_cache/run_910b.sh` or remove a 910B guard to make a helper run.

Collect **unprofiled or separately labelled debug** tiling evidence. Prefer existing
logs. Otherwise use the same direct script with CANN debug logging in separate
fresh processes/output directories; exclude all debug captures from timing:

```bash
for rows in 256 512 2048; do
  ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
    "$PYTHON_BIN" 26_bge_m3_inference/test_add_layer_norm_quant_v2.py \
    --expected-chip 310P --op-api "$BGE_OP_API" --rows "$rows" --width 1024 \
    --output "$BGE_DIAG_ROOT/debug_$rows" \
    > "$BGE_DIAG_ROOT/debug_$rows.log" 2>&1 || break
done
```

Extract the actual V2 tiling key, maxCoreNum, selected blocks, maxUbSize,
firstDimPerCore, rowPerTime/rowStep, tails and normalized stride. Some fields
may require reading the tiling data or deriving them from a known tiler: label
observed and derived values separately. Retain the full logs locally.
If stdout logging is unsupported, collect the process-specific CANN log instead.
Do not enable logging globally or modify another job's configuration.

## 5 Inspect the selected implementation

Confirm the package and selected normal FP16 kernel binary paths/hashes, compiler
architecture and keys. For normal V2 the expected keys are 1000 (no bias), 1002
(broadcast bias), and 1001 (elementwise bias); verify rather than assume.
Retain the selected ELF metadata, not an unrelated object's metadata.

Inspect pinned upstream source plus the **installed 310P SDK** implementation of
these operations. Report paths, hashes and short findings; no rebuild is needed:

- V2's `CopyInX1X2`, `CopyOutLayernormRes`, and `CopyOut`: are x1, x2, FP16 output
  and INT8 output all taking the row-by-row 2002 branch? Are GM and UB row strides
  both exactly 1024 elements, with no padding? This decides the first patch.
- `ReduceSumFP32`: confirm the 310P `ReduceSum` then `GetValue(0)` route and
  generated synchronization, if visible in existing disassembly/tooling.
- `dav_m200` implementations of ReduceSum, WholeReduceSum, Brcb/Broadcast,
  Sqrt/Div/Rsqrt and Cast. Confirm supported FP32 APIs and scratch requirements.
  A vector-looking C++ API can still contain software loops or extra buffers.

If an installed tool can disassemble the selected kernel, retain its supported
command, the relevant function and static instruction counts/types. Distinguish
static instructions from executed counts. An instruction pipeline trace is
optional only if supported on this chip and straightforward to collect.
Do not install a new toolchain or run a broad simulator campaign to obtain it.
Unavailable instruction evidence does not invalidate the earlier steps.

## 6 Return package and decision table

Produce one concise `REPORT.md` plus a tar.gz containing the old report, raw
CSV/trace evidence used, new diagnostics if needed, environment/commands,
compatibility recipe, selected tiling/ELF provenance and relevant SDK excerpts.
Keep profiled and debug runs in different directories. Exclude caches, weights,
installers and bulky unrelated logs. Record the archive's SHA256 and list any
omitted evidence. If an old archive suffices, return it with a small supplement.
No external upload or message is required; Luka will relay the files.

End the report with answers to these questions:

1. Can each of the four row-copy loops safely become one contiguous DataCopy
   per chunk for FP16 width1024 and static single-scale V2? List the exact guards.
2. What are duration, scalar/vector/MTE times, core count and rowStep on this
   device for the three shapes? Is the 0.88–0.95 scalar ratio reproduced?
3. What evidence identifies reduction/UB-to-scalar dependency, copy-command
   dispatch, or another cost as the first bottleneck? State unknowns plainly.
4. Are vector reductions and FP32 vector Sqrt/Div available in this installed
   SDK, and what scratch/layout cost would vector broadcasting require?

## Optimization hypotheses for the authoring agent

**First control: coalesce aligned copies.** On the normal path, combine each
chunk's x1, x2, FP16 normalized output and INT8 output into one DataCopy per
stream. Keep arithmetic, tiling, queue lifetimes and existing synchronization.
Guard actual GM/UB contiguity, byte alignment, transfer-size limits, relevant
attributes, and optional bias behavior. Keep the original fallback for other
layouts. Do not insert new per-row conditionals. Broadcast bias is already
loaded separately; an elementwise-bias fast path must handle its fifth stream.

With an unpartitioned eight-core launch and rowStep13, 2048 rows give 256 rows
per core and 20 chunks. Those four streams would use 80 bulk copy calls rather
than 1024 per-row calls per core. This is a **source-level call-count prediction**,
not measured instructions or a 12.8x speedup. Confirm actual tiling first.
No parameter-cache patch or double-buffer change belongs in this control.

**Next, potentially larger change: keep row statistics in vector/local-memory
form.** Calculate means/variances for a group of rows, consume them with vector
operations, and avoid repeated UB GetValue -> scalar arithmetic -> vector
handoffs. Use installed supported primitives. Changed reduction order can alter
FP16/INT8 results; compare FP32 math, unmodified V2 and real model samples. Start
with vector Sqrt+Div if suitable; native Rsqrt is a separate accuracy experiment.
The 310P Brcb API needs 8 KiB scratch, and generic Broadcast may involve additional
transposes/temporary tensors. Do not assume either is one cheap instruction.
In the authoring host's CANN9.0.1 dav_m200 Brcb implementation, the contiguous
32-bit path uses vadds + v4dtrans with vector barriers per repeat; the other
stride path reads scalars from UB and issues vector_dup with S_V synchronization.
Check the target SDK and chosen layout to avoid reintroducing the dependency
that this change is meant to remove.

The 910B gamma/beta cache was correct but not a consistent speedup. It remains a
separate hypothesis for 310P, not an automatic improvement. Extra UB capacity
alone also does not justify enabling double buffering before inspecting overlap.

## Research sources and scope

The authoring host's CANN9.0.1 `Ascend310P3.ini` lists architecture2002, 8 AI Cores,
7 independent Vector Cores, 256KiB UB, 1MiB L1, 64KiB L0A/B, 256KiB L0C and 16MiB
L2. It lists 24GB device memory, which is **not proof of this server's board or
memory capacity**. Local package/platform files and device queries take priority.

- [Coupled and split architecture](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html).
- [GetCoreNumAiv semantics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_1031.html).
- [Atlas 300I Pro specifications](https://e.huawei.com/cn/products/computing/ascend/atlas-300-ai): 24GB LPDDR4X, advertised 204.8GB/s; applies to that board.
- [ReduceSum implementation caveat](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha003/API/ascendcopapi/atlasascendc_api_07_0078.html).
- [WholeReduceSum support](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0081.html).
- [Brcb support and 8KiB scratch](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0089.html).
- [Rsqrt accuracy caveat and Sqrt/Div alternative](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0030.html).

Public API support is not an installed-version compile/run result. This handoff
collects that distinction without changing the working baseline.
