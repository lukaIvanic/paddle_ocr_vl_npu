# Ascend custom operator handbook

This handbook is the reusable workflow for developing, loading, validating,
profiling, and integrating custom Ascend operators in this repository. It was
written after reconstructing IncreFlashAttention, changing its launch to an
AIV-only runtime shape, exposing it under an independent operator identity,
and testing it from direct PyTorch eager execution through a real compiled B1
PaddleOCR-VL forward pass.

The central rule is simple:

> Create a separately named, narrowly contracted operator and validate it from
> the kernel outward. Run direct eager before TorchAir. Run the real model only
> after both isolated lanes pass.

This workflow is authoritative for the local authoring and Blue Zone 910B
lanes. A 310P result needs a separate run and a separate report.

## Current validated state

The retained 16-block implementation was verified on 2026-08-09 at commit
`b4d0a75`; the forced 32-block split-K control was added and verified at commit
`8a1041f`. Both ran on a physical Ascend 910B2.

- The independent B1 GQA operator works through direct PyTorch eager and
  TorchAir under the name `PaddleGqaIncreFlashAttentionAiv`.
- The final model preset is `combined_apply_gqa_aiv_b1`. It requests 16 AIV
  cores and fails closed outside B1/TorchAir. Stock IncreFA remains the default.
- KV128, KV512, masked KV1536, and KV2048 pass stock tolerance and an
  independent CPU FP32 reference. A real 374-token OCR generation matches
  token IDs, text, and EOS exactly.
- Runtime counters prove zero AIC execution and nonzero AIV execution. The CANN
  task-type label alone is not reliable for this question.
- One controlled same-device NPU6 ABBA run measured 795.31 tok/s stock and
  811.14 tok/s custom. This 1.99% result is promising but small and
  device-sensitive, so it is not yet a production-default decision.
- A separate KV1024 split-K package proves an actual 32-AIV-block launch. Its
  same-device `16 -> 32 -> 32 -> 16` full-decoder sequence averaged 778.38 and
  829.59 tok/s respectively, a 6.58% advantage for 32 blocks. The retained
  16-block preset remains unchanged while this result is repeated.
- Whole attention on AI CPU is correct but 29.9x to 553.1x slower than stock
  IncreFA. Do not use AI CPU for QK, softmax, or AV.

The compact result bundle is
[GQA AIV B1 evidence](../../tmp/09_persistent_page_engine/gqa_aiv_b1_1d16f33/README.md).
The build and operator-specific commands are in the
[GQA AIV operator README](paddle_gqa_increfa_aiv/README.md).

## 1. Evidence vocabulary

Keep four kinds of statements separate:

- **Official contract:** behavior stated by Huawei documentation or pinned
  upstream source.
- **Repository contract:** behavior enforced by this repository's code and
  build gates.
- **Measured result:** output from a retained command on a named physical NPU.
- **Inference:** the best explanation consistent with the evidence, but not yet
  directly proved.

Do not turn an inference into a fact by repeating it. For example, our profiles
show that the first custom eager bridge has much more launch-cadence overhead
than stock. They do not yet prove which internal cache in the stock op-plugin
removes that overhead.

## 2. The operator is a stack of contracts

A custom operator is not one function. It is a stack. A unique name and a valid
contract are required at every callable layer.

| Layer | Responsibility | Current MHA AIV example |
| --- | --- | --- |
| Model call site | Explicitly selects the experiment | `mha_incre_flash_attention_aiv(...)` |
| PyTorch graph op | Stable FX/Dynamo identity | `paddleocr_vl::mha_incre_flash_attention_aiv` |
| PyTorch eager op | Direct NPU dispatcher identity | `paddleocr_vl_npu::paddle_mha_incre_flash_attention_aiv_eager` |
| TorchAir converter | Emits the intended GE node | `PaddleMhaIncreFlashAttentionAiv` |
| Public ACLNN API | Stable external enqueue API | `aclnnPaddleMhaIncreFlashAttentionAiv` |
| Generated inner API | Validation, tiling, executor, launch | `aclnnInnerPaddleMhaIncreFlashAttentionAiv` |
| GE/CANN op definition | Inputs, outputs, attributes, SoC support | `PaddleMhaIncreFlashAttentionAiv` |
| Host tiler | Selects key, workspace, block count, launch mix | copied and patched IncreFA tiler |
| Tiling-data schemas | Binary layout of host-to-kernel data | upstream composite structure names |
| Kernel entry | Device entry point | `paddle_mha_incre_flash_attention_aiv` |
| Kernel metadata | Task ratio, sync, key, binary type | `0:1`, hard sync, key `11000000000100000` |
| Vendor package | Installs host, kernel, and op-api artifacts | `paddle_mha_increfa_aiv_transformer` |

This stack explains why an operator can compile but fail at runtime, run eagerly
but fail under TorchAir, or have a faster kernel but a slower PyTorch call.

## 3. Use an independent identity

Do not overload the stock operator name while developing a new implementation.
Do not hide the experiment behind a branch inside the stock Python wrapper.

An independent identity gives us:

- an unambiguous PyTorch call site;
- stock and candidate operators in the same environment;
- separate GE nodes and graph caches;
- proof that a run did not silently fall back to stock;
- simple removal if the experiment is rejected;
- a contract that can be narrower than the stock operator.

Write the identity ledger before writing code:

```text
PyTorch graph:
PyTorch eager:
GE/CANN:
public aclnn:
generated inner aclnn:
kernel entry:
vendor:
```

Renaming every string is not correct. The callable identities above must be
unique. Nested tiling-data registration names may need to remain identical to
their C++ structure types. Section 8 explains this trap.

## 4. Golden validation ladder

Each rung has a stop condition. Do not skip a failed rung to obtain a later
performance number.

| Rung | Question | Required evidence | Stop when |
| ---: | --- | --- | --- |
| 0 | Is the machine state trustworthy? | commit, source hash, selected physical NPU, process state | source drift or device ambiguity exists |
| 1 | What exact contract are we implementing? | shape/dtype/layout/attribute table | production and test contracts are conflated |
| 2 | Is the package structurally independent? | unique names, exported symbols, selected object proof | stock name or fallback remains possible |
| 3 | Is the intended kernel binary present? | exact required key set, metadata, ELF symbols, package hashes | cube symbol or unexpected key remains |
| 4 | Does direct eager produce the right tensor? | full-output parity and hashes | numerical parity fails |
| 5 | What does direct eager cost? | non-profiled repeated NPU and host timings | first-use or profiler time is mixed into steady time |
| 6 | What does the kernel body do? | isolated profiler trace | helper kernels or unexpected data transforms appear |
| 7 | Does TorchAir emit and run the distinct GE op? | converter proof, graph artifact, exact output | conversion, tiling, or execution fails |
| 8 | What is steady isolated graph latency? | one shape per process, distinct caches | cache warnings or recompilation contaminate timing |
| 9 | Does the production call shape work? | real model contract and exact generation | MHA/GQA, mask, or cache semantics differ |
| 10 | Does the real B1 forward improve? | interleaved fresh-process runs and full-step timing | correctness fails or the full step regresses |

## 5. Rung 0: establish a trustworthy lane

### Local authoring lane

The Mac edits tracked files, commits, pushes, and drives the remote run. It has
no NPU and cannot validate inference.

Before editing:

```sh
git status --short
git log -5 --oneline
```

Preserve unrelated changes. Stage explicit paths only.

### Blue Zone 910B lane

The container is pull-only for tracked source. Always start with:

```sh
ssh blue_zone_npu_container '
  cd /workspace/repos/paddle_ocr_vl_npu &&
  source npu-setup &&
  git rev-parse --short HEAD &&
  printf "ASCEND_RT_VISIBLE_DEVICES=%s\n" "$ASCEND_RT_VISIBLE_DEVICES"
'
```

`npu-setup` selects a free physical device and exposes it as logical `npu:0`.
Record the printed physical device. Do not terminate another user's process.
Do not silently choose a device manually when the selector reports no free
device.

Use the model-only interpreter for isolated probes:

```text
/usr/local/python3.12.13/bin/python3
```

Use the experiment-09 environment for real page/model entrypoints:

```text
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python
```

### Device control rule

If a candidate times out, immediately run the same smallest stock control on
the same physical device. If both time out, the run does not identify a
candidate failure. Exclude it, clean up only your processes, and rerun after
`npu-setup` selects a healthy free device.

`npu-smi` reporting `Health: OK` is not sufficient proof that the runtime lane
is usable. During this work, physical NPU 5 passed that check but both stock and
custom attention calls timed out. All results from that device were excluded.

## 6. Rung 1: freeze the exact operator contract

Write the contract before selecting or editing a tiling key.

```text
SoC:
batch size:
query shape:
key/value shape:
output shape:
dtype for every tensor:
layout:
mask dtype and shape:
actual sequence length representation:
Q heads:
KV heads:
head dimension:
attributes and exact values:
workspace expectations:
aliasing or mutation:
```

The current independent MHA AIV operator is deliberately narrow:

| Field | Value |
| --- | --- |
| SoC | Ascend 910B2 |
| Batch | 1 |
| Q | FP16 BNSD `[1, 16, 1, 128]` |
| K/V | FP16 BNSD `[1, 16, KV, 128]` |
| Mask | bool `[1, 1, 1, KV]` |
| Output | FP16, same shape as Q |
| Actual lengths | none |
| Attention | MHA, 16 Q heads and 16 KV heads |
| Precision | `inner_precise=1` |
| Fixed key | `11000000000100000` |

Production PaddleOCR-VL decode is different. It has 16 Q heads and 2 stored KV
heads. That is GQA.

Huawei's [open IncreFA design](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/common/IFA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)
describes the 910B/A2 all-vector template as an FP16, non-PA, **non-GQA**
path. It also documents GQA as a separate vector G-axis splitting problem.
Therefore the existing MHA all-vector key is not a validated GQA
implementation. Changing `num_key_value_heads` is not enough.

Do not expand the production cache from 2 to 16 KV heads and call that a GQA
optimization. We measured that workaround in the real B1 forward and it
regressed latency by about 2%.

## 7. Source provenance and build isolation

Pin the upstream source at three levels:

1. repository commit;
2. hashes of the kernel entry and host tiler being patched;
3. the exact tiling key being compiled.

The current build pins:

```text
ops-transformer commit:
afe72144f9f2ac8441929035795db88a111b30c5

FP16 MHA all-vector tiling key:
11000000000100000
```

Use a detached build worktree under `.runtime_cache/`. Keep the recovered
upstream checkout unchanged. Apply tracked patches and overlays to the build
worktree. Persist a manifest of the effective source.

The current builder also:

- verifies the upstream commit and source hashes;
- renames the copied operator directory;
- disables the copied stock definitions and API surfaces;
- overlays the new definition, kernel entry, and public ACLNN wrapper;
- compiles only the selected operator and fixed tiling key;
- installs to a run-specific directory;
- validates the installed package before printing its environment path.

Build on Blue Zone:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
bash 09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv/build.sh
```

The script prints:

```text
PADDLE_MHA_INCREFA_AIV_BUILD_ROOT=...
PADDLE_MHA_INCREFA_AIV_PACKAGE=...
PADDLE_MHA_INCREFA_AIV_SET_ENV=...
PADDLE_MHA_INCREFA_AIV_OP_API=...
```

Source the exact printed `SET_ENV` file. Do not use the most recent directory
by guesswork.

### Build time is not runtime time

The recovered `ops-transformer` package build recreates generated CMake output
and bundled host dependencies. In one observed build, the device kernel stage
was about 9 seconds while the full package took about 5 minutes. That is a build
system cost. It is not a TorchAir graph compile and not per-token latency.

Keep these timers separate:

```text
source preparation
device kernel compilation
host tiler/op-api build
package construction
package installation
PyTorch extension build
TorchAir first call
TorchAir cache load
steady graph replay
kernel body
real decoder step
end-to-end request
```

## 8. Callable names versus tiling schema names

The independent callable operator uses the new top-level GE identity:

```text
PaddleMhaIncreFlashAttentionAiv
```

The top-level tiling-data and tiling-template registrations must target this new
operator. However, the copied composite tiling structure contains nested C++
types such as:

```text
IncreFlashAttentionBaseParamsOp
IncreFlashAttentionCoreParamsOp
IncreFlashAttentionTilingDataOp
IncreFlashAttentionTilingDataPrefixOp
```

We initially renamed those nested registrations. TorchAir graph compilation
then failed with:

```text
EB0500 / TBEPythonError
IncreFlashAttentionTilingDataOp is not define
```

CANN's static tiling decoder resolved nested members by the composite C++
structure names. Restoring those nested schema names fixed compilation without
changing the independent callable operator identity.

General rule:

- rename callable operator, kernel, package, public API, top-level tiling-data,
  and tiling-template identities;
- retain nested schema registrations when their names are part of a copied C++
  composite serialization contract;
- gate the installed tiling library with `strings` so required top-level and
  nested names are present.

## 9. AIV launch types and synchronization

Huawei's [kernel task-type API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0218.html)
documents both `KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_MIX_AIV_1_0`. It also
states that a per-tiling-key `KERNEL_TASK_TYPE(key, value)` overrides the
global default.

The recovered all-vector IncreFA implementation performs FlashDecode reduction
with `SyncAll()`. A bare `KERNEL_TYPE_AIV_ONLY` object failed on 910B2 with a
vector UB/D-cache bus error:

```text
507035 at vector PC offset 0x42cc
```

The same failure occurred with 48 blocks and with one block. That ruled out a
simple multi-core participant-count explanation. The working launch is:

```cpp
KERNEL_TASK_TYPE(
    QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING,
    KERNEL_TYPE_MIX_AIV_1_0);
```

This is a hard-synchronization MIX envelope with a `0:1` task ratio. It launches
the vector function and no cube function.

Do not infer cube execution from `coreType: MIX` alone. Prove the launch with
all of the following:

- kernel metadata `taskRation: "0:1"`;
- `intercoreSync: 1`;
- the expected `crossCoreSync` value;
- one `_mix_aiv` ELF function;
- no `_mix_aic` ELF function;
- a profiler trace with zero AICore compute for the custom kernel.

The metadata field is spelled `taskRation` in CANN output. Preserve that spelling
when parsing the JSON.

### Block dimension

Huawei documents
[`CalcTschBlockDim(sliceNum, aicCoreNum, aivCoreNum)`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_1033.html)
for separated Cube/Vector scheduling. The all-vector host tiler must not ask
the scheduler for cube cores:

```cpp
const uint32_t launchAicNum =
    perfMode_ == IfaPerfMode::BMM_ALL_BY_VEC ? 0U : aicNum;
ifaContext_->numBlocks = ascendcPlatform.CalcTschBlockDim(
    aivNum,
    launchAicNum,
    aivNum);
```

Huawei's [`SyncAll` documentation](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0204.html)
also warns that `blockDim` must not exceed the number of cores used by the
operator, or framework multi-round scheduling can cause the kernel to stop
responding. Treat block calculation and synchronization as one contract.

## 10. Static package gates

A successful compiler exit is not enough. The build must fail unless all static
gates pass.

### Kernel artifact gate

- exactly one kernel JSON for the new GE op;
- exactly one matching object;
- exact expected tiling key;
- package exists and is non-empty;
- hashes retained for JSON, object, and package.

### ELF gate

```sh
readelf -Ws <kernel-object>
```

Require `_mix_aiv`. Reject `_mix_aic` for the all-vector candidate.

### Metadata gate

Inspect the JSON and, when available, the CANN
[`msobjdump --dump-elf`](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0103.html)
output. Require the intended values for:

```text
coreType
intercoreSync
kernelList[].tilingKey
kernelList[].taskRation
kernelList[].crossCoreSync
```

### Public API gate

```sh
nm -D <installed-op-api-lib>
```

Require both exported symbols:

```text
aclnn<OpName>GetWorkspaceSize
aclnn<OpName>
```

### Tiling library gate

Find the one installed `libcust_opmaster_rt2.0.so`. Check that it contains the
new top-level op name and all required nested tiling schema names.

### Selection proof

For a decisive proof that stock fallback is impossible, use a disposable copy
of the custom OPP and remove only the selected custom object. The call should
fail while resolving that exact object. Never perform this test in the retained
package or a broad installation directory.

## 11. Direct PyTorch eager comes first

The official
[Ascend C++ extension sample](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation/README.md)
separates two integrations:

- a C++ `PrivateUse1` implementation that invokes ACLNN in eager mode;
- a TorchAir FX-to-GE converter for compiled mode.

Follow the same separation.

The eager extension needs:

1. a unique `TORCH_LIBRARY` or `TORCH_LIBRARY_FRAGMENT` schema;
2. a `PrivateUse1` implementation;
3. a `Meta` implementation for shape propagation;
4. device, dtype, rank, layout, shape, and attribute checks;
5. output allocation with the correct NPU format behavior;
6. an ACLNN enqueue call to the **public separately named API**;
7. no Python fallback to stock.

Verify registration before timing:

```python
table = torch._C._dispatch_dump_table(PYTORCH_OP_NAME)
assert "PrivateUse1" in table
```

Build the current eager extension after sourcing the custom OPP:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
bash \
  09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv/\
pytorch_extension/build.sh
```

Run direct eager before importing TorchAir:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/probes/\
compare_paddle_mha_increfa_aiv_eager.py \
  --output .runtime_cache/paddle_mha_increfa_aiv/compare_eager.json
```

The probe must record:

- `torchair_used: false`;
- `python_stock_fallback: false`;
- `same_name_override: false`;
- physical NPU;
- full contract;
- first call separately;
- output hashes;
- exact/allclose difference;
- repeated NPU-event and host-wall distributions.

### Eager benchmark method

- use deterministic CPU-generated inputs copied to NPU;
- compare the full FP32-materialized output, not only argmax;
- use the real KV lengths of interest;
- separate first use from warm steady calls;
- synchronize before the measured region;
- time many calls per block;
- retain every block value and summary statistics;
- use fresh processes for interleaved control/candidate runs when the expected
  change is small;
- profile only after the non-profiled benchmark is complete.

### What the eager control taught us

At KV128 on physical 910B2 NPU 6:

| Level | Stock | Custom AIV | Interpretation |
| --- | ---: | ---: | --- |
| Direct eager cadence | 51.37 us | 162.64 us | custom bridge is much slower |
| Kernel body | 23.89 us | 20.80 us | custom vector kernel is 12.9% faster |

The profiler showed one attention kernel per call and no helper or TransData
kernels. The regression was outside the kernel body. This is why direct eager
and kernel time must both be measured.

## 12. TorchAir integration is a separate lane

The graph-facing Python op may use stock PyTorch code as its eager correctness
body, provided its registered converter never lowers that body. The converter
must emit only the separately named GE operator.

The current graph integration has:

- `torch.library.custom_op` for a stable PyTorch graph identity;
- a fake implementation returning the correct metadata shape;
- one registered TorchAir converter;
- explicit GE input lists for dynamic K/V inputs;
- typed GE attributes;
- one named output;
- an idempotent converter-registration guard.

The essential converter pattern is:

```python
@register_converter(torch.ops.<namespace>.<op>.default)
def convert(..., meta_outputs=None):
    del meta_outputs
    return ge_custom_op(
        "<IndependentGeOp>",
        inputs={...},
        attrs={...},
        outputs=["<output-name>"],
    )
```

The GE input names, dynamic-list wrapping, attributes, and output name must
match the CANN `OpDef` exactly.

### Cache discipline

Use separate cache directories for:

- stock and custom;
- each operator build;
- each static shape;
- concurrent processes.

After a kernel rebuild, do not reuse the old graph cache. A cached graph can
embed or reference the prior device binary.

For isolated `cache_compile` benchmarks, run one KV shape per process. A
multi-shape attempt during this work emitted repeated `recompiled` cache
warnings. Its timings were excluded even though it eventually returned. A clean
benchmark is cheaper than explaining a contaminated one.

Example clean custom lane:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
PYTHONPATH=09_persistent_page_engine:$PYTHONPATH \
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/probes/\
compare_paddle_mha_increfa_aiv.py \
  --lanes custom \
  --kv-lengths 512 \
  --cache-root \
    .runtime_cache/paddle_mha_increfa_aiv/torchair_custom_<build> \
  --output \
    .runtime_cache/paddle_mha_increfa_aiv/custom_kv512_<build>.json
```

Run stock in a fresh process without sourcing the custom package unless that
package is deliberately required for a same-process parity test.

### Separate compilation from replay

Record:

```text
first call / compile or cache load
warmup calls
steady NPU event time
steady host wall time
graph cache directory
OM and compiled-module timestamps
```

An `.om` that has already been written and stops changing is strong evidence
that a later long wait is execution, not compilation. In the failed zero-cube
GQA experiment, graph artifacts were complete before the process spent more
than ten minutes in a post-compile runtime hang.

### Current isolated graph result

Clean one-shape-per-process results on physical 910B2 NPU 6 were:

| KV | Stock | Custom AIV | Change | Output |
| ---: | ---: | ---: | ---: | --- |
| 128 | 246.47 us | 136.58 us | -44.6% | bit-exact |
| 512 | 245.39 us | 147.84 us | -39.8% | bit-exact |
| 2048 | 240.61 us | 245.21 us | +1.9% | bit-exact |

The result is shape-dependent. The absolute single-op graph times also include
large fixed graph-launch cost. They are not decoder token latency.

## 13. Real model gates

An isolated operator win is only a candidate. The real forward decides whether
the change improves the system.

### First run the full raw-eager control

Running the whole real B1 GQA path without TorchAir established the backend
boundary:

```text
effective decode throughput: 69.07 tok/s
compiled B1 range reported for the project: about 700-800 tok/s
```

Raw eager is therefore a valuable diagnostic lane, but not a production backend
for this decoder.

### Then run the actual compiled production contract

The real test must preserve:

- B1;
- 16 Q heads and 2 stored KV heads;
- static cache capacity and real cache position;
- bool future-slot mask semantics;
- all 18 decoder layers;
- real weights and LM head;
- real prefill state;
- the same EOS and output-materialization behavior;
- exact token, text, stop-reason, and hash comparison.

Do not present an expanded 16-head MHA cache as the production GQA path. Do not
present one isolated attention call as tokens per second.

Use interleaved fresh-process control/candidate order when measuring a small
change. Report distributions and exact commands, not only the best run.

### Adoption gate

Adopt a custom operator only if all are true:

- the real contract enters the separate op explicitly;
- all required correctness checks pass;
- the result repeats across fresh processes;
- the full decoder step improves;
- the end-to-end scheduler/request metric does not regress materially;
- memory and cache behavior remain acceptable;
- the result is labeled with commit, physical NPU, and artifacts.

## 14. Timing taxonomy

Use the correct name for every timing.

| Timing | What it answers | What it does not answer |
| --- | --- | --- |
| Package build | Can CANN construct an installable OPP? | graph compile or runtime speed |
| Extension build | Can PyTorch load the eager adapter? | operator correctness |
| First eager call | Initialization and first launch | steady call latency |
| Eager NPU-event cadence | End-to-end enqueued call cadence | kernel body alone |
| Profiled kernel duration | Device compute inside the kernel | unprofiled throughput |
| TorchAir first call | compile or cache-load path | steady replay |
| Single-op graph replay | isolated graph/operator integration | full decoder token latency |
| Decoder model+argmax | device decode step | host scheduler and prefill |
| Continuous decode wall | realized decode cadence | total page pipeline |
| Full request wall | request-level latency | steady decode alone |

If someone reports a “13-minute compile,” locate the last artifact write and the
active thread first. Classify the time before trying to optimize it.

## 15. Failure atlas

| Symptom | Likely cause | Next action |
| --- | --- | --- |
| `IncreFlashAttentionTilingDataOp is not define` | nested composite schema names were renamed | restore nested C++ structure registration names; retain only the new top-level op identity |
| custom op is not found | custom OPP environment was not sourced, or installed package is incomplete | verify printed `SET_ENV`, `ASCEND_CUSTOM_OPP_PATH`, exported ACLNN symbols, and tiling library |
| eager output is exact but call is much slower | dispatcher, workspace/executor creation, or launch bridge overhead | profile one call family; compare kernel time with full eager cadence; inspect `EXEC_NPU_CMD` integration before changing kernel math |
| repeated `recompiled` warnings in a shape matrix | multiple static shapes share one process/compiler context | run one shape per process with a unique cache directory |
| `.om` exists but the first execution never returns | post-compile runtime hang | inspect artifact timestamps, CPU/runtime threads, and NPU utilization; stop only your process and report it as runtime, not compile |
| stock and custom both time out | unhealthy or wedged device/runtime lane | exclude the run and let `npu-setup` select another free device |
| vector bus error `507035` around `SyncAll` | bare AIV launch does not satisfy the recovered kernel's synchronization envelope | test `MIX_AIV_1_0`, metadata, blockDim, and workspace contract |
| metadata says `MIX` and reviewer assumes cube runs | `coreType` is being interpreted alone | inspect `taskRation`, `_mix_aiv`/`_mix_aic` symbols, and profiler AIC/AIV time |
| MHA passes but GQA hangs or diverges | an MHA tiling/kernel contract was applied to GQA | stop; build a distinct GQA operator and validate its own tiling path |
| package build takes minutes | host tools and package scaffolding rebuild | split build-stage timers; do not call this graph compilation |
| graph result is faster but real forward regresses | graph scheduling, occupancy, cache transformation, or diluted per-op saving | profile the full decoder and quantify predicted per-layer saving before modifying more kernels |
| output mismatch appears only after slot/cache updates | state propagation or aliasing error, not necessarily attention math | test multiple decode steps and compare KV state, not only a single output |
| installed tiling library gate finds zero files | package layout depth changed | inventory the exact install tree and fix the narrow search depth; do not weaken the one-library gate |

## 16. Evidence package

Retain small reproducible evidence under:

```text
tmp/<experiment>/<run-name>_<commit>/
```

Prefer this structure:

```text
README.md
command.txt
environment.txt
exit_code.txt
run.log
result.json
profile_parse_summary.md
sha256.txt
```

`command.txt` is authoritative. It should contain:

```text
local/project commit
upstream source commit
package/object hashes
hostname
physical NPU and logical device
CANN, torch, and torch_npu versions
interpreter
custom OPP environment path
graph cache path
exact command
```

The summary must say which values are measured, which are user-supplied
baselines, and which explanation is inferred.

## 17. Copyable review checklist

### Identity

- [ ] New PyTorch graph name.
- [ ] New PyTorch eager name.
- [ ] New GE/CANN name.
- [ ] New public and inner ACLNN names.
- [ ] New kernel entry and vendor.
- [ ] Stock operator remains callable and unchanged.
- [ ] No Python or package fallback can hide selection.

### Contract

- [ ] SoC, batch, dtype, layout, shapes, heads, mask, lengths, attributes, and
      mutation are written down.
- [ ] Test contract equals the intended model contract.
- [ ] MHA and GQA are not conflated.
- [ ] The tiling key is valid for the exact contract.

### Build

- [ ] Upstream commit and critical source hashes are pinned.
- [ ] Original upstream checkout is clean.
- [ ] Build worktree and overlays are reproducible.
- [ ] Only required operator/key variants are compiled.
- [ ] Package, object, JSON, and installed libraries are hashed.

### AIV proof

- [ ] `taskRation` is the intended value.
- [ ] Synchronization metadata is correct.
- [ ] `_mix_aiv` exists.
- [ ] `_mix_aic` is absent when zero-cube is required.
- [ ] Tiler passes zero requested AIC cores for the all-vector mode.
- [ ] Profile confirms zero cube compute.

### Eager

- [ ] `PrivateUse1` and `Meta` implementations are registered.
- [ ] Public separate ACLNN symbols exist.
- [ ] Full outputs match stock.
- [ ] First use is separate from steady calls.
- [ ] Non-profiled timings precede profiler conclusions.
- [ ] Kernel duration and full eager cadence are reported separately.

### TorchAir

- [ ] Converter emits only the new GE op.
- [ ] GE inputs, lists, attributes, and output names match `OpDef`.
- [ ] Stock and custom use separate caches.
- [ ] One static shape runs per benchmark process.
- [ ] Compile/cache-load time is separate from replay.
- [ ] Output matches stock exactly or to a justified tolerance.

### Real forward

- [ ] The production shape enters the new operator explicitly.
- [ ] Multi-step KV state remains correct.
- [ ] Token IDs, text, stop reason, and hashes match.
- [ ] Full decoder and request metrics are reported.
- [ ] Runs are interleaved across fresh processes.
- [ ] Commit, physical NPU, exact command, and artifacts are retained.

## 18. Completed GQA AIV implementation

The separate GQA operator is now implemented under this identity:

```text
PyTorch graph: paddleocr_vl::gqa_incre_flash_attention_aiv
PyTorch eager: paddleocr_vl_npu::paddle_gqa_incre_flash_attention_aiv_eager
GE/CANN:       PaddleGqaIncreFlashAttentionAiv
public aclnn:  aclnnPaddleGqaIncreFlashAttentionAiv
kernel:        paddle_gqa_incre_flash_attention_aiv
vendor:        paddle_gqa_increfa_aiv
```

The implementation exposed every query head as one all-vector work item and
fixed the non-split and FlashDecode workspace/output offsets. Two bugs are worth
remembering:

1. The first GQA port computed one query head per KV group. An eager output
   allocator reused stock-filled memory and briefly hid the unwritten heads.
   Always run the candidate first, initialize output defensively, compare every
   head, and retain output hashes.
2. The FlashDecode port launched all 16 requested AIV blocks, but an internal
   guard still used `batch * kvHeads * splits`. Only the first two blocks wrote
   workspace. Runtime launch logs disproved the initial launch-count hypothesis;
   changing the guard to `batch * queryHeads * splits` fixed KV2048.

The recovered pipeline supports only one GQA query-head work item per AIV core.
Requests below 16 timed out and are now rejected. For the real KV1536 cache,
requests of 16, 32, or 48 all become the same 16-block non-split launch. At
KV2048, the extra counts create real FlashDecode splits, but 16 had the best
clean custom mean. The production-facing experimental preset therefore requests
16 cores and is explicitly B1/TorchAir-only.

The core-count evidence must be interpreted through the actual launch, not the
requested attribute:

| Shape | Requested cores | Actual behavior | Decision |
| --- | ---: | --- | --- |
| masked KV1536 | 16 | 16-block non-split | keep |
| masked KV1536 | 32 | same 16-block non-split | no additional parallelism |
| masked KV1536 | 48 | same 16-block non-split | no additional parallelism |
| KV2048 | 16 | one query-head work item per core | fastest clean custom mean |
| KV2048 | 32 or 48 | real FlashDecode splits | slower than the 16-core custom row |

Do not select a core count from separate-process timing alone. First confirm
`Block Num`, tiling key, and split count in the profiler export.

Correctness is complete for the narrow contract. Direct eager and TorchAir pass
at KV128, KV512, masked KV1536, and KV2048 against stock tolerance and an
independent CPU FP32 reference. A 374-token real OCR generation is token-,
text-, and EOS-exact.

The runtime is also proven AIV-only. The installed object has no `_mix_aic`
function, metadata uses `taskRation: "0:1"`, and the profile has zero AIC time,
cycles, MAC time, and cube utilization with nonzero AIV time and cycles. CANN
may still label the task `MIX_AIC`; never use that label without the counters.

Performance is a small experimental win, not yet a production decision. A
same-device NPU6 ABBA sequence of four 200-step B1/KV1024 runs measured 795.31
tok/s stock and 811.14 tok/s custom on average: 1.99% more throughput and 1.95%
less step latency. A separate NPU3 pair had the opposite sign, so physical
device and run order must remain part of the evidence. The matched full profile
showed the custom attention kernels themselves were faster—16.282 us versus
18.071 us average across 54 tasks. The next optimization target is still
graph-level cadence or hard-sync scheduling. At this stage, requesting more
cores at KV1024 did not change the actual 16-block launch; Section 21 records
the later forced-split control that made the 32-core question measurable.

After the exploratory preset was renamed, the Blue Zone checkout pulled exact
commit `b4d0a75` and ran `combined_apply_gqa_aiv_b1` on physical NPU6 with B1,
KV1024, and the real full decoder. The short 20-step smoke completed at 741.08
tok/s. This is only final-name and graph-wiring validation; it is not a
replacement for the four-lane 200-step ABBA performance result.

### Build-time classification learned from GQA

The final two-key AscendC object compiled in about 9 seconds. The fresh package
command took about 4 minutes 53 seconds because the upstream wrapper rebuilt
Abseil, protobuf, libprotoc, ONNX plugins, and package scaffolding. Isolated
TorchAir first calls were 4.7 to 17.0 seconds. Classify these as three different
events:

- kernel compilation;
- host/package construction;
- TorchAir graph compile or cache load.

Do not optimize or report them under one “compile time” number.

## 19. AICPU research boundary

A whole B1 GQA attention kernel is technically feasible on AI CPU, but it is
not competitive. The independent direct `.aicpu` implementation used FP16
storage, FP32 accumulation, one AI CPU task, and the same 16Q/2KV/D128 contract.
It matched stock and an independent CPU reference.

Measured on a physical Ascend 910B2:

| KV | AICPU | Stock IncreFA | Slowdown |
| ---: | ---: | ---: | ---: |
| 128 | 1.541 ms | 51.486 us | 29.9x |
| 512 | 6.889 ms | 51.915 us | 132.7x |
| 2048 | 28.535 ms | 51.596 us | 553.1x |

The KV2048 profile reported about 28.76 ms per AI CPU task, while host
`LaunchKernelV2` averaged 7.28 us. Device execution, not dispatch, dominates.
The achieved rate was below 0.7 GFLOP/s and the minimum tensor-byte rate below
0.1 GB/s. Do not move QK, softmax, or AV to AI CPU.

The more detailed scaling evidence is:

| KV | Achieved GMAC/s | Achieved GFLOP/s | Minimum tensor-byte rate |
| ---: | ---: | ---: | ---: |
| 128 | 0.340 | 0.680 | 0.090 GB/s |
| 512 | 0.304 | 0.609 | 0.077 GB/s |
| 2048 | 0.294 | 0.588 | 0.074 GB/s |

At KV2048, process memory moved from 108,284 KiB to 116,404 KiB during the
profile. Direct input and output allocation was 2,107,392 bytes and score
workspace was 8 KiB. The exported run-level counters sampled 26.983 MB/s HBM
read, 14.768 MB/s HBM write, and 39,509.63 MB/s LLC read at a 75.926% hit rate.
Treat these as profiler samples for the whole run, not per-operator theoretical
bandwidth. The direct `.aicpu` launch did not emit `aicpu_*.csv`, so no
compute/memcpy/framework split is claimed.

The realistic AI CPU lane is small branch-heavy metadata, state, scheduler, or
tiling work, and only when it removes a host synchronization or overlaps on a
separate stream. A classic GE AICPU package still needs separate integration
work: its custom scheduler saw an empty `custSoPath`/`LD_LIBRARY_PATH`, while
the direct `.aicpu` route bypassed that lookup and proved device execution.

Huawei describes AI CPU as device-side Arm64 for non-matrix and complex-branch
work. See [AI CPU programming](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00049.html),
[GE parallel streams](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0142.html),
and [profiling fields](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0071.html).

## 20. Test grouped-core ideas with a topology control first

The B1 GQA kernel repeats each KV head across eight query heads. This suggests a
reasonable optimization: assign one AIV block to each KV group, load K/V once,
and process all eight query heads locally. Test the parallelism cost before
writing the much larger shared-UB kernel.

The retained `grouped_serial_control` and `grouped_half_control` packages do
this structurally:

- a separate vendor and cache namespace keep it independent from production;
- the public operator contract and supported `vector_core_count=16` resource
  attribute stay unchanged;
- grouped host tiling sets either one eight-head work item per KV group
  (`Block Num=2`) or two four-head work items per KV group (`Block Num=4`);
- FlashDecode is disabled only for this package so long KV lengths do not split
  back into query-head work items;
- each block runs its assigned eight or four query heads serially through the
  existing math and load functions.

The last point matters. This control measures the compute-parallelism penalty,
but still reloads K/V per query head. Do not call it a shared-K/V kernel.

On physical Ascend 910B2 NPU6, all eager rows from KV128 through KV2048 passed
stock FP16 tolerance and the independent CPU FP32 reference. At KV1024, three
custom-only pipe-profile calls measured:

| Package | Blocks | Task | AIV vector | AIV scalar | AIV MTE2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current query-head parallel | 16 | 21.78 us | 11.40 us | 7.98 us | 4.74 us |
| Half-group control | 4 | 54.83 us | 45.45 us | 18.10 us | 17.93 us |
| Grouped serial control | 2 | 99.96 us | 90.83 us | 31.06 us | 34.78 us |

All packages had zero AIC time/cycles and zero cube utilization. Doubling the
grouped control from two to four blocks reduced task time by 45.14% and vector
time by 49.97%. The four-block task was still 2.52 times the current 16-block
task.

The matched MemoryAccess profiles measured about the same total requested
bytes: 8,215 KiB GM-to-UB for grouped and 8,236 KiB for current, while the
unique direct Q/K/V/mask input is 1,029 KiB. The four-block control measured
8,218 KiB. All paths therefore issue about eight times the unique bytes. Prior
L2 profiling showed that most repeated requests hit cache, so requested bytes
are not the same as HBM traffic.

Use overlapping pipeline counters carefully. Vector, scalar, and MTE times
cannot be added. For the two-block grouped algorithm, deleting all removable
copy work still cannot reduce the 99.96 us task below its measured 90.83 us
vector lane.
The matched real B1/KV1024 TorchAir decoder confirmed the impact:

| Package | Mean step | Throughput |
| --- | ---: | ---: |
| Current 16-block GQA AIV | 1.3636 ms | 733.35 tok/s |
| Half-group four-block control | 1.8382 ms | 544.00 tok/s |
| Four-block copy-free ideal bound | at least 1.6692 ms | at most 599.08 tok/s |
| Grouped two-block control | 2.6582 ms | 376.20 tok/s |

Four blocks improved full-decoder throughput by 44.60% over two blocks, but
remained 25.82% below current. At the eager ACLNN boundary, the same variants
measured 173.30, 171.55, and 170.32 us for 2, 4, and 16 blocks. Fixed eager
overhead masks most of the kernel difference, so the compiled 18-layer forward
pass is the decision metric.

For the four-block control, subtracting every microsecond above its 45.45 us
vector lane from all 18 attention layers gives an optimistic ceiling of 599.08
tok/s. That remains 18.31% below current. This is deliberately more generous
than a real copy-only rewrite because pipeline counters overlap.

Across 18 attention layers, even subtracting the full 9.13 us gap between task
duration and vector time gives an optimistic copy-only upper bound of 400.98
tok/s. This remains 45.32% below current. Stop the shared-UB rewrite at this
gate unless the proposed next design also changes vector arithmetic efficiency,
not only copy count.

This method is general:

1. package a separately named or separately vendored topology control;
2. retain the supported resource attribute and prove actual `Block Num` in the
   profile;
3. pass eager correctness before performance;
4. compare vector and transfer lanes, remembering that they overlap;
5. calculate the best-case removable-work bound;
6. run the real TorchAir forward path only after the microkernel is understood;
7. implement shared-UB dataflow only if that upper bound can beat the current
   kernel.

## 21. Prove and measure a requested 32-core launch

A resource attribute is an upper bound or input to host tiling. It is not proof
that the device launched that many blocks. In the recovered GQA tiler,
requesting 32 cores at KV1024 originally still emitted the 16 query-head work
items. The upstream FlashDecode heuristic did not split GQA until KV2048.

Use a separately packaged tiling control when the experiment needs a topology
that the production heuristic cannot emit. Patch 0009 does this only for the
B1 all-vector contract when `coreNum == 2 * queryHeadWorkItems`. It enables the
existing split-K path and retains its minimum 512-token partition. At KV1024,
the mapping is 16 query heads times two disjoint 512-token sequence partitions.

The proof order is:

1. require the matching 32-core resource attribute in the isolated probe;
2. build into a separate vendor and cache namespace;
3. pass candidate-first eager parity and the independent FP32 reference;
4. inspect profiler `Block Num`, not the requested attribute;
5. prove AIC time/cycles/MAC/cube utilization remain zero;
6. measure GM-to-UB requests before assuming split-K duplicates data;
7. profile the real TorchAir forward after the microkernel gates pass.

The KV1024/valid-769 control passed eager and TorchAir correctness on physical
Ascend910B2 NPU6. The profile reported `Block Num=32`, nonzero AIV counters,
and zero AIC execution. The bounded pipe rows were:

| Package | Blocks | Task | AIV vector | AIV scalar | AIV MTE2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained query-head parallel | 16 | 21.78 us | 11.40 us | 7.98 us | 4.74 us |
| Forced two-way split-K | 32 | 22.63 us | 6.18 us | 8.56 us | 4.19 us |

The vector lane fell by 45.8%. The total profiled task did not fall because the
scalar lane, inter-core synchronization, and FlashDecode combination became
the limit. These pipeline counters overlap; do not add them.

Split-K also did not double data transfer. The paired cores own disjoint
sequence halves. The MemoryAccess profile measured 8,282 KiB GM-to-UB for the
32-block package versus 8,236 KiB for 16 blocks, an increase of 46 KiB or
0.56%. The extra traffic is query, partial-output/workspace, and reduction
traffic. It is not a second copy of all K/V data.

Keep two kinds of prefetch separate:

- **Kernel-local UB staging.** The all-vector kernel uses
  `InitBuffer(inputQue2, 2, 32_KiB)`. Huawei documents `num=2` as enabling
  double buffering. The kernel starts the V `CopyValueToUb` before vector
  softmax, so MTE2 and vector work can overlap at tile granularity.
- **Graph-level `torch_npu.npu_prefetch`.** The installed torch-npu API
  describes preloading into L2 cache. It does not place a tensor into a chosen
  AIV core's UB. Treat it as a separate L2-warming experiment.

A whole 512-token FP16 K/V partition at D128 is about 256 KiB per core: 128 KiB
for K plus 128 KiB for V. This excludes softmax, output, and workspace storage.
The kernel's two main input tiles total about 64 KiB, so it streams tiles and
cannot keep the whole partition resident in UB.

The real 18-layer B1 TorchAir sequence used physical KV1024, initial position
768, 20 warmups, and 200 measured steps per lane:

| Lane | Actual blocks | Mean step | Throughput |
| --- | ---: | ---: | ---: |
| A | 16 | 1.3509 ms | 740.23 tok/s |
| B | 32 | 1.2411 ms | 805.76 tok/s |
| C | 32 | 1.1718 ms | 853.41 tok/s |
| D | 16 | 1.2247 ms | 816.53 tok/s |
| 16-block arithmetic mean | 16 | 1.2878 ms | 778.38 tok/s |
| 32-block arithmetic mean | 32 | 1.2064 ms | 829.59 tok/s |

Both pairwise comparisons favored 32 blocks. The lane means show 6.32% lower
latency and 6.58% higher throughput. The process cadence also drifted enough
that the best single 853.41 tok/s row is not a baseline. Retain the ABBA order,
all four distributions, physical NPU, cache state, and first-call times.

This control changes the next optimization question. The vector arithmetic now
scales, transfer does not double, and full B1 latency improves. The next target
is the partial-softmax/output combine and its `SyncAll`, followed by repeated
same-device 16/32 sequences. Do not expect an external L2 prefetch call to
remove a UB-local reduction or synchronization limit.

## 22. Treat a pipeline counter as active work, not task latency

The forced 32-block GQA control is the clearest example. In a matched physical
NPU6 real-decoder profile, three steps produced 54 attention tasks per lane:

| KV1024 lane | Blocks | Task | AIV total | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 16 | 16.21 us | 14.85 us | 11.25 us | 4.65 us | 5.31 us | 0.12 us |
| Split-K | 32 | 13.38 us | 12.20 us | 6.03 us | 4.87 us | 3.25 us | 0.38 us |

The vector field falls by 46.36%, but task duration falls by 17.46%. Pipeline
counters overlap and describe active issue time on one pipeline. They do not
include every dependency or wait, and they must not be added to estimate task
duration.

Read the source around the measured path. The all-vector split-K kernel does
all of the following after its main vector math:

1. writes partial output and log-sum-exp state to GM workspace;
2. executes a global `SyncAll()` across every launched block;
3. returns the non-reducer blocks only after that barrier;
4. loads every partial on the reducer blocks;
5. recomputes stable log-sum-exp weights with max, subtract, exponential, add,
   and logarithm operations separated by vector barriers;
6. scales and reduces the D128 partial outputs, casts, and copies the result.

This is why “vector is 6 us” does not mean “the kernel should take 6 us.” The
13.38 us task also includes scalar/control work, MTE transfers, pipeline
dependencies, global synchronization and imbalance, the reduction, and about
1.18 us between reported AIV time and task completion.

Use an equal-work control when possible. One 16-block KV512 task and one
32-block KV1024 task each give every worker 512 KV tokens. The bounded direct
profiles measured 19.38 us and 22.63 us respectively. Their vector lanes were
nearly identical at 6.25 and 6.18 us; the extra 3.25 us is the cost of more
blocks plus split workspace/synchronization/combine in that direct context.
Inside the hot compiled graph, better scheduling/cache context lets the
32-block task beat the 16-block KV1024 task, but the combine still consumes
about half of the potential vector saving.

For the next control, prefer a separately named pairwise producer/reducer
package. On supported A2 products, Huawei documents `IBSet`/`IBWait` for a core
to signal and wait on a specific inter-core dependency. That may avoid waiting
for all 32 blocks when each query head only depends on its paired producer.
Treat it as a hypothesis until correctness, deadlock safety, actual topology,
and task timing are measured.

Retained evidence: [lower-KV and bottleneck experiment](../../tmp/09_persistent_page_engine/gqa_split_k32_lower_kv_eeac988/README.md).

## 23. Do not force sub-minimum split-K partitions

The host tiler's partition floor is a safety contract, not only a performance
heuristic. The 48-block control lowered the floor and forced three KV1024
partitions of about 342/341/341 tokens. Its first device task did not complete;
the runtime reported a three-minute synchronization timeout.

The same package completed correctly at KV1536, where every partition was 512
tokens, and the profiler proved 48 AIV blocks with zero AIC execution. It was
still a loss: direct task time increased from 24.79 us at 32 blocks to 33.68 us
at 48 blocks, while the real B1 ABBA throughput changed by only +0.09%.

Fail closed after such a result. The graph wrapper and direct probe now reject
48 cores below KV1536. Do not leave a convenient command that can relaunch a
known-stalling topology.

Retained evidence: [forced 48-core experiment](../../tmp/09_persistent_page_engine/gqa_split_k48_eeac988/README.md).

## 24. Gate direct-kernel wins in the compiled graph

The split-K32 reduction experiments show why an eager task improvement is not
enough. Three separately named controls were measured at B1/KV1024:

| Control | Direct task | Vector | Scalar | MTE2 | Full-graph task |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 22.63 us | 6.18 us | 8.56 us | 4.19 us | 13.380 us |
| Pairwise signal | 26.14 us | 6.24 us | 7.68 us | 5.01 us | gated out |
| Two-way algebra | 22.51 us | 6.13 us | 6.11 us | 4.99 us | gated out |
| Local partial in UB | 20.90 us | 6.11 us | 6.84 us | 3.27 us | 13.319 us |

The local-partial control is a valid direct optimization. The even worker of
each pair is also the correct query-head reducer. It retains partition 0 in
`bmm2ResUb`, scales it in place, and loads only partition 1. This removes one
512-byte GM reload and the generic two-row accumulation path per query head.
The direct task fell by 7.64%.

The real graph changed by only 0.061 us per attention call. Across 18 layers,
that is about 1.10 us per token. The static graph has better scheduling and
cache locality than the isolated ACLNN boundary, so the workspace traffic is
mostly hidden. The warm-cache full B1 sequence did not improve: current
averaged 854.81 tok/s and the control averaged 843.82 tok/s.

Apply this gate to future micro-optimizations:

1. pass candidate-first eager parity and independent reference checks;
2. prove actual blocks and zero AIC execution;
3. compare direct pipeline counters;
4. profile the same operator inside the compiled production graph;
5. multiply the compiled per-call delta by the real layer count;
6. run a reverse-order warm-cache B1 sequence;
7. promote only if the real graph and B1 distribution agree.

Do not multiply the eager delta by layer count when a compiled per-call result
exists. Here that would predict about 31 us per token, while the compiled
profile showed only about 1 us.

The pairwise synchronization control adds another rule. `IBSet`/`IBWait` flags
live in GM workspace, and arbitrary workspace is not guaranteed to start at
zero. A safe per-call initialization required a balanced `SyncAll()` before
compute, so the pairwise control moved the global rendezvous instead of
removing it and became 15.5% slower. Do not omit initialization unless an
external owner provides a race-free generation protocol.

Retained evidence: [split-K32 reduction controls](../../tmp/09_persistent_page_engine/gqa_split_k32_reduction_4cb4678/README.md).

## 25. Repository references

- [Current separate MHA AIV operator](paddle_mha_increfa_aiv/README.md)
- [Reproducible package builder](paddle_mha_increfa_aiv/build.sh)
- [Graph-facing PyTorch op and TorchAir converter](../paddleocr_vl/model/mha_increfa_aiv.py)
- [Direct-eager comparison](../scripts/probes/compare_paddle_mha_increfa_aiv_eager.py)
- [TorchAir comparison](../scripts/probes/compare_paddle_mha_increfa_aiv.py)
- [AIV hard-sync result](../../tmp/09_persistent_page_engine/increfa_aiv_only_hardsync_881d7d3/README.md)
- [Real B1 MHA-cache result](../../tmp/09_persistent_page_engine/increfa_real_forward_b1_mha_cache_5ca3482/README.md)
- [Backend-control result](../../tmp/09_persistent_page_engine/paddle_mha_increfa_aiv_backend_controls_485d8fc/README.md)
- [Current separate GQA AIV operator](paddle_gqa_increfa_aiv/README.md)
- [GQA graph op and TorchAir converter](../paddleocr_vl/model/gqa_increfa_aiv.py)
- [GQA eager/TorchAir comparison probe](../scripts/probes/compare_paddle_gqa_increfa_aiv.py)
- [GQA AIV B1 retained evidence](../../tmp/09_persistent_page_engine/gqa_aiv_b1_1d16f33/README.md)
- [Two-AIV-block GQA experiment](../../tmp/09_persistent_page_engine/gqa_grouped_two_block_994dc8f/README.md)
- [Four-AIV-block GQA experiment](../../tmp/09_persistent_page_engine/gqa_grouped_four_block_ca152b5/README.md)
- [Forced 32-AIV-block split-K experiment](../../tmp/09_persistent_page_engine/gqa_split_k32_8a1041f/README.md)
- [Lower-KV and split-K bottleneck experiment](../../tmp/09_persistent_page_engine/gqa_split_k32_lower_kv_eeac988/README.md)
- [Forced 48-AIV-block split-K experiment](../../tmp/09_persistent_page_engine/gqa_split_k48_eeac988/README.md)
- [Split-K32 reduction controls](../../tmp/09_persistent_page_engine/gqa_split_k32_reduction_4cb4678/README.md)

## 26. Official references

Verified while writing this handbook on 2026-08-09. Recheck them when changing
CANN or torch-npu versions.

- [Huawei AscendC framework-launch sample](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/README.md)
- [Huawei C++ extension eager and compile sample](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation/README.md)
- [Huawei IncreFlashAttention design](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/common/IFA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)
- [AscendC kernel task types](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0218.html)
- [AscendC `SyncAll`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0204.html)
- [AscendC `IBSet`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0202.html)
- [AscendC `CalcTschBlockDim`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_1033.html)
- [AscendC `InitBuffer` and double buffering](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0110.html)
- [AscendC `TQue` buffer limits](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0137.html)
- [AscendC `msobjdump`](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0103.html)
- [Open `ops-transformer` IncreFlashAttention V4 API](https://gitcode.com/cann/ops-transformer/blob/master/attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)

## 27. Start a full B1 decoder mega-kernel from the measured graph

The next optimization target is one complete PaddleOCR-VL text-decoder token
step. Freeze its contract before changing the implementation:

- B1 and one query token;
- hidden size 1024, intermediate size 3072, 18 decoder layers;
- 16 query heads, 2 KV heads, and head dimension 128;
- FP16 activations and weights, with linear weights retained in FRACTAL_NZ;
- static KV capacity 1024 for the first performance target;
- token embedding, all decoder layers, in-place KV updates, final RMSNorm, and
  the 103424-entry LM head inside the model boundary;
- greedy token selection in the measured serving boundary.

The retained connected-kernel profile on physical Ascend 910B2 measured
1.2381 ms per device step, or 807.7 token steps/s. MatMul used 764.21 us across
91 calls and IncreFlashAttention used 322.94 us across 18 calls in the
profiled run. Together they accounted for 77.6% of profiled device work. Host
contribution was only 9.4 us. Therefore, a mega-kernel must preserve Cube
MatMul and FRACTAL_NZ. Reimplementing the decoder as an all-vector kernel is not
a viable path.

Retained evidence: [full B1 decoder profile](../../tmp/09_persistent_page_engine/text_decode_lab/full_head_b1_profile_910b_8a04e95/full_head_b1_profile_report.md).

Use two implementation controls:

1. **TorchAir SuperKernel binary fusion.** Mark the exact full-decoder scope,
   require `strict-scope-check=abort`, and prove one scheduled task with
   profiling. This is the shortest production-shaped route because it keeps
   the already-tuned MatMul, normalization, rotary, scatter, and attention
   subkernels while removing graph task launches and operator-header gaps.
2. **Source-level mixed AIC/AIV kernel.** Keep this as the fallback for an
   operator that cannot participate in binary fusion or for an internal GM
   boundary that still dominates. AscendC supports at most four registered
   MatMul objects in one program. A full source decoder must therefore reuse a
   MatMul object with new static tiling, or fuse at a coarser layer/FFN level;
   it cannot register one object for every linear projection.

The first strict SuperKernel probes established a useful failure ladder:

- `Range` was the first unsupported TBE/TIK node. Hoisting the fixed
  `[0, cache_length)` KV positions into persistent stage state removed it from
  the token graph.
- `GatherV2` was next. It is the one-token vocabulary embedding lookup.
  CANN 9.0 includes an AscendC `GatherV3` source and a 910B config entry, but
  the installed TorchAir/GE operator store exposes neither a generated
  AscendIR wrapper nor a usable FP16 kernel registration. The graph therefore
  gives the lookup an explicit PyTorch identity and lowers it to the independent
  B1/S1/H1024 `PaddleDecodeTokenEmbedding` AscendC operator.
- Stock `MatMul` was the next boundary. The installed implementation selected
  for that GE identity is TBE/TIK, so strict SuperKernel compilation rejects
  it even when its input weight is FRACTAL_NZ. CANN 9.0 also ships the separate
  AscendC `MatMulV3` implementation. An explicit `decode_linear_matmul_v3`
  PyTorch identity can lower a no-bias B1 Linear to `MatMulV3` with
  `transpose_x2=true`. The isolated `[1,1024] x [2560,1024]` test passed strict
  SuperKernel checking with verified weight format code 29 and was bit-exact
  against stock Linear. Reuse this identity for the 91 decoder linears instead
  of copying or simplifying CANN's tuned Cube implementation.

These failures are compatibility inventory, not reasons to relax the scope.
Do not change strict checking to `bypass`: that can make a run appear successful
while silently leaving several tasks outside the requested mega-kernel.

The installed CANN 9.0 SuperKernel compiler also explains the prefetch model.
Its defaults enable early-start v2 and per-function code preloading. The
compiler chooses a mixed AIC/AIV launch from the maximum resource demand of its
subkernels. `feed-sync-all=1` is the compiler's automatic mechanism for a
subkernel whose `SyncAll` participant count differs from the enclosing launch,
but it is not proven safe for this decoder. The generated wrapper faulted in
both split-mode controls before the custom attention body. The current decoder
therefore uses `feed-sync-all=0` and an explicit fixed-16 software barrier
inside the fused cache-update/attention subkernel. Re-test automatic feeding
only as a separately named compiler control.

The acceptance gate is stronger than a successful compile:

1. strict scope compilation succeeds;
2. multi-step cache mutation and logits match the connected-kernel reference;
3. real greedy generation matches;
4. the profiler shows one full-decoder scheduled task and the expected mixed
   AIC/AIV work;
5. clean B1/KV1024 TorchAir throughput is not below the 700--800 token/s
   connected-kernel target;
6. the result survives reverse-order, same-device comparisons.

Relevant official references:

- [AscendC SuperKernel development](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00029.html)
- [TorchAir in-graph SuperKernel scope and strict checking](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00050.html)
- [AscendC `SetNextTaskStart`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00087.html)
- [AscendC `REGIST_MATMUL_OBJ`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0628.html)
- [AscendC Matmul tiling workflow](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0671.html)
- [Official multi-core Matmul sample](https://gitee.com/ascend/samples/blob/fe15fa852f308350496ea8447be08c839fb09f4f/operator/ascendc/0_introduction/10_matmul_frameworklaunch/README.md)

When adding a custom OPP to an existing TorchAir command, preserve CANN's
Python module paths. Prepend the repo package instead of replacing
`PYTHONPATH`:

```sh
export PYTHONPATH="$PWD/09_persistent_page_engine:${PYTHONPATH}"
```

Replacing `PYTHONPATH` with only the repo directory makes GE's custom-TBE
store fail during initialization with `ModuleNotFoundError: No module named
'tbe'`.

## 28. Debug the composed SuperKernel, not only each custom operator

The first complete strict decoder graph reached one binary-fused SuperKernel:
202 scheduled subkernels, 188 flattened inputs, 38 outputs, 18 decoder layers,
and the full 103424-entry LM head. This is the intended mega-kernel shape. It
retains Cube MatMulV3 and vector subkernels instead of rewriting the model as
one all-vector program.

The independent components passed progressively stronger boundaries on
physical Ascend 910B2:

- fused cache update, mask update, and 16-AIV GQA matched stock IncreFA with
  0.0 maximum absolute attention error at positions 128 and 129;
- K, V, and mask mutation were bit-exact on both consecutive calls;
- the same fused operator followed by a FRACTAL_NZ MatMulV3 inside one strict
  `early-start=1` scope was also bit-exact, including the projection output.

The complete 202-subkernel graph still produced an AIV UB-out-of-bounds fault.
Do not attribute such a fault from the outer task name or `tslot` alone. The
runtime reports a device PC relative to the SuperKernel entry. Add that delta
to the entry symbol's object address and inspect sorted symbols in the dumped
`te_superkernel_*_host.o`. This mapped two differently linked failures to the
same custom fused-GQA function offset, `+0x600`, even though one outer report
showed `tslot=3` and another showed `tslot=7`.

That exact mapping first suggested a composition-specific lifecycle problem.
The fused entry constructed one `TPipe` for cache/mask preparation and software
sync, destroyed it, and then entered the stock attention dispatcher, which
constructed another `TPipe`. Installed CANN 9.0 source shows two relevant
rules:

1. only one global `TPipe` can be active at a time;
2. SuperKernel compilation suppresses the final `PIPE_ALL` barrier normally
   emitted by `TPipe::Destroy()`.

An extra barrier before `Destroy()` did not fix the full decoder. The stronger
single-pipe control then used one `TPipe` object for the entire fused subkernel,
executed an explicit all-pipe barrier after software `SyncAll`, called the
supported `TPipe::Reset()` to release phase-one UB/event resources, and passed
the same object into the unchanged stock all-vector attention implementation.
That build passed both the isolated fused operator and the strict
GQA-to-MatMul boundary with 0.0 maximum absolute error, but the complete
202-subkernel decoder still faulted at the identical fused-function offset,
`+0x600`. Reusing one pipe is therefore valid hygiene, but it is not the cause
of the full-graph failure.

This result changes the next debugging question. Do not add more lifecycle
barriers blindly. CANN documents that the SuperKernel compiler inserts
inter-operator synchronization by default. Instead, use separately named
discriminator operators or smaller real-model scopes to decide among these
remaining causes:

1. the all-core software-sync prologue behaves differently in the enclosing
   mixed-core launch;
2. the stock attention body receives a bad argument, tiling pointer, workspace,
   or launch geometry only in the large flattened scope;
3. repeated in-place reference outputs or 18 repeated operator instances expose
   a SuperKernel ABI/composition limit.

A useful first discriminator keeps core-0 cache/mask preparation and the
16-core software barrier but replaces attention with a deterministic output.
If that separately named sync-only operator faults at the same PC, the prologue
or enclosing launch is responsible. If it runs, keep the prologue fixed and
isolate the attention ABI. A one-layer real-model strict scope is the second
control: success at one layer and failure only after repetition points away
from the attention math and toward flattened argument/tiling composition.

The one-layer control failed before a sync-only package was necessary. Its
runtime error log reported `blockDim=24`, while the fused GQA tiling and
software barrier use exactly 16 AIV workers. The failing PC mapped to `+0x600`
inside the split-specific GQA function, exactly as in the 18-layer decoder.
The earlier isolated and GQA-to-MatMul controls did not expose this mismatch
because their enclosing scopes used no more than the attention worker count.

This is a critical SuperKernel rule: every fused subfunction sees the enclosing
launch geometry. A custom subfunction specialized for fewer workers must reject
idle `GetBlockIdx()` values before it creates a `TPipe`, allocates UB, calls a
fixed-participant `SyncAll`, or indexes tiling data. Returning from the
subfunction still returns control to the generated SuperKernel wrapper and its
inter-operator handoff. The fused GQA therefore guards
`GetBlockIdx() >= 16` before all local state. Huawei's `SyncAll` contract also
states that `usedCores` cannot exceed the operator's logical launch dimension;
mixed AIC/AIV launches can expose a larger AIV index range than an AIV-only
launch.

Retained one-layer evidence:

- `tmp/09_persistent_page_engine/text_decode_lab/megakernel_fused_gqa_onepipe_20bc8e0_depth1_ed099ad_npu1/run.log`
- `/root/ascend/log/debug/plog/plog-2703216_20260810095829892.log`
- `extra-info/data-dump/0/te_superkernel_2d66804868f483fa2d4dbd7c8bd50ec1a577c4e1e145e199f61343a433f99c1a_host.o`

The test ran on physical Ascend 910B2 NPU 1 after eight exact full-cache-shape
preflight iterations. Quarantine NPU 1 after the resulting device fault.

Retained remote evidence for the pre-reset build:

- `.runtime_cache/paddle_decode_kv_gqa_aiv/validation/tpipe_2f9c6f1_strict_npu6/result.json`
- `.runtime_cache/paddle_decode_kv_gqa_aiv/validation/gqa_matmul_boundary_tpipe_2f9c6f1_early1_npu6/result.json`
- `tmp/09_persistent_page_engine/text_decode_lab/megakernel_fused_gqa_tpipe_2f9c6f1_early0_npu6/run.log`

Retained remote evidence for the single-pipe control (`20bc8e0`):

- `.runtime_cache/paddle_decode_kv_gqa_aiv/validation/onepipe_20bc8e0_strict_npu0/result.json`
- `.runtime_cache/paddle_decode_kv_gqa_aiv/validation/gqa_matmul_boundary_onepipe_20bc8e0_early1_npu0/result.json`
- `tmp/09_persistent_page_engine/text_decode_lab/megakernel_fused_gqa_onepipe_20bc8e0_early0_npu0/run.log`

The single-pipe package SHA256 is
`99766195710b2cb3d988debeab60cf438291b45a8b3cad73dd074ac07fd56cfe`.
The compiled kernel object SHA256 is
`41184b6b69df84df768351b6ec3d4be3a56718173c59aa93bc5e5f0357e8519f`.
The failing full graph ran on physical Ascend 910B2 NPU 0 after the full-shape
device preflight passed. It then poisoned that device for later work, so NPU 0
was quarantined.

### Device and compile preflight

A device fault can poison later work on one physical NPU while `npu-smi` still
prints `Health=OK`. Before loading an experimental full SuperKernel, run
repeated full B1 cache-shape CPU-to-NPU transfers, an NPU vector operation, and
an exact NPU-to-CPU comparison. Quarantine that physical device after a runtime
fault; do not trust a tiny allocation or the health column as a recovery test.

CANN helper processes invoke the literal command `python3`. On this container,
`/usr/bin/python3` is Python 3.10 without the required package set, while the
operator build uses Python 3.12 and NumPy 1.26. A repo-runtime-cache shim avoids
the misleading helper traceback without changing the system interpreter:

```sh
mkdir -p .runtime_cache/python312_path/bin
ln -s /usr/local/python3.12.13/bin/python3 \
  .runtime_cache/python312_path/bin/python3
export PATH="$PWD/.runtime_cache/python312_path/bin:$PATH"
```

Do not expose Python 3.12 site-packages to `/usr/bin/python3`; the NumPy ABI is
incompatible. Also distinguish the approximately ten-second AscendC device
compile from the current wrapper's multi-minute rebuild of protobuf, Abseil,
and host plugins.

### SSH workflow on the blue-zone gateway

OpenSSH ControlMaster sockets closed immediately on the current gateway with
`Connection closed by UNKNOWN port 65535`. Use the reliable direct form and
amortize its roughly seven-second handshake by running one retained remote job
per experiment, then polling its log and `rc.txt`:

```sh
ssh -S none -o ControlMaster=no -o ConnectTimeout=20 \
  blue_zone_npu_container '<one bounded experiment>'
```

Do not launch `npu-setup` concurrently: each invocation selects a currently
free device, so parallel setup can collide before either process allocates NPU
memory.
