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
| 3 | Is the intended kernel binary present? | one key, metadata, ELF symbols, package hashes | cube symbol or unexpected key remains |
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
graph-level cadence or hard-sync scheduling, not more vector cores.

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

The realistic AI CPU lane is small branch-heavy metadata, state, scheduler, or
tiling work, and only when it removes a host synchronization or overlaps on a
separate stream. A classic GE AICPU package still needs separate integration
work: its custom scheduler saw an empty `custSoPath`/`LD_LIBRARY_PATH`, while
the direct `.aicpu` route bypassed that lookup and proved device execution.

Huawei describes AI CPU as device-side Arm64 for non-matrix and complex-branch
work. See [AI CPU programming](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00049.html),
[GE parallel streams](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0142.html),
and [profiling fields](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0071.html).

## 20. Repository references

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

## 21. Official references

Verified while writing this handbook on 2026-08-09. Recheck them when changing
CANN or torch-npu versions.

- [Huawei AscendC framework-launch sample](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/README.md)
- [Huawei C++ extension eager and compile sample](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation/README.md)
- [Huawei IncreFlashAttention design](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/common/IFA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)
- [AscendC kernel task types](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0218.html)
- [AscendC `SyncAll`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0204.html)
- [AscendC `CalcTschBlockDim`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_1033.html)
- [AscendC `msobjdump`](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0103.html)
- [Open `ops-transformer` IncreFlashAttention V4 API](https://gitcode.com/cann/ops-transformer/blob/master/attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)
