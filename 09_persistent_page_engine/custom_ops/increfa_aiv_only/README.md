# IncreFA FP16/MHA AIV-only launch experiment

This experiment changes only the kernel task type for tiling key
`11000000000100000`:

```text
QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING
```

The validated mixed-core control is recorded under
`tmp/09_persistent_page_engine/increfa_custom_reproduction_764e354/`.

## Why this is a narrow experiment

For the selected key, upstream dispatches
`IncreFlashAttentionAttenAllVecNew` through `INVOKE_IFA_ALL_VEC_OP_IMPL`.
The cube-compiled half of that macro returns immediately because the operation
does not request cube tiling. The generated control metadata nevertheless says:

```text
coreType=MIX
kernelType=MIX_AIC
taskRation=1:2
crossCoreSync=1
```

The patch changes the two default task-type declarations to
`KERNEL_TYPE_MIX_AIV_1_0` while the build is restricted to the single selected
tiling key. CANN maps this mode to the vector code channel and a `0:1` runtime
ratio, so no cube task launches. It also retains the hard-sync runtime contract
required by `SyncAll()`. The host tiler tells `CalcTschBlockDim` that the
all-vector mode uses zero AIC cores. The generated package contains no other key
whose task type could be affected.

The task-type extractor does not reliably resolve a preprocessor-conditioned
`KERNEL_TASK_TYPE_DEFAULT`: an earlier guarded attempt compiled successfully but
still emitted `MIX_AIC` metadata. The fixed-key build uses an unconditional
default and rejects any ratio other than zero-cube `0:1` before reporting
success. CANN 9.0 still writes the legacy per-kernel JSON label `MIX_AIC` for
this package. The runner therefore also checks the ELF symbol table: it must
contain only the `_mix_aiv` function and no `_mix_aic` cube function.

## Important boundaries

- This does not remove `SyncAll()` from the flash-decode reduction. AIV cores
  still need to synchronize with each other before the final reduction.
- Bare `KERNEL_TYPE_AIV_ONLY` was tested with both 48 blocks and one block on a
  910B2. Both failed at the same vector PC offset `0x42cc` with a UB/D-cache bus
  error. The identical instruction works in the validated mixed AIV function.
  Installed open-source kernels that combine AIV-only work with `SyncAll()` use
  a MIX envelope, `taskRation=0:1`, and `intercoreSync=1`. Therefore this
  experiment uses CANN's `MIX_AIV_1_0` hard-sync form instead of bare AIVEC.
- A metadata-only revision kept the upstream mixed block calculation. Its first
  910B smoke launched 24 blocks and failed with a vector-core UB/D-cache
  exception. CANN's `CalcTschBlockDim` contract requires `aicCoreNum=0` when a
  kernel uses no cube API. The host-side part of this patch applies that rule
  only to `BMM_ALL_BY_VEC` in this fixed-key package.
- The patch is tied to upstream source commit
  `afe72144f9f2ac8441929035795db88a111b30c5` and the pristine entry-file
  SHA-256 recorded by the runner. It stops rather than patching drifted source.
- Build only tiling key `11000000000100000`. This package is not a general
  replacement for other IncreFA dtype, layout, GQA, paged-cache, or cube paths.

## Build on Blue Zone

From the project checkout:

```sh
bash 09_persistent_page_engine/custom_ops/increfa_aiv_only/build_aiv_only_package.sh
```

To isolate multi-core launch behavior, build a separate one-AIV diagnostic
package:

```sh
AIV_LAUNCH_BLOCKS=1 \
RUN_ID=aiv_only_one_core_<commit> \
bash 09_persistent_page_engine/custom_ops/increfa_aiv_only/build_aiv_only_package.sh
```

This applies `0002-diagnostic-one-aiv-launch.patch` after the AIV-only patch and
uses vendor name `paddle_increfa_aiv_only_1core`. It is a fault-isolation build,
not a throughput candidate.

The runner sources `npu-setup`, preserves any existing source-tree build
directory, applies the tracked patch temporarily, builds one fixed-key package,
copies the package and metadata under `.runtime_cache/`, and restores the
upstream source file before exit.

Compilation is only the first gate. Do not claim success until the installed
package is proven to load, matches the mixed control at KV128/KV512/KV2048, and
passes repeated non-profiled timing.
