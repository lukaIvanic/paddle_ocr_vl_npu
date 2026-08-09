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
`KERNEL_TYPE_AIV_ONLY` only when the compiler specializes the selected key.
All other tiling keys retain the upstream mixed task type.

## Important boundaries

- This does not remove `SyncAll()` from the flash-decode reduction. AIV cores
  still need to synchronize with each other before the final reduction.
- The upstream host tiler still calculates block dimensions using the physical
  AIC and AIV inventory. The first runtime smoke must therefore check for a
  launch error or synchronization hang before any timing interpretation.
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

The runner sources `npu-setup`, preserves any existing source-tree build
directory, applies the tracked patch temporarily, builds one fixed-key package,
copies the package and metadata under `.runtime_cache/`, and restores the
upstream source file before exit.

Compilation is only the first gate. Do not claim success until the installed
package is proven to load, matches the mixed control at KV128/KV512/KV2048, and
passes repeated non-profiled timing.
