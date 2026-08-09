# Separate Paddle MHA IncreFA AIV operator

This is a separately named Ascend operator. It does not override or replace
CANN's `IncreFlashAttention` registration.

The identity is explicit at every layer:

```text
PyTorch: paddleocr_vl::mha_incre_flash_attention_aiv
GE/CANN: PaddleMhaIncreFlashAttentionAiv
kernel:  paddle_mha_incre_flash_attention_aiv
vendor:  paddle_mha_increfa_aiv
```

The eager PyTorch custom-op body calls stock
`torch_npu.npu_incre_flash_attention` as the correctness reference. During
TorchAir compilation, its converter emits only
`PaddleMhaIncreFlashAttentionAiv`. The stock PyTorch op still emits stock
`IncreFlashAttention`; there is no same-name package selection or fallback.

## Deliberately narrow contract

- Ascend 910B;
- batch size 1;
- FP16 Q/K/V and output;
- BNSD layout;
- one query token;
- equal Q and KV head counts (MHA, not GQA);
- bool attention mask;
- no actual-sequence-length tensor, quantization, paged attention, or KV
  padding;
- `inner_precise=1`;
- fixed upstream all-vector tiling key `11000000000100000`.

The public Python wrapper rejects inputs outside that contract. The CANN
package registers only FP16 and only Ascend 910B. Production PaddleOCR-VL uses
16 query heads and two KV heads, so it cannot enter this operator without an
explicit MHA cache-expansion experiment.

## Source provenance

The build uses the recovered official `ops-transformer` v9.0.0 source at commit
`afe72144f9f2ac8441929035795db88a111b30c5`. It creates a fingerprinted,
detached build worktree under `.runtime_cache/`, renames the upstream source
directory to the new operator, disables the stock registration surfaces in
that build source, overlays the distinct host/kernel entries, and compiles one
tiling key.

The recovered source checkout remains unchanged. The fingerprinted build tree
and its CMake output persist so an identical rebuild is incremental instead of
recompiling from scratch.

The reused all-vector implementation still needs AIV-to-AIV `SyncAll()`. The
kernel therefore uses CANN's hard-sync `MIX_AIV_1_0` envelope. Package gates
require a `0:1` task ratio, the `_mix_aiv` function, no `_mix_aic` function,
inter-core synchronization metadata, and the one expected tiling key.

## Build and isolated PyTorch comparison

Run on Blue Zone after pulling the local commit:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
bash 09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv/build.sh
```

The runner prints `PADDLE_MHA_INCREFA_AIV_SET_ENV`. Source that exact file, then
run the comparison through PyTorch/TorchAir:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
PYTHONPATH=09_persistent_page_engine \
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/probes/compare_paddle_mha_increfa_aiv.py \
  --output .runtime_cache/paddle_mha_increfa_aiv/compare.json \
  --cache-root .runtime_cache/paddle_mha_increfa_aiv/torchair_compare
```

The probe compiles stock and custom operators under different graph-cache
directories in the same process, compares full outputs at KV128/KV512/KV2048,
and reports non-profiled repeated timing. First-call compile/cache-load time is
recorded separately from steady operator timing.

After parity, inspect the custom `.om`/kernel metadata and run the B1 real
forward lane. Do not treat a package build or isolated exact match as an E2E
latency result.
