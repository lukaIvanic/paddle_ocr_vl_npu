# Separate Paddle MHA IncreFA AIV operator

For the reusable development, validation, profiling, and integration workflow,
read the [Ascend custom operator handbook](../ASCEND_CUSTOM_OPERATOR_HANDBOOK.md).

This is a separately named Ascend operator. It does not override or replace
CANN's `IncreFlashAttention` registration.

The identity is explicit at every layer:

```text
PyTorch/TorchAir: paddleocr_vl::mha_incre_flash_attention_aiv
PyTorch eager:    paddleocr_vl_npu::paddle_mha_incre_flash_attention_aiv_eager
GE/CANN:          PaddleMhaIncreFlashAttentionAiv
aclnn eager:      aclnnPaddleMhaIncreFlashAttentionAiv
kernel:           paddle_mha_incre_flash_attention_aiv
vendor:           paddle_mha_increfa_aiv
```

The eager PyTorch custom-op body calls stock
`torch_npu.npu_incre_flash_attention` as the correctness reference. During
TorchAir compilation, its converter emits only
`PaddleMhaIncreFlashAttentionAiv`. The stock PyTorch op still emits stock
`IncreFlashAttention`; there is no same-name package selection or fallback.

The separate C++ extension is the direct-eager lane. It registers a distinct
`PrivateUse1` implementation and enqueues the package's public
`aclnnPaddleMhaIncreFlashAttentionAiv` API. That wrapper forwards to the
package-generated inner API while retaining a stable, independent public name.
The extension has no TorchAir converter and no Python stock-op fallback.

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
`afe72144f9f2ac8441929035795db88a111b30c5`. It creates a detached build
worktree under `.runtime_cache/`, renames the upstream source
directory to the new operator, disables the stock registration surfaces in
that build source, renames the tiling-data and tiling-template registry keys
into the new operator namespace, overlays the distinct host/kernel entries,
and compiles one tiling key.

The recovered source checkout remains unchanged. The active build worktree
persists across overlay revisions, and new tracked patches apply incrementally.
The upstream package wrapper recreates its CMake output and recompiles bundled
host tools such as Protobuf. Treat that multi-minute package construction as a
build-stage cost, never as TorchAir first-call or steady operator latency.

The top-level tiling-data registration uses the separate GE operator name. Its
nested structure registrations retain the upstream structure names because
CANN static graph compilation resolves nested members by their C++ structure
type, such as `IncreFlashAttentionTilingDataOp`. Those names are data-schema
identities, not callable operator identities.

The reused all-vector implementation still needs AIV-to-AIV `SyncAll()`. The
kernel therefore uses CANN's hard-sync `MIX_AIV_1_0` envelope. Package gates
require a `0:1` task ratio, the `_mix_aiv` function, no `_mix_aic` function,
inter-core synchronization metadata, and the one expected tiling key.

## Build and isolated PyTorch comparisons

Run on Blue Zone after pulling the local commit:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
bash 09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv/build.sh
```

The runner prints `PADDLE_MHA_INCREFA_AIV_SET_ENV`. Source that exact file, then
build the direct-eager PyTorch extension:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
bash \
  09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv/pytorch_extension/build.sh
```

Run the direct-eager comparison first. This invokes the separate vendor op and
stock IncreFA directly from PyTorch without TorchAir:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/probes/compare_paddle_mha_increfa_aiv_eager.py \
  --output .runtime_cache/paddle_mha_increfa_aiv/compare_eager.json
```

The direct-eager probe compares full outputs at KV128/KV512/KV2048, records
first-use separately, and reports non-profiled repeated NPU-event and host-wall
timing. Its result includes the PyTorch dispatcher table and asserts that the
separate op has a `PrivateUse1` implementation.

Only after that passes, run the graph-integration comparison through
PyTorch/TorchAir:

```sh
source <PADDLE_MHA_INCREFA_AIV_SET_ENV>
PYTHONPATH=09_persistent_page_engine:$PYTHONPATH \
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/probes/compare_paddle_mha_increfa_aiv.py \
  --lanes both \
  --kv-lengths 128 \
  --output .runtime_cache/paddle_mha_increfa_aiv/compare.json \
  --cache-root .runtime_cache/paddle_mha_increfa_aiv/torchair_compare
```

The TorchAir probe compiles stock and custom operators under different graph
cache directories, compares full outputs, and reports non-profiled repeated
timing. It requires exactly one KV length per process. Launch a fresh process
and use a distinct shape cache for KV512 and KV2048. This prevents the
multi-shape `cache_compile` recompilation warnings observed during development.
First-call compile/cache-load time is recorded separately from steady operator
timing.

After parity, inspect the custom `.om`/kernel metadata and run the B1 real
forward lane. Do not treat a package build or isolated exact match as an E2E
latency result.
