# PaddleDecodeKvScatterQueryV4

This is an independent, shape-specialized AscendC operator for the PaddleOCR-VL
B1 decoder on Ascend 910B2. It writes one FP16 K/V state into persistent BNSD
caches `[1,2,1024,128]`, copies the FP16 query `[1,16,1,128]`, and emits the
bool future mask `[1,1,1,1024]`. The following attention operator consumes the
query and mask outputs, which creates an explicit graph dependency after the
cache write. V4 preserves the downstream GQA operator's dynamic K/V list ABI
and supplies required tensor aliases only for TorchAir's mutable reference
outputs. This avoids a full-cache copy without changing the attention kernel's
argument layout. Folding the mask here removes the unfusible GE `Greater`
boundary from the strict decoder SuperKernel.

The operator launches one AIV block. It is intentionally not a generic scatter.
Build it only on the NPU container after `source npu-setup`:

```sh
bash 09_persistent_page_engine/custom_ops/paddle_decode_kv_scatter_query/build.sh
```
