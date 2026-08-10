# PaddleDecodeKvScatterQueryV2

This is an independent, shape-specialized AscendC operator for the PaddleOCR-VL
B1 decoder on Ascend 910B2. It writes one FP16 K/V state into persistent BNSD
caches `[1,2,1024,128]` and copies the FP16 query `[1,16,1,128]` to its output.
The following attention operator consumes that output, which creates an explicit
graph dependency after the cache write. V2 exposes named K/V ref outputs so
TorchAir auto-functionalization can preserve the persistent mutation without a
full-cache copy.

The operator launches one AIV block. It is intentionally not a generic scatter.
Build it only on the NPU container after `source npu-setup`:

```sh
bash 09_persistent_page_engine/custom_ops/paddle_decode_kv_scatter_query/build.sh
```
