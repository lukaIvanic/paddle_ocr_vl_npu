# PaddleDecodeKvPrepareFunctionalMixed24

This independent AscendC operator is the simple, alias-free K/V handoff for the
PaddleOCR-VL B1 decoder SuperKernel. It copies the two FP16 BNSD caches
`[1,2,1024,128]` in parallel across 24 AIV workers, writes one K/V state into
the fresh outputs, copies query `[1,16,1,128]`, and emits the bool future mask
`[1,1,1,1024]`. The following attention subfunction consumes explicit cache
outputs instead of mutable reference aliases.

The full-cache copy is deliberate. It creates an ordinary functional graph
dependency for the first correct one-launch milestone. Measure its bandwidth
cost before replacing it with a more fragile in-place optimization.

Build only on Ascend 910B2 after `source npu-setup`:

```sh
bash 09_persistent_page_engine/custom_ops/paddle_decode_kv_prepare_functional_mixed24/build.sh
```
