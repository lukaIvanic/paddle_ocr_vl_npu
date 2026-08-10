# PaddleDecodeSwiGluV1

This independent AscendC operator implements the fixed PaddleOCR-VL B1 decoder
MLP gate `SiLU(gate) * up` for two FP16 tensors shaped `[1,1,3072]`. It uses
one AIV block and FP32 intermediate math. Its purpose is to remove CANN 9.0's
unfusible TBE `SwishMul` boundary from the strict whole-decoder SuperKernel.

Build it only on the Ascend 910B2 container after `source npu-setup`:

```sh
bash 09_persistent_page_engine/custom_ops/paddle_decode_swiglu/build.sh
```
