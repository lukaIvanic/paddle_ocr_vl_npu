# Paddle B1 decode QKV split V4

This independent package replaces the packed-QKV `SplitV` and layout chain
with one multi-output AIV-only AscendC operator. It accepts FP16
`qkv[1,1,2560]` and emits contiguous Q `[1,16,1,128]`, K `[1,2,1,128]`, and V
`[1,2,1,128]` tensors.

V4 retains the exact V2 data path, makes all tiling fields live device inputs,
and explicitly destroys its `TPipe` before returning. The earlier three-op V3
detour is removed. Its strict-scope failure was caused by the enclosing
SuperKernel option `feed-sync-all=1`, not by the multi-output ABI. V2 is
bit-exact both normally and under strict binary fusion when the scope uses
`feed-sync-all=0`, `preload-code=none`, `early-start=0`, and `split-mode=1`.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_qkv_split_v4/bin/set_env.bash` before TorchAir
initializes GE.
