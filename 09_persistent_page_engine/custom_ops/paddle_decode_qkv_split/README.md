# Paddle B1 decode QKV split V2

This independent AIV-only AscendC operator replaces the TBE/TIK `SplitV`
plus the packed-QKV layout chain in the strict Paddle decoder SuperKernel. It
accepts FP16 `[1,1,2560]` and emits Q `[1,16,1,128]`, K `[1,2,1,128]`, and V
`[1,2,1,128]`. The one-token layout makes each output a contiguous segment, so
one AIV core performs three small copies without a general transpose kernel.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_qkv_split_v2/bin/set_env.bash` before TorchAir
initializes GE.
