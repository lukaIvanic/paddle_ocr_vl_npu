# Paddle B1 decode QKV slices V1

This package replaces the TBE/TIK `SplitV` plus packed-QKV layout chain with
three independent single-output AIV-only AscendC operators. Each accepts the
same FP16 `[1,1,2560]` tensor and emits one contiguous B1 view: Q
`[1,16,1,128]`, K `[1,2,1,128]`, or V `[1,2,1,128]`. Separate operators avoid
the multi-output custom-kernel ABI that executes normally but faults when
binary-fused into a strict SuperKernel on this CANN 9.0 setup.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_qkv_slices_v1/bin/set_env.bash` before TorchAir
initializes GE.
