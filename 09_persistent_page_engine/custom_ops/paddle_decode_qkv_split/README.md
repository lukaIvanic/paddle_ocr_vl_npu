# Paddle B1 decode QKV slices V2

This package replaces the TBE/TIK `SplitV` plus packed-QKV layout chain with
three independent single-output AIV-only AscendC operators. Each accepts the
same FP16 `[1,1,2560]` tensor and emits one contiguous B1 view: Q
`[1,16,1,128]`, K `[1,2,1,128]`, or V `[1,2,1,128]`. Separate operators avoid
the multi-output custom-kernel ABI that executes normally but faults when
binary-fused into a strict SuperKernel on this CANN 9.0 setup.

V2 uses separate `VECIN` and `VECOUT` queues with a UB-to-UB copy between
them. V1 incorrectly reused one `VECIN` queue for both directions; its final
UB-to-HBM transfer could still be active when the next fused task reused the
same UB address, which corrupted the first 256 query elements.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_qkv_slices_v2/bin/set_env.bash` before TorchAir
initializes GE.
