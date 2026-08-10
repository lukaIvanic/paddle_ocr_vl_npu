# Paddle B1 decode position add V1

This independent one-block AIV-only AscendC operator replaces the B1 decode
position expression `cache_position.view(1, 1) + rope_deltas` inside the strict
TorchAir SuperKernel scope. It accepts two contiguous INT64 `[1,1]` tensors and
returns their exact INT64 sum with the same shape.

The scalar inputs and output use the DCache clean-and-invalidate protocol used
by installed CANN production operators before `GlobalTensor::GetValue` and
after `GlobalTensor::SetValue`. This is required when scalar global-memory state
flows between subkernels inside one binary-fused SuperKernel task.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_position_add_v1/bin/set_env.bash` before
TorchAir initializes GE.
