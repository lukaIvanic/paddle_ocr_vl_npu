# Paddle B1 decode RoPE lookup V1

This independent one-block AIV-only AscendC operator replaces the complete
per-token MRoPE-factor preparation chain in the specialized B1 decoder. It
adds the INT64 cache position and RoPE delta in device scalar state, then
copies the selected FP16 cosine and sine rows from a persistent
`[2,1024,128]` lookup table to two `[1,1,1,128]` outputs.

Because all three Paddle MRoPE position axes are the same during one-token
decode, selecting one scalar row and returning the already-final factors is
equivalent to the former broadcast, FP32 multiply, concatenate, cosine, sine,
section split, and concatenate chain. The scalar reads use the installed-CANN
DCache protocol required inside a binary-fused SuperKernel task.

Build and install on Ascend 910B after `source npu-setup`, then source the
generated `vendors/paddle_decode_rope_lookup_v1/bin/set_env.bash` before
TorchAir initializes GE.
