# Qwen3-MoE Ascend operator notes

These notes map the installed `torch_npu 2.10.0` A2/910B operator contracts to
the Qwen3-30B-A3B BF16 top-8 expert block. They do not make a 310P support claim.

## Current compiled baseline

The static TorchAir graph currently preserves the correctness-first path:

1. FP32 router softmax and top 8;
2. `index_select` complete gate/up and down weights for the selected experts;
3. two small BMM calls;
4. SiLU, multiply, weighting, and reduction as separate operations.

This path fullgraph-compiles without a graph break. The clean second-half run
reaches 153.43 tokens/s, while the complete two-process pipeline reaches 86.07
tokens/s.
It still materializes selected expert weights and therefore performs avoidable
HBM reads and writes.

## Recommended BF16 grouped-matmul lane

The next lane should keep routing on device and use this sequence:

1. Router linear in BF16.
2. `npu_moe_gating_top_k_softmax` on FP32 router logits. It returns selected
   probabilities, `int32` expert IDs, and row IDs. Renormalize the selected
   probabilities afterward because this model has `norm_topk_prob=true`.
3. `npu_moe_init_routing` to replicate and sort the B1 token for its eight
   selected experts.
4. `npu_moe_compute_expert_tokens` to produce the device-side expert group
   boundaries.
5. BF16 `npu_grouped_matmul` with persistent gate/up weights stored as
   `[128, 2048, 1536]`.
6. Split gate/up, then run SiLU and multiply.
7. A second BF16 `npu_grouped_matmul` with down weights stored as
   `[128, 768, 2048]`.
8. `npu_moe_finalize_routing` to apply routing weights and combine the eight
   expert outputs.

`npu_grouped_matmul` supports graph mode, BF16 A2 inputs, 3D expert weights, and
a device Tensor `group_list`. This avoids copying eight complete expert matrices
before each multiplication. The checkpoint loader should write the persistent
weights directly in the operator's `[expert, K, N]` layout; transposing them on
every token would defeat the optimization.

## Operators that do not directly fit

`npu_ffn` looks attractive because it fuses two matrix multiplications and an
activation. Its installed contract does not support this exact lane:

- expert-grouped BF16 supports ordinary `silu`, where the first projection has
  one activation output per intermediate channel;
- `swiglu`, `geglu`, and `reglu` are restricted to the ungrouped FP16
  high-performance case;
- Qwen3-30B-A3B uses grouped BF16 SwiGLU with separate gate and up values.

Therefore `npu_ffn(..., activation="swiglu", expert_tokens=...)` is not a valid
drop-in replacement.

The installed `npu_grouped_matmul_finalize_routing` contract is an int8/int4
operator. It is relevant to a later quantized lane, not the current BF16 lane.

## Source contracts

- [Huawei `torch_npu` API list](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu_list.md)
- [Huawei `npu_grouped_matmul`](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_grouped_matmul.md)
- [Huawei `npu_ffn`](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_ffn.md)
- [Huawei `npu_moe_init_routing`](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000169.html)
- [Huawei `npu_moe_compute_expert_tokens`](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_compute_expert_tokens.md)
- [Huawei `npu_moe_finalize_routing`](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_finalize_routing.md)
