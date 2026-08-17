# Qwen3-MoE Ascend operator notes

These notes map the installed `torch_npu 2.10.0` A2/910B operator contracts to
the Qwen3-30B-A3B BF16 top-8 expert block. They do not make a 310P support claim.

## Verified compiled paths

The original static TorchAir graph used the correctness-first path:

1. FP32 router softmax and top 8;
2. `index_select` complete gate/up and down weights for the selected experts;
3. two small BMM calls;
4. SiLU, multiply, weighting, and reduction as separate operations.

At B1/KV4096, the complete 24-layer second half plus final RMSNorm and LM head
reached 146.50 tokens/s. It materialized selected expert weights and therefore
performed avoidable HBM reads and writes.

The verified optimized BF16 sequence is:

1. Router linear in BF16.
2. `npu_moe_gating_top_k_softmax` on router logits. It returns selected
   probabilities and `int32` expert IDs. Renormalize the selected probabilities
   afterward because this model has `norm_topk_prob=true`.
3. `npu_moe_init_routing_v2` to replicate and sort the B1 token and emit the
   per-expert group counts in one call.
4. BF16 `npu_grouped_matmul` with persistent gate/up weights stored as
   `[128, 2048, 1536]`.
5. Split gate/up, then run SiLU and multiply.
6. A second BF16 `npu_grouped_matmul` with down weights stored as
   `[128, 768, 2048]`.
7. `npu_moe_finalize_routing` to apply routing weights and combine the eight
   expert outputs.

`npu_grouped_matmul` supports graph mode, BF16 A2 inputs, 3D expert weights, and
a device Tensor `group_list`. This avoids copying eight complete expert matrices
before each multiplication. The checkpoint loader should write the persistent
weights directly in the operator's `[expert, K, N]` layout; transposing them on
every token would defeat the optimization.

With fresh per-call Q/K AddRMSNorm zero banks, this complete path reached
203.77 tokens/s at 4.908 ms TPOT over 200 tokens. It
matched all eight captured greedy tokens and used one static TorchAir graph
without recompilation.

The one-layer probe is not a performance benchmark. It exists only to catch
operator contract and numerical failures. Timing decisions use the complete
24-layer stage plus final RMSNorm and LM head.

A direct B1 sort/equality/group-count implementation was retested on that full
stage and reached 170.46 tokens/s. It was 16.35% slower than InitRoutingV2, so
the active implementation does not keep the B1-specialized route.

The installed `npu_moe_gating_top_k_softmax_v2` Python wrapper accepts
`renorm=1`, which is the direct Qwen top-k-normalization contract. Its TorchAir
GE converter rejects that attribute. The compiled path therefore uses
`npu_moe_gating_top_k_softmax` followed by an explicit selected-probability
renormalization.

`npu_add_rms_norm` mutates the add input. Static Q/K zero buffers therefore
become nonzero after one decode call and cause accumulating output error. The
valid graph allocates one Q zero bank and one K zero bank per invocation, then
unbinds one row for each layer. Two fused `Fill` kernels cost about 11 us total
and preserve the zero-input contract across decode steps.

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
