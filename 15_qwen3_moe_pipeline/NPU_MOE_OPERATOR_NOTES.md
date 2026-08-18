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

## B1 InitRoutingV3 reconstruction study

The installed `npu_moe_init_routing_v2` wrapper lowers to the CANN
`MoeInitRoutingV3` operator. At B1/top-8 it produces three values needed by the
BF16 GMM path: eight replicated hidden rows, the inverse expert-order row
indices, and 128 expert counts.

The operator was reconstructed with ordinary Torch and `torch_npu` primitives
and compiled as one static graph containing 24 routing calls. All accepted
candidates matched those three outputs exactly. The harness itself also has 48
common `GatherV2` calls because it selects two inputs for each synthetic layer,
so these figures compare candidates but are not standalone kernel latency:

| routing construction | 24-call graph | result |
| --- | ---: | --- |
| InitRoutingV3, `active_num=8` | 597.56 us | production control |
| InitRoutingV3, `active_num=0` | 576.09 us | microbenchmark winner |
| persistent expert-axis compare and reductions | 712.08 us | exact |
| persistent 128x128 identity lookup and reduction | 702.88 us | exact |
| NPU ScatterUpdate rows and reduction | 708.71 us | exact |
| one-hot and reduction | 719.51 us | exact |
| generic functional scatter | 2330.40 us | exact, AI CPU fallback |

Equivalent expressions lowered very differently. Generic
`torch.Tensor.scatter` became `ScatterElements` on AI CPU and averaged 86.4 us
per call. The contract-correct `torch_npu.scatter_update_` variant became the
AI Vector Core `Scatter` kernel and averaged 1.41 us, but it also required an
8x128 zero tensor, `TensorMove`, reduction, and int64 cast. Its valid shape was
destination `[8,128]`, indices `[8]`, updates `[8,1]`, axis 1: this operator's
one-dimensional index contract specifies one axis offset per leading batch
row. Boundary cases included expert IDs 0 and 127. A flat eight-index update
into one destination row is not this operator's contract.

The fused operator remained faster because the manual graphs launched separate
broadcast, comparison, reduction, cast, and count kernels. CANN's documented
full-load template exists for this exact class of small input: inter-operation
multicore synchronization dominates, so load, sort, index construction, count,
and gather execute inside one kernel. TorchAir also reported that
`MoeInitRoutingV3` cannot join a superkernel, so it remains an opaque graph
boundary.

The `active_num=0` alias was exact and 3.6% faster in the routing-only graph,
but it failed the complete-stage gate. Over 1000 B1/KV4096 tokens on physical
910B2 NPU 3, with all 24 layers, final RMSNorm, and LM head, it reached 207.00
tokens/s versus 210.78 tokens/s for the exact production `active_num=8` path.
Both used one warm static graph, had no recompilations, and matched all eight
captured greedy tokens. The 1.79% full-stage regression rejects the alias.

Evidence is stored in `references/routing_probe_*.json`,
`references/qwen3_moe_active0_l24_k4096_warm1000.json`, and
`references/qwen3_moe_native_active8_l24_k4096_warm1000_rerun.json`. The
temporary implementations were removed after the study.

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
- [Huawei `aclnnMoeInitRoutingV3`](https://gitcode.com/cann/ops-transformer/blob/master/moe/moe_init_routing_v3/docs/aclnnMoeInitRoutingV3.md)
- [Huawei `torch_npu.scatter_update_`](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001258.html)
- [Huawei `aclnnInplaceScatterUpdate`](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnInplaceScatterUpdate.md)
