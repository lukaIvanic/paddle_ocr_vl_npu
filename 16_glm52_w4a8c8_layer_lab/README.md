# Experiment 16: GLM-5.2 W4A8C8 layer lab

This experiment owns a standalone GLM-5.2 decoder-layer implementation. It
uses vLLM-Ascend source only as an architecture and operator-contract reference;
the runtime model imports only PyTorch, torch-npu, and safetensors.

The first rung is layer 3 because it is the first MoE layer and its configured
DSA top-k indexer is skipped. The owned path implements dense, unabsorbed MLA
attention with a static full K/V cache, W8A8 attention and shared-expert
linears, and W4A8 routed experts. This is a correctness and compileability rung,
not yet the final absorbed-MLA performance path.

Eager and compiled timing use independent empty KV caches. Each lane runs the
same ordinary decode loop for warmup and measurement. Warmup includes cold
first use (and compile/cache load for TorchAir) and is excluded from measured
latency. The normal static result is one graph captured during warmup. The
runner reports the graph count after warmup and fails if another graph appears
inside the measured window.

The standalone harness supplies a fresh activation tensor to every layer call,
as an upstream layer would in a real model. The small activation clone is
included in both eager and compiled timing. Reusing one consumed standalone
activation across calls gives invalid multi-token KV state.

## Verified 910B2 result

Commit `e04f03a` ran on physical Ascend 910B2 NPU 7 with B1, BF16 activations,
KV4096, 10 excluded warmup calls, and 1,000 measured layer calls:

| Lane | Mean layer time | Layer calls/s |
| --- | ---: | ---: |
| Owned eager | 2.906 ms | 344.1 |
| Owned TorchAir static compile | 2.014 ms | 496.6 |

The compiled output and all 1,010 written key/value cache rows matched eager
exactly (`max_abs=0`). Dynamo reported one unique graph after warmup and zero
new graphs during measurement. The full summary is in
`references/layer3_steady1000_910b2_e04f03a.json`.

This single-layer result does not establish vLLM-Ascend reference parity or
whole-model token throughput. It decompresses MLA into a full K/V cache, and
the standalone layer 3 has no upstream shared DSA selection. Stack-level DSA,
absorbed MLA, real upstream activations, and final logits require later rungs.

## Verified layers 0-6 stack

The owned seven-layer stack covers the decoder-layer variants present at the
front of GLM-5.2:

- layers 0-2: W8A8 dense MLPs with full DSA indexers;
- layers 3-5: W4A8 MoE layers reusing the last DSA selection;
- layer 6: W4A8 MoE with a refreshed DSA selection.

The current DSA implementation owns the index projections, interleaved RoPE,
index cache, top-2048 selection, shared-index propagation, and sparse attention.
It uses ordinary PyTorch/torch-npu operations rather than vLLM-Ascend runtime
classes or its fused sparse-attention extension.

Commit `107dca3` ran layers 0-6 on physical Ascend 910B2 NPU 7 with B1,
KV4096, 10 ordinary warmup positions, and 2,200 measured positions. The final
162 positions had more than 2,048 valid cache entries, so DSA pruned tokens
rather than selecting the complete valid context.

| Lane | Mean seven-layer time | Stack calls/s |
| --- | ---: | ---: |
| Owned eager | 23.027 ms | 43.43 |
| Owned TorchAir static compile | 9.115 ms | 109.71 |

Measured HBM allocation was 20.13 GiB for weights, 21.89 GiB after the eager
KV/index caches, and 23.97 GiB after the compiled parity lane. The compiled
run captured one graph during warmup and no new graph during measurement.
Compiled-versus-eager output max/mean absolute differences were 0.015625 and
0.000265. Maximum K/V/index cache differences were 0.007813, 0.001953, and
0.023438.

This establishes fit, eager execution, static compilation, state propagation,
and internal eager/compiled parity. It does not yet establish independent
vLLM-Ascend parity, final model logits, or production performance. The full K/V
cache and manual sparse attention must be replaced by absorbed MLA and the
target fused DSA kernels before extrapolating this timing to the full model.
The full result is in
`references/layers0_6_dsa2200_910b2_107dca3.json`.

Run on one Ascend 910B2 after sourcing the NPU environment:

```bash
python3 16_glm52_w4a8c8_layer_lab/run_layer3.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --cache-length 4096 \
  --warmup-steps 10 \
  --decode-steps 1000 \
  --summary-out /tmp/glm52_layer3.json
```
