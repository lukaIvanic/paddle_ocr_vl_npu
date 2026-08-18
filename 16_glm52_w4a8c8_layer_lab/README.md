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

This does not establish vLLM-Ascend reference parity or whole-model token
throughput. The current attention path decompresses MLA into a full K/V cache,
and layer 3 skips the configured DSA top-k indexer. Absorbed MLA, DSA layers,
real upstream activations, and final logits remain future rungs.

Run on one Ascend 910B2 after sourcing the NPU environment:

```bash
python3 16_glm52_w4a8c8_layer_lab/run_layer3.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --cache-length 4096 \
  --warmup-steps 10 \
  --decode-steps 1000 \
  --summary-out /tmp/glm52_layer3.json
```
