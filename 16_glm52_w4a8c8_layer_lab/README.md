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
latency. TorchAir may compile or recompile during warmup. The runner only fails
if a new graph appears inside the measured window.

Run on one Ascend 910B2 after sourcing the NPU environment:

```bash
python3 16_glm52_w4a8c8_layer_lab/run_layer3.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --cache-length 4096 \
  --warmup-steps 2 \
  --decode-steps 20 \
  --summary-out /tmp/glm52_layer3.json
```
