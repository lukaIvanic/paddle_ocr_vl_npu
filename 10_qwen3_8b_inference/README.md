# Experiment 10: Optimized Qwen3-0.6B Decode on Ascend

This experiment contains one implementation: the selected FP16, B1,
fixed-shape Qwen3-0.6B decode path for Ascend NPU.

It does not contain baseline or ablation implementations. The old dynamic
decode, alternative attention contracts, NPU Graph replay, FRACTAL_NZ weights,
packed MLP, batched Q/K RMSNorm, Transformers reference, and Qwen3-8B routes
were removed after the optimization study. Their measured results remain in
this document as historical evidence.

The runtime loads the official Qwen3-0.6B safetensors checkpoint without using
Transformers model classes. Transformers is used only for tokenization. The
runner rejects any checkpoint whose architecture is not the validated
Qwen3-0.6B shape:

- hidden size 1,024;
- intermediate size 3,072;
- 28 decoder layers;
- 16 query heads and 8 K/V heads;
- head dimension 128;
- vocabulary size 151,936.

## Result

The selected path reaches **450.67 decode tok/s** at prefix position 512 and
**447.21 decode tok/s** at prefix position 2,048. Both runs use B1, FP16, 64
decode steps, a KV4096 cache, and the real 151,936-row LM head on an Ascend
910B2.

The benchmark compares compiled execution with the same optimized graph in
eager mode from independent prefills. At both prefix positions:

- all 64 generated tokens matched exactly;
- the final compiled/eager K/V caches had zero maximum absolute difference.

The 450.67 tok/s result is 2.97 times the 151.74 tok/s KV4096 starting point.

After the old code paths were deleted, commit `ad1c83a` repeated the same
contract for five measured runs. It averaged 448.17 tok/s, with individual
runs from 443.13 to 451.84 tok/s, exact compiled/eager tokens, and zero K/V
difference. The cleanup therefore preserved the selected performance within
normal run-to-run variation.

## The first milestone: beating vLLM-Ascend with fullgraph TorchAir

Before the model-specific optimization work, the first local runtime already
beat the recorded vLLM-Ascend B1 result using a small, direct decode graph:

| Runtime | Decode contract | Decode tok/s |
|---|---|---:|
| vLLM-Ascend | Recorded pure B1 decode | 123.5 |
| Local TorchAir | B1, KV128, 64 steps | 239.09 |
| Local TorchAir | B1, KV64, 32 steps | 260.24 |

The stable 239.09 tok/s result was 1.94 times the recorded vLLM-Ascend result.
The short-cache maximum was 2.11 times faster.

This first win did not require a custom kernel or a serving runtime. It came
from a simple static execution contract:

1. Reimplement the dense Qwen3 forward directly in PyTorch and Torch-NPU.
2. Keep the K/V cache caller-owned and fixed in shape.
3. Use one uniform one-token decode signature for every generation step.
4. Express Linear calls as unambiguous 2-D MatMuls.
5. Update K/V state with the NPU scatter operator.
6. Use NPU incremental flash attention.
7. Compile the whole decode step with
   `torch.compile(fullgraph=True, dynamic=False, backend=torchair)`.
8. Disable torch-npu JIT compilation so no unrelated JIT path is measured.

The comparison is a decoder-kernel comparison, not a serving-QPS claim. The
vLLM and local measurements did not use identical K/V capacities, and the local
benchmark excludes request scheduling, HTTP, tokenization, and prefill. A
previous vLLM B4 result of 309.5 aggregate output tok/s included prefill and is
not used as a pure-decode comparison.

## KV4096 optimization ladder

We then fixed the harder target at B1, FP16, prefix position 512, 64 decode
steps, KV4096, and the real LM head. Each accepted row includes all accepted
changes above it.

| Accepted implementation | Decode tok/s | Gain from prior row |
|---|---:|---:|
| KV4096 starting implementation | 151.74 | - |
| Static mask and native NPU RMSNorm | 256.59 | +104.85, +69.1% |
| Packed QKV, fused add-RMSNorm, native RoPE | 378.11 | +121.52, +47.4% |
| Stage-aware weight prefetch | 401.52 | +23.41, +6.2% |
| Complete next-layer prefetch schedule | 421.09 | +19.57, +4.9% |
| Q/K normalization through add-RMSNorm | 443.24 | +22.15, +5.3% |
| Unbound graph-local Q/K zero banks | **450.67** | **+7.43, +1.7%** |

### 1. Static masked decode and native RMSNorm: 151.74 to 256.59

The old long-cache path carried dynamic-length machinery into decode. The
accepted contract fixes the complete K/V tensor shape and builds a boolean
future-slot mask from `cache_position`. Incremental flash attention receives
the static K/V cache and this mask on every step.

All decoder RMSNorm operations that do not include a residual addition use
`torch_npu.npu_rms_norm`. This removes manual FP32 cast, square, mean, reciprocal
square root, multiply, and cast sequences from the graph.

Together these changes improved KV4096 B1 decode by 69.1%, to 256.59 tok/s.

### 2. Packed QKV, fused residual normalization, and NPU RoPE: 378.11

The decode-only setup concatenates the checkpoint Q, K, and V projection
weights once. Each decoder layer then issues one packed QKV MatMul instead of
three small MatMuls. The packed output is split into the model's 16 Q heads and
8 K/V heads.

Residual addition and the following RMSNorm use
`torch_npu.npu_add_rms_norm`. The graph carries the returned summed residual to
the next layer, which removes separate residual-add and normalization sequences.

Q and K rotary embedding uses `torch_npu.npu_apply_rotary_pos_emb` in BSND
layout. RoPE cosine and sine factors for the complete KV capacity are computed
once during setup; each decode step only selects its position from the lookup
table.

This combined rung reached 378.11 tok/s, another 47.4% improvement.

### 3. Stage-aware prefetch: 401.52

The first prefetch schedule moved upcoming decoder weights toward the compute
stage that would consume them. It also prefetched the K/V tensors after their
scatter update and before incremental attention.

This reduced weight and cache movement stalls enough to reach 401.52 tok/s.

### 4. Complete next-layer prefetch: 421.09

The better schedule came from the optimized Paddle decoder. While layer `N`
runs, the graph prefetches the complete weight set for layer `N+1`:

- packed QKV projection;
- attention output projection;
- separate gate and up projections;
- MLP down projection.

The last decoder layer prefetches the LM-head weight instead. One-layer-ahead
prefetch reached 421.09 tok/s at prefix 512 and 419.32 tok/s at prefix 2,048.

### 5. Q/K normalization through add-RMSNorm: 443.24

Qwen3 applies learned RMSNorm separately to every Q and K head. Stock separate
RMSNorm calls remained expensive at this small model size. The accepted path
uses `npu_add_rms_norm` with the original learned Q or K gamma and a zero
residual.

TorchAir lowers this operation to `InplaceAddRmsNorm`; its summed output aliases
the residual input. A persistent zero tensor is therefore invalid because it
stops being zero after the first step. That invalid implementation caused
60 mismatches in 64 generated tokens.

Creating fresh per-layer zero residuals inside the graph preserved semantics
and raised throughput to 443.24 tok/s at prefix 512 and 440.46 tok/s at prefix
2,048.

### 6. Unbound graph-local zero banks: 450.67

Per-layer Q/K zero creation still produced 56 `ZerosLike` kernels per token.
The final path creates one graph-local Q bank and one graph-local K bank for all
28 layers. It unbinds each bank once and passes direct, disjoint layer views to
the two add-RMSNorm calls.

A combined Q+K bank was rejected first. Per-layer indexing introduced 1,984
`GatherV2` and 3,584 `StridedSliceD` calls over 64 tokens and regressed decode
to 414.66 tok/s. Separate unbound banks removed that indexing penalty and
reached the selected result:

- 450.67 tok/s at prefix 512;
- 447.21 tok/s at prefix 2,048.

## Rejected optimizations

Only the winning implementation remains in code. These measured alternatives
were slower and were removed:

| Rejected alternative, prefix 512 | Decode tok/s | Reason not retained |
|---|---:|---|
| Batched stock Q/K RMSNorm | 421.18 | Saved norm calls but added gamma multiply and split work |
| K/V-then-MLP prefetch schedule | 417.87 | Worse than complete-layer-ahead prefetch |
| Complete-layer prefetch plus FRACTAL_NZ | 403.30 | Layout conversion and execution regressed this shape |
| Complete-layer prefetch two layers ahead | 388.27 | Prefetched too early and displaced useful data |
| Complete-layer prefetch plus packed MLP | 409.81 | One larger gate/up MatMul was slower here |
| One combined Q/K zero bank | 414.66 | Gather and slice kernels erased the zero-fill saving |

The final implementation therefore keeps separate gate/up MatMuls and native
ND Linear weights. These are deliberate measured choices, not missing work.

## Current decode contract

There are no runtime optimization selectors. Every invocation uses:

- Qwen3-0.6B only;
- FP16 only;
- Ascend NPU only;
- fixed-shape `dynamic=False` fullgraph TorchAir decode;
- caller-owned static K/V caches;
- boolean masked GQA incremental flash attention;
- packed decode-only QKV weights;
- native NPU add-RMSNorm and RMSNorm;
- RoPE lookup plus native NPU rotary embedding;
- NPU scatter cache updates and post-scatter K/V prefetch;
- complete next-layer weight prefetch;
- separate graph-local, unbound Q/K zero banks;
- unchanged ND Linear weights;
- the real tied 151,936-row LM head and greedy argmax.

Raw eager execution remains internal only as the correctness oracle for the
same optimized graph. It is not exposed as an alternative generation mode.

## Run generation

Run on the Blue Zone container:

```bash
cd /workspace/repos/paddle_ocr_vl_npu/10_qwen3_8b_inference
source npu-setup

PYTHON=/usr/local/python3.12.13/bin/python3
MODEL_DIR=/workspace/models/Qwen3-0.6B

$PYTHON ./run_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --static-kv-cache-len 4096 \
  --prompt "Write a tiny Python function that adds two numbers." \
  --max-new-tokens 64
```

The runner always compiles the optimized decode graph. There is no eager,
dynamic, alternative-attention, weight-format, or optimization-preset flag.

## Benchmark

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 0 \
  --prefill-repeats 0 \
  --decode-warmups 1 \
  --decode-repeats 3 \
  --json-out ../tmp/10_qwen3_8b_inference/optimized_b1_kv4096.json
```

The benchmark first runs 64 eager and compiled decode steps from independent
prefills. It fails on any token mismatch. It records final K/V maximum absolute
difference, excludes graph compile/load from warmed throughput, and reports the
first compiled-call time separately.

Keep `prefill_tokens + decode_steps <= static_kv_cache_len`.

## Profile

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 0 \
  --prefill-repeats 0 \
  --decode-warmups 0 \
  --decode-repeats 1 \
  --profile decode \
  --profile-dir ../tmp/10_qwen3_8b_inference/optimized_profile \
  --topn 20
```

At 450.67 tok/s, summed device-kernel time is:

- 43.5% MatMul;
- 29.8% IncreFlashAttention;
- 12.0% InplaceAddRmsNorm;
- 3.8% cache scatter;
- 3.5% rotary embedding.

The real LM-head MatMul is the largest single kernel. Graph-local bank filling
is only 0.07% and the associated unpacks are 0.28%.
