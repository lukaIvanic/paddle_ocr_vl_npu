# Experiment 10: Qwen3-8B Local Inference

Standalone Hugging Face model replacement for dense Qwen3-8B inference on
Ascend NPU. The implementation is config-driven and loads the official
safetensors checkpoint without using Transformers model classes. Transformers
is retained only for tokenization and the independent reference runner.

Run model commands on the Blue Zone container after preparing the NPU shell:

```bash
cd /workspace/repos/paddle_ocr_vl_npu/10_qwen3_8b_inference
source npu-setup

PYTHON=/usr/local/python3.12.13/bin/python3
MODEL_DIR=/workspace/models/Qwen3-8B
```

`/workspace/models/Qwen3-0.6B` can be used as a smaller architectural smoke
checkpoint. Every NPU entrypoint explicitly disables torch-npu JIT compilation;
TorchAir compilation is enabled only by the corresponding command-line option.

Prefill deliberately produces KV state only. Generation re-feeds the final
prompt token through decode to obtain the first output token, so the first and
all subsequent decode iterations use one identical static graph contract.

The default compiled decode path is model-aware. Qwen3-8B retains its validated
dynamic B1 preset:

- dynamic TorchAir decode with `actual_seq_lengths`;
- native NPU RMSNorm and fused residual-add plus RMSNorm;
- one packed QKV projection per layer;
- native NPU rotary embedding;
- unchanged ND Linear weights.

Qwen3-0.6B instead defaults to the faster fixed-shape KV path validated below:

- static TorchAir decode with a boolean KV mask;
- packed QKV, NPU RoPE lookup, and fused residual-add plus RMSNorm;
- Q/K normalization through add-RMSNorm with graph-local zero residuals;
- post-scatter K/V prefetch;
- complete next-layer weight prefetch one layer ahead;
- unchanged ND Linear weights.

`--decode-optimization baseline --decode-linear-weight-format unchanged`
retains the original implementation for controlled comparisons.

## Transformers Reference

Use this as the correctness reference before comparing the local runtime.

```bash
$PYTHON ./transformers_generate.py \
  --model-id Qwen/Qwen3-8B \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --prompt "Write a tiny Python function that adds two numbers." \
  --max-new-tokens 64
```

## Local Generation

Eager decode:

```bash
$PYTHON ./run_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --prompt "Write a tiny Python function that adds two numbers." \
  --max-new-tokens 64 \
  --static-kv-cache-len 4096
```

Compiled decode:

```bash
$PYTHON ./run_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --prompt "Write a tiny Python function that adds two numbers." \
  --max-new-tokens 64 \
  --static-kv-cache-len 4096
```

The command selects the model-specific default. Explicit flags still override
the automatic selection.

Minimal fixed-shape static-graph validation (64 prompt tokens, 64 decode
iterations, KV length 128):

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --prefill-tokens 64 \
  --decode-steps 64 \
  --static-kv-cache-len 128 \
  --prefill-warmups 1 \
  --prefill-repeats 1 \
  --decode-warmups 1 \
  --decode-repeats 2
```

For compiled decode, the benchmark first runs the full multi-step eager and
compiled lanes from independent prefills. It requires exact generated-token
parity and records the final KV-cache maximum absolute difference before
reporting warmed replay throughput.

Explicit equivalent of the Qwen3-8B default compiled decode contract:

```bash
$PYTHON ./run_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
  --decode-optimization combined_apply \
  --decode-linear-weight-format unchanged \
  --prompt "Write a tiny Python function that adds two numbers." \
  --max-new-tokens 64 \
  --static-kv-cache-len 65536
```

## Benchmark

Basic prefill and compiled decode timing:

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 1 \
  --prefill-repeats 3 \
  --decode-warmups 1 \
  --decode-repeats 3
```

65k static cache benchmark using `actual_seq_lengths`:

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 65536 \
  --prefill-warmups 1 \
  --prefill-repeats 3 \
  --decode-warmups 1 \
  --decode-repeats 3
```

## Torch Profiler

Profile prefill and decode:

```bash
rm -rf ../tmp/10_qwen3_8b_inference/profile

$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 1 \
  --prefill-repeats 3 \
  --decode-warmups 1 \
  --decode-repeats 3 \
  --profile both \
  --profile-dir ../tmp/10_qwen3_8b_inference/profile \
  --topn 20
```

Profile only decode with the 65k `actual_seq_lengths` path:

```bash
rm -rf ../tmp/10_qwen3_8b_inference/profile

$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 65536 \
  --prefill-warmups 1 \
  --prefill-repeats 1 \
  --decode-warmups 1 \
  --decode-repeats 1 \
  --profile decode \
  --profile-dir ../tmp/10_qwen3_8b_inference/profile \
  --topn 20
```

## JSON Output

Save benchmark results to JSON:

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 65536 \
  --prefill-warmups 1 \
  --prefill-repeats 3 \
  --decode-warmups 1 \
  --decode-repeats 3 \
  --json-out ../tmp/10_qwen3_8b_inference/benchmark.json
```

## Notes

- `float16` is the default NPU dtype to test first.
- Checkpoint shards are streamed into the final NPU-resident dtype; the local
  runtime does not retain a second complete CPU state dictionary while loading
  the 8B model.
- `--compile-decode-dynamic --decode-increfa-mode actual_seq_lengths` avoids attending over the full static KV cache during decode.

## Ascend 910B2 B1 Decode Result

### Qwen3-0.6B

Measured on physical Ascend 910B2 with FP16, B1, 64 decode steps, and a
KV4096 static cache. The selected path uses the real 151,936-row Qwen LM head.
It matched the baseline greedy tokens for all 64 steps. Compiled and eager
execution also had exact tokens and zero KV-cache difference.

| Decode implementation | Prefix position | Tokens/s |
|---|---:|---:|
| Old dynamic-length baseline | 512 | 151.74 |
| Static mask plus NPU RMSNorm | 512 | 256.59 |
| Packed QKV and fused add-RMS | 512 | 378.11 |
| Earlier stage-aware prefetch | 512 | 401.52 |
| Paddle one-layer-ahead schedule | 512 | 421.09 |
| Paddle one-layer-ahead schedule | 2048 | 419.32 |
| Batched Q/K RMSNorm stock-op probe | 512 | 421.18 |
| **Q/K through add-RMSNorm (default)** | **512** | **443.24** |
| **Q/K through add-RMSNorm (default)** | **2048** | **440.46** |
| Paddle K/V-then-MLP schedule | 512 | 417.87 |
| One-layer-ahead plus FRACTAL_NZ | 512 | 403.30 |
| Two-layer-ahead prefetch | 512 | 388.27 |
| One-layer-ahead plus packed MLP | 512 | 409.81 |

The selected KV4096 result is 69-70% faster than the previous saved 260.24 tok/s
KV64 maximum. FRACTAL_NZ and packed MLP remain explicit ablations because both
regressed Qwen3-0.6B.

The batched Q/K RMSNorm probe reduced standalone RMSNorm from 57 to 29 kernels
per token and saved about 111 microseconds of RMSNorm kernel time. The required
combined learned-gamma multiply and second split added about 93 microseconds,
leaving end-to-end throughput unchanged. It remains an explicit preset rather
than the default; a useful next step requires fusing normalization, learned
gamma, split, and preferably RoPE into one Qwen-specific kernel.

The successful add-RMSNorm lane retains the separate learned Q and K gamma
weights and supplies a zero residual to `npu_add_rms_norm`. A persistent zero
buffer is invalid under TorchAir because the compiled InplaceAddRmsNorm lowering
aliases the summed output onto that residual; it caused 60/64 token mismatches
after the first step. Graph-local `zeros_like` inputs preserve semantics. They
cost about 59 microseconds per token, but the complete normalization path still
saves about 109 microseconds per token and improves throughput by 5.3%.

Run the selected contract explicitly:

```bash
$PYTHON ./benchmark_local_qwen3_0.py \
  --model-dir /workspace/models/Qwen3-0.6B \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --no-compile-decode-dynamic \
  --decode-increfa-mode mask \
  --decode-optimization combined_apply_complete_layer_prefetch1_qk_add_rms_norm_rope_lut \
  --decode-linear-weight-format unchanged \
  --prefill-tokens 2048 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 0 \
  --prefill-repeats 0 \
  --decode-warmups 1 \
  --decode-repeats 3
```

The remaining profile is 42.7% MatMul, 28.6% IncreFlashAttention, 11.8%
InplaceAddRmsNorm, and 2.6% graph-local zero creation by summed device-kernel
time. The Paddle ~899 tok/s KV4096 lab result also uses 18 decoder layers, two
KV heads, and a 16,384-row decode head, versus Qwen3-0.6B's 28 layers, eight KV
heads, and 151,936-row head.

### Qwen3-8B

Measured on physical Ascend 910B2 NPU 7 with FP16, one 512-token prefix,
64 decode steps, and KV capacity 4096. Every optimized lane matched the
baseline greedy tokens for all 64 steps. The selected `combined_apply` lane
also matched optimized eager versus compiled tokens exactly, with zero KV-cache
maximum absolute difference between those two executions.

| Decode implementation | Tokens/s |
|---|---:|
| Original baseline | 66.17 |
| Native RMSNorm only | 69.41 |
| **`combined_apply` (default)** | **79.11** |
| `combined_apply` plus RoPE lookup | 78.95 |
| RoPE lookup plus stage-aware weight prefetch | 60.75 |
| RoPE lookup plus FRACTAL_NZ decode weights | 60.01 |

The default is 19.6% faster than the original implementation. Qwen3-8B did
not inherit two Paddle-specific wins: the stage-aware prefetch schedule and
FRACTAL_NZ weights both regressed this B1 shape, so they remain explicit lab
options rather than defaults.

The before/after decode profiles verify that the original 9,280 AI-CPU scalar
casts were removed. The optimized profile contains no AI-CPU Cast kernels;
only 384 small AI-vector Cast kernels remain, totaling 886 microseconds across
64 decode steps. The largest remaining non-matmul cost is stock
IncreFlashAttention: 2,304 calls, or 36 layers times 64 steps, with 360,888
microseconds of summed AI-CPU time and 51,924 microseconds of MIX_AIC time.
- Keep `prefill_tokens + decode_steps <= static_kv_cache_len` for benchmark runs.
