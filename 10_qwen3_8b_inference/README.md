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

The default compiled decode path is the validated B1 preset:

- dynamic TorchAir decode with `actual_seq_lengths`;
- native NPU RMSNorm and fused residual-add plus RMSNorm;
- one packed QKV projection per layer;
- native NPU rotary embedding;
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

The command uses `combined_apply`, dynamic compilation, and
`actual_seq_lengths` by default. Use `--no-compile-decode-dynamic` or
`--decode-increfa-mode mask` only for an explicit ablation.

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

Explicit equivalent of the default compiled decode contract:

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
