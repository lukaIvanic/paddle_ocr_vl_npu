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

Compiled decode with `actual_seq_lengths`:

```bash
$PYTHON ./run_local_qwen3_0.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
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
- Keep `prefill_tokens + decode_steps <= static_kv_cache_len` for benchmark runs.
