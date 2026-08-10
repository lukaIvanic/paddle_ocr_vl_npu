# Experiment 13: Qwen3 Reranker

Small, self-contained runtime for `Qwen/Qwen3-Reranker-8B`.

This experiment was transferred from the dedicated
`glm-ocr-inference-reranker` worktree at commit
`32885dbb6c5e04ba2756942b7550c587eb112757`. It contains only the custom
Transformers reference, local runtime, quantization code, and benchmarks. It
does not include vLLM or vLLM-Ascend source.

The goal is to make reranker correctness and performance easy to inspect:
reference Transformers scoring, local fixed-shape scoring, compiled forward
benchmarking, profiler summaries, and the vLLM-shaped offline benchmark all live
in this folder.

Run examples below from the repository root:

```bash
cd /workspace/repos/paddle_ocr_vl_npu
```

## Environment

Use a Python environment with PyTorch, Transformers, Safetensors, and, for NPU
runs, `torch_npu`.

```bash
python3 -m pip install transformers safetensors
```

If the model is not already present locally, download it once:

```bash
huggingface-cli download Qwen/Qwen3-Reranker-8B \
  --local-dir /path/to/models/Qwen3-Reranker-8B
```

If your network requires a Hugging Face mirror, set it before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

All commands below use:

```bash
MODEL_DIR=/path/to/models/Qwen3-Reranker-8B
```

## 1. Transformers Reference

Run this first. It is the ground-truth reference for prompt formatting and
yes/no scoring.

```bash
python3 13_qwen3_reranker/transformers_rerank.py \
  --model-dir "$MODEL_DIR" \
  --query "What is the capital of China?" \
  --documents "The capital of China is Beijing." "Gravity attracts two bodies." \
  --max-length 8192 \
  --dtype float16 \
  --device npu:0
```

Expected behavior: the Beijing document should score higher than the unrelated
gravity document.

## 2. Local Runtime Smoke

This checks the small local model, safetensors loader, tokenizer wrapper, and
fixed-shape batched scoring path.

```bash
python3 13_qwen3_reranker/run_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --query "What is the capital of China?" \
  --documents "The capital of China is Beijing." "Gravity attracts two bodies." \
  --max-length 256 \
  --batch-size 2 \
  --dtype float16 \
  --device npu:0
```

Add `--compile-forward` to compile the fixed-shape forward path:

```bash
python3 13_qwen3_reranker/run_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --query "What is the capital of China?" \
  --documents "The capital of China is Beijing." "Gravity attracts two bodies." \
  --max-length 256 \
  --batch-size 2 \
  --dtype float16 \
  --device npu:0 \
  --attention-impl prompt_flash_attention \
  --compile-forward
```

## 3. Practical Forward Benchmark

Use this for controlled fixed-shape local measurements.

Dense FP16:

```bash
python3 13_qwen3_reranker/benchmark_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --max-length 256 \
  --batch-size 16 \
  --dtype float16 \
  --device npu:0 \
  --compile-forward \
  --attention-impl eager \
  --ffn-weight-mode dense \
  --warmups 1 \
  --repeats 3
```

FFN-only W8A8:

```bash
python3 13_qwen3_reranker/benchmark_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --max-length 256 \
  --batch-size 16 \
  --dtype float16 \
  --device npu:0 \
  --compile-forward \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode w8a8 \
  --warmups 1 \
  --repeats 3
```

With profiler summaries:

```bash
python3 13_qwen3_reranker/benchmark_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --max-length 256 \
  --batch-size 16 \
  --dtype float16 \
  --device npu:0 \
  --compile-forward \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode dense \
  --warmups 1 \
  --repeats 3 \
  --profile forward \
  --topn 20
```

The benchmark prints latency, samples/s, padded tok/s, real tok/s, memory
snapshots, and top profiler operators/kernels when profiling is enabled.
If `--batch-size` is larger than the number of `--documents`, the benchmark
repeats the provided documents to fill the static batch.

## 4. vLLM-Shaped Offline Benchmark

This mimics the useful part of the vLLM reranker benchmark without requiring a
server. It generates synthetic query/document pairs, groups them into forward
batches, and reports pair throughput plus real/padded input-token throughput.

Dense FP16:

```bash
python3 13_qwen3_reranker/benchmark_vllm_shape_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-forward \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode dense \
  --num-prompts 1000 \
  --random-input-len 200 \
  --random-batch-size 1 \
  --forward-batch-sizes 8 16 32 \
  --warmup-batches 1
```

FFN-only W8A8:

```bash
python3 13_qwen3_reranker/benchmark_vllm_shape_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --dtype float16 \
  --device npu:0 \
  --compile-forward \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode w8a8 \
  --num-prompts 1000 \
  --random-input-len 200 \
  --random-batch-size 1 \
  --forward-batch-sizes 8 16 32 \
  --warmup-batches 1
```

`--random-batch-size` controls documents per synthetic rerank request.
`--forward-batch-sizes` controls the local device batch size used to process the
generated pairs. This benchmark is offline, so it does not measure HTTP or
server scheduler overhead.

## Chunked PromptFA Prefill

The eager custom runtime can process a fixed padded sequence as sequential
PromptFA blocks while retaining native per-layer KV states. For the 310P-safe
contract, both the chunk size and padded sequence length must be aligned to 128
tokens:

```bash
python3 13_qwen3_reranker/run_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --max-length 256 \
  --batch-size 2 \
  --dtype float16 \
  --device npu:0 \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode dense \
  --prefill-chunk-size 128
```

Each block uses a bool `[B,1,Q,K]` causal/padding mask and omits
`actual_seq_lengths` and `actual_seq_lengths_kv`. GQA KV states stay compact in
the per-layer cache and are expanded only at the PromptFA boundary. Add
`--compile-forward` to compile the fixed chunk schedule as one static TorchAir
prefill graph; the yes/no projection remains outside that graph.

## Weight Modes

- `dense`: all model weights stay FP16.
- `w8a8`: only FFN projections are W8A8. This is the useful optimized path so
  far.
- `all_w8a8`: all internal transformer linears are W8A8. 

W8A8 calibration is static and happens during runner initialization. It is not
performed inside each forward pass.

## Optimizations Included

- Fixed-shape padded forward path: avoids dynamic control flow and gives compile
  one stable shape per benchmark bucket.
- Optional 310P-compatible PromptFA attention path:
  `--attention-impl prompt_flash_attention` uses FP16 BNSD inputs, a contiguous
  bool causal/padding mask, `sparse_mode=0`, and no actual-sequence-length
  arguments. Atlas inference-series hardware does not support a non-default
  `num_key_value_heads`, so the runtime expands GQA key/value heads before the
  operator call. This uses more KV memory than native GQA on 910B, but keeps one
  operator contract that can also run on 310P.
- Yes/no-only scoring head: reranking only needs the `yes` and `no` logits, so
  the local runtime projects the final hidden state onto those two lm-head rows
  instead of computing the full vocabulary logits.
- FFN-only W8A8 mode: replaces the largest dense FFN projections with calibrated
  W8A8 linears while leaving attention and normalization simple.
- Shared gate/up activation quantization: `gate_proj` and `up_proj` consume the
  same hidden states, so the W8A8 FFN path quantizes that activation once and
  reuses it for both projections.
- Static W8A8 input scales: activation scales are calibrated up front, so the
  hot path does not recompute quantization parameters each forward pass.
- Compile-safe causal mask construction: the local model avoids convenience ops
  that were noisy for compile and builds the fixed-shape mask directly.

## Known Caveats

- The compiled path is fixed-shape. Change `--max-length` or `--batch-size` and
  you should expect a separate compile.
- The PromptFA path deliberately does not pass `actual_seq_lengths`,
  `actual_seq_lengths_kv`, or `num_key_value_heads`. Left padding and causality
  are represented only by the full boolean attention mask. This is required by
  the Atlas inference-series PromptFA contract used by 310P.
- The local runtime currently uses explicit fixed-shape causal attention. Large
  buckets such as `--max-length 8192` can materialize very large attention
  tensors and OOM even when the Transformers reference fits. Use
  `--attention-impl prompt_flash_attention` for larger fixed buckets.
- W8A8 probably changes scores. It is useful for throughput experiments, but relevance
  quality should be checked on real reranking data before production use.
