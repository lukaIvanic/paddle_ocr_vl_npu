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
`actual_seq_lengths` and `actual_seq_lengths_kv`. When cached K/V makes `Q<K`,
the 310P boundary prepends disposable projected-Q rows and a safe dummy mask so
PromptFA physically receives `Q=K`; only the real output rows are retained. GQA
KV states stay compact in the per-layer cache and are expanded only at the
PromptFA boundary. Add
`--compile-forward` to compile each fixed-shape chunk step with TorchAir. The
host sequences the compiled steps while KV tensors remain on NPU; the yes/no
projection remains outside the compiled prefill graphs.

## Static Prefix Cache

The first throughput-oriented prefix-cache lane uses one static shape. It
caches the fixed 60-token system/task prefix once, using manual eager attention,
then runs one disk-cached TorchAir continuation graph with PromptFA:

- prefix build: B1, physical S128, manual eager attention;
- continuation: fixed request batch, real Q128, physical Q256/KV256, PromptFA;
- graph output: final continuation hidden states only; continuation KV is not
  returned or stored.

The semantic maximum length is 60 cached tokens plus 128 continuation tokens,
so the default task requires `--max-length 188`:

```bash
python3 13_qwen3_reranker/run_local_qwen3_reranker.py \
  --model-dir "$MODEL_DIR" \
  --max-length 188 \
  --batch-size 2 \
  --dtype float16 \
  --device npu:0 \
  --attention-impl prompt_flash_attention \
  --ffn-weight-mode dense \
  --prefill-chunk-size 128 \
  --prefix-cache \
  --prefill-optimization combined \
  --compile-forward \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker \
  --graph-warmups 2
```

The prefix/continuation tokenizer split is checked against the original full
prompt token IDs for every pair. If segmentation changes any token ID, the
runtime refuses to use the cache. Graph cache directories include the fixed
shape, model dimensions, and a source hash.

`--prefill-optimization baseline` preserves the original compiled suffix.
`combined` keeps the one-time B1 prefix build on that eager/manual reference,
then applies four changes only to the repeated compiled suffix:

- native `npu_rms_norm` for decoder and Q/K norms;
- one native Q/K rotary call per layer;
- one square PromptFA mask prepared outside the compiled layer stack;
- one-time expansion of reusable prefix K/V heads, while only current K/V is
  repeated per request.

`combined_bsnd` is an experimental derivative. It also converts the prepared
prefix cache to BSND once, keeps projected Q/K/V in BSND, and consumes the BSND
PromptFA output directly. This removes the Q, K, V, and attention-output
transposes from every decoder layer. It does not change square Q padding or the
explicit GQA expansion required by the 310P-safe contract.

The B4, real-Q128, physical-Q/KV256 FP16 comparison on Ascend 910B2 used 50
warm synchronized repetitions. `combined` reduced median latency from 14.863
ms to 10.233 ms and increased executed-model throughput from 34,447 to 50,035
tok/s. Relative to the compiled baseline, maximum final-hidden absolute drift
was 0.0625, maximum yes/no-logit drift was 0.015625, maximum yes-score drift was
0.0007681, and every binary choice matched. This is a 910B2 result, not 310P
validation. The compact result is retained under
`tmp/13_qwen3_reranker/prefix_opt_final_b4_c128_910b2_ce8b947.json`.

At `459a9c2`, the same B4/Q128 comparison tested `combined_bsnd` on Ascend
910B2. A direct square-Q256 PromptFA call was bit-exact between BNSD and BSND,
and operator latency was effectively unchanged. The compiled end-to-end gain
came from the surrounding graph:

- forward-order medians: BNSD 10.418 ms, BSND 10.384 ms (0.3% lower);
- reverse-order warm-cache medians: BNSD 10.373 ms, BSND 10.118 ms (2.5% lower);
- separate clean profile controls: BNSD 10.416 ms, BSND 10.001 ms (4.0% lower);
- device time: 10.205 ms to 9.819 ms (3.8% lower);
- kernel launches: 682 to 570 per forward; all 112 transposes were removed.

Both layouts had the same maximum hidden-state, yes/no-logit, and yes-score
deltas versus the compiled baseline, and every binary choice matched. Because
the latency delta is small and order-sensitive, retain BSND as an experimental
preset until an alternating same-process benchmark and a direct 310P run pass.
Compact evidence is retained in
`tmp/13_qwen3_reranker/prefix_bsnd_b4_c128_910b2_459a9c2.json`,
`tmp/13_qwen3_reranker/prefix_bsnd_b4_c128_reverse_910b2_459a9c2.json`, and
`tmp/13_qwen3_reranker/profile_prefix_combined_bsnd_b4_c128_910b2_459a9c2/`.

### FRACTAL_NZ linear weights

The prefix-cache benchmark can cast the seven transformer linears in every
decoder layer to torch-npu format code 29 before prefix construction and graph
compile:

```bash
python3 13_qwen3_reranker/benchmark_prefix_cache_throughput.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --batch-sizes 4 \
  --continuation-lengths 128 \
  --batch-sweep-continuation 128 \
  --length-sweep-batch 4 \
  --matrix axes \
  --lanes prefix_promptfa_compiled \
  --prefill-optimizations combined_bsnd \
  --linear-weight-format fractal_nz \
  --warmups 5 \
  --repeats 100 \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_nz \
  --json-out tmp/13_qwen3_reranker/prefix_bsnd_fractal_nz.json
```

The option enables `torch.npu.config.allow_internal_format` before the first NPU
allocation, casts with `torch_npu.npu_format_cast(weight, 29)`, and fails if any
target weight does not report format 29. It targets 196 weights in the 28-layer
0.6B model. The tied embedding/lm-head weight stays native because embedding
lookup and the two-row yes/no projection are outside the compiled transformer
prefill graph. The one-time cast is model setup, not timed forward work.

At `223ff9c`, a B4, real-Q128, physical-Q/KV256 FP16 test on Ascend 910B2 found
no decisive steady-state gain. With native run first, native was 10.158 ms and
NZ was 10.171 ms, so NZ was 0.13% slower. With the order reversed and both
graphs warm, NZ was 9.942 ms and native was 10.049 ms, so NZ was 1.07% faster.
All 196 weights changed from format 2 to format 29. The maximum yes/no-logit
difference was 0.015625, the yes-score difference was 0.0003854, and every
binary choice matched.

A three-forward kernel profile at `4dea1ee` also showed no 910B2 MatMul gain:
MatMul time changed from 3.273 to 3.325 ms per forward, while total traced device
time changed from 10.040 to 9.988 ms. Both graphs launched the same 196 MatMuls
and 570 total kernels per forward. Treat FRACTAL_NZ as a 310P experiment, not a
910B2 default. The compact evidence is retained under
`tmp/13_qwen3_reranker/prefix_bsnd_*_910b2_223ff9c.json` and
`tmp/13_qwen3_reranker/profile_prefix_bsnd_*_910b2_4dea1ee/`.

For a fair 310P comparison, also enable internal formats in the native control.
Run the native and NZ commands in separate processes and use separate graph
cache paths:

```bash
python3 13_qwen3_reranker/benchmark_prefix_cache_throughput.py \
  --model-dir /path/to/Qwen3-Reranker-0.6B \
  --device npu:0 --batch-sizes 4 --continuation-lengths 128 \
  --batch-sweep-continuation 128 --length-sweep-batch 4 --matrix axes \
  --lanes prefix_promptfa_compiled --prefill-optimizations combined_bsnd \
  --linear-weight-format native --enable-internal-format \
  --warmups 5 --repeats 100 \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_nz_310p \
  --json-out tmp/13_qwen3_reranker/prefix_native_310p.json

python3 13_qwen3_reranker/benchmark_prefix_cache_throughput.py \
  --model-dir /path/to/Qwen3-Reranker-0.6B \
  --device npu:0 --batch-sizes 4 --continuation-lengths 128 \
  --batch-sweep-continuation 128 --length-sweep-batch 4 --matrix axes \
  --lanes prefix_promptfa_compiled --prefill-optimizations combined_bsnd \
  --linear-weight-format fractal_nz \
  --warmups 5 --repeats 100 \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_nz_310p \
  --json-out tmp/13_qwen3_reranker/prefix_fractal_nz_310p.json
```

### Qwen3-Reranker-4B on 910B2

The same local runtime and compiled prefix-cache graph work without model code
changes for `Qwen/Qwen3-Reranker-4B`. The downloaded checkpoint is 7.6 GiB and
uses 36 layers, hidden size 2560, intermediate size 9728, 32 query heads, and 8
KV heads. A basic eager PromptFA smoke scored the relevant Beijing document at
0.9980 and the unrelated document at `1.14e-5`.

The representative compiled test used B4, real Q128, physical Q/KV256, FP16,
`combined_bsnd`, and internal formats on one Ascend 910B2. The native graph
compiled successfully on its first 71.9-second call. Its fresh-process disk
cache was then reused successfully; each native or FRACTAL_NZ graph directory
occupies about 17 MiB. The NZ graph was built second, so its 37.9-second first
call benefits from shared compiler/operator state and is not a comparable cold
compile measurement.

The order-reversed warm comparison was:

| Weight format | Median | Pairs/s | Executed tok/s | Served tok/s |
|---|---:|---:|---:|---:|
| Native | 30.636 ms | 130.57 | 16,712 | 24,546 |
| FRACTAL_NZ | 28.104 ms | 142.33 | 18,218 | 26,758 |

FRACTAL_NZ reduced latency by 8.26% and increased executed-token throughput by
9.01%. The earlier forward-order comparison showed the same direction, with NZ
6.95% lower latency. All 252 transformer weights reported format 29. The maximum
yes/no-logit difference was 0.015625, the yes-score difference was 0.0004019,
and every binary choice matched.

The independent warm profiles explain the gain:

| Profile component | Native | FRACTAL_NZ |
|---|---:|---:|
| Clean median | 30.167 ms | 27.863 ms |
| Traced device time | 29.923 ms | 27.737 ms |
| 252 MatMuls | 19.562 ms | 17.353 ms |
| 36 PromptFA calls | 3.695 ms | 3.856 ms |
| Total kernels | 730 | 730 |

Native MatMul occupied 65.4% of device time, while PromptFA occupied 12.35%.
The FFN gate/up/down projections alone used about 14.0 ms, or 46.8% of device
time. NZ saved 2.21 ms in MatMul and 2.19 ms in the total traced forward, so the
end-to-end improvement comes almost entirely from the larger linear weights.
This establishes the B4/Q128 910B2 point only; other batches and lengths still
need their own static graphs and measurements. Compact evidence is retained in
`tmp/13_qwen3_reranker/prefix_4b_b4_c128_*_910b2.json` and
`tmp/13_qwen3_reranker/profile_prefix_4b_b4_c128_*_910b2/`.

Run the same controlled matrix on 310P before making `combined` the deployment
default:

```bash
python3 13_qwen3_reranker/benchmark_prefix_cache_throughput.py \
  --model-dir /path/to/Qwen3-Reranker-0.6B \
  --device npu:0 \
  --batch-sizes 4 \
  --continuation-lengths 128 \
  --batch-sweep-continuation 128 \
  --length-sweep-batch 4 \
  --matrix axes \
  --lanes prefix_promptfa_compiled \
  --prefill-optimizations baseline native_rms native_rotary \
    prebuilt_square_mask expanded_prefix_kv native_rms_rotary \
    native_rms_rotary_mask combined combined_bsnd \
  --warmups 3 \
  --repeats 50 \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_opt_310p \
  --json-out tmp/13_qwen3_reranker/prefix_opt_b4_c128_310p.json
```

### Prefix-cache throughput matrix

Use the static-shape benchmark to sweep batch size at continuation length 128
and continuation length at batch size 1:

```bash
python3 13_qwen3_reranker/benchmark_prefix_cache_throughput.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --batch-sizes 1,2,4,8 \
  --continuation-lengths 128,256,512 \
  --matrix axes \
  --warmups 2 \
  --repeats 10 \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_throughput \
  --json-out /tmp/qwen3_reranker_prefix_throughput.json
```

The benchmark loads the model once, builds the 60-token reusable prefix once
with manual eager attention, and compares three lanes for every selected shape:

- full manual prefill;
- full compiled PromptFA prefill;
- prefix-cached compiled PromptFA with square-padded Q.

It excludes the first graph call and all warmups from steady measurements. The
main metrics are `served_input_tok_s` (semantic prompt tokens served),
`executed_model_tok_s` (tokens that execute decoder QKV and MLP in the timed
call), and `physical_attention_q_tok_s` (PromptFA Q rows including disposable
square-padding rows). Use `--matrix cross` only when the full batch-by-length
cross-product is required.

### Prefix-cache forward profile

Profile one warm compiled prefix-cache graph separately from the unprofiled
throughput measurement. The default B4/C128 case reuses the corresponding
static graph cache from the throughput sweep:

```bash
python3 13_qwen3_reranker/profile_prefix_cache_forward.py \
  --model-dir "$MODEL_DIR" \
  --device npu:0 \
  --batch-size 4 \
  --continuation-length 128 \
  --warmups 3 \
  --repeats 20 \
  --profile-iters 3 \
  --prefill-optimization combined \
  --compile-cache-dir .runtime_cache/13_qwen3_reranker/prefix_throughput \
  --profile-dir tmp/13_qwen3_reranker/profile_prefix_b4_c128_910b2
```

The script reports the normal synchronized median and tok/s before enabling
the profiler. It then reports the synchronized profiled-call latency and the
profiler overhead ratio separately. Model loading, prefix-cache construction,
graph cache load/compile, and trace export are outside the profiled forward
window. The generated summary includes top NPU operator types, kernels, core
types, and shape-based module attribution.

The optimized B4, real-Q128, physical-Q/KV256 profile on Ascend 910B2 at
`23d590b` used a warm `combined` graph, 30 clean repetitions, and three profiled
calls. Clean median latency was 10.416 ms (49,153 executed model tok/s); the
profiled median was 10.711 ms, so profiler overhead was 2.83%. The NPU trace
accounted for 10.205 ms and 682 kernels per forward. Compared with the earlier
baseline profile from `c0943f8`, clean latency fell from 15.422 ms, device time
fell from 15.578 ms, and kernel count fell from 1,582. That is 32.5% lower clean
latency, 34.5% lower device time, and 56.9% fewer kernel launches.

The remaining per-forward device profile is:

| Operator family | Kernels | Device share | Interpretation |
|---|---:|---:|---|
| MatMul | 196 | 32.3% | Seven dense projections per decoder layer. |
| PromptFlashAttention | 28 | 17.3% | One square physical Q256/KV256 call per layer. |
| RmsNorm + InplaceAddRmsNorm | 113 | 19.8% | Native Q/K/final norms plus graph-fused residual norms. |
| Transpose | 112 | 10.4% | Q, K, V into BNSD and attention output back to BSND. |
| Concat | 85 | 6.8% | Prefix/current K/V, dummy Q rows, and one RoPE frequency concat. |
| BroadcastTo | 56 | 3.8% | Current GQA K/V expansion from 8 to 16 heads. |
| ApplyRotaryPosEmb | 28 | 3.5% | One native Q/K rotary call per layer. |
| Other | 64 | 6.2% | MLP elementwise fusion, dummy-output slicing, embedding support. |

The original BNSD profile no longer exposed one dominant accidental slow path.
MatMul, PromptFA, native/fused norms, rotary, and MLP elementwise work accounted
for about 76.5% of device time. Transpose, concat, GQA broadcast, and
dummy-output slicing accounted for about 22.9%. The subsequent
`combined_bsnd` experiment removed the four transposes per layer, as described
above. Square Q padding remains required by the currently validated masked 310P
PromptFA contract.

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
  arguments. The runtime keeps compact GQA key/value heads and passes their real
  count through `num_key_value_heads`; this contract was validated directly on
  310P before becoming the default.
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
- The PromptFA path deliberately does not pass `actual_seq_lengths` or
  `actual_seq_lengths_kv`. It does pass `num_key_value_heads` for native GQA.
  Left padding and causality are represented by the full boolean attention mask.
- The local runtime currently uses explicit fixed-shape causal attention. Large
  buckets such as `--max-length 8192` can materialize very large attention
  tensors and OOM even when the Transformers reference fits. Use
  `--attention-impl prompt_flash_attention` for larger fixed buckets.
- W8A8 probably changes scores. It is useful for throughput experiments, but relevance
  quality should be checked on real reranking data before production use.
