# Experiment 26: minimal BGE-M3 dense embeddings

An inference-only PyTorch implementation of BAAI/bge-m3, following the custom
Qwen3 embedder's structure. The model code is independent of Transformers;
Transformers supplies the tokenizer and the validation reference only.

## Contract

FP16 XLM-RoBERTa encoder -> first-token (CLS) vector -> L2 normalization.
Output is `[batch, 1024]`. Inputs are right-padded and include tokenizer special
tokens; queries receive no added instruction. Maximum length is 8192 tokens.
All linear biases, learned type-0 embeddings, padding-aware absolute positions,
LayerNorm epsilons, GELU and post-normalization residuals are preserved.

Removed: training/dropout, decoder/cross attention, KV cache, task heads,
output wrappers, alternate activations and configurable attention backends.
Dense inference only: sparse and ColBERT retrieval heads are outside this
experiment. The tanh pooler in the checkpoint is unused and explicitly dropped;
all remaining weight keys must match strictly.

The FP16 path is the correctness baseline for quantization experiments. It uses
explicit matmul/softmax attention. Optional ordinary static W8A8 lives in
`w8a8.py`; there is no custom norm+quant kernel.
Its attention memory grows quadratically with sequence length; supporting the
position range is not a claim of efficient 8192-token inference.

## Provenance

- Encoder adapted and simplified from Hugging Face Transformers
  [`modeling_xlm_roberta.py`](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py).
  Original Apache-2.0 attribution is retained in the model file.
- Dense pooling and input conventions:
  [official model card](https://huggingface.co/BAAI/bge-m3) and the release's
  `1_Pooling/config.json` (CLS pooling).
- Checkpoint revision: `5617a9f61b028005a4858fdac845db406aefb181`.
  `release.json` pins exact sizes and SHA-256/LFS or Git-blob hashes.
- The download utility follows experiment 25's range-download helper.

## Run

Use the existing Ascend environment, with one idle device selected. On the
current host, run through the persistent host SSH connection and
`docker exec research_vllm_ascend_023_external_workspace`; the old container SSH
forward is stale. Its shared `/workspace/models` mount is read-only, so use
`/workspace/model_downloads`. Download
weights once (about 2.3 GB), outside Git:

```bash
python 26_bge_m3_inference/download_model.py --output /workspace/model_downloads/bge-m3
python 26_bge_m3_inference/run_embedder.py \
  --model-dir /workspace/model_downloads/bge-m3 --max-length 128 \
  --texts 'What is BGE M3?' 'BGE M3 is a multilingual embedding model.'
```

Add `--compile-cache /absolute/path/to/a/fresh/cache` for static, full-graph
TorchAir execution. Use a different cache for each code/checkpoint/dtype revision.
Batch or sequence-shape changes need their own compiled graph.

## Validation

Reproducible accelerator launcher (selects an idle device and records command,
versions, timings, parity results and exit code):

```bash
RUN_ROOT=/workspace/results/bge_m3_UNIQUE_RUN bash 26_bge_m3_inference/run_910b.sh
```

Individual checks:

```bash
python -m unittest discover -s 26_bge_m3_inference -p test_reference.py -v
python 26_bge_m3_inference/validate_910b.py \
  --model-dir /workspace/model_downloads/bge-m3 \
  --compile-cache /absolute/path/to/run/cache --output /absolute/path/to/run/result.json
```

The small CPU test checks the independent implementation against a randomly
initialized Transformers encoder, checkpoint loading, normalization and padding
invariance. It is not NPU validation.

The accelerator script requires a 910B and verifies the real checkpoint hashes.
It checks local FP16 eager and compiled outputs against Transformers FP16 eager
on mixed-length multilingual B2/S128 and truncated long-text B1/S512 inputs.
Embedding gates: maximum absolute error <= 0.002 and cosine >= 0.9999; valid-token
hidden-state eager error <= 0.05; compiled unit-norm error <= 0.002. It exits
nonzero on failure and saves partial results. Two warmups and five repetitions
provide smoke timings, excluding loading and compilation. This is numerical
parity, not a retrieval-quality benchmark.

## Verified 910B2 smoke — 2026-10-10

Validated source commit `e1a5752f`, physical NPU 3, FP16, PyTorch 2.10.0,
torch-npu 2.10.0.post2 and Transformers 5.5.4. The pinned real checkpoint passed
all file hashes. Both local eager and static full-graph TorchAir runs passed.

| Input | Valid tokens | Eager median | Compiled median | Compiled/reference max abs | Minimum cosine |
|---|---|---:|---:|---:|---:|
| B2 / padded S128, multilingual | 9, 25 | 18.55 ms | 4.31 ms | 0.00024414 | 0.99999815 |
| B1 / S512, truncated long text | 512 | 19.03 ms | 5.82 ms | 0.00030518 | 0.99999642 |

Eager hidden states and dense embeddings matched the FP16 Transformers reference
exactly in both cases. Timings are synchronized forward-only medians of five
repetitions after two warmups; tokenization, checkpoint loading and first-call
compilation are excluded. These are small smoke measurements, not a throughput
sweep or retrieval-quality evaluation. 310P has not been tested.

The attention uses explicit three-dimensional `bmm` head batches: the original
four-dimensional broadcast matmul failed GE shape inference on this runtime.
This preserves the Transformers math (covered by CPU and real-NPU reference
checks). The validation harness resets Dynamo between distinct static shapes to
keep their saved graph caches independent.

Evidence: [result.json](../tmp/26_bge_m3_inference/910b_e1a5752f/result.json),
[exact command/environment](../tmp/26_bge_m3_inference/910b_e1a5752f/command.txt),
[exit code](../tmp/26_bge_m3_inference/910b_e1a5752f/exit_code.txt).

The standalone `run_embedder.py --compile-cache ...` command also passed in a
fresh process, reusing the saved B2/S128 graph with different text inputs;
see [CLI output](../tmp/26_bge_m3_inference/910b_e1a5752f/cli.log).

## Ordinary static W8A8

The projection path follows experiment 13's Qwen W8A8 implementation:
`npu_quantize(div_mode=True)` -> `npu_quant_matmul` -> FP16 output.
Weights use symmetric per-output-channel INT8 scales; activation scales are
fixed per input tensor after an FP16 calibration pass. Dequantization scales
are packed once before compilation. There are no calibration hooks, absmax
reductions or weight quantization in timed inference.

BGE's original FP16 linear biases are added after dequantization. Q/K/V share
one activation quantization and retain three separate INT8 matmuls. LayerNorm,
GELU, attention scores/softmax, residuals and embedding tables remain floating
point. This code does not explicitly request norm+quant fusion; ordinary GE
compiler optimizations remain enabled in every lane.

| Mode | INT8 linears | FP16 linears | Activation quantizations per layer |
|---|---:|---:|---:|
| `dense` | 0 | 144 | 0 |
| `ffn_w8a8` | 48 | 96 | 2 |
| `full_w8a8` | 144 | 0 | 4 |

Run one mode (use a separate compiled-cache directory for every mode and
calibration dataset):

```bash
python 26_bge_m3_inference/run_embedder.py \
  --model-dir /workspace/model_downloads/bge-m3 --weight-mode full_w8a8 \
  --compile-cache /workspace/results/bge_full_w8a8_UNIQUE/cache
```

`quantization_texts.json` contains 12 calibration paragraphs and eight disjoint
held-out texts. Calibration runs three batches of four padded to 256 tokens.
This is a deliberately small smoke dataset, not production calibration or a
retrieval benchmark. `--calibration-file` accepts another file with a
`calibration` string list.

Paired compiled comparison:

```bash
RUN_ROOT=/workspace/results/bge_w8a8_UNIQUE bash 26_bge_m3_inference/run_w8a8_910b.sh
```

All three lanes use FRACTAL_NZ for their remaining FP16 and INT8 linear weights,
with internal formats enabled. Compare to this freshly timed dense control,
not to the earlier native-format FP16 measurements above. The benchmark runs
B2/S128, B1/S512 and B4/S512; the latter two fill the sequence. It measures 20
paired repetitions after three warmups, rotating lane order. Setup and compile
costs are excluded. Separate code objects and directories isolate static caches.

The script first checks an INT8 linear with nonzero FP16 bias against a CPU
integer-matmul/dequantization reference. It reports compiled
versus quantized-eager drift for every model/shape, and gates finite normalized
outputs. The dense lane retains its strict reference-parity gate (max abs <=
0.002, cosine >= 0.9999). INT8 threshold crossings can amplify small upstream
floating-point differences, so quantized end-to-end agreement is measured, not
assumed to meet the dense tolerance. The isolated quantizer/linear diagnostic
checks identical-input operator semantics separately.

Two fresh query/document pairs also exercise the compiled graphs: bird migration
and compiler-versus-interpreter explanations. The report records their 2x2
similarity matrices and expected top documents. These simple sanity checks and
eight held-out embedding comparisons are not retrieval-quality evaluation.
A `passed` execution result does not mean retrieval quality has been preserved.

## Verified ordinary W8A8 on 910B2 — 2026-10-10

Source `71201c25`, physical NPU 3, the same pinned checkpoint and runtime versions
as above (compiler logs identify CANN 9.0.1). All nine static graphs compiled and
passed finite/unit-norm checks; the three dense graphs also passed strict parity.
The following are synchronized forward-only medians from 20 paired repetitions:

| Input | FP16 NZ | FFN W8A8 | Full W8A8 | Full speedup |
|---|---:|---:|---:|---:|
| B2 / padded S128 (9, 25 valid tokens) | 3.981 ms | 4.432 ms | 5.092 ms | 0.782x |
| B1 / full S512 | 5.601 ms | 6.341 ms | 6.749 ms | 0.830x |
| B4 / full S512 | 12.860 ms | 12.950 ms | 13.041 ms | 0.986x |

Ordinary W8A8 did not improve latency in this small shape sweep. Its overhead is
more visible on the smaller inputs; B4/S512 is approximately tied. Determining
which kernels dominate needs profiling. Compiler logs show ordinary AddLayerNorm
fusion, so this is already a compiler-optimized FP16 control.

The eight held-out eager embeddings have mean/minimum cosine to dense FP16 of
0.9597/0.9440 for FFN W8A8 and 0.9560/0.9401 for full W8A8. Across the three
compiled cases, full W8A8 mean cosine to dense ranges from 0.9436 to 0.9537.
These are measurable changes, with only 12 short calibration paragraphs.

Both compiled quantized modes chose the expected document for both fresh queries.
For full W8A8, the query/document cosine scores were:

| Query | Bird migration document | Compiler/interpreter document |
|---|---:|---:|
| How do migrating birds navigate over long distances? | **0.6709** | 0.3796 |
| What are the main differences between a compiler and an interpreter? | 0.3016 | **0.7898** |

This meets the experiment's basic semantic sanity criterion, not a retrieval
quality target. Norms, embeddings and attention remain floating point.

On identical captured inputs to the first FFN layer, the isolated quantizer and
biased W8A8 linear each match eager versus compiled **bit-for-bit**. The separate
integer-matmul reference check passes with maximum absolute error 0.001953125.
Full-network compiled/eager quantized mean cosine nevertheless ranges from
0.9756 to 0.9870 across the measured modes/shapes. Accumulated upstream
floating-point differences and INT8 threshold crossings are a plausible
explanation; these probes do not locate every source of divergence.

Evidence: [benchmark result](../tmp/26_bge_m3_inference/bge_m3_w8a8_71201c25/result.json),
[command/environment](../tmp/26_bge_m3_inference/bge_m3_w8a8_71201c25/command.txt),
[exit code](../tmp/26_bge_m3_inference/bge_m3_w8a8_71201c25/exit_code.txt),
[isolated operator diagnostic](../tmp/26_bge_m3_inference/bge_m3_w8a8_diagnostic_71201c25/result.json).
No 310P execution was performed.

The standalone `run_embedder.py --weight-mode full_w8a8` eager CLI also passed in
a fresh process, returning `[2, 1024]` embeddings with norms 0.999745 and
1.000014. See [CLI command](../tmp/26_bge_m3_inference/bge_m3_w8a8_71201c25/cli_command.txt)
and [output](../tmp/26_bge_m3_inference/bge_m3_w8a8_71201c25/cli.log).

## Device kernel profiling

```bash
PROFILE_KERNELS=1 RUN_ROOT=/workspace/results/bge_profile_UNIQUE \
  bash 26_bge_m3_inference/run_w8a8_910b.sh
```

This uses `torch_npu.profiler` (the Ascend PyTorch profiler) with CPU and NPU
activities, Level1 detail, shapes and PipeUtilization metrics. Each mode/shape
gets its own capture: three active synchronized forwards after graph compilation,
three ordinary warmups and one profiler warmup. Calibration, loading and
compilation are outside the capture. Unprofiled paired timings are recorded
before profiling each shape; profiler timings are diagnostic measurements.

`profiles/<case>_<mode>/` contains the original CANN kernel CSVs, Chrome trace,
profiler metadata and `summary.json`. Summaries reuse experiment 05's existing
model-agnostic parser. Counts and duration sums in the main result are normalized
by the three active forwards. Summed kernel durations can overlap and should not
be interpreted as end-to-end latency; the trace retains their timestamps.

### Captured 910B2 kernel results

Source `1ceb71db`, physical NPU 3. All nine captures completed with exit 0.
The timestamp audit assigns every kernel to exactly one of the three active
`ProfilerStep` ranges in each capture. Values below are mean summed device
kernel durations per forward in milliseconds, not profiler wall-clock latency.

| Kernel work | B2/S128 FP16 | B2/S128 full W8A8 | B1/S512 FP16 | B1/S512 full W8A8 | B4/S512 FP16 | B4/S512 full W8A8 |
|---|---:|---:|---:|---:|---:|---:|
| Projection MatMulV2 / QuantBatchMatmulV3 | 1.293 | 1.031 | 1.868 | 1.380 | 5.263 | 2.621 |
| Quantize | 0 | 0.750 | 0 | 0.813 | 0 | 1.358 |
| Separate projection-bias Add kernels | 0 | 0.554 | 0 | 0.627 | 0 | 0.944 |
| AddLayerNorm | 0.422 | 0.508 | 0.655 | 0.846 | 1.262 | 1.614 |
| TransData | 0.284 | 0.283 | 0.280 | 0.289 | 0.435 | 0.434 |
| All kernels | 3.809 | 4.937 | 5.469 | 6.605 | 12.644 | 12.701 |
| Kernel count | 524 | 716 | 500 | 692 | 524 | 716 |

Direct observations:

- The 144 dense projection kernels take bias as their third input. INT8
  QuantBatchMatmulV3 takes dequantization scales there; 72 Q/K/V biases and
  24 intermediate-FFN biases become separate FP16 Add kernels. Their input
  signatures are `[B*S,1024] + [1024]` and `[B*S,4096] + [4096]`.
- The other 48 projection biases are fused into the existing residual
  AddLayerNorm kernels. Their signatures gain a fifth `[1024]` input. These
  kernels take longer in W8A8; the extra bias work is consistent with this,
  although it was not isolated from possible kernel-tiling differences.
- Full W8A8 retains exactly 96 Quantize kernels: 72 on width-1024 inputs and
  24 on width-4096 inputs. Q/K/V already share one quantization per layer.
  No norm+quant fusion is present in these captured graphs.
- FFN-only W8A8 has 48 Quantize and 24 additional separate bias Add kernels,
  increasing the kernel count by 72. Full W8A8 increases it by 192.
- TransData counts and time are nearly unchanged. Median gaps between kernels
  inside each forward span are only about 12–17 microseconds, including both
  dense and quantized modes. Device work accounts for almost the entire span;
  large gaps between launches are not the observed explanation for the slowdown.

At B4/S512, INT8 saves 2.642 ms of projection matmul time, offset by 1.358 ms
quantization, 0.944 ms separate bias additions, 0.351 ms more AddLayerNorm time,
and other small differences. At B2/S128 the matmul saving is only 0.261 ms.
This explains why W8A8 loses on small shapes and approaches a tie at B4.
The fresh unprofiled medians are 4.096/4.709/5.369 ms at B2/S128,
5.721/6.561/7.005 ms at B1/S512, and 12.886/12.833/13.103 ms at B4/S512
(dense / FFN W8A8 / full W8A8).

The next optimization targets supported by these profiles are the explicit
quantization passes and the newly separate bias additions. Norm+quant fusion
addresses only part of that work; the 24 width-4096 quantizers follow GELU,
and Q/K/V and intermediate-FFN bias additions also need attention. Existing
softmax/attention/layout work remains substantial at B4 but affects both modes.
No fusion implementation was changed during profiling.

Compact evidence: [kernel comparison JSON](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/kernel_comparison.json),
[all types CSV](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/kernel_types.csv),
[all shapes CSV](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/kernel_shapes.csv),
[full run result](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/result.json),
[command/environment](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/command.txt).
The original traces/CSVs/CANN data are retained in the 11 MB archive
`bge-m3-910b-torch-profiler.tar.gz`, whose hash is recorded in
[artifacts.json](../tmp/26_bge_m3_inference/bge_m3_profile_1ceb71db/artifacts.json).
Each `trace_view.json` opens in a Chrome-trace-compatible viewer.

Reproduce the compact comparison from the extracted archive:

```bash
python 26_bge_m3_inference/summarize_w8a8_profiles.py \
  --run-root /path/to/bge_m3_profile_1ceb71db --output-dir /path/to/summary
```

## Standalone upstream AddLayerNormQuantV2

`test_add_layer_norm_quant_v2.py` is the single Python entry point for calling
and checking V2. It uses `ctypes` to pass PyTorch-owned NPU buffers to the
upstream ACLNN API. It does not build a PyTorch extension, register a TorchAir
converter, or change the model/linear implementation.

The operator is built unchanged from upstream `cann/ops-nn` tag `v9.1.0`, commit
`ceb4536a2bd6fc99b85aec9d0fdcc0f470376292`. Its kernels, tiler and API retain their
upstream names and implementations. The custom package is installed under
`/workspace/operators/bge-v2-9.1`, separate from CANN 9.0.1.

Build once in a clean upstream checkout after installing its build prerequisites:

```bash
source npu-setup
git clone --depth 1 --branch v9.1.0 https://gitcode.com/cann/ops-nn.git
cd ops-nn
bash build.sh --pkg --soc=ascend910b --ops=add_layer_norm_quant_v2 \
  --no_force --vendor_name=bge_v2 -j8
bash build_out/cann-ops-nn-bge_v2_linux-aarch64.run \
  --quiet --install-path=/workspace/operators/bge-v2-9.1
```

`--no_force` skips recompiling dependency kernels already supplied by CANN; it
does not specialize or modify the V2 source. The upstream build retains V2's
FP16, BF16 and FP32 variants. This test exercises FP16 static quantization only.

From this research repository, run:

```bash
source npu-setup
source /workspace/operators/bge-v2-9.1/vendors/bge_v2_nn/bin/set_env.bash
python3 26_bge_m3_inference/test_add_layer_norm_quant_v2.py \
  --op-api /workspace/operators/bge-v2-9.1/vendors/bge_v2_nn/op_api/lib/libcust_opapi.so \
  --output /workspace/results/bge_v2_UNIQUE
```

The output directory must be fresh. Optional `--rows` and `--width` change the
input shape. The script checks FP16 normalized and INT8 quantized outputs with
and without bias, then captures three warmed calls using the NPU profiler.
It uses multiplication by an FP16 scale, matching V2's documented static mode.
The reference adds and normalizes in FP32; FP16 outputs use atol=0.004,
rtol=0.002, and quantized values may differ by at most one INT8 level.

For this newer package on CANN 9.0.1, the script loads both stock and package
`libes_nn.so` before initializing the NPU backend. This supplies the package's
additional ES symbols while keeping the stock library loaded. No kernel or
installed CANN files are patched.

Validated on physical **910B2 NPU 3**, script commit `f7214189`, using the
unchanged released V2 package. Both shapes passed with and without bias:

| FP16 input shape | Maximum norm error vs FP32 reference (both cases) | INT8 exact agreement | Profiled V2 kernel median |
|---|---:|---:|---:|
| 256 x 1024 | 0.00191832 | 100% | 13.520 us |
| 7 x 768 | 0.00153017 | 100% | 6.940 us |

Profiles use bias and contain exactly three V2 kernels for three calls: one
fused kernel per call, including the FP16 normalized and INT8 quantized outputs.
These three-sample device timings are smoke observations, not a stable
performance comparison. No model or TorchAir integration was performed.

Evidence: [256x1024 result](../tmp/26_bge_m3_inference/v2_f7214189/shape256x1024.json),
[7x768 result](../tmp/26_bge_m3_inference/v2_f7214189/shape7x768.json),
[raw kernel rows](../tmp/26_bge_m3_inference/v2_f7214189/shape256x1024_kernels.csv),
[commands](../tmp/26_bge_m3_inference/v2_f7214189/command.txt),
[build/source provenance](../tmp/26_bge_m3_inference/v2_f7214189/build_provenance.txt).

## Full-W8A8 V2 integration experiment

`fused_norm_v2.py` adds a PyTorch custom-op identity and a TorchAir converter for
unchanged upstream `AddLayerNormQuantV2`. `FusedBGEM3` shares the regular model's
weights and calibrated scales. Its 48 fused boundaries are the embedding norm,
24 attention-output norms, and 23 FFN-output norms. The last FFN norm has no
quantized consumer. Each fusion preserves the FP16 normalized residual and feeds
INT8 directly into the next projection, removing its redundant quantizer.
Projection bias enters V2, matching the baseline GE bias/AddLayerNorm fusion.
The 24 post-GELU and 24 attention-context quantizers remain.

The baseline's division scale and all matmul dequantization scales are retained.
V2 receives the reciprocal rounded to FP16, as its static multiplication mode
requires. This and fused arithmetic can change rounding; numerical comparisons
must distinguish isolated same-input behavior from accumulated model drift.

Two graph-integration details matter on the test host's CANN 9.0.1:

* The release package registers dependency operators too. A private copy keeps
  only V2 kernel dispatch enabled, so baseline AddLayerNorm remains stock.
* The release lacks a V2 graph shape/datatype callback. `v2_graph_infer.cpp`
  supplies metadata for this static, single-scale experiment only. It does not
  modify the kernel, tiler or ACLNN API, and rejects dynamic/second-scale use.

Build the private package once, preserving the vendor basename:

```bash
source npu-setup
python3 26_bge_m3_inference/prepare_v2_graph_package.py \
  --vendor /workspace/operators/bge-v2-9.1/vendors/bge_v2_nn \
  --output /workspace/operators/bge-v2-graph-UNIQUE/vendors/bge_v2_nn
export ASCEND_CUSTOM_OPP_PATH=/workspace/operators/bge-v2-graph-UNIQUE/vendors/bge_v2_nn
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
python3 26_bge_m3_inference/benchmark_norm_v2.py \
  --op-api "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" \
  --output /workspace/results/bge_norm_v2_UNIQUE
```

The harness verifies checkpoint hashes, calibrates once, captures real early,
middle and final attention-norm inputs, checks isolated eager/compiled outputs,
then compares full-model embeddings and two semantic examples. Profiles cover
B2/S128, B1/S512 and B4/S512, with 10 forwards per capture and two captures per
lane in opposite order. `summarize_norm_v2.py` audits every kernel against actual
profiler-step intervals before aggregating counts, shapes and durations.
