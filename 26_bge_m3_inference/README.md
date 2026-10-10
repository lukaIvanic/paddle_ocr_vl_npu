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
