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

This is the correctness baseline for later norm+quant experiments. It uses
explicit matmul/softmax attention and has no W8A8 or custom fused kernels yet.
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
