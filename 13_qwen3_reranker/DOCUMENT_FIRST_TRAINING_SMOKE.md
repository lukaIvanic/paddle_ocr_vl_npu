Qwen3-Reranker-0.6B recovered document-first ranking on a small held-out
PubMedQA sample after ten full fine-tuning updates on Ascend 910B2. The run
took 59.18 seconds inside the Python runner, including data preparation,
model loading, evaluation, and training. This establishes a fast training
smoke route; it does not establish benchmark parity or Qwen3-Reranker-4B
quality.

The validation ran on physical NPU 7 in
`research_vllm_ascend_023_external_workspace`, host `liteserver-c001-4`, using
torch 2.10.0+cpu, torch_npu 2.10.0.post2 and Transformers 5.5.4. The exact
source commit and invocation are in
[command.txt](../tmp/13_qwen3_reranker/train_918ab862_20261006/command.txt).
Source was authored locally, committed and pushed, then pulled through a Git
bundle into an isolated server worktree. No tracked source was edited on the
server. The deployed commit is `918ab8629c3152dc7c2e24390f0f7650ad7b5676` on
`codex/qwen-document-first-training-smoke`; the same changes are also on
`codex/qwen-document-first-probe`.

The prompt keeps the original system prefix, task instruction, assistant
suffix and yes/no scoring. It moves each field's label together with its
content:

```text
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Document>: document text
<Query>: query text
```

All examples fit the 1,024-token limit in both orders; no truncation is used.
The model processes fixed shapes of four rows by 1,024 tokens, with left
padding and a causal mask. Actual non-padding lengths are much shorter: the
ten updates contain 49,652 real tokens across 320 pairs, averaging 155.2
tokens per pair. This is not evidence for long-document training throughput.

The data comes from the `pubmed_qa_labeled_len-0-500` component of the
[BGE-M3 training mixture](https://huggingface.co/datasets/Shitao/bge-m3-data),
acquired through a
[format-only mirror](https://huggingface.co/datasets/hotchpotch/bge-m3-data-finetune-unified)
at revision `b51dd24cbce7d89255911410ef74a36e07bfbab9`. The mirror's claim of
unchanged text was not independently checked against the original 24 GB
archive. The downloaded source SHA-256 is
`74450120e5711e69f3c821d66337d73ead8b6176d595b0435f5ba7acb7921049`.
The original acquisition script is saved with the evidence as
[acquire_source.py](../tmp/13_qwen3_reranker/train_918ab862_20261006/acquire_source.py).

The source allowlist contains only PubMedQA, excluding the source families
in Qwen's published English MTEB-R v2 and Chinese CMTEB-R v1 contracts.
This is source-family exclusion, not exhaustive text-level decontamination
against every benchmark. We also exclude normalized held-out query and
document texts from training. The saved
[inputs.json](../tmp/13_qwen3_reranker/train_918ab862_20261006/inputs.json)
records the selected texts, source row IDs, provenance and split checks.

From the 500 available source queries, seed 731 selects 32 held-out queries
and 256 disjoint training queries. Evaluation uses one positive and three
dataset-provided negatives per query: 128 pairs and 96 positive/negative
comparisons. Training selects one positive and one negative per query,
shuffles groups with seed 732, and consumes the first 160 training queries
over ten updates. Evaluation labels are the mixture's labels, not independent
new judgments. Tied negatives rank ahead of the positive, avoiding bias from
the positive's first position in each saved group.

| Model state and prompt | Positive ranked first | Strict correct comparisons | MRR | NDCG |
|---|---:|---:|---:|---:|
| Original Transformers, query-first | 32/32 | 96/96 | 1.0000 | 1.0000 |
| Original local model, query-first | 32/32 | 96/96 | 1.0000 | 1.0000 |
| Original local model, document-first | 17/32 | 71/96 | 0.7188 | 0.7899 |
| After 1 update, document-first | 20/32 | 79/96 | 0.7917 | 0.8450 |
| After 5 updates, document-first | 31/32 | 95/96 | 0.9844 | 0.9885 |
| After 10 updates, document-first | 32/32 | 96/96 | 1.0000 | 1.0000 |
| After 10 updates, query-first | 32/32 | 96/96 | 1.0000 | 1.0000 |

The original local and Transformers query-first evaluations agree on these
ranking metrics. Their BF16 logits are not identical: maximum yes/no logit
difference is 0.5 and maximum yes-minus-no margin difference is 0.375. The
local path uses the owned model and differentiable `npu_fusion_attention`
with native GQA. An earlier synthetic forward/backward feasibility probe
found finite gradients and close gradient directions, with some gradient
magnitude differences; it was not a full numerical equivalence proof.

The optimizer is `NpuFusedAdamW`, with FP32 master parameters and optimizer
states, BF16 autocast, learning rate 1e-5, betas (0.9, 0.999), epsilon 1e-8,
zero weight decay, and gradient norm clipping at 1.0. Four pairs per
microbatch and eight accumulated microbatches give 32 pairs per update.
The loss is final-position cross entropy over the no/yes logits. The LM
head computes only those two differentiable weight rows, avoiding a full
vocabulary projection during the local training path. There is no gradient
checkpointing or TorchAir graph compilation.

All ten updates had finite gradient norms and changed the sampled Q-projection
parameters. Their combined measured time was 19.17 seconds. The first update
took 4.14 seconds, including optimizer initialization; updates 2–10 averaged
1.67 seconds. Peak allocated NPU memory was 20.47 GiB and peak reserved memory
was 31.45 GiB. The complete runner took 59.18 seconds; this excludes SSH
transfer time, environment sourcing, Python imports before its internal timer,
and process shutdown.

Full results and every evaluation's raw no/yes logits are in
[result.json](../tmp/13_qwen3_reranker/train_918ab862_20261006/result.json),
with [run.log](../tmp/13_qwen3_reranker/train_918ab862_20261006/run.log) and
[exit_code.txt](../tmp/13_qwen3_reranker/train_918ab862_20261006/exit_code.txt).
The preliminary three-update
[timing pilot](../tmp/13_qwen3_reranker/step_timing_918ab862_20261006/result.json)
also passed. The initial pilot failed before an update because the fused
optimizer rejects `zero_grad(set_to_none=True)`; its
[failure log](../tmp/13_qwen3_reranker/step_timing_dcbd841c_20261006/run.log)
is retained. The corrected runner uses `set_to_none=False`. Five local
data/prompt/metric tests pass, including disjointness and tie handling.

No model or optimizer checkpoints were saved. The trained in-memory model
was released when the process exited. The small, single-source, four-candidate
evaluation reached a ceiling for the original model, so broader held-out
reranking evaluation is needed to assess generalization. Cached-document
inference equivalence and speed were not tested in this training smoke.
