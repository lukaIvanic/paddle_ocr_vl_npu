# RWKV validation plan

Agreed sequence, 2026-10-06. Start with correctness and accuracy on Ascend 910B2,
then decide whether larger evaluations need faster inference. This document is
the experiment's single working document, including research findings, source
provenance and reference scores. No weights have been loaded and no model runs
have been completed yet; all reference measurements below are upstream reports.

## 0. Establish a server CPU reference

Use the selected v0.23 container for an explicitly labeled CPU FP32 run of the
0.1B embedding model before the NPU adaptation. Start with a few fixed samples,
including English and Chinese and a sample crossing an EOS chunk boundary.
Save exact input token IDs, preprocessing/pooling metadata, intermediate
recurrent states and final embeddings for comparison with the NPU path.
Inspect the upstream CPU execution path before selecting the reference runner;
the C implementation provides an additional embedding reference but does not
replace the need for intermediate-state comparisons. Keep RWKV dependencies in
a separate environment. No CPU inference has been run yet.

## 1. Adapt the 0.1B embedding model to NPU

- Pin the released checkpoint revision and hashes, tokenizer and upstream code.
  Preserve the evaluation wrapper's instruction, EOS, padding, pooling, context
  cap and title/text behavior. Keep the reference MTEB version and dataset
  revisions explicit.
- Implement a correctness-first NPU path. Begin with a few fixed texts and
  compare token IDs, intermediate recurrent states and final embeddings against
  an independent reference. Record actual execution devices and precision;
  do not silently fall back to CPU. An explicitly labeled CPU numerical oracle
  is separate from the NPU run.
- Evaluate the complete **NanoSCIDOCS** task: 2,210 documents and 50 queries.
  Compare NDCG@10 with the release references, 40.988 for GPU BF16 and 41.058
  for C CPU FP32. Record the precision difference and exact score delta.
- Once that result is understood, evaluate **all 13 NanoBEIR tasks**. Save
  per-task NDCG@10 and the macro mean, comparing with the paper's 59.10 and the
  release's 58.973692 BF16 reproduction.
- Save the 0.1B retriever's top-100 document IDs and scores per query, with a
  manifest of dataset revisions and preprocessing. Reuse these candidates for
  the upstream reranker reproduction.

A substantial unexplained numerical or quality discrepancy is investigated
before expanding the run. Do not select a tolerance after seeing the results.

Start NanoBEIR evaluation on one NPU. If it is not fast enough, use **data
parallelism**, with a complete model replica on each participating NPU and
inputs distributed across replicas. Verify identical query/candidate coverage
and numerically consistent outputs before combining results. Apply this approach
to embedding and reranker evaluations; no model sharding is planned.

## 2. Adapt and verify the rerankers, one size at a time

Order: **90M, then 317M, then 1.3B**. The latter two also require adapting their
matching **0.4B and 1.4B embedding/state backbones**. Candidate retrieval remains
the saved **0.1B** retriever output for all three, as in the paper.

For each released pair:

1. Check a few query/document pairs against the reference computation, including
   intermediate states and final scores. Preserve the exact document/query
   boundaries and reranker preprocessing. Verify batching does not mix states.
2. Rerank the saved top-100 candidates for all **NanoSCIDOCS** queries and record
   NDCG@10 alongside the embedding baseline. This is an early accuracy check;
   no published per-task reranker score has been identified.
3. Evaluate **full NanoBEIR** on the same saved candidates. Compare per-task
   results and the macro mean; published aggregate targets are **63.41**, **68.60**
   and **71.58**, respectively. Resolve unexplained differences before moving
   to the next size or larger suites.

If document states are reused, check that reused-state scores match fresh-state
scores within a justified numerical tolerance. This is a correctness check;
storage/serving optimization comes later if needed.

## 3. Estimate full C-MTEB and English MTEB time for the largest reranker

Only the **1.3B reranker with its 1.4B state backbone** proceeds to this phase.
All three rerankers are verified on NanoSCIDOCS and full NanoBEIR in phase 2;
the later large-suite evaluation and one-hour decision apply only to the largest.

Use the **already saved Qwen3-Embedding-0.6B top-100 candidates per query** from
experiment 22. This phase compares RWKV with Qwen3-Reranker-4B on identical
candidate sets. Do not retrieve a new candidate set with RWKV for this comparison.

Scope is the same retrieval evaluation used for Qwen: the **eight C-MTEB-R
tasks** and **ten English MTEB(eng, v2) retrieval tasks**, with the existing pinned
revisions, splits, judgments and task aggregation. It is not a new evaluation of
every classification, clustering or other task in MTEB.

First audit saved candidate coverage, document IDs/texts, query counts and file
hashes. Locate the actual saved Qwen baseline results. Any missing candidate
artifact is recorded explicitly before estimating or claiming a full run.
Reuse Qwen's dataset/metric contract while applying RWKV's own verified model
preprocessing; do not feed the Qwen tokenized prompt into RWKV.

For the validated largest pair, time a small representative sample spanning
tasks and document/query lengths, including long inputs. Begin with one NPU;
use data parallelism if needed. Measure on the actual available healthy NPUs
and verify that distributing inputs across complete model replicas preserves
scores and coverage relative to a single-device run.

Estimate total wall time from measured rates and full pair counts, accounting
for startup, data loading/tokenization, document-state preparation and transfer,
reranking, output writing and metric aggregation. If caching is used, count the
initial state build and report it separately; do not use warm-cache query timing
alone to estimate the first full evaluation. Prefer a measured multi-device
sample over assuming perfectly linear scaling.

**Decision rule: less than one hour for the combined C-MTEB-R + English MTEB-R
evaluation of the largest reranker pair, using the available NPUs.**

- If the estimate, including reasonable measurement uncertainty, is below
  **3,600 seconds**, run both full suites for the largest pair. This is authorized by
  Luka's plan; another routine confirmation is unnecessary.
- Otherwise, improve speed through **document-state caching and/or inference
  optimization**, check numerical and accuracy parity, and repeat the estimate.
  Do not launch the over-budget full evaluation first.

Record the estimate and decision for the largest pair. Preserve completed pair
scores so an interrupted run can resume without changing candidates or
double-counting results. Verify every expected query/candidate is scored before
reporting full-suite macro NDCG@10 and its difference from the saved Qwen result.

## Boundaries and evidence

Proceed one verified step at a time. This planning request does not start model
downloads, installations, ports or benchmark runs. There is no training or
distillation in this plan. Luka considers training a last resort and requires
explicit approval from his higher-ups before it can be considered. Qwen
prompt-order testing remains Luka's separate work.

Follow the parent repository's local-authoring and pull-only NPU validation
lanes. Each actual run records the exact command, source/checkpoint/dataset
revisions, host/chip/device, dependencies, dtypes, exit code, log and numerical
or benchmark outputs under `tmp/26_rwkv_inference/<run_name>_<commit>/`.
Distinguish upstream reference scores, our smoke checks, our numerical parity
checks and our complete benchmark results.

## Sources and provenance

- [Paper: EmbeddingRWKV: State-Centric Retrieval with Reusable States](https://arxiv.org/abs/2601.07861), version 1.
- [Official source](https://github.com/howard-hou/EmbeddingRWKV), inspected at
  `3c306736c58550f4be6d384be068512ba9bfbd72`.
- [Released checkpoints](https://huggingface.co/howard-hou/EmbeddingRWKV/tree/main).
  The checkpoint repository revision and individual file hashes must be recorded
  when obtaining weights; they have not been pinned here yet.
- [Evaluation implementation](https://github.com/howard-hou/EmbeddingRWKV/tree/3c306736c58550f4be6d384be068512ba9bfbd72/embedding/eval).
- [CPU embedding implementation and reproduction report](https://github.com/howard-hou/EmbeddingRWKV/blob/3c306736c58550f4be6d384be068512ba9bfbd72/rwkv-emb.c/REPRODUCTION.md).
- [NanoBEIR datasets](https://huggingface.co/collections/zeta-alpha-ai/nanobeir).
- Existing Qwen evaluation contracts: experiment 22's
  [Chinese retrieval protocol](../22_qwen3_embedding_benchmark/protocol.py) and
  [English retrieval protocol](../22_qwen3_embedding_benchmark/suite_protocol.py).
  These pin MTEB 1.38.9 and the dataset membership/revisions for the later
  comparison; RWKV's upstream reproduction uses MTEB 1.38.60 separately.

The inspected source repository has an Apache-2.0 license. Record the checkpoint
license metadata separately with the downloaded revision. Keep weights and full
reference checkouts in ignored model/cache locations, outside experiment source.

## Released model pairs and accuracy references

Scores are NDCG@10 multiplied by 100. The reranker sizes refer to the reranking
network; each also requires its matching embedding/state backbone.

| Pair | Embedding checkpoint | Reranker checkpoint | Full NanoBEIR reranking target |
| --- | --- | --- | ---: |
| Tiny: 0.1B backbone + 90M reranker | `rwkv0b1-emb-curriculum.pth` | `rwkv0b1-reranker.pth` | 63.41 |
| Base: 0.4B backbone + 317M reranker | `rwkv0b4-emb-curriculum.pth` | `rwkv0b3-reranker.pth` | 68.60 |
| Large: 1.4B backbone + 1.3B reranker | `rwkv1b4-emb-curriculum.pth` | `rwkv1b3-reranker.pth` | 71.58 |

The smallest embedding model has a paper full-NanoBEIR target of **59.10**.
The release reports a GPU BF16 reproduction of **58.973692** and a C CPU FP32
reproduction of **58.907231**. On NanoSCIDOCS alone, those release references are
**40.988** (GPU BF16) and **41.058** (CPU FP32). No per-dataset accuracy reference
for the three rerankers has been identified; reproducing their aggregate targets
requires all 13 NanoBEIR tasks.

Full NanoBEIR contains 56,723 corpus documents and 649 queries. Top-100 reranking
therefore involves approximately 64,900 pairs per model. The paper's reranking
evaluation retrieves those candidates with the **0.1B embedding model for every
reranker size**, while reranker execution uses each model's matching backbone.
Save and reuse that one candidate set when reproducing the three reported scores.

## Protocol details to preserve

The release's smallest-embedding reproduction uses **MTEB 1.38.60**, batch size 4,
context cap 2,048, EOS chunk size 512, the original generic query instruction,
RETR head and original title/text joining. Its preprocessing includes short-text
repetition, valid-EOS masking, left zero padding and 261 alignment padding.
Preserve the wrapper's actual behavior, including batch effects, rather than
replacing it with a simplified quick-start tokenizer call.

Keep EOS token **65535**, padding, instruction placement, pooling and context
handling explicit. Save dataset revisions/splits, exact candidate ordering,
score aggregation and all preprocessing settings. Do not transfer the embedding
reproduction's settings to reranking without checking the reranker wrapper.
The supplied `run_mteb_rerank.sh` has inconsistent usage/argument parsing; it
requires inspection before use and is not a verified run command for us.

## Execution findings and reported throughput

Selected container: **`research_vllm_ascend_023_external_workspace`** on
`liteserver-c001-4`, verified running on 2026-10-06. Read-only inspection found:

- Architecture: `aarch64`; compiler: GCC 11.4.0.
- Python: `/usr/local/python3.12.13/bin/python3`, version 3.12.13.
- Installed package metadata: torch `2.10.0+cpu`, torch-npu `2.10.0.post2`,
  vLLM `0.23.0+empty`, vLLM-Ascend `0.23.0rc1`.
- Project checkout: `/workspace/repos/paddle_ocr_vl_npu`.
- A separate RWKV environment has not been created yet. Package metadata alone
  does not validate CPU or NPU execution.

The current local session reaches the server through the existing task-specific
SSH configuration, not the default `~/.ssh/config`:

```bash
ssh -F /home/luka/Documents/Codex/2026-10-01/can-you-connect-to-my-mac/work/ssh-blue-zone/config \
  -o ControlPath=/tmp/codex-blue-zone-1000/rwkv-023-master \
  blue_zone_npu_server \
  'docker exec research_vllm_ascend_023_external_workspace hostname'
```

This route was tested successfully. A dedicated multiplexed master was started
and verified on 2026-10-06 at the control socket above, with `ControlPersist=yes`
(indefinite idle persistence), `ServerAliveInterval=30` and
`ServerAliveCountMax=3`. Reuse it for subsequent commands. The pre-existing shared
master remains unchanged; its configuration uses 600-second idle persistence.
The dedicated connection can be re-established with the same SSH config/socket
and `-M -N -f` if it disconnects.

Run workloads inside the selected container;
preserve the parent source-through-Git lane and use `source npu-setup` for NPU
execution. Container inspection did not change packages or launch inference.

The upstream GPU implementation uses custom CUDA kernels; a working Ascend path
has not been established. There is currently no CUDA validation lane. The
released C runtime implements **embeddings only**, not the state reranker. It
can provide an explicitly labeled CPU reference; it is not an NPU fallback.

The [paper's Table 8](https://arxiv.org/html/2601.07861v1#A2.T8) reports the
following GPU results at document length 1,024, query length 64 and batch size
100. The paper does not identify the GPU SKU. These are reported measurements,
not our results or expected Ascend performance.

| Reranker + backbone | Uncached pairs/s | Cached pairs/s | Cached total time for 100 pairs |
| --- | ---: | ---: | ---: |
| 90M + 0.1B | 300.1 | 2,440.0 | 40.98 ms |
| 317M + 0.4B | 109.4 | 1,062.7 | 94.10 ms |
| 1.3B + 1.4B | 41.8 | 539.2 | 185.46 ms |

Their cached mode is labeled "Offline": document states are precomputed. The
reported total includes the remaining query-backbone computation and reranker
head. Inclusion of storage-to-device state fetching was not verified.

The release's CPU reproduction uses the 0.1B embedding model, a **Ryzen 7
9700X**, **16 OpenMP threads**, pure C **FP32**, and no BLAS, PyTorch or GPU
inference. It encoded **57,372 texts in 204.74 inference minutes** (about 4.67
texts/s); the full evaluation took **207.24 minutes**. Full NanoBEIR NDCG@10
was **58.907231**, versus the release's GPU BF16 **58.973692**. No CPU reranker
throughput report was found.

For workload context, our experiment-22 C-MTEB-R run covers **828,252 corpus
documents**, **39,740 queries**, and **3,974,000 top-100 pairs**. Full NanoBEIR
has about **14.6 times fewer documents**, **61.2 times fewer reranking pairs**,
and **15.1 times fewer document-plus-query texts to embed**. These are workload
count ratios, not runtime predictions. NanoBEIR is English; C-MTEB is Chinese,
so their absolute accuracy scores are not interchangeable.
