# Qwen3 ranking distillation pilots

## Contents-swapped follow-up

The follow-up keeps the same dataset, candidate order, instruction, system prefix,
assistant suffix and query-first 4B teacher targets. The student alone receives
`<Query>: document content` followed by `<Document>: query content`. It starts
from the released 0.6B weights with fresh optimizer state, using the same
50-update warmup/linear-decay schedule, peak LR 1e-6 and evaluation cadence.

Record the untrained swapped baseline and compare all updates against the original
query-first baseline saved in the completed pilot. Revalidate the canonical
query-first token hashes against the teacher, retain separate swapped hashes,
and reject any teacher or student truncation for this pilot. Run the real-input
backward/replay/optimizer control with swapped prompts first.

```bash
bash 13_qwen3_reranker/run_margin_distillation_experiment.sh \
  /workspace/results/qwen_margin_distill/data_fast/dataset.json.gz \
  /workspace/results/qwen_margin_distill/npu/contents_swap_RUN_COMMIT \
  contents_swapped /workspace/results/qwen_margin_distill/npu/d66b8e89
```

The query-first results below are the accuracy reference; swapped results must
come from the follow-up's own recorded artifacts. This run does not benchmark
cached-document inference.

## Query-first pilot

This experiment adapts Qwen3-Reranker-0.6B to a frozen Qwen3-Reranker-4B's
within-query score differences. Document-first adaptation is a later experiment.
The real-input implementation control and both 50-update training arms completed
on an Ascend 910B2 on 2026-10-07. The numerical evidence is in the
[run report and artifacts](../tmp/13_qwen3_reranker/margin_distillation_d66b8e89_20261007/README.md).

## Completed pilot

Frequent evaluation uses six fixed queries per task across 10 English and eight
Chinese retrieval tasks. Scores below are percentage NDCG@10, averaged equally
across tasks; teacher agreement uses 64 separate mixture queries.

| Model / endpoint | English | Chinese | Teacher pairwise agreement | Teacher margin MSE |
|---|---:|---:|---:|---:|
| Original 0.6B | 69.646 | 76.898 | 81.859% | 6.3827 |
| Constant LR, update 50 | 70.566 | 76.921 | 84.295% | 4.1122 |
| Warmup / decay, update 50 | 69.931 | 77.231 | 83.175% | 4.4083 |
| Frozen 4B teacher | 72.998 | 78.260 | — | — |

Both arms reduced held-out teacher margin error while preserving the sampled
suite means. The separate four-query-per-task endpoint panel and per-task
regressions are reported in the linked evidence. These small samples do not
establish full-suite parity, and no document-first model was trained.

Training took approximately 18 seconds per update, with 256 pairs per update.
Frequent evaluation took about 72–74 seconds; the additional baseline/endpoint
panel took about 45 seconds. Saving a checkpoint added approximately 17 seconds.
Every evaluated nonzero checkpoint remains on the server with optimizer and RNG
state; large weight files are excluded from the committed artifacts.

## Inputs and separation

Prepare 1,600 training query groups (800 English, 800 Chinese), each with eight
documents. English sources cycle NQ, SQuAD and MIRACL_en; Chinese uses MIRACL_zh.
This is a passage-retrieval pilot, not a representative mixture of every retrieval
subtype. Use the pinned format-repacked BGE-M3 mirror, retaining its provenance.
Choose the first supplied positive and up to seven supplied negatives; supplement
short candidate lists using character-TFIDF retrieval from a split-specific,
source-specific document bank. Teacher scores replace binary labels.

Reserve 64 disjoint mixture queries for teacher agreement. Reject shared normalized
query/document texts between training and this validation set. Exclude all query
texts from the pinned benchmark suites and candidate texts from both sampled
benchmark panels. Source-family exclusions follow `mixture_data.EXCLUDED`.
These checks are not exhaustive corpus-level or semantic decontamination.

Use six fixed queries per task for frequent evaluation and four additional fixed
queries per task for baseline/endpoint evaluation. Cover all 10 English v2 and
eight Chinese v1 retrieval tasks pinned in experiment 22 (MTEB 1.38.9). Reuse the
original Qwen3-Embedding-0.6B top-100 candidates; never inject judged positives.
Score against human qrels. Report separate unweighted English/Chinese task means,
per-task NDCG@10 and hole@10. Small samples are diagnostics, not full-suite scores.

## Objective and updates

Each query has scores s and teacher scores t, both raw yes-minus-no logits. Minimize
the mean of `((s_i-s_j)-(t_i-t_j))**2` over all 28 unordered candidate pairs,
then average equally across 32 queries per optimizer update (256 model inputs).
Save frozen teacher targets once. Assert teacher/student token IDs and truncation
are identical using per-section token hashes.

Use deterministic score-gradient replay to bound activation memory: first obtain
eight scores without autograd, analytically compute their loss gradients, then
replay each microbatch with autograd and accumulate its weighted score gradient.
Do not update parameters until all 32 query groups have contributed. A maximum
of four sequences and 8,192 padded token positions governs each train microbatch.
Left pad to its longest sequence rounded up to 128. Right truncate the prompt body
to fit 8,192 total tokens, preserving the official prefix and suffix.

Train every parameter with FP32 weights and BF16 autocast, using the owned model's
`npu_fusion_attention` path. Construct a fresh NpuFusedAdamW with betas 0.9/0.999,
epsilon 1e-8 and zero weight decay; clip the accumulated gradient norm to 1.0.
Compare constant 1e-6 against five-step warmup to 1e-6 and subsequent linear decay
(last update nonzero, zero reached after the run). Both arms start from the same
released weights and consume the same groups in the same order, without repeats.

Evaluate at 0, 1, 3, 10, 25 and 50. The frequent benchmark+teacher-validation round must fit
three minutes before training starts. Each arm has a 2,400-second wall-time limit.
Save model, optimizer, scheduler description, RNG states and training order at
every evaluated nonzero update. Keep large checkpoints on server storage.

## Validation and execution

`tests/test_margin_distillation.py` checks all-pairs loss/gradient algebra,
offset invariance, microbatch replay and LR scheduling. The real-input NPU control
compares HF eager, owned eager, fusion attention, replay and ordinary/fused AdamW.
Retain numeric differences; BF16 arithmetic is not generally bitwise equivalent.
Training requires the control artifact to pass.

After preparing `dataset.json.gz`, execute the committed pipeline on the 910B2:

```bash
bash 13_qwen3_reranker/run_margin_distillation_experiment.sh \
  /workspace/results/qwen_margin_distill/data_fast/dataset.json.gz \
  /workspace/results/qwen_margin_distill/npu/RUN_COMMIT
```

It selects a free device with `npu-setup` for each stage and records the physical
card, commit, exact command, log and exit status. Never edit tracked source on the
container. Success requires both movement toward held-out teacher rankings and
benchmark preservation; flat scores alone can mean negligible learning.
