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

The follow-up completed all 50 updates on Ascend 910B2, physical NPU 3, on
2026-10-07. The real-input backward/replay/optimizer control passed. The
[completed report and artifacts](../tmp/13_qwen3_reranker/contents_swap_139da06f_20261007/README.md)
retain the full comparison, per-task scores and all evaluation score arrays.

Frequent-panel scores are percentage NDCG@10 on six queries per task:

| Model | English | Chinese | Held-out teacher agreement | Teacher margin MSE |
|---|---:|---:|---:|---:|
| Original query-first 0.6B | 69.646 | 76.898 | 81.859% | 6.3827 |
| Untrained contents swap | 54.626 | 64.088 | 70.241% | 15.0516 |
| Contents swap, 50 updates | 55.358 | 65.575 | 75.364% | 8.4936 |

Teacher error decreased by 43.57%, but benchmark recovery was limited and did
not restore the original query-first accuracy. Across the frequent and reserved
panels combined (10 queries/task), original versus final scores were
71.179 versus 56.690 for English and 73.225 versus 66.285 for Chinese. These
are small diagnostic samples, not full-suite results.

Training averaged 17.29 seconds/update; frequent evaluation took 72.34 seconds.
The process completed in 25.87 minutes including preparation, evaluations and
saves, with 22.04 GiB peak allocated NPU memory. All five evaluated nonzero
checkpoints remain on the server, including model/optimizer/RNG state. This run
does not benchmark cached-document inference.

## Higher-learning-rate contents swap

The matched arm restarted from the same released 0.6B weights and fresh
optimizer, retaining the data order, query-first 4B targets, contents-swapped
prompt, 50 updates and five-step warmup/linear decay. Only the peak learning rate
changes from 1e-6 to 1e-5. Evaluate at 0, 1, 3, 10, 25 and 50; compare against
both the original query-first reference and the completed 1e-6 swap.

Pass the peak learning rate as the optional fifth launcher argument:

```bash
bash 13_qwen3_reranker/run_margin_distillation_experiment.sh \
  /workspace/results/qwen_margin_distill/data_fast/dataset.json.gz \
  /workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_RUN_COMMIT \
  contents_swapped /workspace/results/qwen_margin_distill/npu/d66b8e89 1e-5
```

The 1e-5 arm completed all 50 updates on Ascend 910B2, physical NPU 2.
The [matched comparison report](../tmp/13_qwen3_reranker/contents_swap_lr1e5_d474ba6c_20261007/README.md)
retains all trajectories, per-task scores, original/swapped baseline checks and
five saved-checkpoint inventory entries. The untrained baseline scores and token
hashes match the 1e-6 arm exactly, with no truncation.

Combined endpoint results (10 fixed queries/task), percentage NDCG@10:

| Model | English | Chinese | Teacher agreement | Teacher margin MSE |
|---|---:|---:|---:|---:|
| Original query-first | 71.179 | 73.225 | 81.859% | 6.3827 |
| Swapped, peak LR 1e-6 | 56.690 | 66.285 | 75.364% | 8.4936 |
| Swapped, peak LR 1e-5 | 62.502 | 66.722 | 80.627% | 4.7424 |

The higher learning rate produced better endpoint recovery than 1e-6 in both sampled suite means. Accuracy remains substantially below the original query-first baseline. These small panels do not establish full-suite accuracy.

## Query-first pilot

This experiment adapts Qwen3-Reranker-0.6B to a frozen Qwen3-Reranker-4B's
within-query score differences. It provided the query-first control for the follow-up above.
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
establish full-suite parity; these two control arms used query-first inputs.

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
Save frozen teacher targets once. For query-first controls, assert teacher/student
token IDs and truncation are identical using per-section token hashes. For the
contents swap, validate teacher hashes against canonical query-first encoding,
then separately validate candidate alignment and absence of truncation in the
swapped student inputs.

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
