# Query-first ranking distillation pilot

This experiment adapts Qwen3-Reranker-0.6B to a frozen Qwen3-Reranker-4B's
within-query score differences. Document-first adaptation is a later experiment.
Code and mathematical unit tests exist; NPU validation/results must be read from
the run artifacts before claiming success.

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
eight Chinese v1 retrieval tasks pinned in experiment22 (MTEB1.38.9). Reuse the
original Qwen3-Embedding-0.6B top100 candidates; never inject judged positives.
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
Left pad to its longest sequence rounded up to128. Right truncate the prompt body
to fit8,192 total tokens, preserving the official prefix and suffix.

Train every parameter with FP32 weights and BF16 autocast, using the owned model's
`npu_fusion_attention` path. Construct a fresh NpuFusedAdamW with betas0.9/0.999,
epsilon1e-8 and zero weight decay; clip the accumulated gradient norm to1.0.
Compare constant1e-6 against five-step warmup to1e-6 and subsequent linear decay
(last update nonzero, zero reached after the run). Both arms start from the same
released weights and consume the same groups in the same order, without repeats.

Evaluate at0,1,3,10,25,50. The frequent benchmark+teacher-validation round must fit
three minutes before training starts. Each arm has a2,400-second wall-time limit.
Save model, optimizer, scheduler description, RNG states and training order at
every evaluated nonzero update. Keep large checkpoints on server storage.

## Validation and execution

`tests/test_margin_distillation.py` checks all-pairs loss/gradient algebra,
offset invariance, microbatch replay and LR scheduling. The real-input NPU control
compares HF eager, owned eager, fusion attention, replay and ordinary/fused AdamW.
Retain numeric differences; BF16 arithmetic is not generally bitwise equivalent.
Training requires the control artifact to pass.

After preparing `dataset.json.gz`, execute the committed pipeline on the910B2:

```bash
bash 13_qwen3_reranker/run_margin_distillation_experiment.sh \
  /workspace/results/qwen_margin_distill/data/dataset.json.gz \
  /workspace/results/qwen_margin_distill/npu/RUN_COMMIT
```

It selects a free device with `npu-setup` for each stage and records the physical
card, commit, exact command, log and exit status. Never edit tracked source on the
container. Success requires both movement toward held-out teacher rankings and
benchmark preservation; flat scores alone can mean negligible learning.
