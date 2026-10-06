# Query-first Qwen3-Reranker-0.6B training curve

## Status

The 250-update representative experiment completed successfully on physical NPU 7,
Ascend 910B2, in `research_vllm_ascend_023_external_workspace`, source commit
`6f381fb3`. Execution is fast, but this configuration fails the quality-retention
sanity check: Touché NDCG@10 drops by 13.438 points while held-out mixture ordering
improves by 1.116 percentage points.

| Updates | Touché NDCG@10 × 100 | Held-out positive above negative | Evaluation seconds |
|---:|---:|---:|---:|
| 0 | 72.519 | 94.382% | 94.258 |
| 10 | 66.348 | 94.457% | 94.881 |
| 50 | 63.308 | 94.978% | 94.885 |
| 100 | 64.536 | 95.238% | 94.989 |
| 250 | 59.081 | 95.499% | 94.867 |

At 250 updates, 38 of 49 Touché queries score worse, 10 improve, and one is
unchanged. The frozen Qwen3-Reranker-4B reference on these same candidates is
73.000. All curve measurements use all 49 queries with full qrels and the same
100 candidates per query. This establishes a regression for this configuration;
it does not isolate its cause or establish that the mixture cannot work with
another objective or training recipe.

The complete runner took 1,160.251 seconds (19 minutes 20 seconds), including
473.881 seconds of evaluation. Gradient updates took 598.514 seconds (9 minutes
59 seconds), averaging 2.394 seconds each. Peak allocated memory during updates
was 29.151 GiB and reserved memory was 33.287 GiB. Every update had finite loss
and gradient norm and changed the tracked parameters; exit code was zero.
These times exclude the preceding data acquisition and transfer.

The checked pool contains 6,000 training queries and 384 held-out validation
queries from all 55 eligible source families. The run consumed 4,000 unique
training queries / 8,000 labeled pairs, spanning all 55 sources, without repeating
a query. Validation contains 3,072 labeled pairs / 2,688 positive-negative
comparisons. No model weights were saved.

[Training curve](../tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006/training_curve.png)
· [Raw result](../tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006/result.json)
· [Verification summary](../tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006/verification_summary.json)

The experiment preserves the official query-first prompt and original prefix and
suffix. It uses the owned Qwen model with `npu_fusion_attention`, BF16 autocast,
FP32 master weights and optimizer state, and `NpuFusedAdamW`. The learning rate is
1e-5, weight decay zero, and gradients are clipped at norm 1. We record metrics
throughout the run and do not select a checkpoint or save model weights.

## Three-update quality pilot

Source commit on the isolated validation branch: `b8b53be7`. Physical NPU 7,
Ascend 910B2. Started from the original Qwen3-Reranker-0.6B checkpoint.

This is a small pilot from already completed downloads, not the representative
mixture. Its pool contains 256 queries from 12 sources: ATEC, BQ, LCQMC, PAWSX,
QBQTC_v2, afqmc, and six MIRACL language sources. Three updates consume 48 unique
training queries / 96 labeled pairs. Effective batch size is 32 pairs, with
microbatches of four and eight accumulation passes per update.

Held-out validation contains 48 queries and 383 pairs, including 335
positive-negative ordering comparisons. Its normalized query and document texts
are disjoint from the training pool. Touché uses a fixed eight-query subset and
all 100 frozen retrieval candidates per query; full benchmark qrels are used for
NDCG normalization.

| Metric | Before | After 3 updates | Change |
|---|---:|---:|---:|
| Held-out positive above negative | 96.716% | 98.209% | +1.493 percentage points |
| Touché subset NDCG@10 × 100 | 67.032 | 67.238 | +0.206 points |

These checks show that three updates did not collapse ranking quality. Eight
Touché queries do not establish a reliable accuracy improvement, and this subset
must not be compared directly with the full 49-query Qwen4B benchmark.

Total runner time, including both evaluations, was 50.675 seconds. Update times
were 4.026, 1.586, and 1.639 seconds. Each Touché evaluation took about 8.97
seconds; each complete validation plus Touché round took 13.1–13.8 seconds.
During updates, peak allocated memory was 18.925 GiB and reserved memory was
33.303 GiB. Gradients remained finite and the tracked parameters changed.

The earlier timing-only pilot used identical input data and omitted the final
quality evaluation; its runner took 33.623 seconds on physical NPU 6. It is
retained separately so its timing is not confused with the before/after run.

## Maximum-length backward preflight

A separate diagnostic drew from real downloaded MLDR pairs. Four available
pairs had 8,923–9,984 original tokens and were right-truncated to 8,192 total
while retaining the original prefix and suffix. One update consumed one Russian query's positive
and negative, using microbatch one and accumulation two.

Backward and optimizer execution passed. The update took 3.196 seconds including
first optimizer setup. Peak allocated memory was 24.712 GiB and reserved memory
was 39.738 GiB. This preflight checks execution at the length cap; it is not the
representative training experiment or evidence of an accuracy gain.

## Representative run configuration

The sample contains 6,000 training queries and 384 held-out validation queries from the pinned
format-repacked mirror `hotchpotch/bge-m3-data-finetune-unified`, revision
`b51dd24cbce7d89255911410ef74a36e07bfbab9`. Quotas follow the released source and
length-bin row counts with small source floors. This samples the released data;
it does not reproduce the BGE authors' training sampler or objective. The mirror
claims a format-only repack; that claim has not been independently verified.

Source-family exclusions follow the pinned MTEB English v2 retrieval and MTEB
Chinese v1 retrieval contracts used in this repository: HotpotQA, MS MARCO,
mMARCO Chinese, DuReader, T2Ranking, and cMedQAv2. Exact normalized query and
candidate-document text from the frozen Touché fixture is blocked too. Training
and validation query/document texts are disjoint. This is source-family and
exact Touché exclusion, not exhaustive shared-corpus text decontamination across
every benchmark or MTEB version. Validation is held out from our updates; the
original checkpoint's exposure to these published examples is unknown.

The full curve started afresh from the original checkpoint, with measurements at
0, 10, 50, 100, and 250 updates. Validation uses all 384 held-out mixture
queries and all 49 Touché queries with 100 candidates each. All five rounds took 94.3–95.0 seconds,
meeting the three-minute limit. The runner had a 40-minute budget and saved
metrics, raw scores, sample identities, hashes, timings, and memory readings.

The maximum total sequence length is 8,192 tokens, with right truncation of the
body while preserving the prefix and suffix. Of the complete prepared pool,
208/12,000 training pairs and 64/3,072 validation pairs were truncated. None of
the 4,900 Touché pairs was truncated.

## Interpretation and numerical control

The held-out mixture score measures positive-versus-negative ordering within the
published training-style groups. Touché measures graded argument relevance over
100 fixed retrieval candidates. Improving one does not establish improvement on
the other. This run uses pointwise binary cross entropy and full fine-tuning at
1e-5; it does not reproduce Qwen's or BGE's complete training recipe.

The row-count-weighted pool contains 2,364 NLI-for-SimCSE queries (39.4%),
755 SQuAD queries, 519 Trivia queries, and 504 NQ queries. NLI examples include
entailment/paraphrase positives and contradiction or unrelated negatives. A task
or objective mismatch and forgetting are possible explanations for the Touché
regression, but this single run does not distinguish them. No cause has been
established by an ablation.

The HF control compares the original weights on 32 short pairs against the owned
NPU implementation, both under BF16 autocast. Maximum logit difference is 0.84375
and maximum yes-minus-no margin difference is 0.46875. These values establish
numerical differences, not exact HF parity or verified ranking agreement; the
control does not retain the individual HF scores. The entire training curve
uses the same owned implementation, so the observed before/after regression is
within that fixed implementation.

## Evidence

- `tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006/`: complete representative run, exact input sample, raw scores, plot, diagnostics and evidence verifier.
- `tmp/13_qwen3_reranker/query_first_acquisition_492dc1f5_20261006/`: acquisition, transfer and launch evidence.
- `tmp/13_qwen3_reranker/query_first_pilot_875081d4_20261006/`: initial timing-only pilot.
- `tmp/13_qwen3_reranker/query_first_pilot_b8b53be7_20261006/`: before/after quality pilot.
- `tmp/13_qwen3_reranker/query_first_long_pilot_b8b53be7_20261006/`: maximum-length backward preflight.

Each NPU run records its exact command, commit, hostname, physical device, log,
exit code, and result JSON. The command files are authoritative. The complete
sample is preserved in the main run's `inputs/mixture.json.gz`.

`verify_evidence.py` passed locally. It verifies successful completion, all 250
finite updates, every planned evaluation, fixed validation/Touché membership,
100 finite candidate scores per Touché query, dataset checksum, source-family
exclusions, exact normalized Touché text exclusion, train/validation text
disjointness, per-source consumed counts and recomputed validation metrics.
`diagnostics.json` contains per-query changes and source-level summaries;
length/drift correlations are descriptive associations, not causal findings.

## Data preparation and transfer

Direct `huggingface.co` and its row API were unreachable from the NPU container.
`hf-mirror.com` was reachable and served valid byte ranges, but the last large
Parquet row-group transfers were slow. Local acquisition had an existing cache;
once its row-API throttle cleared, the final cached resume completed all 255
requests and the checked split in 76.683 seconds. This is the final resume time,
not total acquisition time: the preceding acquisition attempts and route checks
took substantially longer. The direct sampler was stopped after the checked
fallback completed.

The selected sample is 11,558,555 compressed bytes, SHA256
`153f4fc8e8297d057563ef87fc0add7e094f63aa644b25490951ff9a58103efd`.
It was sent as 23 independently checked chunks with eight parallel SSH clients;
transfer and whole-file verification took 159.281 seconds. A checksum gate
launched the detached NPU process only after the assembled sample matched.
