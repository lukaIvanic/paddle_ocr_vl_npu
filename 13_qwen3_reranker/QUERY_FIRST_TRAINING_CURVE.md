# Query-first Qwen3-Reranker-0.6B training curve

## Status

The three-update quality pilot and maximum-length backward preflight passed on
Ascend 910B2. The representative BGE-M3 sample is still being acquired directly
inside `research_vllm_ascend_023_external_workspace` through `https://hf-mirror.com`.
The 250-update representative training curve has not started.

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

## Planned representative run

Acquire 6,000 training queries and 384 held-out validation queries from the pinned
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

The full curve starts afresh from the original checkpoint, with measurements at
0, 10, 50, 100, and 250 updates. Planned validation uses all 384 held-out mixture
queries and all 49 Touché queries with 100 candidates each. Each round must fit
within three minutes. The run has a 40-minute runner budget and saves metrics,
raw scores, sample identities, hashes, timings, and memory readings; it saves no
model weights.

## Evidence

- `tmp/13_qwen3_reranker/query_first_pilot_875081d4_20261006/`: initial timing-only pilot.
- `tmp/13_qwen3_reranker/query_first_pilot_b8b53be7_20261006/`: before/after quality pilot.
- `tmp/13_qwen3_reranker/query_first_long_pilot_b8b53be7_20261006/`: maximum-length backward preflight.

Each NPU run records its exact command, commit, hostname, physical device, log,
exit code, and result JSON. The command files are authoritative.
