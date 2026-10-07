# RWKV validation plan

Agreed sequence, 2026-10-06. Start with correctness and accuracy on Ascend 910B2,
then decide whether larger evaluations need faster inference. This document is
the experiment's single working document, including research findings, source
provenance and reference scores. The server CPU environment and C runtime have
been prepared, and the eight-case CPU FP32 smoke passed on 2026-10-06. The owned
full embedding forward passes real-case NPU parity in eager and TorchAir, and
complete NanoBEIR evaluation passed at reference batch 4: macro NDCG@10
58.897462 versus reported CPU 58.907231. The 90M reranker passed numerical
smoke and completed full NanoBEIR at B4 on two data-parallel NPUs: FP16
60.657936 in 12m11s, FP32 60.659794 in 9m31s. FP32 does not close the gap to
published 63.41. The alternate 11-task BM25-plus-positives evaluation gives
63.233688 in 7m25s, narrowing the gap to 0.176312 points; the published run
contract remains unconfirmed. Reference scores below remain upstream reports.

## 0. Establish a server CPU reference

Use the selected v0.23 container for an explicitly labeled CPU FP32 run of the
0.1B embedding model before the NPU adaptation. Start with a few fixed samples,
including English and Chinese and a sample crossing an EOS chunk boundary.
Save exact input token IDs, preprocessing/pooling metadata, layer hidden outputs
and final embeddings for comparison with the NPU path. The C trace API does not
expose raw recurrent matrices; add that tracing later if needed for the NPU port.
Inspect the upstream CPU execution path before selecting the reference runner;
the C implementation provides an additional embedding reference but does not
replace the need for intermediate-state comparisons. Keep RWKV dependencies in
a separate environment. The initial eight-case CPU reference is now saved and
passed its smoke checks; see the CPU smoke evidence below.

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

### NPU port approach and current status, 2026-10-06

The tiny model is **RWKV-7**: 12 layers, width 768, 12 heads of size 64.
It uses FP32 recurrent matrices and previous-token mixing vectors, not a growing
KV cache, RoPE or softmax attention. LayerNorm/GroupNorm, squared-ReLU ChannelMix
and first-layer value mixing must be preserved. Qwen's shared-input QKV packing
and PromptFA/IncreFA do not transfer directly.

`run_npu_embedding.py` owns the inference-only PyTorch model, loading the pinned
checkpoint directly with no Transformers/training framework. Projections use
explicit 2D B*T matmuls; state, normalization and pointwise math remain FP32.
FP16 projections and an FP32 diagnostic control are validated. Preserve the CPU
preprocessing and RETR head; padding tokens remain real recurrent inputs.

Imported candidate: [RWKV-Vibe/rwkv_Ascend](https://github.com/RWKV-Vibe/rwkv_Ascend/tree/1a6eaeb47358001c4fed6e636ed95545fed1f20b),
commit `1a6eaeb47358001c4fed6e636ed95545fed1f20b`. Only the three relevant host/kernel
sources and license were imported, with independent operator/vendor identities.
[`wkv7_npu/provenance.json`](wkv7_npu/provenance.json) records hashes, adaptations
and contract. Its CANN Open Software License v1.0 is separate from the embedding
project's Apache-2.0 license. The kernel uses `[value,key]` state orientation and
`exp(log_decay)`; adapt the C reference's orientation and CUDA's `exp(-exp(w))`
convention explicitly.

The vector-only package compiled/installed privately with **CANN 9.0.1**; symbols,
source hashes and binary metadata passed. On **Ascend 910B2 NPU 7**, the isolated
recurrence passed **32 eager cases** (B1/B2, lengths 1/47/48/49/64/128/512/976,
zero/nonzero states) and **TorchAir B1/T976**. Maximum output/state error against
independent CPU FP64 math was **1.49e-8**; repeat, input-mutation and split-state
continuation checks passed. Saved graphs contain `RwkvReferenceWkv7`. Evidence:
[`eager`](../tmp/26_rwkv_inference/wkv7_shared_052a9fd4/),
[`TorchAir`](../tmp/26_rwkv_inference/wkv7_torchair_f79f4fb4/) and
[`package`](../tmp/26_rwkv_inference/wkv7_eager_a2254ce9/package_manifest.json).

The **full model** passed all eight CPU-anchor cases, comparing all 14 hidden
outputs and final embeddings on identical prepared inputs. Worst embedding
absolute error: **5.74e-7 FP32**, **2.70e-4 FP16**; FP16 minimum cosine **0.9999996**.
B1 lengths were 16/64/96/128/144/976. B2 passed at T96 and T976, including the
empty/long pair. Full **static TorchAir B1/T96, B1/T976 and B2/T96** embeddings
were bitwise identical to eager. Sources and complete logs/results are in
[`tmp/26_rwkv_inference`](../tmp/26_rwkv_inference/), under `embedding_*` run names.

Luka authorized shared NPU 7 and then requested speed comparisons. Health was OK,
with about 8.8 GiB free before model loading. Paired FP16 eager/compiled forward
medians (5 samples) were **42.5/7.25 ms** for English T96, **46.5/24.8 ms** for
the long T976 case and **46.3/7.54 ms** for B2/T96. Corresponding four-thread C
CPU measurements were **1.53/10.54/2.53 s**. These shared-device timings exclude
tokenization, transfers, traces and cold compilation (34–52 s per fresh shape).

**NanoSCIDOCS passed**, source `e8a15dad`, shared Ascend 910B2 NPU 7, eager
FP16 projections/FP32 state. All **2,210 documents, 50 queries and 244 judgments**
were covered at revision `484eb90549fc3f0b9c42b3551e80ceb999515537`.
The largest prepared input **T2032** passed CPU layer/embedding comparison
(embedding max error **7.63e-5**); B4 parity passed separately. The owned runner
preserves MTEB **1.38.60** data, ordering, instruction and metric contracts;
`pytrec_eval` scores passed an independent per-query NDCG crosscheck.
B1 scored **40.946**, evaluating in **103.74 s**; published-setting B4 scored
**41.058** in **37.57 s**, matching CPU FP32 and **+0.070 points** above GPU BF16.
Corpus throughput was stable at **21.9/60.8 documents/s** (B1/B4); total times
including setup and CPU preflight were **143.76/59.14 s**. Logs, timings, per-query
scores, top-100 candidates and provenance: [`B1`](../tmp/26_rwkv_inference/nanoscidocs_eager_b1_fp16_e8a15dad/)
and [`B4`](../tmp/26_rwkv_inference/nanoscidocs_eager_b4_fp16_e8a15dad/).
Higher NanoSCIDOCS batches passed numerical CPU parity but changed padding and
scores: B8 **40.210 / 48.4 documents/s**, B16 **40.572 / 42.2 documents/s**.
Both were slower than B4; preserve reference B4 for benchmark reproduction.
Evidence: [`B8`](../tmp/26_rwkv_inference/nanoscidocs_eager_b8_fp16_e6bd192d/)
and [`B16`](../tmp/26_rwkv_inference/nanoscidocs_eager_b16_fp16_e6bd192d/).

**Full NanoBEIR passed**, source `c3166fd1`, shared Ascend 910B2 NPU 7,
raw eager, FP16 projections/FP32 state. All **13 tasks, 56,723 documents and
649 queries** use pinned MTEB 1.38.60 revisions; coverage and independent
per-query metric checks passed. Macro NDCG@10 was **58.897462**, versus CPU
**58.907231** (−0.009769 points), GPU BF16 **58.973692** (−0.076230 points)
and paper **59.10**. Twelve task scores match CPU to published precision;
DBPedia was **54.551 versus 54.678** (−0.127 points). Luka accepted the match;
no further FP32 diagnostic is planned.

Evaluation took **993.39 s (16m33s)**; including setup and CPU preflight,
**1,161.44 s (19m21s)**. Sixty documents exceeded 2,048 prepared positions,
with maximum **7,936**. WKV alone now runs in state-carrying chunks of at most
2,048 positions; CPU comparisons at T2064/T7936 passed (maximum embedding
error **5.64e-5**). Device row splitting limits token slots to 8,192 while
preserving reference B4 padding, EOS masks and complete input sequences.
Peak reserved HBM was **2.44 GiB**. Top-100 candidates for all 649 queries
(**64,900 pairs**) are saved for all three rerankers. Compact logs, per-task
scores, timings, candidates and acquisition/protocol provenance:
[`full NanoBEIR evidence`](../tmp/26_rwkv_inference/nanobeir_eager_b4_fp16_c3166fd1/).
Large embedding arrays remain on the server with recorded hashes.
The **90M reranker** numerical anchors and full NanoBEIR run are recorded
below; reproduction of the published reranker accuracy remains pending.

Start NanoBEIR evaluation on one NPU. If it is not fast enough, use **data
parallelism**, with a complete model replica on each participating NPU and
inputs distributed across replicas. Verify identical query/candidate coverage
and numerically consistent outputs before combining results. Apply this approach
to embedding and reranker evaluations; no model sharding is planned.

## 2. Adapt and verify the rerankers, one size at a time

Order: **90M, then 317M, then 1.3B**. The latter two also require adapting their
matching **0.4B and 1.4B embedding/state backbones**. Candidate retrieval remains
the saved **0.1B** retriever output for all three, as in the paper.

The smallest reranker checkpoint is pinned in
[`data/reranker_checkpoint.json`](data/reranker_checkpoint.json): **90,963,456
reranker parameters**, 12 layers, width 768, BF16 weights. Its 370,287,479-byte
release also contains unused vision/projection tensors; text inference loads only
`reranker.*`. Downloaded directly on the selected server; SHA256 verified.
`local_modeling_rwkv_embedding.py` and `local_modeling_rwkv_reranker.py` keep the
models separate. The embedder exports/accepts both previous-token vectors and
FP32 `[value,key]` matrices; the reranker reads those matrices with one learned
token. `run_reranker_smoke.py` uses hash-pinned upstream PyTorch functions as an
independent CPU FP32 reference, replacing only the CUDA recurrence, decorators
and device literals. This is a small numerical oracle, not an existing released
CPU reranker runtime.

**Smallest-reranker smoke passed**, source `955f8fb7`, shared Ascend 910B2
NPU 7, eager FP16 projections/FP32 state: five B1 pairs (T25–346) and B2/T40.
All state, hidden-output and 12 reranker-layer comparisons passed. Maximum
CPU/NPU logit difference **0.017441**; ten continuation splits passed, with
maximum cached/full logit difference **0.001465** (CPU continuation was exact).
Saved states remained unchanged and repeated scoring was bitwise stable;
B2 versus identical separately executed padded rows passed (max **0.004395**).
The embedding API regression also passed (max embedding error **6.29e-5**).
The runner took **46.20 s**, including CPU reference work and artifact writing;
these are numerical smoke results, not benchmark accuracy or steady throughput.
[`Evidence and checkpoint provenance`](../tmp/26_rwkv_inference/reranker_smoke_eager_fp16_955f8fb7/)
include hashes and independently verified readback of six server-resident NPZ
anchors.

**90M NanoSCIDOCS completed**, source `edd3c013`, same shared 910B2 NPU 7,
raw eager FP16 projections/FP32 state, **B2**, saved 0.1B top-100 candidates.
All **50 queries / 5,000 pairs** and independent per-query metric checks passed.
NDCG@10 was **36.255**, versus embedding baseline **41.058** (**−4.803 points**).
No published NanoSCIDOCS reranker score is available; this establishes execution
and a task score, not reproduction of the paper's aggregate. Preserve this
quality regression when assessing full NanoBEIR and the upstream batching
protocol; do not attribute it to precision or model quality without evidence.

Scoring took **296.38 s (4m56s; 16.87 pairs/s)**; total including setup and CPU
preflight **375.89 s (6m16s)**. B2 forward median/p95: **95.35/171.69 ms**.
Longest input **B2/T2046** passed CPU FP32 comparison (logit max error
**0.001896**) and matched separately executed identically padded rows exactly.
Left zero padding and one terminal EOS follow the release; no pair was truncated.
Peak reserved HBM **0.93 GiB**.
[`Run, scores, timings and hashes`](../tmp/26_rwkv_inference/nanoscidocs_reranker_eager_b2_fp16_edd3c013/)
include verified readback of the server-resident CPU anchor. The probe below
identifies a padding issue to settle before changing the evaluation path.

B1 scorer probe `run_reranker_buckets.py`, source `ea9aa394`, passed 16 selected
real pairs across static T256/512/1024/2048: compiled scores exactly match
padded eager, repeats are stable, and fresh-process disk-cache loads preserve
identical scores. Warm medians were **11.5/16.5/26.8/45.8 ms**, versus eager
**81.2/84.2/76.4/80.3 ms**; prepared inputs only, profiles/first calls excluded.
Cold compile took 73–99 s; cached first calls about 5 s. The compile-only cleanup
guard also passed a fresh independent CPU/NPU smoke.

**Padding diagnosis**, sources `033c61c2` / `1315ecc7`, 910B2 NPU 7:
token 0 is a nonzero embedding and unmasked recurrent input in the release.
All four selected cases match released token preparation and CPU FP32/NPU FP16
scores. At T512, the 341/306-token pair reverses on both CPU and NPU: NPU
logits change from −8.078/−8.070 to −7.699/−8.602. Right padding also changes
the final matrices; it is not a neutral replacement. These sampled documents
are unjudged and outside the earlier B2 top ten; no NDCG degradation is established.
One long FP16 split-continuation check fails the unchanged strict tolerance
(0.011719 logits); isolated CPU FP32 is exact and NPU FP32 differs by 0.00000334.
Preserve that failure separately from the much larger padding effect.

The inspected MTEB 1.38.60 path calls the released wrapper's default **B32**
(outer chunks 128), padding each group to its maximum before retaining the last
2,048 tokens. Our B2 run matches the padding rule, not those batch boundaries.
Next compare NanoSCIDOCS with upstream logical B32 input preparation; device
execution can split prepared rows without changing their IDs. Extra static
padding can now be placed on the right and excluded through endpoint-state capture
(validated below). Removing upstream left padding would change the benchmark protocol.
[`Diagnosis and retained failure/controls`](../tmp/26_rwkv_inference/reranker_padding_summary_1315ecc7.json);
[`pinned upstream source findings`](data/reranker_padding_sources.json).

**Right-padding endpoint capture passed**, sources `d7348673` / `8a64221a`,
910B2 NPU 7. Independent `RwkvEndpointWkv7` reads int32 device lengths, stops
recurrence at the valid endpoint and retains fixed output shapes; token-shift
vectors are gathered there. CPU FP64 parity passed **72 eager / 18 TorchAir**
recurrence cases (maximum error **1.49e-8**). Eight real B1 pairs, T257–2046,
passed at buckets **512/2048**: separately compiled backbone/head states and
scores are **bitwise equal to padded eager**, including changed lengths and
A/B/A replay in the same graph shapes. Maximum endpoint/unpadded score difference
was **0.0078125 FP16**; T512 FP32 control **0.00000858**. Both sampled pair
orders are preserved. The first full compile failed on a missing reference-head
converter; its evidence is retained and the harness correction passed. No new
accuracy or throughput result; upstream logical B32 preparation remains next.
[`Endpoint test summary, inputs, sources and package/graph hashes`](../tmp/26_rwkv_inference/reranker_endpoint_summary_8a64221a.json).

**Clean endpoint forward timings**, source `2b17b64a`, 910B2 NPU 7, B1 FP16,
20 repetitions/input after warmup, no document caching: T512 (257–510 valid tokens)
**13.7–17.2 ms compiled / 82.3–83.2 ms eager**;
T2048 (1038–2046 valid tokens) **33.1–46.7 / 84.0–84.4 ms**.
The head takes about 1.7–1.8 ms. Setup, tokenization, transfers, validation and
profiling are excluded; all score/state checks still pass. Both fresh processes
used private copies of the existing graph caches.
[`Samples, stage timings and hashes`](../tmp/26_rwkv_inference/reranker_endpoint_timing_summary_2b17b64a.json).

**B2/B4 endpoint probes passed**, source `df84396b`, same 910B2 NPU 7/FP16:
eight real pairs, unchanged row IDs, mixed lengths, row reversal and exact replay.
Compiled states/logits equal batched eager exactly; B1 comparison passes tolerance.
Maximum B1 logit delta **0.015625**; a close pair becomes a tie under B4 row
reversal. This is retained; no benchmark accuracy result is implied.
T512 compiled **96.4–108.5 pairs/s B2 / 117.1 B4**; T2048 **35.5–35.6 / 37.9**.
At B4/T2048 raw eager is faster: **44.2 pairs/s**, 90.4 ms/batch versus compiled
105.5 ms. Peak PyTorch reserved HBM **1.83 GiB**. Same timing exclusions as above.
[`Batch comparisons, stage timings and evidence`](../tmp/26_rwkv_inference/reranker_endpoint_batch_summary_df84396b.json).

**Full 90M NanoBEIR run**, source `8d237960`, 910B2 NPUs **7 and 6**, data
parallel B4/FP16, no document caching: **649 queries / 64,900 pairs / 13 tasks**
completed in **731.39 s (12m11s)** including preparation, warmup and aggregation.
Original task-wide logical B32 token preparation precedes query sharding;
T≤512 uses compiled endpoint extraction, longer inputs use exact-length eager.
Numerical preflights, disjoint coverage and independent NDCG checks passed.
Macro **60.657936** versus published **63.41** (gap **−2.752064**), above the
embedding baseline **58.897462**; NanoSCIDOCS **36.136662** remains below its
embedding baseline **41.057851**. Peak reserved HBM was **2.09 GiB/worker**.
The published accuracy is not reproduced; resolve the gap before larger models.
[`Scores, timings, raw logs and hashed protocol evidence`](../tmp/26_rwkv_inference/nanobeir_reranker_dp2_b4_8d237960/).

**FP16/FP32 dense timing**, sources `99584096`/`37c1bc55`, 910B2 NPU 7,
B4, eight unchanged real pairs, 20 warm synchronized forwards per window:
T512 compiled **34.27/33.90 ms**, eager **82.06/69.82 ms**; T2048 eager
**89.88/97.15 ms** (FP32 **8.1% slower**). Recurrence stays FP32 in both;
matmul HF32 is disabled. Successful runs pass numerical/replay gates, with
maximum FP32–FP16 logit delta **0.008757**. The original FP32/T2048 exact
head check failed; source `4fa423a8` quantified the difference as **4.77e−7**.
A tight **atol 2e−5 / rtol 2e−6** head check passes, with bitwise backbone states
and repeat calls; compiled/eager latency is **113.27/97.52 ms**. These are
endpoint-padded forward timings, distinct from the full-suite run below.
[`Timing evidence and original failure`](../tmp/26_rwkv_inference/reranker_dtype_b4_summary_37c1bc55.json);
[`quantified FP32 compiled-head check`](../tmp/26_rwkv_inference/reranker_fp32_head_b4_t2048_4fa423a8/).

**Full FP32 NanoBEIR**, source `4fa423a8`, same 910B2 NPUs 7/6 and B4:
**60.659794 NDCG@10**, only **+0.001858** over FP16, still **2.750206** below
published 63.41. Only **4/649** query NDCGs change; **10/13** task scores are
identical. All 64,900 pairs complete in **571.39 s (9m31s)**, peak reserved
HBM **2.53 GiB/worker**. Prepared tokens, candidates, shards, checkpoints and
model math match FP16; short inputs use TorchAir, longer inputs exact-length
eager, with matmul HF32 disabled. Coverage and independent metric checks pass.
Changing dense-projection precision does not explain this reproduction gap;
this comparison did not reproduce upstream's full FP16/BF16 arithmetic policy.
[`Full evidence and precision comparison`](../tmp/26_rwkv_inference/nanobeir_reranker_dp2_b4_fp32_4fa423a8/fp16_fp32_comparison.json).

**Protocol audit**, pinned upstream `3c306736`, 910B2 NPU 7: prompt, top-100,
logical B32 padding/truncation/EOS, pair ordering and metrics match the released
evaluator. On 16 real pairs, independent CPU FP32 logits match NPU within
**1.07e−4**. Upstream FP16 backbone/BF16 head math changes logits by up to
**0.397** across four top-100 queries, but all four NDCGs stay identical; the
full-suite effect remains untested. Training evaluator/checkpoint disagree on
`emb.weight` versus `token.weight`; our loader already applies the package's
correct alias. Published run/candidate identity remains unverified. The alternate
`evaluate.py` calls `CrossEncoderNanoBEIREvaluator()` with defaults verified in
Sentence Transformers 5.1.2/5.2.0: **11 tasks**, **BM25 top-100**, and **forced
inclusion of positives**. This differs from our 13-task embedding-candidate run
and from paper Appendix A. Merely excluding ArguAna/Touché from our saved scores
gives **62.012191**; it does not reproduce that alternate protocol. Resolve the
published evaluation contract before another full arithmetic-control run.
[`Two-entrypoint audit and pinned source links`](../tmp/26_rwkv_inference/reranker_protocol_audit_v2_4fa423a8/evaluation_contract_followup.json).
[`Audit, source links, independent comparisons and loader failure`](../tmp/26_rwkv_inference/reranker_protocol_audit_v2_4fa423a8/audit_summary.json).

**BM25 plus positives comparison**, source `5f2bc78b`, 910B2 NPUs **6/4**,
same 90M model and FP32 arithmetic: **63.233688 NDCG@10** versus reported
**63.41** (gap **−0.176312**), up **1.221497** from the previous 11-task mean.
All **550 queries / 57,688 pairs** (2,688 added positives), excluding ArguAna
and Touché, finish in **445.12 s**; BM25 baseline **57.149575**. Preserve the
pinned ST 5.1.2 positives-first/text-matching order, per-query B32 left padding
and sklearn tie-averaged NDCG; independent metrics and B4/tail parity pass.
This tests the alternate evaluator defaults, not a confirmed paper protocol.
[`Task comparison, scores, input hashes and sources`](../tmp/26_rwkv_inference/nanobeir_reranker_bm25_positives_dp2_fp32_5f2bc78b_retry1/protocol_comparison.json).


**Middle-pair speed smoke**, source `ec325479`, 2026-10-07, shared 910B2 NPU 7:
0.4B backbone + released 317M reranker, FP32, B4, uncached prepared inputs.
Eager/TorchAir medians: **156.28/104.66 ms at T512**, **327.87/355.88 ms at
T2048** (compiled **38.22/11.24 pairs/s**). CPU numerical smoke, right-padding/B1
and compiled parity pass; compiled states/logits equal eager. Peak reserved HBM
**4.53 GiB**; cold compilation **180.40/134.98 s**, excluded from speed.
These are shared-device timings, not quality results or isolated throughput.
[`Commands, checks and timings`](../tmp/26_rwkv_inference/reranker_middle_shared_ec325479/).

**Precision/fit smoke**, source `7ce99dc2`, shared 910B2 NPU 7, identical
unpadded **B1/T257**, uncached: middle eager/TorchAir medians are
**147.05/25.64 ms FP32**, **168.66/28.45 ms FP16**, **180.86/28.06 ms BF16**;
all numerical gates pass. States/pointwise math stay FP32. Reduced projections
save memory but show no speed gain here. Largest FP16/BF16 eager takes
**179.55/174.52 ms**, peak reserved **5.84/5.80 GiB**; FP32 live weights
(~9.93 GiB) exceed current spare HBM (~8.16 GiB). Largest FP16 padded and
compiled-state gates failed (relative RMSE ~0.0005); its unpadded eager checks
pass, but no validated largest TorchAir timing yet. These are single-input
smokes, not suite accuracy. Middle FP32 B4 full 13-task NanoBEIR ETA: **50–60
min / 25–30 min** on one/two NPUs; alternate 11-task BM25+positives: **40–45 /
20–25 min**, estimated from saved batch lengths, not a measured middle run.
[`Results, failed gates, precision comparison and estimate scope`](../tmp/26_rwkv_inference/reranker_precision_shared_7ce99dc2/precision_comparison.json).

**Largest accuracy verified**, source `9d712af2`, 2026-10-07: FP32/B4,
exact-length raw eager, data parallel on idle 910B2 NPUs **1/2**, same accepted
**11-task BM25+positives** protocol (**550 queries / 57,688 pairs**). Mean
NDCG@10 **71.61782** versus published **71.58** (**+0.03782 points**). All worker
parity/evaluator checks pass; prepared inputs match the completed middle run.
Measured scoring **29m38s**, complete run **31m27s**, including setup/checks.
[`Final results and provenance`](../tmp/26_rwkv_inference/nanobeir_large_bm25_dp2_fp32_9d712af2/probe/result.json).

**Middle accuracy verified**, source `acbedbc3`, 2026-10-07: FP32/B4,
data parallel on idle 910B2 NPUs **1/2**, accepted **11-task BM25+positives**
protocol, **550 queries / 57,688 pairs**. Mean NDCG@10 **68.79119** versus
published **68.60** (**+0.19119 points**); NanoSCIDOCS **42.98462**. All worker
parity and evaluator checks pass; prepared inputs match the accepted tiny run.
Measured scoring **14m52s**, complete run **16m37s**, including setup/checks.
This verifies our accepted protocol; exact paper protocol identity remains unresolved.
[`Final results and provenance`](../tmp/26_rwkv_inference/nanobeir_base_bm25_dp2_fp32_acbedbc3/probe/result.json).

**Current middle-pair profiles**, source `3ea75a78`, shared 910B2 NPU 7,
B1/T257: clean FP32/FP16 forward **25.78/29.18 ms**; warm real-pair disk read,
tokenization, preparation, H2D, scoring, D2H and JSON write **27.41/30.62 ms**.
Matmuls improve **7.20→4.74 ms**, offset by Cast **0.001→4.37 ms** and added
TransData **1.35 ms** (1,200 extra casts, 285 format conversions). WKV stays
**~7.4–7.5 ms**; graph kernel gaps only **0.16/0.23 ms**. All numerical gates
pass; weight/hash/conversion/H2D startup **3.91/4.47 s**, cached graph first call
**5.80/5.96 s** separately. CPU D2H scopes include queued-work waits; do not call
those transfer-only time. No cold-storage or HBM-saturation claim.
[`Paired traces, compressed CSVs, CPU markers and hashes`](../tmp/26_rwkv_inference/reranker_profile_b1_3ea75a78/profile_comparison.json).

Profiles show eager dispatch gaps and 2,441 kernels/score versus compiled 1,750;
WKV occupies 32% of compiled device kernel time at T256 and 62% at T2048,
followed by casts/normalization. These profiles precede the validated B4 benchmark path above.
[`Summary, evidence paths and hashes`](../tmp/26_rwkv_inference/reranker_b1_bucket_summary_ea9aa394.json);
raw traces remain on server, with verified local summaries/hash manifests.

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

Proceed one verified step at a time. Luka subsequently authorized the single
CPU-reference script, cases, commits/pushes, model download and server setup.
He reviewed the script and authorized the CPU smoke, which has now passed.
There is no training or
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
  The 0.1B checkpoint is pinned to repository revision
  `d6bfff190b6ce4fb6bbd9c574e7e26df3df64075`, with metadata license Apache-2.0.
  `rwkv0b1-emb-curriculum.pth` is 476,851,895 bytes; its published and locally
  verified SHA256 is
  `9033eec92f163d1a710474977fa3fb68b7ee04697e0e961d83434743bd256a15`.
  Source: the Hugging Face model metadata API with `blobs=true`, 2026-10-06.
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

## CPU reference code and preparation

One Python entrypoint: **`run_cpu_reference.py`**, with cases in
**`data/smoke_cases.json`**. It has two commands:

- `prepare`: verify the pinned checkpoint/source/vocabulary hashes, export the
  410 text tensors to upstream's FP32 binary format, copy the unmodified C source
  and license, and build `librwkv_emb.so`. Record source, binary, compiler and
  export provenance in `manifest.json`. This does not execute inference.
- `smoke`: use the C API for all tokenization, preprocessing and forward math;
  save per-batch NPZ files with raw/prepared token IDs, valid EOS masks, layer
  hidden outputs and normalized embeddings. Record exact expanded texts,
  dimensions, source identity, timing and artifact hashes in `result.json`.
  Check finite outputs, normalized embeddings and repeat-call isolation.

The eight cases cover English, Chinese, Unicode, an empty document and a long
document. The long case has 961 raw tokens under the pinned upstream tokenizer,
so it crosses the 512-token chunk boundary. Query instruction insertion is
explicit; no semantic ranking expectation is treated as a correctness check.
Default smoke settings are B1 and four CPU threads. The runtime and result
directories must be new; the script refuses overwrites. Hidden-output traces
are labeled separately from recurrent matrices. MTEB and Transformers are not
needed for this small C-backed smoke.

Verified server paths (prepared on 2026-10-06):

- Source: `/workspace/repos/rwkv-cpu-reference`, branch `codex/rwkv-cpu-reference`.
- Venv: `/workspace/venvs/rwkv_cpu_py312` (inherits torch and NumPy from the
  selected container, without changing its base packages).
- Checkpoint: `/workspace/rwkv_reference/models/rwkv0b1-emb-curriculum.pth`.
- Pinned upstream reference assets: `/workspace/rwkv_reference/upstream`.
- Prepared runtime: `/workspace/rwkv_reference/runtime`.

The container's `/workspace/models` mount is read-only, so RWKV assets use its
writable `/workspace` mount. The checkpoint was downloaded directly on the
server from `hf-mirror.com` at the pinned Hugging Face revision, in 115.11 seconds;
its complete byte count and SHA256 matched the Hugging Face metadata. Direct
`huggingface.co` access failed with network-unreachable errors. An initial local
download was made, but its attempted transfer was stopped; it is not the source
of the final server checkpoint. C source, license and vocabulary were downloaded
directly inside the container from `raw.githubusercontent.com` at the pinned
upstream commit; source and vocabulary hashes were verified.

Project source arrives through Git. The server's unauthenticated request to the
private GitHub repository returned HTTP 401, so source was delivered as an
approximately 20 KB incremental Git bundle against the server's existing
`97890f8384c04e57f0307174e01b61f1a7bd2ce7` commit. The isolated server checkout
fetched that bundle and checked out `codex/rwkv-cpu-reference` at
`27c34182b4c39f1195637fdd9cd674618ea86f4a`. No tracked source was edited there.

Preparation exited **0**: all 410 text tensors were exported, the tokenizer was
built and the unchanged C source compiled successfully on the server with GCC
11.4.0, `-O3`, OpenMP and FP32 weights. This is setup evidence, not numerical
validation. Library dependency resolution, exported-symbol inspection and CLI
help also succeeded without running a forward pass. The runtime manifest records compiler arguments, source/checkpoint
identity, tensor shapes and artifact hashes. Command, logs, manifest and download
provenance are preserved under
[`tmp/26_rwkv_inference/cpu_setup_27c34182/`](../tmp/26_rwkv_inference/cpu_setup_27c34182/).

Equivalent preparation command inside the selected container (already completed;
the existing runtime directory cannot be overwritten):

```bash
cd /workspace/repos/rwkv-cpu-reference
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
/workspace/venvs/rwkv_cpu_py312/bin/python \
  26_rwkv_inference/run_cpu_reference.py prepare \
  --upstream /workspace/rwkv_reference/upstream \
  --checkpoint /workspace/rwkv_reference/models/rwkv0b1-emb-curriculum.pth \
  --runtime /workspace/rwkv_reference/runtime
```

The first model smoke completed after Luka's code review at source commit
`27c34182b4c39f1195637fdd9cd674618ea86f4a`. Equivalent command:

```bash
cd /workspace/repos/rwkv-cpu-reference
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
/workspace/venvs/rwkv_cpu_py312/bin/python -u \
  26_rwkv_inference/run_cpu_reference.py smoke \
  --runtime /workspace/rwkv_reference/runtime \
  --output tmp/26_rwkv_inference/cpu_smoke_27c34182
```

### CPU smoke evidence, 2026-10-06

**Exit 0; all eight cases passed.** Runtime: upstream C, CPU FP32, B1, four
OpenMP threads, on the selected aarch64 server/container. Outputs and all 14
layer-hidden trace slices were finite; embedding norms passed the predeclared
`atol=1e-4, rtol=0` check. Repeating the first batch after the seven other cases
produced a bitwise-identical embedding. The long document contained 961 raw
tokens and 976 prepared positions, exercising the multi-chunk path.

The runner took **32.25 seconds** including setup, trace writing and the repeat
check. The eight initial traced forward calls totaled **24.41 seconds**; the
long-document call took **14.20 seconds**. These are traced smoke timings, not
untraced throughput measurements or NanoBEIR accuracy evidence.

The eight NPZ anchors total **59,660,901 bytes** and remain on the server at
`/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cpu_smoke_27c34182/`.
They contain exact raw/prepared IDs, EOS masks, FP32 embeddings and layer-hidden
outputs, ready for the NPU comparison. A subsequent readback verified every
NPZ hash, shape and finite embedding/trace output. The compact committed
[`result.json`](../tmp/26_rwkv_inference/cpu_smoke_27c34182/result.json),
[`artifact_index.json`](../tmp/26_rwkv_inference/cpu_smoke_27c34182/artifact_index.json),
command, log and exit code preserve source/runtime provenance and each server
artifact's path, hash and shape. Large trace arrays remain outside Git.

These CPU anchors were subsequently used for the NPU comparisons above. Raw
recurrent-matrix tracing remains unavailable in the unchanged C API; the isolated
recurrence uses an independent CPU mathematical reference.

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
- The separate `/workspace/venvs/rwkv_cpu_py312` environment was created with
  `--system-site-packages`, inheriting torch `2.10.0+cpu` and NumPy `1.26.4`.
  Both imports were checked with backend autoload disabled. No base packages
  were changed. Checkpoint export, the C build and the eight-case CPU smoke
  succeeded. Subsequent NPU validation and benchmark results are recorded above.

The current local session reaches the server through the existing task-specific
SSH configuration, not the default `~/.ssh/config`:

```bash
ssh -F /home/luka/Documents/Codex/2026-10-01/can-you-connect-to-my-mac/work/ssh-blue-zone/config \
  -o ControlPath=/tmp/codex-blue-zone-1000/rwkv-023-master-recovered \
  blue_zone_npu_server \
  'docker exec research_vllm_ascend_023_external_workspace hostname'
```

On 2026-10-06 the host rebooted; v0.23 stayed stopped (`restart=no`, exit 255,
not OOM). Restarting that existing container restored the lane; the cause of the
reboot is unknown (no retained previous journal). A foreground `-M -N` host
master at the socket above is retained as session 53254 (`ControlPersist=no`;
keep the session alive). Mac history records detached masters being cleaned up;
prefer retaining the foreground session.
[`Recovery evidence`](../tmp/26_rwkv_inference/ssh_recovery_20261006/result.json).

Run workloads inside the selected container;
preserve the parent source-through-Git lane and use `source npu-setup` for NPU
execution. Models, environments and prior evidence survived the reboot. Keep
code transfers separate from accumulated benchmark evidence; if intervening
evidence commits inflate the bundle, use a source-only Git commit/bundle.
The BM25 evaluator additionally requires scikit-learn `1.7.2` in the RWKV venv
(installed without changing NumPy `1.26.4` or SciPy `1.13.1`).

The upstream GPU implementation uses custom CUDA kernels; the owned Ascend
embedding path is validated above. There is currently no CUDA validation lane. The
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
