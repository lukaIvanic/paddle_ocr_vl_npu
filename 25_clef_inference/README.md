# Clef inference: Transformers and plain PyTorch baselines

Scope: Cloudflare/clef-flash (9B), text-only, BF16, one Ascend 910B. This
experiment does not target 310P. The local Transformers-free baseline is described below. Optimization remains
out of scope.

The official backbone and joint decision head are loaded unchanged using the
release's `load_release_model`; all parameters remain BF16 on NPU. Explicit
`attn_implementation="eager"` selects ordinary Transformers attention, with no
CUDA FlashAttention dependency. Native Transformers Gated DeltaNet execution is
preserved. No vLLM, compilation, quantization, cache reuse, or CPU inference
fallback is used. The official loader still loads the vision parameters, but
text-only requests bypass vision execution.

## Release provenance

Official release: <https://huggingface.co/Cloudflare/clef-flash>

Reference revision: `17f0b0ad64efb65d273590632833508766b2aae6`.
The saved results below used this revision. Supply its downloaded model directory
to either runner; neither runner downloads files or hashes the entire release.
The local loader checks tensor keys, shapes and dtypes. The Transformers runner
imports the official `joint_schema_model.py` from the supplied directory.
The recorded reference revision identifies the baseline, not a digest check of
arbitrary files supplied with `--model`. No weights are stored in this repository.

## Run in the existing 910B container

Source must arrive through Git pull; do not edit tracked files in the container.
Use a free, healthy device; do not trust the setup helper's selection without
checking actual free HBM and process ownership.

```bash
cd /workspace/repos/paddle_ocr_vl_npu
git pull --ff-only
source npu-setup
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/workspace/venvs/clef_transformers_py312/bin/python -u \
  25_clef_inference/run_transformers_smoke.py \
  --model /workspace/models/clef-flash --dtype bf16 \
  --output tmp/25_clef_inference/transformers_$(git rev-parse --short HEAD)/result.json
```

The existing model directory and environment are already prepared. If recreating
the environment, inherit the matching torch/torch-npu installation and install
the original Transformers baseline's overrides:

```bash
/usr/local/python3.12.13/bin/python3 -m venv --system-site-packages \
  /workspace/venvs/clef_transformers_py312
/workspace/venvs/clef_transformers_py312/bin/python -m pip install \
  'transformers==5.17.0' 'accelerate==1.15.0'
```

The local runtime needs torch, torch-npu, safetensors and tokenizers. Transformers
and Accelerate are needed only for the official reference runner. Obtain the
reference model revision separately using standard Hugging Face tooling; custom
download and environment-setup scripts are no longer part of this experiment.

Choose a new output path for each run. Both runners refuse to overwrite an
existing result file. Capture console output and the device/command alongside
the result when recording a new experiment.

## Evidence and interpretation

Six small requests cover English and Chinese, all three decision types, joint
questions over one state, and relevant versus irrelevant document pairs. The
first invoice request is also executed once as a labeled warmup. Cases are
functional smoke observations, not a retrieval-quality benchmark.

The runner saves exact input token IDs, question spans, logits, full-precision
probabilities, formatted answers, actual parameter/activation dtypes, token
counts, memory, and complete-request timings. It logs every completed request
immediately and prints a five-second active-phase heartbeat. Inputs must not be
truncated. Numerical/shape/device failures exit nonzero; semantic expectations
are reported separately, not treated as evidence of an implementation bug.

CPU preprocessing and synchronized model/end-to-end timings are recorded.
Backbone/head NPU events surround their *original invocation within each full
request*, not a replay. Event elapsed spans can contain stream idle time and
host-enqueue gaps; they are not pure kernel duration and should not be called
device utilization. Model tok/s counts the whole encoded input, including the
schema. Cold/setup and warmup are labeled; this smoke is not an optimized
throughput measurement.

## Validated 910B Transformers smoke

Completed on 2026-10-05, physical NPU 6 (Ascend 910B2), source commit
`f9bb6837`. [Saved invocation, logs and results](references/transformers_bf16_910b_f9bb6837/).
Exit code 0; six measured requests plus one labeled warmup. All eight semantic
observations passed; all probability/shape/device checks passed. All backbone
and head parameters, sampled activations, and output logits were BF16 on NPU.
The reference implementation's internal FP32 recurrent math was not changed.

| Smoke check | Relevant / expected result | Irrelevant result |
| --- | --- | --- |
| English invoice (three joint questions) | Overdue; amount above 1000; highest amount category | Not applicable |
| Chinese invoice (three joint questions) | Overdue; amount above 1000; highest amount category | Not applicable |
| English relevance, score 0–3 | 2.7292 | 0.0929 |
| Chinese relevance, score 0–3 | 2.7085 | 0.0271 |

Post-warmup complete-request observations:

| Requests | Encoded input tokens | E2E latency | Backbone event span | Head event span |
| --- | --- | --- | --- | --- |
| English/Chinese invoice | 348–363 | 1.067–1.072 s | 1.050–1.052 s | 12–15 ms |
| English/Chinese relevance pairs | 188–197 | 0.574–0.736 s | 0.559–0.716 s | 10–17 ms |

The first invoice warmup took 1.976 s end-to-end. Model loading took 11.58 s
(release verification is separate). Peak PyTorch-allocated NPU memory was
17.85 GiB. These are small B1 smoke observations, not optimized throughput or
benchmark-level retrieval accuracy; not every shape has its own warmup.

Transformers reported reference PyTorch fallbacks for `causal_conv1d` and
`chunk_gated_delta_rule` because their optimized optional packages are absent.
The backbone dominates the observed request spans; this establishes a working
reference, not a conclusion about Clef's best achievable NPU performance.

The container SSH listener was unavailable, so this run used the existing
`research_vllm_ascend_021_external_workspace` container through `docker exec`
over a multiplexed gateway SSH connection. The existing `/workspace` mount,
source-through-Git lane, and NPU setup were preserved; no services were restarted.

## Local Transformers-free baseline

The local runtime owns the complete text forward and loads the same pinned
release weights directly. It uses PyTorch, safetensors and tokenizers; it never
loads release Python or imports Transformers. Vision weights are omitted. The
untied `lm_head.weight` is retained because the decision head uses its lexical
option embeddings, but no full vocabulary logits are computed.

Like experiment 02, there are two main files:

- `local_modeling_clef.py`: the complete Qwen3.5 text backbone, released joint
  decision head, checkpoint loading and B1 forward.
- `run_local_smoke.py`: text/schema encoding, answer formatting and the parity
  run against the saved Transformers outputs.

`run_transformers_smoke.py` is the one reference/check script: it runs the
original model, or the small backbone checks with `--check-backbone`.
`smoke_cases.json` contains the editable examples, and `references/` preserves
the original run evidence. There are no separate test or setup scripts.

Scope is **B1, unpadded text, BF16, complete requests**. No generation, KV/recurrent
cache reuse, batching, compilation, quantization or custom NPU kernels. Requests
longer than 16,384 tokens and media inputs are rejected, not silently truncated.
The pinned release config is the supported architecture, not a generic Qwen loader.
Source provenance and modifications are recorded below.

After pulling this branch in an isolated server checkout and checking the device:

```bash
source npu-setup
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
/workspace/venvs/clef_transformers_py312/bin/python \
  25_clef_inference/run_transformers_smoke.py --check-backbone
/workspace/venvs/clef_transformers_py312/bin/python -u \
  25_clef_inference/run_local_smoke.py --model /workspace/models/clef-flash \
  --output tmp/25_clef_inference/local_bf16_$(git rev-parse --short HEAD)/result.json
```

Both runners use the existing environment; the local smoke actively blocks
Transformers imports.
The small NPU tests use Transformers only as an independent oracle, and also
compare the chunk scan against a token-at-a-time recurrence across chunk boundaries.
The real-checkpoint smoke requires exact token IDs, question/option spans, BF16
logits and formatted answers on all six saved cases (plus a labeled warmup).
It saves differences and exits nonzero on a mismatch. No tolerance is relaxed to
make a run pass. These are correctness checks, not retrieval-quality benchmarks.

Validated on **2026-10-05**, Ascend **910B2**, physical NPU **6**, source
`55d138dd`. [Saved local-run evidence](references/local_bf16_910b_55d138dd/).

- All six real requests (plus warmup) matched the reference token IDs, question
  and option spans, BF16 logits, and formatted answers exactly. Maximum logit
  difference was zero across all 32 measured option logits.
- The runtime import guard was enabled and `transformers_imported` was false.
- The NPU chunk-scan test passed at lengths 1, 63, 64, 65 and 129 against an
  independent token recurrence. A tiny two-layer hybrid backbone matched
  Transformers exactly in BF16 at lengths 1, 65 and 129.
- All six host fixture/input tests passed. The two head classes were also checked
  structurally against the pinned release: their computation is unchanged.

The small tests use Transformers 5.17.0 as an oracle; the real local runtime uses
only torch 2.10.0, torch-npu 2.10.0, tokenizers 0.23.2 and safetensors 0.8.0.
The `+cpu` torch version suffix in the environment does not mean CPU inference.
Every model parameter and all decision logits are BF16 on NPU; FP32 recurrent
and normalization arithmetic follows the original implementation.

Recorded timing is diagnostic only: `e2e` includes output validation/comparison,
the historical `verify_and_load_s` includes release hashing, and device events
include stream idle/enqueue gaps. Current runs report `load_s` without hashing.
No speedup or broad retrieval-quality claim is made by these checks. Main remains the Transformers baseline; this implementation is on the
review branch `codex/clef-transformers-free`.

## Source provenance

The local implementation adapts these [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
sources. Attribution and the changes to each source are recorded here.

- The backbone in `local_modeling_clef.py`: Transformers **5.17.0**, `models/qwen3_5/modeling_qwen3_5.py`.
  Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
  Inspected file SHA256: `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
  Changed to a text-only, B1, no-cache eager forward; removed the Transformers
  framework, optional kernel dispatch, export path and generation/vision code.
  The chunk scan, normalization and attention retain reference arithmetic.
- The head in `local_modeling_clef.py` and text encoding in `run_local_smoke.py`: Cloudflare **clef-flash**,
  `joint_schema_model.py`, revision `17f0b0ad64efb65d273590632833508766b2aae6`.
  Inspected file SHA256: `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
  The head computation is unchanged (imports and the record type annotation are adapted). Text encoding uses `tokenizers` directly,
  rejects media and overlength inputs, and omits padding and the serving wrapper.

Sources were read from the exact environment/release used for the saved 910B
baseline. The local runtime never imports either upstream Python file. Weights
and tokenizer remain in the separately downloaded release. Historical baseline
verification is recorded in the preserved references; current runners do not
repeat those file-hash checks.

## GPQA Diamond / Decision Index 0.2.1

`run_benchmark.py` runs labeled text-choice JSONL rows using the existing local
model and encoder. `--backend transformers` selects the official implementation
as an oracle; `--reference` requires exact input hashes, logits, probabilities and
choices for every request in that reference result. `--limit` explicitly labels
partial runs. Gold labels never enter either model. One warmup is excluded from
scores and timing summaries; failures abort rather than silently dropping cases.
Results contain IDs, input hashes and decisions, not question text or tokens.

Prepare data outside Git using the public
[Decision Index kit](https://github.com/apolinario/decision-index) at commit
`87d4650b42b377c0291a89c1f1a879f9b31082bf`:

- Download its pinned `data/sources/gpqa/dataset.zip` (SHA256
  `461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc`).
- Call `decision_index.suite.build.normalize_text.gpqa(Layout(work_directory))`.
  It preserves the question verbatim in `instructions`, uses an empty state,
  and shuffles the four options with `20260918:GPQA:<source-index>`.
- Apply `hub/excluded-questions.json`: exclude IDs `GPQA-Diamond:test:89` and
  `GPQA-Diamond:test:126` (duplicate options), leaving 196 of 198 source cases.
  Keep source order. Write each retained row with
  `json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"`.
  Our resulting rows file SHA256 is
  `c314db0fc1129239233eecbd453c50e3be296d462ba9948ca3ffd22909c86eb9`.

After the usual 910B environment setup, run the official sample and then the
complete local evaluation (choose fresh output paths):

```bash
/workspace/venvs/clef_transformers_py312/bin/python -u \
  25_clef_inference/run_benchmark.py --model /workspace/models/clef-flash \
  --rows /workspace/datasets/clef_gpqa_20261005/GPQA-Diamond-0.2.1.jsonl \
  --backend transformers --limit 8 \
  --output tmp/25_clef_inference/gpqa_reference/result.json
/workspace/venvs/clef_transformers_py312/bin/python -u \
  25_clef_inference/run_benchmark.py --model /workspace/models/clef-flash \
  --rows /workspace/datasets/clef_gpqa_20261005/GPQA-Diamond-0.2.1.jsonl \
  --reference tmp/25_clef_inference/gpqa_reference/result.json \
  --output tmp/25_clef_inference/gpqa_local/result.json
```

The comparison target is **51.0% raw accuracy**, reported in the
[Clef-flash model card](https://huggingface.co/Cloudflare/clef-flash#decision-index).
This is the public Decision Index adapter; Cloudflare's exact internal run
manifest and per-case predictions have not been independently verified.

## Tiny uncached reranking protocol check

`run_reranking_smoke.py` reuses experiment 22's pinned MTEB 1.38.9 evaluator,
corpus formatting, task instructions, saved embedding top100 candidates and
Qwen3-Reranker-4B pair scores. It prepares one FiQA2018 test query and one
EcomRetrieval dev query, each with three selected candidates: first positive,
first negative and last negative in embedding-score order. The first sorted
query supporting that selection is used. This deliberately label-balanced
sample checks plumbing; it is not an unbiased quality estimate or a full
MTEB-R/CMTEB-R result.

The document alone is `state`. The original task instruction and query go in
one `noul` relevance question. Rank by unrounded `P(true)` from the decision
logits; retain the official rounded answer separately. No labels or Qwen scores
enter model inference. No input is truncated. Model arithmetic is unchanged.

Prepare in the existing pinned evaluator environment, using the original saved
run directories and offline dataset cache:

```bash
source npu-setup
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
/workspace/venvs/qwen3_embedding_eval_py312/bin/python \
  25_clef_inference/run_reranking_smoke.py prepare \
  --source-repo /workspace/repos/paddle_ocr_vl_npu \
  --english /workspace/repos/paddle_ocr_vl_npu/tmp/22_qwen3_embedding_benchmark/english_5npu_894706ef/evaluation \
  --chinese /workspace/repos/paddle_ocr_vl_npu/tmp/22_qwen3_embedding_benchmark/reranker_ecom_e70e8473/evaluation \
  --output <NEW_FIXTURE_JSON>
```

Run the six uncached pairs on a free 910B2 in the model environment, then evaluate
in the pinned CPU evaluator. These are separate processes so local model
inference can keep its Transformers import guard:

```bash
source npu-setup
/usr/local/python3.12.13/bin/python3 \
  25_clef_inference/run_reranking_smoke.py run \
  --fixture <FIXTURE_JSON> --model /workspace/models/clef-flash \
  --output <NEW_RESULT_JSON>
/workspace/venvs/qwen3_embedding_eval_py312/bin/python \
  25_clef_inference/run_reranking_smoke.py evaluate \
  --fixture <FIXTURE_JSON> --result <RESULT_JSON>
```

The fixture saves exact documents, queries, task instructions, selected IDs,
judgments and source-file hashes. Evaluation preserves full query judgments but
ranks only the three selected candidates, using the existing NDCG@10 and
self-match rules. An independent two-document example checks the evaluator's
NDCG discount. Keep the fixture unchanged for the future cached/uncached check;
these commands do not implement document caching.

### Verified six-pair baseline (910B2)

Validated on 2026-10-05 at code commit `92f4bdf2`. Preparation used the existing
MTEB 1.38.9 environment and offline caches; FiQA's implicit default qrels
configuration was made explicit, and its loaded Arrow-cache revisions verified.
Actual inference ran uncached on physical 910B2 NPU 3 in the isolated CANN 9.0.1
runtime (torch 2.10.0, torch-npu 2.10.0.post2), with Transformers imports blocked.
The model and encoding code were unchanged. All six inputs (191–428 tokens)
completed without truncation and produced finite BF16 noul logits.

| Selected query / three candidates | Clef NDCG@10 | Saved Qwen3-Reranker-4B NDCG@10 |
| --- | ---: | ---: |
| FiQA2018 `10034` | 0.613147 | 0.613147 |
| EcomRetrieval `200000` | 0.630930 | 1.000000 |

The English ordering matched Qwen exactly. On the Chinese query, Clef put
candidate `98713` ahead of the judged positive `260`; Qwen ordered those two the
other way around. Candidate `38006` was last for both. Zero relevance here means
zero or absent relevance in the benchmark judgments, not a proof that a document
is semantically unrelated. Preserve this difference as part of the uncached
baseline; the prompt was not adjusted based on these scores.

NDCG uses the full selected query's judgments with only the three selected
candidates ranked. Consequently, the English perfect ordering among these
three documents scores below 1 because another judged positive is omitted.
These two query numbers are **protocol smoke results, not task-wide accuracy**.
The existing evaluator also passed the independent two-document discount check.

Exact fixture, raw logits/probabilities, ranking/metric results, runtime versions
and commands are saved under
`tmp/25_clef_inference/reranking_smoke_92f4bdf2/`. Keep `fixture.json` byte-for-byte
unchanged for the cached comparison. The original vLLM-Ascend Clef service on
NPU 6 remained healthy and idle after this separate run.

## Fixed 40-pair length sample for document-cache checks

Prepared at `44ca1ad4` using Clef's actual tokenizer and the original pinned
MTEB-R/CMTEB-R datasets, saved embedding candidates and task instructions.
CPU-only preparation profiled 112,246 candidate pairs from 1,123 deterministic
queries across all 18 tasks. Each task contributes up to 64 hash-sampled queries
(TRECCOVID has 50, Touche has 49); protocol self-matches are excluded.
Reference length percentiles give each task equal weight.

The frozen sample contains **10 queries total, five English and five Chinese,
with four documents each**. Queries come from five distinct tasks per language,
chosen near joint query/median-document length percentiles 10/30/50/70/90.
For each query, documents are selected near its candidate-set document-length
percentiles 10/50/90/99. Selection uses neither relevance labels nor model scores.
The fixture retains exact text, original IDs, judgments, saved Qwen scores,
input lengths/hashes, dataset revisions and candidate/tokenizer/source hashes.

| Task / query | Query tokens | Four document lengths | Four complete input lengths |
| --- | ---: | --- | --- |
| CQADupstackGamingRetrieval / `77471` | 7 | 42, 84, 149, 217 | 222, 264, 329, 397 |
| CQADupstackUnixRetrieval / `10158` | 10 | 65, 130, 424, 1953 | 248, 313, 607, 2136 |
| SCIDOCS / `fa3894d83f83d7d05f54b2c87158dc7a2288dc1c` | 14 | 84, 167, 281, 506 | 270, 353, 467, 692 |
| FiQA2018 / `3179` | 17 | 74, 201, 500, 714 | 258, 385, 684, 898 |
| ClimateFEVERHardNegatives / `2983` | 40 | 118, 308, 535, 730 | 328, 518, 745, 940 |
| EcomRetrieval / `200293` | 3 | 16, 18, 23, 26 | 191, 193, 198, 201 |
| MedicalRetrieval / `90` | 5 | 20, 43, 115, 169 | 194, 217, 289, 343 |
| MMarcoRetrieval / `646623` | 6 | 31, 58, 97, 171 | 204, 231, 270, 344 |
| DuRetrieval / `78b5ea5f4b35ec5bdbde24b1609a36bd` | 10 | 46, 148, 208, 725 | 225, 327, 387, 904 |
| CovidRetrieval / `8df6de16294220931f8552ba4f0bdce7` | 15 | 123, 426, 1400, 3016 | 310, 613, 1587, 3203 |

Document-only length summaries:

| Language / population | Min | P50 | P90 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| English reference, 61,046 pairs | 2 | 159 | 475 | 1532 | 9034 |
| English selected, 20 pairs | 42 | 217 | 714 | 1953 | 1953 |
| Chinese reference, 51,200 pairs | 1 | 64 | 491 | 2641 | 6985 |
| Chinese selected, 20 pairs | 16 | 115 | 725 | 3016 | 3016 |

This deliberately gives extra coverage to long documents. It is a small
correctness sample, not a frequency-representative accuracy estimate or coverage
of every extreme input. In particular, the selected English query range is
7–40 tokens while the reference query P90 is 91 tokens. Every selected document
boundary is unaligned to the 64-token arithmetic block; together they cover
26 different nonzero offsets. Separate aligned controls remain useful.

No truncation was applied. One complete input is 3,203 tokens, above the current
3,072-token HTTP limit; a standalone comparison needs `--max-length 3203` or
higher. This preparation did not change the running service or its limits.
It also did not implement or validate complete document-cache reuse. Keep this
fixture unchanged for the later cached/uncached comparison, checking logits,
unrounded relevance probabilities and the four-document ordering per query.
The 40 documents are distinct; a separate query A/B/A check against the same
document is still required to prove reusable caches remain unchanged.

Frozen evidence is in `tmp/25_clef_inference/reranking_lengths_44ca1ad4/`:
`fixture.json`, `command.txt`, `exit_code.txt`, `run.log` and `validation.json`.
The fixture SHA-256 is
`c45e3da7cf069c6407a948494d3b1e18a20f9445c751023845eba83240c2e9cf`.
`prepare-lengths` on `run_reranking_smoke.py` exposes the preparation command;
its full saved command names all four original benchmark run directories.

### Forty-pair Gated DeltaNet boundary test (910B2)

Tested the frozen fixture at `ccf188d6` on physical NPU 3 in the CANN 9.0.1
runtime, with BF16 model weights and FP32 recurrent state. All 40 pairs completed
without truncation, including the 3,203-token input. Transformers imports were
blocked. The existing HTTP service remained ready on its separate NPU.

The probe derives the owned recurrent scan with only initial/final state exposed.
Every pair runs four times: original full execution; all 24 recurrent layers
split at the document boundary with unchanged FP32 state carried into the
question; a split at the preceding 64-token boundary; and original execution
again. Projection, convolution, full-attention and joint-head calculations keep
their original full-sequence shapes. This isolates recurrent block regrouping;
**it does not test complete KV/conv/hidden-state cache reuse**.

| Comparison against original execution | Mean absolute probability change | Median | P90 | P95 | Maximum | Exact final logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Document-boundary split | 0.00251057 | 0.00047146 | 0.00525039 | 0.01069033 | 0.03512251 | 3/40 |
| Aligned split | 0 | 0 | 0 | 0 | 0 | 40/40 |
| Original execution repeated | 0 | 0 | 0 | 0 | 0 | 40/40 |

Probability changes above are on the 0–1 scale; the maximum is **3.512 percentage
points**. Maximum absolute final-logit change was 0.0859375. Aligned controls
include 35 nonzero splits and five short prefixes whose aligned cut is zero.

Nine of ten four-document rankings were unchanged. The first-ranked document
was unchanged in all ten. MedicalRetrieval query `90` swapped second and third:

| Document | Original P(true) | Document-boundary P(true) | Original → split rank |
| --- | ---: | ---: | --- |
| `48284` | 0.52926338 | 0.53995371 | 1 → 1 |
| `9814` | 0.47609246 | 0.47025004 | 2 → 3 |
| `80953` | 0.47073662 | 0.50585914 | 3 → 2 |
| `26402` | 0.16079244 | 0.15765491 | 4 → 4 |

Both swapped candidates have relevance label zero, so this swap leaves the
sample's NDCG unchanged. This small length-selected sample is not an accuracy
estimate. It demonstrates that boundary rounding can change close rankings;
the score drift is not uniformly negligible. The largest change came from the
43-token document `80953` (79-token prefix, 217-token complete input), rather
than the longest document. The 3,203-token input changed by 0.00135126.

The probe is `tmp/25_clef_inference/gdn_boundary_40/probe.py`. Exact command,
exit code, all 160 measured passes, raw logits/probabilities, ranking comparisons
and evidence checks are in `tmp/25_clef_inference/gdn_boundary_40_ccf188d6/`.
Model source was unchanged. No numerical tolerance was selected after the run
to turn these observations into an accuracy pass/fail result.

### FP32 diagnosis of the worst boundary differences (910B2)

At `ff98eeb8`, reran the four pairs with the largest BF16 probability changes,
plus the remaining document from the changed MedicalRetrieval ranking: five
pairs total. Both BF16 and FP32 ran normal, document-boundary, aligned-boundary
and repeated-normal modes, on physical NPU 3. The model implementation and
checkpoint values were unchanged. FP32 promoted the same BF16-loaded weights
and used FP32 activations throughout; it did not recover higher-precision
checkpoint values. FP32 parameter storage was 36,302,264,336 bytes.

Both runs explicitly used `ACL_PRECISION_MODE=must_keep_origin_dtype`, disabled
matmul/conv HF32, and set `CUBE_MATH_TYPE=KEEP_DTYPE`. The option values were read
back and asserted. A dispatch audit checked floating tensor outputs throughout
every measured forward, including functional operations: every observed output
in the FP32 runs was `torch.float32` on `npu:0`. No autocast was used. The BF16
control reproduced every previous logit exactly under these settings and the
same audit, including the original ranking swap.

Absolute normal-versus-document-boundary probability changes (0–1 scale):

| Task / document | BF16 | FP32 |
| --- | ---: | ---: |
| MedicalRetrieval / `80953` | 0.03512251 | 1.19209e-7 |
| MMarcoRetrieval / `7904559` | 0.01565614 | 2.38419e-7 |
| MedicalRetrieval / `48284` | 0.01069033 | 3.57628e-7 |
| MedicalRetrieval / `9814` | 0.00584242 | 6.25849e-7 |
| MedicalRetrieval / `26402` | 0.00313753 | 3.12924e-7 |

For the original worst offender (`80953`), FP32 normal and split scores were
**0.48862370849** and **0.48862382770**: its discrepancy shrank by 294,629 times.
Across all five pairs, the largest probability discrepancy dropped from
0.03512251 to 6.25849e-7 (56,120 times smaller); maximum FP32 logit difference
was 3.63588e-6. Aligned splits and repeated normal execution remained exactly
equal in both dtypes.

The complete medical-query ranking agreed between FP32 normal and split:
`48284`, `80953`, `9814`, `26402`. This is also the BF16 split ordering; the
BF16 normal ordering is not a higher-precision reference. Only one MMarco
document was selected, so no MMarco ranking claim follows from this diagnostic.

These controlled results strongly support amplification of finite-precision
rounding as the cause of the large BF16 differences. FP32 retains tiny rounding
differences, consistent with regrouping equivalent equations. This diagnostic
still isolates the GDN boundary; complete document-cache reuse remains a
separate implementation and validation task.

The existing probe accepts `--dtype float32`, `--strict-precision`, and
`--worst-from <40_PAIR_RESULT_JSON>`. Commands, logs, both result files and
validation hashes are saved in `tmp/25_clef_inference/gdn_fp32_ff98eeb8/`.

### Isolating where the boundary difference enters and grows (910B2)

At `db465f9f`, traced the worst pair, MedicalRetrieval query `90` / document
`80953` (217 total tokens, 79-token prefix), on physical NPU 3. The BF16 model
and original normal/split logits were reproduced exactly. The same strict
precision settings as the FP32 diagnostic were used. Internal scan arithmetic
already runs in FP32 in this BF16 model; its return converts back to BF16.

At the first GDN layer, QKV projections, gating projections and convolution
outputs were exactly equal. All five scan inputs (Q/K/V/g/beta) were exactly
equal too. Q/K normalization and scaling were equal between the full scan and
concatenated prefix/suffix scans. The difference therefore first appears in
the remaining FP32 chunk arithmetic, after Q/K normalization.

For every GDN layer, the probe separately compared whole and split scans on
**identical inputs to that layer**. This distinguishes newly introduced local
rounding from errors propagated from earlier layers. All 24 such comparisons
had identical normalized Q/K and identical BF16 document-prefix outputs.

For the first layer:

| Stage | Measured difference |
| --- | --- |
| Input projections, convolution, Q/K normalization | Exactly zero |
| Raw FP32 scan output | Max absolute 1.04308e-6; relative RMS 1.27261e-7 |
| Scan output cast to BF16, question suffix | 111 / 565,248 values changed (0.01964%); max absolute 3.05176e-5 |
| GDN output projection | 2,600 changed values |
| First decoder block output, after residual and MLP | 15,998 changed values; relative RMS 0.00011333 |
| Final normalized backbone output, all layers split | Relative RMS 0.0246373 |

One first-layer activation (token 79, head 12, component 111) demonstrates the
rounding threshold directly:

| | Whole scan | Split scan |
| --- | ---: | ---: |
| Before BF16 cast | 1.5310940852941712e-6 | 1.531094994788873e-6 |
| After BF16 cast | 1.5273690223693848e-6 | 1.5348196029663086e-6 |

The FP32 gap is 9.09495e-13. These values fall on opposite sides of a BF16
rounding threshold, producing a BF16 gap of 7.45058e-9 (8,192 times larger).
This is one concrete example, not a claim that every cast amplifies error.

Causal interventions on the same full model:

| Intervention | Final P(true) |
| --- | ---: |
| All scans normal | 0.47073662281 |
| Only the first GDN scan split; all later scans normal | 0.50878816843 |
| All 24 GDN scans split | 0.50585913658 |
| Compute every split scan, but pass its same-input normal output onward | 0.47073662281 |

Replacing scan outputs restored **every captured activation and the final
logits exactly**. Repeated normal execution also matched every captured
activation exactly. Thus the changed scan outputs account for the downstream
discrepancy in this example. Later projections, norms, attention, MLPs and the
head run unchanged code on different inputs and propagate/amplify those changes.
Splitting only the first GDN is sufficient for a 3.805-point probability change.

The probe also ran each of the 24 single-layer splits and all 24 cumulative
prefixes of split layers. Effects are non-additive: additional splits can
increase or decrease the score. This isolates the source to the scan's FP32
chunk calculation and its BF16 output boundary; it does not yet attribute the
initial FP32 discrepancy to a single primitive among cumulative decay,
exponentials, matrix multiplications and triangular solves.

Probe: `tmp/25_clef_inference/gdn_isolation/probe.py`. Commands, complete local
error and activation traces, intervention scores, exit code and checks:
`tmp/25_clef_inference/gdn_isolation_db465f9f/`. Production model code and the
live serving process were not changed.

## Actual document-cache reuse and storage

Implementation added at `ec5e1de1`. `local_modeling_clef.py` now supports
preparing the fixed system prompt plus document once, then running only the
schema/question and assistant suffix through the backbone. It resumes all
24 recurrent states and convolution histories, concatenates cached K/V in the
eight full-attention layers, preserves absolute positions and causal masks,
and supplies cached prefix hidden states plus new suffix hidden states to the
unchanged decision head. The head still reads the document memory per query.

The normal uncached forward remains available. `encode_document_prefix()` in
`run_local_smoke.py` shares the original encoding boundaries with `encode_record()`.
One document cache can answer different questions; scoring does not append
questions to it or mutate its tensors.

The model and storage API, used under `torch.inference_mode()`, is:

```python
prefix_ids = encode_document_prefix(tokenizer, document)
cache = model.prepare_document(prefix_ids)       # One-time NPU computation.
cache.save(path)                                # Safetensors, original dtypes.

# Startup: eager copies into owned CPU RAM, not merely open mmap handles.
ram_cache = DocumentCache.load(path)

# Request: move this document to NPU memory, then compute its question.
npu_cache = ram_cache.to("npu:0")
logits = model(full_input_ids, encoded_record, cache=npu_cache)
```

`DocumentCache` stores recurrent states, three pre-convolution history tokens
per recurrent layer, unexpanded attention K/V, normalized document hidden
states, prefix token IDs and a model identity. Files retain FP32 recurrent
states and the selected model dtype for the other tensors. Keys include the
prefix IDs, dtype and model identity. The identity is deliberately local and
conservative: model source/configuration and checkpoint file sizes/mtimes;
it is not a portable content-addressed checkpoint release identifier.
Incompatible model identities, dtypes, devices or document prefixes are rejected.

The `cache` command in `run_reranking_smoke.py` prepares and saves the fixture's
documents, drops the preparation caches, eagerly reloads every file into RAM,
and scores each document with one NPU cache at a time. It measures preparation,
NPU-to-RAM copies, buffered file saves, RAM preload, RAM-to-NPU copies and
uncached/cached inference separately. Query timing uses synchronized medians
of three repeats, excludes model loading/encoding, and has no activation-dtype
audit overhead. File reads follow recent writes and may hit the filesystem
cache; they are not cold-SSD latency measurements. No eviction policy or
asynchronous transfer pipeline is included.

```bash
source npu-setup
/usr/local/python3.12.13/bin/python3 -u \
  25_clef_inference/run_reranking_smoke.py cache \
  --fixture tmp/25_clef_inference/reranking_lengths_44ca1ad4/fixture.json \
  --reference tmp/25_clef_inference/gdn_boundary_40_ccf188d6/result.json \
  --model /workspace/models/clef-flash --max-length 3203 \
  --dtype float32 --repeats 3 \
  --cache-dir /workspace/results/<CACHE_DIRECTORY> \
  --output <NEW_RESULT_JSON>
```

Cache tensors are kept outside Git under `/workspace/results/`. This is the
standalone experiment-25 path; it has not been wired into the running
vLLM-Ascend HTTP service.

### 910B2 cache measurements (40 pairs)

Both BF16 and FP32 completed the frozen ten-query/four-document fixture at
`ec5e1de1`, on physical NPU 3, CANN 9.0.1, torch 2.10.0 and torch-npu
2.10.0.post2. These are sequential eager inference measurements, not HTTP
latencies or full-suite quality results.

| Mean over 40 pairs | BF16 | FP32 |
|---|---:|---:|
| Uncached forward | 1.569 s | 1.618 s |
| Cached forward, cache already on NPU | 0.559 s | 0.576 s |
| First RAM-to-NPU copy per document | 27.6 ms | 10.1 ms |
| Cached forward plus that first copy | 0.587 s | 0.586 s |
| Repeated RAM-to-NPU copy, median per document | 7.47 ms | 7.53 ms |
| Total cache RAM for 40 documents | 2.514 GiB | 3.153 GiB |
| One-time document preparation, all 40 | 45.57 s | 46.91 s |

First-copy numbers include allocator/runtime effects; the BF16/FP32 difference
is not evidence that larger FP32 caches intrinsically transfer faster. The
resident path is about 2.8 times faster by the ratio of mean forward times.
Including the first transfers gives about 2.7 times faster overall in this
length sample. Precomputation is excluded from query latency.

Selected FP32 examples (tokens include the full encoded input):

| Input tokens | Uncached | Cached on NPU | Cached + first RAM upload | Speedup including upload |
|---:|---:|---:|---:|---:|
| 222 | 0.754 s | 0.575 s | 0.604 s | 1.25x |
| 607 | 1.809 s | 0.575 s | 0.585 s | 3.10x |
| 1,587 | 4.467 s | 0.584 s | 0.599 s | 7.46x |
| 3,203 | 8.599 s | 0.581 s | 0.604 s | 14.24x |

FP32 cached versus uncached maximum relevance-probability difference was
`2.74181366e-6`; maximum logit difference was `3.13520432e-5`. All ten document
rankings matched. BF16 retained nine of ten rankings: MedicalRetrieval/90
swapped its second and third documents, as in the earlier boundary probe.
Its maximum probability difference was `0.02340427`. All 40 uncached BF16
logits exactly matched the earlier frozen baseline.

Both modes passed tensor-by-tensor cache immutability and A/B/A reuse checks:
score question A, a different question B, then A again with identical A logits.
Mismatched document prefixes and model identities were rejected without
changing the cache. No Transformers modules were imported.

A separate fresh process at `1fb57148` preloaded all 40 saved FP32 files into
owned RAM in 1.723 s (filesystem cache may be warm). Document preparation was
disabled in that process. Three examples, including the worst prior numerical
case and longest input, reproduced the previous cached logits exactly. An
embedding hook observed only 144, 138 and 151 suffix tokens for full inputs of
222, 217 and 3,203 tokens, respectively. This verifies disk-to-RAM-to-NPU
reuse across process restarts without rebuilding documents.

Evidence: `tmp/25_clef_inference/document_cache_ec5e1de1/` contains commands,
exit codes, logs and per-pair results for both dtypes;
`tmp/25_clef_inference/document_cache_restart_1fb57148/` contains the restart
verification. Bulk Safetensors remain on the server outside Git.

### Full Touché candidate precomputation with BF16 storage

In our pinned English suite, `Touche2020Retrieval.v3` has the fewest reranking
pairs: 49 queries times 100 candidates = 4,900 pairs, using 4,863 distinct
candidate document IDs. TREC-COVID has 5,000 pairs and 4,619 distinct candidate
documents; it is smaller by that latter count. We selected Touché by workload,
not by the size of its full 303,732-document corpus.

`prepare-task` freezes all saved candidate memberships, queries, Qwen scores,
judgments, original corpus formatting and dataset/tokenizer hashes. It checks
MTEB 1.38.9 and the pinned Arrow-cache revision. The complete Touché fixture
contains 2,520,398 prefix tokens: mean 518.28, median 304, maximum 4,131. There
is no truncation. Bulk fixture text stays on the server outside Git.

`precache-task --dtype float32 --storage-dtype bfloat16` computes each complete
document in FP32, rounds **all** cache tensors to BF16, and saves them. This
includes the recurrent states that normally remain FP32. Expected tensor size
is 215.465 GiB (versus 430.930 GiB with FP32 storage), plus file metadata. The
initial runtime estimate is 2–3 hours on one 910B2, extrapolated from the prior
sample, not a measured full-task runtime.

The serving-side conversion is explicit:

```python
# Every tensor loaded from disk and retained in RAM is BF16.
ram_cache = DocumentCache.load(path)
# Transfer BF16 first; expand once on the NPU for FP32 computation.
npu_cache = cache_cast(ram_cache.to("npu:0"), torch.float32)
logits = model(full_input_ids, encoded_record, cache=npu_cache)
```

`cache_cast` lives in `run_reranking_smoke.py`. Expansion cannot recover the
precision discarded by BF16 storage. The 40-pair disk/RAM/NPU roundtrip check
at `07d2de04` preserved all ten rankings versus FP32 storage. Maximum relevance
probability change was `0.00169566274` (0.170 percentage points); maximum logit
change was `0.02713406`. Every stored tensor was BF16, file roundtrips were
bitwise exact, and every restored NPU tensor was FP32. This is a storage
precision check, not a full Touché accuracy result.

The full precompute command is resumable. Each document is written through an
atomic Safetensors replacement, then the file and directory are fsynced before
its manifest entry is saved and fsynced. Resume checks fixture, model, compute
and storage dtypes, file identity and size, and skips completed documents.
The longest document runs first and gets a tensor-by-tensor disk roundtrip
check. Only one document cache is retained at a time during precomputation.
The serving process on NPU 6 is separate.

```bash
source npu-setup
/usr/local/python3.12.13/bin/python3 -u \
  25_clef_inference/run_reranking_smoke.py precache-task \
  --fixture /workspace/results/clef_touche_full/fixture.json \
  --model /workspace/models/clef-flash \
  --dtype float32 --storage-dtype bfloat16 --max-prefix-length 4131 \
  --cache-dir /workspace/results/clef_touche_full/bf16_storage \
  --output /workspace/results/clef_touche_full/bf16_storage-manifest.json
```

Preparation evidence: `tmp/25_clef_inference/touche_prepare_72613d97/`.
Storage accuracy evidence: `tmp/25_clef_inference/bf16_cache_storage_07d2de04/`.
The full-task job's live manifest and bulk caches are on the server at the
paths above; starting precomputation does not establish a completed benchmark.

### Parallel continuation of Touché precomputation

At `e8c66ee5`, the single worker was intentionally interrupted after 114
persisted documents. Its manifest is frozen as the seed of a four-worker run
on physical NPUs 0, 1, 2 and 3. NPUs 4/5 reported health alarms; occupied NPUs
6/7 were not used. The interrupted single-worker status is not the status of
the replacement parallel run.

`tmp/25_clef_inference/parallel_precache/run.py` partitions only unfinished
documents, balancing prefix tokens plus a fixed per-document allowance.
Identical token prefixes stay on one worker, preventing concurrent writes to
the same cache filename. Each worker uses `precache-task --document-ids` with
an independent durable manifest. The coordinator validates complete, disjoint
coverage and existing cache file sizes, then merges every shard with the seed
into the original full-task manifest only after every worker succeeds.

Live plan, seed, per-worker commands/logs/manifests/exit codes and coordinator
status are in `/workspace/results/clef_touche_full/parallel_e8c66ee5/`.
While running, progress is `plan.json`'s `already_completed` plus document
counts in `manifest-0.json` through `manifest-3.json`; inspect `status.json`
for coordinator status and worker logs for failures. The original manifest
retains the seed snapshot until the final validated merge. All workers retain
FP32 computation, BF16 storage and per-document disk flushing.

## Local performance and profiling skeleton

`benchmark_local.py` and `profiling.py` follow experiment 21's real-item
observer. The default workload is the frozen 40-pair length sample. It is a
development workload, not a frequency-weighted benchmark accuracy estimate.
The default run executes document preparation, uncached requests and cached
requests, using FP32 computation and all-BF16 document storage in RAM. The
preparation measurement ends at a usable CPU cache; disk writes are excluded.
Cached requests include BF16 RAM-to-NPU transfer and expansion to FP32 on NPU.

```bash
source npu-setup
/usr/local/python3.12.13/bin/python3 -u \
  25_clef_inference/benchmark_local.py \
  --model /workspace/models/clef-flash --output-dir <NEW_RUN_DIRECTORY>
# Same complete execution with optional CPU/NPU traces:
# add --profile (captures actual second item in each phase), or
# --profile --profile-index 39 (captures the longest fixture pair).
```

`--mode prepare`, `--mode uncached` and `--mode cached` select one complete
workflow. Cached-only requires `--cache-dir` pointing to compatible existing
all-BF16 Safetensors; every file is eagerly preloaded into RAM during setup.
Normal `all` runs retain their prepared caches in RAM without writing them.
Each input is checked against the frozen token lengths/hashes. No truncation,
model arithmetic change, compilation, new batching or scheduling is introduced.

Host spans, whole-item wall time and NPU stream-event intervals remain separate.
Device events are resolved after normal CPU output materialization; there is
no device synchronization between measured inference stages. Nested backbone
and head scopes overlap their parent forward scope and are not added to it.
Stream intervals can include dispatch gaps; they are not active kernel sums.
First use is tagged separately, and subsequent real items retain their actual
shapes; this is not a fixed-shape warm microbenchmark. Peak allocated/reserved
NPU memory is recorded per phase, rather than sampled on every layer.

Every run saves command/workload metadata, source/checkpoint cache identities,
`events.jsonl`, `items.jsonl`, score vectors and `result.json` with distributions
and unresolved-event accounting. A five-second heartbeat reports the active
item/section. `--profile` adds fine source ranges for every GDN/full-attention/
MLP module, evidence-routing layers and the GDN scan. Trace export happens
outside item timing but contributes to phase/job time. Profiler runs are
explicitly labeled and must not be used as clean latency results.

All instrumentation is installed externally and restored on exit. In
particular, `local_modeling_clef.py` is byte-for-byte unchanged, preserving
existing document-cache identities. There is no model implementation copied
into the benchmark.

Validation at `7fe36427` passed on physical NPU 3 (910B2, CANN 9.0.1,
PyTorch 2.10.0 / torch-npu 2.10.0.post2). Four CPU bookkeeping tests pass
(pending-event reuse, nested accounting, wrapper restoration, real-item
profiler scheduling); those tests use fake events, not CPU model execution.
The three-case complete-workflow ABBA check (control/observed/observed/control)
and matched profiled run produced exactly identical scores for each execution
path. Three traces contain the expected Clef source ranges. All device events
resolved, Transformers remained unloaded, and the modeling-file hash was
unchanged. Observed phase overhead was +0.5% preparation, +1.0% uncached and
+2.9% cached; this small three-case measurement is noisy, not a general overhead
guarantee.

The full 40-pair cached baseline, with existing BF16 caches preloaded into RAM,
completed with the following per-request timing (including first use):

| Measurement | Time |
|---|---:|
| Complete request, mean | 632 ms |
| Complete request, median | 609 ms |
| Complete request, p90 | 639 ms |
| Model forward, mean NPU stream interval | 605 ms |
| Cache RAM-to-NPU transfer, mean NPU stream interval | 21.7 ms |
| Cache FP32 expansion, mean NPU stream interval | 1.4 ms |

Complete-request timing includes encoding, input/cache transfer, FP32 expansion,
model execution and output validation. It excludes model loading and initial
disk-to-RAM preload. This is local inference, without HTTP. A matched 40-pair
profiled run produced exactly the same scores and captured the longest pair
(index 39). These runs validate instrumentation; cached and uncached scores
are not claimed to be identical to each other.

Validation script: `tmp/25_clef_inference/profiling_skeleton/validate.py`.
Small results/logs: `tmp/25_clef_inference/profiling_7fe36427/`, including
`validation-now/validation.json`, `cached40-clean/result.json` and
`cached40-profile/result.json`. The four bulk traces remain in the matching
server checkout `/workspace/repos/clef-profiling-7fe36427`, outside Git.
To run immediately, cache shard 3 was paused after 439 saved documents while
shards 0–2 continued; `parallel_precache/resume_after_profile.py` resumes it
from the durable manifest and performs the final validated merge.

## Complete cached Touché reranking evaluation

`run_reranking_task.py` scores all 49 frozen Touché queries and their original
100 candidates each. It uses the unchanged `request_for` relevance question,
unrounded `P(true)`, FP32 model computation and the completed all-BF16 cache
manifest. Every distinct document cache is eagerly preloaded into RAM once;
only one restored cache is on NPU at a time. Inputs are untruncated, using the
existing encoder's 16,384-token ceiling. The 3,072-token HTTP service setting
is not used by this local benchmark.

The result is atomically saved and fsynced after each complete query. Restart
with the same arguments to skip completed queries; partial queries are retried.
Fixture, cache-manifest and modeling-file hashes must remain unchanged.

```bash
source npu-setup
/usr/local/python3.12.13/bin/python3 -u 25_clef_inference/run_reranking_task.py score \
  --fixture /workspace/results/clef_touche_full/fixture.json \
  --model /workspace/models/clef-flash \
  --cache-dir /workspace/results/clef_touche_full/bf16_storage \
  --cache-manifest /workspace/results/clef_touche_full/bf16_storage-manifest.json \
  --output <RESULT_JSON>
# Then in the existing MTEB 1.38.9 evaluator environment:
python 25_clef_inference/run_reranking_task.py evaluate \
  --fixture /workspace/results/clef_touche_full/fixture.json --output <RESULT_JSON>
```

Evaluation reuses experiment 22's `metric_summary`: full qrels, NDCG@10 and
recall@10/@100, with original self-match semantics. It compares Clef against
the frozen Qwen3-Reranker-4B and embedding scores, and retains per-query metrics.
No Qwen scores or relevance labels enter Clef inference. This measures the
complete Touché task, not the entire MTEB-R suite.

The full run was launched at `f268a541` on physical NPU 3 after the cache job
completed. Scoring uses the 0.23 container; a host-side coordinator automatically
runs evaluation in the existing 0.21 container's MTEB 1.38.9 environment, then
copies the evaluated result back. Live logs/status/results are under
`/workspace/results/clef_touche_full/benchmark_f268a541/` in the 0.23 container.

Preflight metric self-comparison using frozen Qwen predictions reproduces
Qwen3-Reranker-4B NDCG@10 **72.9999** and embedding NDCG@10 **69.4927** (0–100
scale). This is an evaluator check, not a Clef result. Small launch/check evidence
and the coordinator source are in `tmp/25_clef_inference/touche_full_f268a541/`.
