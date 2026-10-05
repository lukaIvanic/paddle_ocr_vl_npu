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
