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

Pinned revision: `17f0b0ad64efb65d273590632833508766b2aae6`.
`release.json` contains the Hugging Face API's file sizes, LFS SHA256 digests,
and Git blob IDs. Both download and smoke verify all files before importing
the release's Python code. No weights are stored in this Git repository.

## Run in the existing 910B container

Source must arrive through Git pull; do not edit tracked files in the container.
Use a free, healthy device; do not trust the setup helper's selection without
checking actual free HBM and process ownership.

```bash
cd /workspace/repos/paddle_ocr_vl_npu
git pull --ff-only
bash 25_clef_inference/setup_environment.sh
/workspace/venvs/clef_transformers_py312/bin/python -u \
  25_clef_inference/download_model.py --output /workspace/models/clef-flash
RUN_ROOT=tmp/25_clef_inference/transformers_bf16_smoke_$(git rev-parse --short HEAD) \
  bash 25_clef_inference/run_910b.sh
```

The downloader defaults to hf-mirror.com, pins the upstream commit, validates
byte ranges, and verifies digests. It prints a five-second speed heartbeat and
per-file completion. The isolated venv inherits the existing matching torch and
torch-npu packages; only Transformers/Accelerate overrides are installed.

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

`run_transformers_smoke.py` retains the original reference runner. Download/setup
scripts, the pinned release manifest, fixtures, tests and saved results support
reproduction; they are not separate pieces of the model implementation.

Scope is **B1, unpadded text, BF16, complete requests**. No generation, KV/recurrent
cache reuse, batching, compilation, quantization or custom NPU kernels. Requests
longer than 16,384 tokens and media inputs are rejected, not silently truncated.
The pinned release config is the supported architecture, not a generic Qwen loader.
Source provenance and modifications are recorded below.

After pulling this branch in an isolated server checkout and checking the device:

```bash
source npu-setup
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
/workspace/venvs/clef_transformers_py312/bin/python -m unittest discover \
  -s 25_clef_inference -p 'test_local_backbone.py' -v
RUN_ROOT=tmp/25_clef_inference/local_bf16_$(git rev-parse --short HEAD) \
  bash 25_clef_inference/run_910b.sh local
```

The default `run_910b.sh` still runs the original Transformers baseline. Both use
the existing environment; the local smoke actively blocks Transformers imports.
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
`verify_and_load_s` includes release hashing, and device events include stream
idle/enqueue gaps. No speedup or broad retrieval-quality claim is made by these
checks. Main remains the Transformers baseline; this implementation is on the
review branch `codex/clef-transformers-free`.

## Source provenance

The local implementation adapts these Apache-2.0 sources. The license is included
as `LICENSE.apache-2.0`.

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
and tokenizer remain in the separately downloaded, digest-verified release.
