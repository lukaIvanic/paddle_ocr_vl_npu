# Clef inference: Transformers baseline first

Scope: Cloudflare/clef-flash (9B), text-only, BF16, one Ascend 910B. This
experiment does not target 310P. The local Transformers-free replacement and
optimizations are later stages, not implemented here.

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

Next stage: a simple local Transformers-free implementation, using these saved
inputs, logits and answers as the reference. No local replacement or optimization
has been validated by this smoke.
