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

Validation status: prepared locally; no Clef NPU result established yet.
