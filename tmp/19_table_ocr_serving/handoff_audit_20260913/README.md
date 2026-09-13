# Handoff audit — 2026-09-13

This records the initial audit. The subsequent real RAM/HBM measurement,
output comparison and implemented fix are recorded in
[the live-memory follow-up](../live_memory_20260913/README.md).

No runtime/model edits or new inference runs in this audit. Requirements and
the product README startup command were updated locally; not committed/pushed.
Tests remain in the research checkout, not in the proposed product bundle.

## CPU history retained until shutdown

`ContinuousRecognizer` keeps `submitted_order`, `submitted_ids` and
`completions` for its entire serving lifetime. Each completion keeps its native
token list, DecodeRequest, CPU/prefill timings and route metadata. It does not
keep the input image, rendered HTML or NPU cache after admission releases that
state. Both metrics levels retain this history; it is not a logging-queue issue.

`estimate_retained_memory.py` reconstructs the actual four dataclasses from
source, with released NPU fields set to None and fresh token integers based on
the **actual generated IDs** from the saved 1,000-table run. It uses
tracemalloc on the container's Python 3.12.13, without importing torch or running
inference. Scalars approximate runtime sharing; this is an estimate of retained
Python allocations, not a live RSS/HBM measurement or allocator accounting.

- Mean output: 399.102 native tokens; largest: 3,112.
- Token-list component: about 10,377 bytes/request.
- Full retained-history estimate: **12,579 bytes/request**.
- 1,000 records: 12,557,787 bytes; 2,000: 25,178,286; 3,000: 37,735,969.
- An additional **10 GB (10,000,000,000 bytes)**: about 795,000 completed tables.
- At 1 / 3 / 6 completed requests/s: **220.8 / 73.6 / 36.8 hours**.
- At 6/s: about 272 MB/hour. 10 GiB instead would take about 39.5 hours.

This excludes baseline model/process memory, active requests, logging, native
allocator overhead and fragmentation. It extrapolates this table distribution,
not arbitrary longer text/formula outputs. Actual RSS can differ substantially.
The outstanding fix is to update summary counters at completion and stop
retaining completed records/IDs; none of this requires changing NPU execution.

To reproduce with the same Python environment, run the script on the container
with `--repo /workspace/repos/paddle_ocr_vl_npu` and
`--results /workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/integrated_runtime_20260913/poisson1000/cached/b8/qps6/measured/results.jsonl`.
For this audit it was piped over SSH into `python -` (CPU-only, no multiprocessing).

## Requirements audit

AST scan of all six production scripts, including function-local imports:
`torch`, `torch_npu`, `torchair`, `numpy`, `PIL`, `kornia_rs`, `tokenizers`,
`safetensors`. No production imports of Transformers, OpenCV or PyYAML remain.

The container reports torch 2.10.0+cpu, torch-npu 2.10.0, numpy 1.26.4,
Pillow 12.2.0, kornia-rs 0.1.14, tokenizers 0.22.2 and safetensors 0.8.0.
Requirements now name these directly. Removed the three unused packages.
TorchAir lives inside the installed torch-npu tree, not as a separate pip
distribution in this environment. CANN/driver remain system dependencies.
The exact aarch64 Python 3.12 CPU wheel is listed in the
[official PyTorch wheel index](https://download.pytorch.org/whl/cpu/torch/).
No packages were installed, removed or upgraded during this audit.
An installed-environment `pip install --dry-run --no-index -r /dev/stdin`
confirmed every revised requirement and its dependencies are already satisfied.
This is not a fresh-install test. PyYAML can still arrive transitively through
tokenizers/huggingface-hub; it is no longer a direct dependency of our scripts.

## Manual validation

[MANUAL_TESTS.md](MANUAL_TESTS.md) gives SSH, foreground server, requests,
saved Poisson100/1000, memory watching, asynchronous Python and error checks.
Mixed crop types share one running server; changing vocabulary requires a
restart. Python uses its own worker after HTTP is stopped, not a second worker
competing on the same card. These new checks remain unrun.
