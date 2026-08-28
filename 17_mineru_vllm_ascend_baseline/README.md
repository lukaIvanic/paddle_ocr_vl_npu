# Experiment 17: stock MinerU vLLM-Ascend baseline

This experiment reproduces the photographed Sankalok MinerU2.5-Pro
OmniDocBench benchmark contract on one Ascend 910B2. It uses the stock vLLM
0.21.0 and vLLM-Ascend 0.21.0rc1 engine. It does not import or reuse the custom
MinerU model or scheduler from experiment 11.

The result is configuration-equivalent, not hardware-identical. The source run
used one 310P3 with CANN 8.0.0. This lane uses one 910B2 with CANN 9.0.0. Never
carry a result between chips without labeling both.

The first accepted one-page 910B2 smoke is recorded in
`VERIFIED_910B_SMOKE.md`. The exact 981-page OmniDocBench v1.0 run, 128-page
gate, evaluator results, and graph-cache postmortem are recorded in
`VERIFIED_V1_0_981_BASELINE.md`. The complete 1,651-page OmniDocBench
`v1.6_full` run, recovery, official evaluator score, SOTA comparison, and
online-serving graph-cache behavior are recorded in
`VERIFIED_V1_6_1651_BASELINE.md`. The matched 128-page
`enable_static_kernel` on/off result is recorded in
`STATIC_KERNEL_AB_128.md`.

## Source evidence

The seven supplied screenshots and reconstructed code are transcribed in
`SANKALOK_VLLM_ASCEND_OMNIDOCBENCH_SCREENSHOT_TRANSCRIPTION.md`. The claimed
source result is 981 pages in 6,144 seconds, or 0.1597 pages/s. It has no
preserved command, log, input manifest, output hashes, or accuracy score.

## Compiled and asynchronous contract

The current `compiled_async` operational default fixes these values:

```text
AsyncLLM, tensor_parallel_size=1
float16, no quantization
max_model_len=8192
gpu_memory_utilization=0.9
max_num_seqs=512
max_num_batched_tokens=16384
enforce_eager=False
enable_prefix_caching=True
enable_chunked_prefill=True
enable_npugraph_ex=True
enable_static_kernel=False
fuse_norm_quant=False
cudagraph_mode=FULL_DECODE_ONLY
cudagraph_capture_sizes=1,2,3,4,5,6,7,8,12,16,20,24,28,32
MinerUClient backend=vllm-async-engine
batch_size=0, image_analysis=True
one concurrent_two_step_extract call over the selected corpus
```

The photographed 310P source contract used `enable_static_kernel=True`.
Historical accepted runs retain that setting in their manifests. Pass
`STATIC_KERNEL=on` only when reproducing that source contract or running an
explicit A/B.

The runner also retains two compatibility corrections already established in
experiment 11. It forces tied embeddings at both Hugging Face config levels and
bypasses `mineru-vl-utils` 1.0.5's second multimodal prompt render.

The shell wrapper sets `HI_PYTHON` to the fresh experiment interpreter. CANN's
`op_compiler` otherwise invokes bare `/usr/bin/python3`, which has no NumPy in
this container and makes every requested static-kernel compile fail. Experiment
17 also owns its vLLM compile cache under
`.runtime_cache/17_mineru_vllm_ascend_baseline/`. The wrapper runs the engine
from its per-run directory so CANN's transient static-kernel build trees stay
with the other ignored run artifacts under `tmp/`.

The `eager_sync` mode reproduces the photographed comparison lane. It uses the
synchronous `LLM`, `enforce_eager=True`, no prefix cache, no chunked prefill,
and one `two_step_extract` call per page. This is a faithful comparison, not a
compilation-only ablation, because it changes the engine and scheduling too.

## Fresh environment clone

The known environment is `/workspace/venvs/mineru_pro_vllm_py312`. Clone it to
an experiment-owned path without changing the source:

```sh
cd /workspace/repos/paddle_ocr_vl_npu/17_mineru_vllm_ascend_baseline
bash clone_environment.sh
```

The clone is `/workspace/venvs/mineru_vllm_ascend_exp17_py312`. The verification
step requires this exact package contract:

```text
vllm                 0.21.0+empty
vllm-ascend          0.21.0rc1
torch                2.10.0+cpu
torch-npu            2.10.0
transformers         5.5.4
mineru-vl-utils      1.0.5
httpx-retries        0.6.0
```

Always invoke the clone through its absolute Python path. Copied activation and
pip launcher scripts can retain the source environment's shebang. The runner
does not use them.

## Run ladder

The wrapper sources `npu-setup`, records the selected physical device, rejects
physical NPU5, verifies the environment, and writes the exact command, log,
exit code, manifests, outputs, and summary under `tmp/17_mineru_vllm_ascend_baseline/`.

Start with one page:

```sh
cd /workspace/repos/paddle_ocr_vl_npu/17_mineru_vllm_ascend_baseline
LIMIT=1 bash run_npu_reproduction.sh
```

Every `compiled_async` command below starts a new engine process and captures
all 14 ACL/NPU graphs. The vLLM compile cache does not remove that process-local
capture. Do not run the following page ladder as separate compiled processes
unless the repeated setup cost is intentional:

```sh
LIMIT=10 bash run_npu_reproduction.sh
LIMIT=32 bash run_npu_reproduction.sh
LIMIT=128 bash run_npu_reproduction.sh
```

Run the controlled 128-page static-kernel A/B with the committed matrix runner:

```sh
bash run_static_kernel_ab_128.sh
```

It selects one free physical NPU once, warms the isolated static-off compile
cache with one page, then runs the same first 128 canonical OmniDocBench pages
with static kernels off and on. Normal runs default to `STATIC_KERNEL=off`.

The pull-only 310P environment and one-page compatibility smoke are specified
in `WORK_SERVER_310P_MINERU_STOCK_SMOKE.md`. The handoff verifies the exact
model, full 1,651-page dataset view, and pinned OmniDocBench repository before
loading the model.

For a staged gate without repeated capture, keep one engine process alive
across the gate and continuation. The current wrapper does not implement that
control plane. The verified v1.0 baseline used separate 128-page and 981-page
processes, and its report labels the repeated capture explicitly.

For the historical 981-page corpus, obtain its exact newline-delimited image
list and image files first:

```sh
IMAGE_LIST=/path/to/sankalok_981_images.txt \
IMAGES_DIR=/path/to/OmniDocBenchV1.0 \
LIMIT=all HASH_MODEL_FILES=1 \
bash run_npu_reproduction.sh
```

Do not call the first 981 pages of the current 1,651-page dataset an exact
reproduction. Run the current canonical dataset as a separate extension:

```sh
LIMIT=all HASH_MODEL_FILES=1 bash run_npu_reproduction.sh
```

Run the reported eager comparison in a fresh process:

```sh
MODE=eager_sync LIMIT=128 bash run_npu_reproduction.sh
```

## Timing

Model loading and engine initialization are outside `benchmark_wall_s`, as in
the photographed script. `benchmark_wall_s` starts before client construction
and image loading. It ends after inference, `json2md`, and all Markdown and JSON
writes. The primary result is `completed / benchmark_wall_s`.

The first compiled run is cold. Do not discard its graph-capture cost. Run the
same command in a new process again for a separately labeled warm-cache result.

## Acceptance gates

- Environment verification passes exactly.
- One physical 910B2 is named in the run manifest.
- `completed` equals the selected input count and `failed` is zero.
- Every selected page has Markdown and content-list output.
- The input manifest records every image size and SHA-256.
- The run records setup, image loading, inference, output, and total wall time.
- Speed is not accepted as quality evidence. Score the completed prediction set
  with the evaluator version that matches the selected OmniDocBench corpus.

Prepare a completed run for the repository's pinned OmniDocBench evaluator with
`prepare_omnidocbench_eval.py`. The tool rejects incomplete runs, missing or
extra predictions, duplicate stems, and pages absent from the selected ground
truth. It records the ground-truth, run-summary, input-manifest, and prediction
hashes. Its generated config enables text edit distance, formula edit distance
and CDM, table edit distance and TEDS, and reading-order edit distance.
