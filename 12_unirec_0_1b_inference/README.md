# Experiment 12: UniRec 0.1B Inference

This experiment compares the official UniRec implementations with the local
implementation that we can modify and optimize. It also contains a thin runner
for the official OpenDoc full-page pipeline.

## Files

- `modeling_optimized_unirec.py`: self-contained UniRec encoder, decoder, image
  processor, static KV cache, weight loader, and TorchAir decode integration.
- `run_optimized.py`: local eager or compiled inference.
- `run_original_transformers.py`: exact Transformers model and processor code
  bundled with the official UniRec-0.1B-1217 checkpoint.
- `run_official_opendoc.py`: OpenOCR's official PP-DocLayoutV2 + UniRec page
  pipeline with explicit checkpoint and recognizer-device selection.
- `run_opendoc_custom_unirec.py`: official OpenDoc ONNX layout, crop,
  postprocessing, and output path with only UniRec crop inference replaced by
  the local eager NPU implementation. Its comparison mode feeds each exact
  in-memory crop to stock ONNX and local NPU inference and writes a JSONL trace.
- `run_opendoc_batched_unirec.py`: unmodified OpenDoc layout/crop/assembly
  semantics with a repository-owned cross-page crop queue. Each crop keeps an
  exact B1 vision/decoder prefill. The runner supports fixed padded cohorts and
  a fixed-arena continuous decoder.
- `continuous_unirec.py`: continuous decode scheduler. Each physical batch row
  owns its cache position; an EOS or length-complete row is replaced by the
  next B1-prefilled request without waiting for the other rows.

The local model implementation is copied without architectural changes from
`unirec_research/03_compiled_decode_single_batch` at commit `4b9a9ab`.

## Verified Blue Zone inputs

```text
Model:       /workspace/models/unirec_0_1b_1217
Official Python: /workspace/venvs/unirec1217_npu_py312/bin/python
Custom Python:   /workspace/venvs/unirec_npu_py312/bin/python
Official Transformers: 4.49.0
```

The 1217 model directory is the official `topdu/unirec_0_1b` checkpoint at revision
`d2469d0f50992a380240266fe169b982ea940615`. This is the table-capable
UniRec-0.1B-1217 release. The installed `model.safetensors` is 535,797,520
bytes and has SHA-256:

```text
1a080d683731d2bdae5a4b8c538160d2e8b1733f44de25cb75f264406db8d746
```

This matches the official Hugging Face LFS metadata.

OpenOCR's documented OpenDoc score-reproduction path uses the distinct
`topdu/unirec-0.1b` `model.pth` checkpoint. It is also table-capable in the
OpenDoc pipeline and is not numerically identical to the 1217 safetensors
checkpoint. Use `/workspace/models/unirec-0.1b/model.pth` when reproducing the
published OpenDoc OmniDocBench score. Do not interchange the two checkpoints
when comparing exact outputs or reported metrics.

## Setup

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
```

Create the isolated official environment once. It inherits the installed
Torch-NPU runtime but keeps the checkpoint's Transformers version separate
from vLLM:

```sh
/usr/local/python3.12.13/bin/python3 -m venv --system-site-packages \
  /workspace/venvs/unirec1217_npu_py312
/workspace/venvs/unirec1217_npu_py312/bin/python -m pip install \
  -r 12_unirec_0_1b_inference/requirements-official.txt
```

The commands below use `crops/crop_01_text_block_en.png`. Without an `--image`
argument, each runner uses the first six repository crop images.

## Local implementation

Eager:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16 --decode-mode eager
```

Cached TorchAir decode:

```sh
/workspace/venvs/unirec_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_optimized.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16 --decode-mode compiled \
  --compile-backend torchair
```

## Official bundled Transformers reference

```sh
/workspace/venvs/unirec1217_npu_py312/bin/python \
  12_unirec_0_1b_inference/run_original_transformers.py \
  --model-path /workspace/models/unirec_0_1b_1217 \
  --image crops/crop_01_text_block_en.png \
  --device npu:0 --dtype float16
```

## Official OpenDoc full-page pipeline

OpenOCR provides two full-page implementations:

- `openocr.py --task doc`: the packaged ONNX path. It runs PP-DocLayoutV2 and
  up to four UniRec crop workers. This path is validated in the Blue Zone ARM
  container.
- `tools/infer_doc.py`: the PaddleX + PyTorch path used by the OpenOCR and
  OmniDocBench source trees. The two repositories contain byte-identical
  scripts. PaddlePaddle 3.2.0 currently segfaults while loading the Paddle
  PP-DocLayoutV2 parameters on this ARM container, before page inference.

The official ONNX UniRec export produced exact text parity with the OpenDoc
`model.pth` checkpoint on the same table crop through the tested 128-token
generation. Use the committed full-run wrapper for OmniDocBench v1.6:

```sh
bash 12_unirec_0_1b_inference/run_official_opendoc_omnidocbench.sh
```

The wrapper runs the official CLI over the complete image directory and saves
the exact command, continuous log, exit code, wall time, JSON, and Markdown.
Each page is written immediately. On restart, the wrapper constructs a
symlink-only input directory and processes only pages that do not already have
both final artifacts. Set `RESUME=0` only when deliberate full recomputation is
required.
The current official OmniDocBench v1.6 target is:

```text
Overall:       90.67
Text Edit:      0.049
Formula CDM:   93.02
Table TEDS:    83.88
Reading Edit:   0.140
```

The 90.57 figure in OpenOCR's document is the older OmniDocBench v1.5 result.
The installed 1,651-page dataset is v1.6.

The source-pipeline adapter remains available for environments where the
Paddle predictor loads correctly. It preserves OpenOCR's pipeline and places
only UniRec on the NPU:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/run_official_opendoc.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/unirec-0.1b/model.pth \
  --input /workspace/datasets/OmniDocBench/images \
  --output-dir tmp/12_unirec_0_1b_inference/opendoc_reference \
  --recognizer-device npu:0
```

Use `--limit 1` for a smoke. Omit `--limit` for the complete image directory.
Neither runner replaces the OmniDocBench evaluator. Score the Markdown outputs
with the standard evaluator after inference.

## Official OpenDoc with local eager UniRec

Use one recognition thread while the custom path is being validated. Comparison
mode runs stock ONNX and local eager NPU UniRec on every identical crop, returns
the local result to the unchanged OpenDoc assembler, and records preprocessing,
token, raw-text, and label-postprocessed parity:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/run_opendoc_custom_unirec.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/unirec-0.1b \
  --layout-model /root/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx \
  --stock-encoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx \
  --stock-decoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx \
  --stock-tokenizer-mapping /root/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json \
  --input /workspace/datasets/OmniDocBench/images \
  --output-dir tmp/12_unirec_0_1b_inference/opendoc_custom_compare \
  --mode compare --device npu:0 --dtype float16 --limit 1
```

Use `--mode custom` for timing without the stock recognizer call. The page path
still fixes `max_parallel_blocks=1`; no crop-level concurrency is introduced.
Add `--decode-mode compiled --compile-backend torchair` to compile only the
static-cache decoder step. Image preprocessing, the vision encoder, and decoder
prefill remain eager. Compiled graphs are cached under
`.runtime_cache/12_unirec_0_1b_inference/opendoc_model_pth_decode` by default.

To replace the CPU ONNX layout detector with eager PP-DocLayoutV2 on NPU while
keeping the same OpenDoc layout contract, add:

```sh
  --layout-backend transformers_npu \
  --layout-transformers-model /workspace/models/PP-DocLayoutV2_safetensors \
  --layout-dtype float32
```

The NPU adapter deliberately preserves OpenDoc's original 25-class labels,
overlap filtering, reading-order sort, block numbering, and downstream crop
assembly. `float32` is the parity-first default; `float16` is an explicit
performance experiment.

## Cross-page decode scheduling

The fixed-cohort runner batches decode only. It does not pad images or alter
the vision encoder. Crops are prepared and prefetched one at a time using the
same path as B1, then their static self/cross KV caches are concatenated across
page boundaries. All rows decode until the longest row finishes. Rows that
already reached EOS emit padding EOS tokens and are excluded from effective
token throughput.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/run_opendoc_batched_unirec.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/unirec-0.1b \
  --layout-model /root/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx \
  --layout-backend transformers_npu \
  --layout-transformers-model /workspace/models/PP-DocLayoutV2_safetensors \
  --layout-dtype float32 \
  --stock-encoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx \
  --stock-decoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx \
  --stock-tokenizer-mapping /root/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json \
  --input /workspace/datasets/OmniDocBench/images \
  --output-dir tmp/12_unirec_0_1b_inference/opendoc_batched_b4 \
  --device npu:0 --dtype float16 --max-length 256 \
  --decode-mode compiled --compile-backend torchair \
  --decode-batch-size 4 --limit 32
```

This runner does not edit or patch the OpenOCR checkout. It imports the stock
layout and page-assembly helpers and owns only the scheduling boundary. The
final partial cohort is padded to `--decode-batch-size`. Reports preserve raw
physical decode slots, effective real decode tokens, and padding slots.

Add `--decode-scheduling continuous` to retain the same physical decode graph
while hot-swapping completed rows:

```sh
  --decode-batch-size 4 --decode-scheduling continuous
```

Continuous mode keeps exact B1 image and decoder prefill semantics. It copies
the replacement request's complete static self/cross KV rows into the finished
slot, resets only that row's cache position, and resumes the fixed-shape graph.
Requests may complete out of order; pages are still emitted in input order as
soon as all their crops finish. The initial implementation performs replacement
prefill synchronously between decode iterations. It does not yet overlap NPU
prefill with decode.

Set `UNIREC_STATIC_CACHE_LEN` to select the static self-KV capacity. Set
`UNIREC_STATIC_CROSS_CACHE_LEN` to select a smaller fixed cross-KV capacity.
When the latter is nonzero, crops whose normal encoder sequence exceeds that
capacity are omitted instead of resized or truncated. The remaining crops keep
their exact normal preprocessing, and the cross cache is padded to the exact
requested length. This is a throughput-only experiment because omitted crops
reduce page quality.

## Layout detector lab

`layout_detector_lab.py` isolates the same eager PP-DocLayoutV2 NPU adapter
used by the full runner. It runs sequential B1 pages and excludes recognition,
crop construction, and page assembly. The report separates file read, image
decode, BGR-to-RGB conversion, processor resize/normalize, input H2D, model
forward, Hugging Face box decode, result D2H, Python result construction,
overlap filtering, and reading-order labeling. One warmup page is excluded by
default.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/layout_detector_lab.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/PP-DocLayoutV2_safetensors \
  --input /workspace/datasets/OmniDocBench/images \
  --device npu:0 --dtype float32 --limit 32 \
  --output tmp/12_unirec_0_1b_inference/layout_detector_lab/result.json
```

The JSON contains per-page dimensions, box counts, stage times, aggregate
totals, means, medians, p90 values, and detector pages/s. Device
synchronization is enabled only in this profiling lane so asynchronous NPU work
is charged to the stage that submitted it.

## Guarded-atlas vision lab

`vision_atlas_lab.py` tests a fixed-shape representation for the spatial
FocalSVTR stages. It places variable crop feature maps in one 2D atlas and
surrounds each crop with a zero guard. The crop mask is reapplied after every
focal convolution. This preserves each crop's independent zero-padding
boundary while all crops use one fixed graph shape.

The default lab lane targets stage 2, which contains nine focal blocks. It uses
a 64x192 atlas, a three-cell guard, first-fit-decreasing placement, and at most
16 crops per atlas. A static permutation gathers a padded flat token reservoir
into the atlas. A second permutation returns the output to crop-token order.
The compiled graph includes both permutations. The lab also times a
pessimistic integration path that copies today's separate crop tensors into
the reusable reservoir before each stage call.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/vision_atlas_lab.py \
  --stage 2 --atlas-height 64 --atlas-width 192 --guard 3 \
  --max-members 16 --limit 186 --packing ffd \
  --routing permutation --execution torchair \
  --cache-dir .runtime_cache/12_unirec_0_1b_inference/vision_atlas_lab \
  --output tmp/12_unirec_0_1b_inference/vision_atlas_lab/result.json
```

The validated 32-page shape corpus contains 186 crops and 54,880 real stage-2
tokens. The atlas path used 14 fixed graph calls and 172,032 physical atlas
cells. Median timings on Ascend 910B2 were:

```text
Independent per-crop stage 2:       2041.55 ms
Compiled routed atlas stage 2:       147.02 ms
Separate-crop reservoir assembly:      7.64 ms
Combined routed path:                 154.66 ms
Combined speedup:                      13.20x
```

The compiled lane's mean per-crop mean-absolute difference was 0.00313. Its
worst relative L2 difference was 0.00787, worst cosine similarity was 0.999969,
and worst maximum absolute difference was 1.125. These are intermediate-stage
statistics on deterministic random feature tensors, not an end-to-end OCR
accuracy result.

The 1.887-second stage-2 saving is a measured upper bound for integration. It
would reduce the earlier 32-page 39.84-second pipeline to approximately 37.95
seconds if the surrounding schedule remains unchanged. The lab does not yet
replace stages 0, 1, or 3, the patch stem, stage downsampling, or the final
projection. Use `--routing prebuilt_atlas` only to measure the stage-compute
upper bound without permutation or separate-crop assembly costs.

## Full-page frontend IPC lab

`frontend_ipc_lab.py` isolates the process-to-coordinator transfer after page
decode, layout, and crop construction. It does not load the layout or
recognition model. It replays exact page and crop array shapes from a recorded
recognition trace and compares four transport mechanisms:

- descriptor-only metadata;
- the current list of independently pickled NumPy arrays;
- one packed NumPy arena that is still pickled through the queue;
- one parent-owned POSIX shared-memory arena with descriptor-only queue traffic.

The default trace is the hard 128-page, 7,325-crop run. Worker setup and
shutdown are excluded from measured transport wall time. `--consumer sample`
measures zero-copy receipt and validates representative bytes in every segment.
`--consumer copy` additionally copies every segment into coordinator-owned
memory, which prices a design that cannot retain shared-memory leases.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/frontend_ipc_lab.py \
  --workers 8 --max-inflight 16 \
  --modes metadata,pickle_arrays,pickle_arena,shared_memory_arena \
  --consumer sample \
  --output tmp/12_unirec_0_1b_inference/frontend_ipc_lab/result.json
```

Use `--limit-pages 2 --crop-only` for a fast development smoke. The shared
memory lane requires a writable POSIX shared-memory mount. Blue Zone provides
128 GiB at `/dev/shm`.

Validated on the hard 128-page trace (7,325 crops, 7.866 GiB including page
images), with eight workers and 16 pages maximum in flight:

```text
metadata only:                         0.108 s
pickled array list:                   20.001 s
one packed but still pickled arena:   19.006 s
shared-memory arena, zero-copy view:   2.364 s
shared-memory arena, parent copy:      7.295 s
```

Every lane passed shape and sampled-byte parity. Packing objects without
changing transport saved only 5%. Shared memory was 8.46x faster than the
current pickled-array transport. Copying all 7.866 GiB back into parent-owned
memory cost 4.479 seconds and reduced that gain to 2.74x. This makes page-scoped
shared-memory leases the preferred integration direction.

The page-scoped lease design was then integrated into the hard 128-page E2E
runner. Each worker packs its decoded page and recognition crops into one
shared arena. The coordinator unlinks the public shared-memory name when it
attaches, retains the mapping through OCR and asynchronous output writing, and
closes it when the page write completes. All 7,325 crop token IDs and texts
matched both earlier paths exactly, and `/dev/shm` returned to its idle baseline.

```text
ordinary pickled full frontend: 278.717 s, 0.4592 page/s
shared-memory full frontend:    253.215 s, 0.5055 page/s
saved:                           25.502 s, +10.1% page throughput
```

The process phase fell from 29.921 to 7.447 seconds. Mean queue-delivery delay
fell from 4.433 to 0.093 seconds per page. The remaining frontend boundary is
6.046 seconds of coordinator-side `PageRequest` and PIL-image materialization.

Recognition-specific CPU preprocessing was then moved into the same eight
workers. Workers run the existing PIL bicubic resize, float32 rescale and
normalization path, and place ready BCHW tensors in the page arena. The
coordinator retains only the float32 H2D copy and fp16 conversion. On the same
hard 128-page set:

```text
shared crops, coordinator preprocessing: 253.215 s, 0.5055 page/s
shared crops, worker preprocessing:      198.072 s, 0.6462 page/s
saved:                                    55.142 s, +27.8% page throughput
```

Coordinator recognition preparation fell from 67.077 to 3.341 seconds. The
worker frontend wall rose from 7.447 to 17.664 seconds, and parent descriptor
materialization rose from 6.046 to 7.253 seconds. The move therefore spends
11.424 additional frontend seconds to remove 63.736 seconds of serialized
recognizer preparation. All 7,325 crop sizes, processed sizes, encoder-token
hints, token IDs and texts matched exactly. The measured page arenas occupied
16.826 GB and returned to the idle shared-memory baseline after the run.

## Artifacts

- Run JSON: `tmp/12_unirec_0_1b_inference/`
- TorchAir and compatibility caches:
  `.runtime_cache/12_unirec_0_1b_inference/`

`source npu-setup` is mandatory. It selects a free physical NPU and exposes it
as logical `npu:0`.

## Validation

Validated on Ascend 910B2 with the table crop
`crops/crop_05_table_rwkv_dims.png`, BF16, JIT compile disabled, and a 64-token
limit. The official bundled Transformers implementation and the local eager
implementation produced exact token and text parity across all 64 returned
tokens. Both generated native HTML table markup. The official lane produced
53.2 generated tokens/s. The local eager decode lane produced 115.5 decode
tokens/s.
