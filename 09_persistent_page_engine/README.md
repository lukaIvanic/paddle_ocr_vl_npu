# Experiment 09: persistent page-engine baseline

Experiment 09 is the active continuation of the validated Experiment 08
runtime. It starts from the same real-layout, compiled-prefill, continuous-
decode implementation, but gives the next architecture work a clean home.
Experiment 08 and its NPU reports remain unchanged as the historical evidence
for that baseline.

The immediate design target is a persistent custom request engine behind a
bounded rolling page frontend. Pages should enter continuously, recognition
requests from different pages should share one scheduler, and each completed
page should be emitted immediately. The bounded frontend must provide
backpressure: it should neither submit the entire benchmark at once nor create
fixed page cohorts that temporarily starve cross-page batching.

The faithful PaddleX path now removes its former recognition-batch barrier.
PaddleX still performs the official layout preparation and page assembly, but
all prepared crops enter one run-scoped recognizer schedule and each page is
emitted when its own final crop completes.

The baseline keeps both models resident in one Python process and executes this
path:

```text
stream of full PIL pages
  -> lazy real PP-DocLayoutV3 inference on NPU
  -> reading-ordered layout regions and page collectors
  -> crop and prompt routing into one run-scoped request source
  -> one PaddleOCR-VL prefill at a time
       CPU image/prompt preprocessing
       eager native-resolution patch + position embedding
       shared vision-prefill stage, TorchAir + bucket padding by default
       eager projector
       shared text-prefill stage into a static KV cache,
         TorchAir + bucket padding by default
       eager LM head and first-token argmax
       ready B=1 KV state
  -> bounded cross-page ready reservoir
       high watermark = 4B prepared requests
       low watermark = B prepared requests
       refill in bursts rather than after every completion
  -> persistent power-of-two compiled decode arena
       fill free slots from ready B=1 KV states
       run one autoregressive iteration
       retire EOS/length-complete requests
       hot-swap the next ready KV prefix into each freed slot
       D2H tokens and detokenization
  -> route completions to their page collectors
  -> emit each page immediately when all of its regions finish
```

Vision and text prefill remain sequential B=1. Each stage has one
compiler-safe model path shared by eager and compiled execution. Padding is a
separate input-shaping policy: `none` keeps the real shape, while `bucket`
pads to a configured static shape. `auto` selects bucket padding for TorchAir
and no padding for eager execution. The validated default uses 32 static
TorchAir vision shapes: 32-token steps through 512, 64-token steps through
1024, and 128-token steps through 2048. Only the encoder layers and final
LayerNorm are inside the vision stage; patch embedding, position interpolation,
and the projector stay eager.
Real and dummy rows are isolated by an attention mask, and real rows are sliced
before the projector. Crops above 2048 rows use the faithful eager path without
padding.

Text token embedding and image scatter stay eager. The text transformer uses a
measured static-bucket profile. A padded call may populate physical bucket rows
in its private scratch KV cache, but admission copies only the valid real prefix
into the decode arena and the next-cache position remains the real prompt
length. Inputs above the largest text bucket use the same stage eagerly without
padding.

There are two named operating profiles. The small standalone full-page CLI uses
TorchAir B=4, cache length 2048, and a 768-token cap. The official full
OmniDocBench runner uses B=16, cache length 8192, and PaddleX's 4096-token cap.
`raw_eager` remains an explicit correctness control for the core compiled
boundaries; it is not a competing production path.

Prefills are produced lazily for one run-scoped decode scheduler instead of
draining decode at every page boundary. Decode owns one persistent fixed-shape arena. Slot
indices stay stable; a finished request is replaced in place without moving
other active requests or rebuilding the batch. Admission copies only the valid
prompt KV prefix, while stale cache tails remain safely hidden by each row's
cache position. The ready reservoir is internal and bounded, so the pipeline
does not materialize every page's NPU KV caches before decode.

In both runners, a page is an input/output aggregation boundary rather than a
scheduling boundary. Each request carries its page identity through the engine;
per-page collectors restore reading order and can emit independently.

## What is faithful in this first cut

- Layout is a real `PP-DocLayoutV3` model call through the official
  Transformers implementation and safetensors, not GT or cached boxes.
- Transformers' PP-DocLayoutV3 postprocessor supplies thresholding, polygons,
  and learned reading order.
- Prompt routing follows the official PaddleX PaddleOCR-VL pipeline: table,
  chart, non-number formula, spotting, and seal receive their specialized
  prompts; other labels receive `OCR:`.
- Official v1.6 defaults are retained for image/chart/seal blocks: image blocks
  are not recognized, and chart/seal recognition is opt-in.
- The recognizer is the corrected local PyTorch model from Experiment 05.
  Vision and text-transformer prefill default to compiler-safe static TorchAir
  buckets; eager overflow preserves unpadded behavior outside those profiles.
- Sampled-token D2H uses a second NPU stream, pinned two-row host ring, and
  queue-depth-one control. A request can execute one look-ahead graph call;
  slot epochs discard that old result after the slot is reused.

`scripts/run_offline_e2e.py` intentionally keeps a smaller diagnostic page-preparation
path and its reading-order text is not an OmniDocBench prediction. For faithful
page assembly, `scripts/run_omnidocbench_paddlex.py` keeps the official PaddleX
v1.6 layout filtering, crop/merge policy, table and formula handling, result
objects, and Markdown conversion. Its small page bridge skips only PaddleX's
synchronous VLM batch call and routes the same prepared crops through this
optimized engine as one continuous run.

The full-benchmark runner installs a narrow PP-DocLayoutV3 mask guard.
Transformers can otherwise call OpenCV with a zero-width mask crop when a thin,
positive-size detection collapses after scaling and rounding. Valid detections
still use the installed Transformers method unchanged; only a collapsed slice
uses that detection's integer bounding rectangle. Every fallback is recorded in
`layout_mask_guard.json`.

## Timing model

`run.json` reports four different scopes explicitly:

- Setup: layout-model load, recognizer-model load, optional weight-format probe,
  compile-wrapper creation, and the first compiled call. Setup is excluded from
  page throughput.
- Run wall: first page start through the last page emission. This is the E2E
  throughput denominator; overlapping page latencies are never summed for
  throughput.
- Per-page latency: that page's image load through its completion emission.
- `device_stage_s`: NPU-event execution time for vision/text-prefill substages.
  These values diagnose accelerator work and are not interchangeable with host
  wall latency.

Raw decode tok/s counts every `batch_size * graph_calls` arena slot, including
idle rows and completion look-ahead. Effective decode tok/s counts only real
generated tokens after the prefill-produced first token, including EOS. Their
denominator is conservatively the larger of exclusive decode-control host wall
and serialized decode-plus-admission device time. The JSON also exposes full
run-scoped scheduler wall, lazy ready-source wall, refill count, reservoir
bounds, device timing, idle/look-ahead slots, and copied KV-prefix bytes. E2E
output tok/s includes each request's first token and EOS and divides by run wall.

## Blue Zone run

The smallest complete model path is the one-crop example. It does not construct
the serving engine or scheduler:

```sh
PYTHONPATH=09_persistent_page_engine \
/usr/local/python3.12.13/bin/python3 -m paddleocr_vl.model.example \
  --model /workspace/models/PaddleOCR-VL-1.6 \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:"
```

The normal command only needs its page and model locations because the
optimized decode and vision profile is now the CLI default:

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup

/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/run_offline_e2e.py \
  --image "/workspace/datasets/OmniDocBench/images/PPT_The Right Moves_page_024.png" \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6
```

Experiment 09 always uses the logical device `npu:0` selected by `npu-setup`
and always calls `torch.npu.set_compile_mode(jit_compile=False)`. There is no
device-resolution or JIT-mode option in this experiment.

For a repeatable one-page validation, use `scripts/run_npu_smoke.sh`. Environment
overrides remain available for deliberate controls, including `BATCH_SIZE`,
`DECODE_BACKEND`, `VISION_BACKEND`, `VISION_PADDING`, `TEXT_BACKEND`,
`TEXT_PADDING`, `VISION_BUCKETS`, and `TEXT_BUCKETS`. Omitted bucket overrides use the single source of truth in
`paddleocr_vl/serving/runtime_defaults.py`.

Recognition uses the model's `min_pixels` and `max_pixels` by default. Pass
`--preprocessor-min-pixels N` to override only the recognition-crop minimum;
the model's maximum remains unchanged. The model default, requested override,
effective bounds, resize factor, and nominal minimum image-token count are
recorded under `configuration.preprocessor` in `run.json`.

The default `--vision-backend torchair` builds or loads one B=1 static graph for
every configured bucket during recognizer setup. The first uncached run is therefore
compilation-heavy; per-bucket cache paths and first-call times are recorded in
`configuration.vision_prefill` and `setup_timing_s.vision_runtime_setup`.
Subsequent runs reuse the GE caches under
`.runtime_cache/09_persistent_page_engine_vision_torchair/`. The compiled boundary is
currently the manual-attention path; unset
`PADDLE_OCR_VL_VISION_ATTENTION` or set it to `manual`.

Each recognition result records its real token count, selected physical bucket,
padding, and execution route. Aggregate JSON reports bucket counts and the
overall useful-token fraction so padding cost is visible rather than folded
into a single throughput number.

The default artifact directory is timestamped under
`tmp/09_persistent_page_engine/`. It contains `run.json`, per-page reading-order text,
and an annotated layout image. Pass `--save-crops` only when the actual crop
images are useful. `--max-regions N` is a debug-only partial-page mode and is
recorded as such in the JSON.

`--batch-size` accepts 1, 2, 4, 8, and other powers of two. Additional
`--image` arguments enter the same cross-page scheduling domain by default.
Each page is printed and made available to callbacks as soon as its own regions
finish; the engine does not wait for the whole image list before emitting it.

## Runtime code map

`paddleocr_vl/model/` is the standalone crop model. It can preprocess and
recognize one crop, but has no request queue, page, layout, PaddleX, or
OmniDocBench concepts.

- `paddleocr_vl/model/modeling.py`: faithful model weights/math and the
  connector that assembles the three inference stages.
- `paddleocr_vl/model/vision_prefill.py`: one compiler-safe vision encoder stage,
  exact/bucket input preparation, and eager/TorchAir execution wrapper.
- `paddleocr_vl/model/text_prefill.py`: one compiler-safe text transformer stage,
  exact/bucket preparation, in-place prefix KV writes, and execution wrapper.
- `paddleocr_vl/model/text_decode.py`: one static decode-step stage, its execution
  wrapper/cache identity, and the warmed persistent decode cache.
- `paddleocr_vl/model/preprocessing.py`: crop resize/patchify and multimodal
  prompt construction.
- `paddleocr_vl/model/example.py`: minimal executable one-crop MVP using the
  model directly, with no serving dependency.

`paddleocr_vl/serving/` adds persistent multi-request inference on top of the
model package. The model package never imports it.

- `paddleocr_vl/serving/engine.py`: one persistent model instance, sequential multimodal prefill,
  compact prefilled state, and result materialization.
- `paddleocr_vl/serving/continuous_decode.py`: bounded ready reservoir and persistent
  decode-slot scheduler.
- `paddleocr_vl/serving/types.py`: crop requests, recognition results, and
  decode-schedule contracts.
- `paddleocr_vl/serving/runtime_defaults.py`: measured serving-runtime profiles.

`pipeline/` owns full-page concerns and depends on `paddleocr_vl/`, never the
reverse.

- `pipeline/layout.py`: PP-DocLayoutV3 inference and normalized layout regions.
- `pipeline/page_pipeline.py`: lazy page/layout/crop routing and page completion.
- `pipeline/paddlex_page_bridge.py`: official PaddleX page preparation and
  assembly around one cross-page recognition schedule.
- `pipeline/layout_mask_guard.py`: PP-DocLayout empty-mask fallback and telemetry.
- `pipeline/omnidocbench_defaults.py`: validated full-benchmark execution profile.
- `pipeline/types.py`: boxes, layout regions, page results, and run serialization.

`scripts/` contains serving and pipeline composition roots. It includes the
diagnostic page runner, official PaddleX/OmniDocBench runner, NPU smoke wrapper,
and focused probes. `utils/` contains only shared timing and metric helpers.

The production runtime packages do not import `scripts/` entrypoints or probes.
Those scripts consume the same preprocessing and model-stage modules as the
E2E engine, so diagnostic code cannot silently become a runtime dependency or
invalidate a compiler cache merely because a probe changed.
The recognizer also constructs and warms all three compiled boundaries under
`torch.inference_mode()`, matching real request execution and keeping TorchAir's
dispatch-key guards stable across warmup and serving.

The measured 910B validations remain with Experiment 08 rather than being
duplicated here. Its
[`NPU_FULL_PAGE_RESULT.md`](../08_offline_e2e_b1/NPU_FULL_PAGE_RESULT.md)
records the original B=1 path, while
[`NPU_BATCHED_DECODE_RESULT.md`](../08_offline_e2e_b1/NPU_BATCHED_DECODE_RESULT.md)
records padded fixed B=2 and B=4 decode. Those documents predate the continuous
scheduler and remain historical comparison points.

The persistent-slot implementation and its exact parity/performance comparison
are recorded in
[`NPU_CONTINUOUS_DECODE_RESULT.md`](../08_offline_e2e_b1/NPU_CONTINUOUS_DECODE_RESULT.md).

The five-page B=4 comparison of the model-default `112896` floor against the
half-area `56448` override is recorded in
[`NPU_MIN_PIXELS_RESULT.md`](../08_offline_e2e_b1/NPU_MIN_PIXELS_RESULT.md).

The same-NPU comparison of manual vision attention against eager BNSD
PromptFlashAttention is recorded in
[`NPU_PROMPTFA_RESULT.md`](../08_offline_e2e_b1/NPU_PROMPTFA_RESULT.md).

The bucketed TorchAir vision-encoder integration, exact eager/compiled token
parity control, and full-page B=4 result are recorded in
[`NPU_COMPILED_VISION_RESULT.md`](../08_offline_e2e_b1/NPU_COMPILED_VISION_RESULT.md).
