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

The production path no longer imports PaddleX or PaddleOCR. Experiment 09 owns
the fixed PaddleOCR-VL 1.6 page contract directly: PP-DocLayoutV3 loading and
inference, detector geometry, crop/merge policy, prompt routing, result
assembly, compact JSON, and Markdown formatting. All prepared crops still enter
one run-scoped recognizer schedule and each page is emitted when its own final
crop completes.

The baseline keeps both models resident in one Python process and executes this
path:

```text
stream of full PIL pages
  -> lazy real PP-DocLayoutV3 inference on NPU
  -> reading-ordered layout regions and page collectors
  -> crop and prompt routing into one run-scoped request source
  -> one PaddleOCR-VL prefill group at a time
       CPU image/prompt preprocessing
       eager per-crop native-resolution patch + position embedding
       optionally pack currently ready crops into one shared vision-tower call
       split the packed output back into crop order
       per crop: eager projector and a leased row in a zero-once KV arena
       optionally pack the group's independent text prompts into block-diagonal
         B=1 text-prefill calls, then split their KV prefixes back to each crop
       eager LM head, first-token argmax, and ready B=1 KV-arena view
  -> bounded cross-page ready reservoir
       high watermark = 4B prepared requests
       low watermark = B prepared requests
       refill in bursts rather than after every completion
  -> persistent power-of-two compiled decode arena
       fill free slots with one full-cache ForeachCopy from ready KV rows
       run one autoregressive iteration
       retire EOS/length-complete requests
       hot-swap the next ready KV row into each freed slot
       D2H tokens and detokenization
  -> route completions to their page collectors
  -> emit each page immediately when all of its regions finish
  -> enqueue page artifacts to one bounded, ordered background writer
       persist Markdown and compact JSONL without blocking decode scheduling
       propagate writer failures and drain all pending pages before E2E ends
```

Vision packing and text packing are independent switches. With text packing
disabled, text prefill remains sequential B=1 even when vision is packed.
`--text-packing production_group` applies best-fit decreasing only within the
already selected vision production group, routes each pack to the smallest of
128/256/512/1024, resets MRoPE per segment, and uses a block-diagonal causal
mask. The compiled graph writes a scratch packed KV cache; the valid prefixes
are then copied into leased rows of a persistent zero-initialized KV arena.
Prompts above 1024
tokens retain the normal individual path. Crop results preserve their original
order. Each stage has one
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
in its private scratch KV cache. Its valid prefix is written into a zero-once
private-arena row; admission copies that full fixed cache with one ForeachCopy,
while the next-cache position remains the real prompt length. Inputs above the
largest text bucket use the same stage eagerly without padding.

There are two named operating profiles. The small standalone full-page CLI uses
TorchAir B=4, cache length 2048, and a 768-token cap. The full OmniDocBench
runner uses B=32, cache length 4096, and a 4096-token cap.
`raw_eager` remains an explicit correctness control for the core compiled
boundaries; it is not a competing production path.

Prefills are produced lazily for one run-scoped decode scheduler instead of
draining decode at every page boundary. Decode owns one persistent fixed-shape
arena, and queued prefills lease rows from a second zero-once arena sized for
the ready reservoir plus one production window. Slot indices stay stable; a
finished request is replaced in place without moving other active requests or
rebuilding the batch. Admission copies the full initialized KV row with one
ForeachCopy. The real cache position still hides finite stale tail values.
Stream events guard each returned prefill row before reuse.

In both runners, a page is an input/output aggregation boundary rather than a
scheduling boundary. Each request carries its page identity through the engine;
per-page collectors restore reading order and can emit independently.

## What is faithful in this first cut

- Layout is a real `PP-DocLayoutV3` model call through the official
  Transformers implementation and safetensors, not GT or cached boxes.
- Its fixed-shape, batch-one NPU forward is replayed through a captured NPU
  graph. Preprocessing, thresholding, mask-derived crop geometry, and learned
  reading order remain outside the graph and preserve the reference requests.
- Transformers' PP-DocLayoutV3 postprocessor supplies thresholding, polygons,
  and learned reading order.
- Prompt routing preserves the PaddleOCR-VL 1.6 contract: table,
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

`scripts/run_offline_e2e.py` intentionally keeps a smaller diagnostic
page-preparation path and its reading-order text is not an OmniDocBench
prediction. `scripts/run_omnidocbench.py` is the production entrypoint. It uses
the owned layout frontend and page engine for the complete page-to-Markdown
path. A runtime assertion fails the run if any `paddlex` module was imported;
the summary records the same audit.

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
- Run wall: first page start through the ordered page-artifact writer's final
  drain. This is the E2E throughput denominator, so every prediction is durable
  before the run completes; overlapping page latencies are never summed.
- Per-page latency: that page's image load through submission to the artifact
  writer. The single writer preserves completion order and applies bounded
  backpressure at eight pending pages if storage cannot keep up.
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
The run summary's `page_artifact_writer` block reports accumulated writer work,
queue residence, scheduler backpressure, final-drain wall, and maximum pending
depth. The timeline records submission on the scheduler thread and artifact
work on the `page-artifact-writer` thread separately.

### Text-prefill optimization lab

`scripts/text_lab_corpus.py` turns a recognition trace into an exact text-shape
workload: it reconstructs each crop grid, expands the real prompt with the
installed tokenizer, checks the recorded input/image-token counts and bucket
route, and preserves production request and prefill-group order. The corpus
stores token IDs rather than image tensors.

`scripts/text_lab.py` then runs either a corpus replay or a per-bucket graph
profile. It performs the real token embedding lookup and exact MRoPE layout,
but replaces vision-projector values with a deterministic fixed-seed tensor;
the text transformer has identical shapes, attention positions, weight format,
compiled graph and KV writes. Image contents are deliberately outside this
lab's contract because they do not change text-transformer execution cost.

The headline `text_prefill` measurement is the same production boundary:
decoder transformer plus in-place prefill KV writes. Token embedding, image
scatter, scratch-cache allocation, bucket padding, and optional LM head/argmax
remain separately timed. Replay reports both effective real-token throughput
and physical padded-token throughput, plus per-bucket totals. By default the
lab refuses to compile a missing graph; pass `--allow-compile` only for an
intentional new static bucket.

The E2E path exposes the validated lab mechanism with
`--text-packing production_group`. Its run summary adds call reduction, pack
size and bucket histograms, physical versus real text tokens, and copied KV
bytes. `device_stage_s.text_kv_redistribute` keeps the required cache-split
cost separate from `device_stage_s.text_prefill`.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_lab_corpus.py

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_lab.py \
  --mode replay \
  --name replay_256p_baseline

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_lab.py \
  --mode profile \
  --profile-buckets 64,128,256,640,1312 \
  --warmup 2 \
  --repeats 10 \
  --name profile_current_graphs
```

The experimental `--mode packed` path composes multiple independent prompts
into one B=1 static sequence. It resets MRoPE for every segment and builds a
block-diagonal causal mask inside the compiled graph, so prompts cannot attend
across crop boundaries. The current experiment intentionally stops at a packed
scratch KV cache: its report excludes redistribution of each segment's KV
prefix into independent decode-ready caches. Consequently, its transformer
speedup is a validated optimization target rather than an E2E production
claim. One explicit `--allow-compile` is required when creating a new packed
shape; subsequent runs reuse that cache.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_lab.py \
  --mode packed \
  --pack-length 1024 \
  --max-pack-members 32 \
  --pack-scope production_group \
  --pack-scope global \
  --name packed_text_1024
```

The lab-only `--mode cache_lease` prices the next cache-boundary option without
changing the serving engine. The current path writes a packed scratch cache,
redistributes every segment into a private full-length cache, and later copies
that private prefix into the decode arena. The lease path keeps the compact
packed cache alive and copies each segment slice directly into its arena row.
It alternates both paths pack-by-pack through the same compiled graph, verifies
exact valid-prefix KV equality and multi-step token parity, exercises pooled
buffer reuse, and projects ready-reservoir HBM from the actual pack ownership
sequence. Buffer recycling relies on the existing single-stream ordering:
release occurs only after the last admission read has been enqueued.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_lab.py \
  --mode cache_lease \
  --pack-length 1024 \
  --max-pack-members 32 \
  --arena-batch-size 32 \
  --cache-length 4096 \
  --ready-buffer-capacity 128 \
  --pack-scope production_group \
  --name cache_lease_256p
```

### Text-decode optimization lab

`scripts/text_decode_lab_corpus.py` turns a production recognition trace into
an exact decode-lifetime workload. It preserves request-source order, prompt
lengths, first and generated token IDs, EOS/length stops, the production cache
guard, the exact KV capacity required to replay every recorded graph write, and
the extra active iteration caused by the scheduler's queue-depth-one completion
look-ahead. The builder verifies its request and token totals against the run
summary.

`scripts/text_decode_lab.py` deliberately separates five questions:

- `simulate` reconstructs stable-slot occupancy without loading the model or
  using an NPU. It must reproduce the reference run's graph-call, active,
  effective, idle, and look-ahead accounting before the corpus is trusted.
- `profile` measures one deliberate `(batch_size, cache_length)` production
  graph shape. Its outer device span includes token selection and the
  post-graph arena state updates; the report also keeps the model-plus-argmax
  inner span separate.
- `torch_profile` captures a bounded set of already-warmed compiled decode
  steps with the native `torch_npu` profiler. It records CPU/NPU activities,
  operator shapes, framework stacks, a Chrome timeline, CANN operator/kernel
  tables, and one selected Level1 AI Core metric. Profiler wall time is
  intentionally not treated as throughput because collection perturbs it.
- `replay` runs the real `TextDecodeRuntime`, `DecodeArena`, sampled-token D2H
  ring, retirement, refill, admission-copy, and hot-swap scheduler. Recorded
  request lengths decide completion, so every tested shape sees the same
  workload even when synthetic prompt KV values change model token choices.
  Prompt KV is a shared deterministic zero prefix and every request is ready at
  replay start; vision/text prefill and frontend arrival timing are outside this
  mode's contract.
- `correctness` teacher-forces recorded token paths through the raw-eager and
  selected decode backends for multiple steps, comparing logits, top-1 tokens,
  and newly written KV positions. This is a graph correctness gate; a real-crop
  end-to-end token-parity run remains mandatory before a candidate optimization
  enters production.

The Experiment 09 E2E runners always use the validated `combined_apply` decode
path: MRoPE-factor hoisting, packed QKV, native RMSNorm and rotary operators,
and fused residual-add/RMSNorm. This is deliberately not a production CLI
switch. The text-decode lab retains its explicit `--decode-optimization`
presets for controlled comparisons: `baseline` retains the previous production
implementation, while the other presets isolate
MRoPE-factor hoisting, packed QKV and MLP projections, native RMSNorm and
rotary operators, fused residual-add/RMSNorm, and native SwiGLU. Packed
projection modules are materialized before the decode weight-format pass so
their weights receive the same FRACTAL_NZ treatment as production linears.
Each preset has a distinct TorchAir cache key. Correctness mode always compares
the selected compiled preset against the baseline eager stage.

The faithful OmniDocBench runner defaults to the measured B32/KV4096 decode
shape. Its 2,808-token generation cap leaves room for the retained corpus's
1,289-token maximum prompt in that static cache.

TorchAir graph creation is opt-in. A missing shape fails unless
`--allow-compile` is explicit, and each invocation profiles only the requested
shape rather than expanding a hidden matrix. The default decode cache root is
the production Experiment 09 root, so an already-compiled production shape is
reused without copying it or maintaining a second cache.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab_corpus.py

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode simulate \
  --batch-size 32 \
  --cache-length 4096 \
  --name simulate_b32_k4096

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode profile \
  --batch-size 32 \
  --cache-length 4096 \
  --profile-position 1024 \
  --warmup 3 \
  --repeats 20 \
  --name profile_b32_k4096

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode torch_profile \
  --batch-size 4 \
  --cache-length 1024 \
  --profile-position 1000 \
  --warmup 3 \
  --repeats 3 \
  --profile-metric pipe \
  --name torch_profile_b4_k1024

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode replay \
  --batch-size 32 \
  --cache-length 4096 \
  --name replay_b32_k4096

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode correctness \
  --batch-size 32 \
  --cache-length 4096 \
  --correctness-items 1 \
  --correctness-steps 8 \
  --name correctness_b32_k4096

09_persistent_page_engine/scripts/run_text_decode_optimization_lane.sh
```

### Packed-text E2E result

Commit `5ffd072` was run on the first 32 OmniDocBench pages with min-pixels/4,
B=16 decode, cache length 8192, the profile-guided vision router with lookahead
32, and compiled PromptFA. The two lanes differed only in text packing; setup
is excluded from E2E.

| Lane | E2E (s) | Pages/s | Text calls | Text transformer (s) | KV split (s) | Physical / real text tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text packing off | 27.58 | 1.160 | 510 | 4.45 | 0 | 70,704 / 62,693 |
| 128/256/512/1024 production-group packs | 24.51 | 1.306 | 104 | 1.09 | 0.87 | 73,088 / 62,693 |

Packing reduced E2E time by 11.2% and transformer calls by 79.6%. Transformer
device time fell by 75.4%; including KV redistribution, the combined packed
transformer-plus-split boundary was 1.96 seconds, 55.9% below the unpacked
transformer alone. The packed graph processed 66.8k physical tokens/s and
57.3k useful tokens/s. Outputs had exact token parity for 505/510 crops; the
five differing crops changed the aggregate generated-token count by two.

The owned OmniDocBench runner also writes `timeline_trace.json` and a
self-contained `timeline.html` (template: `utils/timeline_viewer.html`). Every
event declares the resource it describes — host thread, device stream, queue,
or decode slot — and the viewer lays them out accordingly: host threads become
containment-nested flame charts (generator-driving `next()` frames are scopes,
not waits, so nested prefill work is never double-counted as idle), NPU prefill
and decode get their own reconstructed-clock lanes, decode slots draw as
per-slot occupancy rows, and queue waits aggregate into depth charts instead of
overpainting one row. Group headers report busy/occupancy percentages for the
current viewport; clicking a span follows one crop or page end-to-end and opens
a stage-by-stage breakdown. The viewer template renders standalone too: opening
it without an embedded trace offers drag-and-drop for any
`timeline_trace.json`.

Timeline recording adds host timestamping but no accelerator synchronization.
Prefill device spans reuse the existing per-request `DeviceTimeline.resolve()`
boundary, and decode device spans reuse the scheduler's existing final resolve.
Recognition-input H2D timing is device-event copy time on the dedicated prefill
transfer stream, not a synchronized host-wall envelope. Pass `--no-timeline`
only when measuring the small host-side timestamping and in-memory
event-recording overhead itself.

Recognition prefill stages one future group's H2D before yielding any crop from
the current group to the pull-driven decode scheduler. A group is one crop when
packing is disabled. TorchAir occupies the main host thread for much of G's
device work, so one dedicated host worker submits only G+1's H2D concurrently
while the main thread invokes G's compute chain. This avoids both late transfer
submission and the NPU draining two sequentially submitted H2D groups before
compute begins. The future copy is also already submitted before yielding can
suspend the ready source until the decode reservoir next needs a refill. The
staged group's compute chain waits on its own recorded H2D completion event and
starts without submitting a new copy. CPU preparation pins the five copied input
tensors when the runtime supports pinned allocation; failure to pin falls back
to the same tensors and preserves correctness. This dedicated transfer-stream
and compute-stream event dependency mirrors the decode scheduler's established
stream/event pattern without pre-enqueueing multiple future compute groups.
First-token D2H uses the same transfer stream: it waits on the group's argmax
events and copies one pinned K-token vector, so the copy cannot be queued behind
G+1 on the compute stream. The lookahead changes production order only: prefill
and decode kernels remain serialized on the compute stream.

### Packed vision prefill

`--vision-packing greedy` enables impatient arrival-order packing; the default
is `off`. The group former blocks only for its first crop, then consumes every
crop already available from CPU preparation while the sum of real vision rows
fits `--vision-pack-target` (default 1920). It never waits to improve a nonempty
group. Crops above the target stay on the existing faithful single-crop path,
including the unchanged eager-overflow route above 2048 rows.

Each crop is embedded independently, then the rows are concatenated for one
existing shape-routed vision-transformer call. The attention mask prevents
different crop segments from attending to each other and separately isolates
dummy bucket rows. The result is split at the recorded real segment lengths;
the downstream projector and text path then run once per crop as before. The
packed call reuses the existing shape-keyed TorchAir graphs and introduces no
new vision bucket shapes.

The run summary records group count, group-size histogram, crops per group,
real/physical packed tokens, and fill fraction. The timeline shows one
`Packed vision transformer` device span with every member crop as a flow ID,
while downstream spans retain their individual crop IDs.

The owned runner prepares pages sequentially on one background producer. Each
page enters a one-page bounded queue as soon as layout detection, cropping, and
prompt preparation finish. The recognizer can therefore start consuming that
page while the producer prepares the next one; layout work itself is never
parallelized. Superseded all-pages
reference results remain archived under
`tmp/09_persistent_page_engine/prefill_pipeline_streaming_6b1642f/`,
`tmp/09_persistent_page_engine/prefill_pipeline_baseline_6b1642f/`, and the
`sync_scope_*_36f77fa/` run directories for future parity comparisons. The
timeline shows producer work on the `owned-page-producer` thread and queue
residence on the `page-queue` lane.

On NPU, that producer owns one dedicated layout stream and fences it before a
prepared page enters the handoff queue. Recognition input copies use the
dedicated prefill transfer stream and an event dependency on the recognizer's
compute stream, while each prefill timeline resolves by waiting on its last
recorded end event. These scoped waits preserve each stage's data dependencies
without making recognition wait for independent layout kernels; the
continuous-decode scheduler retains its single device-wide synchronization as
the final run-boundary fence.

## Layout frontend lab

`scripts/layout_owned_lab.py` runs the exact owned page path without the
recognizer. The measured boundary includes image loading, layout preprocessing,
PP-DocLayoutV3 inference and postprocessing, cropping, prompt routing, and final
`RecognitionRequest` materialization. It never loads or executes the local OCR
model. The owned frontend requires `kornia-rs==0.1.14` for direct RGB PNG
decoding; JPEG decoding uses the installed TorchVision image backend.

The output `requests.jsonl` records request order, page and block identity,
prompt, pixel profile, crop shape, and an exact hash of the crop pixels. Use
`--reference-requests` to fail unless a candidate run produces the same file:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/layout_owned_lab.py \
  --limit 32 \
  --output-dir tmp/09_persistent_page_engine/layout_owned_32p
```

The lab uses the same output-exact NPU-graph/mask route as production. Its
request manifest includes order, page/block identity, prompt, pixel profile,
crop shape, and exact crop-pixel hash. The first 32 OmniDocBench pages produce
the same 510-request hash as the retired PaddleX oracle.

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

The measured packed min-pixels/4 operating point is explicit rather than a
default change:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/run_omnidocbench.py \
  --limit 256 \
  --batch-size 16 \
  --cache-length 8192 \
  --max-new-tokens 4096 \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-padding auto \
  --vision-packing greedy \
  --vision-pack-target 1920 \
  --preprocessor-min-pixels 28224 \
  --output-dir tmp/09_persistent_page_engine/packed_minpixels_div4_256p
```

The normal page runners expose the vision comparison directly through
`--vision-backend raw_eager|torchair` and
`--vision-attention manual|prompt_flash_attention`. The default remains the
validated TorchAir/manual combination. TorchAir builds or loads one B=1 static
graph for every configured bucket during recognizer setup, and keeps manual and
PromptFA graphs in distinct cache directories. The first uncached run is
therefore compilation-heavy; per-bucket cache paths and first-call times are
recorded in `configuration.vision_prefill` and
`setup_timing_s.vision_runtime_setup`. Subsequent runs reuse the GE caches under
`.runtime_cache/09_persistent_page_engine_vision_torchair/`.

PaddleOCR-VL's native vision head dimension is 72, while compiled PromptFA
requires a dimension divisible by 16. The shared eager/compiled PromptFA path
therefore zero-pads Q/K/V to 80 only for the attention call, retains the scale
derived from the real dimension 72, and slices the result back to 72 before the
output projection. Bucket padding uses PromptFA sparse mode 1 so the supplied
real/dummy-row mask is honored.

The 64-page comparison retained under
`tmp/09_persistent_page_engine/vision_matrix_uniform64_analysis_22a2f21/`
found compiled PromptFA fastest overall: 29.52 seconds of vision-tower device
time for 1,183,256 real tokens (40.1k useful tokens/s), with 3.46% padding.
Compiling remained non-regressive through the largest 5,216-token bucket, while
changing attention or execution by size improved the estimated total by less
than one percent. The simplest measured policy is therefore compiled PromptFA
for every retained bucket.

### Packed-vision staged validation

Commit `44c20b6` was validated on the first 32 or 256 OmniDocBench pages with
B=16, cache length 8192, compiled PromptFA, and a 1920-row pack target. E2E
excludes setup. Score deltas are packed minus the matching unpacked control;
lower edit distance is better. Peak HBM was sampled on cache-warm 32-page
replays because the ready-reservoir and two-group in-flight bound are unchanged
by corpus length.

| Lane | Pages | E2E (s) | Pages/s | Vision tower (s) | Groups / crops per group | Token-diff crops | Score delta (page avg) | Peak HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Stage 0: packing off, default pixels | 32 | 32.87 | 0.974 | 10.67 | 510 / 1.000 | 0 / 510 | exact baseline | not sampled |
| Stage A: packed, default pixels | 32 | 30.79 | 1.039 | 8.04 | 266 / 1.917 | 6 / 510 (1.18%) | text 0; formula -0.00006; table 0; order 0 | 23,828 MB |
| Stage B control: packing off, min-pixels/4 | 32 | 31.13 | 1.028 | 8.49 | 510 / 1.000 | - | unpacked reference | not sampled |
| Stage B: packed, min-pixels/4 | 32 | 28.06 | 1.140 | 4.56 | 141 / 3.617 | 6 / 510 (1.18%) | text +0.00300; formula -0.00131; table 0; order 0 | 25,097 MB |
| Stage B control: packing off, min-pixels/4 | 256 | 229.36 | 1.116 | 73.43 | 3,524 / 1.000 | - | unpacked reference | not sampled |
| Stage B: packed, min-pixels/4 | 256 | 210.88 | 1.214 | 50.38 | 1,079 / 3.266 | 28 / 3,524 (0.795%) | text -0.00270; formula -0.00607; table 0; order -0.00268 | 25,097 MB (32-page replay) |

Packing therefore raised 256-page throughput by 8.8% and reduced serialized
vision-tower device work by 31.4%, with no score regression. It did not reach
the projected 1.30 pages/s acceptance target. The greedy stream formed only
3.266 crops/group, 66.7% of the offline 4.9-crop benchmark and below the
specified 80% policy-discussion threshold, although the groups it did form had
96.65% real-token fill. Bounded-window best-fit is the known next policy to
discuss; it is not implemented here.

The first min-pixels/4 control introduced text buckets 32, 64, 96, and 128.
Its text-runtime setup took 133.74 seconds and total setup took 176.91 seconds;
the next packed run reused those graphs (13.65 seconds text setup, 65.33 seconds
total). The 188 oversized crops in the 256-page packed run remained unchanged
single-crop eager-overflow groups. No OOM occurred; the measured packed peak
left about 40 GB of HBM headroom.

Each recognition result records its real token count, selected physical bucket,
padding, and execution route. Aggregate JSON reports bucket counts and the
overall useful-token fraction so padding cost is visible rather than folded
into a single throughput number. `recognition_trace.jsonl` also records the
vision-stage device time for every real crop, so attention/backend comparisons
can be grouped by exact token length or bucket after an otherwise normal E2E
run.

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

- `paddleocr_vl/serving/engine.py`: one persistent model instance, bounded
  background CPU crop/prompt/MRoPE preparation, sequential multimodal NPU
  prefill, compact prefilled state, and result materialization.
- `paddleocr_vl/serving/continuous_decode.py`: bounded ready reservoir and persistent
  decode-slot scheduler.
- `paddleocr_vl/serving/types.py`: crop requests, recognition results, and
  decode-schedule contracts.
- `paddleocr_vl/serving/runtime_defaults.py`: measured serving-runtime profiles.

`pipeline/` owns full-page concerns and depends on `paddleocr_vl/`, never the
reverse.

- `pipeline/layout_frontend.py`: owned PP-DocLayoutV3 model loading, sequential
  page inference, and exact page-to-request materialization.
- `pipeline/layout_model_runtime.py`: captured NPU model forward and selected
  mask postprocessing.
- `pipeline/layout_postprocess.py`: fixed v1.6 detector geometry, filtering,
  crop, and merge policy.
- `pipeline/layout_output.py`: recognition-result assembly, OTSL tables,
  compact JSON, images, and Markdown.
- `pipeline/page_engine.py`: bounded sequential page producer, one continuous
  crop schedule, per-page collectors, and immediate completion.
- `pipeline/layout.py`: older standalone PP-DocLayoutV3 diagnostic path.
- `pipeline/page_pipeline.py`: lazy page/layout/crop routing and page completion.
- `pipeline/layout_mask_guard.py`: PP-DocLayout empty-mask fallback and telemetry.
- `pipeline/omnidocbench_defaults.py`: validated full-benchmark execution profile.
- `pipeline/types.py`: boxes, layout regions, page results, and run serialization.

`scripts/` contains serving and pipeline composition roots. It includes the
diagnostic page runner, owned OmniDocBench runner, NPU smoke wrapper, labs, and
focused probes. `utils/` contains only shared timing and metric helpers.

The production runtime packages do not import `scripts/` entrypoints or probes.
Those scripts consume the same preprocessing and model-stage modules as the
E2E engine, so diagnostic code cannot silently become a runtime dependency or
invalidate a compiler cache merely because a probe changed.
The recognizer also constructs and warms all three compiled boundaries under
`torch.inference_mode()`, matching real request execution and keeping TorchAir's
dispatch-key guards stable across warmup and serving.

The retired cross-page PaddleX bridge was validated on the same uniformly
sampled 64-page OmniDocBench v1.6 set as the preceding adapter run. All 64
compact JSON results and Markdown files matched exactly. One schedule handled all 1,332
crops; useful decode-slot occupancy rose from 41.36% to 96.01%, decode wall fell
from 51.67s to 23.23s, and E2E time fell from 162.62s to 142.96s (2.234s/page).
The compact evidence is retained in
[`run_summary.json`](../tmp/09_persistent_page_engine/omnidocbench_v16_uniform64_b16_cross_page_0c4eebf_20260720/run_summary.json).

Earlier measured 910B validations remain with Experiment 08. Its
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
