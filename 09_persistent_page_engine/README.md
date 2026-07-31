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

Both page runners accept `--layout-device npu|cpu`. The default remains `npu`.
On NPU, the owned runtime replaces two unsupported Transformers 5.5.4 indexing
forms. Fixed `spatial_shapes` metadata is cached on NPU during the eager graph
warm-up instead of being filled by scalar indexed writes, and top-k memory rows
are selected with the equivalent dimension-1 `gather` instead of tensor-valued
advanced indexing. All detector computation remains on NPU. The summary records
this as `layout_frontend.npu_indexput_compat`.

The production runner also accepts
`--layout-graph-capture/--no-layout-graph-capture`. Disabling capture leaves
the complete layout model on NPU and changes only its execution mode; this is
the intended 310P compatibility route when eager operators work but ACLGraph
capture rejects them.

The CPU option remains the fallback while that NPU compatibility path is being
validated on 310P. It moves only PP-DocLayoutV3 inference and preprocessing off
the NPU; the PaddleOCR-VL recognizer, vision/text prefills, KV cache, and decode
remain on logical `npu:0`. CPU layout disables NPU graph capture and NPU event
timing automatically. The summary records `layout_device`,
`layout_frontend.device`, `layout_frontend.graph_capture`, and
`layout_frontend.npu_indexput_compat`.

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

`scripts/text_decode_lab.py` deliberately separates six questions:

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
- `boundary` runs exactly one real full-decoder/arena step at a selected cache
  position and emits flushed markers before the call, after enqueue, and around
  device synchronization. It exists for externally timed hardware-boundary
  probes where a kernel can hang without raising a Python or CANN error. Its
  inputs are synthetic and it deliberately does not require a replay corpus.
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

`combined_apply_mha_repeat` is a lab-only discriminator for the 310P masked-GQA
boundary issue. It leaves the persistent cache in production's two-KV-head
form, expands one layer's K/V to 16 heads immediately before IncreFA, and calls
the operator in MHA form (`num_key_value_heads=0`). It is not a production
fallback or an accepted optimization: the boundary lab first establishes
whether it avoids the fault, and later measurements must price its transient
memory and repeated-copy cost.

`combined_apply_mha_cache` prices the opposite point in that trade-off. Its
decode arena permanently stores all 16 repeated KV heads, while text prefill
still produces the ordinary two-head cache. Admission broadcasts the prefill
cache once into the expanded arena; each decode layer then repeats only the
new one-token K/V write and calls IncreFA as MHA. At B16/KV4096/fp16 across 18
layers, the decode arena grows from 1.125 GiB to 9.0 GiB. This path is also
lab-only.

The 910B comparison at commit `77ebec8` is recorded under
`tmp/09_persistent_page_engine/mha_cache_910b_77ebec8/`. In the same compiled
100-step B16, position-1279 profile, production GQA measured 2.517 ms/step and
6,356 physical tok/s; per-layer repeated MHA measured 16.276 ms and 983 tok/s;
expanded-cache MHA measured 8.189 ms and 1,954 tok/s. Thus permanent expansion
recovers about half the MHA loss, but remains 3.25x slower than GQA at the
device-step level. The real 16-request OCR generation remained token-, text-,
and EOS-exact against GQA and measured 1,928 effective tok/s, versus 916 for
per-layer repetition and 4,104 for GQA. Measured allocation before the real
run rose from 9.19 GB for GQA to 17.64 GB for expanded-cache MHA, matching the
expected 7.875 GiB arena increase.

`scripts/text_decode_real_generation.py` is the real-generation gate after the
synthetic `boundary` and `correctness` modes. It obtains block 3 from a fixed
OmniDocBench page through the owned layout frontend, verifies the expected
1022-by-772 table crop and 1,021-token real prefill, duplicates that crop into
16 slots, and performs real vision prefill, text prefill, KV admission, and
autoregressive decoding. The selected crop generates 374 tokens before EOS,
so every slot naturally executes cache position 1279/effective length 1280.
The report retains every generated token and requires all requests to cross
the target. `--reference` performs exact per-request token, text, and stop
reason comparison.

The 910B controls at commit `e257add` are under
`tmp/09_persistent_page_engine/real_decode_generation_910b_e257add/`. Both GQA
and repeated-KV MHA completed 16/16 requests and were bit-exact. The sustained
cost is substantial: production GQA measured 4,104 effective decode tok/s and
0.940 s of model-plus-argmax device time, while MHA measured 916 tok/s and
6.057 s. The MHA path is therefore a correctness/termination candidate for
310P, not a performance optimization.

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_real_generation.py \
  --decode-cache-dir .runtime_cache/09_persistent_page_engine_torchair \
  --decode-optimization combined_apply \
  --output tmp/09_persistent_page_engine/real_decode_generation/gqa.json

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_real_generation.py \
  --decode-cache-dir .runtime_cache/09_persistent_page_engine_torchair \
  --decode-optimization combined_apply_mha_repeat \
  --reference tmp/09_persistent_page_engine/real_decode_generation/gqa.json \
  --output tmp/09_persistent_page_engine/real_decode_generation/mha.json
```

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
  --mode boundary \
  --batch-size 16 \
  --active-slots 16 \
  --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply \
  --name boundary_b16_k4096_pos1279

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

### Vision MatMul format/alignment lab

`scripts/vision_matmul_lab.py` measures the exact production
`VisionPrefillStage` with synthetic shape inputs. The boundary keeps all 27
layers, both LayerNorm/residual paths, Q/K/V/output projections, RoPE, real
PromptFA (including the production 72-to-80 head-dimension padding),
FC1/GELU/FC2, and post-LayerNorm. Physical token throughput therefore measures
the same vision-transformer graph used by a real crop rather than an
attention-free MatMul surrogate.

The bounded matrix compares B1xS512, B4xS512, and B1xS2048 at the native
4304-wide MLP and the mathematically equivalent zero-extended 4352-wide MLP,
with native and requested FRACTAL_NZ Linear weights. Physical throughput uses
all `batch_size * sequence_length` tokens in each replay. For FRACTAL_NZ, the
lab enables torch-npu's
`allow_internal_format` runtime gate before the first NPU allocation, casts all
162 Linear weights after model loading, and verifies that every observed weight
has format code 29. It never silently times a native-format fallback. Timed
samples contain multiple complete production-stage replays inside one NPU-event
interval. The optional profiler records MatMulV2 versus MatMulV3 dispatch,
TransData cost, input formats, and cube utilization; its perturbed wall time is
not the throughput result.

When profiling is enabled, `parsed_profile.matmul_only` also records the
strict MatMul-only aggregate. It verifies exactly 162 Linear MatMul kernels
per profiled full-stack replay, divides the known Q/K/V/output and FC1/FC2
FLOP count by only their summed kernel duration, and reports both
`matmul_kernel_duration_per_full_stack_call_ms` and
`matmul_only_linear_tflop_per_s`. This is intentionally distinct from
`measurements.linear_tflop_per_s_device_median`, whose denominator is the
complete 27-layer vision-transformer replay.

#### Detailed multi-metric profiling

`scripts/run_vision_matmul_profile_suite.py` is the repeatable deep-profile
entry point for the Phase-14 optimized lane: B1xS2048, the mathematically
zero-extended 4352-wide MLP, all 162 Linear weights in FRACTAL_NZ, runtime
D72-to-D80 PromptFA padding, separate manual RoPE, and the warm compiled
27-layer production stage. It captures each AI Core PMU family in a separate
process; the unprofiled NPU-event measurement remains the throughput result.

```sh
source npu-setup

PYTHON=/usr/local/python3.12.13/bin/python3
MODEL=/workspace/models/PaddleOCR-VL-1.6

"$PYTHON" \
  09_persistent_page_engine/scripts/run_vision_matmul_profile_suite.py \
  --name 910b_b1s2048_i4352_nz \
  --model "$MODEL" \
  --metrics pipe memory memory_access l2
```

`PYTHON` and `MODEL` are the only environment-specific paths in that command.
On another server, point them at that server's prepared Python environment and
PaddleOCR-VL checkpoint. The runner, source paths, cache layout, raw/evidence
split, and analyzer command remain unchanged.

The runtime is the authority for available PMU families:
`torch_npu.profiler.supported_ai_core_metrics()` is checked before model load
and again before each capture. The cross-product default is `pipe memory`.
Huawei's product matrix lists `arithmetic`, `pipe`, `memory`, `memory_l0`,
`memory_ub`, and `resource_conflict` for Atlas inference products such as
310P; `l2` and `memory_access` are A2/A3 additions. Unsupported requested
lanes fail before a graph is compiled or profiled rather than producing an
empty or zero-valued report.

Raw profiler output stays under
`.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/`. Each lane
writes its exact command, contract, log, and ordinary lab summary under
`tmp/09_persistent_page_engine/vision_matmul_profile_suite/<name>/`.
Large normalized execution tables and SQLite databases remain beside the raw
captures under `.runtime_cache/`; only compact JSON, CSV, and Markdown reports
should be copied into `tmp/` for retention.

The suite prints flushed, timestamped lane transitions and a heartbeat every
15 seconds by default. Each heartbeat includes elapsed time and the current
lane-log size; `--progress-interval-s` changes the interval. For a remotely
followable run, redirect the suite's output to a separate driver log and use
`tail -f` on that file. Child output is written incrementally to
`<suite>/<metric>/run.log`, and `suite_summary.json` is checkpointed after
every successful metric lane.

The combined analysis contains:

- every `kernel_details.csv` execution in normalized CSV and SQLite form;
- the raw CSV and profiler-database schema inventory, including unknown PMU
  columns rather than dropping them;
- replay spans, kernel overlap/gaps, task and stream IDs, Block/Mix Block
  counts, shapes, dtypes, and formats;
- a fail-closed 27-layer mapping. `q_proj`, `k_proj`, `v_proj`, `out_proj`,
  `fc1`, and `fc2` are labeled only when every replay has exactly 162 MatMuls
  and the full six-shape motif matches the saved model contract;
- per-kernel, per-role, and per-layer physical MatMul FLOP/s plus every
  available Pipe, Memory, or L2 counter.

The analyzer can also be rerun later without another NPU execution:

```sh
"$PYTHON" \
  09_persistent_page_engine/scripts/analyze_vision_matmul_profile.py \
  --contract <lane-result>/profile_contract.json \
  --lane pipe=<raw-pipe-profile> \
  --lane memory=<raw-memory-profile> \
  --lane l2=<raw-l2-profile> \
  --output-dir <new-analysis-dir>
```

Interpretation is deliberately conservative. `Block Num` is configured task
parallelism, not proof of balanced useful work on that many physical cores.
On the observed CANN 9 application export, `cube_utilization(%)` is
`aicore_time / task duration`. It is an exported time ratio that can exceed
100%, not physical-core occupancy, achieved MAC utilization, or peak-FLOP
utilization. MAC, MTE1, MTE2, FixPipe, and Vector ratios overlap and must not
be added. Whole-graph captures establish which kernels and layers matter;
targeted `msprof op` replays are the separate second tier for a single
selected kernel's mechanics. `Occupancy` supplies physical-core balance and
`MemoryDetail` supplies deeper active-pipe memory instrumentation, but Huawei
documents both only for Atlas A2/A3 products.

`scripts/run_vision_msprof_op.py` implements that second tier without MSTX,
private APIs, injected libraries, or a hard-coded CANN installation path.
The tested CANN 9 `msprof op` path emitted no data when MSTX targeted the
compiled TorchAir replay. A second bounded test confirmed that even a
one-Linear cached TorchAir graph exposes only setup kernels to `msprof op`,
not its inner MatMulV2. The permanent direct target is therefore deliberately
limited to the square Q/K/V/output-projection shape: its eager call was
validated against the compiled graph as the same MatMulV2, FP16
ND/FRACTAL_NZ/ND contract, dimensions, and Block Dim. FC1 and FC2 remain in
the full-graph profiler; their eager dispatch differs, so no convenient but
non-representative surrogate is retained.

```sh
source npu-setup

"$PYTHON" \
  09_persistent_page_engine/scripts/run_vision_msprof_op.py \
  --run-name vision_square_pipe \
  --metric PipeUtilization

"$PYTHON" \
  09_persistent_page_engine/scripts/run_vision_msprof_op.py \
  --run-name vision_square_memory \
  --metric Memory
```

Those two commands are the portable base for 910B and 310P. The same runner
also accepts `ArithmeticUtilization`, `MemoryL0`, `MemoryUB`, and
`ResourceConflictRatio`. On A2/A3 products such as 910B, add `Occupancy` or
`MemoryDetail` when physical-core balance or active-pipe memory detail is
needed. Huawei documents those two families as unsupported on Atlas inference
products such as 310P, so the work-server workflow does not request them.

These captures are never substitutes for full-stack timing. Before
interpretation, `scripts/analyze_vision_msprof_op.py` matches MatMulV2,
shape-derived FLOPs, formats, Block Dim, and dtype back to the normalized
full-graph reference:

```sh
"$PYTHON" \
  09_persistent_page_engine/scripts/analyze_vision_msprof_op.py \
  --capture-dir tmp/09_persistent_page_engine/vision_msprof_op/<capture> \
  --raw-dir .runtime_cache/09_persistent_page_engine_vision_msprof_op/<capture> \
  --reference-dir .runtime_cache/09_persistent_page_engine_vision_matmul_profiles/<analysis> \
  --output-dir tmp/09_persistent_page_engine/vision_msprof_op/<capture>/analysis
```

The runner discovers `msprof` on `PATH`; paths are repository-relative; the
analyzer is standard-library-only; and binary caches are neither required nor
transferred. The method moves to the work server by changing only `PYTHON`
and the full-graph suite's `MODEL` value. Metric support is selected by
product/runtime capability. Missing fields remain missing, and an unsupported
family is never reported as measured zero.

```sh
/usr/local/python3.12.13/bin/python3 \
  09_persistent_page_engine/scripts/run_vision_matmul_lab_matrix.py \
  --name 910b_compiled \
  --execution torchair \
  --allow-compile-if-missing \
  --profile
```

`--attention-head-padding weights` is the controlled alternative to the
production default `runtime`. It zero-extends every attention Q/K/V projection
once from 1152 to `16 * 80 = 1280` outputs, inserts the eight zero channels in
each head's two-half RoPE layout, supplies neutral 80-wide RoPE inputs, and
zero-extends `out_proj` to 1280 input columns. The compiled graph therefore has
no per-layer Q/K/V `PadV3` or PromptFA-output 80-to-72 slice.

On Ascend 910B2 at B1xS2048 with the 4352-wide MLP and native ND weights, this
changed the full 27-layer stage from 30.1408 ms / 67,947.7 physical tok/s to
25.7301 ms / 79,595.5 tok/s. It removed 189 kernels per replay and reduced
latency by 14.6%, despite the 11.1% larger attention projection MatMuls. The
raw full-stage comparison remained finite with mean absolute difference
0.00268. Evidence is under
`tmp/09_persistent_page_engine/vision_matmul_lab/head80_weight_*`.

The follow-up full-stage RoPE comparison keeps that D80 graph and applies the
same FP32 half-RoPE formula once to one contiguous QK tensor. On the same
B1xS2048 shape, the joint path reproduced at 24.31 ms / 84.2k physical tok/s,
versus a 25.55 ms / 80.1k warm control: 4.84% lower latency and 5.09% higher
throughput, with exact raw D80 output parity. The complete-graph profile halves
the RoPE slice/multiply/add/cast/negate families, removes the QKV split, and
leaves all 27 PromptFA and 162 Linear calls unchanged.

A 910B-only interleaved `_C_ascend::inplace_partial_rotary_mul` lane also
compiled through an explicit TorchAir converter, but regressed to 187.10 ms
per full replay and is rejected. The production stage is unchanged pending a
real-crop/E2E gate. The full table and profile comparison are under
`tmp/09_persistent_page_engine/vision_matmul_lab/rope_full27_comparison_e12cfe8/`.

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

### Project-owned eager detector

The layout lab can replace the Transformers model implementation with the
project-owned eager implementation under
`pipeline/owned_layout_model/`. The owned boundary includes the HGNetV2-L
backbone, hybrid encoder, deformable decoder, mask/order/class/box heads,
checkpoint configuration, and strict safetensors loading. It intentionally
does not include the image processor yet: preprocessing and detector
postprocessing still come from Transformers.

This route is eager-only. It does not use TorchAir, `torch.compile`, or NPU
graph capture:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/layout_owned_lab.py \
  --model-backend owned \
  --no-graph-capture \
  --limit 8 \
  --output-dir \
    tmp/09_persistent_page_engine/owned_layout_model/owned_eager_8p
```

The strict loader accounts for all 858 tensors and 33,288,957 stored tensor
elements in the official checkpoint. The only non-stored model keys are the
known tied decoder-head aliases and PyTorch batch counters. On the first eight
OmniDocBench pages, the owned NPU eager model produced 122
`RecognitionRequest` records byte-for-byte identical to the Transformers NPU
eager oracle, including crop pixels and order. The evidence is under
`tmp/09_persistent_page_engine/owned_layout_model/`.

`scripts/layout_model_parity.py` is the narrower raw-output probe. It loads the
two model implementations sequentially, runs the same processor tensor through
each, and records differences for logits, boxes, order logits, and masks. The
request-manifest comparison remains the acceptance gate because detector query
selection can amplify small internal numerical differences while preserving
the exact selected page regions.

The lab also enables NPU-event stage timing around the detector graph and the
device metadata/mask postprocess tails. The corresponding
`layout_model_device_s`, `layout_device_metadata_postprocess_s`, and
`layout_device_mask_postprocess_s` fields are accelerator execution times.
`layout_metadata_wait_and_d2h_s` and `layout_mask_wait_and_d2h_s` are host-wall
wait-plus-copy measurements at the existing `.cpu()` boundaries; they must not
be added to the device times as independent critical-path work. This
instrumentation is disabled in the production frontend.

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

Experiment 09 recognition always uses the logical device `npu:0` selected by
`npu-setup` and always calls
`torch.npu.set_compile_mode(jit_compile=False)`. Layout also uses `npu:0` by
default; pass `--layout-device cpu` only for the explicit compatibility path
described above.

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

Atlas 310P PromptFA rejects a non-null attention mask when the query/key
sequence length is not 128-aligned. Enable
`--vision-promptfa-align-128` on that hardware. The option rounds every
PromptFA physical sequence and configured vision bucket up to a 128-token
multiple before vision execution; real token counts and useful-token fractions
remain separately reported. It applies to eager, compiled, singleton, and
packed vision routes. The default is off, so existing 910B execution and cache
shapes are unchanged.

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
- `pipeline/owned_layout_model/`: independent eager PP-DocLayoutV3 model,
  HGNetV2-L backbone, configuration, and strict safetensors loader. It has no
  Transformers model dependency.
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
