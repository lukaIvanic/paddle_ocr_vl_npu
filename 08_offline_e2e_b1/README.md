# Experiment 08: offline real-layout E2E with continuous decode

This experiment is the first small offline inference system in the repository,
rather than another isolated kernel benchmark. It keeps both models resident in
one Python process and executes this path:

```text
stream of full PIL pages
  -> lazy real PP-DocLayoutV3 inference on NPU
  -> reading-ordered layout regions and page collectors
  -> crop and prompt routing into one run-scoped request source
  -> one eager PaddleOCR-VL prefill at a time
       CPU image/prompt preprocessing
       eager native-resolution vision prefill
       eager text prefill into a static KV cache
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

Vision and text prefill deliberately remain sequential B=1, but they are now
produced lazily for one run-scoped decode scheduler instead of draining decode
at every page boundary. Decode owns one persistent fixed-shape arena. Slot
indices stay stable; a finished request is replaced in place without moving
other active requests or rebuilding the batch. Admission copies only the valid
prompt KV prefix, while stale cache tails remain safely hidden by each row's
cache position. The ready reservoir is internal and bounded, so the pipeline
does not materialize every page's NPU KV caches before decode.

A page is an input/output aggregation boundary, not a scheduling boundary.
Each request carries its page identity through the engine; per-page collectors
restore reading order and can emit independently. The returned `pages` list
remains in input order even when completion callbacks arrive out of order.

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
- The recognizer is the corrected local PyTorch model from Experiment 05. Vision
  and text prefill are eager and unpadded; only the flat fixed-batch static
  decode module is compiled.
- Sampled-token D2H uses a second NPU stream, pinned two-row host ring, and
  queue-depth-one control. A request can execute one look-ahead graph call;
  slot epochs discard that old result after the slot is reused.

The current page preparation is intentionally smaller than the complete
PaddleX pipeline. It does not yet port PaddleX overlap filtering, bbox unclip and
merge policy, adjacent-block merging, formula margin crop, table figure-token
substitution, or final structured Markdown assembly. The output text is a
reading-order diagnostic artifact, not an OmniDocBench-comparable prediction.
Those missing steps sit between the explicit layout, crop-routing, and page
postprocess boundaries, so adding them does not require replacing the engine.

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

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup

/usr/local/python3.12.13/bin/python3 \
  08_offline_e2e_b1/run_offline_e2e.py \
  --image "/workspace/datasets/OmniDocBench/images/PPT_The Right Moves_page_024.png" \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --recognizer-model /workspace/models/PaddleOCR-VL-1.6 \
  --device npu:0 \
  --decode-backend torchair \
  --batch-size 4 \
  --cache-length 2048 \
  --max-new-tokens 768
```

The default artifact directory is timestamped under
`tmp/08_offline_e2e_b1/`. It contains `run.json`, per-page reading-order text,
and an annotated layout image. Pass `--save-crops` only when the actual crop
images are useful. `--max-regions N` is a debug-only partial-page mode and is
recorded as such in the JSON.

`--batch-size` accepts 1, 2, 4, 8, and other powers of two. Additional
`--image` arguments enter the same cross-page scheduling domain by default.
Each page is printed and made available to callbacks as soon as its own regions
finish; the engine does not wait for the whole image list before emitting it.

Measured 910B validations are recorded in
[`NPU_FULL_PAGE_RESULT.md`](NPU_FULL_PAGE_RESULT.md) for the original B=1 path
and [`NPU_BATCHED_DECODE_RESULT.md`](NPU_BATCHED_DECODE_RESULT.md) for padded
fixed B=2 and B=4 decode. Those documents predate the continuous scheduler and
remain historical comparison points.

The persistent-slot implementation and its exact parity/performance comparison
are recorded in
[`NPU_CONTINUOUS_DECODE_RESULT.md`](NPU_CONTINUOUS_DECODE_RESULT.md).
