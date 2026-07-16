# Experiment 08: offline real-layout E2E with continuous decode

This experiment is the first small offline inference system in the repository,
rather than another isolated kernel benchmark. It keeps both models resident in
one Python process and executes this path:

```text
full PIL page
  -> real PP-DocLayoutV3 inference on NPU
  -> reading-ordered layout regions
  -> crop and prompt routing
  -> one PaddleOCR-VL request at a time
       CPU image/prompt preprocessing
       eager native-resolution vision prefill
       eager text prefill into a static KV cache
       ready B=1 KV state
  -> all page requests enter a ready queue
  -> persistent power-of-two compiled decode arena
       fill free slots from ready B=1 KV states
       run one autoregressive iteration
       retire EOS/length-complete requests
       hot-swap the next ready KV prefix into each freed slot
       D2H tokens and detokenization
  -> reading-order text
```

Vision and text prefill deliberately remain sequential B=1 and finish before
decode starts. Decode owns one persistent fixed-shape arena. Slot indices stay
stable; a finished request is replaced in place without moving other active
requests or rebuilding the batch. Admission copies only the valid prompt KV
prefix, while stale cache tails remain safely hidden by each row's cache
position. Pages remain sequential, so this is decode-iteration continuous
batching rather than concurrent layout/prefill/decode scheduling.

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

`run.json` reports three different scopes explicitly:

- Setup: layout-model load, recognizer-model load, optional weight-format probe,
  compile-wrapper creation, and the first compiled call. Setup is excluded from
  page throughput.
- Coarse synchronized wall time: real layout inference, recognizer H2D, eager
  prefill, compiled decode, each complete request, and each complete page.
- `device_stage_s`: NPU-event execution time for vision/text-prefill substages.
  These values diagnose accelerator work and are not interchangeable with host
  wall latency.

Raw decode tok/s counts every `batch_size * graph_calls` arena slot, including
idle rows and completion look-ahead. Effective decode tok/s counts only real
generated tokens after the prefill-produced first token, including EOS. Both
divide by the same continuous-decode wall time, which includes slot admission,
D2H waits, and host retirement. The JSON separately reports model/argmax device
time, admission device time, idle slots, look-ahead slots, active-slot fraction,
and copied KV-prefix bytes. E2E output tok/s includes each request's first token
and EOS and divides by full page wall time.

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

`--batch-size` accepts 1, 2, 4, 8, and other powers of two. Use additional
`--image` arguments to process multiple pages; pages remain sequential.

Measured 910B validations are recorded in
[`NPU_FULL_PAGE_RESULT.md`](NPU_FULL_PAGE_RESULT.md) for the original B=1 path
and [`NPU_BATCHED_DECODE_RESULT.md`](NPU_BATCHED_DECODE_RESULT.md) for padded
fixed B=2 and B=4 decode. Those documents predate the continuous scheduler and
remain historical comparison points.
