# Experiment 08: offline real-layout E2E, B=1

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
       compiled B=1 single-token decode until EOS/length limit
       D2H tokens and detokenization
  -> reading-order text
```

There is deliberately no batching, overlap, ready queue, or hot-swap scheduler.
The process owns one synchronous `SequentialRecognizer`, and the page pipeline
does not begin region N+1 until region N is complete.

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
  and text prefill are eager; only the flat static B=1 decode module is compiled.
- EOS detection uses the existing NPU event plus pinned-host-flag technique. It
  can execute one look-ahead graph call, then trims exactly at the first EOS.

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

The primary decode rate is generated tokens after the prefill-produced first
token, including EOS, divided by compiled decode wall time. Executed graph calls
per second is separate because asynchronous EOS checking can add one look-ahead
call. E2E output tok/s includes each request's first token and EOS and divides by
full page wall time.

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
  --cache-length 2048 \
  --max-new-tokens 768
```

The default artifact directory is timestamped under
`tmp/08_offline_e2e_b1/`. It contains `run.json`, per-page reading-order text,
and an annotated layout image. Pass `--save-crops` only when the actual crop
images are useful. `--max-regions N` is a debug-only partial-page mode and is
recorded as such in the JSON.

Use additional `--image` arguments to process multiple pages. They remain fully
sequential in this experiment.
