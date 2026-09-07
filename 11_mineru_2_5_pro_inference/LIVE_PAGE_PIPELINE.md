# Live PP-DocLayoutV3 + MinerU page pipeline

`run_page_pipeline.py` accepts ordinary page images. It loads both models once,
runs real PP-DocLayoutV3 detection, and sends in-memory crops into the same
bounded continuous MinerU recognition engine used by the full accuracy test.
No saved layouts, crop files, Paddle recognizer, PaddleX, or ground truth are
required. The existing OmniDocBench entrypoint remains backwards-compatible.

From the repository root on an initialized NPU host:

```sh
source npu-setup
/workspace/venvs/mineru_pro_vllm_py312/bin/python \
  11_mineru_2_5_pro_inference/run_page_pipeline.py \
  --model /workspace/models/MinerU2.5-Pro-2605-1.2B \
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors \
  --input-images /path/page1.png /path/page2.jpg \
  --output-dir /path/new-run
```

The layout frontend additionally requires `kornia-rs==0.1.14` and
`shapely==2.1.2`, as the Paddle
pipeline does. The supplied layout checkpoint must include `inference.yml`,
processor/config JSON, and its safetensors weights. Do not replace model configs
between model families. Model file hashes are recorded in the run manifest.

Use `--layout-backend mineru` for the original native layout model. The new
entrypoint defaults to `pp-doclayout-v3`; the older benchmark script continues
to default to `mineru`. `--dataset-json` plus `--images-dir` is the alternative
to ordinary image files. Input basenames/output stems must be unique.

## Shared execution contract

- The NPU-owning thread executes layout and recognition. CPU executors decode
  page images, prepare detector inputs, resolve polygons, crop, and prepare
  MinerU inputs. No background thread calls the layout model.
- Pages are bounded by `--streaming-page-window` (32). Recognition requests
  from different pages share the existing B32 scheduler. Completed pages are
  emitted individually through bounded artifact writers.
- Live input is also supported by `PaddleLayoutPageSource` with `PageInbox`:
  an empty open inbox is not EOF; explicit `close_input()` drains the stream.
  `run_decode_stream(engine, source)` uses the same source as the CLI. Construct
  and consume the source on the NPU-owning thread. Close the source and layout
  frontend in `finally`, and drain the output writer.
- The default recognizer is FP16, KV4096, PSE-sentinel IncreFA, NZ decode
  weights, NPU RoPE, compiled manual-FP32-layernorm vision, packed text prefill,
  and the 5,632-raw-vision-token pixel cap. Existing recognition caches are
  reused. There are no new recognition kernels or graph shapes.
- The detector follows Paddle's checkpoint dtype/preprocessing and runs eager
  on the same NPU by default. `--layout-graph-capture` is explicit. No automatic
  CPU fallback or per-chip inference claims are made.
- Layout setup is reported under setup, not measured page processing. Layout
  host timings are recorded separately from MinerU vision/text/decode metrics;
  host spans are not advertised as exclusive device time.

## Crop and output policy

This first live integration retains the saved-layout experiment's recognition
policy: original page pixels, white polygon masks, explicit Paddle-to-MinerU
label mapping, image/chart analysis off, and no new rotation inference.
Paddle's group ordering is retained, but crop images are not concatenated or
suppressed within a group. No Paddle-specific formula margin trimming, text
half-resolution scaling, or table-image tokenization is added.

The live detector/geometry replaces saved geometry, so the saved-run score is
not automatically a validated score for live detection. Compare the retained
region metadata and crop-pixel hashes before claiming parity.

Artifacts:

- `predictions/*.md`: MinerU's normal Markdown postprocessing.
- `content_lists/*.json`: recognized blocks and normalized boxes.
- `layout_regions/*.json`: live Paddle labels, pixel boxes, polygons, crop
  hashes/sizes, and layout-stage timings.
- `generation_trace.jsonl`: exact native generation IDs and request identity.
- `progress_shard_00.jsonl`: durable per-page completion journal.
- `run_manifest_shard_00.json` and `run_summary_shard_00.json`: settings,
  hashes, completion counts, layout calls, and recognition metrics.

The CLI currently handles page images; PDF rendering and HTTP transport are
separate interfaces, not implicitly introduced by this integration. Run only
one owner against a recognition cache root at a time. The validation launcher
`run_live_layout_validation.sh` acquires the existing serving cache lock.
