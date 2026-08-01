# Accuracy localization lab

This lab is the small, stable follow-up to the broad 32-page 910B/310P
comparison.  It is designed for exact Git-mediated investigation rather than
benchmark scoring.

The committed [`cases.json`](cases.json) defines two related scopes:

- seven complete OmniDocBench pages, original indices 8 through 14, for layout,
  crop construction, routing, and page-assembly comparisons;
- ten diagnostic crops selected from Phase 38: four first-token cases spanning
  formula/text/table, two later-prefix cases, the 34-versus-1120-token runaway,
  one additional long text case, and two control candidates from the
  low-divergence page 11.

Cases are keyed by original dataset image name plus layout block index.  They
remain stable when a smaller `--offset 8 --limit 7` run renumbers request IDs.
The manifest also preserves the historical 32-page 910B token contract so a
changed small-corpus routing result is explicit rather than silently treated as
the old reference.

## Input identity

Pass `--recognition-input-fingerprints` to the real OmniDocBench runner.  For
every crop, the trace then contains:

- a SHA-256 over the exact PIL mode, dimensions, and raw crop bytes;
- individual dtype/shape/byte hashes for `pixel_values`, `image_grid_thw`,
  `input_ids`, `attention_mask`, `position_ids`, and `rope_deltas`;
- one combined prepared-input hash.

The hashes are computed on CPU before pinning and H2D.  They add CPU work and
are disabled by default outside this lab.  They do not add NPU synchronization.

## Fixed seven-page run

Both devices must use the same command contract, including an explicit text
bucket ladder.  Device-specific cache roots may differ, but the selected graph
shapes may not.

```sh
python 09_persistent_page_engine/scripts/run_omnidocbench.py \
  --offset 8 --limit 7 \
  --batch-size 32 --cache-length 4096 \
  --preprocessor-min-pixels 28224 \
  --decode-backend torchair \
  --decode-optimization combined_apply_static_actual \
  --vision-backend torchair \
  --vision-attention prompt_flash_attention \
  --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 \
  --vision-promptfa-align-128 --vision-padding bucket \
  --vision-packing greedy --vision-pack-target 1920 \
  --vision-router-lookahead 32 \
  --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 \
  --text-packing production_group \
  --text-pack-buckets 128,256,512,1024 \
  --text-pack-max-members 32 \
  --layout-device npu --no-layout-graph-capture \
  --preprocess-all-pages-first --no-timeline \
  --recognition-input-fingerprints \
  --output-dir <output>
```

The normal model, frontend, packing, prefill, decode, and page-assembly paths
remain in use.  This is not a mocked crop runner.

## Comparison

```sh
python 09_persistent_page_engine/scripts/accuracy_lab.py \
  --reference-output <910b-output> \
  --candidate-output <310p-output> \
  --output-dir <comparison-output>
```

The report contains the complete seven-page crop population as well as the ten
selected cases.  It cross-tabulates token divergence against raw-crop identity,
prepared-input identity, vision route, text-prefill route, and recorded request
metadata.  Its strongest classification is
`MODEL_EXECUTION_DIFFERENCE_PROVEN`: at least one divergent crop has exact
prepared inputs and exact execution routes on both devices.  That isolates the
difference to NPU/compiler model execution and justifies a later per-stage
tensor comparison.

The lab does not call the OmniDocBench evaluator.  Exact token differences are
localization evidence, not by themselves an accuracy score.
