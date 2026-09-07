# PP-DocLayoutV3 regions → MinerU recognition accuracy experiment

Opt-in only. `saved_paddle_crops.py` exports the final `page_regions.jsonl`
geometry from a completed PP-DocLayoutV3 full-corpus run. Coverage must match
the dataset exactly. It reads **no saved OCR content** and uses no GT boxes,
labels, or reading order. Original page pixels are cropped using experiment
09's white polygon mask, saved losslessly as PNG, and individually hashed.

This is not a replay of byte-identical historical Paddle recognizer inputs:
no Paddle text-crop half scaling or group concatenation is applied. Regions
remain in saved `parsing_res_list` order. MinerU's edge-ratio/min-edge helper
and image processor still run. Generic Paddle captions map to text, without
guessing table/image ownership; all other mappings are explicit in the exporter.
Image/chart recognition stays off. There is no embedded-table-image masking
or rotation inference in this adapter. These are part of the hybrid frontend
contract, not claims of a pure detector-box-only ablation.

`--saved-layout-manifest` feeds these crops into the existing production
streaming recognizer, decoder, postprocessor and Markdown writer. It requires
`--warmup-pages 0` so no native layout warmup can accidentally execute.
Model code and graph cache keys are unchanged. Run summaries record the input
manifest hash and request counts; there must be zero layout requests.

For the 910B validation launcher set `SAVED_LAYOUT_MANIFEST`, `DATASET_JSON`,
`LIMIT`, `RUN_ROOT`, `WARMUP_PAGES=0`, `PROCESSOR_MAX_PIXELS=1103872`, and
`INCREFA_LENGTH_MODE=pse_sentinel_310p`. Run smoke using the export's
`smoke_dataset.json`, then the unchanged full dataset. Use the existing
`run_serving_accuracy.sh` with matching `RUN_ROOT`, `DATASET_JSON`, and `LIMIT`.

Compare full overall/text/TEDS/CDM scores and per-page deltas using the frozen
evaluator. Historical full custom scores predate the pixel cap; explicitly
label that confound unless a matched native-layout control is rerun.
