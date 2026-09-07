# Live PP-DocLayoutV3 + MinerU validation — 910B

2026-09-07. The live detector integration passed a seven-page NPU smoke and
an independent ordinary-image CLI invocation. **This is not a new full-corpus
accuracy result or a 310P validation.**

Seven-page run: `live_paddle_smoke7_5fff57a5`, source `5fff57a5`, physical
910B2 NPU 3, FP32 PP-DocLayoutV3 detector with graph capture off and the current
FP16/PSE-sentinel/B32/KV4096 MinerU recognizer. Recognition graphs reused the
existing production cache roots.

- 7/7 pages completed, zero failures.
- 7 real PP-DocLayoutV3 model calls, 151 MinerU recognition requests, zero
  generated MinerU layout requests.
- Compared with the prior saved-layout seven-page experiment: ordered
  recognition labels/bboxes exact on 7/7 pages, crop pixels exact on 7/7,
  Markdown byte-exact on 7/7.
- Page processing, including live layout and final artifact drain: 36.102 s.
  This tiny smoke includes first-use cache loads and is not a throughput claim.
- `parity.json` contains every page comparison. `run_summary_shard_00.json`
  records the effective model hashes, settings, layout and recognition metrics.

The additional `live_paddle_image_661fcd1b` invocation passed the vocabulary
page via `--input-images`, without a dataset JSON or saved crop manifest. It
completed successfully and its Markdown matched the seven-page invocation
byte-for-byte. Its summary is `direct_image_run_summary.json`.

The initial `live_paddle_smoke7_7676790d` attempt failed at CPU polygon handling
because the MinerU environment lacked Shapely. We installed the same pinned
dependencies as the working Paddle frontend: `kornia-rs==0.1.14` and
`shapely==2.1.2`, without changing torch/model packages. Explicit dependency
and label-map checks now run during layout setup.

The live path reuses Paddle's image decoding, NPU detector, polygon geometry,
and group ordering. It deliberately retains the previously tested MinerU crop
policy instead of adding Paddle-specific image concatenation, formula-margin
trimming, or table-image tokenization. Both models remain loaded, CPU work is
bounded, and only the inference-owning thread runs the detector.

See `../../LIVE_PAGE_PIPELINE.md` for the user-facing CLI and integration
contract. The new CLI defaults to live Paddle layout; the historical benchmark
entrypoint retains its native-MinerU default. Native layout remains selectable.
