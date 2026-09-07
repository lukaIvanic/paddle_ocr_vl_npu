# Live PP-DocLayoutV3 → custom MinerU: full 1,651-page 910B validation

Inference commit `3e3fd74b`; reporting script `7e136abc`. Run completed on
2026-09-07 on one **Ascend 910B2**, physical NPU 4, `liteserver-c001-4`.
Both inference and evaluation exited 0. All 1,651 original OmniDocBench images
were processed successfully through live layout detection, cropping, custom
MinerU recognition and Markdown assembly. This is not saved-crop-only timing.

## Main results

| Metric | Live PP-DocLayoutV3 + MinerU | Saved Paddle layout + MinerU | Historical native MinerU layout + recognition |
|---|---:|---:|---:|
| Pipeline wall time | 1,769.729 s | 1,568.131 s | 2,029.966 s |
| Pages/s | **0.932911** | 1.052846 (excludes layout/crop export) | 0.813314 |
| Text accuracy | 96.303273 | 96.303417 | 96.306322 |
| Table TEDS | 93.101917 | 93.099046 | 92.303376 |
| Formula CDM | 96.993320 | 96.993653 | 96.729681 |
| Overall (mean of preceding three) | **95.466170** | 95.465372 | 95.113126 |
| Table structure TEDS | 95.787192 | 95.787192 | 95.103325 |
| Reading-order edit distance, lower is better | 0.138871 | 0.138871 | 0.125259 |

The live run is **14.70% faster** than the historical native-layout run,
saving 260.24 seconds (4.34 minutes). This is a historical comparison, **not a
controlled layout-only ablation**: the native run predates the crop pixel cap,
PSE-sentinel decode and the current vision path. It also used two warmup pages;
the live and saved-layout runs used zero. Native and live ran on physical NPU 4;
the saved-layout run used physical NPU 3.

Pipeline timing excludes model setup (24.222 s for the live run), includes
first-use cache loads within page processing and final output-writer drain,
and excludes accuracy evaluation. Completion 10 → completion 1,651 throughput
is 0.944254 pg/s: `(1651 - 10) / (last_completion_s - tenth_completion_s)`.
Inputs are dataset images, not original PDFs; no PDF rendering is measured.

## Accuracy and input audit

- The same frozen evaluator `2b161d010d2e3aff77a0edef359ea3a6411d23cd` and
  TeX Live 2025 / ImageMagick runtime were used. Evaluation took 865.83 s.
- All 1,651 pages matched. CDM scored 2,352 formula samples; TEDS scored 665
  table samples. Zero matcher timeout fallbacks, metric timeouts, errors or
  exceptions. Raw results and runtime verification are in `evaluation/`.
- 32,051 recognition requests; 32,016 EOS completions, 35 length-capped
  completions. No MinerU-generated layout requests.
- Relative to the saved-layout full run, all ordered recognition labels,
  bounding boxes, prompts and prompt token IDs match. 61 crop-pixel hashes on
  53 pages differ; their exact cause was not established by this benchmark.
- Only 12 requests have different generated token IDs: five with different
  crop inputs and seven with identical logged inputs. 1,639/1,651 complete
  Markdown pages are byte-identical. Overall changes by only +0.000798 points.
- The better three-metric overall does not imply every quality metric improved:
  reading-order edit distance is worse than the historical native-layout run.

## Settings and timing observations

`command.txt` is the invocation; `run_manifest_shard_00.json` records the fully
expanded arguments, model/config hashes and dataset hash. Both models reside
on the same NPU in the same process.

- PP-DocLayoutV3 checkpoint FP32, eager, layout graph capture off.
- MinerU FP16, continuous B32/KV4096 decode, PSE-sentinel IncreFA, NZ text
  weights; compiled manual-FP32-LayerNorm/nn.Linear vision path.
- Crop min/max pixels: 25,088 / 1,103,872 (5,632 raw vision tokens maximum).
- Existing production vision, text-prefill and decode cache roots reused.
- Streaming page window 32, CPU preparation depth 64. `image_analysis=False`:
  this is page parsing, not validated chart-content extraction.
- 1,651 live layout calls; main-thread layout host wall total 177.826 s.
  Detailed layout fields are host timings and overlap other pipeline activity;
  do not add them as non-overlapping NPU device times.
- Decode active-slot fraction 97.82%; no idle rows while ready work waited.
  Device decode 247.038 s; vision transformer blocks 441.359 s; text
  transformer prefill 355.108 s. Full details are in the run summary.
- Live wall time exceeds the saved-layout run by 201.60 s. This difference
  includes live detection/cropping and runtime variability, not a clean
  isolated layout-kernel cost.

## Reproduction and retained artifacts

From the 910B repo root, after pulling the source and selecting a free NPU via
the committed launcher:

```sh
RUN_ROOT=tmp/11_mineru_2_5_pro_inference/live_paddle_full1651_3e3fd74b LIMIT=1651 \
  bash 11_mineru_2_5_pro_inference/run_live_layout_validation.sh
RUN_ROOT=tmp/11_mineru_2_5_pro_inference/live_paddle_full1651_3e3fd74b LIMIT=1651 \
  bash 11_mineru_2_5_pro_inference/run_serving_accuracy.sh
```

The above is the recorded run root: **choose a new output directory when
rerunning**, but retain the existing graph-cache roots. No 310P run is implied.

Full predictions, token traces, progress logs, live layout geometry/crop hashes
and evaluation details remain at:

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/live_paddle_full1651_3e3fd74b
```

`comparison.json` is generated by `report_live_pipeline.py`, comparing the run
above against `paddle_layout_full1651_542b3fdc` and
`serving_streaming_1651_ae4c947c` under the same remote experiment root.
Reporting changes did not alter inference code or rerun prediction generation.
