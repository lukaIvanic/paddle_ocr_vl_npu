# Selective 5,632 → 3,072 vision-token crop cap experiment (910B)

Completed 2026-09-07, inference commit `a23fec49`, report `3a1dc3c8`.
One Ascend 910B2, physical NPU **6**, for both paired runs.
The production default remains **1,103,872 pixels / 5,632 raw vision tokens**.
Candidate: **602,112 pixels / 3,072 tokens**; min_pixels remains 25,088.

## Results: full reconstructed 1,651-page corpus

| Metric | Control: 5,632 tokens | Candidate: 3,072 tokens | Delta (points) |
|---|---:|---:|---:|
| Text accuracy | 96.303273 | 96.234624 | -0.068648 |
| Table TEDS | 93.101917 | 93.520705 | +0.418788 |
| Formula CDM | 96.993320 | 97.053718 | +0.060398 |
| Overall | **95.466170** | **95.603016** | **+0.136846** |
| Table structure TEDS | 95.787192 | 96.152605 | +0.365413 |

Reading-order edit distance (lower is better): 0.138871 → 0.138992.
Both full evaluations passed, with 1,651 matched pages, 2,352 CDM samples,
665 TEDS samples and zero timeout fallbacks, metric errors or exceptions.
Control scores match the original full live-layout baseline exactly.

### Local regressions matter despite the overall gain

| Scored pages | Improved | Regressed | Unchanged |
|---|---:|---:|---:|
| Text (1,557) | 22 | 22 | 1,513 |
| Tables (458) | 44 | 47 | 367 |
| Formulas (313) | 4 | 4 | 305 |

Two large regressions were inspected directly in the saved crop traces:

- `book_en_[搬书匠#893][Pyomo—Optimization Modeling in Python].2012.英文版_page_016.png`,
  text block 2: page text accuracy **100 → 1.511335**. Control finishes at EOS
  after 529 tokens; capped output repeats table-of-contents leader dots and
  hits the 3,336-token generation limit. Its -98.488665 page-point change
  accounts for about 0.0633 of the 0.0686 corpus-level text-point loss.
- `page-dca64e05-1ce6-49ee-857a-9b5d05a87357.png`, table block 0:
  page TEDS **85.316734 → 0.228833**. Control finishes at EOS after 1,560
  tokens; capped output degenerates into repeated `<lcel>` tokens and hits
  its 3,351-token limit.

These are observable generation failures, not merely cosmetic metric drift.
This experiment does not establish the internal cause of the repetition.
A lower-cap-first policy with a high-resolution retry for detected failures is
a plausible next experiment, **not implemented or evaluated here**.

## Measured performance versus estimated full-pipeline speed

Only **1,227 affected crops across 469 pages** were rerun at each cap.
The other **30,824 recognition outputs were reused**, and layout detection
was not rerun. Timings include selected-crop inference plus full-page
reconstruction, not full E2E page processing.

| Measured selected-replay quantity | Control | Candidate |
|---|---:|---:|
| Wall time, excluding model setup | 313.071 s | 204.847 s |
| Selected crops/s | 3.919 | 5.990 |
| Vision transformer device time | 122.303 s | 61.050 s |
| Text prefill device time | 70.524 s | 20.711 s |
| Decode device time | 48.617 s | 50.028 s |
| Raw vision tokens | 5,915,168 | 3,609,956 |
| Text prefill overflow calls | 949 | 0 |
| Text packed-prefill calls | 278 | 1,227 |
| Decode active-slot fraction | 86.33% | 86.22% |

This is **1.528× measured selected-replay throughput**, 34.57% less wall
time, saving 108.224 s. Model setup (23.522 / 22.780 s) is excluded;
existing caches were reused, and first decode calls took 0.304 / 0.370 s.

The lower cap both halves vision time and puts every selected input onto the
compiled packed text-prefill path instead of the overflow path. It also leaves
more output capacity under KV4096: affected-table length-capped outputs drop
from **12 to 3**, while generated table tokens rise from 187,009 to 194,964.
Those counts include new failures as well as resolved ones; they are not proof
that the same nine crops were simply fixed.

**Estimated, not measured, full-pipeline throughput:** subtracting the paired
108.224 s saving from the previous 1,769.729 s live run yields 1,661.504 s,
or **0.993678 pg/s**, versus 0.932911 (about +6.5%). Different batching,
overlap, device/run variability and the historical baseline's physical NPU 4
limit this estimate. Do not quote 1,651 divided by selective-replay wall as
E2E page throughput. A full rerun is needed to measure the actual E2E gain.

## Selection and validation

- Affected types: 896 text, 256 table, 34 equation, 33 title, 6 algorithm,
  1 header and 1 footer. Full selection is in `selection.json`.
- Selection uses baseline image-pad token count × 4 × 196 > 602,112. The
  checkpoint's patch_size=14 and merge_size=2 are validated, as are the
  original model, processor, tokenizer and dataset hashes.
- Exact live geometry and the original page decoder/polygon crop routine are
  used to regenerate only selected crop PNGs. Both raw and helper-resized
  pixel hashes match the full live baseline before either cap is applied.
- Baseline raw-output reconstruction reproduces all 1,651 Markdown pages
  byte-for-byte before inference. No GT boxes or GT text are model inputs.
- The 2-page NPU smoke reruns two selected crops and passes. Both full replay
  traces contain exactly the expected 1,227 request IDs, no extras.
- Control reproduces **all 1,227 selected token sequences** and **all 1,651
  Markdown pages** exactly. Candidate changes 337 selected token sequences;
  1,431 complete Markdown pages remain byte-identical to baseline. All 1,182
  pages without selected crops remain byte-identical in both runs.
- Unaffected raw block outputs seed the normal full-page postprocessor. The
  replay traces contain newly inferred crops only, not copied output records
  mislabeled as new inference. Input/output membership checks enforce this.
- Frozen evaluator commit is `2b161d010d2e3aff77a0edef359ea3a6411d23cd`, with
  TeX Live 2025 / ImageMagick, identical to the baseline. Evaluation wall times:
  control 875.19 s, capped 865.35 s.

## Provenance and reproduction

Remote baseline:

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/live_paddle_full1651_3e3fd74b
```

Remote experiment root:

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/crop_cap3072_2bd4f39e
```

`frozen/` retains the hash-verified crop PNGs, geometry, saved raw outputs and
manifest. `attempt2/{smoke,control,capped}` retains full logs, predictions,
token traces, timing records and evaluations. An initial launcher attempt
stopped before model loading because an older helper required
`global_request_stream=True`; its failure is retained in the outer root.
The corrected runner uses the successful live entrypoint's actual streaming
preset, with an explicit test against its recorded settings.

```sh
# Run on 910B after sourcing npu-setup. Use new OUTPUT/ROOT directories.
python 11_mineru_2_5_pro_inference/prepare_crop_cap_replay.py \
  --reference-run BASELINE --output ROOT/frozen --max-pixels 602112
python 11_mineru_2_5_pro_inference/run_crop_cap_replay.py \
  --reference-run BASELINE --root ROOT
python 11_mineru_2_5_pro_inference/report_crop_cap_replay.py \
  --reference-run BASELINE --root ROOT
```

Use the validated MinerU interpreter from each retained `run.log.command`.
The runner takes the existing cache roots from the baseline, locks the serving
cache owner, runs smoke → control → capped → both evaluations, and stops on
validation failure. No production max-pixels default or 310P result changed.
