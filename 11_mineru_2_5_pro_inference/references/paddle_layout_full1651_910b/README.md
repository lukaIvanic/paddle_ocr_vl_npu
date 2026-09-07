# Saved PP-DocLayoutV3 regions + MinerU: full 910B accuracy result

2026-09-07. **All 1,651 pages completed and evaluated successfully.** This
demonstrates a working hybrid parser, not a controlled proof that one layout
model is globally superior.

| Metric | Historical custom MinerU layout | Saved Paddle layout + MinerU | Delta (points) |
| --- | ---: | ---: | ---: |
| Overall | 95.113126 | 95.465372 | +0.352246 |
| Text accuracy | 96.306322 | 96.303417 | -0.002905 |
| Table TEDS | 92.303376 | 93.099046 | +0.795669 |
| Formula CDM | 96.729681 | 96.993653 | +0.263972 |
| Table structure TEDS | 95.103325 | 95.787192 | +0.683867 |

Reading-order edit distance (lower is better) worsened from **0.125259** to
**0.138871**. Overall is the arithmetic mean of text accuracy, TEDS, and CDM;
reading order is not included in it.

## Coverage and execution

- Verified saved Paddle geometry covers the dataset exactly: no missing, extra,
  or duplicate page names. Exported 32,051 original-resolution polygon-masked
  PNG crops (2.3 GB), with per-file hashes. Saved Paddle OCR text is never used.
- Seven-page 910B smoke completed first: 151 recognition requests, zero layout
  requests, six representative cases plus an additional formula page. One
  math-heavy text crop repeated to its length limit; its output was not altered.
- Full run: 32,051 recognition requests, **zero MinerU layout requests**;
  32,016 EOS completions and 35 length-limited requests (10 equations, 10 text,
  13 tables, 2 footers). No failed pages or missing outputs.
- Recognition/assembly window: **1568.131 seconds (26.1 minutes)**. Evaluation:
  **865.17 seconds (14.4 minutes)**. Precomputed layout detection and crop export
  are outside that recognition window; do not label this full PDF-to-Markdown
  E2E throughput.
- Frozen evaluator: 2,352 formula samples and 665 table samples; zero page-match
  timeout fallbacks, zero CDM/TEDS timeouts, exceptions, or errors. Page-weighted
  denominators: text 1,557, tables 458, formulas 313, reading order 1,638.

## What changes across pages

- Text: 223 pages improve, 379 regress, 955 are unchanged. Aggregate text
  accuracy is virtually identical despite major changes on individual pages.
- Tables: 77 pages improve, 63 regress, 318 unchanged.
- Formulas: 62 pages improve, 54 regress, 197 unchanged.
- Vocabulary example `notes_f7f010b78016aeebd76e56d9283eb67f_72.jpg`:
  text accuracy **0.464 → 96.918**. Paddle supplies individual text regions
  instead of the native MinerU table interpretation.
- Pyomo contents example: **98.405 → 100.000** text accuracy.
- Vertical-text example `page-56023e3d-ae9b-486d-b7bb-419624d18356.png`:
  **85.350 → 7.036**. The output includes repeated text; this adapter does not
  infer crop rotations. Other inspected regressions involve handwritten math
  repetition and text grouped into image/table regions. These observations do
  not establish a single cause for every regression.
- The largest -100-point text outlier has only the short scored GT phrase
  “联系我们”; it is not evidence that an entire text-heavy page was lost.

All per-page changes and the worst/best lists are in `accuracy_comparison.json`.

## Comparison caveats

The baseline is the historical `serving_streaming_1651_ae4c947c` run, not a new
matched control. The hybrid uses the current **1,103,872 max-pixel cap** (5,632
raw vision tokens), PSE-sentinel IncreFA, and current manual-FP32 vision
layernorm path; the historical run predates these settings. Model weight and
dataset hashes match. Thus **+0.352 points cannot be attributed solely to
layout**. No matched native-layout control was launched without Luka's answer
to the optional follow-up question.

This experiment uses final saved Paddle regions and their order, not raw
detector output. It does not reproduce Paddle's old half-resolution text crop
scaling or crop-group concatenation. Generic captions map to text; image/chart
recognition is disabled; no orientation inference or embedded-table-image
masking is added. See the exporter and `SAVED_PADDLE_LAYOUT_ACCURACY.md` for
the explicit hybrid contract. No tuning was performed after inspecting smoke
or full-corpus accuracy.

## Reproduction / provenance

- Inference source: `542b3fdc0edcf67742bfd870a4c2c655b99603c1`; full run on
  physical **910B2 NPU 3**. Smoke used NPU 4. NPU 4 subsequently became occupied,
  so the initial full-run launch refused to run there; no other job was touched.
- Run root on 910B: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/paddle_layout_full1651_542b3fdc`.
- Crop export: `tmp/11_mineru_2_5_pro_inference/paddle_saved_crops_1651_f1bbec69/manifest.json`.
- Crop manifest SHA256: `fad1db4db11a6d90781966d9c1f36c5b54551dd1e0dfc2c3bf9f0e62fc5c2761`.
- Source Paddle run: `tmp/09_persistent_page_engine/910b_full_cap4096_text05_b64_pse_target768_898ced7/output/page_regions.jsonl`.
- Source geometry SHA256: `eab1bd3c93194a0c8ad762501ee480fae78ff0d0b03da373ed8e1912a372de12`.
- Dataset SHA256: `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
- The exact command, device, model hashes, scoring diagnostics and comparison
  are preserved beside this report. Raw crops, predictions and token traces
  remain in the run directories; multi-gigabyte crops are not committed.
