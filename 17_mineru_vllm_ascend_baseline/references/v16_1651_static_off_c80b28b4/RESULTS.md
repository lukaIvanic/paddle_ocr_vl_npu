# Stock vLLM-Ascend full v1.6, static kernels off

Stock vLLM-Ascend completed and scored all 1,651 OmniDocBench v1.6 pages on one
Ascend 910B2, physical NPU4. It completed with zero inference failures. Hot
end-to-end throughput was **0.72344 pages/s** and overall accuracy was
**95.4996**.

Inference used source commit `c80b28b44e366d93408b4228d148768972e3bf49` on
2026-09-03. The evaluation launcher was committed as `3f6b4ca2`. The model and
dataset hashes match the prior full stock run and the same-day custom run.

## Runtime contract

- stock vLLM 0.21.0 and vLLM-Ascend 0.21.0rc1;
- torch-npu 2.10.0, FP16, no quantization, tensor parallel size one;
- `AsyncLLM`, model length 8,192, maximum sequences 512;
- maximum batched tokens 16,384, prefix caching and chunked prefill enabled;
- `FULL_DECODE_ONLY`, NPU graphs enabled, static kernels disabled;
- capture sizes 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28 and 32;
- `MinerUClient` async two-step extraction with `image_analysis=True`;
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

The normal vLLM compile cache loaded in 0.099 seconds. Total `torch.compile`
setup took 4.48 seconds. All 14 graphs captured in about four seconds. Full
engine setup took 83.58 seconds and is outside the hot benchmark timer.

## Performance

| Measurement | Result |
| --- | ---: |
| Completed / failed pages | 1,651 / 0 |
| Hot benchmark wall, including image load and output writes | 2,282.1487 s |
| Hot pages/s | 0.723441 |
| Inference | 2,168.4659 s |
| Inference-only pages/s | 0.761368 |
| Client setup | 0.0269 s |
| Image load | 112.0026 s |
| Output write | 1.6533 s |
| Engine setup, excluded from hot timing | 83.5801 s |

The custom full run completed the same corpus on NPU4 at 0.813314 pages/s in
2,029.9659 seconds. Custom was 12.42% faster in pages/s and saved 252.18 seconds
inside the hot window. This is a full-pipeline comparison, not an isolated
scheduler A/B. Stock uses image analysis and an 8,192-token model limit. Custom
uses no image analysis, KV4096 and exact token tracing. The custom trace work is
included in its hot timing.

## Accuracy

| Metric | Current stock, static off | Custom | Page denominator |
| --- | ---: | ---: | ---: |
| Overall | 95.4996 | 95.1131 | 1,651 selected pages |
| Text accuracy | 95.8639 | 96.3063 | 1,557 |
| Text edit distance, lower is better | 0.041361 | 0.036937 | 1,557 |
| Formula CDM | 96.9764 | 96.7297 | 313 |
| Table TEDS | 93.6585 | 92.3034 | 458 |
| Table structure TEDS | 96.2604 | 95.1033 | 458 |
| Reading-order edit distance, lower is better | 0.127648 | 0.125259 | 1,638 |

Overall is the mean of page-weighted text accuracy, formula CDM and table TEDS.
The stock result is 0.3865 points above custom. Custom is better on text and
reading order. Stock is better on formula CDM and table TEDS. The custom result
has 39 KV4096-capped requests; the stock run uses model length 8,192.

The evaluator is pinned to `2b161d010d2e3aff77a0edef359ea3a6411d23cd`, with
the same frozen TeX Live, ImageMagick and Ghostscript runtime as the custom run.
All pages matched with no fallback. CDM scored 2,352 samples with no timeout,
error or exception. TEDS scored 665 samples. One 33,693-byte prediction against
an 8,990-byte ground-truth table hit the 120-second timeout, the same case as
the prior full stock evaluation. Evaluation exited zero in 992.98 seconds.
Prediction Markdown was not altered before scoring.

The previous stitched static-kernel-on stock result was 95.5064. The current
clean static-off result differs by -0.0068 points. Of 1,651 Markdown files,
1,521 are byte-identical and 130 differ. There are no missing or extra pages.
The one empty prediction is the expected figure-only page.

## Static-kernel interpretation

The controlled 128-page A/B remains the authority for this option. Static
kernels improved inference-only throughput by 0.80% in one pair, from 0.46075
to 0.46443 pages/s. This was too small to separate from variance. They added
443.65 seconds to engine setup because process-local packages were rebuilt.
The measured break-even estimate was about 25,780 pages. This full run therefore
uses the operational static-off default. It is not a rerun of that A/B.

## Evidence

`stock_output.tar.gz` contains every prediction, content list, manifest, exact
command, exit status and inference log. `evaluation.tar.gz` contains the
evaluator configuration, prediction hashes, ground truth, top-level metric
JSON files, runtime record, logs and exit status. Regenerable CDM render images
are omitted. Verify both files with `SHA256SUMS` before extracting.

All 1,651 local prediction files match the evaluator manifest hashes. The
overall score was independently recomputed from the three page-weighted metric
components. Compact copies of the run and evaluator summaries are beside this
report.

Server run root:
`/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_block_default_static_kernel_off_nall_20260903T173522Z_c80b28b4`.
