# Custom MinerU full OmniDocBench v1.6 result

The bounded streaming custom pipeline completed all 1,651 pages on one Ascend
910B2, physical NPU4, with zero failures or skipped pages. Its overall score is
**95.1131**. Hot end-to-end throughput is **0.81331 pages/s**.

Inference used commit `ae4c947c15d43d9858dfe3608e25bdbb6fa43565` on 2026-09-03.
The evaluation launcher was committed as `6c62a26f`. The model, processor,
FP16 kernels, B32/KV4096 settings, 32-page window and existing graph-cache paths
match the earlier 384-page streaming validation. No vLLM engine is used by the
custom pipeline.

## Accuracy

All metrics below use the full 1,651-page v1.6 corpus, ground-truth SHA256
`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.

| Metric | Custom | Previous stock run | Page denominator |
| --- | ---: | ---: | ---: |
| Overall | 95.1131 | 95.5064 | 1,651 selected pages |
| Text accuracy, 100 × (1 − edit distance) | 96.3063 | 95.7955 | 1,557 |
| Text edit distance, lower is better | 0.036937 | 0.042045 | 1,557 |
| Formula CDM | 96.7297 | 97.0395 | 313 |
| Table TEDS | 92.3034 | 93.6841 | 458 |
| Table structure TEDS | 95.1033 | 96.2852 | 458 |
| Reading-order edit distance, lower is better | 0.125259 | 0.128741 | 1,638 |

Overall is the mean of page-weighted text accuracy, formula CDM and table TEDS.
The independently recomputed result equals the evaluator's notebook summary.
The custom result is 0.3933 points below the prior stock result. The comparison
is to our recorded stock run, not a new stock rerun or a claim about the latest
public leaderboard.

The evaluator is pinned to `2b161d010d2e3aff77a0edef359ea3a6411d23cd`, with
TeX Live 2025/pdfTeX 1.40.28, ImageMagick 7.1.1-47 and Ghostscript 9.55.0.
The experiment-09 wrapper uses process-isolated page matching and TEDS, with
unchanged matching/metric functions. Workers: 24 matching, 12 CDM, 12 TEDS.
The primary page deadline is 420 seconds; TEDS has a 120-second deadline.
All 1,651 pages matched with zero fallbacks. All 2,352 formula samples and 665
table samples scored with zero timeouts, errors or exceptions. Evaluation
exited zero in 843.97 seconds. Raw Markdown was not transformed before scoring.

The previous stock run used the same evaluator commit and frozen tools, but
recorded two page fallbacks and one TEDS timeout.
Its model/pipeline settings also differ, including an 8,192-token model limit.
This is a benchmark-score comparison, not an isolated implementation ablation.

## Performance and output audit

| Measurement | Result |
| --- | ---: |
| Hot pipeline wall time, including token trace and writer drain | 2,029.96594 s |
| Hot pages/s | 0.813314 |
| Reported model setup | 22.88045 s |
| Two-page warmup, excluded from hot timing | 16.75965 s |
| Decode device time | 329.72092 s |
| Prefill wall time | 1,333.85166 s |
| Decode-slot occupancy, iteration weighted | 99.7682% |
| Decode-slot occupancy, decode-device-time weighted | 99.7742% |
| Empty decode slots while ready work waited | 0 |
| Layout / recognition requests | 1,651 / 35,285 |
| Generated tokens, including EOS | 2,577,276 |

Setup and warmup are separate from hot timing. Model-file checksum work is
also outside the hot timer; the reported setup field is not total launch wall
time. The prefill and device timing counters are diagnostic and should not be
added indiscriminately. Decode-slot occupancy is not whole-program NPU usage.

The run kept at most 32 live pages, 64 queued CPU preparations, 63 live
generation states and two pending page writes. All trace/request/token counts
reconcile. All 1,651 local prediction hashes match the files supplied to the
evaluator. The raw token trace SHA256 is
`c626235e19cdd523a34bd45b80fd584a47f8e81db205ba0d63a22c111f5befbb`, verified on
both the server and Mac.

The first 384 Markdown pages are byte-identical to the prior streaming run.
All layout token sequences and 5,100/5,102 recognition sequences are exact.
The two differences are verified random table-image placeholder renamings with
unchanged final Markdown and block JSON. Their only changed input field is the
crop image hash. Both empty Markdown pages have only figure/text-mask ground
truth and no scored text.

## KV4096 limitation

There are 39 length-capped requests on 37 pages: 12 layouts, 20 tables, five
text regions and two equations. Every capped request has exactly 4,096 total
prompt-plus-generated tokens. These predictions were retained unchanged in the
score; this result is not an uncapped quality baseline.

Of the 458 scored table pages, 21 contain a capped layout or table request.
Those pages account for 1.0245 of the 1.3807-point net TEDS gap to stock,
about 74%. The remaining 437 pages contribute 0.3562 points. This localizes
most of the gap to capped pages, but does not prove that increasing KV alone
would recover it. No KV8192 rerun was performed.

## Reproduction and preserved artifacts

Run from the pull-only Blue Zone checkout, with NPU4 free:

```sh
MODE=streaming LIMIT=1651 RUN_ROOT=tmp/11_mineru_2_5_pro_inference/my_full1651 \
  bash 11_mineru_2_5_pro_inference/run_serving_validation.sh
LIMIT=1651 RUN_ROOT=tmp/11_mineru_2_5_pro_inference/my_full1651 \
  bash 11_mineru_2_5_pro_inference/run_serving_accuracy.sh
```

Use a new run directory. The launcher refuses to overwrite prior outputs.
The evaluator preparation checks the dataset hash, complete page membership,
prediction/content/progress agreement and zero failures before scoring.
The 25 experiment-11 CPU tests passed, including the new preparation checks.

Verify `SHA256SUMS` before extraction:

- `streaming.tar.gz`: every prediction, content list, exact token trace,
  progress record, manifest, summary, inference command, commit and log.
- `evaluation.tar.gz`: ground truth, prediction hashes, evaluator configuration,
  commands, source hashes, frozen-runtime record, all top-level per-sample and
  per-page metric JSON files, final summary, logs and exit status. Regenerable
  CDM render images are omitted; the originals remain on the server.
- `trace_audit.json`, `table_gap_audit.json`, `evaluation_prep_summary.json` and
  `evaluator_run_summary.json`: directly readable audit and score summaries.

Server run root:
`/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/serving_streaming_1651_ae4c947c`.
