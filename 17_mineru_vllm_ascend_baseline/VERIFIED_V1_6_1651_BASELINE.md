# Verified OmniDocBench v1.6 full baseline

This record covers the stock vLLM-Ascend MinerU2.5-Pro baseline on the complete
1,651-page OmniDocBench `v1.6_full` corpus. It ran on one Ascend 910B2 and was
scored with the pinned official evaluator. The measured overall score is
95.5064. The official MinerU2.5-Pro row is 95.75, so the absolute difference is
0.2436 points. This is accepted as quality parity within a quarter point.

## Corpus provenance

- Ground-truth file: `/workspace/datasets/OmniDocBench/OmniDocBench.json`
- Ground-truth SHA-256:
  `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- Ordered pages: 1,651
- Unique image names: 1,651
- Image files present: 1,651
- Input-manifest SHA-256:
  `22c85ca2d2ce13dfa88078d93a5502bd45cd2df24151afd4b8d4fcce762abf40`

This is the current canonical OmniDocBench `v1.6_full` corpus. It is not the
historical 981-page v1.0 corpus used in the earlier baseline.

## Fixed runtime contract

- Hardware: one Ascend 910B2
- Physical device: NPU 6
- Model: `MinerU2.5-Pro-2605-1.2B`
- vLLM: 0.21.0
- vLLM-Ascend: 0.21.0rc1
- torch-npu: 2.10.0
- Tensor parallel size: 1
- Weight dtype: float16
- Quantization: none
- Engine: `AsyncLLM`
- Model length: 8,192
- GPU memory utilization: 0.9
- Maximum sequences: 512
- Maximum batched tokens: 16,384
- Prefix caching: enabled
- Chunked prefill: enabled
- Static-kernel and NPU graph compilation: enabled
- Graph mode: `FULL_DECODE_ONLY`
- Capture sizes: 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32
- MinerU pipeline: one concurrent two-step extraction call, layout then OCR

## Inference and recovery

Primary run directory:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n1651_20260828T081412Z_d703317
```

The primary process loaded both normal compile caches:

```text
Loaded npugraph_ex compilation cache ... artifact_compile_range_1_16384_subgraph_0
Directly load AOT compilation ... torch_aot_compile/.../rank_0_0/model
```

The vLLM compile-cache load took 0.097 seconds and total `torch.compile` setup
took 4.40 seconds. Stock vLLM-Ascend still rebuilt the process-local static
kernel run packages and spent 362 seconds capturing the 14 decode graphs.

Inference reached 1,651 of 1,651 pages in 37 minutes 18 seconds. Output writing
then failed on dataset index 1,491 because the atomic JSON temporary filename
exceeded the filesystem's 255-byte component limit. At failure time, the run
had saved complete Markdown and content-list pairs for indices 0 through 1,490.
The failure record reports 2,353.5981 seconds from benchmark start through the
partial output-writing failure.

The writer was corrected to use a short PID-and-hash temporary name. A recovery
run processed the exact suffix at offset 1,491:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n160_20260828T090645Z_d9b83b8
```

The recovery completed all 160 pages with no failures. Its separately measured
result was 355.3296 seconds end to end, or 0.4503 pages/s; inference alone was
332.9922 seconds, or 0.4805 pages/s. The overlapping page at index 1,491 had
the same prediction SHA-256 in the primary process and recovery process:

```text
64ee15c71793324e1e69aae0e1daafd095674b1e6770d7072bf41a507cb62c26
```

The stitched output is an accuracy artifact only. It contains 1,651 Markdown
files and 1,651 content-list JSON files, with no missing or extra page stems.
One Markdown file is intentionally empty; its ground truth contains only a
figure and text-mask elements. There are no duplicate Markdown payload groups
and no Unicode replacement characters. Predictions were not edited before
scoring.

The stitched run-summary SHA-256 is
`1b9cb56c694c54c5ee317fbbbe20f385db510a97f278a6462a6967a6e3e81877`.
The prediction-manifest SHA-256 is
`c376d23733f59a9999653f5a7a89e324067ba1531788b133caaa6c17a13ab408`.

## Pinned evaluator result

Evaluation directory:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n1651_20260828T081412Z_d703317/evaluation_v16_full_recovered_b626658
```

Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

| Metric | Measured | Official | Difference | Page denominator |
| --- | ---: | ---: | ---: | ---: |
| OmniDocBench overall | 95.5064 | 95.75 | -0.2436 points | 1,651 |
| Text edit distance | 0.04204 | 0.036 | +0.00604 | 1,557 |
| Formula CDM | 97.0395 | 97.45 | -0.4105 points | 313 |
| Table TEDS | 93.6841 | 93.42 | +0.2641 points | 458 |
| Table structure TEDS | 96.2852 | 95.92 | +0.3652 points | 458 |
| Reading-order edit distance | 0.12874 | 0.120 | +0.00874 | 1,638 |

The evaluator matched all 1,651 pages and exited zero. Two pages required its
bounded page-match fallback. Formula CDM scored 2,352 samples with zero
timeouts, errors, or exceptions. Table TEDS scored 665 samples. One 33,693-byte
prediction against an 8,990-byte ground-truth table reached the 120-second TEDS
timeout; no TEDS errors or exceptions occurred.

The official row comes from the
[OmniDocBench `v1.6_full` end-to-end table](https://github.com/opendatalab/OmniDocBench#end-to-end-evaluation).
It uses the same overall formula:

```text
((1 - text edit distance) * 100 + table TEDS + formula CDM) / 3
```

The measured 95.5064 is also 0.1836 points below the MinerU2.5-Pro paper's
95.69 result, 0.1164 points above MinerU's documented 95.39 high-VLM result,
and 0.2064 points above its documented 95.30 `vlm-engine` result. Those three
are context anchors; the official OmniDocBench table row is the direct
comparison.

## Throughput limit

Do not report the stitched prediction set as one clean throughput run. The
primary process completed all model inference but failed during output writing,
so it did not save the exact inference-only timing field. The recovery process
was a separate 160-page run and paid another engine startup and graph capture.
The accuracy score is valid because every prediction is preserved unchanged
and the exact suffix recovery was verified. A single-run 1,651-page throughput
number would require one clean rerun and is not inferred here.

## Online-serving graph cache behavior

Stock online `vllm serve` uses the same EngineCore and graph compiler as this
offline `AsyncLLM` lane. It loads the normal vLLM compile cache and Torch AOT
cache. It captures the static decode graphs once during server startup and then
reuses them for every request handled by that process. A persistent server
therefore avoids repeated capture.

In vLLM-Ascend 0.21, a server restart still rebuilds the static-kernel run
packages. The installed compiler path does not call
`torch.npu.npugraph_ex.inference.cache_compile`, and worker shutdown uninstalls
the process's static-kernel package. Online mode improves process lifetime; it
does not add persistence for those run packages across restarts. The public
[graph-mode guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/graph_mode.md)
also describes static-kernel compilation as a service-startup cost.

## Evidence

The compact manifests, commands, failure and recovery records, evaluator
configuration, metric outputs, runtime report, and stage report are in
`references/v16_1651_b626658/`. The 1,651 generated predictions remain on the
910B container and are not duplicated in Git.
