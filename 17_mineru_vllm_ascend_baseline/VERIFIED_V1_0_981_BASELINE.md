# Verified OmniDocBench v1.0 baseline

This record covers the stock vLLM-Ascend MinerU2.5-Pro baseline on one Ascend
910B2. It uses the exact 981-page OmniDocBench v1.0 corpus. The 128-page prefix
is a quality gate. The full 981-page result is accepted only after inference
and the pinned OmniDocBench evaluator both complete without missing pages.

## Corpus provenance

- OmniDocBench dataset revision:
  `f5f559bddf50e36f7f9899d842d0006f13ce8afc`
- Ground-truth file: `OmniDocBench_v1.0/OmniDocBench_v1_0.json`
- Ground-truth SHA-256:
  `2fafe9329dc92fc426b30036aee51c716b3fcdcc1d20cb964dc7670579533817`
- Ordered pages: 981, all unique
- Expected-image manifest SHA-256:
  `44eb67dc24fde6647e4cf83a55ff1289b175b953f4dc7507eb17ac365e74b051`
- Image verification: 981 present, 0 missing, 0 SHA-256 mismatches
- Original v1.0 image overrides: 227
- Byte-identical images reused from the current dataset: 754

The 227 overrides are the 116 note pages and 111 newspaper pages whose source
resolution changed after v1.0. The benchmark therefore does not substitute the
current higher-resolution copies for the historical v1.0 inputs.

The v1.0 source distribution is:

| Source | Pages |
| --- | ---: |
| PPT2PDF | 133 |
| academic literature | 129 |
| note | 116 |
| exam paper | 114 |
| newspaper | 111 |
| book | 104 |
| magazine | 97 |
| colorful text | 96 |
| research report | 81 |

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
- Model length: 8192
- GPU memory utilization: 0.9
- Maximum sequences: 512
- Maximum batched tokens: 16384
- Prefix caching: enabled
- Chunked prefill: enabled
- Static-kernel and NPU graph compilation: enabled
- Graph mode: `FULL_DECODE_ONLY`
- MinerU pipeline: one concurrent two-step extraction call, layout then OCR

## 128-page quality gate

Run directory:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n128_20260827T175952Z_af7293e
```

Inference source commit: `af7293e5812cbd420098604eaa67249ef5ea22ae`.

| Field | Result |
| --- | ---: |
| Selected pages | 128 |
| Completed pages | 128 |
| Failed pages | 0 |
| Engine setup | 527.8359 s |
| Benchmark wall | 104.3774 s |
| Inference | 101.3007 s |
| End-to-end throughput | 1.2263 pages/s |
| Inference-only throughput | 1.2636 pages/s |

All 128 pages produced non-empty Markdown and content-list JSON. The output
size median was 394 bytes; the range was 26 to 4,022 bytes. Fourteen outputs
were shorter than 100 bytes. Manual comparison against their ground-truth page
content found that all fourteen were genuinely sparse title or slide pages.
There were no duplicate payload groups, replacement characters, tracebacks, or
compiler import errors.

The prefix is not representative of all v1.0 sources: it contains only
PPT2PDF pages, with 114 Chinese, 11 English, and 3 mixed-language pages.

### Pinned evaluator result

Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

| Metric | Result | Page denominator |
| --- | ---: | ---: |
| OmniDocBench overall | 90.9406 | 128 |
| Text edit distance | 0.02519 | 123 |
| Formula CDM | 81.2667 | 3 |
| Table TEDS | 94.0738 | 13 |
| Table structure TEDS | 100.0000 | 13 |
| Reading-order edit distance | 0.08064 | 127 |

The evaluator matched all 128 pages. It used no page-match fallbacks and had no
CDM or TEDS errors or timeouts. The formula denominator is only three pages, so
the 128-page formula result is a smoke signal, not a stable corpus score.

## Full 981-page result

Run directory:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n981_20260827T181613Z_127542b
```

Inference source commit: `127542b2075bb3ea3f562ce714d2c179eebca246`.

| Field | Result |
| --- | ---: |
| Selected pages | 981 |
| Completed pages | 981 |
| Failed pages | 0 |
| Engine setup | 528.9945 s |
| Graph capture | 372 s |
| Image loading | 41.2388 s |
| Inference | 1,245.7371 s |
| Output writing | 1.0092 s |
| Benchmark wall | 1,288.0127 s |
| End-to-end throughput | 0.7616 pages/s |
| Inference-only throughput | 0.7875 pages/s |

The benchmark produced 981 non-empty Markdown files and 981 non-empty
content-list JSON files. There were no duplicate Markdown payloads,
tracebacks, or compiler import errors. Markdown size ranged from 9 to 89,145
bytes, with a median of 2,371 bytes and a 90th percentile of 7,474 bytes. The
shortest prediction was `NO.\n\nDate`; its ground truth is `NO. Date`.

Two pages contain one Unicode replacement character each. Both are localized
OCR decoding errors inside otherwise populated pages. The predictions were not
edited before scoring. A spot check of one page from each of the nine source
categories found plausible text, tables, and reading structure.

### Pinned evaluator result

Evaluation directory:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n981_20260827T181613Z_127542b/evaluation_v16_protocol_v10_pages
```

Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

Evidence hashes:

- Input manifest:
  `1d2be34ecaf04b992420dac0a379c740ae05ef6689ab72cb2cacab1b4f5ad3fd`
- Run summary:
  `cca5f1d865d6570f1a9d2d4a74f2251a37419b3b1baa182d20ee1ffd42249377`
- Prediction manifest:
  `da62b19e9c8ac9fc543e3711be35b1de6a65c365b05ba38f15f2dc9be4c4e9c8`

| Metric | Result | Page denominator |
| --- | ---: | ---: |
| OmniDocBench overall | 93.6797 | 981 |
| Text edit distance | 0.05705 | 921 |
| Formula CDM | 94.1947 | 53 |
| Formula edit distance | 0.12943 | 53 |
| Table TEDS | 92.5488 | 317 |
| Table structure TEDS | 95.8978 | 317 |
| Table edit distance | 0.05941 | 317 |
| Reading-order edit distance | 0.14063 | 973 |

The overall notebook score is the mean of text accuracy (`1 - edit distance`),
formula CDM, and table TEDS. Formula CDM evaluated 385 samples. Table TEDS
evaluated 428 samples. Two pages required the evaluator's bounded
page-timeout fallback. CDM and TEDS had zero timeouts, errors, or exceptions.

The full score is 2.7390 points higher than the 128-page PPT-only gate. The
gate's low formula denominator and single-source composition made it useful for
sanity checking, not for predicting the corpus score.

### Throughput comparison

The photographed 310P3 run reported 0.1597 pages/s on 981 pages. This 910B2 run
reached 0.7616 pages/s under the same measured-window contract, or 4.77 times
the reported 310P3 throughput. This is a cross-chip comparison, not a 910B
speedup attribution.

## Graph-cache postmortem

The one-page smoke, 128-page gate, and 981-page run used separate engine
processes. Each process captured the same 14 NPU graphs. This was avoidable.

The 128-page and 981-page logs both prove that two caches hit:

```text
Loaded npugraph_ex compilation cache ... artifact_compile_range_1_16384_subgraph_0
Directly load AOT compilation ... torch_aot_compile/.../rank_0_0/model
```

Those caches saved model graph compilation, but they did not persist the
static-kernel run packages used during ACL graph capture. The installed
vLLM-Ascend 0.21 path uses ordinary `torch.compile` integration rather than
`torch.npu.npugraph_ex.inference.cache_compile`. Its worker also uninstalls the
static-kernel package during shutdown. The result was 378 seconds of capture
for the 128-page process and 372 seconds for the 981-page process despite both
compile-cache hits.

For a future staged quality gate, keep one engine process alive across the gate
and continuation. Do not run a `compiled_async` page ladder as separate
processes and assume the compile cache removes graph capture. No extra compiled
rerun was performed after the 981-page result.

## Interpretation limits

The photographed 310P3 run reported 0.1597 pages/s but preserved no accuracy
score, input manifest, output hashes, or evaluator commit. This 910B2 result is
configuration-equivalent, not chip-equivalent.

No official MinerU2.5-Pro score was found for OmniDocBench v1.0. Current MinerU
documentation reports 95.39 for its high VLM lane and 95.30 for `vlm-engine`
on OmniDocBench v1.6. The MinerU2.5-Pro paper reports 95.69 on v1.6. This v1.0
score is 1.62 to 2.01 points below those contextual anchors, but the corpus,
checkpoint, and evaluator version differ. It is not a direct parity comparison.
