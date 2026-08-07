# Ascend 310P table OCR latency fundamentals

## Bottom line

The full OmniDocBench result of approximately **0.7 pages/s** is a throughput result. It is not a per-page latency result.

The system gets this throughput by processing many OCR crops concurrently. Decode batching amortizes one model-weight read across many active sequences. A standalone table request at batch size 1 cannot use that same amortization.

For the current PaddleOCR-VL model on one Ascend 310P3:

- measured batch-size-1 decode throughput is approximately **150 tokens/s**;
- the theoretical FP16 weight-bandwidth roof is only approximately **283 tokens/s**;
- a P90 table needs **951 decode steps**;
- a P99 table needs **3,091 decode steps**.

Therefore, **P99 latency below 2 seconds is physically impossible for this model in FP16 on one 310P**, even before image encoding, HTTP, preprocessing, queueing, and other work.

## Why batch-size-1 decode is memory-bound

The exact checkpoint used for this analysis is `PaddleOCR-VL-1.6`.

The weights used across every autoregressive decode step are:

| Component | Parameters | FP16 size |
|---|---:|---:|
| Text transformer layers | 254,840,832 | 509.7 MB |
| Language-model head | 105,906,176 | 211.8 MB |
| **Repeatedly used per decode step** | **360,747,008** | **721.5 MB** |

The input embedding table is not included in this lower bound. Decode reads only the selected embedding row, not the complete table, on each step.

The Atlas 300I Duo specification gives **408 GB/s total memory bandwidth across two Ascend 310-series processors**. This is approximately **204 GB/s per processor**. The absolute weight-streaming roof is therefore:

\[
\text{maximum decode throughput}
= \frac{204\ \text{GB/s}}{0.7215\ \text{GB/token}}
\approx 283\ \text{tokens/s}
\]

This roof assumes all of the following impossible conditions:

- 100% of memory bandwidth is available to model weights;
- every weight byte is transferred exactly once;
- no KV-cache traffic exists;
- no activation traffic exists;
- attention, normalization, RoPE, vector operations, argmax, and host work take zero time;
- there are no kernel-launch gaps or bandwidth inefficiencies.

The measured **150 tokens/s** is approximately **53% of this ideal roof**. More kernel tuning can improve it, but it cannot close the large gap between the physical roof and a 2-second P99 target.

## Real table output-length distribution

This distribution comes from all **665 OmniDocBench table crops** in the saved production table-API run.

`Decode steps` equals generated output tokens minus the first token produced during prefill.

| Percentile | Decode steps | Decode-only at 150 tok/s | Impossible bandwidth roof at 283 tok/s |
|---|---:|---:|---:|
| Minimum | 9 | 0.06 s | 0.03 s |
| Median | 211 | 1.41 s | 0.75 s |
| P75 | 451 | 3.01 s | 1.60 s |
| P90 | 951 | 6.34 s | 3.36 s |
| P95 | 1,496 | 9.97 s | 5.29 s |
| P99 | 3,091 | 20.61 s | 10.93 s |
| Maximum | 3,111 | 20.74 s | 11.00 s |
| Mean | 402.6 | 2.68 s | 1.42 s |

At the measured 150 tokens/s:

- **36.4%** of tables require more than 2 seconds for decode alone;
- **14.6%** require more than 5 seconds;
- **5.0%** require more than 10 seconds;
- **1.5%** require more than 20 seconds.

Even at the impossible 283 tokens/s bandwidth roof, **20.9%** of tables still require more than 2 seconds for decode alone.

## Encoder and fixed work

Decode is not the complete request.

At approximately **8,000 real vision tokens/s**, the vision encoder adds:

| Statistic | Vision time |
|---|---:|
| Mean | 0.295 s |
| Median | 0.299 s |
| P90 | 0.500 s |
| P99 | 0.508 s |

The current image-size cap makes the upper vision-time tail relatively flat. The output-token distribution causes the large total-latency tail.

Adding only vision execution to measured 150-token/s decode gives this optimistic compute estimate:

| Percentile | Decode + vision |
|---|---:|
| Median | 1.73 s |
| P75 | 3.39 s |
| P90 | 6.84 s |
| P95 | 10.48 s |
| P99 | 21.10 s |
| Maximum | 21.23 s |
| Mean | 2.98 s |

These values still exclude HTTP, image decoding, prompt preparation, text prefill, scheduling, result serialization, and queueing. These costs should be added as time, not represented as a fixed `0.5x` scaling factor. Queueing latency also depends on server load and has no fixed upper bound.

## Why 0.7 pages/s and high page latency can both be true

Throughput and latency measure different properties.

- **Throughput** measures how many pages the whole server completes per second under concurrent work.
- **Latency** measures how long one request waits from submission to completion.
- A page can contain many OCR crops.
- The page cannot finish before its slowest required crop finishes.
- Batching improves aggregate throughput because one decoder-weight pass advances many crops.
- Batching does not make a single isolated crop generate 64 tokens during one decode step.

Thus, a server can sustain approximately 0.7 pages/s while individual complex pages take many seconds to finish.

## What would be required for P99 below 2 seconds?

A P99 table requires 3,091 decode steps. Decode alone would need:

\[
\frac{3{,}091\ \text{tokens}}{2\ \text{s}}
= 1{,}546\ \text{tokens/s}
\]

This is:

- **10.3 times** the measured 150 tokens/s;
- **5.5 times** the ideal FP16 weight-bandwidth roof.

The target therefore cannot be reached by ordinary kernel tuning. It requires a fundamental change, such as:

- a much smaller decoder;
- substantially fewer generated tokens;
- a different output representation;
- aggressive weight quantization together with a much faster decode path;
- a device with much more memory bandwidth;
- or a latency contract below P99, such as median or P75.

## Source and scope

- Dataset: 665 OmniDocBench table crops from the production table-API result.
- Checkpoint: `PaddleOCR-VL-1.6`.
- Weight count: calculated directly from `model.safetensors` tensor shapes.
- Decode speed: measured batch-size-1 310P result, approximately 150 tokens/s.
- Memory-bandwidth source: [Huawei Atlas 300I Duo specifications](https://support.huawei.com/enterprise/en/doc/EDOC1100285916?section=j00e), which report 408 GB/s for the two-processor card.
- Scope: isolated table OCR on one 310P processor. Queueing under concurrent production load is separate.
