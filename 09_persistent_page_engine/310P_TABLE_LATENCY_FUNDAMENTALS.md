# PaddleOCR-VL table latency on Ascend 310P3

## 1. Measured end-to-end result

We created our custom PaddleOCR-VL 1.6 e2e page pipeline, and evaluated on **OmniDocBench v1.6** using **1 x Ascend 310P3**.

The measured full-benchmark result was approximately:

| Metric                    | Result |
|---------------------------|---:|
| Dataset                   | OmniDocBench v1.6 |
| Pages                     | 1,651 |
| Hardware                  | 1 × Ascend 310P3 |
| Concurrency               | ×64 |
| End-to-end throughput     | **0.7 pages/s** |
| Text-block Edit Distance  | 94.9 |
| Table Page-TEDS           | 94.4 |
| Formula Page-CDM          | 97.4 |
| Official Overall accuracy | **95.59** |

Although **throughput** is 0.7 page/s, this does not mean latency per page is `1 / 0.7 = 1.43` seconds.

## 2. Latency is not throughput

Concurrency is good for the throughput metric. If you give us 70 pages, we will return OCR in <=100 seconds for all of them. However, each page may return at for example 30s, 60s, 80s. This would mean >>10s latency per page.

## 3. CBG latency requirement

The CBG team requested:

> **P99 table latency below 2 seconds.**

The metric would be all tables from OmniDocBench v1.6.
To evaluate this requirement, we must first know how many output tokens each table needs.

## 4. Output-token distribution for OmniDocBench v1.6 tables

The following distribution comes from all **665 table crops** in OmniDocBench v1.6.

| Statistic | Decode tokens |
|---|---:|
| Minimum | 9 |
| Median | 211 |
| Mean | 402.6 |
| P75 | 451 |
| P90 | 951 |
| P95 | 1,496 |
| P99 | 3,091 |
| Maximum | 3,111 |


## 5. Decode speed required for latency below 2 seconds

If we wanted to achieve P99 <2 seconds, it is clear we need to produce 3000+ output tokens in that time:

$$
\frac{3{,}091\ \text{tokens}}{2\ \text{seconds}}
= 1{,}546\ \text{tokens/second}
$$

The lower percentiles also require high batch-size-1 throughput:

| Target | Decode tokens | Throughput required for 2 s |
|---|---:|---:|
| P90 | 951 | 476 tok/s |
| P95 | 1,496 | 748 tok/s |
| P99 | 3,091 | **1,546 tok/s** |

These numbers allow only two seconds for decode. They leave no time for image loading, preprocessing, vision encoding, text prefill, HTTP, scheduling, or result serialization.

## 6. What is theoretically possible?

If we want to minimize latency, we should use 1x concurrency. That way, multiple tables won't fight for the same pipeline resources.

This means we want batch size 1 (B1) decoding.

To understand limits of B1 decoding, we look at the following fact: for one output token, the NPU needs to load all model weights from HBM to L2 cache. So if we know our NPU memory bandwidth, and model size, we can get peak theoretical tok/s for B1 decoding.

The NPU hardware bandwidths are:

| Device | Memory bandwidth | Source |
|---|---:|---|
| Ascend 310P3 | **204 GB/s per processor** | [Atlas 300I Duo specifies 408 GB/s across two processors](https://support.huawei.com/enterprise/en/doc/EDOC1100285916?section=j00e) |
| Ascend 910B2 environment | **1.6 TB/s** | [64 GB Atlas 300I A2 specification](https://www.hiascend.com/hardware/accelerator-card) |

## 7. Why one output token requires reading the decoder weights

Autoregressive decoding runs the complete text transformer and language-model head once for each new output token.

The exact PaddleOCR-VL 1.6 checkpoint contains:

| Decode component | Parameters | FP16 weight bytes |
|---|---:|---:|
| Text transformer layers | 254,840,832 | 509.7 MB |
| Language-model head | 105,906,176 | 211.8 MB |
| **Total used for every decode iteration** | **360,747,008** | **721.5 MB** |

The complete decoder checkpoint also contains an input embedding table. It is not included here because one decode iteration reads only the selected embedding row, not the complete table.

The active 721.5 MB weight set is much larger than on-chip cache. At batch size 1, producing one new token therefore requires loading approximately **721.5 MB of FP16 decoder weights from device memory at least once**.

This gives a simple bandwidth roof:

$$
\text{Peak decode tokens/second}
\leq
\frac{\text{memory bandwidth}}{\text{FP16 decoder weight bytes per token}}
$$

This is an optimistic upper bound. It assumes:

- 100% memory-bandwidth utilization;
- every weight byte is transferred exactly once;
- zero KV-cache and activation traffic;
- zero attention, normalization, RoPE, vector-operation, and argmax cost;
- zero kernel-launch and host overhead.

No real implementation can meet all these assumptions.

## 8. Theoretical peak and measured batch-size-1 decode throughput

### Ascend 310P3

$$
\frac{204\ \text{GB/s}}{0.7215\ \text{GB/token}}
= 283\ \text{tokens/s}
$$

### Ascend 910B2

$$
\frac{1{,}600\ \text{GB/s}}{0.7215\ \text{GB/token}}
= 2{,}218\ \text{tokens/s}
$$

### Comparison with measured results

| Device | Theoretical FP16 roof | Measured B1 decode | Measured fraction of roof | P99 decode time at measured speed |
|---|---:|---:|---:|---:|
| Ascend 310P3 | 283 tok/s | **150 tok/s** | 53% | **20.6 s** |
| Ascend 910B2 | 2,218 tok/s | **750 tok/s** | 34% | **4.12 s** |

Even the impossible 310P3 roof gives:

$$
\frac{3{,}091}{283} = 10.9\ \text{s}
$$

The requested P99 throughput of 1,546 tokens/s is:

- **10.3 times** the measured 310P3 result;
- **5.5 times** the 310P3 physical FP16 roof;
- **2.1 times** the measured 910B2 result;
- approximately 70% of the impossible 910B2 bandwidth roof, before any other work.

## 9. Conclusion

For the current 360.7M-active-parameter FP16 decoder:

- **P99 below 2 seconds is physically impossible on one Ascend 310P3.**
- Ordinary kernel tuning cannot bridge a 5.5× gap beyond the memory-bandwidth roof.
- The current 150 tok/s result already reaches approximately 53% of that ideal roof.
- The current 910B2 path is much faster, but its measured 750 tok/s still gives more than four seconds of P99 decode time before encoder and service overhead.

Meeting the requirement needs a fundamental change, such as:

- fewer generated tokens or a more compact table representation;
- a substantially smaller decoder;
- lower-bit decoder weights with a genuinely faster low-bit execution path;
- hardware with much more memory bandwidth;
- or a less strict latency percentile.

## Data and calculation scope

- Dataset: OmniDocBench v1.6.
- Table population: 665 ground-truth table crops.
- Model: PaddleOCR-VL 1.6.
- Parameter counts: calculated directly from the production `model.safetensors` tensor shapes.
- Decode results: warmed batch-size-1 measurements, approximately 150 tok/s on 310P3 and 750 tok/s on 910B2.
- Latency calculations are decode-only unless stated otherwise. Production queueing can increase latency further.
