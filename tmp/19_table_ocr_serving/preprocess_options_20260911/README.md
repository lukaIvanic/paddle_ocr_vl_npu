# Existing preprocessing options: 100-table serving comparisons

2026-09-11, one Ascend 910B2 physical NPU6. Runtime and benchmark launcher
commit `875863c09ae43492c5c633ab5ad5a02bc60fd723`. No product implementation,
model graph, scheduler or default was changed for this comparison.

## Contract

- Eight sequential runs: B2/C2 and B8/C8, each with Pillow/CPU normalization,
  Kornia/CPU normalization, Pillow/uint8, and Kornia/uint8.
- Current 60,416-row decode head; full first-token head, KV4096 and stopping
  behavior unchanged. All uploaded images remain encoded bytes: image decoding
  is included in latency. Pixel limits remain 28,224–802,816.
- Identical random-100 sequence, seed 1, from the frozen `be691de1` client.
  Order SHA256: `944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b`.
- Independent server per variant, existing constructor cache loading, one
  complete real-request warmup outside each measured client window. E2E means
  actual client submission to complete response. Device timing enabled equally.
- `preprocess_options_server.py` wraps the unchanged product HTTP/worker path
  solely to supply the two existing constructor arguments. Its readiness
  metadata records their resolved values. No research switches were added to
  the public product CLI. Exact commands and configuration are saved per run.
- `uint8` leaves CPU patches uint8, transfers them without floating conversion,
  then normalizes on the NPU before vision. All of that work counts in E2E.

## Results

All durations are seconds. CPU mean is total CPU preparation service time,
not exclusively exposed critical-path delay. QPS is closed-loop completed
tables/s, not an open-loop arrival-rate guarantee.

| B/C | Variant | CPU mean | E2E mean | E2E P95 | Tables/s |
|---|---|---:|---:|---:|---:|
| 2/2 | Current Pillow/CPU | 0.061505 | 0.684608 | 1.928796 | 2.867619 |
| 2/2 | Kornia | 0.056776 | 0.667768 | 1.917245 | 2.940667 |
| 2/2 | Uint8 → NPU normalization | 0.045093 | 0.654354 | 1.881408 | 2.998609 |
| 2/2 | Both | 0.038739 | 0.641529 | 1.870831 | 3.061967 |
| 8/8 | Current Pillow/CPU | 0.057958 | 1.207451 | 3.320476 | 5.882611 |
| 8/8 | Kornia | 0.054209 | 1.217371 | 3.365673 | 5.864473 |
| 8/8 | Uint8 → NPU normalization | 0.044200 | 1.170147 | 3.260791 | 6.077528 |
| 8/8 | Both | 0.036314 | 1.154483 | 3.209186 | 6.150543 |

Relative to fresh matching controls:

- Uint8: throughput +4.57% / +3.31%; P95 lower by 0.0474 / 0.0597 s (B2/B8).
- Both: throughput +6.78% / +4.55%; P95 lower by 0.0580 / 0.1113 s.
  CPU service reduced by 37.0% / 37.3%.
- Kornia alone: small improvement at B2, no demonstrated E2E benefit at B8.

These are single 100-request screening runs, without randomized variant order
or repeats. Small differences may include run variance. The current B2 control
was 2.0% slower in throughput than the previous CPU-readiness run, while P95
was close. Do not claim a repeatable 1000-request or open-loop milestone here.

## Exposed CPU-readiness delay

Observed FIFO-head/free-slot eligibility until CPU preparation completes.
Includes zeros for already-ready requests; not a counterfactual E2E saving.

| B/C | Variant | Requests >0.001 s | Mean | P95 | Max |
|---|---|---:|---:|---:|---:|
| 2/2 | Current | 97 | 0.057235 | 0.123560 | 0.333376 |
| 2/2 | Kornia | 96 | 0.052801 | 0.104808 | 0.282939 |
| 2/2 | Uint8 | 94 | 0.042040 | 0.101588 | 0.307218 |
| 2/2 | Both | 98 | 0.036971 | 0.090606 | 0.257962 |
| 8/8 | Current | 65 | 0.035488 | 0.114715 | 0.267641 |
| 8/8 | Kornia | 73 | 0.037581 | 0.107024 | 0.280594 |
| 8/8 | Uint8 | 64 | 0.028908 | 0.095366 | 0.249602 |
| 8/8 | Both | 66 | 0.025183 | 0.079572 | 0.196917 |

Counts need not fall when individual preparation gets faster: scheduling and
the moment each request becomes eligible also change. The combined option
reduces mean exposed delay by 0.0203 s at B2 and 0.0103 s at B8.

The image/prompt CPU stage fell from 0.0376 to 0.0170 s at B2 and from 0.0361
to 0.0151 s at B8. Uint8 also reduced pinning time. Uploaded-image decoding
remained about 0.018 s on average and is untouched by either option.

H2D device-region time fell from approximately 0.0035–0.0037 s to 0.0002 s
with uint8. That region is not the entire cost of the new path: normalization
runs afterward. The runtime measures `vision_input_normalize` internally, but
its response's selected `device_stage_s` fields omit it. Its duration therefore
cannot be isolated from these saved responses; it is included in E2E timing.

## Output and ownership checks

All eight runs: 100/100 successful requests, all EOS, 41,120 generated tokens
including EOS. Per-request crop size, prompt token count and projected image
token count match the corresponding control. No shorter output cap or missing
request accounts for the speed differences.

- B2 uint8: 100/100 native token streams and formatted texts match Pillow.
- B2 Kornia and both: 99/100 match; both change the same cell in
  `page_001227_table_10`, while preserving its 910-token output length.
- B8: all three alternatives match their B8 control 100/100.

The B2 change adds a `C` in the S750A_rp primer sequence. Ground truth was
checked read-only in `/workspace/datasets/OmniDocBench/OmniDocBench.json`,
page `scihub_mol.66.4.921.pdf_1.jpg`, `layout_dets` table HTML:

```text
Pillow: AACAGCCGTGCTGTGCACCCGGGGCGGGACGATACTGCGGGACAG
Kornia: AACAGCCGTGCTGTGCACCCGGGGCGCGGACGATACTGCGGGACAG
GT:     AACAGCCGTGCTGTGCACCCGGGCGCGGACGATACTGCGGGACAG
```

Kornia is closer in this cell, but retains an extra G. This local comparison
is not a full OmniDocBench quality evaluation or proof of general resize parity.

276 clean ownership snapshots across runs; each final snapshot is free. Direct
host `npu-smi` inspection after the sweep also found no process on NPU6. All
owned servers stopped. Cached vision setup was 6.23–6.79 s and cached decode
setup 0.210–0.247 s; no fresh model-graph compilation delay was observed.

## Evidence

`analysis.json` includes full latency and CPU distributions, device stage
summaries, generated-token totals, completion reasons and output diffs.
Run `python3 tmp/19_table_ocr_serving/preprocess_options_20260911/analyze.py`
to regenerate it from saved request records. Analysis asserts sequence/shape
parity and clean device ownership. The 30 local product tests also passed.

Downloaded archive SHA256:
`26f8a4d1b5ab87d4f36f1e6bb2f0badcc4e163cdb516354bfe0f8f19fcd698ac`.
