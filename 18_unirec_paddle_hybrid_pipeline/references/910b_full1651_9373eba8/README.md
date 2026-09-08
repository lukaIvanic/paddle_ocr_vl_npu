# Full 1,651-page hybrid, one 910B2

Inference commit `9373eba8`, physical NPU 6, 2026-09-08. Exact invocation is in
[command.txt](command.txt). Existing caches reused; no change to model kernels,
preprocessing, routing or scheduling from the accepted hybrid smoke. This run
adds prefill counters/event envelopes, not a new execution path.

Exit 0. All 1,651 expected annotation stems have both Markdown and page JSON,
with no missing or extra stems. All 30,557 crop request IDs are unique:
28,125 UniRec text, 751 Paddle tables, 1,681 Paddle formulas. Per-crop token IDs
and text remain in the full run's `hybrid/recognition_trace.jsonl` on the 910B;
a local copy is `/tmp/hybrid-full1651.ErER31/recognition_trace.jsonl`.

## Throughput and timing

Processing E2E: **928.063 s / 1.779 pages/s**, including layout, crop production,
recognition, assembly and synchronous output writes. Measured model/frontend
setup: 42.383 s; including this setup: **1.701 pages/s**. Imports and initial
dataset enumeration precede the setup timer. Cache/first-use work within the
processing window remains included; this is not a strictly warmed-kernels-only
throughput result.

| Coordinator operation | Seconds | Processing share |
|---|---:|---:|
| Layout/crop preparation | 238.10 | 25.7% |
| UniRec prefill | 357.53 | 38.5% |
| Paddle prefill | 141.94 | 15.3% |
| UniRec decode turns | 127.06 | 13.7% |
| Paddle decode turns | 62.74 | 6.8% |

These are non-overlapping host wall intervals. Decode turns include engine
admission/copy/bookkeeping and completion callbacks, including page writes;
prefill includes its CPU work. They are not pure device-kernel durations.

| Decoder | Capacity | Mean active slots | Active utilization | Useful tok/s in decode turns | Raw slots/s in decode turns |
|---|---:|---:|---:|---:|---:|
| UniRec | 128 | 113.75 | 88.87% | 13,369 | 15,043 |
| Paddle | 64 | 44.47 | 69.48% | 6,506 | 9,426 |

UniRec: 14,933 calls, 1,698,670 useful decode tokens. Its narrower graph + argmax
+ blocking CPU-token-read interval totals 92.336 s (18,397 useful / 20,701 raw
tok/s). This is **not device-only timing**.

Paddle: 9,240 calls, 408,192 useful decode tokens. Device events around decode
model + argmax total 56.005 s (7,288 useful / 10,559 raw tok/s). Active utilization
includes 2,675 lookahead slots; excluding those, useful utilization is 69.03%.
Per-model tokenizers differ: do not treat their token rates as identical work.
Whole-engine elapsed/residual timers in raw summaries include cooperative
pauses and are unsuitable as exclusive scheduler-overhead measurements.

## Findings

- Prefill plus layout account for 79.5% of processing time. Decode underfilling
  exists, particularly Paddle, but decode is not the largest wall-time category.
- UniRec packed text prefill: 1,952,568 real source tokens versus 15,867,720
  physical tokens: **12.3% useful density**, across 12,021 S1320 calls (2.34
  crops/call on average). UniRec compiled vision row utilization is **41.3%**.
  This makes grouping at slot-driven refill boundaries a concrete investigation
  target. It does not by itself establish the speedup from changing that policy.
- Paddle vision: 2,301,560 real patches / 2,653,568 physical patches. Transformer
  device time 52.371 s: **43,947 real patches/s**. Text prefill device time
  17.687 s: **34,319 input tokens/s**.
- UniRec vision-call event envelopes total 213.138 s: **9,161 real output
  tokens/s**. Text-prefill envelopes total 44.477 s: **43,901 real source
  tokens/s**. These include transfer/submission gaps and use encoder-output
  tokens, not Paddle's vision-patch units; do not compare the two vision rates
  as equal work.
- Peak Torch allocation **22.20 GiB**, reservation **23.57 GiB**. This is not
  total process/device memory, and unchanged residency is not demonstrated to
  fit 310P.
- 65 UniRec length-capped crops, 9 Paddle KV-cap stops, and 2 Paddle repetition
  stops. All remaining crops ended at EOS. No accuracy evaluation was run;
  successful completion is not a quality/parity claim.

Raw [run_summary.json](run_summary.json), [run.log](run.log), and derived
[metrics.json](metrics.json) are retained. Recreate the latter with:

```sh
python 18_unirec_paddle_hybrid_pipeline/summarize_run.py /path/to/hybrid
```

The report's output-length distributions include EOS when produced and exclude
UniRec's decoder-start token. Eight local routing/scheduling/reporting tests
passed; those tests are separate from this real-NPU full-corpus validation.
