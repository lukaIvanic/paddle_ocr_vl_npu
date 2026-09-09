# Half-ready capacity: first 384 pages, one 910B2

Both runs at source `3a45bf83` exited zero on physical NPU 6. Candidate ran
first, followed by a fresh original-capacity control. Exact commands, remote
artifact roots, timing partitions, counters and parity are in
[comparison.json](comparison.json). Existing graph caches were reused.

| Metric | Original | Half ready |
|---|---:|---:|
| UniRec / Paddle NPU ready capacity | 128 / 64 | 64 / 32 |
| UniRec / Paddle CPU preparation capacity | 128 / 64 | 128 / 64 |
| UniRec / Paddle active decode batch | 128 / 64 | 128 / 64 |
| Pipeline wall seconds | 185.498 | 184.085 |
| Pages/s | 2.0701 | 2.0860 |
| Torch peak allocated GiB | 17.1771 | 16.3339 |
| Torch peak reserved GiB | 18.5137 | 17.6445 |
| Whole-device sampled peak MB | 22,859 | 21,969 |
| Whole-device baseline MB | 3,425 | 3,419 |
| Peak increase over baseline MB | 19,434 | 18,550 |
| NPU sampler observations | 261 | 272 |
| Setup seconds, outside pipeline wall | 43.814 | 48.188 |

The smaller pools saved 890 MB of sampled total device peak (0.869 GiB), or
884 MB after subtracting each run's baseline. NPU-SMI was sampled every second
with no query errors: these are sampled peaks, not a guarantee against shorter
transient peaks. Total device usage includes the baseline; it is not an
additional allocation to add to Torch memory. Torch, logical pool bytes and
NPU-SMI are overlapping views and must not be added together.

Throughput was 0.77% higher with the smaller pools in this pair: no observed
material throughput penalty, not evidence of a repeatable speedup. The earlier
`abe23c9b` original-capacity run was 176.412 s (2.1767 pg/s), so comparing only
against that older run would have suggested a slowdown that this fresh control
did not reproduce. Both current runs retain large vision-prefill envelope
outliers; these timings alone do not identify their cause.

## Where the memory saving came from

- Paddle fixed ready-KV storage: 1,811,939,328 -> 905,969,664 bytes, exactly
  0.84375 GiB saved. Ready-row length remains 1,536; active KV remains 4,096.
  Peak leased rows were 64 -> 32, max prompt length 1,033 in both runs, and all
  689 leases were released in each run.
- UniRec logical compact ready cross-KV: high-water rows 128 -> 64, but the
  byte high-water was **118,996,992 bytes (113.484 MiB) in both runs**. Row and
  byte maxima need not occur together: real source lengths vary. Halving the
  row ceiling did not reduce this observed byte peak. This counter excludes
  temporary prefill tensors and allocator retention; it is not total UniRec
  HBM usage or a time-weighted occupancy metric.
- CPU preparation was unchanged, including observed high-water requests:
  UniRec 65 and Paddle 14 in both runs. No RAM queue reduction was needed.
- Active self/cross KV tensors, decode batch sizes, model weights, generation
  limits, preprocessing and default sequential vision execution were unchanged.

Thus almost all of the observed allocation saving is explained by Paddle's
smaller fixed pool, not a multi-GiB reduction in UniRec's dynamic ready queue.

## Correctness and work counters

All **4,346 complete crop trace records** matched by unique request ID, including
token IDs, text, stop reason, model, page, prompt and block index: 3,657 UniRec
and 689 Paddle crops. This is output parity, not a new ground-truth evaluation.
The candidate also matched the earlier sequential `abe23c9b` control.

UniRec used 3,099 decode calls and 166,917 effective decode tokens in both runs.
Paddle used 4,958 -> 4,953 calls with unchanged 193,205 active token-slots and
192,399 effective tokens. Continuous admission order can change batch packing;
the smaller reservoir did not force smaller active decode batches. Both owner
timing partitions closed with zero accounting error.

Local policy tests passed (39 total, five torch-dependent skips); all three
small-ready tests, including the real Paddle CPU scheduler admission test,
also passed in the 910B environment before inference. A full active batch can
be admitted through multiple smaller-ready groups using the existing copy path.

## Reproduction / scope

Use the normal benchmark command with only these NPU-ready overrides:

```sh
--unirec-ready-capacity 64 --paddle-ready-capacity 32
```

Paddle private ready rows default to the selected ready capacity. An explicit
`--paddle-ready-cache-rows 64` would retain the larger allocation and should not
be carried over when reproducing this memory experiment. CPU preparation stays
at 128/64, active decode B128/B64, and ready-row length stays 1,536.

Defaults have not been changed. This test does not prove the full 1,651-page
run fits on 310P: its runtime, total HBM and later-page peaks still need direct
validation there. No 310P result is claimed.
