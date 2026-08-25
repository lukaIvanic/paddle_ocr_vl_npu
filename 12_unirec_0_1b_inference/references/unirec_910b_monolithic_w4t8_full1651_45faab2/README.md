# UniRec monolithic W4/T8 full-1651 PSS baseline

This is the proper architectural baseline for the low-memory runner. It ran the
original integrated `run_opendoc_batched_unirec.py` path over all 1,651
OmniDocBench pages on physical 910B2 NPU 7 at commit `45faab2`.

The run used four layout-and-prefill worker processes, T8 recognition crop
preprocessing, compiled FP32 B2 layout, K20 L4 vision buckets, continuous B128
decode, cross-KV 1320, self-KV 2048, NZ decoder weights, and a 57,344-row LM
head. TBE and knowledge-bank service counts were already reduced, so the result
isolates process ownership rather than compiler-service duplication.

## Result

| Metric | Monolithic W4/T8 |
|---|---:|
| Pages | 1,651 |
| Crops | 32,110 |
| Process-tree peak PSS | 30,981,315,584 bytes, 30.981 GB |
| Process-tree peak summed RSS | 34,158,673,920 bytes, 34.159 GB |
| External process wall | 329.700 s |
| Cold process throughput | 5.0076 pages/s |
| Measured pipeline wall | 229.108 s |
| Measured pipeline throughput | 7.2062 pages/s |
| Peak live CPU cross-KV budget | 3,006,775,296 bytes, 3.007 GB |

The PSS peak contained these dominant processes:

| Process | PSS |
|---|---:|
| Coordinator and decoder | 2.527 GB |
| Layout/prefill worker 0 | 6.446 GB |
| Layout/prefill worker 1 | 7.142 GB |
| Layout/prefill worker 2 | 8.334 GB |
| Layout/prefill worker 3 | 6.393 GB |
| Small forkservers and resource tracker | approximately 0.139 GB |

The four heavy workers account for 28.315 GB. Each owns a layout model, UniRec
model, compiled graph objects, and a CANN/Torch runtime. This duplication is the
dominant memory cost. Cross-KV is not the main cause.

## Comparison with the final low-memory path

| Metric | Monolithic | Low-memory | Change |
|---|---:|---:|---:|
| Peak PSS | 30.981 GB | 4.369 GB | -85.90% |
| PSS ratio | 7.09x | 1.00x | 7.09x less RAM |
| Exact recognition rows | 32,110 | 32,110 | exact |

The sorted request ID, label, text, and token rows are byte-identical between
the two runs. Both produce this SHA-256:

```text
5656893a9bac377717df75a19d8a26ee51306a7482eb8ba7c07fd59ffdb9300e
```

The low-memory architecture keeps four CPU crop workers but makes them
Torch-free and approximately 65 MB PSS each. One process owns the NPU models
and graphs. A bounded NPU queue replaces the 3.007 GB CPU cross-KV bank.

The memory saving has a real throughput cost. The monolithic external process
rate is 5.0076 pages/s. The low-memory inference process rate is 3.9349 pages/s,
21.4% lower, before its deferred 76.5-second output writer. Therefore the
earlier 2.1% throughput comparison applies only to two low-memory allocator
variants, not to this original monolithic architecture.

## Evidence

- `preflight.txt`: commit and physical NPU.
- `run_summary.json`: complete pipeline settings and timings.
- `process_tree_memory.json`: full process-tree PSS/RSS sampled every 50 ms.

