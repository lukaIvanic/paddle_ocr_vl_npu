# UniRec full-1651 low-memory result

This directory records the completed four-worker low-memory UniRec run on one
Ascend 910B2. Inference ran at project commit
`8272f87a9d1ca4cd7f58dd66525fa301daf0e6e5`. The evaluator compatibility fix
was committed afterward as `c64b615` and does not change inference.

## Result

| Gate | Result |
|---|---:|
| Pages | 1,651 |
| Recognition crops | 32,110 |
| CPU crop processes | 4 |
| Resize threads per crop process | 8 |
| Full inference process wall | 419.577 s |
| Full inference throughput | 3.9349 pages/s |
| Peak inference process-tree PSS | 4,368,691,200 bytes, 4.369 GB |
| Deferred writer peak PSS | 1,223,302,144 bytes, 1.223 GB |
| Deferred writer wall | 76.488 s |
| Frozen evaluator Overall | 90.1876% |

The matching pre-main-allocator baseline was commit `9bb6dd4` with identical
W4/T8, layout, vision, and decode settings. It ran at 4.0191 pages/s and peaked
at 5,421,636,608 bytes PSS. The final allocator change reduced peak PSS by
19.4% and changed throughput by -2.1%.

The final and baseline recognition traces contain the same 32,110 request IDs,
texts, and generated token rows. Their sorted normalized trace SHA-256 is:

```text
5656893a9bac377717df75a19d8a26ee51306a7482eb8ba7c07fd59ffdb9300e
```

## Accuracy

The evaluation used the clean OmniDocBench evaluator at commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd`, TeX Live 2025/pdfTeX 1.40.28,
ImageMagick 7.1.1-47, Ghostscript 9.55.0, 12 page/TEDS workers, and 64 CDM
workers. HTML image tags were removed only from evaluator copies.

| Metric | Result |
|---|---:|
| Page text edit, lower is better | 0.053843 |
| Page text accuracy | 94.6157% |
| Page CDM | 92.1385% |
| Page TEDS | 83.8087% |
| Reading-order edit | 0.145533 |
| Overall | 90.1876% |

No page-match, TEDS, or CDM timeout, exception, or error occurred. The prior
known-good full-accuracy result was 90.1878%, a 0.0002-point difference.

## Memory mechanism

The run keeps the process-isolated execution design. It does not merge or
remove CPU workers:

- one dedicated layout process;
- four crop processes;
- eight resize threads in each crop process;
- four vision executor lanes;
- one bounded NPU decode owner.

The layout owner, crop workers, and main process use this jemalloc policy:

```text
narenas:2,background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000
```

The process launcher applies the main-process policy before Python imports
PyTorch or CANN. Child policies are scoped only while each process is spawned.
TBE compiler children are deinitialized after all serving graphs are warm.

The memory changes are host-only. They do not change an NPU operator, tensor
shape, graph, weight format, or numerical path. They should therefore apply on
310P when its Python process uses jemalloc. A runtime without jemalloc ignores
`MALLOC_CONF`; that case needs direct host validation and is not claimed here.

## Evidence

- `run_summary.json`: complete inference settings and stage timings.
- `process_tree_memory.json`: 50 ms process-tree PSS/RSS sampling.
- `writer_process_tree_memory.json`: deferred writer PSS/RSS sampling.
- `deferred_write_summary.json`: all 1,651 pages materialized.
- `transform_summary.json`: 1,545 image tags removed from evaluator copies.
- `full_eval_summary.json`: frozen OmniDocBench metrics and failure counts.

