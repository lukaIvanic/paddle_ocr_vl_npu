# MinerU optimization rung 1: asynchronous request H2D

## Result

Rejected. Both variants preserved exact content-list output but made the same
32-page workload slower. The source path was removed after measurement.

| Variant | Commit | Measured wall | Pages/s | Delta vs baseline | Content-list parity |
|---|---:|---:|---:|---:|---|
| Existing synchronous H2D | `8fa7e3e` | 41.129 s | 0.7790 | baseline | exact |
| Pinned async H2D | `10b63d1` | 45.814 s | 0.6985 | +11.4% wall | exact |
| Pageable async H2D | `8c08522` | 45.762 s | 0.7000 | +11.3% wall | exact |

All lanes used the first 32 OmniDocBench pages, B32/KV4096, the same warm
vision, text, and decode graph caches, PromptFA vision prefill, packed text
prefill, IncreFA decode, NZ decode weights, and the two-page warmup.

The pageable variant changed the measured generation stages as follows:

| Stage | Existing | Pageable async | Change |
|---|---:|---:|---:|
| Generation wall | 35.181 s | 39.742 s | +4.562 s |
| Prefill wall | 18.803 s | 23.516 s | +4.714 s |
| Decode wall | 10.260 s | 10.472 s | +0.212 s |
| CPU preparation service | 11.263 s | 15.717 s | +4.455 s |

The transfer stream competed with the active prefill path. The prefill
regression was larger than any hidden host-copy time. Per-request pinning also
added allocator work and did not help. Do not carry this path into later rungs.
