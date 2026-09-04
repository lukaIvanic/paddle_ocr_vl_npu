# Production-locked B/Q anchor benchmark

Four independent processes benchmarked the easy anchor lanes on one otherwise
idle Ascend 910B2 (physical NPU 6). Each process loaded the model independently,
then ran 10 real-input warmup calls and 50 measured calls. There was no separate
synthetic graph-build phase.

Source commit: `ab925dd51fbe73a8ad6f2c9b7fc203eeeea2760f`

| Lane | Median call | P95 call | Calls/s | Physical positions/s |
|---|---:|---:|---:|---:|
| B8Q1 | 1.094 ms | 1.101 ms | 908.5 | 7,268.3 |
| B1Q8 | 1.338 ms | 1.351 ms | 743.5 | 5,947.9 |
| B16Q1 | 1.357 ms | 1.365 ms | 735.5 | 11,768.4 |
| B1Q16 | 1.914 ms | 1.919 ms | 520.9 | 8,334.5 |

The first warmup in each fresh process took 13.1-16.8 seconds because it loaded
or compiled the real graph. The other nine warmups were steady-state calls.

No lane emitted a warning. The Q1 lanes used production IncreFA with the locked
complete-layer-prefetch and RoPE-lookup preset. The Q8/Q16 lanes used the
production manual grouped-attention verifier with the locked speculative
prefetch and multimodal-RoPE preset. All lanes used the frequency-selected
16,384-row LM head and native token-ID remapping.

These are independent forward-call anchors. They do not yet measure a mixed
draft-plus-verifier schedule or end-to-end OCR latency.
