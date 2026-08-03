# 910B text-decode length-mode matrix

Commit `24acb27989bcc02a89813d284ad5c2747aa24b97`; Ascend 910B2; FP16; TorchAir; full 18-layer decoder + LM head + argmax; KV2048; profile positions 1024-1053; 3 warmups and 30 measured steps; all slots active.

| B | mode | mean ms | median ms | p95 ms | physical tok/s | vs normal | peak delta MiB | KV MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | normal masked GQA | 1.9912 | 1.9763 | 1.9800 | 8035.3 | +0.00% | 10.3 | 576.0 |
| 16 | static actual | 2.1184 | 2.1022 | 2.1076 | 7552.9 | -6.00% | 10.3 | 576.0 |
| 16 | PSE sentinel | 2.0521 | 2.0364 | 2.0412 | 7797.0 | -2.97% | 10.3 | 576.0 |
| 32 | normal masked GQA | 2.9847 | 2.9668 | 2.9746 | 10721.4 | +0.00% | 18.9 | 1152.0 |
| 32 | static actual | 3.0434 | 3.0262 | 3.0337 | 10514.5 | -1.93% | 18.9 | 1152.0 |
| 32 | PSE sentinel | 3.0571 | 3.0397 | 3.0480 | 10467.3 | -2.37% | 18.9 | 1152.0 |
| 64 | normal masked GQA | 4.1555 | 4.1384 | 4.1508 | 15401.2 | +0.00% | 38.6 | 2304.0 |
| 64 | static actual | 4.2941 | 4.3202 | 4.3386 | 14904.1 | -3.23% | 38.6 | 2304.0 |
| 64 | PSE sentinel | 4.2539 | 4.2344 | 4.2467 | 15045.1 | -2.31% | 38.6 | 2304.0 |
| 128 | normal masked GQA | 6.2948 | 6.2711 | 6.3034 | 20334.3 | +0.00% | 75.8 | 4608.0 |
| 128 | static actual | 6.1382 | 6.1134 | 6.1374 | 20853.2 | +2.55% | 75.8 | 4608.0 |
| 128 | PSE sentinel | 6.2771 | 6.2578 | 6.2736 | 20391.7 | +0.28% | 75.8 | 4608.0 |

All eight static-actual/PSE boundary gates passed at cache_position=1279 / effective_length=1280.

All lanes reported `decode_native_fallback`; the current 910B torch_npu runtime did not materialize FRACTAL_NZ weights.
