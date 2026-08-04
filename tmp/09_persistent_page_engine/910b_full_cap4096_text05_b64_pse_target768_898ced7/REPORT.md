# 910B full OmniDocBench: KV4096, cap4096, text scale 0.5, B64 PSE

Source commit at execution: `898ced73e4a13208e7d2d6af757daaa0d5f988d1`.
Physical device: Ascend 910B2, physical NPU 5.

## Configuration

- 1651 OmniDocBench v1.6 pages; all pages prepared before recognition.
- Layout: NPU eager, eight page-preparation workers.
- Decode: batch 64, KV4096, TorchAir `combined_apply_pse_sentinel`.
- Vision: TorchAir PromptFA, 4352-padded MLP, FRACTAL_NZ weights, greedy packing target 768.
- Vision buckets: 256,384,512,640,768,1408,1920,2048,2944,4096.
- Pixel policy: min_pixels 28224; global max_pixels 802816 (4096 raw vision tokens); text crop scale 0.5.
- Text packing: 128,256,384,512,768,1024; max 32 members.
- Repetition guard enabled.

## Runtime

- Setup: 243.230 s (excluded from pipeline time; graph materialization/cache setup).
- Pipeline: 800.495 s, 2.06248 pages/s, 0.48485 s/page.
- Completed: 1651/1651 pages, 30557/30557 recognition requests.
- Vision prefill: 224.227 device-s; 10,666,368 physical tokens at 47,569 tok/s; 9,542,424 useful tokens at 42,557 tok/s.
- Text prefill: 77.752 device-s; 3,373,824 physical tokens at 43,392 tok/s; 2,782,847 useful tokens at 35,791 tok/s.
- Decode: 169.931 device-s; 1,782,016 raw slots at 10,487 tok/s; 1,607,069 effective tokens at 9,457 tok/s.
- Decode stops: 30,485 EOS, 60 repetition guard, 12 KV full.
- Vision useful-token fraction: 89.46%; text: 82.48%; decode: 90.18%.

Layout `stage_s` values in `run_summary.json` are summed work across concurrent workers, not elapsed critical-path wall time. `page_total_s=930.547` must not be reported as layout wall time. The first completed page occurred at 162.677 s after the pipeline timer began, which is only an upper bound on layout-first completion plus the first OCR result.

## Evaluation

Official evaluator wall: 255 s, exit 0.

- Text block Edit distance: 0.0507017.
- Display-formula Edit distance: 0.0903191.
- Table TEDS (official page aggregate): 0.944425.
- Table TEDS (per-table sample aggregate, diagnostic): 0.930515.
- Table structure-only TEDS (sample): 0.956921.
- Table Edit distance: 0.0541462.
- Reading-order Edit distance: 0.140596.
- Matching: 1651 pages; four process-isolated fallback pages; no quick-match timeouts.
- TEDS: 665/665, zero errors and zero timeouts.

Direct native CDM: 2352/2352 samples across 313 formula-bearing pages in 88.6545 s with 96 workers, zero errors/timeouts. Official page aggregate: 0.974080; per-formula sample aggregate: 0.970524.

Official OmniDocBench overall `((1-text_edit)*100 + page_CDM*100 + page_TEDS*100)/3`: **95.5935**. Reading-order Edit distance is reported separately and is not part of Overall.

## Comparison

- Old quality-first KV4096/default-pixels run: 1055.523 s, 1.564 pages/s, page TEDS 0.943450.
- Aggressive KV2048/cap2048/text-scale-0.5 run: 866.768 s, 1.905 pages/s, page TEDS 0.921781.
- Later B64/PSE execution of that KV2048/cap2048 regime: 717.705 s, 2.300 pages/s; its official edit-distance/TEDS metrics were essentially the same, but that run did not record a direct CDM artifact.
- This run: 800.495 s, 2.062 pages/s, page TEDS 0.944425, official Overall 95.5935.

This run is 24.2% less wall time / 31.9% higher pages/s than the old quality-first run while retaining most of its quality. Relative to the aggressive KV2048 result, it restores table TEDS from 0.913860 to 0.930515 and CDM from 0.964700 to 0.970524.
