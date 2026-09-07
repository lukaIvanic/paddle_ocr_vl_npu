# Live lower-cap smoke before the 310P handoff

2026-09-07, inference source `8528414f`, one 910B2, physical NPU 4.
Live PP-DocLayoutV3 (checkpoint FP32, eager, graph capture off) + custom MinerU
FP16, PSE-sentinel decode, unchanged production recognition settings/caches,
`processor_max_pixels=602112` (3,072 raw vision tokens).

Two validation passes:

| Pass | Pages | Live layout calls | Recognition requests | Pipeline wall | Setup |
|---|---:|---:|---:|---:|---:|
| First two dataset pages (root files) | 2 | 2 | 25 | 12.004 s | 24.617 s |
| Known affected text/table pages (`affected/`) | 2 | 2 | 20 | 13.215 s | 23.972 s |

Both exit 0 with no failed/skipped pages. Neither uses saved layouts/crop
replay; both load the detector and run it on the same NPU as recognition.

The first pass's largest crop was only 1,280 tokens, so a second pass was added
to exercise the cap on known oversized crops. This second pass is the one
specified in the 310P handoff:

- `page-573c437e-c309-4483-a038-ef2f440b104a.png`, table block 3: 4,032 tokens
  in the higher-cap baseline → **2,976** here; EOS, 868 raw output characters.
  Output begins with a structured table of spline strategies and equations.
- `page-9bcba6da-bdb0-4403-97cb-8874898ac8ab.png`, text block 5: 3,456 baseline
  tokens → **2,904** here; EOS, 649 raw output characters. Output begins with
  Chinese mathematical text and equations.

All recognition traces satisfy raw image-token count ≤ 3,072. These are NPU
integration smoke checks, not independent accuracy evaluations. Full-corpus
lower-cap quality evidence remains `../crop_cap3072_live_910b/`; the lower-cap
full-live 910B pg/s has not been measured.

Full logs, layout metadata, predictions and token IDs remain on the 910B at:

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/live_cap3072_handoff_smoke_8528414f
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/live_cap3072_binding_smoke_8528414f
```

Exact commands, device identity, summaries and exit files are retained here.
No graph cache was reset and no production defaults were changed.
