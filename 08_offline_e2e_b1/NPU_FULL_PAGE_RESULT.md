# 910B full-page validation

Observed on 2026-07-16 from commit `bc59615`, using one logical `npu:0` on a
910B server. Both models were local:

```text
PP-DocLayoutV3: /workspace/models/PP-DocLayoutV3_safetensors
PaddleOCR-VL:   /workspace/models/PaddleOCR-VL-1.6
```

The input was the 2000x1500 OmniDocBench page
`PPT_The Right Moves_page_024.png`. The run used real layout inference,
`cache_length=2048`, `max_new_tokens=768`, eager vision/text prefill, TorchAir
compiled B=1 decode, and no region cap.

## Result

- PP-DocLayoutV3 produced five reading-ordered regions: one
  `paragraph_title`, three `text`, and one `number`.
- All five regions ran sequentially and stopped at EOS.
- The recognized strings were plausible and followed the page's reading order.
- Page inference took 1.872002 s, excluding model setup and diagnostic artifact
  writes. Annotated image/text artifact writes added 0.225688 s, for 2.097715 s
  including artifacts.
- Real layout took 0.703594 s total: 0.058053 s processor work, 0.001339 s H2D,
  0.509972 s synchronized model inference, and 0.134042 s postprocessing.
- Strict sequential recognition took 1.136537 s.
- Compiled decode consumed 0.208081 s for 76 effective post-first-token tokens
  including EOS: 365.242 tok/s. The page produced 81 tokens including each
  prefill-produced first token and EOS, giving 43.269 E2E output tok/s over the
  page-inference wall time.

Per-region decode rates were 184.0, 390.3, 413.9, 392.5, and 268.8 tok/s. These
are short, heterogeneous B=1 sequences with per-request EOS handling and a
2048-token static cache, so the aggregate is not directly comparable to a
fixed-length decode-only microbenchmark. The longest real region generated 42
tokens total and sustained 413.9 effective decode tok/s.

The compile cache was warm for this run: compile-wrapper construction took
0.736 s and the first compiled call took 0.547 s. Total process setup was
31.345 s. The result JSON and annotated page remain in the Blue Zone checkout:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page/run.json
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page/pages/
```

This validates the Experiment 08 execution architecture and timing surfaces. It
does not establish complete PaddleX output parity: overlap filtering, box merge
policy, adjacent-block merging, formula margin crop, table figure-token
substitution, and structured Markdown assembly remain explicit follow-on work.
