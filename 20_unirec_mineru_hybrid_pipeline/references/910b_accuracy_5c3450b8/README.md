# Full-corpus UniRec + MinerU accuracy on the 910B host

2026-09-11. Inference and evaluator launcher source: `5c3450b8`.
All 1,651 saved image predictions from the full experiment-20 run were scored.
No inference rerun, crop-level oracle mixing, omitted pages or removal of
length-stopped outputs. Original predictions were preserved and hash-checked.

| Page-aggregated metric | Percent |
|---|---:|
| Text accuracy (1 - page edit distance) | 95.028590 |
| Page Table TEDS | 93.593239 |
| Page Formula CDM | 96.662795 |
| Overall (mean of those three) | 95.094875 |
| Page table-structure TEDS | 96.206169 |

Reading-order page edit distance: 0.141184891. Formula page edit distance:
0.091627099. Sample-averaged TEDS/CDM are different metrics and are not used
as substitutes for the page-aggregated anchors above.

For historical context, the earlier UniRec+Paddle 910B result was 95.426424
overall: this result is lower by 0.331550 points. Text changes +0.016942 points,
tables -0.885287, formulas -0.126304. The largest aggregate difference is tables;
this does not isolate resolution, recognizer or scheduling effects by itself.

## Evaluation validity and speed

- Frozen evaluator commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`;
  unchanged executable source, verified TeX Live 2025/pdfTeX 1.40.28 and
  ImageMagick 7.1.1-47. Exact runtime check and command are retained here.
- Matching: all 1,651 pages, 48 process-isolated workers. Three page timeouts
  used the existing bounded recovery; cases are in `full_eval_summary.json`.
  No new recovery or scoring changes were introduced.
- TEDS: 665 matched samples, 32 workers, zero timeout/error/exception cases.
- Direct CDM: 2,352 samples on 313 pages, 64 workers, zero timeout/exception
  cases. Rendering workers used OMP/BLAS/ImageMagick thread limits of one.
- Evaluation wall: **357 seconds (5m57s)** including runtime verification,
  preparation, matching, TEDS, CDM, aggregation and hash checks. It does not
  count toward inference throughput. This exceeds the earlier five-minute
  aspiration; matching's difficult-page tail remains a contributor.
- Final source/prediction hash checks passed; exit code 0.

Only HTML image tags were removed from separate evaluation copies, exactly
as in the established UniRec/hybrid evaluator convention. See `preparation.json`
for ground-truth/source hashes. The complete transform manifest and per-page /
per-item results remain at:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/20_unirec_mineru_hybrid_pipeline/full1651_accuracy_5c3450b8/
```

The launcher's printed `UNIREC_FULL_EVAL` prefix is legacy; its recorded lane
is `unirec_mineru_910b_full1651_5c3450b8`, not an all-UniRec result.
