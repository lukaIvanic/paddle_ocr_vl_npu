# 910B table vision-token cap sweep

Commit: `d88d6ed`. Scope: **458** OmniDocBench v1.6 pages containing **665 GT tables**; the fixed layout pass produced **731 table recognition requests** per lane.

Only table requests were capped. Text, formula, and every other prompt retained the default 5120-token pixel ceiling. All four lanes were fresh, internally comparable runs on separate Ascend 910B2 NPUs.

| Nominal cap | Actual max | Table vision tokens | Token reduction | Page TEDS | Delta vs 5120 | Structure TEDS | Table Edit dist | Changed generations | Improved / same / worsened |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5120 | 5104 | 1,741,512 | 0.00% | 0.944513 | +0.0000 pp | 0.968790 | 0.054794 | 0/731 | 0 / 665 / 0 |
| 4096 | 4092 | 1,566,664 | 10.04% | 0.944393 | -0.0120 pp | 0.968762 | 0.054161 | 86/731 | 34 / 587 / 44 |
| 3072 | 3060 | 1,341,552 | 22.97% | 0.943644 | -0.0868 pp | 0.968650 | 0.055077 | 124/731 | 44 / 556 / 65 |
| 2048 | 2048 | 1,020,272 | 41.41% | 0.935920 | -0.8592 pp | 0.962949 | 0.062378 | 194/731 | 62 / 488 / 115 |

## Interpretation

- **4096 is effectively free:** 10.04% fewer table vision tokens for -0.0120 TEDS percentage points; table Edit distance slightly improves.
- **3072 is still nearly free:** 22.97% fewer table vision tokens for -0.0868 TEDS percentage points.
- **2048 is the first clear quality knee:** 41.41% fewer table vision tokens, but -0.8592 TEDS percentage points and table Edit distance rises from 0.05479 to 0.06238.
- The 2048 loss is not universal: 488/665 GT tables are unchanged, 62 improve, and 115 worsen; 19 lose at least 0.10 TEDS.
- Evaluator health: every lane evaluated 665/665 tables with zero TEDS errors/timeouts and zero page-match fallbacks.

## Runtime context

| Cap | Pipeline e2e | Pages/s | All-page vision-prefill device time |
|---:|---:|---:|---:|
| 5120 | 247.80s | 1.848 | 83.14s |
| 4096 | 245.87s | 1.863 | 80.99s |
| 3072 | 241.36s | 1.898 | 76.57s |
| 2048 | 233.80s | 1.959 | 69.45s |

The timing lanes ran concurrently, so use the within-lane device totals directionally. Accuracy is the primary result. The first launch also warmed two missing native vision graphs; `pipeline_e2e_s` excludes recognizer setup.

## Artifacts

Root: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_table_cap_sweep_d88d6ed`. Each `cap*/output/` contains predictions, trace, and run summary; each `cap*/evaluation/work/result/` contains official metric and per-table TEDS JSON.
