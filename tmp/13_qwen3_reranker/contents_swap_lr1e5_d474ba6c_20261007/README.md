# Contents-swapped Qwen3 distillation: peak LR 1e-5 — 2026-10-07

This matched follow-up changes the peak learning rate from 1e-6 to 1e-5. Both arms start from the released Qwen3-Reranker-0.6B weights with fresh optimizer state, using the same 1,600 training queries in the same order and the same frozen query-first Qwen3-Reranker-4B targets. The student sees `<Query>: document content` followed by `<Document>: query content`; the original instruction, system prefix and assistant suffix are unchanged.

## Endpoint comparison

NDCG@10 is reported as a percentage, averaged equally across tasks. The combined panel has 10 fixed queries per task (six frequent plus four reserved) across 10 English and eight Chinese retrieval tasks. These are diagnostic samples, not full-suite scores.

| Model | English NDCG@10 | Chinese NDCG@10 | Held-out teacher agreement | Teacher margin MSE |
|---|---:|---:|---:|---:|
| Original query-first 0.6B | 71.179 | 73.225 | 81.859% | 6.3827 |
| Untrained contents swap | 55.943 | 64.957 | 70.241% | 15.0516 |
| Contents swap, peak 1e-6, 50 updates | 56.690 | 66.285 | 75.364% | 8.4936 |
| Contents swap, peak 1e-5, 50 updates | 62.502 | 66.722 | 80.627% | 4.7424 |

The 1e-5 arm changes the combined English score by +6.559 points and Chinese by +1.764 points relative to the untrained swap. Its final differences from the original query-first baseline are -8.677 English and -6.503 Chinese points.
Held-out teacher agreement moves from 70.241% to 80.627%, while margin MSE changes from 15.0516 to 4.7424 (68.49% reduction). Teacher agreement measures learning toward the teacher; human-judgment NDCG measures sampled benchmark quality.

## Frequent evaluation trajectory

Each row uses six fixed queries per task, with a separate 64-query mixture validation set for teacher agreement. Evaluation points and candidate lists are identical between the arms.

| Updates | LR 1e-6 English | LR 1e-5 English | LR 1e-6 Chinese | LR 1e-5 Chinese | LR 1e-6 agreement | LR 1e-5 agreement | LR 1e-6 margin MSE | LR 1e-5 margin MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.626 | 54.626 | 64.088 | 64.088 | 70.241% | 70.241% | 15.0516 | 15.0516 |
| 1 | 53.755 | 54.827 | 64.147 | 64.230 | 70.241% | 70.577% | 15.0801 | 14.5422 |
| 3 | 53.874 | 54.894 | 64.028 | 64.286 | 70.269% | 72.340% | 14.9417 | 11.2165 |
| 10 | 55.006 | 59.248 | 64.249 | 68.138 | 70.969% | 77.632% | 13.1239 | 7.0778 |
| 25 | 54.232 | 59.353 | 65.169 | 66.731 | 74.160% | 80.179% | 9.5498 | 5.3181 |
| 50 | 55.358 | 59.040 | 65.575 | 66.137 | 75.364% | 80.627% | 8.4936 | 4.7424 |

## Frequent panel: six queries/task

| Task | Original query-first | Untrained swap | Trained swap 1e-6 | Trained swap 1e-5 | 1e-5 Δ versus original | 4B teacher |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 81.546 | 84.360 | 78.209 | 87.698 | +6.151 | 93.849 |
| CQADupstackGamingRetrieval | 63.933 | 64.116 | 63.933 | 62.048 | -1.886 | 66.116 |
| CQADupstackUnixRetrieval | 45.556 | 35.130 | 32.596 | 30.537 | -15.019 | 59.093 |
| ClimateFEVERHardNegatives | 57.062 | 16.737 | 15.942 | 21.409 | -35.653 | 62.505 |
| FEVERHardNegatives | 100.000 | 81.546 | 79.364 | 85.515 | -14.485 | 100.000 |
| FiQA2018 | 65.908 | 44.327 | 50.184 | 49.205 | -16.702 | 56.433 |
| HotpotQAHardNegatives | 92.214 | 57.412 | 57.725 | 73.501 | -18.713 | 96.616 |
| SCIDOCS | 24.124 | 17.924 | 21.946 | 18.378 | -5.746 | 23.234 |
| TRECCOVID | 89.901 | 86.676 | 88.487 | 87.293 | -2.609 | 96.291 |
| Touche2020Retrieval.v3 | 76.219 | 58.033 | 65.194 | 74.814 | -1.404 | 75.844 |
| EcomRetrieval | 63.889 | 34.120 | 35.275 | 41.691 | -22.198 | 63.350 |
| VideoRetrieval | 100.000 | 85.515 | 91.667 | 91.667 | -8.333 | 100.000 |
| MedicalRetrieval | 50.000 | 18.849 | 18.849 | 16.667 | -33.333 | 43.849 |
| MMarcoRetrieval | 83.333 | 72.900 | 73.845 | 78.540 | -4.793 | 89.781 |
| CmedqaRetrieval | 41.404 | 50.980 | 50.715 | 42.298 | +0.894 | 41.975 |
| DuRetrieval | 94.376 | 80.248 | 81.978 | 79.119 | -15.257 | 93.752 |
| CovidRetrieval | 93.849 | 85.515 | 87.698 | 87.698 | -6.151 | 100.000 |
| T2Retrieval | 88.336 | 84.577 | 84.577 | 91.418 | +3.082 | 93.369 |

| Model | English mean | Chinese mean |
|---|---:|---:|
| Original query-first | 69.646 | 76.898 |
| Untrained swap | 54.626 | 64.088 |
| Swap 1e-6 | 55.358 | 65.575 |
| Swap 1e-5 | 59.040 | 66.137 |

## Reserved endpoint panel: four queries/task

| Task | Original query-first | Untrained swap | Trained swap 1e-6 | Trained swap 1e-5 | 1e-5 Δ versus original | 4B teacher |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 100.000 | 85.767 | 87.500 | 87.500 | -12.500 | 100.000 |
| CQADupstackGamingRetrieval | 73.302 | 81.595 | 81.595 | 72.659 | -0.643 | 77.566 |
| CQADupstackUnixRetrieval | 61.562 | 62.704 | 62.704 | 70.789 | +9.227 | 64.273 |
| ClimateFEVERHardNegatives | 47.636 | 11.267 | 10.993 | 16.291 | -31.345 | 36.688 |
| FEVERHardNegatives | 100.000 | 64.072 | 63.773 | 90.773 | -9.227 | 100.000 |
| FiQA2018 | 66.689 | 48.215 | 48.662 | 59.463 | -7.227 | 84.463 |
| HotpotQAHardNegatives | 100.000 | 85.766 | 84.033 | 92.336 | -7.664 | 100.000 |
| SCIDOCS | 20.459 | 22.910 | 22.170 | 31.788 | +11.329 | 32.820 |
| TRECCOVID | 95.269 | 96.720 | 97.371 | 96.456 | +1.187 | 97.396 |
| Touche2020Retrieval.v3 | 69.872 | 20.174 | 28.091 | 58.907 | -10.965 | 76.999 |
| EcomRetrieval | 40.560 | 21.405 | 22.171 | 32.172 | -8.388 | 37.307 |
| VideoRetrieval | 87.500 | 72.171 | 72.171 | 75.445 | -12.055 | 100.000 |
| MedicalRetrieval | 50.000 | 50.000 | 50.000 | 50.000 | +0.000 | 50.000 |
| MMarcoRetrieval | 50.000 | 50.000 | 50.000 | 50.000 | +0.000 | 65.859 |
| CmedqaRetrieval | 43.875 | 63.594 | 66.273 | 55.110 | +11.235 | 40.551 |
| DuRetrieval | 92.419 | 90.831 | 93.223 | 86.569 | -5.850 | 98.569 |
| CovidRetrieval | 87.500 | 87.500 | 90.773 | 100.000 | +12.500 | 100.000 |
| T2Retrieval | 89.868 | 94.591 | 94.193 | 91.494 | +1.626 | 94.577 |

| Model | English mean | Chinese mean |
|---|---:|---:|
| Original query-first | 73.479 | 67.715 |
| Untrained swap | 57.919 | 66.262 |
| Swap 1e-6 | 58.689 | 67.351 |
| Swap 1e-5 | 67.696 | 67.599 |

## Recipe, execution and validation

- Ascend 910B2, physical NPU 2. Executed source `d474ba6cb3e196e87b7a1a0246794471c1ad889b`; published branch `codex/qwen-contents-swapped-distill`.
- Full-model NpuFusedAdamW with fresh state: betas 0.9/0.999, epsilon 1e-8, zero weight decay, gradient norm clipping 1.0. FP32 parameters, BF16 autocast and owned Ascend fusion-attention model.
- Peak LR 1e-5. Updates 1–5 use 2e-6, 4e-6, 6e-6, 8e-6, 1e-5; subsequent updates use the same linear decay as the 1e-6 control. The final nonzero LR is 1e-5/45. No optimizer state is resumed from the previous run.
- 50 updates, 32 queries × eight candidates = 256 pairs/update. Mean Margin-MSE over all 28 unordered within-query candidate pairs, averaged equally across queries. Targets are raw teacher yes-minus-no score differences.
- Deterministic score-gradient replay. At most four inputs and 8,192 padded positions per training microbatch; left padding to the longest input rounded up to 128. The total prompt limit is 8,192 tokens including preserved prefix/suffix.
- Train sources: 800 MIRACL_zh, 267 NQ, 267 MIRACL_en, 266 SQuAD. Held-out teacher validation: 64 queries × eight candidates. Frequent human-judgment panel: 108 queries ×100 candidates; reserved panel: 72 queries ×100 candidates.
- MTEB 1.38.9 pinned task definitions and original Qwen3-Embedding-0.6B top-100 candidates, without injected positives. Generic train/mixture-validation instructions and fixed task-specific benchmark instructions are inherited unchanged.
- Source-family and exact-text exclusions, benchmark query blocking, selected candidate blocking and training/validation separation are inherited from the previous pilot. This four-source mixture does not cover every retrieval task type or constitute exhaustive semantic decontamination.
- Matching data, released model and teacher file hashes, prompt token hashes, recipe and candidate counts were verified against the 1e-6 arm. Untrained baseline candidate scores and metrics are identical. Both input orders have zero truncated pairs in every section.
- Swapped-input HF/owned backward, replay and ordinary/fused optimizer control passed. The unchanged optimizer control uses LR 1e-6. Maximum checked ordinary/fused optimizer parameter difference: 0.0; maximum replay/joint gradient relative L2: 0.211%. The fused forward/backward path has BF16 numerical differences; this control does not establish bitwise parity for all tensors.

## Runtime

Training averaged 18.22 seconds/update (range 17.62–20.78); the 50 updates took 15.18 minutes. The training process including input preparation, evaluations and checkpoint saves took 26.80 minutes.
Frequent benchmark plus teacher validation averaged 73.26 seconds. The extra reserved panel took 46.02 seconds at the endpoint. Checkpoint saves averaged 17.24 seconds; peak allocated NPU memory was 22.04 GiB.

## Evidence and saved states

Dataset SHA256 `0fc73a4a5df7f755ea0cdbc58afdee1e87141d07f9575f01ee720588bf86899e`; teacher SHA256 `7b130b26a8ff74b3f2c8a788ca7bfb05114bbf593842fec41d8ec873dbe94cfc`.

- `warmup_linear/result.json`: all per-query metrics, candidate scores, teacher agreement, original query-first baseline, both token layouts and training/evaluation/checkpoint timings.
- `reference/swapped_lr1e6.json`: completed matched lower-LR run; `reference/query_first_warmup.json`: original query-first control; `reference/teacher.json`: frozen teacher targets; `reference/manifest.json`: source/split and panel provenance.
- `control/`: real-input implementation check; `command.txt` and `run.log` record actual execution. Recorded command text has terminal trailing spaces normalized.
- `touche_changes.json`: original query-first, untrained swap and both trained arms, with top-ten document IDs, raw scores, qrels and character lengths.
- Every nonzero evaluated checkpoint (1, 3, 10, 25, 50) contains model and optimizer states, scheduler description, CPU/NPU RNG states, group order and data/teacher hashes. Large weights remain on the server:

- `/workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_d474ba6c/warmup_linear/checkpoint_001.pt` (7,770,900,082 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_d474ba6c/warmup_linear/checkpoint_003.pt` (7,770,900,082 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_d474ba6c/warmup_linear/checkpoint_010.pt` (7,770,900,082 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_d474ba6c/warmup_linear/checkpoint_025.pt` (7,770,900,082 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_lr1e5_d474ba6c/warmup_linear/checkpoint_050.pt` (7,770,900,082 bytes)

## Interpretation

The higher learning rate produced better endpoint recovery than 1e-6 in both sampled suite means. Accuracy remains substantially below the original query-first baseline.

Teacher agreement continues to improve between updates 10 and 25 while English NDCG is nearly flat and Chinese NDCG falls on the frequent panel. This separates progress on the held-out mixture distillation objective from recovery across external retrieval tasks. The endpoint and per-task tables show the subsequent changes; gains are not uniform across the evaluation distribution.

## Scope of conclusions

Assess recovery relative to original query-first accuracy, not just the damaged untrained swap. The small diagnostic panels are useful for observing large changes; individual-task gains can reflect a few queries. This matched comparison tests the learning-rate change under the contents-swapped format. It does not test another document-first prompt, longer training, full-suite parity or cached-document serving performance.
