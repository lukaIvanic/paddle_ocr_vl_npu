# Qwen3 query-first Margin-MSE distillation pilot — 2026-10-07

Both 50-update arms completed on an Ascend 910B2. This is a query-first pilot; it does not validate document-first inference or cached-document scoring. The JSONs, command records and logs beside this file are the numerical evidence.

## Measurements

NDCG@10 below is a percentage. English and Chinese values are unweighted task means on six fixed queries per task (10 English and 8 Chinese tasks), with 100 original embedding candidates per query. These are diagnostic samples, not full-suite benchmark scores. Teacher agreement uses 64 separate mixture queries with eight candidates each.

| Arm | Updates | English NDCG@10 | Chinese NDCG@10 | Teacher pairwise agreement | Teacher margin MSE | Top-3 overlap | Touché NDCG@10 (6 queries) |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 0 | 69.646 | 76.898 | 81.859% | 6.3827 | 76.042% | 76.219 |
| constant | 1 | 69.562 | 77.238 | 82.111% | 6.2952 | 76.562% | 75.489 |
| constant | 3 | 69.868 | 77.546 | 82.279% | 6.0472 | 77.083% | 75.489 |
| constant | 10 | 69.841 | 77.298 | 81.999% | 5.3952 | 76.042% | 75.466 |
| constant | 25 | 71.008 | 77.573 | 83.287% | 4.5882 | 77.083% | 78.716 |
| constant | 50 | 70.566 | 76.921 | 84.295% | 4.1122 | 77.083% | 78.111 |
| warmup_linear | 0 | 69.646 | 76.898 | 81.859% | 6.3827 | 76.042% | 76.219 |
| warmup_linear | 1 | 69.401 | 76.839 | 81.971% | 6.3307 | 76.562% | 75.321 |
| warmup_linear | 3 | 70.183 | 77.132 | 81.943% | 6.3118 | 76.562% | 77.058 |
| warmup_linear | 10 | 69.859 | 77.511 | 82.223% | 5.4316 | 76.562% | 76.400 |
| warmup_linear | 25 | 70.980 | 77.371 | 82.363% | 4.6502 | 76.562% | 78.077 |
| warmup_linear | 50 | 69.931 | 77.231 | 83.175% | 4.4083 | 76.042% | 77.191 |

Held-out teacher agreement by language (32 queries each):

| Arm | Updates | Language | Pairwise agreement | Margin MSE |
|---|---:|---|---:|---:|
| constant | 0 | en | 80.223% | 4.7840 |
| constant | 0 | zh | 83.502% | 7.9813 |
| constant | 50 | en | 82.514% | 3.5385 |
| constant | 50 | zh | 86.083% | 4.6859 |
| warmup_linear | 0 | en | 80.223% | 4.7840 |
| warmup_linear | 0 | zh | 83.502% | 7.9813 |
| warmup_linear | 50 | en | 81.453% | 3.7832 |
| warmup_linear | 50 | zh | 84.905% | 5.0333 |

## Frequent panel: baseline and endpoints by task

| Task | Original 0.6B | 4B teacher | Constant endpoint | Δ constant | Warmup/decay endpoint | Δ scheduled |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 81.546 | 93.849 | 81.546 | +0.000 | 81.546 | +0.000 |
| CQADupstackGamingRetrieval | 63.933 | 66.116 | 72.267 | +8.333 | 66.116 | +2.182 |
| CQADupstackUnixRetrieval | 45.556 | 59.093 | 44.218 | -1.338 | 44.218 | -1.338 |
| ClimateFEVERHardNegatives | 57.062 | 62.505 | 59.836 | +2.774 | 59.949 | +2.887 |
| FEVERHardNegatives | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| FiQA2018 | 65.908 | 56.433 | 63.021 | -2.887 | 63.021 | -2.887 |
| HotpotQAHardNegatives | 92.214 | 96.616 | 92.214 | +0.000 | 92.214 | +0.000 |
| SCIDOCS | 24.124 | 23.234 | 23.894 | -0.230 | 23.938 | -0.186 |
| TRECCOVID | 89.901 | 96.291 | 90.555 | +0.654 | 91.117 | +1.215 |
| Touche2020Retrieval.v3 | 76.219 | 75.844 | 78.111 | +1.892 | 77.191 | +0.972 |
| EcomRetrieval | 63.889 | 63.350 | 66.071 | +2.182 | 66.071 | +2.182 |
| VideoRetrieval | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| MedicalRetrieval | 50.000 | 43.849 | 50.000 | +0.000 | 50.000 | +0.000 |
| MMarcoRetrieval | 83.333 | 89.781 | 82.178 | -1.155 | 82.178 | -1.155 |
| CmedqaRetrieval | 41.404 | 41.975 | 37.438 | -3.966 | 42.256 | +0.852 |
| DuRetrieval | 94.376 | 93.752 | 94.549 | +0.173 | 94.376 | +0.000 |
| CovidRetrieval | 93.849 | 100.000 | 93.849 | +0.000 | 93.849 | +0.000 |
| T2Retrieval | 88.336 | 93.369 | 91.280 | +2.944 | 89.118 | +0.781 |

Suite means:

| Model | English | Chinese |
|---|---:|---:|
| Original 0.6B | 69.646 | 76.898 |
| 4B teacher | 72.998 | 78.260 |
| Constant endpoint | 70.566 | 76.921 |
| Warmup/decay endpoint | 69.931 | 77.231 |

## Reserved endpoint panel: baseline and endpoints by task

| Task | Original 0.6B | 4B teacher | Constant endpoint | Δ constant | Warmup/decay endpoint | Δ scheduled |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| CQADupstackGamingRetrieval | 73.302 | 77.566 | 73.302 | +0.000 | 73.302 | +0.000 |
| CQADupstackUnixRetrieval | 61.562 | 64.273 | 61.562 | +0.000 | 61.562 | +0.000 |
| ClimateFEVERHardNegatives | 47.636 | 36.688 | 46.819 | -0.817 | 51.115 | +3.479 |
| FEVERHardNegatives | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| FiQA2018 | 66.689 | 84.463 | 59.463 | -7.227 | 66.689 | +0.000 |
| HotpotQAHardNegatives | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| SCIDOCS | 20.459 | 32.820 | 20.459 | +0.000 | 20.459 | +0.000 |
| TRECCOVID | 95.269 | 97.396 | 95.352 | +0.083 | 95.268 | -0.001 |
| Touche2020Retrieval.v3 | 69.872 | 76.999 | 73.777 | +3.905 | 73.863 | +3.991 |
| EcomRetrieval | 40.560 | 37.307 | 40.560 | +0.000 | 33.333 | -7.227 |
| VideoRetrieval | 87.500 | 100.000 | 87.500 | +0.000 | 87.500 | +0.000 |
| MedicalRetrieval | 50.000 | 50.000 | 50.000 | +0.000 | 50.000 | +0.000 |
| MMarcoRetrieval | 50.000 | 65.859 | 50.000 | +0.000 | 50.000 | +0.000 |
| CmedqaRetrieval | 43.875 | 40.551 | 53.101 | +9.227 | 53.101 | +9.227 |
| DuRetrieval | 92.419 | 98.569 | 92.541 | +0.122 | 92.541 | +0.122 |
| CovidRetrieval | 87.500 | 100.000 | 90.773 | +3.273 | 87.500 | +0.000 |
| T2Retrieval | 89.868 | 94.577 | 96.693 | +6.825 | 94.541 | +4.673 |

Suite means:

| Model | English | Chinese |
|---|---:|---:|
| Original 0.6B | 73.479 | 67.715 |
| 4B teacher | 77.021 | 73.358 |
| Constant endpoint | 73.073 | 70.146 |
| Warmup/decay endpoint | 74.226 | 68.565 |

Combined baseline/endpoint means (all 10 preselected queries per task):

| Model | English | Chinese |
|---|---:|---:|
| Original 0.6B | 71.179 | 73.225 |
| 4B teacher | 74.607 | 76.299 |
| Constant endpoint | 71.569 | 74.211 |
| Warmup/decay endpoint | 71.649 | 73.764 |

This combines the six-query and four-query panels within each task before averaging tasks equally; it requires no additional model evaluation. Both separate panels remain reported above.

## Execution cost

| Arm | Training seconds/update (mean; min–max) | 50 training updates | Frequent evaluation | Extra endpoint panel | Checkpoint save | Total process time | Peak allocated memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 18.16; 17.57–20.50 | 15.13 min | 73.55 s | 45.43 s | 16.89 s | 26.32 min | 22.04 GiB |
| warmup_linear | 17.64; 17.22–19.65 | 14.70 min | 72.54 s | 44.73 s | 16.53 s | 25.72 min | 22.04 GiB |

Frozen-teacher scoring took 9.13 minutes for 31,312 pairs; teacher-process total including setup was 10.72 minutes.

Evaluation runs at updates 0, 1, 3, 10, 25 and 50. The additional four-query-per-task panel runs only at 0 and 50. Checkpoint saving runs before each nonzero evaluated update. Training timing includes the no-grad scoring pass, replayed forwards/backwards, clipping, optimizer step and final synchronization; it excludes checkpointing and evaluation.

## Recipe and data

- Student: released Qwen3-Reranker-0.6B, all parameters trainable. Frozen teacher: released Qwen3-Reranker-4B. Both use the same official query-first prompt structure, per-example instruction, token IDs and truncation policy. No input-order changes were made.
- Objective: raw yes-minus-no score differences. For each eight-document query, average squared student-minus-teacher margin error over all 28 unordered document pairs; average query losses equally over 32 queries per update. Binary relevance labels do not enter the training objective.
- Each update consumes 32 queries × 8 documents = 256 inputs. All 50 updates consume 1,600 query groups once, in the same shuffled order in both arms. Microbatches are at most four sequences and 8,192 padded token positions, left-padded to their longest sequence rounded up to 128. A no-grad pass computes the eight score gradients; replayed microbatches accumulate parameter gradients before a single optimizer update.
- FP32 parameters, BF16 autocast, owned PyTorch model with Ascend fusion attention; no TorchAir compile. Activation memory is bounded by the explicit score-gradient replay described above. Evaluation uses up to 16 sequences with a 16,384 padded-position budget. Total prompt limit 8,192 tokens, retaining prefix and suffix. No selected teacher/student input was truncated.
- Fresh NpuFusedAdamW moments and step count; betas 0.9/0.999, epsilon 1e-8, weight decay 0, global gradient-norm clipping at 1.0. Constant LR 1e-6; scheduled LR rises over five updates to 1e-6 and linearly decays, with a nonzero final update. Integrated LR is 50e-6 for constant and 26e-6 for scheduled; this comparison does not isolate warmup from total update size.
- Training sources: 800 MIRACL_zh, 267 NQ, 267 MIRACL_en and 266 SQuAD queries. Held-out teacher validation: 32 English and 32 Chinese queries. This deliberately avoids the earlier NLI-heavy mixture, but these four passage-retrieval sources are not representative of every retrieval subtype.
- Candidate groups select one supplied positive and up to seven supplied negatives, supplementing if needed with split/source-specific lexical retrieval. The 4B teacher determines the relative training scores; it is a reference model, not human ground truth.
- Source-family exclusions, all target-suite normalized query texts, and candidate document texts from both sampled panels are excluded from training/mixture validation. Selected training/validation query and document texts are mutually disjoint. These are exact-text and family checks, not exhaustive semantic/corpus decontamination.
- Human evaluation uses pinned MTEB 1.38.9 task/dataset versions and saved Qwen3-Embedding-0.6B top-100 candidates, with no injected judged positives. NDCG@10 uses the full query qrels for ideal gain; unjudged retrieved documents receive zero gain. English and Chinese means weight tasks equally.

## Implementation control

The real unequal-length, left-padded NPU control passed before teacher scoring/training. HF eager and owned eager scores and loss agreed exactly on the four-example control. Fusion attention is numerically close rather than bitwise identical; checked gradient cosines exceed 0.997, and relative L2 differences are approximately 5–8%. Replay versus joint fusion gradients differ by about 0.17–0.23%. Replaying identical stored gradients through ordinary and fused AdamW gives identical checked parameter updates. This checks representative tensors on one batch, not every gradient or full-benchmark HF parity. See `control/control.json` for exact differences and thresholds.

## Touché judgments

| Panel | Original hole@10 | Constant endpoint hole@10 | Scheduled endpoint hole@10 |
|---|---:|---:|---:|
| benchmark | 13.333% | 11.667% | 11.667% |
| reserved_benchmark | 30.000% | 25.000% | 25.000% |

The frequent Touché panel has six queries and the reserved panel four. These results cannot be compared directly with the earlier 49-query Touché score. Candidate IDs, qrels and per-query metrics are retained in the manifest/results for inspecting promoted and demoted documents.

## Reproducibility and saved state

- Executed source commit: `d66b8e894674f9b472fdeb2de33c316e20b0723f` (the same commit for both arms).
- Container: `research_vllm_ascend_023_external_workspace`; host `liteserver-c001-4`. Stage-specific physical NPU IDs and exact invocations are in each `command.txt`.
- Environment: `{"chip": "Ascend 910B2", "host": "liteserver-c001-4", "physical_npu": "3", "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post2", "transformers": "5.5.4"}`.
- Dataset: `/workspace/results/qwen_margin_distill/data_fast/dataset.json.gz`; SHA256 `0fc73a4a5df7f755ea0cdbc58afdee1e87141d07f9575f01ee720588bf86899e`. See `preparation/manifest.json` for source revision, exclusions, panel IDs, candidate-file hashes and query/doc split metadata.
- Both arms have identical baseline benchmark, held-out validation and reserved-panel score arrays; both assert identical teacher/student token hashes. Released model-file hashes and teacher/control artifact hashes are in each result JSON.
- Evaluated checkpoints 1, 3, 10, 25 and 50 for each arm remain on the server. Each contains model and optimizer states, schedule configuration/completed step, CPU/NPU RNG states, dataset/teacher hashes and training order. Large `.pt` files were not copied into Git.

Checkpoint paths:

- `/workspace/results/qwen_margin_distill/npu/d66b8e89/constant/checkpoint_001.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/constant/checkpoint_003.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/constant/checkpoint_010.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/constant/checkpoint_025.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/constant/checkpoint_050.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/warmup_linear/checkpoint_001.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/warmup_linear/checkpoint_003.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/warmup_linear/checkpoint_010.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/warmup_linear/checkpoint_025.pt` (7,770,899,826 bytes)
- `/workspace/results/qwen_margin_distill/npu/d66b8e89/warmup_linear/checkpoint_050.pt` (7,770,899,826 bytes)

## Interpretation limits

The tables show numerical learning and retention on a short diagnostic run. With six/four queries per task, differences can be driven by one ranking change; they do not establish full MTEB/CMTEB parity or superiority. Both schedules and the 50-update endpoints were fixed before their results were observed.

This recipe changes the data mixture, objective, learning rate and effective batch relative to the earlier binary-label run. Its behavior does not identify which single change caused the difference. It also does not establish that a document-first student will retain these results.
