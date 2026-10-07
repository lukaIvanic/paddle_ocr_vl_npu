# Contents-swapped Qwen3-Reranker distillation — 2026-10-07

This follow-up trains the released Qwen3-Reranker-0.6B against the same frozen query-first Qwen3-Reranker-4B targets used in the previous pilot. It changes only the student prompt body to `<Query>: document content` followed by `<Document>: query content`. The instruction and official prefix/assistant suffix remain fixed. All training and evaluation here uses the swapped student format.

## Recovery against the original query-first baseline

The frequent human-judgment panel has six queries per task across 10 English and eight Chinese retrieval tasks. NDCG@10 is reported as a percentage, with tasks weighted equally. Held-out teacher agreement uses 64 distinct mixture queries × eight candidates.

| Model / updates | English NDCG@10 | Chinese NDCG@10 | Teacher ordering agreement | Teacher margin MSE | Touché NDCG@10 (6 queries) |
|---|---:|---:|---:|---:|---:|
| Original query-first 0.6B | 69.646 | 76.898 | 81.859% | 6.3827 | 76.219 |
| Query-first warmup, 50 updates | 69.931 | 77.231 | 83.175% | 4.4083 | 77.191 |
| Contents swapped, 0 updates | 54.626 | 64.088 | 70.241% | 15.0516 | 58.033 |
| Contents swapped, 1 updates | 53.755 | 64.147 | 70.241% | 15.0801 | 57.729 |
| Contents swapped, 3 updates | 53.874 | 64.028 | 70.269% | 14.9417 | 56.632 |
| Contents swapped, 10 updates | 55.006 | 64.249 | 70.969% | 13.1239 | 58.153 |
| Contents swapped, 25 updates | 54.232 | 65.169 | 74.160% | 9.5498 | 60.596 |
| Contents swapped, 50 updates | 55.358 | 65.575 | 75.364% | 8.4936 | 65.194 |

## Frequent panel: per-task baseline, damage and recovery

| Task | Original query-first | Untrained swapped | Trained swapped (50) | Δ versus original | 4B query-first teacher |
|---|---:|---:|---:|---:|---:|
| ArguAna | 81.546 | 84.360 | 78.209 | -3.338 | 93.849 |
| CQADupstackGamingRetrieval | 63.933 | 64.116 | 63.933 | +0.000 | 66.116 |
| CQADupstackUnixRetrieval | 45.556 | 35.130 | 32.596 | -12.960 | 59.093 |
| ClimateFEVERHardNegatives | 57.062 | 16.737 | 15.942 | -41.120 | 62.505 |
| FEVERHardNegatives | 100.000 | 81.546 | 79.364 | -20.636 | 100.000 |
| FiQA2018 | 65.908 | 44.327 | 50.184 | -15.723 | 56.433 |
| HotpotQAHardNegatives | 92.214 | 57.412 | 57.725 | -34.490 | 96.616 |
| SCIDOCS | 24.124 | 17.924 | 21.946 | -2.178 | 23.234 |
| TRECCOVID | 89.901 | 86.676 | 88.487 | -1.415 | 96.291 |
| Touche2020Retrieval.v3 | 76.219 | 58.033 | 65.194 | -11.025 | 75.844 |
| EcomRetrieval | 63.889 | 34.120 | 35.275 | -28.614 | 63.350 |
| VideoRetrieval | 100.000 | 85.515 | 91.667 | -8.333 | 100.000 |
| MedicalRetrieval | 50.000 | 18.849 | 18.849 | -31.151 | 43.849 |
| MMarcoRetrieval | 83.333 | 72.900 | 73.845 | -9.489 | 89.781 |
| CmedqaRetrieval | 41.404 | 50.980 | 50.715 | +9.310 | 41.975 |
| DuRetrieval | 94.376 | 80.248 | 81.978 | -12.398 | 93.752 |
| CovidRetrieval | 93.849 | 85.515 | 87.698 | -6.151 | 100.000 |
| T2Retrieval | 88.336 | 84.577 | 84.577 | -3.759 | 93.369 |

| Model | English mean | Chinese mean |
|---|---:|---:|
| Original query-first | 69.646 | 76.898 |
| Query-first trained | 69.931 | 77.231 |
| Untrained swapped | 54.626 | 64.088 |
| Trained swapped | 55.358 | 65.575 |

## Reserved endpoint panel: per-task baseline, damage and recovery

| Task | Original query-first | Untrained swapped | Trained swapped (50) | Δ versus original | 4B query-first teacher |
|---|---:|---:|---:|---:|---:|
| ArguAna | 100.000 | 85.767 | 87.500 | -12.500 | 100.000 |
| CQADupstackGamingRetrieval | 73.302 | 81.595 | 81.595 | +8.293 | 77.566 |
| CQADupstackUnixRetrieval | 61.562 | 62.704 | 62.704 | +1.142 | 64.273 |
| ClimateFEVERHardNegatives | 47.636 | 11.267 | 10.993 | -36.643 | 36.688 |
| FEVERHardNegatives | 100.000 | 64.072 | 63.773 | -36.227 | 100.000 |
| FiQA2018 | 66.689 | 48.215 | 48.662 | -18.027 | 84.463 |
| HotpotQAHardNegatives | 100.000 | 85.766 | 84.033 | -15.967 | 100.000 |
| SCIDOCS | 20.459 | 22.910 | 22.170 | +1.711 | 32.820 |
| TRECCOVID | 95.269 | 96.720 | 97.371 | +2.102 | 97.396 |
| Touche2020Retrieval.v3 | 69.872 | 20.174 | 28.091 | -41.781 | 76.999 |
| EcomRetrieval | 40.560 | 21.405 | 22.171 | -18.389 | 37.307 |
| VideoRetrieval | 87.500 | 72.171 | 72.171 | -15.329 | 100.000 |
| MedicalRetrieval | 50.000 | 50.000 | 50.000 | +0.000 | 50.000 |
| MMarcoRetrieval | 50.000 | 50.000 | 50.000 | +0.000 | 65.859 |
| CmedqaRetrieval | 43.875 | 63.594 | 66.273 | +22.399 | 40.551 |
| DuRetrieval | 92.419 | 90.831 | 93.223 | +0.804 | 98.569 |
| CovidRetrieval | 87.500 | 87.500 | 90.773 | +3.273 | 100.000 |
| T2Retrieval | 89.868 | 94.591 | 94.193 | +4.325 | 94.577 |

| Model | English mean | Chinese mean |
|---|---:|---:|
| Original query-first | 73.479 | 67.715 |
| Query-first trained | 74.226 | 68.565 |
| Untrained swapped | 57.919 | 66.262 |
| Trained swapped | 58.689 | 67.351 |

Combined endpoint comparison (all 10 preselected queries per task):

| Model | English | Chinese |
|---|---:|---:|
| Original query-first | 71.179 | 73.225 |
| Query-first trained | 71.649 | 73.764 |
| Untrained swapped | 55.943 | 64.957 |
| Trained swapped | 56.690 | 66.285 |

## Timing and memory

Training averaged **17.29 seconds/update** (range 16.79–19.26), with 32 queries × eight documents = 256 pairs per update. All 50 training updates took 14.41 minutes; process wall time including preparation, evaluation and saves was 25.87 minutes.
Frequent evaluation averaged 72.34 seconds. The extra baseline/endpoint panel took about 45 seconds. Checkpoint saving averaged 16.98 seconds. Peak allocated NPU memory was 22.04 GiB.

Evaluation runs at updates 0, 1, 3, 10, 25 and 50. The reserved four-query-per-task panel runs only at 0 and 50. The source, loss and batch recipe are otherwise the same as the query-first warmup/decay control.

## Recipe and alignment checks

- Same 1,600 training groups: 800 MIRACL_zh, 267 NQ, 267 MIRACL_en, 266 SQuAD; same 64 held-out mixture queries and fixed human-judgment panels. Dataset SHA256 and frozen teacher-file SHA256 match the previous experiment.
- Mean Margin-MSE over all 28 unordered candidate pairs per query, then equally over 32 queries. Raw yes-minus-no score differences are the targets. Fresh full-model NpuFusedAdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0, gradient clipping 1.0.
- FP32 parameters, BF16 autocast, owned model with Ascend fusion attention. Deterministic score-gradient replay bounds activation memory; train microbatches contain at most four sequences and 8,192 padded positions, left-padded to the longest sequence rounded up to 128.
- Five-update warmup to peak LR 1e-6, then the same linear decay with nonzero last update. Both this run and the query-first warmup control consume the same groups in the same order, without repeats.
- The original canonical query-first token hashes are checked against the saved 4B targets. Swapped token hashes are recorded separately and must differ. Teacher and student have the same candidate counts and zero truncated inputs in every section. The query-first reference model hashes, dataset hash, teacher hash and original baseline arrays are also checked.
- The teacher remains query first. The student preserves `<Instruct>`, `<Query>` and `<Document>` label order while swapping only query/document contents. Instruction, system prefix and assistant suffix are unchanged.
- Exact-text/source-family exclusions, benchmark query blocking and selected panel document blocking are inherited from the original dataset. This is a four-source passage-retrieval pilot, not exhaustive semantic decontamination or a full retrieval training mixture.

## Implementation checks

Six mathematical tests passed. The swapped-input real-example backward control passed, with HF and owned eager scores/loss/checked gradients identical. Fusion versus HF representative gradient relative L2 differences range from 3.59% to 14.84%; these are numerical differences, not bitwise parity. Replay versus joint fusion backward differs by at most 0.211% relative L2. Ordinary/fused AdamW gives maximum checked parameter difference 0.0. The control checks representative tensors on one four-example batch; see `control/control.json` for all values.

## Saved evidence

- Executed source commit: `139da06fe810d6861c7e3402796dcf09cc7ddd9d`. Published experiment branch: `codex/qwen-contents-swapped-distill`.
- Environment: `{"chip": "Ascend 910B2", "host": "liteserver-c001-4", "physical_npu": "3", "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post2", "transformers": "5.5.4"}`.
- Dataset SHA256: `0fc73a4a5df7f755ea0cdbc58afdee1e87141d07f9575f01ee720588bf86899e`; teacher SHA256: `7b130b26a8ff74b3f2c8a788ca7bfb05114bbf593842fec41d8ec873dbe94cfc`.
- `warmup_linear/result.json` retains all per-query metrics and candidate score arrays, both token layouts, the original baseline, update timings and checkpoint records.
- `reference/` retains the original query-first warmup run, frozen teacher targets and source/split/candidate provenance manifest. Training groups and candidate order are identical to that reference.
- `touche_changes.json` records original query-first, untrained swapped and trained swapped top-ten document IDs, judgments, scores and character lengths.
- Every evaluated nonzero checkpoint contains model/optimizer states, scheduler description, CPU/NPU RNG states, training order and data/teacher hashes. The five large checkpoint files remain on the server:

- `/workspace/results/qwen_margin_distill/npu/contents_swap_139da06f/warmup_linear/checkpoint_001.pt` (7,770,900,018 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_139da06f/warmup_linear/checkpoint_003.pt` (7,770,900,018 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_139da06f/warmup_linear/checkpoint_010.pt` (7,770,900,018 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_139da06f/warmup_linear/checkpoint_025.pt` (7,770,900,018 bytes)
- `/workspace/results/qwen_margin_distill/npu/contents_swap_139da06f/warmup_linear/checkpoint_050.pt` (7,770,900,018 bytes)

## Interpretation

**The model learned toward the teacher, but 50 conservative updates did not restore query-first benchmark accuracy.** Held-out teacher margin MSE fell from 15.0516 to 8.4936 (43.57%), and pairwise agreement rose from 70.241% to 75.364%. Both remain worse than the original query-first student's 6.3827 MSE and 81.859% agreement.

Across all 10 sampled queries per task, English NDCG@10 increased from 55.943 to 56.690 and Chinese from 64.957 to 66.285. The original query-first references were 71.179 and 73.225: the final deficits are 14.489 English and 6.940 Chinese points. The separate reserved panel shows the same direction of modest improvement. Recovery is uneven across tasks; for example, Touché improves from its swapped baseline while the sampled FEVER and ClimateFEVER scores do not.

The matched query-first warmup control preserved sampled accuracy and improved teacher agreement. Under this short recipe, changing the input arrangement adds a much harder adaptation problem. The result does not establish that document-first adaptation is impossible, or that contents-only swapping is easier than correctly relabeling the fields. Neither a longer run nor another document-first prompt was tested here.

Success is judged against the original query-first accuracy reference, alongside teacher agreement; improvement over the damaged swapped baseline alone is insufficient for parity. The tables report both comparisons. With only six/four queries per task, numerical gains or regressions can reflect one ranking change. These diagnostic samples do not establish full MTEB/CMTEB parity or 4B-level accuracy. This run evaluates full swapped prompts and does not measure cached-document serving speed.
