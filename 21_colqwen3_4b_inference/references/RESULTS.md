# HF baseline evidence — 910B2, 2026-10-01

Scope: direct Ops-Colqwen3-4B embedding and MaxSim smoke, not a retrieval
benchmark. No 310P validation is claimed.

## Execution contract

- Container: `research_vllm_ascend_021_external_workspace`, host
  `liteserver-c001-4`; free healthy physical 910B2 device 7 selected by
  `source npu-setup`, exposed as `npu:0`.
- Environment: `/workspace/venvs/colqwen3_hf_py312/bin/python`, Python 3.12.13,
  torch 2.10.0+cpu with torch-npu 2.10.0, Transformers 4.57.1.
- Checkpoint: `/workspace/models/Ops-Colqwen3-4B`; checkpoint-supplied
  `OpsColQwen3Model` and `OpsColQwen3Processor`, FP16, 2560-dimensional output,
  `attn_implementation=eager`, no `torch.compile`, no processor overrides.
- Queries batch together; images B1. First forward excluded from three-repeat
  warm means. Forward timings synchronize NPU and exclude preprocessing,
  transfers, serialization and MaxSim scoring.
- Source edits committed locally and pulled on the container. Persistent SSH
  master to the host plus `docker exec` avoids the unavailable port-22021 alias.

## Results

| Input | Embedding shape | Raw vision tokens | Warm forward mean |
|---|---|---:|---:|
| Two crop-smoke queries | 2 × 20 × 2560 | — | 86.48 ms |
| Committed text crop | 1 × 189 × 2560 | 700 | 136.90 ms |
| Committed table crop | 1 × 142 × 2560 | 512 | 135.23 ms |
| Two ViDoRe English queries | 2 × 51 × 2560 | — | 93.85 ms |
| HR page, corpus ID 372 | 1 × 1274 × 2560 | 5040 | 473.38 ms |
| Computer science page, corpus ID 157 | 1 × 1254 × 2560 | 4960 | 458.09 ms |

Both runs passed: 715 expected checkpoint tensors, no missing/unexpected or
shape-mismatched weights, finite embeddings, maximum active-row norm error
below 0.00046, and exactly repeated embeddings. Padding query rows are excluded
from active-row normalization checks.

The full-page MaxSim matrix was `[[33.3125, 13.6015625], [9.2578125, 26.40625]]`.
Both queries preferred their linked page among these two candidates. This is
only a sanity check, not nDCG or evidence of full-benchmark accuracy.

## Artifacts

- Crop smoke source: `b25734d4`; [result](hf4571_smoke_910b/result.json).
- Full-page smoke source: `247c121b`; [result](vidore_smoke_910b/result.json),
  [input provenance](vidore_smoke_910b/samples.json),
  [run log](vidore_smoke_910b/run.log), [exit code](vidore_smoke_910b/exit_code.txt).
- Server run roots (relative to `/workspace/repos/paddle_ocr_vl_npu`):
  `tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4` and
  `tmp/21_colqwen3_4b_inference/vidore_smoke_247c121b`.
  Their output directories retain processor tensors and embeddings as `.pt`.
- Result files hash model code/config/tokenizer and input images. Weight-shard
  SHA256 fields are null in these successful runs: do not present the recorded
  weight sizes or tensor-contract checks as full weight-content verification.
- Eight dataset [manifests](vidore_v3_download/) record pinned revisions and
  verified per-file hashes, including embedded corpus images and source PDFs.

## Compatibility finding

Transformers 5.5.4 passed text forward but failed image forward because the
checkpoint wrapper drops the newly required `mm_token_type_ids` argument.
Pinning its recorded Transformers 4.57.1 version solved this without changing
checkpoint code. The isolated HF environment must not be used for inherited
vLLM packages that require Transformers 5.5.4.

Next evaluation work is to pin the official evaluator and language/aggregation
protocol, encode each domain's full corpus and queries, and score bounded MaxSim
chunks against supplied qrels. None of that evaluation is claimed complete.
