# Target benchmark: ViDoRe v3 public retrieval

Sources inspected 2026-10-01:

- Official collection: https://huggingface.co/collections/vidore/vidore-benchmark-v3
- Dataset organization: https://huggingface.co/vidore
- Official evaluator: https://github.com/illuin-tech/vidore-benchmark
- Model card: https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B

Eight public domains: HR, finance EN, industrial, pharmaceuticals, computer
science, energy, physics, finance FR. Two further domains are private and are
not part of this downloadable public benchmark. Do not label a public-only run
as all ten domains.

Each repository contains test Parquet components `corpus`, `queries`, `qrels`,
document metadata, and source PDFs. Use supplied corpus page images as visual
model inputs; rerendering PDFs would change the benchmark. Corpus Markdown is
available but is not the input to this image-retrieval baseline. Download all
languages; choose evaluation language explicitly later. Do not pool translated
queries and call the result an English score.

The official single-model evaluation route now points to MTEB. The ViDoRe
repository also retains legacy model evaluation and a separate pipeline
evaluation framework. Pin the evaluator/version and language selection before
comparing scores. The model card reports public-v3 nDCG@10 of 61.27 at 2560
dimensions, but this is a reported reference, not our measured result; verify
the exact query/language aggregation protocol before claiming reproduction.
Do not compare v3 nDCG@10 with v1/v2 nDCG@5.

## Storage and reproducible download

`vidore_v3_revisions.json` pins the eight dataset revisions observed through
the reachable HF mirror. The full repositories total about 9.53 GB, including
PDFs. Corpus Parquet already embeds the page images. Store under the persistent
workspace volume, never the container overlay or Git.

```sh
cd /workspace/repos/paddle_ocr_vl_npu
HF_HOME=/workspace/.cache/huggingface \
  /workspace/venvs/mineru_pro_vllm_py312/bin/python -u \
  21_colqwen3_4b_inference/download_vidore_v3.py \
  --root /workspace/datasets/ViDoRe_v3 --endpoint https://hf-mirror.com
```

Official HF is the default endpoint; the server currently times out connecting
to it, while the mirror is reachable. No token is sent to the mirror. The script
is resumable and verifies every downloaded file against the pinned repo's
size and LFS SHA256 or Git blob ID. Per-domain manifests record SHA256 hashes,
Parquet counts/schemas, language counts and qrels referential integrity.
The mirror supplies metadata too; this is not independent official-origin
attestation. Keep the endpoint recorded with the lock.

Offline consumers can load local Parquet directly, e.g.
`load_dataset('parquet', data_files={'test': sorted(path.glob('corpus/*.parquet'))}, split='test')`.
Keep query/corpus IDs and supplied relevance grades unchanged. A future scoring
runner should persist corpus/query embeddings and rankings, compute MaxSim in
bounded chunks, and report each domain plus the specified macro aggregation.
Downloading the dataset does not run or validate the retrieval evaluation.
