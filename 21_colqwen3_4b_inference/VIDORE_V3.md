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
  /workspace/venvs/colqwen3_hf_py312/bin/python -u \
  21_colqwen3_4b_inference/download_vidore_v3.py \
  --root /workspace/datasets/ViDoRe_v3 --endpoint https://hf-mirror.com
```

Official HF is the default endpoint; the server currently times out connecting
to it, while the mirror is reachable. No token is sent to the mirror. The script
is resumable and verifies every downloaded file against the pinned repo's
size and LFS SHA256 or Git blob ID. Per-domain manifests record SHA256 hashes,
Parquet counts/schemas, language counts and qrels referential integrity.
The irrelevant `pdfs/.DS_Store` Finder metadata file in finance FR is excluded
and explicitly recorded: the mirror denies that file with HTTP 403. No corpus,
query, qrels, document metadata, or PDF content is excluded.
The mirror supplies metadata too; this is not independent official-origin
attestation. Keep the endpoint recorded with the lock.

Offline consumers can load local Parquet directly, e.g.
`load_dataset('parquet', data_files={'test': sorted(path.glob('corpus/*.parquet'))}, split='test')`.
Keep query/corpus IDs and supplied relevance grades unchanged. A future scoring
runner should persist corpus/query embeddings and rankings, compute MaxSim in
bounded chunks, and report each domain plus the specified macro aggregation.
Downloading the dataset does not run or validate the retrieval evaluation.

## Verified local inventory — 2026-10-01

All eight public domains passed file-hash/size, unique-ID and qrels-reference
checks under `/workspace/datasets/ViDoRe_v3`. The checked-in
[manifests](references/vidore_v3_download/) contain file-level evidence.

| Domain | Pages | Queries, all six languages | English queries |
|---|---:|---:|---:|
| HR | 1,110 | 1,908 | 318 |
| Finance EN | 2,942 | 1,854 | 309 |
| Industrial | 5,244 | 1,698 | 283 |
| Pharmaceuticals | 2,313 | 2,184 | 364 |
| Computer science | 1,360 | 1,290 | 215 |
| Energy | 2,225 | 1,848 | 308 |
| Physics | 1,674 | 1,812 | 302 |
| Finance FR | 2,384 | 1,920 | 320 |
| **Total** | **19,252** | **14,514** | **2,419** |

Verified files total approximately 9.53 GB including source PDFs, excluding
download-cache metadata and the 6,148-byte Finder file described above.
Two concurrent groups with eight download workers each achieved observed
interval rates of roughly 15–37 MB/s after the initial domain download.
These are transfer observations, not guaranteed endpoint bandwidth.

`prepare_vidore_smoke.py` exports one linked English query and original corpus
image from each of HR and computer science, with IDs and hashes preserved.
Those two pages passed the HF baseline; no full-domain ranking or nDCG score
has been computed yet.
