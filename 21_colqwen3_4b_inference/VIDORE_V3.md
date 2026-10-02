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
Keep query/corpus IDs and supplied relevance grades unchanged. Scoring should
retain corpus/query embeddings in memory, persist compact rankings and metrics,
compute MaxSim in bounded chunks, and report the specified domain aggregation.
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
Those two pages passed the initial HF baseline. The subsequent full HR English
evaluation is described below, with measured results in the README.

## Current performance-testing contract (experiment 21 only)

`run_hr_evaluation.py` defaults to the fixed **HR development workload**:
111 pages at corpus indices `5,15,...,1105`, plus 32 deterministically selected
English queries with a positive qrel among those pages. The query selection
uses SHA256 of `hr-dev-v1:<query_id>`, and preserves source order after selection.
The exact IDs and selection hash are saved in `workload.json`. Development qrels
are restricted to the selected corpus; these scores are internal anchors, not
comparable with the published full-HR score. Full HR requires `--workload full`
and is to be run **only when Luka explicitly requests it**.

One B1/sequential pipeline and one default observer. `--profile` is the only
observation toggle. No embedding files are saved: CPU embeddings remain in
memory through scoring, then are released when the process exits. Compact
scores, rankings, metrics and timing evidence are retained. No changes to
batching, prefetch, backpressure, or other experiments are included.

```sh
# Default: complete pipeline on the development subset, NOT full HR.
"$PYTHON" -u 21_colqwen3_4b_inference/run_hr_evaluation.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --dataset-root /workspace/datasets/ViDoRe_v3_hr_mteb_reference \
  --output-dir tmp/21_colqwen3_4b_inference/hr_dev_new
# Add --profile to capture actual runtime items, not subsection replays.
```

`pipeline_timing.py` records host spans for setup, dataset verification/read,
selection, model/processor preparation, preprocessing, transfers/materialization,
vision preparation/dispatch/transformer, text preparation/dispatch/transformer,
retrieval projection, output validation, scoring, ranking, metrics and artifacts.
The corpus is encoded first, then the query batch is encoded and ranked against
the existing in-memory corpus. `query_to_ranked_batch_s` is the wall time of
that **offline query-batch workflow**, not a single online-query latency.

Device events are recorded around device-producing sections, with **no timing
synchronization before/after each section**. Events are resolved nonblockingly
after the natural CPU-output materialization required by the consumer. An
incomplete event remains explicitly pending; later `device_resolution` records
retain its identity and elapsed interval. No host/device absolute-clock
alignment is inferred. `host_s` can include submission work and implicit waits;
`device_interval_ms` is stream elapsed time, potentially including launch gaps
and waits, not summed active kernel time. In particular, a blocking CPU copy may
wait for previous work and is called `output_materialize_wait`, not pure D2H
execution. Neither host nor device section sums should be added together.

- Every completed page, query, or scoring item prints and flushes its full timing,
  token and progress record **immediately**. It also flushes `items.jsonl` and
  buffered section events. Completion does not wait for the heartbeat.
- A **5-second heartbeat** reports progress and the active item/host section.
  A host section taking 120 seconds triggers a diagnostic stack dump; no wait is
  introduced into inference to detect it.
- Fine section events are buffered, not printed individually. Raw section
  starts/finishes and device resolutions remain in `events.jsonl`.
- `result.json` reports mean/p50/p90/p99/max and weighted device-interval tok/s
  by section, exact token length and execution route. Host timings are not
  mislabeled as device tok/s. Whole-item latency and end-to-end pg/s use wall
  time; item accounting residuals and `encoding_outside_item_spans_s` remain
  explicitly separate rather than being assigned to a nearby section.
- Existing compatible graphs may load on their **first real invocation**.
  Its output is used, with `stage_first_use` tagging; there is no throwaway stage
  call. Uncached shapes stay explicitly optimized eager. Caches are not cleared.

With `--profile`, the entire same pipeline runs. The profiler observes one
actual page, one actual query, and one actual scoring item, each preceded by a
real warmup item in its workflow. Captured items are the second item of each
phase. Trace export can distort timings and is labeled by `profiler=true`;
use normal observer runs for throughput. Full pipelines, never isolated replay
functions, are used to assess performance.

`bench_observer_overhead.py` is a **development-only ABBA test**, not a production
observation-level option. It runs the identical complete HR-dev pipeline in four
fresh processes (control, observed, observed, control). The private control
omits event/logging observation but retains workload metadata, validation,
scoring and result output. It checks exact score/ID parity and reports observed
versus control wall times for pages, queries, encoding, scoring and the whole
job. Thus reported overhead concerns the observer, not every shared metadata
operation. Run variance must be considered; no single difference proves a
precise overhead percentage. Comparisons to the historical 2.79 pg/s anchor also
need to account for removing embedding serialization and timing barriers.

## Historical full English HR evaluation (`fbfaedc5`)

`run_hr_evaluation.py` evaluates the entire HR English retrieval task: 1,110
candidate pages, 318 queries, all supplied relevance labels, 2560-dimensional
embeddings, original image processing, and no resolution/token reduction.
The score reference is **0.66088 nDCG@10** from MTEB 2.4.2 for checkpoint
`4894b7d451ff33981650acc693bb482dbef302d3` (FP16/FlashAttention2).
The [published result](references/hr_protocol/published_result.json) and
[model metadata](references/hr_protocol/published_model_meta.json) are copied
from `embeddings-benchmark/results`, under that model/revision directory.

The published dataset SHA belongs to **a different repository**,
`vidore/vidore_v3_hr_mteb_format`, not the original HR repository above.
Use `download_hr_reference.py` to fetch just its pinned English components at
`bc7d43d64815ed30f664168c8052106484aba7fd`; all three SHA256 values are enforced
by both downloader and evaluator. The reference is thus not casually compared
against a potentially different revision of the original-format data.
Its English qrels component contains all 1,908 multilingual query IDs, while
the English query component contains 318. The runner validates corpus references
and English-query coverage, then evaluates exactly those 318 query IDs. It does
not average in the 1,590 other-language query IDs that were never requested.

```sh
# From the 910B repo root, with the experiment venv and source npu-setup:
PYTHON=/workspace/venvs/colqwen3_hf_py312/bin/python
"$PYTHON" 21_colqwen3_4b_inference/download_hr_reference.py \
  --root /workspace/datasets/ViDoRe_v3_hr_mteb_reference
"$PYTHON" -u 21_colqwen3_4b_inference/run_hr_evaluation.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --dataset-root /workspace/datasets/ViDoRe_v3_hr_mteb_reference \
  --workload full \
  --output-dir tmp/21_colqwen3_4b_inference/hr_new/output
```

Requires `pytrec-eval-terrier==0.5.10` in the experiment venv. On the current
server its build-time GitHub download is unreachable; the official
`usnistgov/trec_eval` v9.0.8 archive was transferred from local and unpacked
into the package's `trec_eval/` directory before installation, without source
modifications. Do not replace this metric with an exponential-gain nDCG formula.

Execution uses the existing optimized native-weight/PromptFA/Linear-patch path.
Only matching on-disk transformer graphs are loaded; unseen exact shapes use
explicitly labeled optimized raw eager. A compiled-call error is fatal, not an
automatic fallback. No graphs are freshly compiled and no cache is cleared.
This mixed-route evaluation is not an all-shapes-compiled performance claim.

Observability and artifacts:

- `events.jsonl` and flushed stdout: item/section starts and finishes, token
  counts, route, elapsed time, completion count, throughput and ETA. A 10-second
  heartbeat names the active section; a section taking 120 seconds triggers
  repeating Python stack dumps. The stack dump is diagnostic, not a timeout.
- `items.jsonl`: per-query, per-page and per-scoring-page wall latency, raw
  section latencies, NPU event elapsed time, image size/grid/hash, vision tokens,
  merged image tokens, text/prompt tokens, and embedding-row counts.
- `encoding_summary.json`, final `result.json`: aggregate sum/mean/p50/p90/p99/max,
  weighted wall/device tok/s by section and by exact length/route, setup/cache
  loading separately identified, encoding pg/s and scoring duration.
- `embeddings/*.pt`, `scores.npy`, `ids.json`, `rankings.jsonl`, and
  `per_query_metrics.json`: reproducible embeddings, full score matrix and
  rankings, plus pytrec_eval nDCG@10, Recall@10 and MAP@10.
- `progress.json`: latest periodic encoding/scoring progress snapshot.

Stage timings synchronize the NPU and include diagnostic overhead; event elapsed
time is an instrumented stream interval, not a sum of individual kernel times.
Page throughput includes image processing, transfer, preparation, both
transformer stacks, retrieval projection, validation and embedding serialization.
Query encoding, model load and retrieval scoring are separately reported.
Scoring uses FP32 NPU dot products and maxima, with small per-query sums on CPU;
the first scoring column is cross-checked against the checkpoint processor's
official MaxSim implementation. Original zero-valued document rows are preserved.
For documents shorter than the corpus-wide maximum embedding length, one
additional zero row reproduces the MaxSim effect of the official adapter's
global zero padding without computing every padded dot product.

The historical structural preflight used `--limit-pages 8 --limit-queries 8`. Such a run
is explicitly labeled `completed_partial_smoke`, and does **not** produce a
published-score comparison. A full-domain run validates HR only, not the
eight-domain mean or all six query languages.

The historical runner's per-section synchronization, embedding serialization,
10-second heartbeat and smoke-limit CLI described above have been superseded by
the single-observer development contract. The archived result remains unchanged.
