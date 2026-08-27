# Canonical persistent UniRec service configuration

This record freezes the persistent-service configuration accepted on Atlas 310P
on 2026-08-27. The corresponding runner is
`run_persistent_unirec_service_benchmark.py`. The full work-server procedure is
`WORK_SERVER_310P_UNIREC_PERSISTENT_RESIDENT_K20_FULL1651.md`.

## Reported 310P result

The 310P work agent reported these full OmniDocBench v1.6 results after the
fresh persistent decode-cache gate and the 1,651-page hot window:

| Metric | Result |
|---|---:|
| Hot throughput | 3.2 pages/s |
| Overall accuracy | 90.22% |
| Peak process-tree PSS | 7.993 GB |
| Peak process-tree RSS | 9.350 GB |
| Peak HBM | 16.330 GB |

The raw run remains on the 310P work server. These values are the report relayed
by Luka, not copied raw artifacts.

## Default service configuration

The runner defaults encode the accepted service path:

| Setting | Default |
|---|---:|
| Excluded real-page warmup | 512 pages |
| CPU crop workers | 4 |
| Recognition threads per worker | 8 |
| Recognition resize chunk | 0 |
| Layout lanes | 1 |
| Layout batch | 2 |
| Layout threshold | 0.5 |
| Vision preset | `310p_k20_l4` |
| Vision lanes | 4 |
| Vision graph residency | `all` |
| Require every K20 graph during warmup | yes |
| Same-key vision shards | 1 |
| Sharded vision-key count | 0 |
| Vision record budget | 128 |
| Maximum calls per vision key | 64 |
| Vision queue | 128 |
| Tall-crop fallback | `eager` |
| Decode batch | 128 |
| Cross-KV capacity | 1,320 |
| Self-KV capacity | 2,048 |
| Maximum decoded length | 2,048 |
| Decode-ready queue | 128 |
| Progress interval | 16 pages |

The model path, compiled-FP32 B2 layout cache, K20 cache, decode cache parent,
dataset, output directory, and spool directory remain required arguments because
their absolute paths differ by host.

The work-server run also applies CPU affinity `0-63`, samples process-tree memory
every 200 ms, samples HBM every 1 s, and uses
`CANN_KNOWLEDGE_BANK_PROCESS_NUM=0` during replay and measured inference. One
knowledge-bank process is allowed only while building the single fresh
B128/C1320/S2048 decode graph through normal real requests.

`--write-outputs` remains explicit. It controls excluded benchmark artifact
writing, not the hot serving path. The service always returns its in-memory page
result.

## Accuracy contract

The full evaluator uses the frozen repository-local OmniDocBench environment and
strips embedded HTML image tags from the evaluation copy. The source Markdown
files remain unchanged. Do not compare a result from another TeX installation
or an evaluation that scores the image tags as text.
