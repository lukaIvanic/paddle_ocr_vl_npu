# UniRec low-memory host and HBM measurement

This directory records the full 1,651-page W4/T8 low-memory UniRec run on one
Ascend 910B2 at commit `4fc7311`. The run used the same production settings as
`../unirec_910b_lowmem_full1651_8272f87/` and added external physical-device
HBM sampling.

## Result

| Metric | Result |
|---|---:|
| Pipeline status | pass |
| Pages | 1,651 |
| Recognition crops | 32,110 |
| Internal pipeline wall | 414.968 s |
| Internal pipeline throughput | 3.9786 pages/s |
| External process wall | 423.580 s |
| External process throughput | 3.8977 pages/s |
| Peak host process-tree PSS | 4,410,212,352 bytes, 4.410 GB |
| Peak host process-tree RSS | 4.694 GB |
| Host PSS samples | 4,025 at 50 ms |
| Physical NPU | 7 |
| Idle HBM baseline | 3,415 MiB |
| Absolute HBM peak | 16,205 MiB, 15.825 GiB |
| HBM increase above idle | 12,790 MiB, 12.490 GiB |
| HBM samples | 212 at 2 s |
| HBM sampler errors | 0 |

The host PSS peak occurred at 84.637 seconds during layout. The HBM peak
occurred at 399.108 seconds during recognition and decode. The selected-device
process table reported 12,859 MiB for the task at the HBM peak. HBM returned to
3,509 MiB by the last shutdown sample.

The HBM phase ranges were:

| Phase | Absolute HBM range |
|---|---:|
| Layout, 0 to 135.55 s | 3,414 to 4,075 MiB |
| Layout release and recognition setup, 135.55 to 168.59 s | 3,415 to 3,418 MiB |
| Vision, text prefill, and decode, 168.59 to 414.97 s | 3,415 to 16,205 MiB |

`npu-smi info` reports physical-device total HBM. Its idle baseline includes
driver and device runtime memory, so the baseline-subtracted 12,790 MiB is the
useful pipeline footprint. The `npu-smi` polling processes were children of the
outer sampler, not the measured pipeline root, and therefore did not contribute
to process-tree PSS.

## Parity

The new and prior validated traces contain the same 32,110 request IDs, texts,
and 2,287,945 generated token IDs. Their sorted normalized SHA-256 is:

```text
528c2841067619b09879be8a843a396845471222a35197f581f4c3da622d7ac4
```

## Evidence

- `process_tree_and_hbm.json` contains the exact command, 50 ms host-memory
  peak, all compact HBM samples, idle baseline, and raw `npu-smi` output at the
  HBM peak.
- `run_summary.json` contains the pipeline settings, stage timings, and decode
  statistics.
