# Qwen3-8B KV64 decode batch sweep

## Contract

- Device: Ascend 910B2, physical NPU 7
- Model: `/workspace/models/Qwen3-8B`, FP16
- Prompt/prefill tokens: 32
- Generated decode tokens: 32 per sequence
- Static KV capacity: 64 tokens
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- Eager and static `torch.compile`/TorchAir were measured independently.
- Timed decode excludes model loading, prefill/cache construction, graph
  compilation, and warmup.
- All compiled lanes used exactly one Dynamo graph and matched eager tokens
  exactly over all 32 decode steps.
- B2 used two measured repeats during the initial validation; all other lanes
  used three. Timings were stable enough for the comparison.

`tok/s` is total generated tokens across all batch members. `batch/s` is the
number of complete B-wide one-token decode iterations per second, so
`tok/s = B * batch/s`.

## Results

| B | eager tok/s | compiled tok/s | eager batch/s | compiled batch/s | compiled/eager | compiled 32-step wall (s) | compiled peak allocated (GiB) |
|---:|------------:|---------------:|--------------:|-----------------:|---------------:|--------------------------:|------------------------------:|
| 1 | 15.90 | 68.93 | 15.90 | 68.93 | 4.33x | 0.464 | 16.42 |
| 2 | 30.47 | 124.16 | 15.24 | 62.08 | 4.07x | 0.515 | 16.42 |
| 4 | 62.36 | 236.13 | 15.59 | 59.03 | 3.79x | 0.542 | 16.42 |
| 8 | 124.98 | 467.39 | 15.62 | 58.42 | 3.74x | 0.548 | 16.42 |
| 16 | 236.55 | 817.36 | 14.78 | 51.09 | 3.46x | 0.626 | 16.42 |
| 32 | 463.55 | 1,536.15 | 14.49 | 48.00 | 3.31x | 0.667 | 16.54 |
| 64 | 915.72 | 2,745.49 | 14.31 | 42.90 | 3.00x | 0.746 | 17.81 |
| 128 | 2,057.38 | 4,575.53 | 16.07 | 35.75 | 2.22x | 0.895 | 20.37 |
| 256 | 3,899.47 | 6,231.45 | 15.23 | 24.34 | 1.60x | 1.315 | 25.48 |
| 512 | 7,434.43 | 8,710.73 | 14.52 | 17.01 | 1.17x | 1.881 | 35.69 |

## Interpretation

- Compiled throughput is best at B512 in absolute terms: 8.71k tok/s.
- Compilation has its largest relative value at small batches, about 4.3x at
  B1, then falls continuously to 1.17x at B512 as the eager path amortizes its
  Python and dispatch overhead.
- Compiled one-token iteration throughput is nearly flat around 58--69 batch/s
  through B8, then falls as device compute grows. Eager stays around 14--16
  batch/s across the entire sweep.
- B512 is not an OOM edge under this KV64 contract: compiled peak allocation
  was 35.69 GiB on the 60.96 GiB device.

Each lane directory contains its exact JSON result and full log.
