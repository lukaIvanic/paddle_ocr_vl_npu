# Two-AIV-block GQA experiment

This retained summary tests the proposal to assign one AIV block to each of the
two KV groups. It ran on physical Ascend 910B2 NPU6 on 2026-08-09.

## Result

The topology is correct but not competitive. The grouped control launched
exactly two AIV blocks, passed the stock FP16-tolerance check and independent
CPU FP32 reference at KV128, KV512, KV1024, KV1536, and KV2048, and ran through
the real B1 TorchAir full decoder.

The matched B1/KV1024 run used 20 warmups and 200 measured steps:

| Kernel package | Mean step | Throughput |
| --- | ---: | ---: |
| Current 16-block GQA AIV | 1.3636 ms | 733.35 tok/s |
| Two-block grouped control | 2.6582 ms | 376.20 tok/s |

The grouped topology increased latency by 94.94% and reduced throughput by
48.70%.

## What the control does and does not prove

The control serializes the eight query heads in each KV group on one AIV block.
It intentionally keeps the current per-head K/V load path. It is not yet the
larger UB-resident shared-K/V rewrite.

That distinction is measured, not assumed. At KV1024, both packages requested
about eight times the 1,029 KiB of unique direct input:

| Package | Blocks | GM to UB | Main-memory read |
| --- | ---: | ---: | ---: |
| Current 16-block | 16 | 8,236 KiB | 8,442.46 KiB |
| Grouped control | 2 | 8,215 KiB | 8,255.67 KiB |

The pipe profile also shows why removing those copies cannot recover the lost
parallelism when the vector algorithm stays the same:

| Package | Task | Vector | Scalar | MTE2 |
| --- | ---: | ---: | ---: | ---: |
| Current 16-block | 21.78 us | 11.40 us | 7.98 us | 4.74 us |
| Grouped control | 99.96 us | 90.83 us | 31.06 us | 34.78 us |

The grouped vector time is 7.97 times the current vector time. This is the eight
query heads becoming serial on each of two cores. Pipeline counters overlap, so
they must not be added. For this measured algorithm, task duration cannot fall
below its 90.83 us vector lane merely by deleting MTE2 copies. Across 18 decoder
layers, the most optimistic copy-only calculation gives at most about 401
tok/s. That is still 45.3% below the matched current result.

This is not a proof that every conceivable grouped vector algorithm is slow. A
new batched-head algorithm could change vector instruction efficiency. It does
show that “use two cores and remove K/V copies” is insufficient: the current
per-head vector kernels already use the vector width, and serializing eight of
them produces an almost exact eightfold vector-time penalty.

## Reproduction

Build the separate package. The resource attribute stays at the supported value
16; grouped tiling is what launches two blocks.

```sh
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
PADDLE_GQA_EXPERIMENT_VARIANT=grouped_serial_control \
PADDLE_GQA_BUILD_SOURCE_ROOT="$PWD/.runtime_cache/paddle_gqa_grouped_serial_increfa_aiv/sources/grouped_noflash" \
RUN_ID=grouped_serial_noflash \
bash 09_persistent_page_engine/custom_ops/paddle_gqa_increfa_aiv/build.sh
```

Run one clean eager row:

```sh
cd /workspace/repos/paddle_ocr_vl_npu/09_persistent_page_engine
source npu-setup
source <GROUPED_BUILD_ROOT>/installed/vendors/paddle_gqa_grouped_serial_increfa_aiv_transformer/bin/set_env.bash
/usr/local/python3.12.13/bin/python3 \
  scripts/probes/compare_paddle_gqa_increfa_aiv.py \
  --backend eager --kv-length 1024 --vector-core-count 16 \
  --experimental-grouped-serial-control \
  --warmup 20 --blocks 7 --repeats-per-block 200 \
  --output ../.runtime_cache/paddle_gqa_grouped_serial_increfa_aiv/results/kv1024.json
```

Add `--profile-dir`, `--profile-metric pipe`, and `--profile-calls 3` for the
bounded pipe profile. Use `memory_access` for the request-byte counters.

The full machine-readable summary is [summary.json](summary.json). Raw JSON and
profiler exports remain under the remote paths recorded there; large profiler
traces are not committed.
