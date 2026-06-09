# NPU Hot-Swap Bottleneck Matrix

Run this from the work/NPU lane after pulling the latest `main`:

```sh
cd /home/lukaiv/paddle_ocr_vl_npu/04_batched_fixed_cohort_decode
bash run_npu_hotswap_bottleneck_matrix.sh
```

The runner executes the fixed-cohort warmup/reference and the hot-swap
`num-items=8,9,16,32,100` matrix twice:

- `off_*.json`: clean throughput with `--step-timing off`.
- `both_*.json`: diagnostic timing with CPU and NPU per-step records.

Both passes use the same batch size, cache length, TorchAir cache directory,
dtype, and EOS mode. The runner writes one JSON file per run under
`outputs/hotswap_bottleneck_matrix/` and prints the same JSON to stdout.

Environment overrides are supported without editing tracked files:

```sh
PYTHON_BIN=/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python \
MODEL=/home/lukaiv/models/paddle_ocr_0_9b_v_1_6 \
DEVICE=npu:0 \
CACHE_LENGTH=1269 \
TIMING_MODES="off both" \
bash run_npu_hotswap_bottleneck_matrix.sh
```

If `CACHE_LENGTH=1269` is too small, stop and report the exact error. Then run
one discovery command without `CACHE_LENGTH` manually, note the selected
`cache_length` from the JSON, and rerun the whole matrix with that explicit
larger value.

Read the matrix this way:

- `02_fixed_warm_reference` is the speed baseline.
- `03_hotswap_no_replacement_8` isolates scheduler overhead with no real
  replacement.
- `04_hotswap_one_extra_9` isolates the smallest nontrivial replacement case.
- `05_hotswap_one_wave_16` and `06_hotswap_several_waves_32` show replacement
  scaling.
- `07_hotswap_full_100` is the real workload.

For clean throughput, read `off_*.json`: paste back `correctness`, `tok_per_s`,
`phase_timing_s`, and `timing_s`.

For diagnostic timing, read `both_*.json`: paste back `correctness`,
`tok_per_s`, `phase_timing_s`, `speed_debug`, `loop.step_timing_summary`,
`timing_s`, and `timing_accounting`.

In hot-swap results, `hotswap_total_*` includes active-cache setup, initial
slot loads, steady loop, final drain, and result materialization. The
`hotswap_steady_*` rates use only `phase_timing_s.steady_decode_loop_s` and are
the better comparison for scheduler steady-state decode. Do not write inline
parsing scripts.
