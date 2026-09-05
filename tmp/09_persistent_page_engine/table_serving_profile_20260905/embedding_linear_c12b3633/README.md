# Linear patch projection: real embedding profile

Source `c12b36335d3c9daf200c92b45ffc69b811dcc82f`, physical NPU6, 910B2.
Same diagnostic contract as `../embedding_bc201dda`: one real crop processed
independently eight times through the ordinary B2 API at client C1. Five
complete requests warm the path, followed by three profiled embedding forwards.
Only the equivalent linear patch projection is enabled. Input remains
`[1,3840,3,14,14]`, grid `[1,48,80]`, text input973 tokens.

| Mean across three captures | Original convolution | Linear projection |
|---|---:|---:|
| Projection kernel (microseconds) | 8215.153333333334 | 84.64 |
| All embedding kernels (microseconds) | 8547.593333333334 | 335.16 |
| Kernels per capture | 15 | 11 |

MatMulV2 replaces the full-patch Conv2D. Position interpolation remains about
116 microseconds. The analyzer verifies all three complete kernel lists,
including projection, interpolation and final output move; the profiler emits
a RECORD-state stop warning but the expected stage kernels are present.
Profiler export/synchronization blocks requests, so this capture's HTTP
latencies are **not performance measurements**.

On identical real patches, original Conv2D versus linear output has maximum
absolute difference0.001953125, mean absolute difference
5.3718004267011565e-08, relative L2 difference5.453072390082525e-06;
0.013879846665076911% of FP16 elements differ. All outputs are finite.
All eight full OCR requests reach EOS and match the convolution diagnostic's
native token stream. This single-crop diagnostic does not prove corpus parity;
the separate matched development100 run found one character-level difference.

The initial profile process spent222.464920s in setup because the changed
vision source invalidated its graph cache. The measured unprofiled100 runs
subsequently used cached setup (~35.5s). No synthetic precompile was used.
The model receives freshly prepared pixels/features/KV for every request.

Commands, configuration, raw native outputs, CSV exports, compressed Chrome
traces and same-input differences are retained. Full binary captures remain
at the remote path recorded by `command.txt`; duplicate raw profiler exports
are not committed. The common host monitor and manual PID mapping are in
`../../table_patch_linear_c12b3633_20260906/`. Workers and monitor were stopped;
a direct-host check at2026-09-06T07:14:16+08:00 confirmed NPU6 free.

Reproduce the CPU audit without loading a model:

```sh
python3 tmp/09_persistent_page_engine/table_serving_profile_20260905/embedding_bc201dda/analyze.py \
  tmp/09_persistent_page_engine/table_serving_profile_20260905/embedding_linear_c12b3633
```
