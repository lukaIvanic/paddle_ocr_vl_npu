# Vision diagnostic accounting

Kernel durations normalized by the recorded profile-forward count (default 3). Wait and observed gaps are separate, non-additive measurements.

| Chip | Lane | Mode | Wall ms | Event ms | Useful tok/s | Status |
|---|---|---|---:|---:|---:|---|
| Ascend910B2 | matmul/M768_qkv_compiled_nd_internal_on | compiled | NA | NA | NA | failed |
| Ascend910B2 | matmul/M768_qkv_eager_nd_internal_on | eager | 0.225 | 0.095 | NA | completed |
| Ascend910B2 | matmul/M768_qkv_eager_nz_internal_on | eager | 0.241 | 0.101 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on | compiled | 0.503 | 0.359 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on | compiled | 0.502 | 0.355 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on | compiled | 0.515 | 0.369 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on | compiled | 0.500 | 0.357 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nd_internal_on | compiled | 0.506 | 0.350 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nz_internal_on | compiled | 0.516 | 0.364 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on | compiled | 0.523 | 0.369 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on | compiled | 0.536 | 0.376 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on | compiled | 0.549 | 0.396 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on | compiled | 0.559 | 0.418 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on | compiled | 0.552 | 0.408 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on | compiled | 0.581 | 0.439 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nd_internal_on | compiled | 0.515 | 0.360 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nz_internal_on | compiled | 0.492 | 0.350 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on | compiled | 0.542 | 0.393 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on | compiled | 0.503 | 0.366 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nd_internal_on | compiled | 0.505 | 0.355 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nz_internal_on | compiled | 0.494 | 0.347 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nd_internal_on | compiled | 0.495 | 0.351 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nz_internal_on | compiled | 0.562 | 0.376 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nd_internal_on | compiled | 0.506 | 0.350 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nz_internal_on | compiled | 0.482 | 0.330 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nd_internal_on | compiled | 0.501 | 0.353 | NA | completed |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nz_internal_on | compiled | 0.507 | 0.354 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nd_internal_on | eager | 0.224 | 0.086 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nz_internal_on | eager | 0.236 | 0.093 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nd_internal_on | eager | 0.221 | 0.090 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nz_internal_on | eager | 0.239 | 0.098 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nd_internal_on | eager | 0.227 | 0.091 | NA | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nz_internal_on | eager | 0.244 | 0.098 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nd_internal_on | eager | 0.292 | 0.159 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nz_internal_on | eager | 0.306 | 0.168 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nd_internal_on | eager | 0.333 | 0.164 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nz_internal_on | eager | 0.306 | 0.169 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nd_internal_on | eager | 0.243 | 0.093 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nz_internal_on | eager | 0.247 | 0.106 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nd_internal_on | eager | 0.288 | 0.139 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nz_internal_on | eager | 0.282 | 0.142 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nd_internal_on | eager | 0.403 | 0.261 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nz_internal_on | eager | 0.416 | 0.267 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nd_internal_on | eager | 0.403 | 0.283 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nz_internal_on | eager | 0.426 | 0.285 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nd_internal_on | eager | 0.227 | 0.102 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nz_internal_on | eager | 0.258 | 0.111 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nd_internal_on | eager | 0.329 | 0.201 | NA | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nz_internal_on | eager | 0.371 | 0.225 | NA | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off | compiled | 0.488 | 0.338 | NA | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nz_internal_off | compiled | NA | NA | NA | unsupported_configuration |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nd_internal_off | eager | 0.217 | 0.092 | NA | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nz_internal_off | eager | NA | NA | NA | unsupported_configuration |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | torchair_fullgraph | 18.733 | 18.403 | 38435.221 | completed |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | torchair_fullgraph | 50.177 | 49.699 | 60505.255 | completed |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | torchair_fullgraph | 18.948 | 18.658 | 37999.668 | completed |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | torchair_fullgraph | 50.040 | 49.749 | 60671.856 | completed |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | torchair_fullgraph | 20.040 | 19.759 | 35928.258 | completed |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | torchair_fullgraph | 51.040 | 50.756 | 59483.122 | completed |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | torchair_fullgraph | 18.243 | 17.950 | 39467.076 | completed |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | torchair_fullgraph | 50.006 | 49.604 | 60712.918 | completed |
| unreported | capture.receipt | None | NA | NA | NA | completed |

| Chip | Lane | Bucket/type | Calls/forward | Kernel ms/forward | Wait ms/forward | Block Num histogram | Mix Block Num | Accelerator Core |
|---|---|---|---:|---:|---:|---|---|---|
| Ascend910B2 | matmul/M768_qkv_eager_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.032 | 0.173 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul/M768_qkv_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.173 | — | — | — |
| Ascend910B2 | matmul/M768_qkv_eager_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.035 | 0.211 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul/M768_qkv_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.211 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.129 | 0.291 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.291 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.145 | 0.277 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.277 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.126 | 0.335 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.335 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.145 | 0.287 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.287 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.067 | 0.341 | {'21': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.341 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.072 | 0.353 | {'21': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.353 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.099 | 0.317 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.317 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.113 | 0.318 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.318 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on | remaining: Data | 0.3 | 0.002 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.226 | 0.250 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.250 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.259 | 0.247 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.247 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.241 | 0.248 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.248 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.264 | 0.308 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.308 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.072 | 0.331 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.331 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.071 | 0.326 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.326 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.176 | 0.270 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.270 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.194 | 0.269 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.269 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.039 | 0.368 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.368 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nz_internal_on | remaining: Data | 0.3 | 0.002 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.044 | 0.366 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.366 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.045 | 0.374 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.374 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.046 | 0.404 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.404 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.021 | 0.350 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.350 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.030 | 0.340 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.340 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nd_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.033 | 0.342 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.342 | — | — | — |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nz_internal_on | remaining: Data | 0.3 | 0.001 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.036 | 0.379 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.379 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.038 | 0.161 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.161 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.039 | 0.213 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.213 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.045 | 0.184 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.184 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.045 | 0.203 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.203 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.018 | 0.177 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.177 | — | — | — |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.019 | 0.239 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.239 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.124 | 0.304 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.304 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.127 | 0.209 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.209 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.126 | 0.243 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.243 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.127 | 0.201 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.201 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nd_internal_on | matmul: MatMulV2 | 1.0 | 0.066 | 0.172 | {'21': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.172 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nz_internal_on | matmul: MatMulV2 | 1.0 | 0.068 | 0.199 | {'21': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.199 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.098 | 0.195 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.195 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.098 | 0.218 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.218 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.225 | 0.185 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.185 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.226 | 0.286 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.286 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.237 | 0.180 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.180 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.238 | 0.234 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.234 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.068 | 0.178 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.178 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.067 | 0.200 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.200 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nd_internal_on | matmul: MatMulV3 | 1.0 | 0.172 | 0.383 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nd_internal_on | **TOTAL WAIT (separate)** | — | — | 0.383 | — | — | — |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nz_internal_on | matmul: MatMulV3 | 1.0 | 0.174 | 0.203 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nz_internal_on | **TOTAL WAIT (separate)** | — | — | 0.203 | — | — | — |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off | remaining: Data | 0.3 | 0.002 | 0.000 | {'1': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off | matmul: MatMulV2 | 1.0 | 0.032 | 0.353 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off | **TOTAL WAIT (separate)** | — | — | 0.353 | — | — | — |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nd_internal_off | matmul: MatMulV2 | 1.0 | 0.032 | 0.170 | {'24': 3} | {'0': 3} | {'AI_CORE': 3} |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nd_internal_off | **TOTAL WAIT (separate)** | — | — | 0.170 | — | — | — |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Add | 128.0 | 1.285 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: AutomaticBufferFusionOp | 160.0 | 1.106 | 0.007 | {'1': 192, '32': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Cast | 65.0 | 0.646 | 0.596 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: ConcatV2D | 64.0 | 0.581 | 0.004 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Data | 0.3 | 0.002 | 0.000 | {'4': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | matmul: MatMulV2 | 128.0 | 4.593 | 0.008 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Mul | 192.0 | 1.635 | 0.010 | {'48': 192, '43': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Neg | 64.0 | 0.586 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | attention: PromptFlashAttention | 32.0 | 2.660 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: ReduceMeanD | 128.0 | 0.963 | 0.006 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: StridedSliceD | 128.0 | 1.886 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Sub | 64.0 | 0.739 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Transpose | 128.0 | 1.216 | 0.037 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | remaining: Unpack | 32.0 | 0.277 | 0.002 | {'16': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/baseline_crop_0_bucket_768 | **TOTAL WAIT (separate)** | — | — | 0.728 | — | — | — |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Add | 128.0 | 1.875 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: AutomaticBufferFusionOp | 160.0 | 2.160 | 0.009 | {'3': 192, '43': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Cast | 65.0 | 0.848 | 0.457 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: ConcatV2D | 64.0 | 0.878 | 0.004 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Data | 0.3 | 0.002 | 0.000 | {'4': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | matmul: MatMulV2 | 32.0 | 2.118 | 0.002 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | matmul: MatMulV3 | 96.0 | 11.501 | 0.005 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Mul | 192.0 | 2.632 | 0.011 | {'48': 192, '47': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Neg | 64.0 | 0.691 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | attention: PromptFlashAttention | 32.0 | 16.674 | 0.039 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: ReduceMeanD | 128.0 | 1.747 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: StridedSliceD | 128.0 | 4.557 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Sub | 64.0 | 1.202 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Transpose | 128.0 | 1.798 | 0.036 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | remaining: Unpack | 32.0 | 0.689 | 0.002 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/baseline_crop_1_bucket_3072 | **TOTAL WAIT (separate)** | — | — | 0.591 | — | — | — |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Add | 128.0 | 1.218 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: AutomaticBufferFusionOp | 160.0 | 1.145 | 0.008 | {'1': 192, '32': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Cast | 65.0 | 0.594 | 0.968 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: ConcatV2D | 64.0 | 0.599 | 0.005 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: ConfusionTransposeD | 32.0 | 0.180 | 0.032 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Data | 0.3 | 0.002 | 0.000 | {'7': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | matmul: GroupedMatmul | 32.0 | 1.007 | 0.037 | {'24': 96} | {'0': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | matmul: MatMulV2 | 96.0 | 3.434 | 0.006 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Mul | 192.0 | 1.584 | 0.012 | {'48': 192, '43': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Neg | 64.0 | 0.586 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | attention: PromptFlashAttention | 32.0 | 2.661 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: ReduceMeanD | 128.0 | 0.893 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: StridedSliceD | 128.0 | 1.869 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Sub | 64.0 | 0.678 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Transpose | 128.0 | 1.821 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | remaining: Unpack | 32.0 | 0.281 | 0.033 | {'16': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_0_bucket_768 | **TOTAL WAIT (separate)** | — | — | 1.173 | — | — | — |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Add | 128.0 | 1.797 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: AutomaticBufferFusionOp | 160.0 | 2.214 | 0.009 | {'3': 192, '43': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Cast | 65.0 | 0.778 | 0.362 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: ConcatV2D | 64.0 | 0.907 | 0.003 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: ConfusionTransposeD | 32.0 | 0.331 | 0.032 | {'41': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Data | 0.3 | 0.002 | 0.000 | {'7': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | matmul: GroupedMatmul | 32.0 | 3.081 | 0.038 | {'24': 96} | {'0': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | matmul: MatMulV2 | 32.0 | 2.119 | 0.002 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | matmul: MatMulV3 | 64.0 | 8.272 | 0.003 | {'24': 192} | {'0': 192} | {'AI_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Mul | 192.0 | 2.643 | 0.011 | {'48': 192, '47': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Neg | 64.0 | 0.688 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | attention: PromptFlashAttention | 32.0 | 16.699 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: ReduceMeanD | 128.0 | 1.657 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: StridedSliceD | 128.0 | 4.483 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Sub | 64.0 | 1.127 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Transpose | 128.0 | 2.145 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | remaining: Unpack | 32.0 | 0.702 | 0.032 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_crop_1_bucket_3072 | **TOTAL WAIT (separate)** | — | — | 0.567 | — | — | — |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Add | 128.0 | 1.214 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: AutomaticBufferFusionOp | 160.0 | 1.145 | 0.040 | {'1': 192, '32': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Cast | 65.0 | 0.588 | 0.378 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: ConcatV2D | 64.0 | 0.606 | 0.004 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: ConfusionTransposeD | 32.0 | 0.182 | 0.032 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Data | 0.3 | 0.002 | 0.000 | {'9': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | matmul: GroupedMatmul | 64.0 | 2.309 | 0.076 | {'24': 192} | {'0': 192} | {'MIX_AIC': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | matmul: MatMulV2 | 64.0 | 2.140 | 0.004 | {'24': 192} | {'0': 192} | {'AI_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Mul | 192.0 | 1.600 | 0.012 | {'48': 192, '43': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Neg | 64.0 | 0.589 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | attention: PromptFlashAttention | 32.0 | 2.689 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: ReduceMeanD | 128.0 | 0.960 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: StridedSliceD | 128.0 | 1.865 | 0.006 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Sub | 64.0 | 0.667 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Transpose | 160.0 | 2.799 | 0.010 | {'48': 480} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | remaining: Unpack | 32.0 | 0.283 | 0.032 | {'16': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | **TOTAL WAIT (separate)** | — | — | 0.653 | — | — | — |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Add | 128.0 | 1.791 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: AutomaticBufferFusionOp | 160.0 | 2.203 | 0.038 | {'3': 192, '43': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Cast | 65.0 | 0.778 | 0.510 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: ConcatV2D | 64.0 | 0.906 | 0.004 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: ConfusionTransposeD | 32.0 | 0.329 | 0.031 | {'41': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Data | 0.3 | 0.002 | 0.000 | {'9': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | matmul: GroupedMatmul | 64.0 | 7.068 | 0.075 | {'24': 192} | {'0': 192} | {'MIX_AIC': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | matmul: MatMulV2 | 32.0 | 2.119 | 0.001 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | matmul: MatMulV3 | 32.0 | 4.107 | 0.002 | {'24': 96} | {'0': 96} | {'AI_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Mul | 192.0 | 2.655 | 0.010 | {'48': 192, '47': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Neg | 64.0 | 0.690 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | attention: PromptFlashAttention | 32.0 | 16.716 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: ReduceMeanD | 128.0 | 1.665 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: StridedSliceD | 128.0 | 4.496 | 0.008 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Sub | 64.0 | 1.129 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Transpose | 160.0 | 3.217 | 0.009 | {'48': 480} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | remaining: Unpack | 32.0 | 0.702 | 0.032 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | **TOTAL WAIT (separate)** | — | — | 0.779 | — | — | — |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Add | 128.0 | 1.274 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: AutomaticBufferFusionOp | 160.0 | 1.101 | 0.009 | {'1': 192, '32': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Cast | 65.0 | 0.644 | 0.464 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: ConcatV2D | 64.0 | 0.578 | 0.005 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: ConfusionTransposeD | 32.0 | 0.176 | 0.031 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Data | 0.3 | 0.002 | 0.000 | {'4': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | matmul: MatMulV2 | 128.0 | 4.565 | 0.007 | {'24': 384} | {'0': 384} | {'AI_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Mul | 192.0 | 1.631 | 0.009 | {'48': 192, '43': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Neg | 64.0 | 0.586 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | attention: PromptFlashAttention | 32.0 | 2.650 | 0.036 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: ReduceMeanD | 128.0 | 0.959 | 0.006 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: StridedSliceD | 128.0 | 1.887 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Sub | 64.0 | 0.739 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Transpose | 96.0 | 0.905 | 0.006 | {'48': 288} | {'0': 288} | {'AI_VECTOR_CORE': 288} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | remaining: Unpack | 32.0 | 0.281 | 0.001 | {'16': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_0_bucket_768 | **TOTAL WAIT (separate)** | — | — | 0.596 | — | — | — |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Add | 128.0 | 1.862 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: AutomaticBufferFusionOp | 160.0 | 2.164 | 0.010 | {'3': 192, '43': 192, '48': 96} | {'0': 480} | {'AI_VECTOR_CORE': 480} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Cast | 65.0 | 0.841 | 0.562 | {'48': 195} | {'0': 195} | {'AI_VECTOR_CORE': 195} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: ConcatV2D | 64.0 | 0.867 | 0.003 | {'41': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: ConfusionTransposeD | 32.0 | 0.361 | 0.032 | {'41': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Data | 0.3 | 0.002 | 0.000 | {'4': 1} | {'0': 1} | {'AI_VECTOR_CORE': 1} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | matmul: MatMulV2 | 32.0 | 2.122 | 0.002 | {'21': 96} | {'0': 96} | {'AI_CORE': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | matmul: MatMulV3 | 96.0 | 11.531 | 0.005 | {'24': 288} | {'0': 288} | {'AI_CORE': 288} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Mul | 192.0 | 2.631 | 0.010 | {'48': 192, '47': 384} | {'0': 576} | {'AI_VECTOR_CORE': 576} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Neg | 64.0 | 0.685 | 0.004 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | attention: PromptFlashAttention | 32.0 | 16.849 | 0.038 | {'24': 96} | {'48': 96} | {'MIX_AIC': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: ReduceMeanD | 128.0 | 1.733 | 0.007 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: StridedSliceD | 128.0 | 4.528 | 0.008 | {'48': 384} | {'0': 384} | {'AI_VECTOR_CORE': 384} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Sub | 64.0 | 1.203 | 0.003 | {'48': 192} | {'0': 192} | {'AI_VECTOR_CORE': 192} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Transpose | 96.0 | 1.379 | 0.006 | {'48': 288} | {'0': 288} | {'AI_VECTOR_CORE': 288} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | remaining: Unpack | 32.0 | 0.690 | 0.002 | {'21': 96} | {'0': 96} | {'AI_VECTOR_CORE': 96} |
| Ascend910B2 | vision_matrix/internal_format_opposite_crop_1_bucket_3072 | **TOTAL WAIT (separate)** | — | — | 0.699 | — | — | — |

**Synthetic calibration only — not model performance.**

| Chip | Lane / M,K,N | Execution | Requested/actual weight format | Internal format | Matmul types / Block Num | Kernel us/forward | TFLOPS | Status |
|---|---|---|---|---|---|---:|---:|---|
| Ascend910B2 | matmul/M768_qkv_compiled_nd_internal_on / {'M': 768, 'K': 1280, 'N': 3840} | compiled | nd/None | True | {} | NA | NA | failed |
| Ascend910B2 | matmul/M768_qkv_eager_nd_internal_on / {'M': 768, 'K': 1280, 'N': 3840} | eager | nd/2 | True | {'MatMulV2': {'24': 3}} | 31.574 | 239.115 | completed |
| Ascend910B2 | matmul/M768_qkv_eager_nz_internal_on / {'M': 768, 'K': 1280, 'N': 3840} | eager | nz/29 | True | {'MatMulV2': {'24': 3}} | 34.628 | 218.026 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 5120} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 128.536 | 313.261 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 5120} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 144.529 | 278.596 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on / {'M': 3072, 'K': 5120, 'N': 1280} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 125.662 | 320.425 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on / {'M': 3072, 'K': 5120, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 144.836 | 278.006 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 1280} | compiled | nd/2 | True | {'MatMulV2': {'21': 3}} | 67.229 | 149.733 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_proj_compiled_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'21': 3}} | 71.981 | 139.846 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 3840} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 99.055 | 304.870 | completed |
| Ascend910B2 | matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 3840} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 113.016 | 267.211 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 5120} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 226.058 | 326.553 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 5120} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 258.579 | 285.483 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on / {'M': 5632, 'K': 5120, 'N': 1280} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 240.625 | 306.784 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on / {'M': 5632, 'K': 5120, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 264.452 | 279.142 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 1280} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 71.514 | 258.059 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_proj_compiled_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 71.421 | 258.397 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 3840} | compiled | nd/2 | True | {'MatMulV3': {'24': 3}} | 175.810 | 314.913 | completed |
| Ascend910B2 | matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 3840} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 193.657 | 285.891 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nd_internal_on / {'M': 768, 'K': 1280, 'N': 5120} | compiled | nd/2 | True | {'MatMulV2': {'24': 3}} | 38.854 | 259.081 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc1_compiled_nz_internal_on / {'M': 768, 'K': 1280, 'N': 5120} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 43.794 | 229.856 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nd_internal_on / {'M': 768, 'K': 5120, 'N': 1280} | compiled | nd/2 | True | {'MatMulV2': {'24': 3}} | 44.534 | 226.035 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_fc2_compiled_nz_internal_on / {'M': 768, 'K': 5120, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 45.668 | 220.426 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nd_internal_on / {'M': 768, 'K': 1280, 'N': 1280} | compiled | nd/2 | True | {'MatMulV2': {'24': 3}} | 21.134 | 119.079 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_proj_compiled_nz_internal_on / {'M': 768, 'K': 1280, 'N': 1280} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 29.854 | 84.296 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nd_internal_on / {'M': 768, 'K': 1280, 'N': 3840} | compiled | nd/2 | True | {'MatMulV2': {'24': 3}} | 33.201 | 227.397 | completed |
| Ascend910B2 | matmul_compiled_retry/M768_qkv_compiled_nz_internal_on / {'M': 768, 'K': 1280, 'N': 3840} | compiled | nz/29 | True | {'MatMulV2': {'24': 3}} | 35.907 | 210.256 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nd_internal_on / {'M': 768, 'K': 1280, 'N': 5120} | eager | nd/2 | True | {'MatMulV2': {'24': 3}} | 37.714 | 266.910 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc1_eager_nz_internal_on / {'M': 768, 'K': 1280, 'N': 5120} | eager | nz/29 | True | {'MatMulV2': {'24': 3}} | 38.614 | 260.693 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nd_internal_on / {'M': 768, 'K': 5120, 'N': 1280} | eager | nd/2 | True | {'MatMulV2': {'24': 3}} | 45.301 | 222.212 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_fc2_eager_nz_internal_on / {'M': 768, 'K': 5120, 'N': 1280} | eager | nz/29 | True | {'MatMulV2': {'24': 3}} | 45.367 | 221.885 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nd_internal_on / {'M': 768, 'K': 1280, 'N': 1280} | eager | nd/2 | True | {'MatMulV2': {'24': 3}} | 18.014 | 139.701 | completed |
| Ascend910B2 | matmul_eager_768_remaining/M768_proj_eager_nz_internal_on / {'M': 768, 'K': 1280, 'N': 1280} | eager | nz/29 | True | {'MatMulV2': {'24': 3}} | 18.814 | 133.764 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 5120} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 124.389 | 323.705 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc1_eager_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 5120} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 127.089 | 316.827 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nd_internal_on / {'M': 3072, 'K': 5120, 'N': 1280} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 126.303 | 318.799 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_fc2_eager_nz_internal_on / {'M': 3072, 'K': 5120, 'N': 1280} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 127.095 | 316.812 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 1280} | eager | nd/2 | True | {'MatMulV2': {'21': 3}} | 66.334 | 151.751 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_proj_eager_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 1280} | eager | nz/29 | True | {'MatMulV2': {'21': 3}} | 68.341 | 147.296 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nd_internal_on / {'M': 3072, 'K': 1280, 'N': 3840} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 98.348 | 307.062 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M3072_qkv_eager_nz_internal_on / {'M': 3072, 'K': 1280, 'N': 3840} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 97.809 | 308.756 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 5120} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 224.978 | 328.120 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc1_eager_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 5120} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 226.464 | 325.966 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nd_internal_on / {'M': 5632, 'K': 5120, 'N': 1280} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 237.112 | 311.329 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_fc2_eager_nz_internal_on / {'M': 5632, 'K': 5120, 'N': 1280} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 238.372 | 309.683 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 1280} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 67.795 | 272.218 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_proj_eager_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 1280} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 67.208 | 274.596 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nd_internal_on / {'M': 5632, 'K': 1280, 'N': 3840} | eager | nd/2 | True | {'MatMulV3': {'24': 3}} | 172.324 | 321.283 | completed |
| Ascend910B2 | matmul_eager_large_remaining/M5632_qkv_eager_nz_internal_on / {'M': 5632, 'K': 1280, 'N': 3840} | eager | nz/29 | True | {'MatMulV3': {'24': 3}} | 173.511 | 319.086 | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off / {'M': 768, 'K': 1280, 'N': 3840} | compiled | nd/2 | False | {'MatMulV2': {'24': 3}} | 32.481 | 232.438 | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_compiled_nz_internal_off / {'M': 768, 'K': 1280, 'N': 3840} | compiled | nz/None | False | {} | NA | NA | unsupported_configuration |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nd_internal_off / {'M': 768, 'K': 1280, 'N': 3840} | eager | nd/2 | False | {'MatMulV2': {'24': 3}} | 31.707 | 238.107 | completed |
| Ascend910B2 | matmul_internal_off_validation/M768_qkv_eager_nz_internal_off / {'M': 768, 'K': 1280, 'N': 3840} | eager | nz/None | False | {} | NA | NA | unsupported_configuration |

| Lane | Physical card | Before/after UTC | Before/after load average | CPUs / affinity | Device errors | Clock data |
|---|---|---|---|---|---|---|
| matmul/M768_qkv_compiled_nd_internal_on | 2 | 2026-10-07T12:51:55.584122+00:00 / 2026-10-07T12:52:14.130991+00:00 | [22.2958984375, 23.0087890625, 23.67431640625] / [21.8857421875, 22.87548828125, 23.61572265625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul/M768_qkv_eager_nd_internal_on | 2 | 2026-10-07T12:50:47.787318+00:00 / 2026-10-07T12:51:15.523561+00:00 | [23.2177734375, 23.31201171875, 23.818359375] / [22.4052734375, 23.10595703125, 23.73291015625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul/M768_qkv_eager_nz_internal_on | 2 | 2026-10-07T12:51:19.775824+00:00 / 2026-10-07T12:51:51.314908+00:00 | [22.29248046875, 23.07080078125, 23.7177734375] / [22.4091796875, 23.04296875, 23.68896484375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_fc1_compiled_nd_internal_on | 2 | 2026-10-07T13:04:01.466324+00:00 / 2026-10-07T13:04:34.179231+00:00 | [25.4345703125, 25.5810546875, 24.583984375] / [25.48974609375, 25.57958984375, 24.6220703125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_fc1_compiled_nz_internal_on | 2 | 2026-10-07T13:04:38.515476+00:00 / 2026-10-07T13:05:10.660682+00:00 | [25.48974609375, 25.57958984375, 24.6220703125] / [25.8623046875, 25.67138671875, 24.68896484375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_fc2_compiled_nd_internal_on | 2 | 2026-10-07T13:05:14.963865+00:00 / 2026-10-07T13:05:47.861813+00:00 | [25.95361328125, 25.69384765625, 24.70166015625] / [26.7177734375, 25.92578125, 24.8125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_fc2_compiled_nz_internal_on | 2 | 2026-10-07T13:05:52.160669+00:00 / 2026-10-07T13:06:24.816257+00:00 | [26.580078125, 25.91015625, 24.81396484375] / [25.81884765625, 25.8193359375, 24.826171875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_proj_compiled_nd_internal_on | 2 | 2026-10-07T13:02:47.409988+00:00 / 2026-10-07T13:03:18.949471+00:00 | [26.2724609375, 25.57666015625, 24.48974609375] / [27.02734375, 25.84521484375, 24.62060546875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_proj_compiled_nz_internal_on | 2 | 2026-10-07T13:03:23.224295+00:00 / 2026-10-07T13:03:57.184259+00:00 | [27.02734375, 25.84521484375, 24.62060546875] / [25.47265625, 25.59130859375, 24.58154296875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_qkv_compiled_nd_internal_on | 2 | 2026-10-07T13:01:28.661281+00:00 / 2026-10-07T13:02:03.348374+00:00 | [27.1064453125, 25.365234375, 24.330078125] / [26.5673828125, 25.43310546875, 24.3876953125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M3072_qkv_compiled_nz_internal_on | 2 | 2026-10-07T13:02:07.662746+00:00 / 2026-10-07T13:02:43.169231+00:00 | [26.84228515625, 25.50927734375, 24.41796875] / [26.99267578125, 25.70458984375, 24.52490234375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_fc1_compiled_nd_internal_on | 2 | 2026-10-07T13:09:35.034927+00:00 / 2026-10-07T13:10:21.043663+00:00 | [24.3603515625, 25.3564453125, 24.86181640625] / [25.24072265625, 25.49853515625, 24.935546875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_fc1_compiled_nz_internal_on | 2 | 2026-10-07T13:10:25.347720+00:00 / 2026-10-07T13:11:03.306950+00:00 | [24.98095703125, 25.4404296875, 24.91943359375] / [26.17236328125, 25.71826171875, 25.033203125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_fc2_compiled_nd_internal_on | 2 | 2026-10-07T13:11:07.621048+00:00 / 2026-10-07T13:11:55.377459+00:00 | [25.7578125, 25.6396484375, 25.01123046875] / [26.83447265625, 25.88330078125, 25.12353515625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_fc2_compiled_nz_internal_on | 2 | 2026-10-07T13:11:59.651265+00:00 / 2026-10-07T13:12:40.701450+00:00 | [26.52734375, 25.8349609375, 25.11181640625] / [25.64111328125, 25.7353515625, 25.1103515625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_proj_compiled_nd_internal_on | 2 | 2026-10-07T13:08:00.554182+00:00 / 2026-10-07T13:08:49.773397+00:00 | [24.65380859375, 25.705078125, 24.90380859375] / [24.6083984375, 25.544921875, 24.89306640625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_proj_compiled_nz_internal_on | 2 | 2026-10-07T13:08:54.031888+00:00 / 2026-10-07T13:09:30.760494+00:00 | [24.64013671875, 25.53564453125, 24.89404296875] / [24.6533203125, 25.43017578125, 24.8828125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_qkv_compiled_nd_internal_on | 2 | 2026-10-07T13:06:29.106595+00:00 / 2026-10-07T13:07:14.103322+00:00 | [26.15380859375, 25.88916015625, 24.85400390625] / [26.85302734375, 26.1572265625, 24.9990234375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M5632_qkv_compiled_nz_internal_on | 2 | 2026-10-07T13:07:18.398825+00:00 / 2026-10-07T13:07:56.249204+00:00 | [26.85302734375, 26.1572265625, 24.9990234375] / [24.62353515625, 25.71728515625, 24.90283203125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_fc1_compiled_nd_internal_on | 2 | 2026-10-07T12:58:55.738426+00:00 / 2026-10-07T12:59:28.248511+00:00 | [26.1396484375, 24.326171875, 23.8720703125] / [26.55810546875, 24.5732421875, 23.9677734375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_fc1_compiled_nz_internal_on | 2 | 2026-10-07T12:59:32.578892+00:00 / 2026-10-07T13:00:08.921361+00:00 | [26.75390625, 24.64697265625, 23.9951171875] / [25.8935546875, 24.69091796875, 24.03662109375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_fc2_compiled_nd_internal_on | 2 | 2026-10-07T13:00:13.212438+00:00 / 2026-10-07T13:00:44.891524+00:00 | [25.8935546875, 24.69091796875, 24.03662109375] / [26.5693359375, 25.0068359375, 24.16845703125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_fc2_compiled_nz_internal_on | 2 | 2026-10-07T13:00:49.178916+00:00 / 2026-10-07T13:01:24.375330+00:00 | [26.36328125, 24.98974609375, 24.16748046875] / [27.2900390625, 25.37158203125, 24.326171875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_proj_compiled_nd_internal_on | 2 | 2026-10-07T12:57:41.144137+00:00 / 2026-10-07T12:58:14.257469+00:00 | [25.4423828125, 23.53369140625, 23.58837890625] / [26.60009765625, 24.080078125, 23.7724609375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_proj_compiled_nz_internal_on | 2 | 2026-10-07T12:58:18.579283+00:00 / 2026-10-07T12:58:51.440488+00:00 | [26.5517578125, 24.1123046875, 23.78466796875] / [26.67431640625, 24.3994140625, 23.89306640625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_qkv_compiled_nd_internal_on | 2 | 2026-10-07T12:56:27.660738+00:00 / 2026-10-07T12:56:59.338319+00:00 | [22.21240234375, 22.5283203125, 23.29150390625] / [24.51708984375, 23.10888671875, 23.46044921875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_compiled_retry/M768_qkv_compiled_nz_internal_on | 2 | 2026-10-07T12:57:03.648921+00:00 / 2026-10-07T12:57:36.817576+00:00 | [24.55615234375, 23.140625, 23.46875] / [26.00341796875, 23.6103515625, 23.61328125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_fc1_eager_nd_internal_on | 2 | 2026-10-07T13:13:48.874729+00:00 / 2026-10-07T13:14:12.290987+00:00 | [25.166015625, 25.53955078125, 25.08251953125] / [24.80224609375, 25.443359375, 25.0634765625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_fc1_eager_nz_internal_on | 2 | 2026-10-07T13:14:16.544649+00:00 / 2026-10-07T13:14:47.267462+00:00 | [24.65771484375, 25.40234375, 25.05224609375] / [24.83154296875, 25.40380859375, 25.0654296875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_fc2_eager_nd_internal_on | 2 | 2026-10-07T13:14:51.590286+00:00 / 2026-10-07T13:15:15.429881+00:00 | [25.16552734375, 25.4638671875, 25.0869140625] / [24.75830078125, 25.345703125, 25.05810546875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_fc2_eager_nz_internal_on | 2 | 2026-10-07T13:15:19.798511+00:00 / 2026-10-07T13:15:52.344119+00:00 | [24.77783203125, 25.33984375, 25.0576171875] / [27.60498046875, 25.98974609375, 25.28466796875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_proj_eager_nd_internal_on | 2 | 2026-10-07T13:12:45.103069+00:00 / 2026-10-07T13:13:09.007082+00:00 | [25.34912109375, 25.6728515625, 25.09326171875] / [24.818359375, 25.53466796875, 25.06298828125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_768_remaining/M768_proj_eager_nz_internal_on | 2 | 2026-10-07T13:13:13.310905+00:00 / 2026-10-07T13:13:44.553153+00:00 | [24.818359375, 25.53466796875, 25.06298828125] / [25.166015625, 25.53955078125, 25.08251953125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_fc1_eager_nd_internal_on | 2 | 2026-10-07T13:18:04.202636+00:00 / 2026-10-07T13:18:27.600984+00:00 | [27.23193359375, 26.57958984375, 25.6083984375] / [27.388671875, 26.67578125, 25.66748046875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_fc1_eager_nz_internal_on | 2 | 2026-10-07T13:18:31.914349+00:00 / 2026-10-07T13:19:02.677151+00:00 | [27.1171875, 26.63134765625, 25.658203125] / [29.31884765625, 27.23486328125, 25.89208984375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_fc2_eager_nd_internal_on | 2 | 2026-10-07T13:19:06.974925+00:00 / 2026-10-07T13:19:30.645021+00:00 | [29.37353515625, 27.28125, 25.91455078125] / [28.39697265625, 27.2373046875, 25.93798828125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_fc2_eager_nz_internal_on | 2 | 2026-10-07T13:19:34.908995+00:00 / 2026-10-07T13:20:06.391193+00:00 | [28.765625, 27.3330078125, 25.97607421875] / [28.271484375, 27.3603515625, 26.029296875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_proj_eager_nd_internal_on | 2 | 2026-10-07T13:17:01.510570+00:00 / 2026-10-07T13:17:24.928013+00:00 | [28.53955078125, 26.5830078125, 25.54150390625] / [27.94384765625, 26.6005859375, 25.576171875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_proj_eager_nz_internal_on | 2 | 2026-10-07T13:17:29.205369+00:00 / 2026-10-07T13:17:59.929968+00:00 | [27.94873046875, 26.6240234375, 25.58935546875] / [27.23193359375, 26.57958984375, 25.6083984375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_qkv_eager_nd_internal_on | 2 | 2026-10-07T13:15:56.746721+00:00 / 2026-10-07T13:16:22.147782+00:00 | [27.796875, 26.056640625, 25.31005859375] / [27.1552734375, 26.076171875, 25.337890625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M3072_qkv_eager_nz_internal_on | 2 | 2026-10-07T13:16:26.458664+00:00 / 2026-10-07T13:16:57.237655+00:00 | [26.90234375, 26.04150390625, 25.33056640625] / [28.673828125, 26.57568359375, 25.533203125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_fc1_eager_nd_internal_on | 2 | 2026-10-07T13:22:19.729698+00:00 / 2026-10-07T13:22:43.657660+00:00 | [28.166015625, 27.6025390625, 26.30126953125] / [27.5966796875, 27.51318359375, 26.30029296875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_fc1_eager_nz_internal_on | 2 | 2026-10-07T13:22:47.988974+00:00 / 2026-10-07T13:23:18.855459+00:00 | [27.46875, 27.48779296875, 26.29833984375] / [27.166015625, 27.43505859375, 26.32080078125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_fc2_eager_nd_internal_on | 2 | 2026-10-07T13:23:23.122687+00:00 / 2026-10-07T13:23:54.729419+00:00 | [27.23291015625, 27.44482421875, 26.330078125] / [26.4892578125, 27.2607421875, 26.310546875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_fc2_eager_nz_internal_on | 2 | 2026-10-07T13:23:59.033935+00:00 / 2026-10-07T13:24:33.162519+00:00 | [26.4892578125, 27.2607421875, 26.310546875] / [25.96484375, 27.09375, 26.2919921875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_proj_eager_nd_internal_on | 2 | 2026-10-07T13:21:14.949055+00:00 / 2026-10-07T13:21:39.803022+00:00 | [28.7978515625, 27.69189453125, 26.24169921875] / [27.2060546875, 27.40966796875, 26.18603515625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_proj_eager_nz_internal_on | 2 | 2026-10-07T13:21:44.191145+00:00 / 2026-10-07T13:22:15.421019+00:00 | [27.2060546875, 27.40966796875, 26.18603515625] / [28.00634765625, 27.56201171875, 26.28125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_qkv_eager_nd_internal_on | 2 | 2026-10-07T13:20:10.657957+00:00 / 2026-10-07T13:20:34.619062+00:00 | [28.41015625, 27.404296875, 26.05078125] / [28.3466796875, 27.48193359375, 26.11328125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_eager_large_remaining/M5632_qkv_eager_nz_internal_on | 2 | 2026-10-07T13:20:38.903003+00:00 / 2026-10-07T13:21:10.645222+00:00 | [28.3466796875, 27.48193359375, 26.11328125] / [28.431640625, 27.60205078125, 26.205078125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_internal_off_validation/M768_qkv_compiled_nd_internal_off | 1 | 2026-10-07T13:13:02.503150+00:00 / 2026-10-07T13:13:37.931244+00:00 | [25.34619140625, 25.65625, 25.0966796875] / [25.02197265625, 25.52392578125, 25.072265625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_internal_off_validation/M768_qkv_compiled_nz_internal_off | 1 | 2026-10-07T13:13:42.206915+00:00 / 2026-10-07T13:13:58.503972+00:00 | [25.1806640625, 25.548828125, 25.0830078125] / [25.44189453125, 25.587890625, 25.103515625] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_internal_off_validation/M768_qkv_eager_nd_internal_off | 1 | 2026-10-07T13:12:13.112329+00:00 / 2026-10-07T13:12:36.701016+00:00 | [25.89794921875, 25.72412109375, 25.0830078125] / [26.04541015625, 25.8154296875, 25.1328125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| matmul_internal_off_validation/M768_qkv_eager_nz_internal_off | 1 | 2026-10-07T13:12:40.997841+00:00 / 2026-10-07T13:12:58.211595+00:00 | [25.64111328125, 25.7353515625, 25.1103515625] / [25.2890625, 25.650390625, 25.091796875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/baseline_crop_0_bucket_768 | 1 | 2026-10-07T12:54:51.344214+00:00 / 2026-10-07T12:55:54.547500+00:00 | [22.615234375, 22.6826171875, 23.42138671875] / [22.67529296875, 22.63720703125, 23.3505859375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/baseline_crop_1_bucket_3072 | 1 | 2026-10-07T12:55:58.833800+00:00 / 2026-10-07T12:56:52.250648+00:00 | [22.62109375, 22.62646484375, 23.34326171875] / [24.6826171875, 23.09521484375, 23.4599609375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/grouped_qkv_crop_0_bucket_768 | 1 | 2026-10-07T13:00:11.687118+00:00 / 2026-10-07T13:02:04.776573+00:00 | [25.8935546875, 24.69091796875, 24.03662109375] / [26.84228515625, 25.50927734375, 24.41796875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/grouped_qkv_crop_1_bucket_3072 | 1 | 2026-10-07T13:02:09.077125+00:00 / 2026-10-07T13:03:48.915488+00:00 | [26.93505859375, 25.55078125, 24.4375] / [25.33935546875, 25.5673828125, 24.568359375] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/grouped_qkv_mlp_fc1_crop_0_bucket_768 | 1 | 2026-10-07T13:03:53.219705+00:00 / 2026-10-07T13:05:42.442256+00:00 | [25.33935546875, 25.5673828125, 24.568359375] / [26.95458984375, 25.95849609375, 24.81689453125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/grouped_qkv_mlp_fc1_crop_1_bucket_3072 | 1 | 2026-10-07T13:05:46.720628+00:00 / 2026-10-07T13:07:32.182952+00:00 | [26.7177734375, 25.92578125, 24.8125] / [26.841796875, 26.18408203125, 25.0263671875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/internal_format_opposite_crop_0_bucket_768 | 1 | 2026-10-07T12:56:56.539841+00:00 / 2026-10-07T12:58:27.673456+00:00 | [24.3876953125, 23.06005859375, 23.4462890625] / [26.748046875, 24.19384765625, 23.81298828125] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| vision_matrix/internal_format_opposite_crop_1_bucket_3072 | 1 | 2026-10-07T12:58:32.013224+00:00 / 2026-10-07T13:00:07.430112+00:00 | [27.0888671875, 24.30712890625, 23.85205078125] / [25.7099609375, 24.634765625, 24.01513671875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |
| capture.receipt | 1 | 2026-10-07T12:52:44.658669+00:00 / 2026-10-07T12:54:46.979010+00:00 | [21.8095703125, 22.77294921875, 23.55859375] / [22.49462890625, 22.66015625, 23.41796875] | 192/192 | False / False | not exposed by successful queried interfaces / not exposed by successful queried interfaces |

Failures, missing data and launch-only records:

- matmul/M768_qkv_compiled_nd_internal_on: failed; launch receipt overrides worker status; the worker may have failed after writing timings; expected one kernel CSV, found 0; no aggregate invented
- matmul_internal_off_validation/M768_qkv_compiled_nz_internal_off: unsupported_configuration; expected one kernel CSV, found 0; no aggregate invented
- matmul_internal_off_validation/M768_qkv_eager_nz_internal_off: unsupported_configuration; expected one kernel CSV, found 0; no aggregate invented
- capture.receipt: completed; no model/calibration result; inspect receipt/log (capture launches are expected here)

Full per-call columns, shape/format/HF32 histograms, conversion directions, remaining types and per-stream gaps are retained in the companion JSON. Unclassified does not mean erroneous; it means no semantic attribution was guessed.
