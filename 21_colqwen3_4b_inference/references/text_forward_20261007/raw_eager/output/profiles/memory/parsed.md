# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/raw_eager/output/profiles/memory/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/raw_eager/output/profiles/memory/raw/liteserver-c001-4_10776_20261007110807092_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `230759.652 us`
- `Free`: `1047262.303 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3277.750 us`
- `Stage`: `1278022.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 59296.485 |
| `MatMulV2` | 108 | 50095.159 |
| `PromptFlashAttention` | 108 | 33169.362 |
| `Mul` | 1410 | 22697.015 |
| `Transpose` | 324 | 11060.690 |
| `Cast` | 870 | 9638.414 |
| `Pows` | 435 | 8791.322 |
| `ReduceMean` | 435 | 8426.499 |
| `Add` | 876 | 7316.985 |
| `Slice` | 432 | 5829.507 |
| `AsStrided` | 540 | 5747.464 |
| `Swish` | 108 | 3701.430 |
| `ConcatD` | 216 | 2030.214 |
| `Neg` | 216 | 1708.731 |
| `Rsqrt` | 435 | 1252.357 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59296.485 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50095.159 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33169.362 |
| `aclnnMul_MulAiCore_Mul` | 1410 | 22697.015 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9638.414 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 8791.322 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8426.499 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 7107.245 |
| `aclnnAdd_AddAiCore_Add` | 441 | 6016.071 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 108 | 3953.445 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3701.430 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2491.714 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2384.258 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2377.068 |
| `aclnnCat_ConcatD_ConcatD` | 216 | 2030.214 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 1987.839 |
| `aclnnNeg_NegAiCore_Neg` | 216 | 1708.731 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1457.410 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1300.914 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1252.357 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 878.682 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59296.485 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50095.159 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50095.159 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28964.719 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17469.089 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12862.677 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.750 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.369 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.349 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.249 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.249 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.049 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.029 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.009 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.710 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.710 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.510 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.349 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.870 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.730 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.729 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.609 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.490 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.150 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.030 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.869 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.849 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.550 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.529 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.490 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.290 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.269 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.009 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.970 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.749 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.730 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.489 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1277842.940 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 432215.490 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 423660.740 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 422957.940 |
| `empty_tensor` | 5865 | 203231.440 |
| `aten::to` | 870 | 195154.820 |
| `aten::_to_copy` | 870 | 164242.430 |
| `aten::mul` | 1410 | 157391.030 |
| `aten::linear` | 432 | 135917.622 |
| `aten::add` | 876 | 127041.110 |
| `aten::matmul` | 432 | 111061.089 |
| `aclnnMatmul` | 432 | 109391.646 |
| `aten::reshape` | 1296 | 99677.530 |
| `aten::copy_` | 1086 | 90558.330 |
| `aten::pow` | 435 | 67892.550 |
| `aten::empty` | 870 | 60059.630 |
| `aten::as_strided` | 1842 | 59356.790 |
| `aten::transpose` | 864 | 58210.150 |
| `aten::chunk` | 216 | 54174.150 |
| `aten::mean` | 435 | 53216.060 |
| `aclnnMul` | 1410 | 53081.882 |
| `aten::rsqrt` | 435 | 48620.760 |
| `aten::split` | 216 | 46931.370 |
| `npu::npu_prompt_flash_attention` | 108 | 46065.380 |
| `aten::view` | 1296 | 43361.610 |
| `aten::t` | 432 | 42869.000 |
| `aclnnInplaceCopy` | 1086 | 42079.324 |
| `aten::contiguous` | 216 | 41062.160 |
| `aten::narrow` | 432 | 40081.050 |
| `aten::clone` | 216 | 33495.300 |
| `aclnnPromptFlashAttentionV3` | 108 | 33169.363 |
| `aten::item` | 435 | 27191.080 |
| `aten::slice` | 432 | 26800.110 |
| `aten::cat` | 216 | 25145.580 |
| `aten::split_with_sizes` | 216 | 24737.440 |
| `aten::neg` | 216 | 24355.640 |
| `aten::_reshape_alias` | 324 | 19629.570 |
| `aclnnPowTensorScalar` | 435 | 17073.120 |
| `aclnnMean` | 435 | 16627.330 |
| `aclnnAdds` | 435 | 16334.630 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 6837 | 33554.260 |
| `aclrtLaunchKernelWithHostArgs` | 6837 | 25493.150 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 22185.590 |
| `PromptFlashAttention_Tiling` | 216 | 15630.840 |
| `aclnnMul` | 1410 | 11765.290 |
| `aclnnInplaceCopy` | 1086 | 9300.720 |
| `InnerPromptFlashAttention` | 108 | 5770.150 |
| `aclnnMean` | 435 | 4344.130 |
| `aclnnMatmul` | 432 | 3698.350 |
| `aclnnAdd` | 441 | 3248.020 |
| `aclnnAdds` | 435 | 3235.580 |
| `aclnnPowTensorScalar` | 435 | 3225.050 |
| `aclnnRsqrt` | 435 | 3151.630 |
| `aclnnCat` | 216 | 2602.180 |
| `aclnnNeg` | 216 | 2557.920 |
| `aclrtGetStreamAttribute` | 5757 | 2155.980 |
| `aclnnSilu` | 108 | 1440.570 |
| `aclrtGetHardwareSyncAddr` | 867 | 542.900 |
| `aclrtGetResInCurrentThread` | 216 | 175.060 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 141.370 |
