# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/bsnd/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/bsnd/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_92605_20261007121314914_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `221470.202 us`
- `Free`: `1061360.029 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2384.250 us`
- `Stage`: `1282830.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 58767.003 |
| `MatMulV2` | 108 | 50481.908 |
| `PromptFlashAttention` | 108 | 36073.398 |
| `Mul` | 1410 | 22489.249 |
| `Slice` | 972 | 10921.359 |
| `Pows` | 435 | 10312.767 |
| `ReduceMean` | 435 | 9117.359 |
| `Cast` | 870 | 8790.890 |
| `Add` | 876 | 5949.576 |
| `Swish` | 108 | 3735.914 |
| `ConcatD` | 216 | 1799.620 |
| `Neg` | 216 | 1745.872 |
| `Rsqrt` | 435 | 1287.027 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58767.003 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50481.908 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 36073.398 |
| `aclnnMul_MulAiCore_Mul` | 1410 | 22489.249 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10312.767 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 9117.359 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 8790.890 |
| `aclnnAdd_AddAiCore_Add` | 441 | 4513.842 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3735.914 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 324 | 3214.443 |
| `aclnnNeg_SliceAiCore_Slice` | 216 | 2176.183 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2073.721 |
| `aclnnCat_SliceAiCore_Slice` | 216 | 1993.441 |
| `aclnnCat_ConcatD_ConcatD` | 216 | 1799.620 |
| `aclnnNeg_NegAiCore_Neg` | 216 | 1745.872 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1463.571 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1435.734 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1287.027 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58767.003 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50481.908 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50481.908 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28965.542 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17190.556 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12610.905 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 475.529 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.630 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.409 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.869 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.889 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.229 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.850 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.830 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.669 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.090 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.050 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.010 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.949 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.930 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.809 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.670 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.649 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.490 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.450 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.370 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.330 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.249 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.190 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.089 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1282187.020 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 429843.170 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 427523.520 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 425782.600 |
| `empty_tensor` | 5757 | 207970.280 |
| `aten::to` | 870 | 203289.330 |
| `aten::_to_copy` | 870 | 170868.550 |
| `aten::mul` | 1410 | 164714.380 |
| `aten::linear` | 432 | 141971.189 |
| `aten::add` | 876 | 134029.870 |
| `aten::matmul` | 432 | 112578.691 |
| `aclnnMatmul` | 432 | 109248.923 |
| `aten::reshape` | 1296 | 107095.910 |
| `aten::copy_` | 978 | 83021.140 |
| `aten::pow` | 435 | 69905.700 |
| `aten::empty` | 870 | 62834.880 |
| `aten::chunk` | 216 | 58193.890 |
| `aten::as_strided` | 1620 | 55021.320 |
| `aten::mean` | 435 | 54757.720 |
| `aclnnMul` | 1410 | 53407.740 |
| `aten::rsqrt` | 435 | 51476.410 |
| `aten::split` | 216 | 50507.260 |
| `aten::view` | 1296 | 46396.570 |
| `aten::t` | 432 | 46361.220 |
| `npu::npu_prompt_flash_attention` | 108 | 44540.010 |
| `aten::narrow` | 432 | 43250.200 |
| `aclnnInplaceCopy` | 978 | 38664.760 |
| `aclnnPromptFlashAttentionV3` | 108 | 36073.401 |
| `aten::transpose` | 432 | 31544.710 |
| `aten::item` | 435 | 28944.650 |
| `aten::slice` | 432 | 28936.130 |
| `aten::split_with_sizes` | 216 | 26271.530 |
| `aten::cat` | 216 | 26176.200 |
| `aten::neg` | 216 | 25539.470 |
| `aten::_reshape_alias` | 324 | 20646.900 |
| `aten::contiguous` | 108 | 18962.280 |
| `aclnnPowTensorScalar` | 435 | 17284.980 |
| `aclnnAdds` | 435 | 17070.970 |
| `aclnnMean` | 435 | 17047.410 |
| `aclnnRsqrt` | 435 | 16949.410 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 6513 | 38829.490 |
| `aclrtLaunchKernelWithHostArgs` | 6513 | 30743.180 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21903.410 |
| `PromptFlashAttention_Tiling` | 216 | 15383.060 |
| `aclnnMul` | 1410 | 12564.210 |
| `aclnnInplaceCopy` | 978 | 9640.380 |
| `InnerPromptFlashAttention` | 108 | 5911.660 |
| `aclnnMean` | 435 | 4967.820 |
| `aclnnMatmul` | 432 | 4438.750 |
| `aclnnAdds` | 435 | 3936.070 |
| `aclnnAdd` | 441 | 3787.270 |
| `aclnnPowTensorScalar` | 435 | 3767.280 |
| `aclnnRsqrt` | 435 | 3744.360 |
| `aclnnCat` | 216 | 2851.530 |
| `aclnnNeg` | 216 | 2817.610 |
| `aclrtGetStreamAttribute` | 5649 | 2291.920 |
| `aclnnSilu` | 108 | 1490.940 |
| `aclrtGetHardwareSyncAddr` | 867 | 604.580 |
| `aclrtGetResInCurrentThread` | 216 | 157.350 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 134.350 |

