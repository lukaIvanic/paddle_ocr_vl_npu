# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_90705_20261007121148972_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `232021.118 us`
- `Free`: `1104617.448 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2856.250 us`
- `Stage`: `1336638.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 58792.833 |
| `MatMulV2` | 108 | 50314.078 |
| `PromptFlashAttention` | 108 | 33346.351 |
| `Mul` | 1410 | 22419.104 |
| `Transpose` | 324 | 10732.019 |
| `Pows` | 435 | 10406.544 |
| `Cast` | 870 | 9925.812 |
| `ReduceMean` | 435 | 8386.678 |
| `Add` | 876 | 6427.005 |
| `Slice` | 432 | 6201.005 |
| `AsStrided` | 540 | 6037.301 |
| `Swish` | 108 | 3719.584 |
| `ConcatD` | 216 | 2010.557 |
| `Neg` | 216 | 1920.915 |
| `Rsqrt` | 435 | 1383.268 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58792.833 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50314.078 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33346.351 |
| `aclnnMul_MulAiCore_Mul` | 1410 | 22419.104 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10406.544 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9925.812 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8386.678 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 6897.648 |
| `aclnnAdd_AddAiCore_Add` | 441 | 4984.488 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 108 | 3834.371 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3719.584 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2618.430 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2583.953 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2509.049 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2175.955 |
| `aclnnCat_ConcatD_ConcatD` | 216 | 2010.557 |
| `aclnnNeg_NegAiCore_Neg` | 216 | 1920.915 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1442.517 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1441.097 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1383.268 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 909.822 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58792.833 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50314.078 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50314.078 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28998.903 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17134.897 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12659.033 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.089 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.930 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.709 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.469 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.690 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.549 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.409 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.070 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.929 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.829 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.610 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.549 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.529 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.449 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.290 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.209 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.130 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.109 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.990 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.850 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.849 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.770 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.669 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1336179.850 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 451464.900 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 444807.560 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 440986.510 |
| `empty_tensor` | 5865 | 214940.040 |
| `aten::to` | 870 | 211835.800 |
| `aten::_to_copy` | 870 | 178175.270 |
| `aten::mul` | 1410 | 166616.230 |
| `aten::linear` | 432 | 139910.455 |
| `aten::add` | 876 | 133327.090 |
| `aten::matmul` | 432 | 111586.017 |
| `aclnnMatmul` | 432 | 109106.920 |
| `aten::reshape` | 1296 | 104965.930 |
| `aten::copy_` | 1086 | 97689.160 |
| `aten::pow` | 435 | 70814.580 |
| `aten::empty` | 870 | 64793.230 |
| `aten::as_strided` | 1842 | 62545.760 |
| `aten::transpose` | 864 | 61390.970 |
| `aten::chunk` | 216 | 57052.110 |
| `aten::mean` | 435 | 55712.530 |
| `aclnnMul` | 1410 | 54888.971 |
| `aten::rsqrt` | 435 | 51039.530 |
| `aten::split` | 216 | 49423.320 |
| `npu::npu_prompt_flash_attention` | 108 | 47629.620 |
| `aten::view` | 1296 | 45717.770 |
| `aten::t` | 432 | 45169.860 |
| `aclnnInplaceCopy` | 1086 | 45031.781 |
| `aten::contiguous` | 216 | 43153.440 |
| `aten::narrow` | 432 | 42280.770 |
| `aten::clone` | 216 | 35112.160 |
| `aclnnPromptFlashAttentionV3` | 108 | 33346.346 |
| `aten::item` | 435 | 28776.600 |
| `aten::slice` | 432 | 28289.690 |
| `aten::cat` | 216 | 26142.930 |
| `aten::split_with_sizes` | 216 | 25936.700 |
| `aten::neg` | 216 | 25081.990 |
| `aten::_reshape_alias` | 324 | 20629.590 |
| `aclnnPowTensorScalar` | 435 | 17521.800 |
| `aclnnMean` | 435 | 17269.870 |
| `aclnnAdds` | 435 | 16977.380 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 6837 | 39886.680 |
| `aclrtLaunchKernelWithHostArgs` | 6837 | 31658.510 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21800.050 |
| `PromptFlashAttention_Tiling` | 216 | 15418.330 |
| `aclnnMul` | 1410 | 13668.230 |
| `aclnnInplaceCopy` | 1086 | 10706.810 |
| `InnerPromptFlashAttention` | 108 | 6011.100 |
| `aclnnMean` | 435 | 4916.110 |
| `aclnnMatmul` | 432 | 4177.680 |
| `aclnnAdd` | 441 | 3779.540 |
| `aclnnPowTensorScalar` | 435 | 3747.180 |
| `aclnnAdds` | 435 | 3737.820 |
| `aclnnRsqrt` | 435 | 3658.140 |
| `aclnnNeg` | 216 | 2891.910 |
| `aclnnCat` | 216 | 2832.200 |
| `aclrtGetStreamAttribute` | 5757 | 2245.130 |
| `aclnnSilu` | 108 | 1433.490 |
| `aclrtGetHardwareSyncAddr` | 867 | 573.300 |
| `aclrtGetResInCurrentThread` | 216 | 156.640 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 141.170 |

