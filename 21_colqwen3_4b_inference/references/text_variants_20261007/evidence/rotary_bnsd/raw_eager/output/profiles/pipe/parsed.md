# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/rotary_bnsd/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/rotary_bnsd/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_96361_20261007121527745_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `222005.060 us`
- `Free`: `1062852.111 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2572.500 us`
- `Stage`: `1284857.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 58790.917 |
| `MatMulV2` | 108 | 50317.353 |
| `PromptFlashAttention` | 108 | 33616.594 |
| `Mul` | 978 | 17540.860 |
| `Transpose` | 324 | 10658.656 |
| `Cast` | 870 | 10641.098 |
| `Pows` | 435 | 10436.095 |
| `ReduceMean` | 435 | 8412.416 |
| `Slice` | 432 | 5864.972 |
| `RotaryPositionEmbedding` | 216 | 5102.233 |
| `Add` | 660 | 4264.925 |
| `Swish` | 108 | 3814.757 |
| `Rsqrt` | 435 | 1528.469 |
| `AsStrided` | 108 | 1017.483 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58790.917 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50317.353 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33616.594 |
| `aclnnMul_MulAiCore_Mul` | 978 | 17540.860 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 324 | 10658.656 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 10641.098 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10436.095 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8412.416 |
| `aclnnRotaryPositionEmbeddingV2_RotaryPositionEmbedding_RotaryPositionEmbedding` | 216 | 5102.233 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3814.757 |
| `aclnnAdd_AddAiCore_Add` | 225 | 2833.140 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2349.632 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2075.801 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1528.469 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1439.539 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1431.785 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 1017.483 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58790.917 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50317.353 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50317.353 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28991.653 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17150.370 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12648.894 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.369 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.050 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.710 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.450 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.170 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.889 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.809 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.749 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.670 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.649 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.630 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.810 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.649 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.410 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.089 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.050 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.030 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.010 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.010 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.889 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.870 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.709 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.630 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.569 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.530 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.510 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.369 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.310 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.309 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.210 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.149 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1284189.840 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 429062.020 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 429046.740 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 427136.450 |
| `aten::to` | 870 | 212833.990 |
| `empty_tensor` | 5217 | 198825.110 |
| `aten::_to_copy` | 870 | 178718.960 |
| `aten::linear` | 432 | 147408.505 |
| `aten::mul` | 978 | 122598.130 |
| `aten::copy_` | 1302 | 115085.980 |
| `aten::matmul` | 432 | 114547.024 |
| `aten::add` | 660 | 114292.720 |
| `aten::reshape` | 1296 | 112310.610 |
| `aclnnMatmul` | 432 | 110790.637 |
| `aten::contiguous` | 432 | 82773.700 |
| `aten::pow` | 435 | 72908.690 |
| `aten::clone` | 432 | 67652.890 |
| `aten::empty` | 870 | 65745.960 |
| `aten::transpose` | 864 | 63135.690 |
| `aten::mean` | 435 | 57031.320 |
| `aten::as_strided` | 1620 | 56635.650 |
| `aclnnInplaceCopy` | 1302 | 53145.336 |
| `aten::rsqrt` | 435 | 52842.920 |
| `npu::npu_prompt_flash_attention` | 108 | 51859.950 |
| `aten::view` | 1296 | 49288.000 |
| `aten::t` | 432 | 48361.400 |
| `aclnnMul` | 978 | 39325.450 |
| `aclnnPromptFlashAttentionV3` | 108 | 33616.592 |
| `aten::item` | 435 | 30203.870 |
| `npu::npu_rotary_mul` | 216 | 28450.080 |
| `aten::split_with_sizes` | 216 | 27332.000 |
| `aten::_reshape_alias` | 324 | 21363.750 |
| `aclnnPowTensorScalar` | 435 | 18186.960 |
| `aclnnMean` | 435 | 17832.120 |
| `aclnnAdds` | 435 | 17671.820 |
| `aclnnRsqrt` | 435 | 17488.090 |
| `aten::result_type` | 435 | 15118.680 |
| `aten::_local_scalar_dense` | 435 | 14813.700 |
| `aten::silu` | 108 | 14446.660 |
| `aten::unsqueeze` | 216 | 13874.340 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 5541 | 38000.260 |
| `aclrtLaunchKernelWithHostArgs` | 5541 | 30007.570 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 27468.160 |
| `PromptFlashAttention_Tiling` | 216 | 19220.290 |
| `aclnnInplaceCopy` | 1302 | 14347.590 |
| `aclnnMul` | 978 | 9063.670 |
| `InnerPromptFlashAttention` | 108 | 6749.490 |
| `aclnnMean` | 435 | 5351.980 |
| `aclnnMatmul` | 432 | 4726.660 |
| `aclnnAdds` | 435 | 4108.910 |
| `aclnnPowTensorScalar` | 435 | 3979.180 |
| `aclnnRsqrt` | 435 | 3868.960 |
| `aclnnRotaryPositionEmbeddingV2` | 216 | 2560.260 |
| `aclnnAdd` | 225 | 2078.570 |
| `aclrtGetStreamAttribute` | 5109 | 2065.800 |
| `aclnnSilu` | 108 | 1535.550 |
| `aclrtGetHardwareSyncAddr` | 1083 | 768.240 |
| `aclrtGetResInCurrentThread` | 216 | 181.950 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 151.110 |

