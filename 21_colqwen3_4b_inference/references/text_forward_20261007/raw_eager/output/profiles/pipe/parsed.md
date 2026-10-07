# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_10776_20261007110752481_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `230817.828 us`
- `Free`: `1025002.944 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2685.750 us`
- `Stage`: `1255820.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 59623.915 |
| `MatMulV2` | 108 | 49731.542 |
| `PromptFlashAttention` | 108 | 33201.388 |
| `Mul` | 1410 | 22687.132 |
| `Transpose` | 324 | 11063.940 |
| `Cast` | 870 | 9717.053 |
| `Pows` | 435 | 9008.323 |
| `ReduceMean` | 435 | 8384.751 |
| `Add` | 876 | 7212.045 |
| `AsStrided` | 540 | 5771.139 |
| `Slice` | 432 | 5716.159 |
| `Swish` | 108 | 3707.677 |
| `ConcatD` | 216 | 2047.318 |
| `Neg` | 216 | 1704.318 |
| `Rsqrt` | 435 | 1243.101 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59623.915 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49731.542 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33201.388 |
| `aclnnMul_MulAiCore_Mul` | 1410 | 22687.132 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9717.053 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 9008.323 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8384.751 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 7145.244 |
| `aclnnAdd_AddAiCore_Add` | 441 | 5923.718 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 108 | 3918.696 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3707.677 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2495.978 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2391.921 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2281.295 |
| `aclnnCat_ConcatD_ConcatD` | 216 | 2047.318 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 1969.776 |
| `aclnnNeg_NegAiCore_Neg` | 216 | 1704.318 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1465.088 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1288.327 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1243.101 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 883.240 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59623.915 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49731.542 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49731.542 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 29005.521 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17656.997 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12961.397 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 463.929 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 463.610 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 463.529 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 463.469 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 463.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.789 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.649 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.569 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.550 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.530 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.529 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.369 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.289 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.230 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.209 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.110 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 462.010 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.930 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.829 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.829 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.810 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.809 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.750 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.429 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.389 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.389 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 461.329 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1255317.280 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 420774.220 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 417931.430 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 417574.130 |
| `empty_tensor` | 5865 | 200744.370 |
| `aten::to` | 870 | 192693.420 |
| `aten::_to_copy` | 870 | 162093.370 |
| `aten::mul` | 1410 | 154502.340 |
| `aten::linear` | 432 | 134664.440 |
| `aten::add` | 876 | 125216.560 |
| `aten::matmul` | 432 | 110658.016 |
| `aclnnMatmul` | 432 | 109355.445 |
| `aten::reshape` | 1296 | 98764.760 |
| `aten::copy_` | 1086 | 88913.090 |
| `aten::pow` | 435 | 66492.690 |
| `aten::empty` | 870 | 59622.560 |
| `aten::as_strided` | 1842 | 58920.280 |
| `aten::transpose` | 864 | 57596.630 |
| `aten::chunk` | 216 | 53838.500 |
| `aten::mean` | 435 | 52020.860 |
| `aclnnMul` | 1410 | 51669.858 |
| `aten::rsqrt` | 435 | 48135.610 |
| `aten::split` | 216 | 46641.380 |
| `npu::npu_prompt_flash_attention` | 108 | 45008.920 |
| `aten::view` | 1296 | 42980.150 |
| `aten::t` | 432 | 42362.660 |
| `aclnnInplaceCopy` | 1086 | 40808.631 |
| `aten::contiguous` | 216 | 40585.090 |
| `aten::narrow` | 432 | 39908.530 |
| `aclnnPromptFlashAttentionV3` | 108 | 33201.383 |
| `aten::clone` | 216 | 33037.430 |
| `aten::item` | 435 | 27008.170 |
| `aten::slice` | 432 | 26700.550 |
| `aten::split_with_sizes` | 216 | 24576.190 |
| `aten::cat` | 216 | 24527.050 |
| `aten::neg` | 216 | 23851.980 |
| `aten::_reshape_alias` | 324 | 19607.430 |
| `aclnnPowTensorScalar` | 435 | 16510.620 |
| `aclnnMean` | 435 | 16067.290 |
| `aclnnRsqrt` | 435 | 15840.940 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 6837 | 33575.380 |
| `aclrtLaunchKernelWithHostArgs` | 6837 | 25477.360 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21723.700 |
| `PromptFlashAttention_Tiling` | 216 | 15389.090 |
| `aclnnMul` | 1410 | 11870.810 |
| `aclnnInplaceCopy` | 1086 | 9344.480 |
| `InnerPromptFlashAttention` | 108 | 5928.480 |
| `aclnnMean` | 435 | 4364.930 |
| `aclnnMatmul` | 432 | 3747.560 |
| `aclnnAdd` | 441 | 3283.360 |
| `aclnnAdds` | 435 | 3247.910 |
| `aclnnPowTensorScalar` | 435 | 3176.730 |
| `aclnnRsqrt` | 435 | 3175.450 |
| `aclnnCat` | 216 | 2622.010 |
| `aclnnNeg` | 216 | 2581.880 |
| `aclrtGetStreamAttribute` | 5757 | 2046.540 |
| `aclnnSilu` | 108 | 1383.810 |
| `aclrtGetHardwareSyncAddr` | 867 | 559.230 |
| `aclrtGetResInCurrentThread` | 216 | 156.490 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 132.080 |
