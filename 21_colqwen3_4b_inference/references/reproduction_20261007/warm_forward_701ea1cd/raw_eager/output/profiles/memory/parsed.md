# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/raw_eager/output/profiles/memory/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/raw_eager/output/profiles/memory/raw/liteserver-c001-4_7223_20261007104323966_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `442713.151 us`
- `Free`: `1904332.442 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2219.750 us`
- `Stage`: `2347045.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 180 | 126855.760 |
| `MatMulV3` | 564 | 91992.849 |
| `MatMulV2` | 186 | 53110.073 |
| `Mul` | 1998 | 36816.929 |
| `Cast` | 1647 | 19235.621 |
| `Transpose` | 552 | 17137.912 |
| `ReduceMean` | 723 | 15599.798 |
| `Add` | 1464 | 14402.081 |
| `Slice` | 1014 | 13551.334 |
| `Pows` | 579 | 12652.770 |
| `AsStrided` | 612 | 7193.994 |
| `MaskedScatter` | 3 | 6643.352 |
| `GeluV2` | 72 | 5710.551 |
| `ConcatD` | 366 | 4367.588 |
| `Swish` | 108 | 3769.755 |
| `Sub` | 144 | 3769.151 |
| `Neg` | 360 | 3670.271 |
| `Rsqrt` | 579 | 1856.217 |
| `ViewCopy` | 6 | 1249.244 |
| `LayerNormV3` | 12 | 741.615 |
| `Index` | 3 | 308.966 |
| `Gelu` | 12 | 308.466 |
| `GatherV2` | 6 | 277.465 |
| `IndexPutV2` | 9 | 182.205 |
| `GreaterEqual` | 3 | 179.805 |
| `ZerosLike` | 12 | 142.543 |
| `MemSet` | 9 | 88.640 |
| `RealDiv` | 3 | 87.961 |
| `Cos` | 6 | 79.220 |
| `Sin` | 6 | 78.205 |
| `ReduceAny` | 6 | 71.442 |
| `TensorMove` | 6 | 70.661 |
| `LpNormV2` | 3 | 69.940 |
| `SelectV2` | 3 | 66.941 |
| `Range` | 6 | 56.540 |
| `LogicalAnd` | 3 | 53.262 |
| `BroadcastTo` | 6 | 49.661 |
| `NonZero` | 9 | 46.881 |
| `Tile` | 3 | 34.220 |
| `Less` | 3 | 32.020 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 180 | 126855.760 |
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59448.563 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49459.102 |
| `aclnnMul_MulAiCore_Mul` | 1995 | 36778.047 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 240 | 32544.286 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 1608 | 19115.759 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 723 | 15599.798 |
| `aclnnAdd_AddAiCore_Add` | 885 | 12824.755 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 579 | 12652.770 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 336 | 9923.532 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 7214.380 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 3 | 6643.352 |
| `aclnnGeluV2_GeluV2_GeluV2` | 72 | 5710.551 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 510 | 5420.876 |
| `aclnnCat_ConcatD_ConcatD` | 366 | 4367.588 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3769.755 |
| `aclnnSub_SubAiCore_Sub` | 144 | 3769.151 |
| `aclnnNeg_NegAiCore_Neg` | 360 | 3670.271 |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 78 | 3650.971 |
| `aclnnNeg_SliceAiCore_Slice` | 144 | 2631.487 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2506.266 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2387.687 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 180 | 2300.041 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2057.301 |
| `aclnnCat_SliceAiCore_Slice` | 144 | 1997.623 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 579 | 1856.217 |
| `aclnnAdds_AddAiCore_Add` | 579 | 1577.326 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1444.047 |
| `aclnnInplaceCopy_ViewCopyAiCpu_ViewCopy` | 6 | 1249.244 |
| `aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3` | 12 | 741.615 |
| `aclnnIndex_IndexAiCore_Index` | 3 | 308.966 |
| `aclnnGelu_Gelu_Gelu` | 12 | 308.466 |
| `aclnnEmbedding_GatherV2AiCore_GatherV2` | 6 | 277.465 |
| `aclnnIndexPutImpl_IndexPutV2_IndexPutV2` | 9 | 182.205 |
| `aclnnGeTensor_GreaterEqual_GreaterEqual` | 3 | 179.805 |
| `aclnnInplaceZero_ZerosLikeAiCore_ZerosLike` | 12 | 142.543 |
| `aclnnNonzeroV2_NonzeroAiCore_MemSet` | 9 | 88.640 |
| `aclnnDiv_RealDivAiCore_RealDiv` | 3 | 87.961 |
| `aclnnCos_CosAiCore_Cos` | 6 | 79.220 |
| `aclnnSin_SinAiCore_Sin` | 6 | 78.205 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59448.563 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49459.102 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 240 | 32544.286 |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 78 | 3650.971 |
| `aclnnMatmul_MulAiCore_Mul` | 3 | 38.882 |
| `aclnnMatmul_BroadcastToAiCore_BroadcastTo` | 3 | 5.420 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49459.102 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 29166.398 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17531.811 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12750.354 |
| `MatMulV3 | "5040,1024;4096,1024;4096" -> "5040,4096" | ND;ND;ND -> ND` | 72 | 10455.727 |
| `MatMulV3 | "5040,4096;1024,4096;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 10321.589 |
| `MatMulV3 | "5040,1024;3072,1024;3072" -> "5040,3072" | ND;ND;ND -> ND` | 72 | 8295.962 |
| `MatMulV2 | "5040,1024;1024,1024;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 3198.862 |
| `MatMulV3 | "1260,4096;4096,4096;4096" -> "1260,4096" | ND;ND;ND -> ND` | 12 | 2057.462 |
| `MatMulV3 | "1260,4096;2560,4096;2560" -> "1260,2560" | ND;ND;ND -> ND` | 12 | 1413.546 |
| `MatMulV2 | "1274,2560;2560,2560;2560" -> "1274,2560" | ND;ND;ND -> ND` | 3 | 268.146 |
| `MatMulV2 | "5040,1536;1024,1536;1024" -> "5040,1024" | ND;ND;ND -> ND` | 3 | 183.963 |
| `Mul | "3,1,64,1;3,1,1,1274" -> "3,1,64,1274" | ND;ND -> ND` | 3 | 38.882 |
| `BroadcastTo | "1,1,64,1;4" -> "3,1,64,1" | ND;ND -> ND` | 3 | 5.420 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2220.404 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2213.744 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2209.204 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1317.966 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1312.626 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1312.426 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1311.626 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1310.867 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1310.427 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1310.067 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1310.006 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.726 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.106 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.547 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.346 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.167 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.046 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.026 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1307.646 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.946 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.806 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.666 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.626 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.546 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.286 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.186 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.146 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.926 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.926 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.886 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.646 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.306 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.266 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.086 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.026 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.006 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.906 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.666 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.666 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.666 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.forward.text_transformer` | 3 | 1357512.560 |
| `colqwen.forward.vision_transformer` | 3 | 841775.670 |
| `colqwen.raw_eager.forward.step0` | 1 | 789041.490 |
| `colqwen.raw_eager.forward.step2` | 1 | 783267.710 |
| `colqwen.raw_eager.forward.step1` | 1 | 775076.230 |
| `aten::to` | 1725 | 382668.710 |
| `empty_tensor` | 9810 | 355511.520 |
| `aten::_to_copy` | 1698 | 320814.940 |
| `aten::mul` | 2031 | 237968.600 |
| `aten::linear` | 750 | 229691.374 |
| `aten::add` | 1506 | 214526.780 |
| `aten::copy_` | 2232 | 190304.893 |
| `aten::reshape` | 2052 | 161824.670 |
| `npu::npu_prompt_flash_attention` | 180 | 143642.085 |
| `aclnnPromptFlashAttentionV3` | 180 | 126855.754 |
| `aten::as_strided` | 3519 | 116492.660 |
| `aten::empty` | 1713 | 116461.270 |
| `aten::matmul` | 435 | 112096.791 |
| `aclnnMatmul` | 435 | 108999.476 |
| `aten::transpose` | 1473 | 104391.890 |
| `aten::contiguous` | 504 | 101272.000 |
| `colqwen.forward.vision_prepare` | 3 | 97125.940 |
| `aten::pow` | 591 | 97042.760 |
| `aten::chunk` | 360 | 96291.080 |
| `aten::mean` | 723 | 93478.770 |
| `aten::clone` | 525 | 85839.360 |
| `aclnnInplaceCopy` | 2127 | 85275.110 |
| `aten::split` | 360 | 83261.140 |
| `aten::t` | 750 | 78865.690 |
| `aclnnMul` | 1995 | 77227.091 |
| `aten::view` | 2124 | 75065.540 |
| `aten::narrow` | 744 | 73323.550 |
| `aten::rsqrt` | 579 | 68962.920 |
| `aten::slice` | 756 | 49679.970 |
| `aten::cat` | 384 | 48958.990 |
| `colqwen.forward.text_prepare_mergers` | 3 | 46306.170 |
| `aten::addmm` | 318 | 44802.765 |
| `aten::neg` | 360 | 43035.160 |
| `aten::item` | 594 | 39725.990 |
| `aclnnAddmm` | 318 | 36224.116 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 11316 | 81049.460 |
| `aclrtLaunchKernelWithHostArgs` | 11316 | 61586.820 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 180 | 38777.460 |
| `PromptFlashAttention_Tiling` | 360 | 27461.940 |
| `aclnnInplaceCopy` | 2127 | 26744.590 |
| `aclnnMul` | 1995 | 19875.620 |
| `InnerPromptFlashAttention` | 180 | 10996.240 |
| `aclnnMean` | 723 | 8738.710 |
| `aclnnAdd` | 885 | 8207.300 |
| `aclnnPowTensorScalar` | 579 | 6785.460 |
| `aclnnRsqrt` | 579 | 6683.870 |
| `aclnnNeg` | 360 | 6470.230 |
| `aclnnAdds` | 579 | 5405.320 |
| `aclnnCat` | 369 | 5102.860 |
| `aclnnMatmul` | 435 | 4576.260 |
| `aclnnNonzeroV2` | 9 | 4217.200 |
| `aclrtGetStreamAttribute` | 9594 | 3871.340 |
| `aclrtSynchronizeStream` | 36 | 3475.490 |
| `aclnnAddmm` | 318 | 3368.230 |
| `aclrtMemcpy` | 48 | 2076.260 |
| `aclnnSilu` | 108 | 1613.660 |
| `aclnnSub` | 144 | 1365.710 |
| `aclrtGetHardwareSyncAddr` | 1494 | 1069.040 |
| `aclnnGeluV2` | 72 | 706.390 |
| `aclnnInplaceCopyGetWorkspaceSize` | 6 | 650.720 |
| `aclnnNonzeroV2GetWorkspaceSize` | 9 | 508.460 |
| `aclnnIndexPutImpl` | 9 | 494.070 |
| `aclnnIndexPutImplGetWorkspaceSize` | 9 | 470.920 |
| `aclrtGetResInCurrentThread` | 408 | 325.020 |
| `aclnnEqScalar` | 12 | 273.790 |
| `aclnnLayerNorm` | 12 | 260.800 |
| `aclrtSynchronizeStreamWithTimeout` | 21 | 242.140 |
| `aclnnEmbedding` | 6 | 209.410 |
| `aclnnAll` | 6 | 167.220 |
| `IndexPutV2_Tiling` | 9 | 166.700 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 164.210 |
| `Slice_Tiling` | 6 | 131.490 |
| `aclnnReduceSum` | 3 | 124.990 |
| `aclnnInplaceZero` | 12 | 121.320 |
| `NonZero_Tiling` | 18 | 110.560 |

