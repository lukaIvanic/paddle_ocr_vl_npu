# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_7223_20261007104259534_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `442128.356 us`
- `Free`: `1778494.551 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `1991.500 us`
- `Stage`: `2220622.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 180 | 126702.615 |
| `MatMulV3` | 564 | 91875.684 |
| `MatMulV2` | 186 | 53001.918 |
| `Mul` | 1998 | 36774.821 |
| `Cast` | 1647 | 19498.649 |
| `Transpose` | 552 | 16917.282 |
| `ReduceMean` | 723 | 15607.695 |
| `Add` | 1464 | 14331.813 |
| `Slice` | 1014 | 13590.886 |
| `Pows` | 579 | 12602.607 |
| `AsStrided` | 612 | 7194.183 |
| `MaskedScatter` | 3 | 6641.293 |
| `GeluV2` | 72 | 5711.358 |
| `ConcatD` | 366 | 4396.273 |
| `Swish` | 108 | 3769.558 |
| `Sub` | 144 | 3759.158 |
| `Neg` | 360 | 3670.369 |
| `Rsqrt` | 579 | 1851.542 |
| `ViewCopy` | 6 | 1079.101 |
| `LayerNormV3` | 12 | 746.215 |
| `Index` | 3 | 314.447 |
| `Gelu` | 12 | 307.966 |
| `GatherV2` | 6 | 284.706 |
| `IndexPutV2` | 9 | 183.104 |
| `GreaterEqual` | 3 | 174.463 |
| `ZerosLike` | 12 | 136.666 |
| `MemSet` | 9 | 97.445 |
| `RealDiv` | 3 | 87.803 |
| `Cos` | 6 | 80.142 |
| `Sin` | 6 | 78.860 |
| `LpNormV2` | 3 | 75.742 |
| `ReduceAny` | 6 | 71.981 |
| `TensorMove` | 6 | 69.981 |
| `SelectV2` | 3 | 66.640 |
| `Range` | 6 | 55.881 |
| `LogicalAnd` | 3 | 53.400 |
| `NonZero` | 9 | 46.360 |
| `BroadcastTo` | 6 | 46.060 |
| `Less` | 3 | 35.701 |
| `Tile` | 3 | 33.881 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 180 | 126702.615 |
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59346.273 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49346.488 |
| `aclnnMul_MulAiCore_Mul` | 1995 | 36735.859 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 240 | 32529.411 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 1608 | 19376.567 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 723 | 15607.695 |
| `aclnnAdd_AddAiCore_Add` | 885 | 12753.297 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 579 | 12602.607 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 336 | 9705.584 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 7211.698 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 3 | 6641.293 |
| `aclnnGeluV2_GeluV2_GeluV2` | 72 | 5711.358 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 510 | 5383.328 |
| `aclnnCat_ConcatD_ConcatD` | 366 | 4396.273 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3769.558 |
| `aclnnSub_SubAiCore_Sub` | 144 | 3759.158 |
| `aclnnNeg_NegAiCore_Neg` | 360 | 3670.369 |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 78 | 3655.430 |
| `aclnnNeg_SliceAiCore_Slice` | 144 | 2710.504 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2500.928 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2394.910 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 180 | 2298.345 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2060.018 |
| `aclnnCat_SliceAiCore_Slice` | 144 | 1998.149 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 579 | 1851.542 |
| `aclnnAdds_AddAiCore_Add` | 579 | 1578.516 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1438.887 |
| `aclnnInplaceCopy_ViewCopyAiCpu_ViewCopy` | 6 | 1079.101 |
| `aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3` | 12 | 746.215 |
| `aclnnIndex_IndexAiCore_Index` | 3 | 314.447 |
| `aclnnGelu_Gelu_Gelu` | 12 | 307.966 |
| `aclnnEmbedding_GatherV2AiCore_GatherV2` | 6 | 284.706 |
| `aclnnIndexPutImpl_IndexPutV2_IndexPutV2` | 9 | 183.104 |
| `aclnnGeTensor_GreaterEqual_GreaterEqual` | 3 | 174.463 |
| `aclnnInplaceZero_ZerosLikeAiCore_ZerosLike` | 12 | 136.666 |
| `aclnnNonzeroV2_NonzeroAiCore_MemSet` | 9 | 97.445 |
| `aclnnDiv_RealDivAiCore_RealDiv` | 3 | 87.803 |
| `aclnnCos_CosAiCore_Cos` | 6 | 80.142 |
| `aclnnSin_SinAiCore_Sin` | 6 | 78.860 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59346.273 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 49346.488 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 240 | 32529.411 |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 78 | 3655.430 |
| `aclnnMatmul_MulAiCore_Mul` | 3 | 38.962 |
| `aclnnMatmul_BroadcastToAiCore_BroadcastTo` | 3 | 5.140 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49346.488 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 29229.920 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17487.857 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12628.496 |
| `MatMulV3 | "5040,1024;4096,1024;4096" -> "5040,4096" | ND;ND;ND -> ND` | 72 | 10435.288 |
| `MatMulV3 | "5040,4096;1024,4096;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 10327.567 |
| `MatMulV3 | "5040,1024;3072,1024;3072" -> "5040,3072" | ND;ND;ND -> ND` | 72 | 8278.647 |
| `MatMulV2 | "5040,1024;1024,1024;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 3209.461 |
| `MatMulV3 | "1260,4096;4096,4096;4096" -> "1260,4096" | ND;ND;ND -> ND` | 12 | 2068.560 |
| `MatMulV3 | "1260,4096;2560,4096;2560" -> "1260,2560" | ND;ND;ND -> ND` | 12 | 1419.349 |
| `MatMulV2 | "1274,2560;2560,2560;2560" -> "1274,2560" | ND;ND;ND -> ND` | 3 | 266.606 |
| `MatMulV2 | "5040,1536;1024,1536;1024" -> "5040,1024" | ND;ND;ND -> ND` | 3 | 179.363 |
| `Mul | "3,1,64,1;3,1,1,1274" -> "3,1,64,1274" | ND;ND -> ND` | 3 | 38.962 |
| `BroadcastTo | "1,1,64,1;4" -> "3,1,64,1" | ND;ND -> ND` | 3 | 5.140 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2219.425 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2211.984 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2209.884 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1319.026 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1316.546 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1315.387 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1314.046 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1313.187 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1311.626 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1311.587 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1311.387 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1310.226 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.786 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.446 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.306 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.286 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.267 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.206 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1309.046 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1308.226 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1307.387 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1307.326 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1307.026 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.186 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1306.086 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.726 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1305.566 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.886 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.886 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.866 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.866 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.506 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.486 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.307 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.286 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1304.266 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1303.786 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1303.286 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1302.666 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 1 | 1302.346 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.forward.text_transformer` | 3 | 1286581.590 |
| `colqwen.forward.vision_transformer` | 3 | 795266.920 |
| `colqwen.raw_eager.forward.step0` | 1 | 742964.320 |
| `colqwen.raw_eager.forward.step2` | 1 | 739264.250 |
| `colqwen.raw_eager.forward.step1` | 1 | 738532.530 |
| `aten::to` | 1725 | 362860.010 |
| `empty_tensor` | 9810 | 338037.900 |
| `aten::_to_copy` | 1698 | 304704.540 |
| `aten::mul` | 2031 | 226783.350 |
| `aten::linear` | 750 | 220570.615 |
| `aten::add` | 1506 | 204610.000 |
| `aten::copy_` | 2232 | 180266.076 |
| `aten::reshape` | 2052 | 153371.170 |
| `npu::npu_prompt_flash_attention` | 180 | 140606.884 |
| `aclnnPromptFlashAttentionV3` | 180 | 126702.612 |
| `aten::matmul` | 435 | 111443.460 |
| `aten::as_strided` | 3519 | 110395.580 |
| `aten::empty` | 1713 | 110366.490 |
| `aclnnMatmul` | 435 | 108777.192 |
| `aten::transpose` | 1473 | 98681.290 |
| `aten::contiguous` | 504 | 95839.660 |
| `aten::pow` | 591 | 92486.630 |
| `aten::chunk` | 360 | 90621.260 |
| `colqwen.forward.vision_prepare` | 3 | 89753.570 |
| `aten::mean` | 723 | 88528.200 |
| `aten::clone` | 525 | 81173.510 |
| `aclnnInplaceCopy` | 2127 | 80460.128 |
| `aten::split` | 360 | 78548.780 |
| `aten::t` | 750 | 74369.110 |
| `aclnnMul` | 1995 | 73978.964 |
| `aten::view` | 2124 | 71372.070 |
| `aten::narrow` | 744 | 69148.710 |
| `aten::rsqrt` | 579 | 65653.390 |
| `aten::slice` | 756 | 46786.090 |
| `aten::cat` | 384 | 46141.630 |
| `colqwen.forward.text_prepare_mergers` | 3 | 44808.820 |
| `aten::addmm` | 318 | 43693.955 |
| `aten::neg` | 360 | 40607.240 |
| `aten::item` | 594 | 37883.720 |
| `aclnnAddmm` | 318 | 36184.843 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 11316 | 73725.160 |
| `aclrtLaunchKernelWithHostArgs` | 11316 | 55154.970 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 180 | 37509.990 |
| `PromptFlashAttention_Tiling` | 360 | 26309.950 |
| `aclnnInplaceCopy` | 2127 | 22757.360 |
| `aclnnMul` | 1995 | 19306.620 |
| `InnerPromptFlashAttention` | 180 | 10675.940 |
| `aclnnMean` | 723 | 8487.470 |
| `aclnnAdd` | 885 | 7896.220 |
| `aclnnAdds` | 579 | 5246.370 |
| `aclnnPowTensorScalar` | 579 | 5200.100 |
| `aclnnRsqrt` | 579 | 5025.950 |
| `aclnnCat` | 369 | 4949.730 |
| `aclnnNeg` | 360 | 4901.420 |
| `aclnnNonzeroV2` | 9 | 4429.650 |
| `aclnnMatmul` | 435 | 4413.130 |
| `aclrtGetStreamAttribute` | 9594 | 3859.090 |
| `aclrtSynchronizeStream` | 36 | 3724.280 |
| `aclnnAddmm` | 318 | 3227.440 |
| `aclrtMemcpy` | 48 | 1955.150 |
| `aclnnSilu` | 108 | 1526.470 |
| `aclnnSub` | 144 | 1314.580 |
| `aclrtGetHardwareSyncAddr` | 1494 | 1037.000 |
| `aclnnGeluV2` | 72 | 673.700 |
| `aclnnInplaceCopyGetWorkspaceSize` | 6 | 568.350 |
| `aclnnNonzeroV2GetWorkspaceSize` | 9 | 465.230 |
| `aclnnIndexPutImplGetWorkspaceSize` | 9 | 453.860 |
| `aclnnIndexPutImpl` | 9 | 440.520 |
| `aclrtGetResInCurrentThread` | 408 | 330.950 |
| `aclnnLayerNorm` | 12 | 248.130 |
| `aclrtSynchronizeStreamWithTimeout` | 21 | 229.790 |
| `aclnnEqScalar` | 12 | 209.970 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 143.650 |
| `IndexPutV2_Tiling` | 9 | 129.380 |
| `aclnnEmbedding` | 6 | 122.190 |
| `aclnnInplaceZero` | 12 | 112.040 |
| `aclnnGelu` | 12 | 108.280 |
| `aclnnAll` | 6 | 95.600 |
| `Slice_Tiling` | 6 | 95.230 |
| `NonZero_Tiling` | 18 | 89.890 |

