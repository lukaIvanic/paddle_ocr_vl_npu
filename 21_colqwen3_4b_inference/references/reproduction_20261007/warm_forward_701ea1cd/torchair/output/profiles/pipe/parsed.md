# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/torchair/output/profiles/pipe/raw/liteserver-c001-4_7875_20261007104425958_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `368119.074 us`
- `Free`: `124256.104 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `1985.750 us`
- `Stage`: `492375.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 180 | 125934.820 |
| `MatMulV3` | 564 | 88094.521 |
| `MatMul` | 108 | 49774.617 |
| `Mul` | 1314 | 16899.831 |
| `AutomaticBufferFusionOp` | 1047 | 12030.436 |
| `Transpose` | 804 | 10481.807 |
| `SplitVD` | 576 | 9637.696 |
| `Add` | 525 | 7627.550 |
| `Square` | 435 | 6677.045 |
| `MaskedScatter` | 3 | 6620.272 |
| `GeluV2` | 72 | 5655.672 |
| `Cast` | 432 | 4494.359 |
| `ReduceMeanD` | 288 | 4333.832 |
| `ConcatV2D` | 360 | 4085.355 |
| `Neg` | 360 | 3507.902 |
| `MatMulV2` | 78 | 3435.869 |
| `Sub` | 144 | 2858.131 |
| `Unpack` | 72 | 1804.797 |
| `ViewCopy` | 6 | 989.701 |
| `LayerNormV3` | 12 | 785.055 |
| `GatherV2` | 6 | 292.846 |
| `Index` | 3 | 292.826 |
| `Gelu` | 12 | 292.486 |
| `GreaterEqual` | 3 | 169.823 |
| `IndexPutV2` | 9 | 164.881 |
| `ZerosLike` | 12 | 133.785 |
| `MemSet` | 9 | 90.521 |
| `RealDiv` | 3 | 88.542 |
| `Cos` | 6 | 72.541 |
| `Sin` | 6 | 71.061 |
| `TensorMove` | 6 | 69.922 |
| `SelectV2` | 3 | 67.242 |
| `LpNormV2` | 3 | 66.241 |
| `ReduceAny` | 6 | 61.122 |
| `LogicalAnd` | 3 | 52.361 |
| `Range` | 6 | 49.880 |
| `BroadcastTo` | 6 | 47.301 |
| `NonZero` | 9 | 45.240 |
| `ConcatD` | 6 | 40.721 |
| `Equal` | 12 | 34.883 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 3 | 6620.272 |
| `PromptFlashAttention_2` | 6 | 4830.557 |
| `PromptFlashAttention_16` | 6 | 4819.936 |
| `PromptFlashAttention_19` | 6 | 4810.598 |
| `PromptFlashAttention_3` | 6 | 4808.636 |
| `PromptFlashAttention_1` | 6 | 4806.617 |
| `PromptFlashAttention` | 6 | 4805.596 |
| `PromptFlashAttention_18` | 6 | 4805.295 |
| `PromptFlashAttention_8` | 6 | 4802.077 |
| `PromptFlashAttention_15` | 6 | 4800.476 |
| `PromptFlashAttention_17` | 6 | 4800.457 |
| `PromptFlashAttention_5` | 6 | 4799.635 |
| `PromptFlashAttention_21` | 6 | 4799.196 |
| `PromptFlashAttention_4` | 6 | 4799.156 |
| `PromptFlashAttention_9` | 6 | 4795.816 |
| `PromptFlashAttention_12` | 6 | 4791.817 |
| `PromptFlashAttention_20` | 6 | 4790.556 |
| `PromptFlashAttention_11` | 6 | 4789.196 |
| `PromptFlashAttention_7` | 6 | 4781.714 |
| `PromptFlashAttention_10` | 6 | 4780.356 |
| `PromptFlashAttention_23` | 6 | 4777.756 |
| `PromptFlashAttention_6` | 6 | 4773.657 |
| `PromptFlashAttention_13` | 6 | 4773.154 |
| `PromptFlashAttention_22` | 6 | 4764.736 |
| `PromptFlashAttention_14` | 6 | 4761.215 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 24 | 3478.491 |
| `MatMul_34` | 3 | 1386.508 |
| `MatMul_78` | 3 | 1386.429 |
| `MatMul_2` | 3 | 1385.827 |
| `MatMul_38` | 3 | 1385.688 |
| `MatMul_46` | 3 | 1385.088 |
| `MatMul_142` | 3 | 1384.827 |
| `MatMul_130` | 3 | 1384.567 |
| `MatMul_26` | 3 | 1384.008 |
| `MatMul_18` | 3 | 1383.907 |
| `MatMul_70` | 3 | 1383.808 |
| `MatMul_134` | 3 | 1383.227 |
| `MatMul_90` | 3 | 1382.868 |
| `MatMul_98` | 3 | 1382.867 |
| `MatMul_50` | 3 | 1382.808 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 24 | 3478.491 |
| `MatMul_34` | 3 | 1386.508 |
| `MatMul_78` | 3 | 1386.429 |
| `MatMul_2` | 3 | 1385.827 |
| `MatMul_38` | 3 | 1385.688 |
| `MatMul_46` | 3 | 1385.088 |
| `MatMul_142` | 3 | 1384.827 |
| `MatMul_130` | 3 | 1384.567 |
| `MatMul_26` | 3 | 1384.008 |
| `MatMul_18` | 3 | 1383.907 |
| `MatMul_70` | 3 | 1383.808 |
| `MatMul_134` | 3 | 1383.227 |
| `MatMul_90` | 3 | 1382.868 |
| `MatMul_98` | 3 | 1382.867 |
| `MatMul_50` | 3 | 1382.808 |
| `MatMul_10` | 3 | 1382.788 |
| `MatMul_30` | 3 | 1382.788 |
| `MatMul_42` | 3 | 1382.748 |
| `MatMul_6` | 3 | 1382.648 |
| `MatMul_94` | 3 | 1382.347 |
| `MatMul_138` | 3 | 1382.269 |
| `MatMul_110` | 3 | 1382.167 |
| `MatMul_54` | 3 | 1382.147 |
| `MatMul_114` | 3 | 1381.807 |
| `MatMul_66` | 3 | 1381.488 |
| `MatMul_62` | 3 | 1381.427 |
| `MatMul_22` | 3 | 1381.408 |
| `MatMul_58` | 3 | 1381.389 |
| `MatMul_86` | 3 | 1381.388 |
| `MatMul_102` | 3 | 1381.168 |
| `MatMul_82` | 3 | 1381.088 |
| `MatMul_14` | 3 | 1380.788 |
| `MatMul_126` | 3 | 1380.588 |
| `MatMul_118` | 3 | 1380.527 |
| `MatMul_106` | 3 | 1380.247 |
| `MatMul_74` | 3 | 1379.668 |
| `MatMul_122` | 3 | 1379.307 |
| `MatMul_135_to_v3` | 3 | 796.015 |
| `MatMul_31_to_v3` | 3 | 793.796 |
| `MatMul_131_to_v3` | 3 | 793.775 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49774.617 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28341.860 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15860.356 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12600.155 |
| `MatMulV3 | "5040,4096;1024,4096;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 10389.268 |
| `MatMulV3 | "5040,1024;4096,1024;4096" -> "5040,4096" | ND;ND;ND -> ND` | 72 | 9924.160 |
| `MatMulV3 | "5040,1024;3072,1024;3072" -> "5040,3072" | ND;ND;ND -> ND` | 72 | 7500.231 |
| `MatMulV2 | "5040,1024;1024,1024;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 2980.680 |
| `MatMulV3 | "1260,4096;4096,4096;4096" -> "1260,4096" | ND;ND;ND -> ND` | 12 | 2057.982 |
| `MatMulV3 | "1260,4096;2560,4096;2560" -> "1260,2560" | ND;ND;ND -> ND` | 12 | 1420.509 |
| `MatMulV2 | "1274,2560;2560,2560;2560" -> "1274,2560" | ND;ND;ND -> ND` | 3 | 277.486 |
| `MatMulV2 | "5040,1536;1024,1536;1024" -> "5040,1024" | ND;ND;ND -> ND` | 3 | 177.703 |
| `Mul | "3,1,64,1;3,1,1,1274" -> "3,1,64,1274" | ND;ND -> ND` | 3 | 34.382 |
| `BroadcastTo | "1,1,64,1;4" -> "3,1,64,1" | ND;ND -> ND` | 3 | 5.140 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2207.964 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2207.364 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2204.944 |
| `PromptFlashAttention_16` | 1 | 1308.226 |
| `PromptFlashAttention_16` | 1 | 1306.506 |
| `PromptFlashAttention_12` | 1 | 1305.486 |
| `PromptFlashAttention_9` | 1 | 1304.626 |
| `PromptFlashAttention_15` | 1 | 1304.486 |
| `PromptFlashAttention_1` | 1 | 1304.306 |
| `PromptFlashAttention_2` | 1 | 1303.946 |
| `PromptFlashAttention_11` | 1 | 1303.826 |
| `PromptFlashAttention_2` | 1 | 1303.726 |
| `PromptFlashAttention_2` | 1 | 1303.546 |
| `PromptFlashAttention_18` | 1 | 1302.706 |
| `PromptFlashAttention_21` | 1 | 1302.466 |
| `PromptFlashAttention_15` | 1 | 1301.966 |
| `PromptFlashAttention` | 1 | 1301.206 |
| `PromptFlashAttention_17` | 1 | 1300.506 |
| `PromptFlashAttention_19` | 1 | 1300.266 |
| `PromptFlashAttention_20` | 1 | 1299.706 |
| `PromptFlashAttention_17` | 1 | 1299.606 |
| `PromptFlashAttention_1` | 1 | 1299.166 |
| `PromptFlashAttention_10` | 1 | 1298.946 |
| `PromptFlashAttention_8` | 1 | 1298.846 |
| `PromptFlashAttention_3` | 1 | 1298.685 |
| `PromptFlashAttention_8` | 1 | 1298.126 |
| `PromptFlashAttention_23` | 1 | 1297.626 |
| `PromptFlashAttention_21` | 1 | 1297.526 |
| `PromptFlashAttention_3` | 1 | 1297.206 |
| `PromptFlashAttention_23` | 1 | 1297.106 |
| `PromptFlashAttention_23` | 1 | 1297.006 |
| `PromptFlashAttention_16` | 1 | 1296.966 |
| `PromptFlashAttention_1` | 1 | 1296.806 |
| `PromptFlashAttention_3` | 1 | 1296.406 |
| `PromptFlashAttention_7` | 1 | 1296.365 |
| `PromptFlashAttention_15` | 1 | 1296.306 |
| `PromptFlashAttention_5` | 1 | 1296.066 |
| `PromptFlashAttention_14` | 1 | 1295.926 |
| `PromptFlashAttention_20` | 1 | 1295.906 |
| `PromptFlashAttention_12` | 1 | 1295.686 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.forward.text_prepare_mergers` | 3 | 200103.840 |
| `colqwen.torchair.forward.step0` | 1 | 168236.610 |
| `aten::index_put_` | 12 | 166358.190 |
| `aten::_index_put_impl_` | 12 | 166043.820 |
| `colqwen.torchair.forward.step2` | 1 | 162307.050 |
| `colqwen.torchair.forward.step1` | 1 | 161711.830 |
| `colqwen.forward.vision_prepare` | 3 | 93304.620 |
| `aten::to` | 129 | 16196.180 |
| `aten::_to_copy` | 108 | 12658.590 |
| `cache_compiler inference` | 6 | 8768.840 |
| `empty_tensor` | 288 | 8741.000 |
| `aten::copy_` | 138 | 8126.805 |
| `aten::as_strided` | 303 | 7718.560 |
| `aten::linear` | 30 | 7508.650 |
| `aten::arange` | 78 | 6926.080 |
| `aten::masked_scatter` | 3 | 6748.355 |
| `aten::masked_scatter_` | 3 | 6714.514 |
| `aclnnInplaceMaskedScatter` | 3 | 6714.514 |
| `aten::empty` | 138 | 6076.700 |
| `TorchNpuGraphBase::Run` | 6 | 5738.130 |
| `colqwen.forward.vision_transformer` | 3 | 5625.940 |
| `aten::unsqueeze` | 111 | 5525.990 |
| `aten::cat` | 24 | 4541.650 |
| `aten::addmm` | 30 | 4222.606 |
| `colqwen.forward.text_transformer` | 3 | 4170.700 |
| `aten::mul` | 45 | 4015.210 |
| `aclnnAddmm` | 30 | 3933.679 |
| `aten::layer_norm` | 12 | 3583.780 |
| `aten::flatten` | 42 | 3476.430 |
| `aten::add` | 54 | 3449.540 |
| `aten::repeat` | 6 | 3140.720 |
| `aten::select` | 60 | 3105.490 |
| `aten::native_layer_norm` | 12 | 3078.000 |
| `RefreshAtTensorFromGeTensor` | 6 | 3051.930 |
| `aten::view` | 108 | 2964.910 |
| `aten::reshape` | 36 | 2894.000 |
| `aten::clone` | 21 | 2878.930 |
| `aten::empty_strided` | 81 | 2857.560 |
| `colqwen.forward.retrieval_projection` | 3 | 2822.530 |
| `aten::gelu` | 24 | 2800.980 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 2 | 422588.350 |
| `launch` | 2721 | 192827.510 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 184904.410 |
| `aclnnNonzeroV2` | 9 | 159802.640 |
| `aclrtSynchronizeStream` | 36 | 159113.300 |
| `aclrtLaunchKernelWithHostArgs` | 309 | 2261.920 |
| `aclrtMemcpy` | 48 | 1781.420 |
| `aclnnInplaceCopy` | 33 | 1019.110 |
| `aclnnAdd` | 12 | 555.320 |
| `aclnnIndexPutImpl` | 9 | 500.190 |
| `aclnnInplaceCopyGetWorkspaceSize` | 6 | 452.970 |
| `aclnnNonzeroV2GetWorkspaceSize` | 9 | 430.750 |
| `aclnnIndexPutImplGetWorkspaceSize` | 9 | 385.260 |
| `aclnnAddmm` | 30 | 369.880 |
| `InputCopy` | 6 | 362.440 |
| `aclnnLayerNorm` | 12 | 324.680 |
| `aclrtSynchronizeStreamWithTimeout` | 21 | 250.600 |
| `aclnnEqScalar` | 12 | 247.530 |
| `aclnnMul` | 9 | 140.790 |
| `aclnnInplaceZero` | 12 | 127.830 |
| `aclnnGelu` | 12 | 126.520 |
| `aclrtGetStreamAttribute` | 237 | 116.550 |
| `IndexPutV2_Tiling` | 9 | 113.480 |
| `ModelExecute` | 6 | 108.540 |
| `aclnnInplaceMaskedScatter` | 3 | 107.170 |
| `aclnnAll` | 6 | 104.080 |
| `aclnnEmbedding` | 6 | 102.130 |
| `aclnnCat` | 9 | 100.760 |
| `aclnnAny` | 6 | 92.400 |
| `aclnnReduceSum` | 3 | 87.070 |
| `aclnnArange` | 6 | 86.550 |
| `NonZero_Tiling` | 18 | 80.270 |
| `aclnnSin` | 6 | 78.090 |
| `Slice_Tiling` | 6 | 77.850 |
| `aclnnCos` | 6 | 60.020 |
| `aclnnInplaceFillScalar` | 6 | 59.960 |
| `Transpose_Tiling` | 12 | 56.670 |
| `step_info` | 12 | 54.530 |
| `aclnnMatmul` | 3 | 54.120 |
| `aclrtGetResInCurrentThread` | 48 | 51.100 |

