# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/torchair/output/profiles/memory/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/torchair/output/profiles/memory/raw/liteserver-c001-4_7875_20261007104437518_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `368201.914 us`
- `Free`: `129670.393 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2335.750 us`
- `Stage`: `497872.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 180 | 126090.779 |
| `MatMulV3` | 564 | 87982.800 |
| `MatMul` | 108 | 49759.577 |
| `Mul` | 1314 | 16880.377 |
| `AutomaticBufferFusionOp` | 1047 | 12028.538 |
| `Transpose` | 804 | 10458.615 |
| `SplitVD` | 576 | 9646.996 |
| `Add` | 525 | 7625.407 |
| `Square` | 435 | 6676.682 |
| `MaskedScatter` | 3 | 6625.972 |
| `GeluV2` | 72 | 5655.532 |
| `Cast` | 432 | 4478.476 |
| `ReduceMeanD` | 288 | 4328.382 |
| `ConcatV2D` | 360 | 4085.591 |
| `Neg` | 360 | 3507.727 |
| `MatMulV2` | 78 | 3433.731 |
| `Sub` | 144 | 2860.082 |
| `Unpack` | 72 | 1800.597 |
| `ViewCopy` | 6 | 1154.543 |
| `LayerNormV3` | 12 | 751.575 |
| `GatherV2` | 6 | 298.005 |
| `Gelu` | 12 | 292.687 |
| `Index` | 3 | 290.406 |
| `GreaterEqual` | 3 | 171.184 |
| `IndexPutV2` | 9 | 162.641 |
| `ZerosLike` | 12 | 134.282 |
| `RealDiv` | 3 | 87.942 |
| `MemSet` | 9 | 86.923 |
| `Cos` | 6 | 71.321 |
| `Sin` | 6 | 70.002 |
| `TensorMove` | 6 | 69.902 |
| `SelectV2` | 3 | 66.581 |
| `LpNormV2` | 3 | 64.241 |
| `ReduceAny` | 6 | 57.820 |
| `LogicalAnd` | 3 | 52.301 |
| `Range` | 6 | 49.802 |
| `BroadcastTo` | 6 | 47.562 |
| `NonZero` | 9 | 44.701 |
| `ConcatD` | 6 | 40.481 |
| `Tile` | 3 | 32.661 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 3 | 6625.972 |
| `PromptFlashAttention_16` | 6 | 4840.256 |
| `PromptFlashAttention` | 6 | 4831.677 |
| `PromptFlashAttention_18` | 6 | 4828.777 |
| `PromptFlashAttention_19` | 6 | 4819.577 |
| `PromptFlashAttention_2` | 6 | 4818.896 |
| `PromptFlashAttention_12` | 6 | 4815.557 |
| `PromptFlashAttention_8` | 6 | 4815.476 |
| `PromptFlashAttention_4` | 6 | 4814.496 |
| `PromptFlashAttention_5` | 6 | 4805.257 |
| `PromptFlashAttention_17` | 6 | 4804.436 |
| `PromptFlashAttention_20` | 6 | 4804.137 |
| `PromptFlashAttention_15` | 6 | 4803.616 |
| `PromptFlashAttention_3` | 6 | 4803.056 |
| `PromptFlashAttention_1` | 6 | 4802.595 |
| `PromptFlashAttention_7` | 6 | 4800.695 |
| `PromptFlashAttention_22` | 6 | 4796.756 |
| `PromptFlashAttention_14` | 6 | 4793.436 |
| `PromptFlashAttention_9` | 6 | 4791.096 |
| `PromptFlashAttention_21` | 6 | 4790.916 |
| `PromptFlashAttention_10` | 6 | 4782.496 |
| `PromptFlashAttention_23` | 6 | 4779.755 |
| `PromptFlashAttention_11` | 6 | 4778.115 |
| `PromptFlashAttention_6` | 6 | 4773.515 |
| `PromptFlashAttention_13` | 6 | 4768.715 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 24 | 3462.847 |
| `MatMul_34` | 3 | 1386.227 |
| `MatMul_78` | 3 | 1385.348 |
| `MatMul_54` | 3 | 1385.109 |
| `MatMul_70` | 3 | 1385.007 |
| `MatMul_26` | 3 | 1384.588 |
| `MatMul_46` | 3 | 1384.089 |
| `MatMul_2` | 3 | 1384.049 |
| `MatMul_18` | 3 | 1384.027 |
| `MatMul_38` | 3 | 1383.928 |
| `MatMul_114` | 3 | 1383.648 |
| `MatMul_142` | 3 | 1383.028 |
| `MatMul_90` | 3 | 1382.708 |
| `MatMul_74` | 3 | 1382.268 |
| `MatMul_66` | 3 | 1382.247 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 24 | 3462.847 |
| `MatMul_34` | 3 | 1386.227 |
| `MatMul_78` | 3 | 1385.348 |
| `MatMul_54` | 3 | 1385.109 |
| `MatMul_70` | 3 | 1385.007 |
| `MatMul_26` | 3 | 1384.588 |
| `MatMul_46` | 3 | 1384.089 |
| `MatMul_2` | 3 | 1384.049 |
| `MatMul_18` | 3 | 1384.027 |
| `MatMul_38` | 3 | 1383.928 |
| `MatMul_114` | 3 | 1383.648 |
| `MatMul_142` | 3 | 1383.028 |
| `MatMul_90` | 3 | 1382.708 |
| `MatMul_74` | 3 | 1382.268 |
| `MatMul_66` | 3 | 1382.247 |
| `MatMul_130` | 3 | 1382.208 |
| `MatMul_58` | 3 | 1382.147 |
| `MatMul_82` | 3 | 1382.048 |
| `MatMul_42` | 3 | 1381.928 |
| `MatMul_62` | 3 | 1381.687 |
| `MatMul_14` | 3 | 1381.567 |
| `MatMul_106` | 3 | 1381.528 |
| `MatMul_134` | 3 | 1381.428 |
| `MatMul_6` | 3 | 1381.308 |
| `MatMul_10` | 3 | 1381.287 |
| `MatMul_30` | 3 | 1381.187 |
| `MatMul_98` | 3 | 1381.187 |
| `MatMul_122` | 3 | 1381.148 |
| `MatMul_138` | 3 | 1380.927 |
| `MatMul_94` | 3 | 1380.727 |
| `MatMul_110` | 3 | 1380.587 |
| `MatMul_50` | 3 | 1380.509 |
| `MatMul_86` | 3 | 1380.309 |
| `MatMul_118` | 3 | 1380.167 |
| `MatMul_126` | 3 | 1380.167 |
| `MatMul_102` | 3 | 1380.028 |
| `MatMul_22` | 3 | 1379.227 |
| `MatMul_135_to_v3` | 3 | 795.597 |
| `MatMul_51_to_v3` | 3 | 794.537 |
| `MatMul_71_to_v3` | 3 | 792.096 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49759.577 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28307.012 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15837.678 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12579.150 |
| `MatMulV3 | "5040,4096;1024,4096;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 10378.229 |
| `MatMulV3 | "5040,1024;4096,1024;4096" -> "5040,4096" | ND;ND;ND -> ND` | 72 | 9918.277 |
| `MatMulV3 | "5040,1024;3072,1024;3072" -> "5040,3072" | ND;ND;ND -> ND` | 72 | 7499.607 |
| `MatMulV2 | "5040,1024;1024,1024;1024" -> "5040,1024" | ND;ND;ND -> ND` | 72 | 2974.440 |
| `MatMulV3 | "1260,4096;4096,4096;4096" -> "1260,4096" | ND;ND;ND -> ND` | 12 | 2048.820 |
| `MatMulV3 | "1260,4096;2560,4096;2560" -> "1260,2560" | ND;ND;ND -> ND` | 12 | 1414.027 |
| `MatMulV2 | "1274,2560;2560,2560;2560" -> "1274,2560" | ND;ND;ND -> ND` | 3 | 276.286 |
| `MatMulV2 | "5040,1536;1024,1536;1024" -> "5040,1024" | ND;ND;ND -> ND` | 3 | 183.005 |
| `Mul | "3,1,64,1;3,1,1,1274" -> "3,1,64,1274" | ND;ND -> ND` | 3 | 35.841 |
| `BroadcastTo | "1,1,64,1;4" -> "3,1,64,1" | ND;ND -> ND` | 3 | 5.460 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2213.004 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2206.864 |
| `aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter` | 1 | 2206.104 |
| `PromptFlashAttention_16` | 1 | 1307.466 |
| `PromptFlashAttention_20` | 1 | 1307.147 |
| `PromptFlashAttention_16` | 1 | 1307.106 |
| `PromptFlashAttention_8` | 1 | 1306.226 |
| `PromptFlashAttention_17` | 1 | 1306.226 |
| `PromptFlashAttention_21` | 1 | 1305.246 |
| `PromptFlashAttention_7` | 1 | 1304.766 |
| `PromptFlashAttention_14` | 1 | 1304.706 |
| `PromptFlashAttention_18` | 1 | 1304.166 |
| `PromptFlashAttention_1` | 1 | 1304.066 |
| `PromptFlashAttention` | 1 | 1303.766 |
| `PromptFlashAttention_2` | 1 | 1303.366 |
| `PromptFlashAttention` | 1 | 1303.106 |
| `PromptFlashAttention_15` | 1 | 1302.686 |
| `PromptFlashAttention_12` | 1 | 1302.426 |
| `PromptFlashAttention` | 1 | 1301.646 |
| `PromptFlashAttention_14` | 1 | 1301.526 |
| `PromptFlashAttention_4` | 1 | 1301.066 |
| `PromptFlashAttention_2` | 1 | 1300.986 |
| `PromptFlashAttention_17` | 1 | 1300.646 |
| `PromptFlashAttention_2` | 1 | 1300.246 |
| `PromptFlashAttention_8` | 1 | 1299.946 |
| `PromptFlashAttention_16` | 1 | 1299.646 |
| `PromptFlashAttention_18` | 1 | 1299.426 |
| `PromptFlashAttention_22` | 1 | 1299.266 |
| `PromptFlashAttention_15` | 1 | 1299.246 |
| `PromptFlashAttention_12` | 1 | 1298.746 |
| `PromptFlashAttention_20` | 1 | 1298.726 |
| `PromptFlashAttention_5` | 1 | 1298.506 |
| `PromptFlashAttention_23` | 1 | 1298.446 |
| `PromptFlashAttention_9` | 1 | 1297.346 |
| `PromptFlashAttention_14` | 1 | 1296.926 |
| `PromptFlashAttention_19` | 1 | 1296.766 |
| `PromptFlashAttention_12` | 1 | 1296.226 |
| `PromptFlashAttention_9` | 1 | 1296.026 |
| `PromptFlashAttention_4` | 1 | 1295.906 |
| `PromptFlashAttention_23` | 1 | 1295.846 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.forward.text_prepare_mergers` | 3 | 201182.830 |
| `colqwen.torchair.forward.step0` | 1 | 170996.720 |
| `aten::index_put_` | 12 | 166218.760 |
| `aten::_index_put_impl_` | 12 | 165894.730 |
| `colqwen.torchair.forward.step2` | 1 | 163597.130 |
| `colqwen.torchair.forward.step1` | 1 | 163348.360 |
| `colqwen.forward.vision_prepare` | 3 | 96395.760 |
| `aten::to` | 129 | 16739.480 |
| `aten::_to_copy` | 108 | 13150.950 |
| `cache_compiler inference` | 6 | 10194.590 |
| `empty_tensor` | 288 | 9377.560 |
| `aten::copy_` | 138 | 8699.572 |
| `aten::as_strided` | 303 | 7933.100 |
| `aten::linear` | 30 | 7739.410 |
| `aten::arange` | 78 | 7095.580 |
| `aten::masked_scatter` | 3 | 6752.495 |
| `aten::masked_scatter_` | 3 | 6719.294 |
| `aclnnInplaceMaskedScatter` | 3 | 6719.294 |
| `TorchNpuGraphBase::Run` | 6 | 6492.670 |
| `colqwen.forward.vision_transformer` | 3 | 6454.960 |
| `aten::empty` | 138 | 6244.150 |
| `aten::unsqueeze` | 111 | 5682.140 |
| `colqwen.forward.text_transformer` | 3 | 4819.100 |
| `aten::cat` | 24 | 4559.560 |
| `aten::addmm` | 30 | 4278.233 |
| `aten::mul` | 45 | 4126.260 |
| `aclnnAddmm` | 30 | 3922.138 |
| `aten::layer_norm` | 12 | 3735.780 |
| `aten::add` | 54 | 3729.380 |
| `aten::flatten` | 42 | 3553.470 |
| `aten::native_layer_norm` | 12 | 3212.820 |
| `aten::repeat` | 6 | 3202.490 |
| `aten::select` | 60 | 3195.620 |
| `RefreshAtTensorFromGeTensor` | 6 | 3144.070 |
| `aten::clone` | 21 | 3027.010 |
| `aten::empty_strided` | 81 | 3007.360 |
| `aten::view` | 108 | 3007.110 |
| `aten::reshape` | 36 | 2938.750 |
| `colqwen.forward.retrieval_projection` | 3 | 2892.970 |
| `aten::gelu` | 24 | 2799.840 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 2 | 422588.350 |
| `launch` | 2720 | 191690.470 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 184757.220 |
| `aclnnNonzeroV2` | 9 | 159103.940 |
| `aclrtSynchronizeStream` | 36 | 158397.250 |
| `aclrtLaunchKernelWithHostArgs` | 308 | 2039.530 |
| `aclrtMemcpy` | 48 | 1887.520 |
| `aclnnInplaceCopy` | 33 | 1051.450 |
| `aclnnNonzeroV2GetWorkspaceSize` | 9 | 679.430 |
| `aclnnInplaceCopyGetWorkspaceSize` | 6 | 656.900 |
| `aclnnIndexPutImpl` | 9 | 536.520 |
| `InputCopy` | 6 | 508.930 |
| `aclnnIndexPutImplGetWorkspaceSize` | 9 | 503.520 |
| `aclnnAddmm` | 30 | 398.970 |
| `aclnnLayerNorm` | 12 | 353.600 |
| `aclnnEqScalar` | 12 | 297.180 |
| `aclrtSynchronizeStreamWithTimeout` | 21 | 271.420 |
| `NonZero_Tiling` | 18 | 159.300 |
| `aclnnGelu` | 12 | 154.910 |
| `aclnnMul` | 9 | 141.460 |
| `IndexPutV2_Tiling` | 9 | 141.190 |
| `aclnnCat` | 9 | 139.080 |
| `aclnnInplaceZero` | 12 | 135.400 |
| `ModelExecute` | 6 | 134.900 |
| `aclnnAdd` | 12 | 134.310 |
| `aclnnEmbedding` | 6 | 125.600 |
| `aclrtGetStreamAttribute` | 237 | 122.800 |
| `aclnnAll` | 6 | 113.280 |
| `aclnnReduceSum` | 3 | 105.680 |
| `aclnnInplaceMaskedScatter` | 3 | 102.170 |
| `aclnnAny` | 6 | 92.840 |
| `Transpose_Tiling` | 12 | 92.080 |
| `Slice_Tiling` | 6 | 90.970 |
| `aclnnArange` | 6 | 84.750 |
| `aclnnCos` | 6 | 73.190 |
| `aclnnSin` | 6 | 68.250 |
| `aclnnInplaceFillScalar` | 6 | 66.320 |
| `step_info` | 12 | 61.670 |
| `aclrtGetResInCurrentThread` | 48 | 55.720 |
| `aclnnMatmul` | 3 | 55.410 |

