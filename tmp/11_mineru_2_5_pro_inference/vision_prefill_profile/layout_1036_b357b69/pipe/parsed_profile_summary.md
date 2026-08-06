# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles/layout_1036_b357b69/pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles/layout_1036_b357b69/pipe/liteserver-c001-4_3508133_20260806080527522_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `414089.880 us`
- `Free`: `26882.580 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4266.000 us`
- `Stage`: `440972.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 96 | 164678.820 |
| `Conv3D` | 3 | 108359.720 |
| `MatMulV3` | 387 | 69819.540 |
| `StridedSliceD` | 384 | 22706.500 |
| `AddLayerNorm` | 189 | 9328.440 |
| `Mul` | 390 | 7211.340 |
| `AutomaticBufferFusionOp` | 96 | 6836.220 |
| `Transpose` | 384 | 5918.000 |
| `Add` | 195 | 3496.320 |
| `Cast` | 201 | 3453.760 |
| `ConcatV2D` | 192 | 3359.080 |
| `Unpack` | 96 | 3136.320 |
| `Neg` | 192 | 2209.900 |
| `TransData` | 9 | 1382.900 |
| `Index` | 3 | 470.020 |
| `LayerNormV3` | 6 | 411.220 |
| `NotEqual` | 3 | 253.500 |
| `MatMulV2` | 3 | 205.540 |
| `PadV3` | 12 | 195.100 |
| `MemSet` | 18 | 183.020 |
| `Range` | 12 | 128.360 |
| `Gelu` | 3 | 94.000 |
| `BroadcastTo` | 6 | 63.860 |
| `Cos` | 3 | 45.600 |
| `Sin` | 3 | 44.760 |
| `ConcatD` | 3 | 18.320 |
| `Tile` | 3 | 17.680 |
| `Less` | 3 | 16.080 |
| `Data` | 3 | 13.680 |
| `Pack` | 3 | 9.840 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `Conv3D1` | 3 | 108359.720 |
| `PromptFlashAttention_17` | 3 | 5192.740 |
| `PromptFlashAttention` | 3 | 5163.820 |
| `PromptFlashAttention_27` | 3 | 5163.220 |
| `PromptFlashAttention_24` | 3 | 5161.040 |
| `PromptFlashAttention_15` | 3 | 5160.180 |
| `PromptFlashAttention_1` | 3 | 5159.700 |
| `PromptFlashAttention_25` | 3 | 5157.740 |
| `PromptFlashAttention_8` | 3 | 5156.360 |
| `PromptFlashAttention_28` | 3 | 5155.800 |
| `PromptFlashAttention_14` | 3 | 5155.080 |
| `PromptFlashAttention_12` | 3 | 5152.500 |
| `PromptFlashAttention_23` | 3 | 5150.580 |
| `PromptFlashAttention_18` | 3 | 5149.880 |
| `PromptFlashAttention_9` | 3 | 5149.680 |
| `PromptFlashAttention_26` | 3 | 5148.060 |
| `PromptFlashAttention_4` | 3 | 5147.140 |
| `PromptFlashAttention_16` | 3 | 5144.940 |
| `PromptFlashAttention_13` | 3 | 5144.880 |
| `PromptFlashAttention_29` | 3 | 5142.080 |
| `PromptFlashAttention_5` | 3 | 5141.680 |
| `PromptFlashAttention_2` | 3 | 5141.640 |
| `PromptFlashAttention_3` | 3 | 5141.280 |
| `PromptFlashAttention_7` | 3 | 5137.480 |
| `PromptFlashAttention_10` | 3 | 5136.000 |
| `PromptFlashAttention_30` | 3 | 5135.560 |
| `PromptFlashAttention_19` | 3 | 5135.400 |
| `PromptFlashAttention_6` | 3 | 5132.800 |
| `PromptFlashAttention_20` | 3 | 5131.960 |
| `PromptFlashAttention_31` | 3 | 5128.960 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 3 | 861.520 |
| `MatMulV2_31_to_v3` | 3 | 736.480 |
| `MatMulV2_115_to_v3` | 3 | 735.400 |
| `MatMulV2_111_to_v3` | 3 | 734.660 |
| `MatMulV2_71_to_v3` | 3 | 734.220 |
| `MatMulV2_3_to_v3` | 3 | 734.100 |
| `MatMulV2_59_to_v3` | 3 | 733.580 |
| `MatMulV2_103_to_v3` | 3 | 732.740 |
| `MatMulV2_11_to_v3` | 3 | 732.440 |
| `MatMulV2_123_to_v3` | 3 | 732.420 |
| `MatMulV2_43_to_v3` | 3 | 731.420 |
| `MatMulV2_83_to_v3` | 3 | 731.140 |
| `MatMulV2_55_to_v3` | 3 | 730.780 |
| `MatMulV2_91_to_v3` | 3 | 730.700 |
| `MatMulV2_35_to_v3` | 3 | 730.540 |
| `MatMulV2_75_to_v3` | 3 | 730.480 |
| `MatMulV2_107_to_v3` | 3 | 729.300 |
| `MatMulV2_19_to_v3` | 3 | 729.280 |
| `MatMulV2_79_to_v3` | 3 | 729.260 |
| `MatMulV2_63_to_v3` | 3 | 728.880 |
| `MatMulV2_47_to_v3` | 3 | 728.460 |
| `MatMulV2_39_to_v3` | 3 | 728.120 |
| `MatMulV2_67_to_v3` | 3 | 727.960 |
| `MatMulV2_23_to_v3` | 3 | 727.620 |
| `MatMulV2_95_to_v3` | 3 | 727.360 |
| `MatMulV2_87_to_v3` | 3 | 726.980 |
| `MatMulV2_51_to_v3` | 3 | 726.840 |
| `MatMulV2_7_to_v3` | 3 | 726.680 |
| `MatMulV2_15_to_v3` | 3 | 726.480 |
| `MatMulV2_127_to_v3` | 3 | 725.400 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV3 | "5632,5120;1280,5120;1280" -> "5632,1280" | ND;ND;ND -> ND` | 96 | 23353.380 |
| `MatMulV3 | "5632,1280;5120,1280;5120" -> "5632,5120" | ND;ND;ND -> ND` | 96 | 22306.840 |
| `MatMulV3 | "5632,1280;3840,1280;3840" -> "5632,3840" | ND;ND;ND -> ND` | 96 | 16792.960 |
| `MatMulV3 | "5632,1280;1280,1280;1280" -> "5632,1280" | ND;ND;ND -> ND` | 96 | 6504.840 |
| `MatMulV3 | "1369,5120;5120,5120;5120" -> "1369,5120" | ND;ND;ND -> ND` | 3 | 861.520 |
| `MatMulV2 | "1369,5120;896,5120;896" -> "1369,896" | ND;ND;ND -> ND` | 3 | 205.540 |

### TransData Names
| name | count | total_us |
|---|---:|---:|
| `trans_TransData_2` | 3 | 978.240 |
| `trans_TransData_0` | 3 | 262.760 |
| `trans_TransData_1` | 3 | 141.900 |
| `trans_TransData_1_MemSet` | 3 | 56.820 |

### TransData Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `TransData | "5476,1,80,1,1,16" -> "5476,1280,1,1,1" | NDC1HWC0 -> NCDHW` | 3 | 978.240 |
| `TransData | "5476,3,2,14,14" -> "5476,2,1,14,14,16" | NCDHW -> NDC1HWC0` | 3 | 262.760 |
| `TransData | "1280,3,2,14,14" -> "392,80,16,16" | NCDHW -> FRACTAL_Z_3D` | 3 | 141.900 |
| `MemSet | N/A -> N/A | N/A -> N/A` | 3 | 56.820 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_17` | 1 | 1734.100 |
| `PromptFlashAttention_17` | 1 | 1729.660 |
| `PromptFlashAttention_17` | 1 | 1728.980 |
| `PromptFlashAttention` | 1 | 1726.460 |
| `PromptFlashAttention_25` | 1 | 1723.980 |
| `PromptFlashAttention_27` | 1 | 1723.420 |
| `PromptFlashAttention_1` | 1 | 1723.180 |
| `PromptFlashAttention_15` | 1 | 1722.760 |
| `PromptFlashAttention_24` | 1 | 1722.760 |
| `PromptFlashAttention_8` | 1 | 1721.520 |
| `PromptFlashAttention_27` | 1 | 1720.780 |
| `PromptFlashAttention_28` | 1 | 1720.280 |
| `PromptFlashAttention_28` | 1 | 1720.040 |
| `PromptFlashAttention_14` | 1 | 1719.800 |
| `PromptFlashAttention_23` | 1 | 1719.800 |
| `PromptFlashAttention_13` | 1 | 1719.740 |
| `PromptFlashAttention` | 1 | 1719.680 |
| `PromptFlashAttention_4` | 1 | 1719.640 |
| `PromptFlashAttention_18` | 1 | 1719.560 |
| `PromptFlashAttention_5` | 1 | 1719.500 |
| `PromptFlashAttention_24` | 1 | 1719.400 |
| `PromptFlashAttention_26` | 1 | 1719.280 |
| `PromptFlashAttention_14` | 1 | 1719.280 |
| `PromptFlashAttention_1` | 1 | 1719.240 |
| `PromptFlashAttention_27` | 1 | 1719.020 |
| `PromptFlashAttention_24` | 1 | 1718.880 |
| `PromptFlashAttention_15` | 1 | 1718.860 |
| `PromptFlashAttention_8` | 1 | 1718.620 |
| `PromptFlashAttention_12` | 1 | 1718.560 |
| `PromptFlashAttention_15` | 1 | 1718.560 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.vision_prefill.pipe.step0` | 1 | 151806.300 |
| `mineru.vision_prefill.pipe.step2` | 1 | 145614.880 |
| `mineru.vision_prefill.pipe.step1` | 1 | 145290.540 |
| `aten::item` | 15 | 113107.920 |
| `aten::_local_scalar_dense` | 15 | 112556.540 |
| `npu::npu_conv3d` | 3 | 109799.440 |
| `Conv3D` | 3 | 109799.440 |
| `cache_compiler inference` | 3 | 4658.830 |
| `empty_tensor` | 102 | 3774.290 |
| `TorchNpuGraphBase::Run` | 3 | 2814.920 |
| `aten::as_strided` | 69 | 2101.430 |
| `aten::linear` | 6 | 1906.950 |
| `aten::repeat_interleave` | 3 | 1902.160 |
| `aten::pad` | 12 | 1878.890 |
| `aten::arange` | 12 | 1518.830 |
| `aten::flatten` | 9 | 1512.760 |
| `aten::constant_pad_nd` | 12 | 1447.110 |
| `aten::select` | 24 | 1433.630 |
| `aten::unsqueeze` | 21 | 1316.000 |
| `aten::addmm` | 6 | 1287.740 |
| `aten::reshape` | 9 | 1154.780 |
| `aclnnAddmm` | 6 | 1067.060 |
| `aten::layer_norm` | 3 | 1065.250 |
| `RefreshAtTensorFromGeTensor` | 3 | 1031.100 |
| `aten::gelu` | 6 | 982.970 |
| `aten::cat` | 6 | 950.910 |
| `aten::native_layer_norm` | 3 | 912.150 |
| `aten::clone` | 6 | 911.810 |
| `aten::unbind` | 6 | 907.860 |
| `aten::view` | 24 | 881.660 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 296242.800 |
| `ModelLoad` | 1 | 242539.140 |
| `aclrtSynchronizeStreamWithTimeout` | 15 | 108758.610 |
| `launch` | 1052 | 17010.230 |
| `aclrtMemcpy` | 15 | 2194.830 |
| `aclopCompileAndExecute` | 3 | 1414.340 |
| `aclrtLaunchKernelWithHostArgs` | 108 | 822.750 |
| `InputCopy` | 3 | 302.690 |
| `aclnnConstantPadNd` | 12 | 295.830 |
| `aclnnArange` | 12 | 239.800 |
| `aclnnCat` | 6 | 149.690 |
| `aclnnCatGetWorkspaceSize` | 3 | 147.790 |
| `aclnnMul` | 6 | 112.230 |
| `aclnnLayerNorm` | 3 | 109.040 |
| `aclnnAddmm` | 6 | 106.510 |
| `aclnnRepeatInterleaveGetWorkspaceSize` | 3 | 101.870 |
| `aclnnCumsum` | 3 | 99.350 |
| `opCompile` | 3 | 96.440 |
| `aclnnInplaceCopy` | 6 | 79.720 |
| `aclrtMemcpyAsync` | 6 | 75.940 |
| `aclnnRepeatInterleave` | 3 | 67.400 |
| `aclnnIndex` | 3 | 65.930 |
| `aclnnNeTensor` | 3 | 62.300 |
| `aclnnMax` | 3 | 62.110 |
| `ModelExecute` | 3 | 56.630 |
| `aclnnStack` | 3 | 56.290 |
| `aclnnLtScalar` | 3 | 55.530 |
| `aclnnReduceSum` | 3 | 52.480 |
| `aclnnCos` | 3 | 41.660 |
| `aclrtGetStreamAttribute` | 87 | 40.700 |

