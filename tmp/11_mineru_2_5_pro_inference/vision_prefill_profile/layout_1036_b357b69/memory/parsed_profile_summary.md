# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles/layout_1036_b357b69/memory`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles/layout_1036_b357b69/memory/liteserver-c001-4_3508133_20260806080537046_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `414143.260 us`
- `Free`: `22293.360 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3377.250 us`
- `Stage`: `436436.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 96 | 164677.640 |
| `Conv3D` | 3 | 108454.520 |
| `MatMulV3` | 387 | 69892.620 |
| `StridedSliceD` | 384 | 22697.920 |
| `AddLayerNorm` | 189 | 9322.400 |
| `Mul` | 390 | 7212.860 |
| `AutomaticBufferFusionOp` | 96 | 6831.480 |
| `Transpose` | 384 | 5883.380 |
| `Add` | 195 | 3491.980 |
| `Cast` | 201 | 3451.040 |
| `ConcatV2D` | 192 | 3357.080 |
| `Unpack` | 96 | 3122.260 |
| `Neg` | 192 | 2209.820 |
| `TransData` | 9 | 1357.260 |
| `Index` | 3 | 459.780 |
| `LayerNormV3` | 6 | 409.800 |
| `NotEqual` | 3 | 252.820 |
| `MatMulV2` | 3 | 204.100 |
| `PadV3` | 12 | 191.520 |
| `MemSet` | 18 | 182.480 |
| `Range` | 12 | 135.300 |
| `Gelu` | 3 | 93.860 |
| `BroadcastTo` | 6 | 65.680 |
| `Cos` | 3 | 45.180 |
| `Sin` | 3 | 44.820 |
| `ConcatD` | 3 | 18.460 |
| `Tile` | 3 | 16.440 |
| `Less` | 3 | 15.500 |
| `Data` | 3 | 14.380 |
| `Pack` | 3 | 9.560 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `Conv3D1` | 3 | 108454.520 |
| `PromptFlashAttention_17` | 3 | 5189.060 |
| `PromptFlashAttention_27` | 3 | 5173.540 |
| `PromptFlashAttention_25` | 3 | 5169.480 |
| `PromptFlashAttention_15` | 3 | 5159.740 |
| `PromptFlashAttention_18` | 3 | 5155.080 |
| `PromptFlashAttention_14` | 3 | 5154.480 |
| `PromptFlashAttention` | 3 | 5153.120 |
| `PromptFlashAttention_16` | 3 | 5151.860 |
| `PromptFlashAttention_29` | 3 | 5151.340 |
| `PromptFlashAttention_8` | 3 | 5151.120 |
| `PromptFlashAttention_1` | 3 | 5150.720 |
| `PromptFlashAttention_12` | 3 | 5150.700 |
| `PromptFlashAttention_31` | 3 | 5150.020 |
| `PromptFlashAttention_20` | 3 | 5149.500 |
| `PromptFlashAttention_3` | 3 | 5149.480 |
| `PromptFlashAttention_2` | 3 | 5148.940 |
| `PromptFlashAttention_24` | 3 | 5148.680 |
| `PromptFlashAttention_9` | 3 | 5147.720 |
| `PromptFlashAttention_28` | 3 | 5146.180 |
| `PromptFlashAttention_26` | 3 | 5145.120 |
| `PromptFlashAttention_4` | 3 | 5141.700 |
| `PromptFlashAttention_13` | 3 | 5141.220 |
| `PromptFlashAttention_6` | 3 | 5135.980 |
| `PromptFlashAttention_30` | 3 | 5135.640 |
| `PromptFlashAttention_7` | 3 | 5132.420 |
| `PromptFlashAttention_23` | 3 | 5132.300 |
| `PromptFlashAttention_10` | 3 | 5131.440 |
| `PromptFlashAttention_11` | 3 | 5129.540 |
| `PromptFlashAttention_5` | 3 | 5127.500 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 3 | 863.260 |
| `MatMulV2_115_to_v3` | 3 | 734.660 |
| `MatMulV2_103_to_v3` | 3 | 734.320 |
| `MatMulV2_31_to_v3` | 3 | 734.260 |
| `MatMulV2_59_to_v3` | 3 | 733.420 |
| `MatMulV2_71_to_v3` | 3 | 733.220 |
| `MatMulV2_11_to_v3` | 3 | 733.200 |
| `MatMulV2_35_to_v3` | 3 | 733.100 |
| `MatMulV2_3_to_v3` | 3 | 733.080 |
| `MatMulV2_123_to_v3` | 3 | 732.740 |
| `MatMulV2_107_to_v3` | 3 | 731.040 |
| `MatMulV2_83_to_v3` | 3 | 730.980 |
| `MatMulV2_39_to_v3` | 3 | 730.080 |
| `MatMulV2_91_to_v3` | 3 | 730.020 |
| `MatMulV2_15_to_v3` | 3 | 729.940 |
| `MatMulV2_23_to_v3` | 3 | 729.760 |
| `MatMulV2_75_to_v3` | 3 | 729.760 |
| `MatMulV2_111_to_v3` | 3 | 729.620 |
| `MatMulV2_19_to_v3` | 3 | 728.820 |
| `MatMulV2_43_to_v3` | 3 | 728.800 |
| `MatMulV2_55_to_v3` | 3 | 728.600 |
| `MatMulV2_99_to_v3` | 3 | 728.160 |
| `MatMulV2_47_to_v3` | 3 | 728.120 |
| `MatMulV2_127_to_v3` | 3 | 728.120 |
| `MatMulV2_79_to_v3` | 3 | 728.020 |
| `MatMulV2_87_to_v3` | 3 | 727.860 |
| `MatMulV2_63_to_v3` | 3 | 727.800 |
| `MatMulV2_67_to_v3` | 3 | 727.640 |
| `MatMulV2_7_to_v3` | 3 | 727.380 |
| `MatMulV2_95_to_v3` | 3 | 727.140 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV3 | "5632,5120;1280,5120;1280" -> "5632,1280" | ND;ND;ND -> ND` | 96 | 23357.080 |
| `MatMulV3 | "5632,1280;5120,1280;5120" -> "5632,5120" | ND;ND;ND -> ND` | 96 | 22354.580 |
| `MatMulV3 | "5632,1280;3840,1280;3840" -> "5632,3840" | ND;ND;ND -> ND` | 96 | 16833.920 |
| `MatMulV3 | "5632,1280;1280,1280;1280" -> "5632,1280" | ND;ND;ND -> ND` | 96 | 6483.780 |
| `MatMulV3 | "1369,5120;5120,5120;5120" -> "1369,5120" | ND;ND;ND -> ND` | 3 | 863.260 |
| `MatMulV2 | "1369,5120;896,5120;896" -> "1369,896" | ND;ND;ND -> ND` | 3 | 204.100 |

### TransData Names
| name | count | total_us |
|---|---:|---:|
| `trans_TransData_2` | 3 | 969.120 |
| `trans_TransData_0` | 3 | 250.740 |
| `trans_TransData_1` | 3 | 137.400 |
| `trans_TransData_1_MemSet` | 3 | 57.220 |

### TransData Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `TransData | "5476,1,80,1,1,16" -> "5476,1280,1,1,1" | NDC1HWC0 -> NCDHW` | 3 | 969.120 |
| `TransData | "5476,3,2,14,14" -> "5476,2,1,14,14,16" | NCDHW -> NDC1HWC0` | 3 | 250.740 |
| `TransData | "1280,3,2,14,14" -> "392,80,16,16" | NCDHW -> FRACTAL_Z_3D` | 3 | 137.400 |
| `MemSet | N/A -> N/A | N/A -> N/A` | 3 | 57.220 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_17` | 1 | 1732.540 |
| `PromptFlashAttention_17` | 1 | 1729.180 |
| `PromptFlashAttention_27` | 1 | 1728.660 |
| `PromptFlashAttention_17` | 1 | 1727.340 |
| `PromptFlashAttention_25` | 1 | 1727.000 |
| `PromptFlashAttention_27` | 1 | 1724.600 |
| `PromptFlashAttention_14` | 1 | 1724.240 |
| `PromptFlashAttention_25` | 1 | 1724.200 |
| `PromptFlashAttention_31` | 1 | 1723.280 |
| `PromptFlashAttention_3` | 1 | 1722.100 |
| `PromptFlashAttention_18` | 1 | 1721.600 |
| `PromptFlashAttention_15` | 1 | 1721.260 |
| `PromptFlashAttention_16` | 1 | 1721.100 |
| `PromptFlashAttention_8` | 1 | 1720.860 |
| `PromptFlashAttention_18` | 1 | 1720.460 |
| `PromptFlashAttention_27` | 1 | 1720.280 |
| `PromptFlashAttention_4` | 1 | 1719.680 |
| `PromptFlashAttention_9` | 1 | 1719.680 |
| `PromptFlashAttention_15` | 1 | 1719.600 |
| `PromptFlashAttention_29` | 1 | 1719.520 |
| `PromptFlashAttention_12` | 1 | 1719.480 |
| `PromptFlashAttention_2` | 1 | 1719.200 |
| `PromptFlashAttention` | 1 | 1718.880 |
| `PromptFlashAttention_15` | 1 | 1718.880 |
| `PromptFlashAttention_26` | 1 | 1718.820 |
| `PromptFlashAttention_20` | 1 | 1718.740 |
| `PromptFlashAttention_9` | 1 | 1718.540 |
| `PromptFlashAttention_25` | 1 | 1718.280 |
| `PromptFlashAttention_1` | 1 | 1718.240 |
| `PromptFlashAttention` | 1 | 1718.220 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.vision_prefill.memory.step0` | 1 | 147395.810 |
| `mineru.vision_prefill.memory.step1` | 1 | 145521.680 |
| `mineru.vision_prefill.memory.step2` | 1 | 144579.590 |
| `aten::item` | 15 | 110446.550 |
| `aten::_local_scalar_dense` | 15 | 109936.020 |
| `npu::npu_conv3d` | 3 | 109869.000 |
| `Conv3D` | 3 | 109869.000 |
| `cache_compiler inference` | 3 | 4381.090 |
| `empty_tensor` | 102 | 3835.150 |
| `TorchNpuGraphBase::Run` | 3 | 2678.230 |
| `aten::as_strided` | 69 | 2123.060 |
| `aten::linear` | 6 | 1917.730 |
| `aten::pad` | 12 | 1875.110 |
| `aten::flatten` | 9 | 1487.350 |
| `aten::constant_pad_nd` | 12 | 1458.050 |
| `aten::arange` | 12 | 1431.450 |
| `aten::select` | 24 | 1419.680 |
| `aten::unsqueeze` | 21 | 1341.910 |
| `aten::addmm` | 6 | 1304.120 |
| `aten::repeat_interleave` | 3 | 1122.090 |
| `aten::layer_norm` | 3 | 1105.660 |
| `aclnnAddmm` | 6 | 1067.360 |
| `RefreshAtTensorFromGeTensor` | 3 | 1034.950 |
| `aten::gelu` | 6 | 1031.960 |
| `aten::native_layer_norm` | 3 | 968.670 |
| `aten::unbind` | 6 | 926.050 |
| `aten::cat` | 6 | 897.640 |
| `aten::view` | 24 | 865.720 |
| `aten::clone` | 6 | 855.440 |
| `aten::reshape` | 9 | 786.460 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 296137.840 |
| `ModelLoad` | 1 | 242539.140 |
| `aclrtSynchronizeStreamWithTimeout` | 15 | 108871.370 |
| `launch` | 1052 | 16786.470 |
| `aclrtLaunchKernelWithHostArgs` | 108 | 665.610 |
| `aclopCompileAndExecute` | 3 | 548.730 |
| `aclrtMemcpy` | 15 | 500.570 |
| `aclnnConstantPadNd` | 12 | 257.640 |
| `aclnnArange` | 12 | 207.030 |
| `InputCopy` | 3 | 180.970 |
| `aclnnCatGetWorkspaceSize` | 3 | 114.530 |
| `aclnnCat` | 6 | 113.480 |
| `aclnnLayerNorm` | 3 | 90.200 |
| `aclnnRepeatInterleaveGetWorkspaceSize` | 3 | 89.310 |
| `aclnnAddmm` | 6 | 84.280 |
| `opCompile` | 3 | 83.410 |
| `aclrtMemcpyAsync` | 6 | 82.210 |
| `ModelExecute` | 3 | 77.880 |
| `aclnnCumsum` | 3 | 77.100 |
| `aclnnInplaceCopy` | 6 | 75.470 |
| `aclnnMul` | 6 | 71.610 |
| `aclnnMax` | 3 | 60.010 |
| `aclnnRepeatInterleave` | 3 | 56.070 |
| `aclnnStack` | 3 | 54.800 |
| `aclnnIndex` | 3 | 47.970 |
| `aclnnReduceSum` | 3 | 47.440 |
| `aclnnGelu` | 3 | 46.470 |
| `aclnnCos` | 3 | 43.040 |
| `aclnnNeTensor` | 3 | 37.620 |
| `aclrtGetStreamAttribute` | 87 | 36.450 |

