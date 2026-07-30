# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b4s512_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b4s512_pipe/liteserver-c001-4_757329_20260730140003492_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `73592.180 us`
- `Free`: `2986.400 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3840.250 us`
- `Stage`: `76578.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 21248.260 |
| `ScaledMaskedSoftmax` | 81 | 12733.480 |
| `StridedSliceD` | 324 | 8728.880 |
| `Transpose` | 324 | 5445.860 |
| `AddLayerNorm` | 162 | 3923.240 |
| `Mul` | 324 | 3379.340 |
| `Gelu` | 81 | 3237.160 |
| `BatchMatMul` | 162 | 3037.540 |
| `ConcatV2D` | 243 | 2916.400 |
| `TransData` | 243 | 2051.200 |
| `Add` | 162 | 1964.120 |
| `Cast` | 162 | 1825.700 |
| `Neg` | 162 | 1704.100 |
| `SplitVD` | 81 | 1290.720 |
| `LayerNormV3` | 3 | 91.040 |
| `Data` | 3 | 15.140 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax` | 3 | 474.480 |
| `ScaledMaskedSoftmax_25` | 3 | 472.400 |
| `ScaledMaskedSoftmax_3` | 3 | 472.360 |
| `ScaledMaskedSoftmax_20` | 3 | 471.900 |
| `ScaledMaskedSoftmax_1` | 3 | 471.880 |
| `ScaledMaskedSoftmax_5` | 3 | 471.840 |
| `ScaledMaskedSoftmax_26` | 3 | 471.820 |
| `ScaledMaskedSoftmax_22` | 3 | 471.820 |
| `ScaledMaskedSoftmax_9` | 3 | 471.700 |
| `ScaledMaskedSoftmax_17` | 3 | 471.680 |
| `ScaledMaskedSoftmax_15` | 3 | 471.660 |
| `ScaledMaskedSoftmax_7` | 3 | 471.640 |
| `ScaledMaskedSoftmax_13` | 3 | 471.540 |
| `ScaledMaskedSoftmax_10` | 3 | 471.480 |
| `ScaledMaskedSoftmax_8` | 3 | 471.460 |
| `ScaledMaskedSoftmax_12` | 3 | 471.400 |
| `ScaledMaskedSoftmax_24` | 3 | 471.360 |
| `ScaledMaskedSoftmax_19` | 3 | 471.320 |
| `ScaledMaskedSoftmax_11` | 3 | 471.280 |
| `ScaledMaskedSoftmax_21` | 3 | 471.200 |
| `ScaledMaskedSoftmax_23` | 3 | 471.140 |
| `ScaledMaskedSoftmax_4` | 3 | 471.120 |
| `ScaledMaskedSoftmax_18` | 3 | 471.120 |
| `ScaledMaskedSoftmax_16` | 3 | 471.080 |
| `ScaledMaskedSoftmax_14` | 3 | 470.940 |
| `ScaledMaskedSoftmax_2` | 3 | 470.940 |
| `ScaledMaskedSoftmax_6` | 3 | 470.920 |
| `MatMulV2_5` | 3 | 263.900 |
| `MatMulV2_4` | 3 | 248.440 |
| `MatMulV2_143` | 3 | 244.720 |
| `MatMulV2_95` | 3 | 244.180 |
| `MatMulV2_125` | 3 | 244.160 |
| `MatMulV2_47` | 3 | 243.760 |
| `MatMulV2_83` | 3 | 243.600 |
| `MatMulV2_77` | 3 | 243.320 |
| `MatMulV2_29` | 3 | 242.860 |
| `MatMulV2_17` | 3 | 242.700 |
| `MatMulV2_137` | 3 | 242.360 |
| `MatMulV2_59` | 3 | 242.100 |
| `MatMulV2_71` | 3 | 242.060 |
| `MatMulV2_101` | 3 | 241.840 |
| `MatMulV2_65` | 3 | 241.800 |
| `MatMulV2_119` | 3 | 241.800 |
| `MatMulV2_161` | 3 | 241.800 |
| `MatMulV2_113` | 3 | 241.480 |
| `MatMulV2_149` | 3 | 241.340 |
| `MatMulV2_155` | 3 | 241.320 |
| `MatMulV2_107` | 3 | 241.240 |
| `MatMulV2_131` | 3 | 240.480 |
| `MatMulV2_41` | 3 | 240.140 |
| `MatMulV2_23` | 3 | 239.800 |
| `MatMulV2_89` | 3 | 238.880 |
| `MatMulV2_53` | 3 | 238.720 |
| `MatMulV2_11` | 3 | 238.120 |
| `MatMulV2_35` | 3 | 237.980 |
| `MatMulV2_22` | 3 | 230.600 |
| `MatMulV2_28` | 3 | 229.080 |
| `MatMulV2_100` | 3 | 228.660 |
| `MatMulV2_16` | 3 | 228.600 |
| `MatMulV2_40` | 3 | 228.000 |
| `MatMulV2_136` | 3 | 227.940 |
| `MatMulV2_160` | 3 | 227.660 |
| `MatMulV2_118` | 3 | 227.640 |
| `MatMulV2_46` | 3 | 227.560 |
| `MatMulV2_154` | 3 | 227.420 |
| `MatMulV2_94` | 3 | 227.340 |
| `MatMulV2_82` | 3 | 227.320 |
| `MatMulV2_148` | 3 | 227.240 |
| `MatMulV2_58` | 3 | 227.060 |
| `MatMulV2_76` | 3 | 226.700 |
| `MatMulV2_130` | 3 | 226.640 |
| `MatMulV2_106` | 3 | 226.620 |
| `MatMulV2_70` | 3 | 226.240 |
| `MatMulV2_52` | 3 | 225.960 |
| `MatMulV2_124` | 3 | 225.780 |
| `MatMulV2_34` | 3 | 224.460 |
| `MatMulV2_10` | 3 | 223.960 |
| `MatMulV2_64` | 3 | 223.220 |
| `MatMulV2_112` | 3 | 223.160 |
| `MatMulV2_142` | 3 | 223.140 |
| `MatMulV2_88` | 3 | 222.580 |
| `Gelu_21` | 3 | 138.560 |
| `Gelu_14` | 3 | 137.520 |
| `Gelu_17` | 3 | 119.760 |
| `Gelu_5` | 3 | 119.440 |
| `Gelu_10` | 3 | 119.100 |
| `Gelu_23` | 3 | 119.060 |
| `Gelu_1` | 3 | 118.880 |
| `Gelu_15` | 3 | 118.560 |
| `Gelu_25` | 3 | 118.560 |
| `Gelu_3` | 3 | 118.520 |
| `Gelu_20` | 3 | 118.480 |
| `Gelu_7` | 3 | 118.480 |
| `Gelu_19` | 3 | 118.400 |
| `Gelu_24` | 3 | 118.380 |
| `Gelu_13` | 3 | 118.360 |
| `Gelu_8` | 3 | 118.320 |
| `Gelu_9` | 3 | 118.320 |
| `Gelu_12` | 3 | 118.180 |
| `Gelu_22` | 3 | 118.180 |
| `Gelu` | 3 | 118.160 |
| `Gelu_26` | 3 | 118.080 |
| `Gelu_2` | 3 | 118.020 |
| `Gelu_16` | 3 | 118.020 |
| `Gelu_4` | 3 | 118.000 |
| `Gelu_11` | 3 | 117.960 |
| `Gelu_18` | 3 | 117.960 |
| `Gelu_6` | 3 | 117.900 |
| `MatMulV2_3` | 3 | 115.740 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 107.580 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 103.580 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 101.980 |
| `StridedSliceV2_28` | 3 | 98.740 |
| `StridedSliceV2_29` | 3 | 97.480 |
| `StridedSliceV2_27` | 3 | 97.360 |
| `StridedSliceV2_26` | 3 | 96.860 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 96.320 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 96.280 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 96.160 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 95.540 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 95.480 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 95.320 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 95.120 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 95.020 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 95.000 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 94.320 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 94.280 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 94.240 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 94.140 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 93.900 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 93.760 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 93.640 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 93.600 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 93.360 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 93.000 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 92.960 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 92.880 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 92.860 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 92.780 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 92.760 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 91.220 |
| `LayerNormV4_LayerNormV3` | 3 | 91.040 |
| `MatMulV2` | 3 | 85.960 |
| `MatMulV2_18` | 3 | 82.800 |
| `MatMulV2_20` | 3 | 82.540 |
| `MatMulV2_43` | 3 | 82.380 |
| `StridedSliceV2_36` | 3 | 82.360 |
| `MatMulV2_6` | 3 | 82.320 |
| `MatMulV2_36` | 3 | 82.240 |
| `StridedSliceV2_37` | 3 | 82.240 |
| `MatMulV2_8` | 3 | 82.240 |
| `StridedSliceV2_87` | 3 | 82.140 |
| `StridedSliceV2_47` | 3 | 82.060 |
| `StridedSliceV2_66` | 3 | 82.060 |
| `StridedSliceV2_76` | 3 | 81.980 |
| `MatMulV2_38` | 3 | 81.820 |
| `StridedSliceV2_77` | 3 | 81.820 |
| `MatMulV2_56` | 3 | 81.760 |
| `StridedSliceV2_56` | 3 | 81.760 |
| `StridedSliceV2_104` | 3 | 81.740 |
| `MatMulV2_157` | 3 | 81.720 |
| `MatMulV2_66` | 3 | 81.680 |
| `MatMulV2_91` | 3 | 81.680 |
| `MatMulV2_13` | 3 | 81.640 |
| `StridedSliceV2_67` | 3 | 81.600 |
| `MatMulV2_102` | 3 | 81.580 |
| `MatMulV2_116` | 3 | 81.580 |
| `StridedSliceV2_8` | 3 | 81.560 |
| `StridedSliceV2_84` | 3 | 81.540 |
| `StridedSliceV2_46` | 3 | 81.540 |
| `StridedSliceV2_57` | 3 | 81.440 |
| `MatMulV2_150` | 3 | 81.400 |
| `MatMulV2_152` | 3 | 81.400 |
| `StridedSliceV2_16` | 3 | 81.380 |
| `StridedSliceV2_89` | 3 | 81.300 |
| `MatMulV2_84` | 3 | 81.280 |
| `StridedSliceV2_86` | 3 | 81.280 |
| `MatMulV2_109` | 3 | 81.260 |
| `StridedSliceV2_18` | 3 | 81.260 |
| `StridedSliceV2_95` | 3 | 81.240 |
| `MatMulV2_114` | 3 | 81.220 |
| `StridedSliceV2_6` | 3 | 81.180 |
| `StridedSliceV2_38` | 3 | 81.180 |
| `StridedSliceV2_7` | 3 | 81.120 |
| `StridedSliceV2_72` | 3 | 81.100 |
| `StridedSliceV2_100` | 3 | 81.040 |
| `MatMulV2_12` | 3 | 81.040 |
| `MatMulV2_104` | 3 | 81.000 |
| `MatMulV2_134` | 3 | 81.000 |
| `StridedSliceV2_103` | 3 | 81.000 |
| `StridedSliceV2_39` | 3 | 80.980 |
| `MatMulV2_127` | 3 | 80.980 |
| `StridedSliceV2_9` | 3 | 80.900 |
| `StridedSliceV2_94` | 3 | 80.900 |
| `MatMulV2_7` | 3 | 80.860 |
| `StridedSliceV2_85` | 3 | 80.860 |
| `StridedSliceV2_92` | 3 | 80.860 |
| `StridedSliceV2` | 3 | 80.760 |
| `StridedSliceV2_17` | 3 | 80.740 |
| `StridedSliceV2_34` | 3 | 80.740 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax | "4,16,512,512;4,1,512,512" -> "4,16,512,512" | NCHW;NCHW -> NCHW` | 81 | 12733.480 |
| `StridedSliceD | "4,512,16,80" -> "4,512,16,40" | ND -> ND` | 324 | 8728.880 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6546.460 |
| `MatMulV2 | "2048,1152;72,80,16,16;1280" -> "2048,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 6500.840 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6139.020 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 3923.240 |
| `Transpose | "4,512,16,80;4" -> "4,16,512,80" | ND;ND -> ND` | 243 | 3749.420 |
| `Mul | "4,512,16,80;4,512,1,80" -> "4,512,16,80" | ND;ND -> ND` | 324 | 3379.340 |
| `Gelu | "4,512,4352" -> "4,512,4352" | ND -> ND` | 81 | 3237.160 |
| `MatMulV2 | "2048,1280;80,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2061.940 |
| `TransData | "64,512,80" -> "64,5,32,16,16" | ND -> FRACTAL_NZ` | 243 | 2051.200 |
| `Add | "4,512,16,80;4,512,16,80" -> "4,512,16,80" | ND;ND -> ND` | 162 | 1964.120 |
| `ConcatV2D | "4,512,16,40;4,512,16,40" -> "4,512,16,80" | ND;ND -> ND` | 162 | 1911.460 |
| `Cast | "4,512,16,80" -> "4,512,16,80" | ND -> ND` | 162 | 1825.700 |
| `Neg | "4,512,16,40" -> "4,512,16,40" | ND -> ND` | 162 | 1704.100 |
| `Transpose | "4,16,512,80;4" -> "4,512,16,80" | ND;ND -> ND` | 81 | 1696.440 |
| `BatchMatMul | "64,5,32,16,16;64,5,32,16,16" -> "64,512,512" | FRACTAL_NZ;FRACTAL_NZ -> ND` | 81 | 1601.980 |
| `BatchMatMul | "64,512,512;64,5,32,16,16" -> "64,512,80" | ND;FRACTAL_NZ -> ND` | 81 | 1435.560 |
| `SplitVD | "4,512,3840" -> "4,512,1280;4,512,1280;4,512,1280" | ND -> ND;ND;ND` | 81 | 1290.720 |
| `ConcatV2D | "4,512,1280;4,512,1280;4,512,1280" -> "4,512,3840" | ND;ND;ND -> ND` | 81 | 1004.940 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 91.040 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.140 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;FRACTAL_NZ;ND` | 486 | 21248.260 |
| `ND` | 1053 | 18837.760 |
| `NCHW;NCHW` | 81 | 12733.480 |
| `ND;ND` | 972 | 12700.780 |
| `ND;ND;ND;ND` | 162 | 3923.240 |
| `FRACTAL_NZ;FRACTAL_NZ` | 81 | 1601.980 |
| `ND;FRACTAL_NZ` | 81 | 1435.560 |
| `ND;ND;ND` | 84 | 1095.980 |
| `N/A` | 3 | 15.140 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax` | 0 | 158.240 |
| `ScaledMaskedSoftmax` | 0 | 158.240 |
| `ScaledMaskedSoftmax` | 0 | 158.000 |
| `ScaledMaskedSoftmax_3` | 0 | 157.880 |
| `ScaledMaskedSoftmax_5` | 0 | 157.540 |
| `ScaledMaskedSoftmax_25` | 0 | 157.540 |
| `ScaledMaskedSoftmax_13` | 0 | 157.500 |
| `ScaledMaskedSoftmax_25` | 0 | 157.480 |
| `ScaledMaskedSoftmax_1` | 0 | 157.460 |
| `ScaledMaskedSoftmax_22` | 0 | 157.460 |
| `ScaledMaskedSoftmax_26` | 0 | 157.460 |
| `ScaledMaskedSoftmax_9` | 0 | 157.440 |
| `ScaledMaskedSoftmax_11` | 0 | 157.420 |
| `ScaledMaskedSoftmax_7` | 0 | 157.400 |
| `ScaledMaskedSoftmax_25` | 0 | 157.380 |
| `ScaledMaskedSoftmax_17` | 0 | 157.380 |
| `ScaledMaskedSoftmax_20` | 0 | 157.360 |
| `ScaledMaskedSoftmax_1` | 0 | 157.340 |
| `ScaledMaskedSoftmax_20` | 0 | 157.340 |
| `ScaledMaskedSoftmax_24` | 0 | 157.320 |
| `ScaledMaskedSoftmax_23` | 0 | 157.320 |
| `ScaledMaskedSoftmax_3` | 0 | 157.300 |
| `ScaledMaskedSoftmax_15` | 0 | 157.300 |
| `ScaledMaskedSoftmax_19` | 0 | 157.260 |
| `ScaledMaskedSoftmax_10` | 0 | 157.260 |
| `ScaledMaskedSoftmax_12` | 0 | 157.260 |
| `ScaledMaskedSoftmax_15` | 0 | 157.260 |
| `ScaledMaskedSoftmax_7` | 0 | 157.240 |
| `ScaledMaskedSoftmax_9` | 0 | 157.240 |
| `ScaledMaskedSoftmax_12` | 0 | 157.240 |
| `ScaledMaskedSoftmax_5` | 0 | 157.220 |
| `ScaledMaskedSoftmax_8` | 0 | 157.220 |
| `ScaledMaskedSoftmax_18` | 0 | 157.220 |
| `ScaledMaskedSoftmax_4` | 0 | 157.200 |
| `ScaledMaskedSoftmax_10` | 0 | 157.200 |
| `ScaledMaskedSoftmax_17` | 0 | 157.200 |
| `ScaledMaskedSoftmax_20` | 0 | 157.200 |
| `ScaledMaskedSoftmax_22` | 0 | 157.200 |
| `ScaledMaskedSoftmax_26` | 0 | 157.200 |
| `ScaledMaskedSoftmax_14` | 0 | 157.180 |
| `ScaledMaskedSoftmax_21` | 0 | 157.180 |
| `ScaledMaskedSoftmax_3` | 0 | 157.180 |
| `ScaledMaskedSoftmax_8` | 0 | 157.180 |
| `ScaledMaskedSoftmax_16` | 0 | 157.180 |
| `ScaledMaskedSoftmax_19` | 0 | 157.160 |
| `ScaledMaskedSoftmax_26` | 0 | 157.160 |
| `ScaledMaskedSoftmax_22` | 0 | 157.160 |
| `ScaledMaskedSoftmax_17` | 0 | 157.100 |
| `ScaledMaskedSoftmax_15` | 0 | 157.100 |
| `ScaledMaskedSoftmax_6` | 0 | 157.080 |
| `ScaledMaskedSoftmax_1` | 0 | 157.080 |
| `ScaledMaskedSoftmax_4` | 0 | 157.080 |
| `ScaledMaskedSoftmax_18` | 0 | 157.080 |
| `ScaledMaskedSoftmax_24` | 0 | 157.080 |
| `ScaledMaskedSoftmax_5` | 0 | 157.080 |
| `ScaledMaskedSoftmax_8` | 0 | 157.060 |
| `ScaledMaskedSoftmax_13` | 0 | 157.060 |
| `ScaledMaskedSoftmax_23` | 0 | 157.060 |
| `ScaledMaskedSoftmax_11` | 0 | 157.060 |
| `ScaledMaskedSoftmax_21` | 0 | 157.040 |
| `ScaledMaskedSoftmax_6` | 0 | 157.040 |
| `ScaledMaskedSoftmax_16` | 0 | 157.020 |
| `ScaledMaskedSoftmax_9` | 0 | 157.020 |
| `ScaledMaskedSoftmax_10` | 0 | 157.020 |
| `ScaledMaskedSoftmax_2` | 0 | 157.000 |
| `ScaledMaskedSoftmax_7` | 0 | 157.000 |
| `ScaledMaskedSoftmax_13` | 0 | 156.980 |
| `ScaledMaskedSoftmax_2` | 0 | 156.980 |
| `ScaledMaskedSoftmax_21` | 0 | 156.980 |
| `ScaledMaskedSoftmax_2` | 0 | 156.960 |
| `ScaledMaskedSoftmax_24` | 0 | 156.960 |
| `ScaledMaskedSoftmax_14` | 0 | 156.920 |
| `ScaledMaskedSoftmax_12` | 0 | 156.900 |
| `ScaledMaskedSoftmax_19` | 0 | 156.900 |
| `ScaledMaskedSoftmax_16` | 0 | 156.880 |
| `ScaledMaskedSoftmax_4` | 0 | 156.840 |
| `ScaledMaskedSoftmax_14` | 0 | 156.840 |
| `ScaledMaskedSoftmax_18` | 0 | 156.820 |
| `ScaledMaskedSoftmax_6` | 0 | 156.800 |
| `ScaledMaskedSoftmax_11` | 0 | 156.800 |
| `ScaledMaskedSoftmax_23` | 0 | 156.760 |
| `MatMulV2_5` | 0 | 91.000 |
| `MatMulV2_5` | 0 | 87.800 |
| `MatMulV2_5` | 0 | 85.100 |
| `MatMulV2_4` | 0 | 83.880 |
| `MatMulV2_4` | 0 | 83.560 |
| `MatMulV2_29` | 0 | 82.240 |
| `MatMulV2_29` | 0 | 82.140 |
| `MatMulV2_95` | 0 | 82.020 |
| `MatMulV2_143` | 0 | 81.760 |
| `MatMulV2_143` | 0 | 81.620 |
| `MatMulV2_125` | 0 | 81.580 |
| `MatMulV2_47` | 0 | 81.500 |
| `MatMulV2_83` | 0 | 81.500 |
| `MatMulV2_77` | 0 | 81.480 |
| `MatMulV2_125` | 0 | 81.480 |
| `MatMulV2_95` | 0 | 81.400 |
| `MatMulV2_119` | 0 | 81.380 |
| `MatMulV2_143` | 0 | 81.340 |
| `MatMulV2_83` | 0 | 81.320 |
| `MatMulV2_47` | 0 | 81.280 |
| `MatMulV2_59` | 0 | 81.220 |
| `MatMulV2_59` | 0 | 81.160 |
| `MatMulV2_77` | 0 | 81.120 |
| `MatMulV2_17` | 0 | 81.120 |
| `MatMulV2_125` | 0 | 81.100 |
| `MatMulV2_161` | 0 | 81.040 |
| `MatMulV2_4` | 0 | 81.000 |
| `MatMulV2_17` | 0 | 81.000 |
| `MatMulV2_47` | 0 | 80.980 |
| `MatMulV2_65` | 0 | 80.980 |
| `MatMulV2_71` | 0 | 80.900 |
| `MatMulV2_137` | 0 | 80.860 |
| `MatMulV2_65` | 0 | 80.840 |
| `MatMulV2_137` | 0 | 80.820 |
| `MatMulV2_113` | 0 | 80.800 |
| `MatMulV2_149` | 0 | 80.800 |
| `MatMulV2_83` | 0 | 80.780 |
| `MatMulV2_23` | 0 | 80.780 |
| `MatMulV2_95` | 0 | 80.760 |
| `MatMulV2_119` | 0 | 80.740 |
| `MatMulV2_77` | 0 | 80.720 |
| `MatMulV2_101` | 0 | 80.720 |
| `MatMulV2_137` | 0 | 80.680 |
| `MatMulV2_155` | 0 | 80.640 |
| `MatMulV2_23` | 0 | 80.640 |
| `MatMulV2_17` | 0 | 80.580 |
| `MatMulV2_71` | 0 | 80.580 |
| `MatMulV2_71` | 0 | 80.580 |
| `MatMulV2_161` | 0 | 80.580 |
| `MatMulV2_101` | 0 | 80.560 |
| `MatMulV2_101` | 0 | 80.560 |
| `MatMulV2_107` | 0 | 80.540 |
| `MatMulV2_107` | 0 | 80.520 |
| `MatMulV2_35` | 0 | 80.500 |
| `MatMulV2_149` | 0 | 80.480 |
| `MatMulV2_155` | 0 | 80.480 |
| `MatMulV2_131` | 0 | 80.440 |
| `MatMulV2_131` | 0 | 80.440 |
| `MatMulV2_41` | 0 | 80.420 |
| `MatMulV2_53` | 0 | 80.420 |
| `MatMulV2_11` | 0 | 80.420 |
| `MatMulV2_113` | 0 | 80.380 |
| `MatMulV2_113` | 0 | 80.300 |
| `MatMulV2_155` | 0 | 80.200 |
| `MatMulV2_107` | 0 | 80.180 |
| `MatMulV2_161` | 0 | 80.180 |
| `MatMulV2_35` | 0 | 80.160 |
| `MatMulV2_89` | 0 | 80.060 |
| `MatMulV2_149` | 0 | 80.060 |
| `MatMulV2_41` | 0 | 80.020 |
| `MatMulV2_65` | 0 | 79.980 |
| `MatMulV2_11` | 0 | 79.780 |
| `MatMulV2_59` | 0 | 79.720 |
| `MatMulV2_41` | 0 | 79.700 |
| `MatMulV2_119` | 0 | 79.680 |
| `MatMulV2_89` | 0 | 79.640 |
| `MatMulV2_131` | 0 | 79.600 |
| `MatMulV2_53` | 0 | 79.340 |
| `MatMulV2_89` | 0 | 79.180 |
| `MatMulV2_53` | 0 | 78.960 |
| `MatMulV2_29` | 0 | 78.480 |
| `MatMulV2_23` | 0 | 78.380 |
| `MatMulV2_22` | 0 | 78.080 |
| `MatMulV2_11` | 0 | 77.920 |
| `MatMulV2_35` | 0 | 77.320 |
| `MatMulV2_16` | 0 | 77.300 |
| `MatMulV2_28` | 0 | 77.120 |
| `MatMulV2_40` | 0 | 76.400 |
| `MatMulV2_100` | 0 | 76.380 |
| `MatMulV2_22` | 0 | 76.360 |
| `MatMulV2_160` | 0 | 76.300 |
| `MatMulV2_34` | 0 | 76.260 |
| `MatMulV2_100` | 0 | 76.220 |
| `MatMulV2_22` | 0 | 76.160 |
| `MatMulV2_100` | 0 | 76.060 |
| `MatMulV2_136` | 0 | 76.060 |
| `MatMulV2_136` | 0 | 76.040 |
| `MatMulV2_28` | 0 | 76.020 |
| `MatMulV2_46` | 0 | 76.000 |
| `MatMulV2_28` | 0 | 75.940 |
| `MatMulV2_118` | 0 | 75.920 |
| `MatMulV2_118` | 0 | 75.900 |
| `MatMulV2_154` | 0 | 75.900 |
| `MatMulV2_40` | 0 | 75.880 |
| `MatMulV2_82` | 0 | 75.880 |
| `MatMulV2_136` | 0 | 75.840 |
| `MatMulV2_46` | 0 | 75.840 |
| `MatMulV2_106` | 0 | 75.820 |
| `MatMulV2_118` | 0 | 75.820 |
| `MatMulV2_94` | 0 | 75.820 |
| `MatMulV2_160` | 0 | 75.800 |
| `MatMulV2_82` | 0 | 75.780 |
| `MatMulV2_94` | 0 | 75.780 |
| `MatMulV2_148` | 0 | 75.780 |
| `MatMulV2_154` | 0 | 75.760 |
| `MatMulV2_154` | 0 | 75.760 |
| `MatMulV2_94` | 0 | 75.740 |
| `MatMulV2_130` | 0 | 75.740 |
| `MatMulV2_148` | 0 | 75.740 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 27767.540 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step1` | 1 | 26232.650 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step3` | 1 | 25753.430 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step2` | 1 | 25703.380 |
| `TorchDynamo Cache Lookup` | 3 | 24707.690 |
| `Torch-Compiled Region: 0/0` | 3 | 4050.130 |
| `TorchNpuGraphBase::Run` | 3 | 2930.740 |
| `RefreshAtTensorFromGeTensor` | 3 | 1245.360 |
| `aten::empty` | 3 | 600.850 |
| `ExecuteGraph` | 3 | 503.350 |
| `AssembleInputs` | 3 | 422.930 |
| `AssembleOutputs` | 3 | 344.900 |
| `aten::set_` | 3 | 320.400 |
| `empty_tensor` | 3 | 297.480 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 190188.930 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 71438.930 |
| `launch` | 1003 | 12612.240 |
| `InputCopy` | 3 | 140.850 |
| `ModelExecute` | 3 | 49.030 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 27.320 |
| `step_info` | 6 | 14.130 |
| `OutputCopy` | 3 | 0.930 |

