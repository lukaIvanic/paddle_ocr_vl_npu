# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b1s2048_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b1s2048_pipe/liteserver-c001-4_761725_20260730140358649_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `93263.900 us`
- `Free`: `3099.520 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4032.750 us`
- `Stage`: `96363.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax` | 81 | 27866.980 |
| `MatMulV2` | 486 | 21819.300 |
| `BatchMatMul` | 162 | 10074.480 |
| `StridedSliceD` | 324 | 8564.400 |
| `AddLayerNorm` | 162 | 3949.400 |
| `Transpose` | 324 | 3758.500 |
| `Mul` | 324 | 3322.800 |
| `Gelu` | 81 | 3143.340 |
| `ConcatV2D` | 243 | 2664.560 |
| `TransData` | 243 | 1856.020 |
| `Add` | 162 | 1850.740 |
| `Cast` | 162 | 1540.600 |
| `Neg` | 162 | 1483.720 |
| `SplitVD` | 81 | 1266.340 |
| `LayerNormV3` | 3 | 87.600 |
| `Data` | 3 | 15.120 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax` | 3 | 1047.620 |
| `ScaledMaskedSoftmax_19` | 3 | 1045.460 |
| `ScaledMaskedSoftmax_4` | 3 | 1034.740 |
| `ScaledMaskedSoftmax_20` | 3 | 1034.000 |
| `ScaledMaskedSoftmax_16` | 3 | 1033.520 |
| `ScaledMaskedSoftmax_14` | 3 | 1033.400 |
| `ScaledMaskedSoftmax_12` | 3 | 1033.040 |
| `ScaledMaskedSoftmax_1` | 3 | 1032.780 |
| `ScaledMaskedSoftmax_8` | 3 | 1032.340 |
| `ScaledMaskedSoftmax_10` | 3 | 1031.880 |
| `ScaledMaskedSoftmax_24` | 3 | 1031.560 |
| `ScaledMaskedSoftmax_25` | 3 | 1031.320 |
| `ScaledMaskedSoftmax_6` | 3 | 1031.140 |
| `ScaledMaskedSoftmax_22` | 3 | 1030.960 |
| `ScaledMaskedSoftmax_9` | 3 | 1030.760 |
| `ScaledMaskedSoftmax_26` | 3 | 1030.220 |
| `ScaledMaskedSoftmax_2` | 3 | 1030.140 |
| `ScaledMaskedSoftmax_11` | 3 | 1029.920 |
| `ScaledMaskedSoftmax_15` | 3 | 1029.880 |
| `ScaledMaskedSoftmax_21` | 3 | 1029.820 |
| `ScaledMaskedSoftmax_17` | 3 | 1029.820 |
| `ScaledMaskedSoftmax_3` | 3 | 1029.540 |
| `ScaledMaskedSoftmax_7` | 3 | 1029.460 |
| `ScaledMaskedSoftmax_5` | 3 | 1028.960 |
| `ScaledMaskedSoftmax_23` | 3 | 1028.540 |
| `ScaledMaskedSoftmax_18` | 3 | 1028.280 |
| `ScaledMaskedSoftmax_13` | 3 | 1027.880 |
| `MatMulV2_5` | 3 | 274.640 |
| `MatMulV2_65` | 3 | 253.780 |
| `MatMulV2_77` | 3 | 253.320 |
| `MatMulV2_89` | 3 | 252.640 |
| `MatMulV2_149` | 3 | 252.220 |
| `MatMulV2_41` | 3 | 252.140 |
| `MatMulV2_101` | 3 | 252.120 |
| `MatMulV2_113` | 3 | 252.080 |
| `MatMulV2_125` | 3 | 251.780 |
| `MatMulV2_53` | 3 | 251.700 |
| `MatMulV2_29` | 3 | 251.660 |
| `MatMulV2_137` | 3 | 251.260 |
| `MatMulV2_17` | 3 | 251.040 |
| `MatMulV2_4` | 3 | 250.800 |
| `MatMulV2_161` | 3 | 248.760 |
| `MatMulV2_148` | 3 | 243.020 |
| `MatMulV2_119` | 3 | 242.520 |
| `MatMulV2_16` | 3 | 242.260 |
| `MatMulV2_155` | 3 | 242.100 |
| `MatMulV2_59` | 3 | 241.720 |
| `MatMulV2_142` | 3 | 241.680 |
| `MatMulV2_124` | 3 | 241.600 |
| `MatMulV2_47` | 3 | 241.540 |
| `MatMulV2_64` | 3 | 241.440 |
| `MatMulV2_131` | 3 | 241.320 |
| `MatMulV2_160` | 3 | 241.260 |
| `MatMulV2_106` | 3 | 241.240 |
| `MatMulV2_76` | 3 | 241.100 |
| `MatMulV2_143` | 3 | 241.100 |
| `MatMulV2_71` | 3 | 240.920 |
| `MatMulV2_88` | 3 | 240.780 |
| `MatMulV2_28` | 3 | 240.640 |
| `MatMulV2_82` | 3 | 239.880 |
| `MatMulV2_107` | 3 | 239.840 |
| `MatMulV2_40` | 3 | 239.820 |
| `MatMulV2_95` | 3 | 239.820 |
| `MatMulV2_136` | 3 | 239.700 |
| `MatMulV2_34` | 3 | 239.640 |
| `MatMulV2_58` | 3 | 239.600 |
| `MatMulV2_70` | 3 | 239.500 |
| `MatMulV2_46` | 3 | 239.460 |
| `MatMulV2_118` | 3 | 239.340 |
| `MatMulV2_52` | 3 | 239.240 |
| `MatMulV2_83` | 3 | 239.240 |
| `MatMulV2_100` | 3 | 239.200 |
| `MatMulV2_10` | 3 | 239.160 |
| `MatMulV2_154` | 3 | 239.040 |
| `MatMulV2_22` | 3 | 238.660 |
| `MatMulV2_35` | 3 | 238.520 |
| `MatMulV2_112` | 3 | 238.500 |
| `MatMulV2_94` | 3 | 238.340 |
| `MatMulV2_23` | 3 | 238.140 |
| `MatMulV2_11` | 3 | 237.840 |
| `MatMulV2_130` | 3 | 237.480 |
| `BatchMatMul_2` | 3 | 201.200 |
| `BatchMatMul_30` | 3 | 200.760 |
| `BatchMatMul_18` | 3 | 200.720 |
| `BatchMatMul_34` | 3 | 200.720 |
| `BatchMatMul_6` | 3 | 200.500 |
| `BatchMatMul_26` | 3 | 200.400 |
| `BatchMatMul_46` | 3 | 200.360 |
| `BatchMatMul_14` | 3 | 200.240 |
| `BatchMatMul` | 3 | 200.200 |
| `BatchMatMul_38` | 3 | 200.160 |
| `BatchMatMul_50` | 3 | 200.160 |
| `BatchMatMul_10` | 3 | 199.880 |
| `BatchMatMul_42` | 3 | 199.660 |
| `BatchMatMul_22` | 3 | 199.420 |
| `BatchMatMul_20` | 3 | 198.780 |
| `BatchMatMul_48` | 3 | 198.760 |
| `BatchMatMul_40` | 3 | 198.720 |
| `BatchMatMul_16` | 3 | 198.720 |
| `BatchMatMul_32` | 3 | 198.680 |
| `BatchMatMul_12` | 3 | 198.660 |
| `BatchMatMul_28` | 3 | 198.660 |
| `BatchMatMul_52` | 3 | 198.660 |
| `BatchMatMul_44` | 3 | 198.620 |
| `BatchMatMul_24` | 3 | 198.620 |
| `BatchMatMul_8` | 3 | 198.600 |
| `BatchMatMul_36` | 3 | 198.500 |
| `BatchMatMul_4` | 3 | 197.860 |
| `BatchMatMul_1` | 3 | 177.960 |
| `BatchMatMul_19` | 3 | 177.960 |
| `BatchMatMul_7` | 3 | 177.600 |
| `BatchMatMul_15` | 3 | 177.460 |
| `BatchMatMul_35` | 3 | 177.400 |
| `BatchMatMul_39` | 3 | 177.320 |
| `BatchMatMul_23` | 3 | 177.200 |
| `BatchMatMul_51` | 3 | 176.740 |
| `BatchMatMul_43` | 3 | 176.340 |
| `BatchMatMul_31` | 3 | 176.240 |
| `BatchMatMul_11` | 3 | 176.000 |
| `BatchMatMul_27` | 3 | 175.920 |
| `BatchMatMul_47` | 3 | 175.860 |
| `BatchMatMul_21` | 3 | 172.080 |
| `BatchMatMul_3` | 3 | 171.620 |
| `BatchMatMul_5` | 3 | 171.120 |
| `BatchMatMul_29` | 3 | 171.020 |
| `BatchMatMul_33` | 3 | 171.020 |
| `BatchMatMul_53` | 3 | 170.760 |
| `BatchMatMul_17` | 3 | 170.480 |
| `BatchMatMul_13` | 3 | 170.260 |
| `BatchMatMul_41` | 3 | 170.180 |
| `BatchMatMul_49` | 3 | 170.160 |
| `BatchMatMul_37` | 3 | 170.080 |
| `BatchMatMul_45` | 3 | 169.940 |
| `BatchMatMul_25` | 3 | 169.780 |
| `BatchMatMul_9` | 3 | 169.760 |
| `Gelu_21` | 3 | 116.920 |
| `Gelu_23` | 3 | 116.620 |
| `Gelu_2` | 3 | 116.540 |
| `Gelu_1` | 3 | 116.540 |
| `Gelu_17` | 3 | 116.540 |
| `Gelu_10` | 3 | 116.520 |
| `Gelu_8` | 3 | 116.500 |
| `Gelu_26` | 3 | 116.500 |
| `Gelu_14` | 3 | 116.480 |
| `Gelu_20` | 3 | 116.480 |
| `Gelu_3` | 3 | 116.480 |
| `Gelu_25` | 3 | 116.420 |
| `Gelu_6` | 3 | 116.400 |
| `Gelu_13` | 3 | 116.400 |
| `Gelu_15` | 3 | 116.400 |
| `Gelu_18` | 3 | 116.400 |
| `Gelu_5` | 3 | 116.400 |
| `Gelu_7` | 3 | 116.380 |
| `Gelu` | 3 | 116.360 |
| `Gelu_11` | 3 | 116.360 |
| `Gelu_9` | 3 | 116.340 |
| `Gelu_19` | 3 | 116.340 |
| `Gelu_16` | 3 | 116.320 |
| `Gelu_24` | 3 | 116.240 |
| `Gelu_4` | 3 | 116.220 |
| `Gelu_12` | 3 | 116.200 |
| `Gelu_22` | 3 | 116.040 |
| `MatMulV2_3` | 3 | 101.900 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 101.760 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 99.300 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 99.260 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 98.480 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 97.140 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 96.740 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 96.620 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 96.400 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 96.360 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 96.300 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 96.000 |
| `StridedSliceV2_28` | 3 | 95.920 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 95.880 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 95.880 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 95.700 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 95.500 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 95.440 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 95.140 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 94.960 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 94.600 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 94.200 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 94.160 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 94.100 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 94.020 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 93.920 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 93.740 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 93.720 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 93.380 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 93.160 |
| `StridedSliceV2_26` | 3 | 93.080 |
| `StridedSliceV2_27` | 3 | 93.040 |
| `StridedSliceV2_29` | 3 | 92.500 |
| `LayerNormV4_LayerNormV3` | 3 | 87.600 |
| `MatMulV2` | 3 | 85.800 |
| `MatMulV2_140` | 3 | 84.640 |
| `MatMulV2_150` | 3 | 84.020 |
| `MatMulV2_18` | 3 | 83.780 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax | "1,16,2048,2048;1,1,2048,2048" -> "1,16,2048,2048" | NCHW;NCHW -> NCHW` | 81 | 27866.980 |
| `StridedSliceD | "1,2048,16,80" -> "1,2048,16,40" | ND -> ND` | 324 | 8564.400 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6673.760 |
| `MatMulV2 | "2048,1152;72,80,16,16;1280" -> "2048,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 6543.580 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6492.380 |
| `BatchMatMul | "16,5,128,16,16;16,5,128,16,16" -> "16,2048,2048" | FRACTAL_NZ;FRACTAL_NZ -> ND` | 81 | 5386.220 |
| `BatchMatMul | "16,2048,2048;16,5,128,16,16" -> "16,2048,80" | ND;FRACTAL_NZ -> ND` | 81 | 4688.260 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 3949.400 |
| `Mul | "1,2048,16,80;1,2048,1,80" -> "1,2048,16,80" | ND;ND -> ND` | 324 | 3322.800 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3143.340 |
| `Transpose | "2048,16,80;3" -> "16,2048,80" | ND;ND -> ND` | 243 | 2809.740 |
| `MatMulV2 | "2048,1280;80,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2109.580 |
| `TransData | "16,2048,80" -> "16,5,128,16,16" | ND -> FRACTAL_NZ` | 243 | 1856.020 |
| `Add | "1,2048,16,80;1,2048,16,80" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1850.740 |
| `ConcatV2D | "1,2048,16,40;1,2048,16,40" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1729.520 |
| `Cast | "1,2048,16,80" -> "1,2048,16,80" | ND -> ND` | 162 | 1540.600 |
| `Neg | "1,2048,16,40" -> "1,2048,16,40" | ND -> ND` | 162 | 1483.720 |
| `SplitVD | "1,2048,3840" -> "1,2048,1280;1,2048,1280;1,2048,1280" | ND -> ND;ND;ND` | 81 | 1266.340 |
| `Transpose | "16,2048,80;3" -> "2048,16,80" | ND;ND -> ND` | 81 | 948.760 |
| `ConcatV2D | "1,2048,1280;1,2048,1280;1,2048,1280" -> "1,2048,3840" | ND;ND;ND -> ND` | 81 | 935.040 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 87.600 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.120 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `NCHW;NCHW` | 81 | 27866.980 |
| `ND;FRACTAL_NZ;ND` | 486 | 21819.300 |
| `ND` | 1053 | 17854.420 |
| `ND;ND` | 972 | 10661.560 |
| `FRACTAL_NZ;FRACTAL_NZ` | 81 | 5386.220 |
| `ND;FRACTAL_NZ` | 81 | 4688.260 |
| `ND;ND;ND;ND` | 162 | 3949.400 |
| `ND;ND;ND` | 84 | 1022.640 |
| `N/A` | 3 | 15.120 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax` | 0 | 350.420 |
| `ScaledMaskedSoftmax` | 0 | 349.260 |
| `ScaledMaskedSoftmax_19` | 0 | 348.600 |
| `ScaledMaskedSoftmax_19` | 0 | 348.560 |
| `ScaledMaskedSoftmax_19` | 0 | 348.300 |
| `ScaledMaskedSoftmax` | 0 | 347.940 |
| `ScaledMaskedSoftmax_20` | 0 | 345.640 |
| `ScaledMaskedSoftmax_12` | 0 | 345.420 |
| `ScaledMaskedSoftmax_16` | 0 | 345.400 |
| `ScaledMaskedSoftmax_4` | 0 | 345.360 |
| `ScaledMaskedSoftmax_16` | 0 | 345.140 |
| `ScaledMaskedSoftmax_25` | 0 | 345.120 |
| `ScaledMaskedSoftmax_8` | 0 | 345.080 |
| `ScaledMaskedSoftmax_14` | 0 | 345.060 |
| `ScaledMaskedSoftmax_26` | 0 | 344.840 |
| `ScaledMaskedSoftmax_14` | 0 | 344.820 |
| `ScaledMaskedSoftmax_4` | 0 | 344.700 |
| `ScaledMaskedSoftmax_4` | 0 | 344.680 |
| `ScaledMaskedSoftmax_1` | 0 | 344.540 |
| `ScaledMaskedSoftmax_10` | 0 | 344.540 |
| `ScaledMaskedSoftmax_1` | 0 | 344.480 |
| `ScaledMaskedSoftmax_21` | 0 | 344.480 |
| `ScaledMaskedSoftmax_6` | 0 | 344.460 |
| `ScaledMaskedSoftmax_20` | 0 | 344.220 |
| `ScaledMaskedSoftmax_5` | 0 | 344.220 |
| `ScaledMaskedSoftmax_10` | 0 | 344.180 |
| `ScaledMaskedSoftmax_24` | 0 | 344.160 |
| `ScaledMaskedSoftmax_20` | 0 | 344.140 |
| `ScaledMaskedSoftmax_22` | 0 | 344.140 |
| `ScaledMaskedSoftmax_18` | 0 | 344.120 |
| `ScaledMaskedSoftmax_9` | 0 | 344.020 |
| `ScaledMaskedSoftmax_7` | 0 | 343.940 |
| `ScaledMaskedSoftmax_17` | 0 | 343.940 |
| `ScaledMaskedSoftmax_12` | 0 | 343.880 |
| `ScaledMaskedSoftmax_2` | 0 | 343.800 |
| `ScaledMaskedSoftmax_1` | 0 | 343.760 |
| `ScaledMaskedSoftmax_24` | 0 | 343.740 |
| `ScaledMaskedSoftmax_12` | 0 | 343.740 |
| `ScaledMaskedSoftmax_11` | 0 | 343.700 |
| `ScaledMaskedSoftmax_8` | 0 | 343.680 |
| `ScaledMaskedSoftmax_24` | 0 | 343.660 |
| `ScaledMaskedSoftmax_15` | 0 | 343.640 |
| `ScaledMaskedSoftmax_6` | 0 | 343.600 |
| `ScaledMaskedSoftmax_8` | 0 | 343.580 |
| `ScaledMaskedSoftmax_3` | 0 | 343.580 |
| `ScaledMaskedSoftmax_14` | 0 | 343.520 |
| `ScaledMaskedSoftmax_22` | 0 | 343.460 |
| `ScaledMaskedSoftmax_9` | 0 | 343.440 |
| `ScaledMaskedSoftmax_11` | 0 | 343.420 |
| `ScaledMaskedSoftmax_22` | 0 | 343.360 |
| `ScaledMaskedSoftmax_2` | 0 | 343.320 |
| `ScaledMaskedSoftmax_23` | 0 | 343.300 |
| `ScaledMaskedSoftmax_9` | 0 | 343.300 |
| `ScaledMaskedSoftmax_21` | 0 | 343.240 |
| `ScaledMaskedSoftmax_25` | 0 | 343.240 |
| `ScaledMaskedSoftmax_15` | 0 | 343.180 |
| `ScaledMaskedSoftmax_26` | 0 | 343.180 |
| `ScaledMaskedSoftmax_10` | 0 | 343.160 |
| `ScaledMaskedSoftmax_17` | 0 | 343.140 |
| `ScaledMaskedSoftmax_6` | 0 | 343.080 |
| `ScaledMaskedSoftmax_13` | 0 | 343.060 |
| `ScaledMaskedSoftmax_15` | 0 | 343.060 |
| `ScaledMaskedSoftmax_2` | 0 | 343.020 |
| `ScaledMaskedSoftmax_3` | 0 | 343.000 |
| `ScaledMaskedSoftmax_16` | 0 | 342.980 |
| `ScaledMaskedSoftmax_3` | 0 | 342.960 |
| `ScaledMaskedSoftmax_23` | 0 | 342.960 |
| `ScaledMaskedSoftmax_25` | 0 | 342.960 |
| `ScaledMaskedSoftmax_7` | 0 | 342.960 |
| `ScaledMaskedSoftmax_5` | 0 | 342.880 |
| `ScaledMaskedSoftmax_11` | 0 | 342.800 |
| `ScaledMaskedSoftmax_13` | 0 | 342.760 |
| `ScaledMaskedSoftmax_17` | 0 | 342.740 |
| `ScaledMaskedSoftmax_18` | 0 | 342.640 |
| `ScaledMaskedSoftmax_7` | 0 | 342.560 |
| `ScaledMaskedSoftmax_23` | 0 | 342.280 |
| `ScaledMaskedSoftmax_26` | 0 | 342.200 |
| `ScaledMaskedSoftmax_21` | 0 | 342.100 |
| `ScaledMaskedSoftmax_13` | 0 | 342.060 |
| `ScaledMaskedSoftmax_5` | 0 | 341.860 |
| `ScaledMaskedSoftmax_18` | 0 | 341.520 |
| `MatMulV2_5` | 0 | 91.800 |
| `MatMulV2_5` | 0 | 91.560 |
| `MatMulV2_5` | 0 | 91.280 |
| `MatMulV2_29` | 0 | 84.940 |
| `MatMulV2_65` | 0 | 84.920 |
| `MatMulV2_89` | 0 | 84.880 |
| `MatMulV2_113` | 0 | 84.800 |
| `MatMulV2_77` | 0 | 84.640 |
| `MatMulV2_29` | 0 | 84.580 |
| `MatMulV2_77` | 0 | 84.560 |
| `MatMulV2_113` | 0 | 84.500 |
| `MatMulV2_65` | 0 | 84.480 |
| `MatMulV2_41` | 0 | 84.460 |
| `MatMulV2_53` | 0 | 84.400 |
| `MatMulV2_65` | 0 | 84.380 |
| `MatMulV2_137` | 0 | 84.380 |
| `MatMulV2_41` | 0 | 84.260 |
| `MatMulV2_89` | 0 | 84.240 |
| `MatMulV2_4` | 0 | 84.240 |
| `MatMulV2_101` | 0 | 84.220 |
| `MatMulV2_149` | 0 | 84.200 |
| `MatMulV2_4` | 0 | 84.140 |
| `MatMulV2_53` | 0 | 84.140 |
| `MatMulV2_77` | 0 | 84.120 |
| `MatMulV2_101` | 0 | 84.100 |
| `MatMulV2_149` | 0 | 84.080 |
| `MatMulV2_125` | 0 | 84.060 |
| `MatMulV2_17` | 0 | 84.060 |
| `MatMulV2_125` | 0 | 84.020 |
| `MatMulV2_149` | 0 | 83.940 |
| `MatMulV2_137` | 0 | 83.900 |
| `MatMulV2_101` | 0 | 83.800 |
| `MatMulV2_17` | 0 | 83.780 |
| `MatMulV2_125` | 0 | 83.700 |
| `MatMulV2_161` | 0 | 83.540 |
| `MatMulV2_89` | 0 | 83.520 |
| `MatMulV2_41` | 0 | 83.420 |
| `MatMulV2_17` | 0 | 83.200 |
| `MatMulV2_53` | 0 | 83.160 |
| `MatMulV2_137` | 0 | 82.980 |
| `MatMulV2_161` | 0 | 82.840 |
| `MatMulV2_113` | 0 | 82.780 |
| `MatMulV2_4` | 0 | 82.420 |
| `MatMulV2_161` | 0 | 82.380 |
| `MatMulV2_16` | 0 | 82.240 |
| `MatMulV2_29` | 0 | 82.140 |
| `MatMulV2_148` | 0 | 81.580 |
| `MatMulV2_143` | 0 | 81.560 |
| `MatMulV2_119` | 0 | 81.520 |
| `MatMulV2_155` | 0 | 81.400 |
| `MatMulV2_59` | 0 | 81.260 |
| `MatMulV2_142` | 0 | 81.240 |
| `MatMulV2_160` | 0 | 81.100 |
| `MatMulV2_47` | 0 | 81.080 |
| `MatMulV2_106` | 0 | 81.080 |
| `MatMulV2_148` | 0 | 81.060 |
| `MatMulV2_155` | 0 | 81.000 |
| `MatMulV2_131` | 0 | 80.980 |
| `MatMulV2_124` | 0 | 80.920 |
| `MatMulV2_59` | 0 | 80.880 |
| `MatMulV2_82` | 0 | 80.880 |
| `MatMulV2_64` | 0 | 80.860 |
| `MatMulV2_88` | 0 | 80.860 |
| `MatMulV2_76` | 0 | 80.820 |
| `MatMulV2_76` | 0 | 80.740 |
| `MatMulV2_64` | 0 | 80.720 |
| `MatMulV2_71` | 0 | 80.720 |
| `MatMulV2_136` | 0 | 80.700 |
| `MatMulV2_28` | 0 | 80.660 |
| `MatMulV2_119` | 0 | 80.620 |
| `MatMulV2_10` | 0 | 80.620 |
| `MatMulV2_40` | 0 | 80.500 |
| `MatMulV2_142` | 0 | 80.500 |
| `MatMulV2_47` | 0 | 80.460 |
| `MatMulV2_100` | 0 | 80.420 |
| `MatMulV2_107` | 0 | 80.400 |
| `MatMulV2_124` | 0 | 80.400 |
| `MatMulV2_143` | 0 | 80.400 |
| `MatMulV2_22` | 0 | 80.380 |
| `MatMulV2_119` | 0 | 80.380 |
| `MatMulV2_148` | 0 | 80.380 |
| `MatMulV2_131` | 0 | 80.360 |
| `MatMulV2_11` | 0 | 80.340 |
| `MatMulV2_95` | 0 | 80.320 |
| `MatMulV2_160` | 0 | 80.320 |
| `MatMulV2_70` | 0 | 80.300 |
| `MatMulV2_124` | 0 | 80.280 |
| `MatMulV2_71` | 0 | 80.260 |
| `MatMulV2_106` | 0 | 80.260 |
| `MatMulV2_100` | 0 | 80.240 |
| `MatMulV2_46` | 0 | 80.200 |
| `MatMulV2_88` | 0 | 80.160 |
| `MatMulV2_118` | 0 | 80.100 |
| `MatMulV2_16` | 0 | 80.100 |
| `MatMulV2_40` | 0 | 80.080 |
| `MatMulV2_70` | 0 | 80.060 |
| `MatMulV2_46` | 0 | 80.060 |
| `MatMulV2_58` | 0 | 80.060 |
| `MatMulV2_34` | 0 | 80.040 |
| `MatMulV2_136` | 0 | 80.040 |
| `MatMulV2_58` | 0 | 80.020 |
| `MatMulV2_47` | 0 | 80.000 |
| `MatMulV2_28` | 0 | 80.000 |
| `MatMulV2_28` | 0 | 79.980 |
| `MatMulV2_131` | 0 | 79.980 |
| `MatMulV2_35` | 0 | 79.980 |
| `MatMulV2_34` | 0 | 79.960 |
| `MatMulV2_71` | 0 | 79.940 |
| `MatMulV2_142` | 0 | 79.940 |
| `MatMulV2_16` | 0 | 79.920 |
| `MatMulV2_52` | 0 | 79.920 |
| `MatMulV2_83` | 0 | 79.920 |
| `MatMulV2_82` | 0 | 79.900 |
| `MatMulV2_106` | 0 | 79.900 |
| `MatMulV2_83` | 0 | 79.880 |
| `MatMulV2_64` | 0 | 79.860 |
| `MatMulV2_160` | 0 | 79.840 |
| `MatMulV2_94` | 0 | 79.800 |
| `MatMulV2_95` | 0 | 79.800 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 34327.590 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step1` | 1 | 32952.340 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step3` | 1 | 32315.290 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step2` | 1 | 32300.270 |
| `TorchDynamo Cache Lookup` | 3 | 31262.080 |
| `Torch-Compiled Region: 0/0` | 3 | 4188.110 |
| `TorchNpuGraphBase::Run` | 3 | 2971.140 |
| `RefreshAtTensorFromGeTensor` | 3 | 1255.790 |
| `aten::empty` | 3 | 603.750 |
| `ExecuteGraph` | 3 | 519.000 |
| `AssembleInputs` | 3 | 436.690 |
| `AssembleOutputs` | 3 | 355.800 |
| `aten::set_` | 3 | 310.990 |
| `empty_tensor` | 3 | 286.990 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 187314.120 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 91003.560 |
| `launch` | 1003 | 12769.850 |
| `InputCopy` | 3 | 162.300 |
| `ModelExecute` | 3 | 48.370 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 45.690 |
| `step_info` | 6 | 26.650 |
| `OutputCopy` | 3 | 1.090 |

