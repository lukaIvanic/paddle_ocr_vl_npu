# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b1s512_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/scaled_masked_softmax_b1s512_pipe/liteserver-c001-4_754080_20260730135531601_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `34128.900 us`
- `Free`: `2433.940 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3161.250 us`
- `Stage`: `36562.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 8094.880 |
| `StridedSliceD` | 324 | 4529.140 |
| `ScaledMaskedSoftmax` | 81 | 4241.140 |
| `Transpose` | 324 | 2759.180 |
| `AddLayerNorm` | 162 | 2024.560 |
| `Mul` | 324 | 1889.480 |
| `TransData` | 243 | 1821.380 |
| `ConcatV2D` | 243 | 1622.460 |
| `BatchMatMul` | 162 | 1546.820 |
| `Neg` | 162 | 1299.900 |
| `Add` | 162 | 1276.760 |
| `Cast` | 162 | 1237.980 |
| `Gelu` | 81 | 1183.640 |
| `SplitVD` | 81 | 543.080 |
| `LayerNormV3` | 3 | 43.060 |
| `Data` | 3 | 15.440 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax_19` | 3 | 163.980 |
| `ScaledMaskedSoftmax_5` | 3 | 157.240 |
| `ScaledMaskedSoftmax_16` | 3 | 157.220 |
| `ScaledMaskedSoftmax_14` | 3 | 157.140 |
| `ScaledMaskedSoftmax_18` | 3 | 157.120 |
| `ScaledMaskedSoftmax_25` | 3 | 157.120 |
| `ScaledMaskedSoftmax_4` | 3 | 157.040 |
| `ScaledMaskedSoftmax_8` | 3 | 157.020 |
| `ScaledMaskedSoftmax_24` | 3 | 157.020 |
| `ScaledMaskedSoftmax_2` | 3 | 156.980 |
| `ScaledMaskedSoftmax_22` | 3 | 156.960 |
| `ScaledMaskedSoftmax_10` | 3 | 156.960 |
| `ScaledMaskedSoftmax_3` | 3 | 156.880 |
| `ScaledMaskedSoftmax_13` | 3 | 156.860 |
| `ScaledMaskedSoftmax_23` | 3 | 156.860 |
| `ScaledMaskedSoftmax_9` | 3 | 156.840 |
| `ScaledMaskedSoftmax_21` | 3 | 156.840 |
| `ScaledMaskedSoftmax_26` | 3 | 156.840 |
| `ScaledMaskedSoftmax_12` | 3 | 156.840 |
| `ScaledMaskedSoftmax_1` | 3 | 156.820 |
| `ScaledMaskedSoftmax_17` | 3 | 156.800 |
| `ScaledMaskedSoftmax_20` | 3 | 156.800 |
| `ScaledMaskedSoftmax_6` | 3 | 156.780 |
| `ScaledMaskedSoftmax_15` | 3 | 156.720 |
| `ScaledMaskedSoftmax_7` | 3 | 156.700 |
| `ScaledMaskedSoftmax_11` | 3 | 156.660 |
| `ScaledMaskedSoftmax` | 3 | 154.100 |
| `MatMulV2_5` | 3 | 85.060 |
| `MatMulV2_149` | 3 | 79.820 |
| `MatMulV2_53` | 3 | 79.020 |
| `MatMulV2_4` | 3 | 78.360 |
| `MatMulV2_47` | 3 | 76.900 |
| `MatMulV2_113` | 3 | 76.660 |
| `MatMulV2_35` | 3 | 76.600 |
| `MatMulV2_17` | 3 | 76.360 |
| `MatMulV2_65` | 3 | 76.160 |
| `MatMulV2_29` | 3 | 75.980 |
| `MatMulV2_83` | 3 | 75.940 |
| `MatMulV2_41` | 3 | 75.860 |
| `MatMulV2_59` | 3 | 75.820 |
| `MatMulV2_89` | 3 | 75.700 |
| `MatMulV2_107` | 3 | 75.700 |
| `MatMulV2_161` | 3 | 75.680 |
| `MatMulV2_23` | 3 | 75.440 |
| `MatMulV2_142` | 3 | 75.360 |
| `MatMulV2_40` | 3 | 75.340 |
| `MatMulV2_155` | 3 | 75.320 |
| `MatMulV2_64` | 3 | 75.280 |
| `MatMulV2_77` | 3 | 75.280 |
| `MatMulV2_16` | 3 | 75.020 |
| `MatMulV2_119` | 3 | 75.000 |
| `MatMulV2_125` | 3 | 74.900 |
| `MatMulV2_46` | 3 | 74.840 |
| `MatMulV2_71` | 3 | 74.760 |
| `MatMulV2_22` | 3 | 74.500 |
| `MatMulV2_101` | 3 | 74.400 |
| `MatMulV2_52` | 3 | 74.380 |
| `MatMulV2_131` | 3 | 74.380 |
| `MatMulV2_11` | 3 | 74.280 |
| `MatMulV2_137` | 3 | 74.080 |
| `MatMulV2_143` | 3 | 73.980 |
| `MatMulV2_58` | 3 | 73.420 |
| `MatMulV2_130` | 3 | 73.060 |
| `MatMulV2_10` | 3 | 73.040 |
| `MatMulV2_28` | 3 | 72.980 |
| `MatMulV2_160` | 3 | 72.960 |
| `MatMulV2_94` | 3 | 72.700 |
| `MatMulV2_112` | 3 | 72.600 |
| `MatMulV2_76` | 3 | 72.560 |
| `MatMulV2_100` | 3 | 72.500 |
| `MatMulV2_118` | 3 | 72.400 |
| `MatMulV2_95` | 3 | 72.280 |
| `MatMulV2_82` | 3 | 72.100 |
| `MatMulV2_88` | 3 | 71.840 |
| `MatMulV2_124` | 3 | 71.840 |
| `MatMulV2_136` | 3 | 71.820 |
| `MatMulV2_154` | 3 | 71.600 |
| `MatMulV2_34` | 3 | 71.260 |
| `MatMulV2_148` | 3 | 70.020 |
| `MatMulV2_70` | 3 | 69.200 |
| `MatMulV2_106` | 3 | 68.780 |
| `MatMulV2` | 3 | 68.700 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 63.220 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 60.400 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 59.720 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 58.900 |
| `StridedSliceV2_28` | 3 | 56.220 |
| `StridedSliceV2_29` | 3 | 56.140 |
| `MatMulV2_3` | 3 | 55.540 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 55.240 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 48.500 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 48.340 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 48.300 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 48.180 |
| `MatMulV2_96` | 3 | 48.140 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 48.120 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 48.100 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 47.920 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 47.880 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 47.860 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 47.840 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 47.780 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 47.780 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 47.760 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 47.760 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 47.700 |
| `MatMulV2_42` | 3 | 47.680 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 47.640 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 47.640 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 47.560 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 47.540 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 47.520 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 47.520 |
| `StridedSliceV2_26` | 3 | 47.480 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 47.480 |
| `StridedSliceV2_27` | 3 | 47.340 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 47.120 |
| `MatMulV2_30` | 3 | 45.820 |
| `StridedSliceV2` | 3 | 45.760 |
| `MatMulV2_150` | 3 | 45.700 |
| `MatMulV2_48` | 3 | 45.660 |
| `StridedSliceV2_8` | 3 | 45.660 |
| `StridedSliceV2_44` | 3 | 45.400 |
| `MatMulV2_90` | 3 | 45.380 |
| `MatMulV2_36` | 3 | 45.320 |
| `MatMulV2_138` | 3 | 45.320 |
| `MatMulV2_102` | 3 | 45.300 |
| `StridedSliceV2_32` | 3 | 45.280 |
| `MatMulV2_72` | 3 | 45.280 |
| `StridedSliceV2_84` | 3 | 45.280 |
| `MatMulV2_6` | 3 | 45.260 |
| `StridedSliceV2_16` | 3 | 45.260 |
| `StridedSliceV2_100` | 3 | 45.160 |
| `MatMulV2_18` | 3 | 45.060 |
| `MatMulV2_114` | 3 | 45.040 |
| `StridedSliceV2_56` | 3 | 44.920 |
| `StridedSliceV2_92` | 3 | 44.880 |
| `StridedSliceV2_96` | 3 | 44.880 |
| `StridedSliceV2_40` | 3 | 44.700 |
| `StridedSliceV2_60` | 3 | 44.660 |
| `StridedSliceV2_88` | 3 | 44.620 |
| `MatMulV2_60` | 3 | 44.580 |
| `StridedSliceV2_64` | 3 | 44.580 |
| `MatMulV2_24` | 3 | 44.500 |
| `MatMulV2_144` | 3 | 44.480 |
| `StridedSliceV2_20` | 3 | 44.360 |
| `StridedSliceV2_36` | 3 | 44.340 |
| `StridedSliceV2_1` | 3 | 44.280 |
| `StridedSliceV2_12` | 3 | 44.280 |
| `MatMulV2_84` | 3 | 44.280 |
| `MatMulV2_54` | 3 | 44.180 |
| `Gelu_17` | 3 | 44.180 |
| `Gelu_4` | 3 | 44.160 |
| `StridedSliceV2_68` | 3 | 44.160 |
| `Gelu_3` | 3 | 44.140 |
| `MatMulV2_156` | 3 | 44.100 |
| `Gelu_15` | 3 | 44.100 |
| `Gelu_24` | 3 | 44.100 |
| `Gelu_21` | 3 | 44.080 |
| `StridedSliceV2_80` | 3 | 44.060 |
| `Gelu_23` | 3 | 44.060 |
| `Gelu_16` | 3 | 44.040 |
| `Gelu_10` | 3 | 44.020 |
| `Gelu_5` | 3 | 44.000 |
| `Gelu_9` | 3 | 44.000 |
| `Gelu_11` | 3 | 43.960 |
| `MatMulV2_132` | 3 | 43.960 |
| `BatchMatMul_1` | 3 | 43.840 |
| `MatMulV2_66` | 3 | 43.820 |
| `Gelu_25` | 3 | 43.800 |
| `Gelu` | 3 | 43.740 |
| `Gelu_8` | 3 | 43.700 |
| `Gelu_13` | 3 | 43.700 |
| `Gelu_18` | 3 | 43.700 |
| `MatMulV2_126` | 3 | 43.700 |
| `StridedSliceV2_52` | 3 | 43.660 |
| `Gelu_6` | 3 | 43.660 |
| `Gelu_12` | 3 | 43.660 |
| `Gelu_19` | 3 | 43.660 |
| `Gelu_1` | 3 | 43.640 |
| `Gelu_7` | 3 | 43.620 |
| `MatMulV2_108` | 3 | 43.620 |
| `Gelu_20` | 3 | 43.620 |
| `Gelu_26` | 3 | 43.620 |
| `Gelu_2` | 3 | 43.600 |
| `Gelu_22` | 3 | 43.580 |
| `Gelu_14` | 3 | 43.500 |
| `MatMulV2_120` | 3 | 43.500 |
| `MatMulV2_78` | 3 | 43.420 |
| `StridedSliceV2_76` | 3 | 43.420 |
| `StridedSliceV2_105` | 3 | 43.260 |
| `StridedSliceV2_48` | 3 | 43.240 |
| `LayerNormV4_LayerNormV3` | 3 | 43.060 |
| `StridedSliceV2_9` | 3 | 42.980 |
| `StridedSliceV2_24` | 3 | 42.960 |
| `StridedSliceV2_72` | 3 | 42.880 |
| `StridedSliceV2_45` | 3 | 42.780 |
| `StridedSliceV2_101` | 3 | 42.720 |
| `StridedSliceV2_81` | 3 | 42.700 |
| `StridedSliceV2_5` | 3 | 42.660 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "1,512,16,80" -> "1,512,16,40" | ND -> ND` | 324 | 4529.140 |
| `ScaledMaskedSoftmax | "1,16,512,512;1,1,512,512" -> "1,16,512,512" | NCHW;NCHW -> NCHW` | 81 | 4241.140 |
| `MatMulV2 | "512,1152;72,80,16,16;1280" -> "512,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 3179.740 |
| `MatMulV2 | "512,4352;272,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2051.360 |
| `Transpose | "512,16,80;3" -> "16,512,80" | ND;ND -> ND` | 243 | 2033.940 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2024.560 |
| `MatMulV2 | "512,1152;72,272,16,16;4352" -> "512,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 1969.760 |
| `Mul | "1,512,16,80;1,512,1,80" -> "1,512,16,80" | ND;ND -> ND` | 324 | 1889.480 |
| `TransData | "16,512,80" -> "16,5,32,16,16" | ND -> FRACTAL_NZ` | 243 | 1821.380 |
| `Neg | "1,512,16,40" -> "1,512,16,40" | ND -> ND` | 162 | 1299.900 |
| `Add | "1,512,16,80;1,512,16,80" -> "1,512,16,80" | ND;ND -> ND` | 162 | 1276.760 |
| `Cast | "1,512,16,80" -> "1,512,16,80" | ND -> ND` | 162 | 1237.980 |
| `Gelu | "1,512,4352" -> "1,512,4352" | ND -> ND` | 81 | 1183.640 |
| `ConcatV2D | "1,512,16,40;1,512,16,40" -> "1,512,16,80" | ND;ND -> ND` | 162 | 1000.300 |
| `MatMulV2 | "512,1280;80,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 894.020 |
| `BatchMatMul | "16,512,512;16,5,32,16,16" -> "16,512,80" | ND;FRACTAL_NZ -> ND` | 81 | 821.340 |
| `BatchMatMul | "16,5,32,16,16;16,5,32,16,16" -> "16,512,512" | FRACTAL_NZ;FRACTAL_NZ -> ND` | 81 | 725.480 |
| `Transpose | "16,512,80;3" -> "512,16,80" | ND;ND -> ND` | 81 | 725.240 |
| `ConcatV2D | "1,512,1280;1,512,1280;1,512,1280" -> "1,512,3840" | ND;ND;ND -> ND` | 81 | 622.160 |
| `SplitVD | "1,512,3840" -> "1,512,1280;1,512,1280;1,512,1280" | ND -> ND;ND;ND` | 81 | 543.080 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 43.060 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.440 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND` | 1053 | 10615.120 |
| `ND;FRACTAL_NZ;ND` | 486 | 8094.880 |
| `ND;ND` | 972 | 6925.720 |
| `NCHW;NCHW` | 81 | 4241.140 |
| `ND;ND;ND;ND` | 162 | 2024.560 |
| `ND;FRACTAL_NZ` | 81 | 821.340 |
| `FRACTAL_NZ;FRACTAL_NZ` | 81 | 725.480 |
| `ND;ND;ND` | 84 | 665.220 |
| `N/A` | 3 | 15.440 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `ScaledMaskedSoftmax_19` | 0 | 54.780 |
| `ScaledMaskedSoftmax_19` | 0 | 54.740 |
| `ScaledMaskedSoftmax_19` | 0 | 54.460 |
| `ScaledMaskedSoftmax_14` | 0 | 52.560 |
| `ScaledMaskedSoftmax_5` | 0 | 52.500 |
| `ScaledMaskedSoftmax_25` | 0 | 52.500 |
| `ScaledMaskedSoftmax_16` | 0 | 52.460 |
| `ScaledMaskedSoftmax_9` | 0 | 52.460 |
| `ScaledMaskedSoftmax_16` | 0 | 52.460 |
| `ScaledMaskedSoftmax_18` | 0 | 52.460 |
| `ScaledMaskedSoftmax_4` | 0 | 52.440 |
| `ScaledMaskedSoftmax_26` | 0 | 52.440 |
| `ScaledMaskedSoftmax_8` | 0 | 52.420 |
| `ScaledMaskedSoftmax_1` | 0 | 52.420 |
| `ScaledMaskedSoftmax_4` | 0 | 52.420 |
| `ScaledMaskedSoftmax_17` | 0 | 52.400 |
| `ScaledMaskedSoftmax_14` | 0 | 52.400 |
| `ScaledMaskedSoftmax_5` | 0 | 52.380 |
| `ScaledMaskedSoftmax_25` | 0 | 52.380 |
| `ScaledMaskedSoftmax_3` | 0 | 52.380 |
| `ScaledMaskedSoftmax_22` | 0 | 52.380 |
| `ScaledMaskedSoftmax_18` | 0 | 52.360 |
| `ScaledMaskedSoftmax_24` | 0 | 52.360 |
| `ScaledMaskedSoftmax_6` | 0 | 52.360 |
| `ScaledMaskedSoftmax_21` | 0 | 52.360 |
| `ScaledMaskedSoftmax_2` | 0 | 52.360 |
| `ScaledMaskedSoftmax_5` | 0 | 52.360 |
| `ScaledMaskedSoftmax_8` | 0 | 52.360 |
| `ScaledMaskedSoftmax_10` | 0 | 52.360 |
| `ScaledMaskedSoftmax_24` | 0 | 52.360 |
| `ScaledMaskedSoftmax_2` | 0 | 52.340 |
| `ScaledMaskedSoftmax_22` | 0 | 52.340 |
| `ScaledMaskedSoftmax_23` | 0 | 52.340 |
| `ScaledMaskedSoftmax_10` | 0 | 52.340 |
| `ScaledMaskedSoftmax_20` | 0 | 52.340 |
| `ScaledMaskedSoftmax_13` | 0 | 52.320 |
| `ScaledMaskedSoftmax_13` | 0 | 52.320 |
| `ScaledMaskedSoftmax_20` | 0 | 52.300 |
| `ScaledMaskedSoftmax_21` | 0 | 52.300 |
| `ScaledMaskedSoftmax_15` | 0 | 52.300 |
| `ScaledMaskedSoftmax_18` | 0 | 52.300 |
| `ScaledMaskedSoftmax_24` | 0 | 52.300 |
| `ScaledMaskedSoftmax_12` | 0 | 52.300 |
| `ScaledMaskedSoftmax_16` | 0 | 52.300 |
| `ScaledMaskedSoftmax_7` | 0 | 52.280 |
| `ScaledMaskedSoftmax_11` | 0 | 52.280 |
| `ScaledMaskedSoftmax_12` | 0 | 52.280 |
| `ScaledMaskedSoftmax_2` | 0 | 52.280 |
| `ScaledMaskedSoftmax_3` | 0 | 52.280 |
| `ScaledMaskedSoftmax_6` | 0 | 52.280 |
| `ScaledMaskedSoftmax_23` | 0 | 52.280 |
| `ScaledMaskedSoftmax_10` | 0 | 52.260 |
| `ScaledMaskedSoftmax_7` | 0 | 52.260 |
| `ScaledMaskedSoftmax_12` | 0 | 52.260 |
| `ScaledMaskedSoftmax_1` | 0 | 52.240 |
| `ScaledMaskedSoftmax_9` | 0 | 52.240 |
| `ScaledMaskedSoftmax_8` | 0 | 52.240 |
| `ScaledMaskedSoftmax_22` | 0 | 52.240 |
| `ScaledMaskedSoftmax_23` | 0 | 52.240 |
| `ScaledMaskedSoftmax_25` | 0 | 52.240 |
| `ScaledMaskedSoftmax_3` | 0 | 52.220 |
| `ScaledMaskedSoftmax_15` | 0 | 52.220 |
| `ScaledMaskedSoftmax_11` | 0 | 52.220 |
| `ScaledMaskedSoftmax_13` | 0 | 52.220 |
| `ScaledMaskedSoftmax_17` | 0 | 52.200 |
| `ScaledMaskedSoftmax_26` | 0 | 52.200 |
| `ScaledMaskedSoftmax_15` | 0 | 52.200 |
| `ScaledMaskedSoftmax_17` | 0 | 52.200 |
| `ScaledMaskedSoftmax_26` | 0 | 52.200 |
| `ScaledMaskedSoftmax_4` | 0 | 52.180 |
| `ScaledMaskedSoftmax_14` | 0 | 52.180 |
| `ScaledMaskedSoftmax_21` | 0 | 52.180 |
| `ScaledMaskedSoftmax_20` | 0 | 52.160 |
| `ScaledMaskedSoftmax_1` | 0 | 52.160 |
| `ScaledMaskedSoftmax_7` | 0 | 52.160 |
| `ScaledMaskedSoftmax_11` | 0 | 52.160 |
| `ScaledMaskedSoftmax_6` | 0 | 52.140 |
| `ScaledMaskedSoftmax_9` | 0 | 52.140 |
| `ScaledMaskedSoftmax` | 0 | 51.500 |
| `ScaledMaskedSoftmax` | 0 | 51.380 |
| `ScaledMaskedSoftmax` | 0 | 51.220 |
| `MatMulV2_5` | 0 | 28.700 |
| `MatMulV2_5` | 0 | 28.580 |
| `MatMulV2_5` | 0 | 27.780 |
| `MatMulV2_53` | 0 | 27.160 |
| `MatMulV2_149` | 0 | 27.060 |
| `MatMulV2_4` | 0 | 26.840 |
| `MatMulV2_16` | 0 | 26.840 |
| `MatMulV2_17` | 0 | 26.840 |
| `MatMulV2_64` | 0 | 26.820 |
| `MatMulV2_46` | 0 | 26.800 |
| `MatMulV2_40` | 0 | 26.740 |
| `MatMulV2_149` | 0 | 26.700 |
| `MatMulV2_10` | 0 | 26.620 |
| `MatMulV2_47` | 0 | 26.420 |
| `MatMulV2_23` | 0 | 26.360 |
| `MatMulV2_53` | 0 | 26.340 |
| `MatMulV2_59` | 0 | 26.300 |
| `MatMulV2_22` | 0 | 26.120 |
| `MatMulV2_11` | 0 | 26.100 |
| `MatMulV2_149` | 0 | 26.060 |
| `MatMulV2_35` | 0 | 25.960 |
| `MatMulV2_113` | 0 | 25.960 |
| `MatMulV2_52` | 0 | 25.840 |
| `MatMulV2_28` | 0 | 25.800 |
| `MatMulV2_4` | 0 | 25.800 |
| `MatMulV2_4` | 0 | 25.720 |
| `MatMulV2_17` | 0 | 25.640 |
| `MatMulV2_53` | 0 | 25.520 |
| `MatMulV2_41` | 0 | 25.480 |
| `MatMulV2_107` | 0 | 25.480 |
| `MatMulV2_161` | 0 | 25.460 |
| `MatMulV2_29` | 0 | 25.460 |
| `MatMulV2_65` | 0 | 25.460 |
| `MatMulV2_65` | 0 | 25.440 |
| `MatMulV2_113` | 0 | 25.440 |
| `MatMulV2_47` | 0 | 25.420 |
| `MatMulV2_101` | 0 | 25.420 |
| `MatMulV2_83` | 0 | 25.420 |
| `MatMulV2_35` | 0 | 25.400 |
| `MatMulV2_29` | 0 | 25.360 |
| `MatMulV2_131` | 0 | 25.360 |
| `MatMulV2_89` | 0 | 25.340 |
| `MatMulV2_41` | 0 | 25.320 |
| `MatMulV2_59` | 0 | 25.320 |
| `MatMulV2_137` | 0 | 25.300 |
| `MatMulV2_142` | 0 | 25.300 |
| `MatMulV2_143` | 0 | 25.300 |
| `MatMulV2_83` | 0 | 25.280 |
| `MatMulV2_65` | 0 | 25.260 |
| `MatMulV2_113` | 0 | 25.260 |
| `MatMulV2_83` | 0 | 25.240 |
| `MatMulV2_23` | 0 | 25.240 |
| `MatMulV2_35` | 0 | 25.240 |
| `MatMulV2_107` | 0 | 25.240 |
| `MatMulV2_89` | 0 | 25.200 |
| `MatMulV2_137` | 0 | 25.200 |
| `MatMulV2_142` | 0 | 25.200 |
| `MatMulV2_77` | 0 | 25.180 |
| `MatMulV2_29` | 0 | 25.160 |
| `MatMulV2_155` | 0 | 25.160 |
| `MatMulV2_161` | 0 | 25.160 |
| `MatMulV2_89` | 0 | 25.160 |
| `MatMulV2_101` | 0 | 25.160 |
| `MatMulV2_119` | 0 | 25.100 |
| `MatMulV2_77` | 0 | 25.100 |
| `MatMulV2_155` | 0 | 25.100 |
| `MatMulV2_71` | 0 | 25.080 |
| `MatMulV2_95` | 0 | 25.060 |
| `MatMulV2_155` | 0 | 25.060 |
| `MatMulV2_41` | 0 | 25.060 |
| `MatMulV2_47` | 0 | 25.060 |
| `MatMulV2_119` | 0 | 25.060 |
| `MatMulV2_161` | 0 | 25.060 |
| `MatMulV2_71` | 0 | 25.020 |
| `MatMulV2_77` | 0 | 25.000 |
| `MatMulV2_107` | 0 | 24.980 |
| `MatMulV2_125` | 0 | 24.980 |
| `MatMulV2_125` | 0 | 24.980 |
| `MatMulV2_131` | 0 | 24.940 |
| `MatMulV2_125` | 0 | 24.940 |
| `MatMulV2_143` | 0 | 24.920 |
| `MatMulV2_58` | 0 | 24.880 |
| `MatMulV2_142` | 0 | 24.860 |
| `MatMulV2_119` | 0 | 24.840 |
| `MatMulV2_130` | 0 | 24.760 |
| `MatMulV2_71` | 0 | 24.660 |
| `MatMulV2_160` | 0 | 24.480 |
| `MatMulV2_112` | 0 | 24.420 |
| `MatMulV2_52` | 0 | 24.420 |
| `MatMulV2` | 0 | 24.400 |
| `MatMulV2_112` | 0 | 24.400 |
| `MatMulV2_40` | 0 | 24.380 |
| `MatMulV2_94` | 0 | 24.360 |
| `MatMulV2_118` | 0 | 24.360 |
| `MatMulV2_160` | 0 | 24.320 |
| `MatMulV2_76` | 0 | 24.300 |
| `MatMulV2_94` | 0 | 24.300 |
| `MatMulV2_58` | 0 | 24.300 |
| `MatMulV2_64` | 0 | 24.300 |
| `MatMulV2_100` | 0 | 24.280 |
| `MatMulV2_16` | 0 | 24.280 |
| `MatMulV2_154` | 0 | 24.280 |
| `MatMulV2_82` | 0 | 24.260 |
| `MatMulV2_22` | 0 | 24.240 |
| `MatMulV2_58` | 0 | 24.240 |
| `MatMulV2_28` | 0 | 24.240 |
| `MatMulV2_34` | 0 | 24.220 |
| `MatMulV2_40` | 0 | 24.220 |
| `MatMulV2_154` | 0 | 24.200 |
| `MatMulV2_34` | 0 | 24.200 |
| `MatMulV2_59` | 0 | 24.200 |
| `MatMulV2_46` | 0 | 24.200 |
| `MatMulV2_130` | 0 | 24.180 |
| `MatMulV2_64` | 0 | 24.160 |
| `MatMulV2_148` | 0 | 24.160 |
| `MatMulV2_160` | 0 | 24.160 |
| `MatMulV2_76` | 0 | 24.160 |
| `MatMulV2_82` | 0 | 24.160 |
| `MatMulV2_11` | 0 | 24.140 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 13749.260 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step1` | 1 | 13083.330 |
| `TorchNpuGraphBase::Run` | 3 | 13079.340 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step2` | 1 | 12334.820 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.scaled_masked_softmax.separate_manual.torchair.active.step3` | 1 | 12231.120 |
| `AssembleInputs` | 3 | 11658.980 |
| `RefreshAtTensorFromGeTensor` | 3 | 1074.110 |
| `ExecuteGraph` | 3 | 568.420 |
| `aten::empty` | 3 | 540.830 |
| `empty_tensor` | 3 | 282.050 |
| `AssembleOutputs` | 3 | 272.320 |
| `aten::set_` | 3 | 256.470 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 178711.200 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 32355.890 |
| `launch` | 1003 | 11812.870 |
| `InputCopy` | 3 | 217.700 |
| `ModelExecute` | 3 | 63.690 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 37.410 |
| `step_info` | 6 | 32.250 |
| `OutputCopy` | 3 | 1.610 |

