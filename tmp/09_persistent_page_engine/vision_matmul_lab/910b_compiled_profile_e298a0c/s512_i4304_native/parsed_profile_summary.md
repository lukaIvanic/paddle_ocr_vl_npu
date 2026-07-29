# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s512_i4304_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s512_i4304_native/liteserver-c001-4_616610_20260729131916297_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `16797.660 us`
- `Free`: `2682.680 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3891.000 us`
- `Stage`: `19480.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 14237.960 |
| `AddLayerNorm` | 162 | 1358.620 |
| `Gelu` | 81 | 839.060 |
| `AutomaticBufferFusionOp` | 81 | 305.500 |
| `LayerNormV3` | 3 | 43.220 |
| `Data` | 3 | 13.300 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_77` | 3 | 251.400 |
| `MatMulV2_5` | 3 | 249.500 |
| `MatMulV2_131` | 3 | 249.040 |
| `MatMulV2_17` | 3 | 248.960 |
| `MatMulV2_125` | 3 | 247.560 |
| `MatMulV2_83` | 3 | 247.480 |
| `MatMulV2_95` | 3 | 246.320 |
| `MatMulV2_155` | 3 | 246.240 |
| `MatMulV2_47` | 3 | 245.800 |
| `MatMulV2_119` | 3 | 245.620 |
| `MatMulV2_65` | 3 | 245.500 |
| `MatMulV2_59` | 3 | 245.180 |
| `MatMulV2_35` | 3 | 245.140 |
| `MatMulV2_23` | 3 | 244.360 |
| `MatMulV2_137` | 3 | 244.320 |
| `MatMulV2_89` | 3 | 244.240 |
| `MatMulV2_71` | 3 | 244.180 |
| `MatMulV2_143` | 3 | 243.700 |
| `MatMulV2_161` | 3 | 243.380 |
| `MatMulV2_29` | 3 | 243.320 |
| `MatMulV2_149` | 3 | 243.260 |
| `MatMulV2_11` | 3 | 243.220 |
| `MatMulV2_101` | 3 | 243.220 |
| `MatMulV2_53` | 3 | 243.140 |
| `MatMulV2_41` | 3 | 242.580 |
| `MatMulV2_107` | 3 | 241.360 |
| `MatMulV2_113` | 3 | 240.800 |
| `MatMulV2_4` | 3 | 129.540 |
| `MatMulV2_28` | 3 | 125.380 |
| `MatMulV2_88` | 3 | 123.800 |
| `MatMulV2_148` | 3 | 123.740 |
| `MatMulV2_124` | 3 | 123.720 |
| `MatMulV2_112` | 3 | 123.240 |
| `MatMulV2_76` | 3 | 123.040 |
| `MatMulV2_100` | 3 | 122.880 |
| `MatMulV2_16` | 3 | 122.680 |
| `MatMulV2_52` | 3 | 122.540 |
| `MatMulV2_46` | 3 | 122.040 |
| `MatMulV2_40` | 3 | 122.020 |
| `MatMulV2_136` | 3 | 121.960 |
| `MatMulV2_94` | 3 | 121.360 |
| `MatMulV2_160` | 3 | 121.140 |
| `MatMulV2_106` | 3 | 121.060 |
| `MatMulV2_82` | 3 | 121.020 |
| `MatMulV2_10` | 3 | 120.940 |
| `MatMulV2_154` | 3 | 120.860 |
| `MatMulV2_142` | 3 | 120.840 |
| `MatMulV2_58` | 3 | 120.740 |
| `MatMulV2_64` | 3 | 120.660 |
| `MatMulV2_118` | 3 | 120.460 |
| `MatMulV2_34` | 3 | 120.340 |
| `MatMulV2_70` | 3 | 120.060 |
| `MatMulV2_130` | 3 | 120.060 |
| `MatMulV2_22` | 3 | 119.720 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 55.940 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 51.280 |
| `Gelu` | 3 | 49.520 |
| `MatMulV2` | 3 | 49.060 |
| `LayerNormV4_LayerNormV3` | 3 | 43.220 |
| `MatMulV2_73` | 3 | 41.060 |
| `MatMulV2_75` | 3 | 41.060 |
| `MatMulV2_27` | 3 | 41.040 |
| `MatMulV2_18` | 3 | 40.960 |
| `MatMulV2_51` | 3 | 40.960 |
| `MatMulV2_36` | 3 | 40.940 |
| `MatMulV2_1` | 3 | 40.900 |
| `MatMulV2_102` | 3 | 40.880 |
| `MatMulV2_150` | 3 | 40.860 |
| `MatMulV2_147` | 3 | 40.800 |
| `MatMulV2_97` | 3 | 40.760 |
| `MatMulV2_90` | 3 | 40.740 |
| `MatMulV2_12` | 3 | 40.700 |
| `MatMulV2_85` | 3 | 40.700 |
| `MatMulV2_157` | 3 | 40.700 |
| `MatMulV2_133` | 3 | 40.660 |
| `MatMulV2_78` | 3 | 40.560 |
| `MatMulV2_84` | 3 | 40.560 |
| `MatMulV2_114` | 3 | 40.560 |
| `MatMulV2_126` | 3 | 40.540 |
| `MatMulV2_61` | 3 | 40.480 |
| `MatMulV2_30` | 3 | 40.460 |
| `MatMulV2_156` | 3 | 40.420 |
| `MatMulV2_96` | 3 | 40.400 |
| `MatMulV2_39` | 3 | 40.380 |
| `MatMulV2_25` | 3 | 40.340 |
| `MatMulV2_42` | 3 | 40.340 |
| `MatMulV2_99` | 3 | 40.340 |
| `MatMulV2_6` | 3 | 40.320 |
| `MatMulV2_121` | 3 | 40.300 |
| `MatMulV2_153` | 3 | 40.260 |
| `MatMulV2_145` | 3 | 40.260 |
| `MatMulV2_123` | 3 | 40.240 |
| `MatMulV2_54` | 3 | 40.220 |
| `MatMulV2_105` | 3 | 40.220 |
| `MatMulV2_109` | 3 | 40.200 |
| `MatMulV2_66` | 3 | 40.180 |
| `MatMulV2_44` | 3 | 40.160 |
| `MatMulV2_48` | 3 | 40.160 |
| `MatMulV2_120` | 3 | 40.160 |
| `MatMulV2_24` | 3 | 40.120 |
| `MatMulV2_57` | 3 | 40.080 |
| `MatMulV2_81` | 3 | 40.080 |
| `MatMulV2_3` | 3 | 40.060 |
| `MatMulV2_33` | 3 | 40.060 |
| `MatMulV2_91` | 3 | 40.060 |
| `MatMulV2_87` | 3 | 40.000 |
| `MatMulV2_108` | 3 | 40.000 |
| `MatMulV2_37` | 3 | 39.980 |
| `MatMulV2_129` | 3 | 39.940 |
| `MatMulV2_43` | 3 | 39.940 |
| `MatMulV2_49` | 3 | 39.940 |
| `MatMulV2_2` | 3 | 39.920 |
| `MatMulV2_9` | 3 | 39.900 |
| `MatMulV2_93` | 3 | 39.880 |
| `MatMulV2_31` | 3 | 39.860 |
| `MatMulV2_67` | 3 | 39.860 |
| `MatMulV2_98` | 3 | 39.860 |
| `MatMulV2_134` | 3 | 39.820 |
| `MatMulV2_19` | 3 | 39.800 |
| `MatMulV2_138` | 3 | 39.800 |
| `MatMulV2_132` | 3 | 39.780 |
| `MatMulV2_21` | 3 | 39.760 |
| `MatMulV2_151` | 3 | 39.760 |
| `MatMulV2_127` | 3 | 39.740 |
| `MatMulV2_74` | 3 | 39.720 |
| `MatMulV2_86` | 3 | 39.720 |
| `MatMulV2_146` | 3 | 39.720 |
| `MatMulV2_111` | 3 | 39.700 |
| `MatMulV2_144` | 3 | 39.700 |
| `MatMulV2_139` | 3 | 39.700 |
| `MatMulV2_92` | 3 | 39.680 |
| `MatMulV2_63` | 3 | 39.660 |
| `MatMulV2_7` | 3 | 39.620 |
| `MatMulV2_26` | 3 | 39.600 |
| `MatMulV2_122` | 3 | 39.600 |
| `MatMulV2_8` | 3 | 39.580 |
| `MatMulV2_72` | 3 | 39.580 |
| `MatMulV2_56` | 3 | 39.560 |
| `MatMulV2_62` | 3 | 39.560 |
| `MatMulV2_79` | 3 | 39.540 |
| `MatMulV2_140` | 3 | 39.520 |
| `MatMulV2_152` | 3 | 39.500 |
| `MatMulV2_135` | 3 | 39.480 |
| `MatMulV2_13` | 3 | 39.460 |
| `MatMulV2_20` | 3 | 39.460 |
| `MatMulV2_103` | 3 | 39.420 |
| `MatMulV2_116` | 3 | 39.420 |
| `MatMulV2_128` | 3 | 39.420 |
| `MatMulV2_60` | 3 | 39.380 |
| `MatMulV2_141` | 3 | 39.340 |
| `MatMulV2_115` | 3 | 39.320 |
| `MatMulV2_55` | 3 | 39.300 |
| `MatMulV2_68` | 3 | 39.280 |
| `MatMulV2_117` | 3 | 39.280 |
| `MatMulV2_38` | 3 | 39.240 |
| `MatMulV2_110` | 3 | 39.200 |
| `MatMulV2_50` | 3 | 39.180 |
| `MatMulV2_104` | 3 | 39.180 |
| `MatMulV2_80` | 3 | 39.160 |
| `MatMulV2_32` | 3 | 39.160 |
| `MatMulV2_15` | 3 | 39.140 |
| `MatMulV2_158` | 3 | 39.080 |
| `MatMulV2_159` | 3 | 39.040 |
| `MatMulV2_69` | 3 | 38.860 |
| `MatMulV2_45` | 3 | 38.760 |
| `MatMulV2_14` | 3 | 38.640 |
| `Gelu_15` | 3 | 30.920 |
| `Gelu_10` | 3 | 30.860 |
| `Gelu_8` | 3 | 30.480 |
| `Gelu_17` | 3 | 30.480 |
| `Gelu_21` | 3 | 30.420 |
| `Gelu_7` | 3 | 30.420 |
| `Gelu_14` | 3 | 30.400 |
| `Gelu_13` | 3 | 30.380 |
| `Gelu_5` | 3 | 30.360 |
| `Gelu_11` | 3 | 30.360 |
| `Gelu_18` | 3 | 30.360 |
| `Gelu_20` | 3 | 30.340 |
| `Gelu_1` | 3 | 30.320 |
| `Gelu_9` | 3 | 30.320 |
| `Gelu_16` | 3 | 30.320 |
| `AddAdd_1Muls` | 3 | 30.300 |
| `Gelu_4` | 3 | 30.300 |
| `Gelu_6` | 3 | 30.300 |
| `Gelu_2` | 3 | 30.280 |
| `Gelu_3` | 3 | 30.280 |
| `Gelu_19` | 3 | 30.280 |
| `Gelu_12` | 3 | 30.260 |
| `Gelu_23` | 3 | 30.260 |
| `Gelu_25` | 3 | 30.260 |
| `Gelu_22` | 3 | 30.240 |
| `Gelu_26` | 3 | 30.180 |
| `Gelu_24` | 3 | 30.160 |
| `LayerNormV4_48_LayerNormV3/AddLayerNorm` | 3 | 27.360 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 26.820 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 26.680 |
| `LayerNormV4_8_LayerNormV3/AddLayerNorm` | 3 | 26.420 |
| `LayerNormV4_2_LayerNormV3/AddLayerNorm` | 3 | 26.060 |
| `LayerNormV4_46_LayerNormV3/AddLayerNorm` | 3 | 25.760 |
| `LayerNormV4_22_LayerNormV3/AddLayerNorm` | 3 | 25.700 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "512,4304;1152,4304;1152" -> "512,1152" | ND;ND;ND -> ND` | 81 | 6618.820 |
| `MatMulV2 | "512,1152;1152,1152;1152" -> "512,1152" | ND;ND;ND -> ND` | 324 | 4323.300 |
| `MatMulV2 | "512,1152;4304,1152;4304" -> "512,4304" | ND;ND;ND -> ND` | 81 | 3295.840 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 1358.620 |
| `Gelu | "1,512,4304" -> "1,512,4304" | ND -> ND` | 81 | 839.060 |
| `AutomaticBufferFusionOp | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,1152" | ND;ND;ND -> ND` | 81 | 305.500 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 43.220 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 13.300 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 14586.680 |
| `ND;ND;ND;ND` | 162 | 1358.620 |
| `ND` | 81 | 839.060 |
| `N/A` | 3 | 13.300 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_77` | 0 | 86.680 |
| `MatMulV2_131` | 0 | 84.440 |
| `MatMulV2_5` | 0 | 84.220 |
| `MatMulV2_125` | 0 | 83.980 |
| `MatMulV2_83` | 0 | 83.800 |
| `MatMulV2_143` | 0 | 83.740 |
| `MatMulV2_101` | 0 | 83.660 |
| `MatMulV2_35` | 0 | 83.400 |
| `MatMulV2_95` | 0 | 83.360 |
| `MatMulV2_17` | 0 | 83.340 |
| `MatMulV2_17` | 0 | 83.300 |
| `MatMulV2_161` | 0 | 83.080 |
| `MatMulV2_47` | 0 | 83.060 |
| `MatMulV2_77` | 0 | 82.920 |
| `MatMulV2_5` | 0 | 82.820 |
| `MatMulV2_131` | 0 | 82.780 |
| `MatMulV2_71` | 0 | 82.720 |
| `MatMulV2_65` | 0 | 82.680 |
| `MatMulV2_119` | 0 | 82.680 |
| `MatMulV2_41` | 0 | 82.660 |
| `MatMulV2_155` | 0 | 82.640 |
| `MatMulV2_119` | 0 | 82.520 |
| `MatMulV2_155` | 0 | 82.500 |
| `MatMulV2_5` | 0 | 82.460 |
| `MatMulV2_23` | 0 | 82.460 |
| `MatMulV2_83` | 0 | 82.360 |
| `MatMulV2_17` | 0 | 82.320 |
| `MatMulV2_125` | 0 | 82.260 |
| `MatMulV2_59` | 0 | 82.160 |
| `MatMulV2_89` | 0 | 82.120 |
| `MatMulV2_29` | 0 | 81.940 |
| `MatMulV2_23` | 0 | 81.900 |
| `MatMulV2_149` | 0 | 81.900 |
| `MatMulV2_113` | 0 | 81.900 |
| `MatMulV2_131` | 0 | 81.820 |
| `MatMulV2_107` | 0 | 81.800 |
| `MatMulV2_77` | 0 | 81.800 |
| `MatMulV2_149` | 0 | 81.740 |
| `MatMulV2_11` | 0 | 81.720 |
| `MatMulV2_137` | 0 | 81.720 |
| `MatMulV2_65` | 0 | 81.720 |
| `MatMulV2_53` | 0 | 81.620 |
| `MatMulV2_71` | 0 | 81.600 |
| `MatMulV2_59` | 0 | 81.580 |
| `MatMulV2_11` | 0 | 81.580 |
| `MatMulV2_47` | 0 | 81.580 |
| `MatMulV2_95` | 0 | 81.520 |
| `MatMulV2_137` | 0 | 81.460 |
| `MatMulV2_59` | 0 | 81.440 |
| `MatMulV2_95` | 0 | 81.440 |
| `MatMulV2_83` | 0 | 81.320 |
| `MatMulV2_125` | 0 | 81.320 |
| `MatMulV2_89` | 0 | 81.200 |
| `MatMulV2_47` | 0 | 81.160 |
| `MatMulV2_137` | 0 | 81.140 |
| `MatMulV2_65` | 0 | 81.100 |
| `MatMulV2_155` | 0 | 81.100 |
| `MatMulV2_143` | 0 | 81.060 |
| `MatMulV2_35` | 0 | 80.920 |
| `MatMulV2_89` | 0 | 80.920 |
| `MatMulV2_29` | 0 | 80.900 |
| `MatMulV2_41` | 0 | 80.860 |
| `MatMulV2_53` | 0 | 80.860 |
| `MatMulV2_35` | 0 | 80.820 |
| `MatMulV2_53` | 0 | 80.660 |
| `MatMulV2_107` | 0 | 80.520 |
| `MatMulV2_29` | 0 | 80.480 |
| `MatMulV2_119` | 0 | 80.420 |
| `MatMulV2_161` | 0 | 80.280 |
| `MatMulV2_161` | 0 | 80.020 |
| `MatMulV2_23` | 0 | 80.000 |
| `MatMulV2_11` | 0 | 79.920 |
| `MatMulV2_71` | 0 | 79.860 |
| `MatMulV2_101` | 0 | 79.800 |
| `MatMulV2_101` | 0 | 79.760 |
| `MatMulV2_149` | 0 | 79.620 |
| `MatMulV2_113` | 0 | 79.520 |
| `MatMulV2_113` | 0 | 79.380 |
| `MatMulV2_41` | 0 | 79.060 |
| `MatMulV2_107` | 0 | 79.040 |
| `MatMulV2_143` | 0 | 78.900 |
| `MatMulV2_4` | 0 | 43.240 |
| `MatMulV2_4` | 0 | 43.240 |
| `MatMulV2_4` | 0 | 43.060 |
| `MatMulV2_28` | 0 | 42.940 |
| `MatMulV2_148` | 0 | 42.640 |
| `MatMulV2_76` | 0 | 42.400 |
| `MatMulV2_100` | 0 | 42.360 |
| `MatMulV2_112` | 0 | 42.020 |
| `MatMulV2_124` | 0 | 41.980 |
| `MatMulV2_52` | 0 | 41.960 |
| `MatMulV2_88` | 0 | 41.760 |
| `MatMulV2_16` | 0 | 41.660 |
| `MatMulV2_154` | 0 | 41.520 |
| `MatMulV2_106` | 0 | 41.500 |
| `MatMulV2_28` | 0 | 41.440 |
| `MatMulV2_46` | 0 | 41.320 |
| `MatMulV2_64` | 0 | 41.320 |
| `MatMulV2_160` | 0 | 41.300 |
| `MatMulV2_88` | 0 | 41.200 |
| `MatMulV2_82` | 0 | 41.120 |
| `MatMulV2_118` | 0 | 41.080 |
| `MatMulV2_136` | 0 | 41.060 |
| `MatMulV2_142` | 0 | 41.060 |
| `MatMulV2_10` | 0 | 41.000 |
| `MatMulV2_28` | 0 | 41.000 |
| `MatMulV2_124` | 0 | 41.000 |
| `MatMulV2_40` | 0 | 40.960 |
| `MatMulV2_130` | 0 | 40.940 |
| `MatMulV2_94` | 0 | 40.860 |
| `MatMulV2_88` | 0 | 40.840 |
| `MatMulV2_58` | 0 | 40.780 |
| `MatMulV2_112` | 0 | 40.780 |
| `MatMulV2_124` | 0 | 40.740 |
| `MatMulV2_16` | 0 | 40.680 |
| `MatMulV2_40` | 0 | 40.580 |
| `MatMulV2_148` | 0 | 40.580 |
| `MatMulV2_70` | 0 | 40.520 |
| `MatMulV2_148` | 0 | 40.520 |
| `MatMulV2_46` | 0 | 40.500 |
| `MatMulV2_76` | 0 | 40.500 |
| `MatMulV2_40` | 0 | 40.480 |
| `MatMulV2_136` | 0 | 40.460 |
| `MatMulV2_94` | 0 | 40.440 |
| `MatMulV2_112` | 0 | 40.440 |
| `MatMulV2_136` | 0 | 40.440 |
| `MatMulV2_160` | 0 | 40.360 |
| `MatMulV2_52` | 0 | 40.340 |
| `MatMulV2_16` | 0 | 40.340 |
| `MatMulV2_82` | 0 | 40.320 |
| `MatMulV2_34` | 0 | 40.280 |
| `MatMulV2_100` | 0 | 40.280 |
| `MatMulV2_52` | 0 | 40.240 |
| `MatMulV2_100` | 0 | 40.240 |
| `MatMulV2_46` | 0 | 40.220 |
| `MatMulV2_70` | 0 | 40.140 |
| `MatMulV2_76` | 0 | 40.140 |
| `MatMulV2_130` | 0 | 40.100 |
| `MatMulV2_94` | 0 | 40.060 |
| `MatMulV2_34` | 0 | 40.040 |
| `MatMulV2_10` | 0 | 40.040 |
| `MatMulV2_58` | 0 | 40.020 |
| `MatMulV2_34` | 0 | 40.020 |
| `MatMulV2_22` | 0 | 40.000 |
| `MatMulV2_106` | 0 | 39.980 |
| `MatMulV2_22` | 0 | 39.980 |
| `MatMulV2_58` | 0 | 39.940 |
| `MatMulV2_118` | 0 | 39.920 |
| `MatMulV2_142` | 0 | 39.920 |
| `MatMulV2_10` | 0 | 39.900 |
| `MatMulV2_64` | 0 | 39.880 |
| `MatMulV2_142` | 0 | 39.860 |
| `MatMulV2_154` | 0 | 39.800 |
| `MatMulV2_22` | 0 | 39.740 |
| `MatMulV2_82` | 0 | 39.580 |
| `MatMulV2_106` | 0 | 39.580 |
| `MatMulV2_154` | 0 | 39.540 |
| `MatMulV2_160` | 0 | 39.480 |
| `MatMulV2_64` | 0 | 39.460 |
| `MatMulV2_118` | 0 | 39.460 |
| `MatMulV2_70` | 0 | 39.400 |
| `MatMulV2_130` | 0 | 39.020 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 18.820 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 18.620 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 18.500 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 17.400 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 17.120 |
| `MatMulV2` | 0 | 16.820 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 16.760 |
| `Gelu` | 0 | 16.520 |
| `Gelu` | 0 | 16.500 |
| `Gelu` | 0 | 16.500 |
| `MatMulV2` | 0 | 16.180 |
| `MatMulV2` | 0 | 16.060 |
| `LayerNormV4_LayerNormV3` | 0 | 15.580 |
| `MatMulV2_18` | 0 | 14.640 |
| `MatMulV2_97` | 0 | 14.460 |
| `MatMulV2_75` | 0 | 14.440 |
| `MatMulV2_27` | 0 | 14.420 |
| `MatMulV2_147` | 0 | 14.400 |
| `MatMulV2_51` | 0 | 14.360 |
| `MatMulV2_114` | 0 | 14.360 |
| `MatMulV2_42` | 0 | 14.240 |
| `MatMulV2_90` | 0 | 14.240 |
| `MatMulV2_12` | 0 | 14.220 |
| `MatMulV2_25` | 0 | 14.220 |
| `MatMulV2_102` | 0 | 14.220 |
| `MatMulV2_36` | 0 | 14.200 |
| `MatMulV2_145` | 0 | 14.180 |
| `MatMulV2_157` | 0 | 14.180 |
| `MatMulV2_30` | 0 | 14.160 |
| `MatMulV2_66` | 0 | 14.160 |
| `MatMulV2_123` | 0 | 14.160 |
| `MatMulV2_150` | 0 | 14.160 |
| `MatMulV2_126` | 0 | 14.140 |
| `MatMulV2_54` | 0 | 14.120 |
| `MatMulV2_85` | 0 | 14.120 |
| `MatMulV2_108` | 0 | 14.120 |
| `MatMulV2_61` | 0 | 14.080 |
| `MatMulV2_73` | 0 | 14.080 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 8556.360 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step1` | 1 | 7696.880 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step2` | 1 | 6668.370 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step3` | 1 | 6588.850 |
| `TorchDynamo Cache Lookup` | 3 | 5799.520 |
| `Torch-Compiled Region: 0/0` | 3 | 4077.430 |
| `TorchNpuGraphBase::Run` | 3 | 2856.210 |
| `RefreshAtTensorFromGeTensor` | 3 | 1149.340 |
| `ExecuteGraph` | 3 | 578.760 |
| `aten::empty` | 3 | 551.800 |
| `AssembleInputs` | 3 | 430.850 |
| `AssembleOutputs` | 3 | 284.930 |
| `aten::set_` | 3 | 284.530 |
| `empty_tensor` | 3 | 281.180 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 68738.320 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 14607.550 |
| `launch` | 274 | 3667.890 |
| `InputCopy` | 3 | 202.750 |
| `ModelExecute` | 3 | 83.450 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 38.740 |
| `step_info` | 6 | 30.330 |
| `OutputCopy` | 3 | 1.020 |

