# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s512_i4352_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s512_i4352_native/liteserver-c001-4_618680_20260729132043395_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `12096.640 us`
- `Free`: `2557.380 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3184.000 us`
- `Stage`: `14654.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 9517.940 |
| `AddLayerNorm` | 162 | 1369.820 |
| `Gelu` | 81 | 844.860 |
| `AutomaticBufferFusionOp` | 81 | 305.020 |
| `LayerNormV3` | 3 | 46.340 |
| `Data` | 3 | 12.660 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_4` | 3 | 109.940 |
| `MatMulV2_76` | 3 | 105.620 |
| `MatMulV2_112` | 3 | 105.320 |
| `MatMulV2_88` | 3 | 105.300 |
| `MatMulV2_100` | 3 | 105.160 |
| `MatMulV2_28` | 3 | 104.700 |
| `MatMulV2_16` | 3 | 104.480 |
| `MatMulV2_124` | 3 | 104.400 |
| `MatMulV2_40` | 3 | 104.320 |
| `MatMulV2_148` | 3 | 103.760 |
| `MatMulV2_52` | 3 | 103.420 |
| `MatMulV2_106` | 3 | 103.320 |
| `MatMulV2_154` | 3 | 103.100 |
| `MatMulV2_10` | 3 | 102.800 |
| `MatMulV2_64` | 3 | 102.740 |
| `MatMulV2_58` | 3 | 102.700 |
| `MatMulV2_136` | 3 | 102.560 |
| `MatMulV2_46` | 3 | 102.440 |
| `MatMulV2_94` | 3 | 102.140 |
| `MatMulV2_82` | 3 | 102.100 |
| `MatMulV2_160` | 3 | 101.900 |
| `MatMulV2_70` | 3 | 101.840 |
| `MatMulV2_130` | 3 | 101.780 |
| `MatMulV2_34` | 3 | 101.540 |
| `MatMulV2_22` | 3 | 101.480 |
| `MatMulV2_142` | 3 | 101.180 |
| `MatMulV2_118` | 3 | 100.920 |
| `MatMulV2_5` | 3 | 93.640 |
| `MatMulV2_107` | 3 | 90.560 |
| `MatMulV2_23` | 3 | 89.600 |
| `MatMulV2_155` | 3 | 89.500 |
| `MatMulV2_83` | 3 | 89.400 |
| `MatMulV2_131` | 3 | 89.280 |
| `MatMulV2_95` | 3 | 89.240 |
| `MatMulV2_59` | 3 | 89.140 |
| `MatMulV2_71` | 3 | 89.140 |
| `MatMulV2_11` | 3 | 89.120 |
| `MatMulV2_35` | 3 | 88.940 |
| `MatMulV2_47` | 3 | 88.500 |
| `MatMulV2_119` | 3 | 87.800 |
| `MatMulV2_41` | 3 | 87.680 |
| `MatMulV2_125` | 3 | 87.580 |
| `MatMulV2_77` | 3 | 87.500 |
| `MatMulV2_65` | 3 | 87.380 |
| `MatMulV2_17` | 3 | 86.980 |
| `MatMulV2_101` | 3 | 86.920 |
| `MatMulV2_143` | 3 | 86.900 |
| `MatMulV2_149` | 3 | 86.900 |
| `MatMulV2_113` | 3 | 86.800 |
| `MatMulV2_29` | 3 | 86.540 |
| `MatMulV2_89` | 3 | 86.400 |
| `MatMulV2_161` | 3 | 86.160 |
| `MatMulV2_137` | 3 | 86.140 |
| `MatMulV2_53` | 3 | 85.860 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 54.660 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 54.160 |
| `MatMulV2` | 3 | 48.020 |
| `Gelu` | 3 | 47.840 |
| `LayerNormV4_LayerNormV3` | 3 | 46.340 |
| `MatMulV2_97` | 3 | 41.740 |
| `MatMulV2_147` | 3 | 41.480 |
| `MatMulV2_78` | 3 | 41.420 |
| `MatMulV2_123` | 3 | 41.380 |
| `MatMulV2_121` | 3 | 41.360 |
| `MatMulV2_102` | 3 | 41.320 |
| `MatMulV2_156` | 3 | 41.240 |
| `MatMulV2_96` | 3 | 41.200 |
| `MatMulV2_85` | 3 | 41.120 |
| `MatMulV2_157` | 3 | 41.080 |
| `MatMulV2_6` | 3 | 41.000 |
| `MatMulV2_144` | 3 | 40.940 |
| `MatMulV2_12` | 3 | 40.900 |
| `MatMulV2_24` | 3 | 40.900 |
| `MatMulV2_75` | 3 | 40.900 |
| `MatMulV2_120` | 3 | 40.900 |
| `MatMulV2_25` | 3 | 40.840 |
| `MatMulV2_60` | 3 | 40.840 |
| `MatMulV2_66` | 3 | 40.840 |
| `MatMulV2_150` | 3 | 40.840 |
| `MatMulV2_13` | 3 | 40.780 |
| `MatMulV2_19` | 3 | 40.780 |
| `MatMulV2_42` | 3 | 40.780 |
| `MatMulV2_61` | 3 | 40.760 |
| `MatMulV2_48` | 3 | 40.760 |
| `MatMulV2_1` | 3 | 40.720 |
| `MatMulV2_51` | 3 | 40.720 |
| `MatMulV2_133` | 3 | 40.700 |
| `MatMulV2_18` | 3 | 40.700 |
| `MatMulV2_43` | 3 | 40.680 |
| `MatMulV2_37` | 3 | 40.660 |
| `MatMulV2_90` | 3 | 40.660 |
| `MatMulV2_132` | 3 | 40.620 |
| `MatMulV2_145` | 3 | 40.620 |
| `MatMulV2_67` | 3 | 40.480 |
| `MatMulV2_73` | 3 | 40.460 |
| `MatMulV2_99` | 3 | 40.440 |
| `MatMulV2_138` | 3 | 40.440 |
| `MatMulV2_27` | 3 | 40.420 |
| `MatMulV2_20` | 3 | 40.400 |
| `MatMulV2_91` | 3 | 40.400 |
| `MatMulV2_30` | 3 | 40.360 |
| `MatMulV2_72` | 3 | 40.300 |
| `MatMulV2_151` | 3 | 40.300 |
| `MatMulV2_26` | 3 | 40.260 |
| `MatMulV2_33` | 3 | 40.260 |
| `MatMulV2_49` | 3 | 40.220 |
| `MatMulV2_57` | 3 | 40.220 |
| `MatMulV2_84` | 3 | 40.220 |
| `MatMulV2_54` | 3 | 40.200 |
| `MatMulV2_104` | 3 | 40.200 |
| `MatMulV2_111` | 3 | 40.200 |
| `MatMulV2_126` | 3 | 40.200 |
| `MatMulV2_109` | 3 | 40.180 |
| `MatMulV2_9` | 3 | 40.080 |
| `MatMulV2_31` | 3 | 40.080 |
| `MatMulV2_45` | 3 | 40.060 |
| `MatMulV2_108` | 3 | 40.060 |
| `MatMulV2_153` | 3 | 40.060 |
| `MatMulV2_86` | 3 | 40.040 |
| `MatMulV2_159` | 3 | 40.040 |
| `MatMulV2_7` | 3 | 40.040 |
| `MatMulV2_98` | 3 | 40.000 |
| `MatMulV2_68` | 3 | 39.940 |
| `MatMulV2_32` | 3 | 39.940 |
| `MatMulV2_74` | 3 | 39.920 |
| `MatMulV2_128` | 3 | 39.880 |
| `MatMulV2_21` | 3 | 39.860 |
| `MatMulV2_105` | 3 | 39.860 |
| `MatMulV2_114` | 3 | 39.860 |
| `MatMulV2_115` | 3 | 39.840 |
| `MatMulV2_92` | 3 | 39.840 |
| `MatMulV2_127` | 3 | 39.820 |
| `MatMulV2_146` | 3 | 39.820 |
| `MatMulV2_2` | 3 | 39.800 |
| `MatMulV2_103` | 3 | 39.780 |
| `MatMulV2_140` | 3 | 39.780 |
| `MatMulV2_36` | 3 | 39.760 |
| `MatMulV2_44` | 3 | 39.760 |
| `MatMulV2_3` | 3 | 39.720 |
| `MatMulV2_55` | 3 | 39.700 |
| `MatMulV2_38` | 3 | 39.680 |
| `MatMulV2_8` | 3 | 39.640 |
| `MatMulV2_79` | 3 | 39.640 |
| `MatMulV2_87` | 3 | 39.620 |
| `MatMulV2_56` | 3 | 39.560 |
| `MatMulV2_80` | 3 | 39.560 |
| `MatMulV2_135` | 3 | 39.560 |
| `MatMulV2_116` | 3 | 39.480 |
| `MatMulV2_134` | 3 | 39.480 |
| `MatMulV2_129` | 3 | 39.460 |
| `MatMulV2_141` | 3 | 39.460 |
| `MatMulV2_69` | 3 | 39.440 |
| `MatMulV2_110` | 3 | 39.440 |
| `MatMulV2_122` | 3 | 39.440 |
| `MatMulV2_81` | 3 | 39.420 |
| `MatMulV2_152` | 3 | 39.420 |
| `MatMulV2_158` | 3 | 39.420 |
| `MatMulV2_139` | 3 | 39.360 |
| `MatMulV2_14` | 3 | 39.320 |
| `MatMulV2_50` | 3 | 39.260 |
| `MatMulV2_117` | 3 | 39.220 |
| `MatMulV2_93` | 3 | 39.140 |
| `MatMulV2_39` | 3 | 39.080 |
| `MatMulV2_63` | 3 | 39.080 |
| `MatMulV2_62` | 3 | 38.800 |
| `MatMulV2_15` | 3 | 38.700 |
| `Gelu_6` | 3 | 31.840 |
| `Gelu_10` | 3 | 31.280 |
| `Gelu_5` | 3 | 30.940 |
| `Gelu_26` | 3 | 30.900 |
| `Gelu_9` | 3 | 30.720 |
| `Gelu_1` | 3 | 30.680 |
| `Gelu_4` | 3 | 30.680 |
| `Gelu_3` | 3 | 30.660 |
| `Gelu_7` | 3 | 30.620 |
| `Gelu_8` | 3 | 30.620 |
| `Gelu_13` | 3 | 30.620 |
| `Gelu_2` | 3 | 30.600 |
| `Gelu_21` | 3 | 30.600 |
| `Gelu_11` | 3 | 30.580 |
| `Gelu_22` | 3 | 30.560 |
| `Gelu_24` | 3 | 30.540 |
| `Gelu_16` | 3 | 30.540 |
| `Gelu_25` | 3 | 30.500 |
| `Gelu_14` | 3 | 30.480 |
| `Gelu_23` | 3 | 30.480 |
| `Gelu_19` | 3 | 30.480 |
| `Gelu_12` | 3 | 30.440 |
| `Gelu_17` | 3 | 30.440 |
| `Gelu_18` | 3 | 30.420 |
| `Gelu_20` | 3 | 30.420 |
| `Gelu_15` | 3 | 30.380 |
| `AddAdd_1Muls` | 3 | 29.820 |
| `LayerNormV4_48_LayerNormV3/AddLayerNorm` | 3 | 27.560 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 27.160 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 26.620 |
| `LayerNormV4_8_LayerNormV3/AddLayerNorm` | 3 | 26.400 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 26.000 |
| `LayerNormV4_46_LayerNormV3/AddLayerNorm` | 3 | 25.960 |
| `LayerNormV4_22_LayerNormV3/AddLayerNorm` | 3 | 25.880 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "512,1152;1152,1152;1152" -> "512,1152" | ND;ND;ND -> ND` | 324 | 4347.380 |
| `MatMulV2 | "512,1152;4352,1152;4352" -> "512,4352" | ND;ND;ND -> ND` | 81 | 2790.960 |
| `MatMulV2 | "512,4352;1152,4352;1152" -> "512,1152" | ND;ND;ND -> ND` | 81 | 2379.600 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 1369.820 |
| `Gelu | "1,512,4352" -> "1,512,4352" | ND -> ND` | 81 | 844.860 |
| `AutomaticBufferFusionOp | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,1152" | ND;ND;ND -> ND` | 81 | 305.020 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 46.340 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 12.660 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 9869.300 |
| `ND;ND;ND;ND` | 162 | 1369.820 |
| `ND` | 81 | 844.860 |
| `N/A` | 3 | 12.660 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_4` | 0 | 37.680 |
| `MatMulV2_88` | 0 | 36.380 |
| `MatMulV2_76` | 0 | 36.280 |
| `MatMulV2_100` | 0 | 36.240 |
| `MatMulV2_4` | 0 | 36.200 |
| `MatMulV2_4` | 0 | 36.060 |
| `MatMulV2_28` | 0 | 35.900 |
| `MatMulV2_124` | 0 | 35.740 |
| `MatMulV2_10` | 0 | 35.660 |
| `MatMulV2_16` | 0 | 35.560 |
| `MatMulV2_112` | 0 | 35.540 |
| `MatMulV2_154` | 0 | 35.480 |
| `MatMulV2_148` | 0 | 35.340 |
| `MatMulV2_46` | 0 | 35.220 |
| `MatMulV2_58` | 0 | 35.180 |
| `MatMulV2_40` | 0 | 35.000 |
| `MatMulV2_70` | 0 | 34.960 |
| `MatMulV2_142` | 0 | 34.940 |
| `MatMulV2_22` | 0 | 34.940 |
| `MatMulV2_112` | 0 | 34.940 |
| `MatMulV2_52` | 0 | 34.920 |
| `MatMulV2_34` | 0 | 34.900 |
| `MatMulV2_100` | 0 | 34.880 |
| `MatMulV2_106` | 0 | 34.860 |
| `MatMulV2_94` | 0 | 34.840 |
| `MatMulV2_112` | 0 | 34.840 |
| `MatMulV2_40` | 0 | 34.820 |
| `MatMulV2_76` | 0 | 34.760 |
| `MatMulV2_82` | 0 | 34.740 |
| `MatMulV2_124` | 0 | 34.680 |
| `MatMulV2_16` | 0 | 34.580 |
| `MatMulV2_106` | 0 | 34.580 |
| `MatMulV2_76` | 0 | 34.580 |
| `MatMulV2_136` | 0 | 34.560 |
| `MatMulV2_88` | 0 | 34.560 |
| `MatMulV2_160` | 0 | 34.520 |
| `MatMulV2_28` | 0 | 34.500 |
| `MatMulV2_40` | 0 | 34.500 |
| `MatMulV2_64` | 0 | 34.420 |
| `MatMulV2_130` | 0 | 34.380 |
| `MatMulV2_88` | 0 | 34.360 |
| `MatMulV2_64` | 0 | 34.340 |
| `MatMulV2_16` | 0 | 34.340 |
| `MatMulV2_28` | 0 | 34.300 |
| `MatMulV2_52` | 0 | 34.280 |
| `MatMulV2_148` | 0 | 34.220 |
| `MatMulV2_52` | 0 | 34.220 |
| `MatMulV2_148` | 0 | 34.200 |
| `MatMulV2_136` | 0 | 34.160 |
| `MatMulV2_118` | 0 | 34.120 |
| `MatMulV2_130` | 0 | 34.080 |
| `MatMulV2_100` | 0 | 34.040 |
| `MatMulV2_46` | 0 | 33.980 |
| `MatMulV2_64` | 0 | 33.980 |
| `MatMulV2_124` | 0 | 33.980 |
| `MatMulV2_70` | 0 | 33.940 |
| `MatMulV2_58` | 0 | 33.920 |
| `MatMulV2_106` | 0 | 33.880 |
| `MatMulV2_136` | 0 | 33.840 |
| `MatMulV2_82` | 0 | 33.820 |
| `MatMulV2_154` | 0 | 33.820 |
| `MatMulV2_154` | 0 | 33.800 |
| `MatMulV2_160` | 0 | 33.760 |
| `MatMulV2_10` | 0 | 33.700 |
| `MatMulV2_94` | 0 | 33.680 |
| `MatMulV2_160` | 0 | 33.620 |
| `MatMulV2_94` | 0 | 33.620 |
| `MatMulV2_58` | 0 | 33.600 |
| `MatMulV2_118` | 0 | 33.580 |
| `MatMulV2_82` | 0 | 33.540 |
| `MatMulV2_142` | 0 | 33.500 |
| `MatMulV2_10` | 0 | 33.440 |
| `MatMulV2_34` | 0 | 33.380 |
| `MatMulV2_130` | 0 | 33.320 |
| `MatMulV2_22` | 0 | 33.300 |
| `MatMulV2_34` | 0 | 33.260 |
| `MatMulV2_22` | 0 | 33.240 |
| `MatMulV2_46` | 0 | 33.240 |
| `MatMulV2_118` | 0 | 33.220 |
| `MatMulV2_70` | 0 | 32.940 |
| `MatMulV2_142` | 0 | 32.740 |
| `MatMulV2_5` | 0 | 32.220 |
| `MatMulV2_107` | 0 | 32.100 |
| `MatMulV2_11` | 0 | 31.040 |
| `MatMulV2_131` | 0 | 30.960 |
| `MatMulV2_23` | 0 | 30.900 |
| `MatMulV2_71` | 0 | 30.880 |
| `MatMulV2_5` | 0 | 30.840 |
| `MatMulV2_125` | 0 | 30.740 |
| `MatMulV2_155` | 0 | 30.700 |
| `MatMulV2_47` | 0 | 30.660 |
| `MatMulV2_41` | 0 | 30.620 |
| `MatMulV2_59` | 0 | 30.580 |
| `MatMulV2_5` | 0 | 30.580 |
| `MatMulV2_95` | 0 | 30.340 |
| `MatMulV2_35` | 0 | 30.320 |
| `MatMulV2_83` | 0 | 30.280 |
| `MatMulV2_77` | 0 | 30.260 |
| `MatMulV2_65` | 0 | 30.240 |
| `MatMulV2_89` | 0 | 29.940 |
| `MatMulV2_17` | 0 | 29.840 |
| `MatMulV2_119` | 0 | 29.760 |
| `MatMulV2_95` | 0 | 29.660 |
| `MatMulV2_83` | 0 | 29.620 |
| `MatMulV2_155` | 0 | 29.600 |
| `MatMulV2_101` | 0 | 29.560 |
| `MatMulV2_149` | 0 | 29.560 |
| `MatMulV2_113` | 0 | 29.540 |
| `MatMulV2_83` | 0 | 29.500 |
| `MatMulV2_107` | 0 | 29.420 |
| `MatMulV2_23` | 0 | 29.380 |
| `MatMulV2_35` | 0 | 29.380 |
| `MatMulV2_161` | 0 | 29.320 |
| `MatMulV2_23` | 0 | 29.320 |
| `MatMulV2_29` | 0 | 29.300 |
| `MatMulV2_59` | 0 | 29.300 |
| `MatMulV2_59` | 0 | 29.260 |
| `MatMulV2_95` | 0 | 29.240 |
| `MatMulV2_35` | 0 | 29.240 |
| `MatMulV2_131` | 0 | 29.240 |
| `MatMulV2_155` | 0 | 29.200 |
| `MatMulV2_71` | 0 | 29.140 |
| `MatMulV2_71` | 0 | 29.120 |
| `MatMulV2_11` | 0 | 29.100 |
| `MatMulV2_119` | 0 | 29.100 |
| `MatMulV2_143` | 0 | 29.080 |
| `MatMulV2_131` | 0 | 29.080 |
| `MatMulV2_137` | 0 | 29.060 |
| `MatMulV2_107` | 0 | 29.040 |
| `MatMulV2_143` | 0 | 29.020 |
| `MatMulV2_47` | 0 | 29.020 |
| `MatMulV2_53` | 0 | 28.980 |
| `MatMulV2_11` | 0 | 28.980 |
| `MatMulV2_119` | 0 | 28.940 |
| `MatMulV2_149` | 0 | 28.940 |
| `MatMulV2_47` | 0 | 28.820 |
| `MatMulV2_143` | 0 | 28.800 |
| `MatMulV2_101` | 0 | 28.760 |
| `MatMulV2_137` | 0 | 28.760 |
| `MatMulV2_113` | 0 | 28.740 |
| `MatMulV2_77` | 0 | 28.700 |
| `MatMulV2_53` | 0 | 28.680 |
| `MatMulV2_161` | 0 | 28.680 |
| `MatMulV2_41` | 0 | 28.660 |
| `MatMulV2_125` | 0 | 28.660 |
| `MatMulV2_65` | 0 | 28.640 |
| `MatMulV2_29` | 0 | 28.640 |
| `MatMulV2_29` | 0 | 28.600 |
| `MatMulV2_101` | 0 | 28.600 |
| `MatMulV2_17` | 0 | 28.580 |
| `MatMulV2_17` | 0 | 28.560 |
| `MatMulV2_77` | 0 | 28.540 |
| `MatMulV2_113` | 0 | 28.520 |
| `MatMulV2_65` | 0 | 28.500 |
| `MatMulV2_89` | 0 | 28.480 |
| `MatMulV2_41` | 0 | 28.400 |
| `MatMulV2_149` | 0 | 28.400 |
| `MatMulV2_137` | 0 | 28.320 |
| `MatMulV2_53` | 0 | 28.200 |
| `MatMulV2_125` | 0 | 28.180 |
| `MatMulV2_161` | 0 | 28.160 |
| `MatMulV2_89` | 0 | 27.980 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 18.480 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 18.480 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 18.320 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 18.180 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 0 | 18.160 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 0 | 17.200 |
| `MatMulV2` | 0 | 16.380 |
| `MatMulV2` | 0 | 16.180 |
| `Gelu` | 0 | 15.980 |
| `Gelu` | 0 | 15.940 |
| `Gelu` | 0 | 15.920 |
| `LayerNormV4_LayerNormV3` | 0 | 15.900 |
| `MatMulV2` | 0 | 15.460 |
| `LayerNormV4_LayerNormV3` | 0 | 15.380 |
| `LayerNormV4_LayerNormV3` | 0 | 15.060 |
| `MatMulV2_147` | 0 | 14.900 |
| `MatMulV2_121` | 0 | 14.840 |
| `MatMulV2_102` | 0 | 14.740 |
| `MatMulV2_48` | 0 | 14.680 |
| `MatMulV2_97` | 0 | 14.680 |
| `MatMulV2_51` | 0 | 14.620 |
| `MatMulV2_78` | 0 | 14.580 |
| `MatMulV2_123` | 0 | 14.580 |
| `MatMulV2_60` | 0 | 14.560 |
| `MatMulV2_43` | 0 | 14.540 |
| `MatMulV2_90` | 0 | 14.500 |
| `MatMulV2_25` | 0 | 14.480 |
| `MatMulV2_85` | 0 | 14.460 |
| `MatMulV2_157` | 0 | 14.460 |
| `MatMulV2_13` | 0 | 14.460 |
| `MatMulV2_75` | 0 | 14.420 |
| `MatMulV2_96` | 0 | 14.400 |
| `MatMulV2_144` | 0 | 14.400 |
| `MatMulV2_145` | 0 | 14.400 |
| `MatMulV2_133` | 0 | 14.360 |
| `MatMulV2_150` | 0 | 14.360 |
| `MatMulV2_54` | 0 | 14.340 |
| `MatMulV2_73` | 0 | 14.340 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 6881.610 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step1` | 1 | 5426.100 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step3` | 1 | 5045.310 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step2` | 1 | 5040.450 |
| `TorchDynamo Cache Lookup` | 3 | 4219.680 |
| `Torch-Compiled Region: 0/0` | 3 | 3469.890 |
| `TorchNpuGraphBase::Run` | 3 | 2557.790 |
| `RefreshAtTensorFromGeTensor` | 3 | 1134.280 |
| `aten::empty` | 3 | 530.590 |
| `ExecuteGraph` | 3 | 419.680 |
| `AssembleInputs` | 3 | 383.520 |
| `aten::set_` | 3 | 293.270 |
| `AssembleOutputs` | 3 | 266.680 |
| `empty_tensor` | 3 | 266.480 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 68698.450 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 10067.380 |
| `launch` | 274 | 3451.040 |
| `InputCopy` | 3 | 114.920 |
| `ModelExecute` | 3 | 40.180 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 20.710 |
| `step_info` | 6 | 13.540 |
| `OutputCopy` | 3 | 0.980 |

