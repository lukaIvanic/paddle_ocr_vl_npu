# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s2048_i4304_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s2048_i4304_native/liteserver-c001-4_621073_20260729132208840_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `30425.120 us`
- `Free`: `2846.220 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3520.500 us`
- `Stage`: `33271.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 162 | 15399.580 |
| `MatMulV2` | 324 | 7641.260 |
| `AddLayerNorm` | 162 | 3221.720 |
| `Gelu` | 81 | 3140.060 |
| `AutomaticBufferFusionOp` | 81 | 916.200 |
| `LayerNormV3` | 3 | 93.200 |
| `Data` | 3 | 13.100 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_5_to_v3` | 3 | 303.460 |
| `MatMulV2_71_to_v3` | 3 | 293.840 |
| `MatMulV2_4_to_v3` | 3 | 292.980 |
| `MatMulV2_11_to_v3` | 3 | 289.220 |
| `MatMulV2_23_to_v3` | 3 | 286.920 |
| `MatMulV2_106_to_v3` | 3 | 286.880 |
| `MatMulV2_35_to_v3` | 3 | 286.480 |
| `MatMulV2_142_to_v3` | 3 | 286.380 |
| `MatMulV2_29_to_v3` | 3 | 286.320 |
| `MatMulV2_137_to_v3` | 3 | 286.160 |
| `MatMulV2_131_to_v3` | 3 | 286.040 |
| `MatMulV2_136_to_v3` | 3 | 285.960 |
| `MatMulV2_95_to_v3` | 3 | 285.780 |
| `MatMulV2_59_to_v3` | 3 | 285.700 |
| `MatMulV2_119_to_v3` | 3 | 285.660 |
| `MatMulV2_101_to_v3` | 3 | 285.620 |
| `MatMulV2_83_to_v3` | 3 | 285.560 |
| `MatMulV2_113_to_v3` | 3 | 285.380 |
| `MatMulV2_70_to_v3` | 3 | 285.340 |
| `MatMulV2_89_to_v3` | 3 | 285.240 |
| `MatMulV2_28_to_v3` | 3 | 285.160 |
| `MatMulV2_125_to_v3` | 3 | 285.060 |
| `MatMulV2_53_to_v3` | 3 | 285.020 |
| `MatMulV2_22_to_v3` | 3 | 284.980 |
| `MatMulV2_40_to_v3` | 3 | 284.980 |
| `MatMulV2_161_to_v3` | 3 | 284.920 |
| `MatMulV2_65_to_v3` | 3 | 284.860 |
| `MatMulV2_149_to_v3` | 3 | 284.840 |
| `MatMulV2_155_to_v3` | 3 | 284.680 |
| `MatMulV2_107_to_v3` | 3 | 284.640 |
| `MatMulV2_88_to_v3` | 3 | 284.560 |
| `MatMulV2_17_to_v3` | 3 | 284.480 |
| `MatMulV2_10_to_v3` | 3 | 284.300 |
| `MatMulV2_41_to_v3` | 3 | 284.220 |
| `MatMulV2_143_to_v3` | 3 | 284.120 |
| `MatMulV2_94_to_v3` | 3 | 283.920 |
| `MatMulV2_34_to_v3` | 3 | 283.880 |
| `MatMulV2_100_to_v3` | 3 | 283.820 |
| `MatMulV2_46_to_v3` | 3 | 283.800 |
| `MatMulV2_52_to_v3` | 3 | 283.800 |
| `MatMulV2_154_to_v3` | 3 | 283.800 |
| `MatMulV2_47_to_v3` | 3 | 283.740 |
| `MatMulV2_77_to_v3` | 3 | 283.220 |
| `MatMulV2_148_to_v3` | 3 | 283.080 |
| `MatMulV2_76_to_v3` | 3 | 283.020 |
| `MatMulV2_58_to_v3` | 3 | 282.600 |
| `MatMulV2_130_to_v3` | 3 | 282.280 |
| `MatMulV2_160_to_v3` | 3 | 282.240 |
| `MatMulV2_112_to_v3` | 3 | 282.120 |
| `MatMulV2_16_to_v3` | 3 | 281.940 |
| `MatMulV2_64_to_v3` | 3 | 281.920 |
| `MatMulV2_124_to_v3` | 3 | 281.920 |
| `MatMulV2_118_to_v3` | 3 | 281.500 |
| `MatMulV2_82_to_v3` | 3 | 281.240 |
| `Gelu_16` | 3 | 117.300 |
| `Gelu_4` | 3 | 117.000 |
| `Gelu_26` | 3 | 116.920 |
| `Gelu_7` | 3 | 116.920 |
| `Gelu_21` | 3 | 116.880 |
| `Gelu_11` | 3 | 116.780 |
| `Gelu` | 3 | 116.240 |
| `Gelu_6` | 3 | 116.200 |
| `Gelu_8` | 3 | 116.180 |
| `Gelu_2` | 3 | 116.160 |
| `Gelu_9` | 3 | 116.160 |
| `Gelu_10` | 3 | 116.160 |
| `Gelu_23` | 3 | 116.140 |
| `Gelu_13` | 3 | 116.120 |
| `Gelu_22` | 3 | 116.120 |
| `Gelu_15` | 3 | 116.100 |
| `Gelu_20` | 3 | 116.100 |
| `Gelu_24` | 3 | 116.100 |
| `Gelu_1` | 3 | 116.100 |
| `Gelu_12` | 3 | 116.080 |
| `Gelu_18` | 3 | 116.080 |
| `Gelu_5` | 3 | 116.060 |
| `Gelu_19` | 3 | 116.060 |
| `Gelu_3` | 3 | 116.040 |
| `Gelu_25` | 3 | 116.040 |
| `Gelu_14` | 3 | 116.040 |
| `Gelu_17` | 3 | 115.980 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 106.320 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 98.940 |
| `LayerNormV4_LayerNormV3` | 3 | 93.200 |
| `MatMulV2` | 3 | 85.780 |
| `MatMulV2_114` | 3 | 73.540 |
| `MatMulV2_102` | 3 | 72.940 |
| `MatMulV2_56` | 3 | 72.920 |
| `MatMulV2_108` | 3 | 72.760 |
| `MatMulV2_37` | 3 | 72.660 |
| `MatMulV2_54` | 3 | 72.620 |
| `MatMulV2_60` | 3 | 72.620 |
| `MatMulV2_66` | 3 | 72.500 |
| `MatMulV2_43` | 3 | 72.460 |
| `MatMulV2_116` | 3 | 72.440 |
| `MatMulV2_62` | 3 | 72.360 |
| `MatMulV2_68` | 3 | 72.260 |
| `MatMulV2_110` | 3 | 72.240 |
| `MatMulV2_104` | 3 | 72.180 |
| `MatMulV2_81` | 3 | 72.160 |
| `MatMulV2_6` | 3 | 72.080 |
| `MatMulV2_20` | 3 | 72.080 |
| `MatMulV2_33` | 3 | 72.000 |
| `MatMulV2_39` | 3 | 71.980 |
| `MatMulV2_127` | 3 | 71.980 |
| `MatMulV2_31` | 3 | 71.960 |
| `MatMulV2_14` | 3 | 71.940 |
| `MatMulV2_91` | 3 | 71.940 |
| `MatMulV2_103` | 3 | 71.860 |
| `MatMulV2_8` | 3 | 71.840 |
| `MatMulV2_93` | 3 | 71.820 |
| `MatMulV2_150` | 3 | 71.680 |
| `MatMulV2_12` | 3 | 71.640 |
| `MatMulV2_45` | 3 | 71.620 |
| `MatMulV2_139` | 3 | 71.600 |
| `MatMulV2_156` | 3 | 71.540 |
| `MatMulV2_135` | 3 | 71.420 |
| `MatMulV2_158` | 3 | 71.420 |
| `MatMulV2_85` | 3 | 71.380 |
| `MatMulV2_87` | 3 | 71.340 |
| `MatMulV2_141` | 3 | 71.240 |
| `MatMulV2_79` | 3 | 71.200 |
| `MatMulV2_99` | 3 | 71.080 |
| `MatMulV2_123` | 3 | 71.000 |
| `MatMulV2_133` | 3 | 70.960 |
| `MatMulV2_84` | 3 | 70.940 |
| `MatMulV2_42` | 3 | 70.660 |
| `MatMulV2_51` | 3 | 70.520 |
| `MatMulV2_1` | 3 | 70.520 |
| `MatMulV2_18` | 3 | 70.460 |
| `MatMulV2_48` | 3 | 70.440 |
| `MatMulV2_27` | 3 | 70.420 |
| `MatMulV2_111` | 3 | 70.420 |
| `MatMulV2_55` | 3 | 70.380 |
| `MatMulV2_122` | 3 | 70.340 |
| `MatMulV2_86` | 3 | 70.300 |
| `MatMulV2_90` | 3 | 70.260 |
| `MatMulV2_109` | 3 | 70.220 |
| `MatMulV2_96` | 3 | 70.200 |
| `MatMulV2_159` | 3 | 70.200 |
| `MatMulV2_7` | 3 | 70.200 |
| `MatMulV2_67` | 3 | 70.180 |
| `MatMulV2_75` | 3 | 70.160 |
| `MatMulV2_36` | 3 | 70.120 |
| `MatMulV2_50` | 3 | 70.080 |
| `MatMulV2_69` | 3 | 70.080 |
| `MatMulV2_2` | 3 | 70.060 |
| `LayerNormV4_12_LayerNormV3/AddLayerNorm` | 3 | 70.060 |
| `MatMulV2_63` | 3 | 70.060 |
| `MatMulV2_26` | 3 | 70.020 |
| `MatMulV2_30` | 3 | 70.000 |
| `MatMulV2_57` | 3 | 69.960 |
| `MatMulV2_126` | 3 | 69.960 |
| `MatMulV2_25` | 3 | 69.940 |
| `MatMulV2_78` | 3 | 69.900 |
| `MatMulV2_98` | 3 | 69.880 |
| `MatMulV2_132` | 3 | 69.860 |
| `MatMulV2_32` | 3 | 69.840 |
| `MatMulV2_105` | 3 | 69.760 |
| `MatMulV2_152` | 3 | 69.760 |
| `MatMulV2_13` | 3 | 69.740 |
| `MatMulV2_74` | 3 | 69.740 |
| `MatMulV2_72` | 3 | 69.720 |
| `MatMulV2_128` | 3 | 69.720 |
| `LayerNormV4_20_LayerNormV3/AddLayerNorm` | 3 | 69.680 |
| `MatMulV2_44` | 3 | 69.660 |
| `MatMulV2_19` | 3 | 69.640 |
| `MatMulV2_24` | 3 | 69.640 |
| `MatMulV2_38` | 3 | 69.620 |
| `MatMulV2_117` | 3 | 69.600 |
| `MatMulV2_115` | 3 | 69.580 |
| `MatMulV2_121` | 3 | 69.560 |
| `MatMulV2_21` | 3 | 69.540 |
| `MatMulV2_120` | 3 | 69.520 |
| `MatMulV2_151` | 3 | 69.440 |
| `MatMulV2_9` | 3 | 69.420 |
| `MatMulV2_80` | 3 | 69.420 |
| `MatMulV2_61` | 3 | 69.400 |
| `MatMulV2_97` | 3 | 69.400 |
| `MatMulV2_92` | 3 | 69.360 |
| `MatMulV2_140` | 3 | 69.360 |
| `MatMulV2_144` | 3 | 69.320 |
| `MatMulV2_49` | 3 | 69.280 |
| `MatMulV2_153` | 3 | 69.260 |
| `MatMulV2_157` | 3 | 69.260 |
| `MatMulV2_3` | 3 | 69.240 |
| `MatMulV2_147` | 3 | 69.220 |
| `MatMulV2_138` | 3 | 69.200 |
| `MatMulV2_146` | 3 | 69.180 |
| `MatMulV2_145` | 3 | 69.160 |
| `MatMulV2_129` | 3 | 69.120 |
| `LayerNormV4_38_LayerNormV3/AddLayerNorm` | 3 | 69.100 |
| `MatMulV2_15` | 3 | 69.020 |
| `MatMulV2_73` | 3 | 68.980 |
| `LayerNormV4_40_LayerNormV3/AddLayerNorm` | 3 | 68.900 |
| `LayerNormV4_18_LayerNormV3/AddLayerNorm` | 3 | 68.820 |
| `MatMulV2_134` | 3 | 68.800 |
| `LayerNormV4_16_LayerNormV3/AddLayerNorm` | 3 | 68.600 |
| `LayerNormV4_2_LayerNormV3/AddLayerNorm` | 3 | 68.400 |
| `LayerNormV4_42_LayerNormV3/AddLayerNorm` | 3 | 68.400 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV3 | "2048,4304;1152,4304;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 7731.180 |
| `MatMulV3 | "2048,1152;4304,1152;4304" -> "2048,4304" | ND;ND;ND -> ND` | 81 | 7668.400 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7641.260 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 3221.720 |
| `Gelu | "1,2048,4304" -> "1,2048,4304" | ND -> ND` | 81 | 3140.060 |
| `AutomaticBufferFusionOp | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,1152" | ND;ND;ND -> ND` | 81 | 916.200 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 93.200 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 13.100 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 24050.240 |
| `ND;ND;ND;ND` | 162 | 3221.720 |
| `ND` | 81 | 3140.060 |
| `N/A` | 3 | 13.100 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_5_to_v3` | 0 | 101.980 |
| `MatMulV2_5_to_v3` | 0 | 100.880 |
| `MatMulV2_5_to_v3` | 0 | 100.600 |
| `MatMulV2_71_to_v3` | 0 | 99.020 |
| `MatMulV2_4_to_v3` | 0 | 98.480 |
| `MatMulV2_71_to_v3` | 0 | 97.460 |
| `MatMulV2_71_to_v3` | 0 | 97.360 |
| `MatMulV2_4_to_v3` | 0 | 97.320 |
| `MatMulV2_4_to_v3` | 0 | 97.180 |
| `MatMulV2_11_to_v3` | 0 | 96.860 |
| `MatMulV2_11_to_v3` | 0 | 96.420 |
| `MatMulV2_137_to_v3` | 0 | 96.320 |
| `MatMulV2_23_to_v3` | 0 | 96.320 |
| `MatMulV2_106_to_v3` | 0 | 96.280 |
| `MatMulV2_35_to_v3` | 0 | 96.140 |
| `MatMulV2_136_to_v3` | 0 | 95.960 |
| `MatMulV2_11_to_v3` | 0 | 95.940 |
| `MatMulV2_59_to_v3` | 0 | 95.860 |
| `MatMulV2_131_to_v3` | 0 | 95.820 |
| `MatMulV2_142_to_v3` | 0 | 95.820 |
| `MatMulV2_29_to_v3` | 0 | 95.740 |
| `MatMulV2_107_to_v3` | 0 | 95.600 |
| `MatMulV2_70_to_v3` | 0 | 95.580 |
| `MatMulV2_29_to_v3` | 0 | 95.560 |
| `MatMulV2_125_to_v3` | 0 | 95.540 |
| `MatMulV2_28_to_v3` | 0 | 95.540 |
| `MatMulV2_35_to_v3` | 0 | 95.520 |
| `MatMulV2_155_to_v3` | 0 | 95.520 |
| `MatMulV2_119_to_v3` | 0 | 95.460 |
| `MatMulV2_95_to_v3` | 0 | 95.440 |
| `MatMulV2_106_to_v3` | 0 | 95.440 |
| `MatMulV2_22_to_v3` | 0 | 95.420 |
| `MatMulV2_83_to_v3` | 0 | 95.400 |
| `MatMulV2_89_to_v3` | 0 | 95.400 |
| `MatMulV2_107_to_v3` | 0 | 95.360 |
| `MatMulV2_23_to_v3` | 0 | 95.340 |
| `MatMulV2_83_to_v3` | 0 | 95.320 |
| `MatMulV2_53_to_v3` | 0 | 95.320 |
| `MatMulV2_142_to_v3` | 0 | 95.300 |
| `MatMulV2_161_to_v3` | 0 | 95.300 |
| `MatMulV2_17_to_v3` | 0 | 95.300 |
| `MatMulV2_131_to_v3` | 0 | 95.280 |
| `MatMulV2_101_to_v3` | 0 | 95.280 |
| `MatMulV2_23_to_v3` | 0 | 95.260 |
| `MatMulV2_113_to_v3` | 0 | 95.260 |
| `MatMulV2_119_to_v3` | 0 | 95.260 |
| `MatMulV2_142_to_v3` | 0 | 95.260 |
| `MatMulV2_149_to_v3` | 0 | 95.260 |
| `MatMulV2_101_to_v3` | 0 | 95.240 |
| `MatMulV2_41_to_v3` | 0 | 95.200 |
| `MatMulV2_95_to_v3` | 0 | 95.180 |
| `MatMulV2_10_to_v3` | 0 | 95.180 |
| `MatMulV2_136_to_v3` | 0 | 95.180 |
| `MatMulV2_59_to_v3` | 0 | 95.160 |
| `MatMulV2_95_to_v3` | 0 | 95.160 |
| `MatMulV2_106_to_v3` | 0 | 95.160 |
| `MatMulV2_137_to_v3` | 0 | 95.160 |
| `MatMulV2_40_to_v3` | 0 | 95.140 |
| `MatMulV2_113_to_v3` | 0 | 95.140 |
| `MatMulV2_65_to_v3` | 0 | 95.120 |
| `MatMulV2_65_to_v3` | 0 | 95.120 |
| `MatMulV2_101_to_v3` | 0 | 95.100 |
| `MatMulV2_40_to_v3` | 0 | 95.100 |
| `MatMulV2_28_to_v3` | 0 | 95.080 |
| `MatMulV2_143_to_v3` | 0 | 95.060 |
| `MatMulV2_77_to_v3` | 0 | 95.040 |
| `MatMulV2_88_to_v3` | 0 | 95.040 |
| `MatMulV2_29_to_v3` | 0 | 95.020 |
| `MatMulV2_89_to_v3` | 0 | 95.000 |
| `MatMulV2_113_to_v3` | 0 | 94.980 |
| `MatMulV2_88_to_v3` | 0 | 94.960 |
| `MatMulV2_119_to_v3` | 0 | 94.940 |
| `MatMulV2_131_to_v3` | 0 | 94.940 |
| `MatMulV2_34_to_v3` | 0 | 94.920 |
| `MatMulV2_53_to_v3` | 0 | 94.920 |
| `MatMulV2_161_to_v3` | 0 | 94.920 |
| `MatMulV2_76_to_v3` | 0 | 94.900 |
| `MatMulV2_47_to_v3` | 0 | 94.900 |
| `MatMulV2_70_to_v3` | 0 | 94.900 |
| `MatMulV2_100_to_v3` | 0 | 94.880 |
| `MatMulV2_70_to_v3` | 0 | 94.860 |
| `MatMulV2_154_to_v3` | 0 | 94.860 |
| `MatMulV2_47_to_v3` | 0 | 94.840 |
| `MatMulV2_89_to_v3` | 0 | 94.840 |
| `MatMulV2_22_to_v3` | 0 | 94.840 |
| `MatMulV2_83_to_v3` | 0 | 94.840 |
| `MatMulV2_77_to_v3` | 0 | 94.840 |
| `MatMulV2_136_to_v3` | 0 | 94.820 |
| `MatMulV2_35_to_v3` | 0 | 94.820 |
| `MatMulV2_46_to_v3` | 0 | 94.820 |
| `MatMulV2_17_to_v3` | 0 | 94.800 |
| `MatMulV2_41_to_v3` | 0 | 94.800 |
| `MatMulV2_149_to_v3` | 0 | 94.800 |
| `MatMulV2_154_to_v3` | 0 | 94.800 |
| `MatMulV2_53_to_v3` | 0 | 94.780 |
| `MatMulV2_125_to_v3` | 0 | 94.780 |
| `MatMulV2_149_to_v3` | 0 | 94.780 |
| `MatMulV2_34_to_v3` | 0 | 94.760 |
| `MatMulV2_52_to_v3` | 0 | 94.760 |
| `MatMulV2_155_to_v3` | 0 | 94.760 |
| `MatMulV2_40_to_v3` | 0 | 94.740 |
| `MatMulV2_94_to_v3` | 0 | 94.740 |
| `MatMulV2_125_to_v3` | 0 | 94.740 |
| `MatMulV2_10_to_v3` | 0 | 94.720 |
| `MatMulV2_22_to_v3` | 0 | 94.720 |
| `MatMulV2_143_to_v3` | 0 | 94.720 |
| `MatMulV2_161_to_v3` | 0 | 94.700 |
| `MatMulV2_137_to_v3` | 0 | 94.680 |
| `MatMulV2_59_to_v3` | 0 | 94.680 |
| `MatMulV2_94_to_v3` | 0 | 94.660 |
| `MatMulV2_65_to_v3` | 0 | 94.620 |
| `MatMulV2_88_to_v3` | 0 | 94.560 |
| `MatMulV2_100_to_v3` | 0 | 94.560 |
| `MatMulV2_46_to_v3` | 0 | 94.540 |
| `MatMulV2_28_to_v3` | 0 | 94.540 |
| `MatMulV2_52_to_v3` | 0 | 94.520 |
| `MatMulV2_94_to_v3` | 0 | 94.520 |
| `MatMulV2_52_to_v3` | 0 | 94.520 |
| `MatMulV2_46_to_v3` | 0 | 94.440 |
| `MatMulV2_148_to_v3` | 0 | 94.440 |
| `MatMulV2_155_to_v3` | 0 | 94.400 |
| `MatMulV2_10_to_v3` | 0 | 94.400 |
| `MatMulV2_16_to_v3` | 0 | 94.400 |
| `MatMulV2_148_to_v3` | 0 | 94.400 |
| `MatMulV2_58_to_v3` | 0 | 94.380 |
| `MatMulV2_17_to_v3` | 0 | 94.380 |
| `MatMulV2_100_to_v3` | 0 | 94.380 |
| `MatMulV2_143_to_v3` | 0 | 94.340 |
| `MatMulV2_160_to_v3` | 0 | 94.340 |
| `MatMulV2_130_to_v3` | 0 | 94.300 |
| `MatMulV2_58_to_v3` | 0 | 94.280 |
| `MatMulV2_148_to_v3` | 0 | 94.240 |
| `MatMulV2_112_to_v3` | 0 | 94.240 |
| `MatMulV2_41_to_v3` | 0 | 94.220 |
| `MatMulV2_34_to_v3` | 0 | 94.200 |
| `MatMulV2_76_to_v3` | 0 | 94.180 |
| `MatMulV2_64_to_v3` | 0 | 94.160 |
| `MatMulV2_124_to_v3` | 0 | 94.160 |
| `MatMulV2_154_to_v3` | 0 | 94.140 |
| `MatMulV2_16_to_v3` | 0 | 94.140 |
| `MatMulV2_112_to_v3` | 0 | 94.040 |
| `MatMulV2_130_to_v3` | 0 | 94.020 |
| `MatMulV2_47_to_v3` | 0 | 94.000 |
| `MatMulV2_160_to_v3` | 0 | 94.000 |
| `MatMulV2_82_to_v3` | 0 | 93.980 |
| `MatMulV2_130_to_v3` | 0 | 93.960 |
| `MatMulV2_118_to_v3` | 0 | 93.960 |
| `MatMulV2_58_to_v3` | 0 | 93.940 |
| `MatMulV2_76_to_v3` | 0 | 93.940 |
| `MatMulV2_160_to_v3` | 0 | 93.900 |
| `MatMulV2_64_to_v3` | 0 | 93.900 |
| `MatMulV2_118_to_v3` | 0 | 93.900 |
| `MatMulV2_124_to_v3` | 0 | 93.900 |
| `MatMulV2_124_to_v3` | 0 | 93.860 |
| `MatMulV2_64_to_v3` | 0 | 93.860 |
| `MatMulV2_112_to_v3` | 0 | 93.840 |
| `MatMulV2_82_to_v3` | 0 | 93.740 |
| `MatMulV2_107_to_v3` | 0 | 93.680 |
| `MatMulV2_118_to_v3` | 0 | 93.640 |
| `MatMulV2_82_to_v3` | 0 | 93.520 |
| `MatMulV2_16_to_v3` | 0 | 93.400 |
| `MatMulV2_77_to_v3` | 0 | 93.340 |
| `Gelu_16` | 0 | 39.260 |
| `Gelu_26` | 0 | 39.060 |
| `Gelu_7` | 0 | 39.040 |
| `Gelu_16` | 0 | 39.020 |
| `Gelu_21` | 0 | 39.020 |
| `Gelu_16` | 0 | 39.020 |
| `Gelu_4` | 0 | 39.000 |
| `Gelu_4` | 0 | 39.000 |
| `Gelu_4` | 0 | 39.000 |
| `Gelu_21` | 0 | 38.960 |
| `Gelu_7` | 0 | 38.940 |
| `Gelu_26` | 0 | 38.940 |
| `Gelu_7` | 0 | 38.940 |
| `Gelu_11` | 0 | 38.940 |
| `Gelu_11` | 0 | 38.920 |
| `Gelu_11` | 0 | 38.920 |
| `Gelu_26` | 0 | 38.920 |
| `Gelu_21` | 0 | 38.900 |
| `Gelu` | 0 | 38.840 |
| `Gelu_2` | 0 | 38.800 |
| `Gelu_9` | 0 | 38.780 |
| `Gelu_19` | 0 | 38.760 |
| `Gelu_1` | 0 | 38.740 |
| `Gelu_2` | 0 | 38.740 |
| `Gelu_3` | 0 | 38.740 |
| `Gelu_6` | 0 | 38.740 |
| `Gelu_8` | 0 | 38.740 |
| `Gelu_23` | 0 | 38.740 |
| `Gelu_6` | 0 | 38.740 |
| `Gelu_10` | 0 | 38.740 |
| `Gelu_15` | 0 | 38.720 |
| `Gelu_20` | 0 | 38.720 |
| `Gelu_22` | 0 | 38.720 |
| `Gelu_6` | 0 | 38.720 |
| `Gelu_8` | 0 | 38.720 |
| `Gelu_10` | 0 | 38.720 |
| `Gelu_13` | 0 | 38.720 |
| `Gelu_8` | 0 | 38.720 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 13030.350 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step1` | 1 | 11868.110 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step2` | 1 | 11265.270 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step3` | 1 | 11233.250 |
| `TorchDynamo Cache Lookup` | 3 | 10289.430 |
| `Torch-Compiled Region: 0/0` | 3 | 3793.410 |
| `TorchNpuGraphBase::Run` | 3 | 2776.400 |
| `RefreshAtTensorFromGeTensor` | 3 | 1152.000 |
| `aten::empty` | 3 | 544.590 |
| `ExecuteGraph` | 3 | 542.870 |
| `AssembleInputs` | 3 | 381.550 |
| `aten::set_` | 3 | 302.290 |
| `AssembleOutputs` | 3 | 298.320 |
| `empty_tensor` | 3 | 278.520 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 68245.230 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 28485.750 |
| `launch` | 274 | 5058.770 |
| `InputCopy` | 3 | 190.720 |
| `ModelExecute` | 3 | 60.850 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 37.030 |
| `step_info` | 6 | 25.800 |
| `OutputCopy` | 3 | 1.430 |

