# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s2048_i4352_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_compiled_profile_e298a0c/s2048_i4352_native/liteserver-c001-4_623159_20260729132347233_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `26681.240 us`
- `Free`: `2638.420 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3945.000 us`
- `Stage`: `29319.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 162 | 12228.680 |
| `MatMulV2` | 324 | 7570.580 |
| `Gelu` | 81 | 3103.720 |
| `AddLayerNorm` | 162 | 2718.920 |
| `AutomaticBufferFusionOp` | 81 | 953.000 |
| `LayerNormV3` | 3 | 93.400 |
| `Data` | 3 | 12.940 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_4_to_v3` | 3 | 256.520 |
| `MatMulV2_28_to_v3` | 3 | 249.340 |
| `MatMulV2_40_to_v3` | 3 | 246.680 |
| `MatMulV2_112_to_v3` | 3 | 246.320 |
| `MatMulV2_64_to_v3` | 3 | 246.180 |
| `MatMulV2_148_to_v3` | 3 | 245.260 |
| `MatMulV2_154_to_v3` | 3 | 245.260 |
| `MatMulV2_34_to_v3` | 3 | 245.180 |
| `MatMulV2_46_to_v3` | 3 | 244.920 |
| `MatMulV2_88_to_v3` | 3 | 244.840 |
| `MatMulV2_76_to_v3` | 3 | 244.820 |
| `MatMulV2_52_to_v3` | 3 | 244.760 |
| `MatMulV2_82_to_v3` | 3 | 244.600 |
| `MatMulV2_100_to_v3` | 3 | 244.460 |
| `MatMulV2_22_to_v3` | 3 | 244.460 |
| `MatMulV2_136_to_v3` | 3 | 244.400 |
| `MatMulV2_70_to_v3` | 3 | 244.180 |
| `MatMulV2_142_to_v3` | 3 | 244.160 |
| `MatMulV2_124_to_v3` | 3 | 244.100 |
| `MatMulV2_58_to_v3` | 3 | 243.920 |
| `MatMulV2_16_to_v3` | 3 | 243.860 |
| `MatMulV2_118_to_v3` | 3 | 243.820 |
| `MatMulV2_94_to_v3` | 3 | 243.120 |
| `MatMulV2_10_to_v3` | 3 | 242.740 |
| `MatMulV2_130_to_v3` | 3 | 242.740 |
| `MatMulV2_106_to_v3` | 3 | 242.720 |
| `MatMulV2_160_to_v3` | 3 | 242.720 |
| `MatMulV2_5_to_v3` | 3 | 219.620 |
| `MatMulV2_113_to_v3` | 3 | 210.680 |
| `MatMulV2_65_to_v3` | 3 | 209.520 |
| `MatMulV2_95_to_v3` | 3 | 209.420 |
| `MatMulV2_71_to_v3` | 3 | 208.720 |
| `MatMulV2_47_to_v3` | 3 | 208.640 |
| `MatMulV2_101_to_v3` | 3 | 208.240 |
| `MatMulV2_161_to_v3` | 3 | 208.060 |
| `MatMulV2_119_to_v3` | 3 | 207.940 |
| `MatMulV2_125_to_v3` | 3 | 207.900 |
| `MatMulV2_89_to_v3` | 3 | 207.880 |
| `MatMulV2_41_to_v3` | 3 | 207.840 |
| `MatMulV2_143_to_v3` | 3 | 207.780 |
| `MatMulV2_83_to_v3` | 3 | 207.520 |
| `MatMulV2_23_to_v3` | 3 | 207.420 |
| `MatMulV2_53_to_v3` | 3 | 207.300 |
| `MatMulV2_107_to_v3` | 3 | 207.240 |
| `MatMulV2_77_to_v3` | 3 | 207.120 |
| `MatMulV2_17_to_v3` | 3 | 207.080 |
| `MatMulV2_149_to_v3` | 3 | 206.600 |
| `MatMulV2_29_to_v3` | 3 | 206.500 |
| `MatMulV2_35_to_v3` | 3 | 206.180 |
| `MatMulV2_59_to_v3` | 3 | 206.100 |
| `MatMulV2_131_to_v3` | 3 | 205.780 |
| `MatMulV2_137_to_v3` | 3 | 205.520 |
| `MatMulV2_11_to_v3` | 3 | 205.260 |
| `MatMulV2_155_to_v3` | 3 | 204.740 |
| `Gelu_10` | 3 | 116.220 |
| `Gelu_15` | 3 | 115.860 |
| `Gelu_20` | 3 | 115.860 |
| `Gelu` | 3 | 115.800 |
| `Gelu_5` | 3 | 115.780 |
| `Gelu_23` | 3 | 114.820 |
| `Gelu_1` | 3 | 114.820 |
| `Gelu_4` | 3 | 114.820 |
| `Gelu_3` | 3 | 114.780 |
| `Gelu_13` | 3 | 114.780 |
| `Gelu_16` | 3 | 114.780 |
| `Gelu_21` | 3 | 114.780 |
| `Gelu_26` | 3 | 114.760 |
| `Gelu_25` | 3 | 114.760 |
| `Gelu_6` | 3 | 114.740 |
| `Gelu_18` | 3 | 114.740 |
| `Gelu_7` | 3 | 114.740 |
| `Gelu_14` | 3 | 114.740 |
| `Gelu_24` | 3 | 114.720 |
| `Gelu_9` | 3 | 114.720 |
| `Gelu_8` | 3 | 114.700 |
| `Gelu_2` | 3 | 114.680 |
| `Gelu_11` | 3 | 114.680 |
| `Gelu_19` | 3 | 114.680 |
| `Gelu_17` | 3 | 114.680 |
| `Gelu_22` | 3 | 114.660 |
| `Gelu_12` | 3 | 114.620 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 108.340 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 99.120 |
| `LayerNormV4_LayerNormV3` | 3 | 93.400 |
| `MatMulV2` | 3 | 86.680 |
| `MatMulV2_66` | 3 | 71.640 |
| `MatMulV2_25` | 3 | 71.260 |
| `MatMulV2_7` | 3 | 71.240 |
| `MatMulV2_12` | 3 | 71.200 |
| `MatMulV2_61` | 3 | 71.200 |
| `MatMulV2_27` | 3 | 71.160 |
| `MatMulV2_90` | 3 | 70.980 |
| `MatMulV2_67` | 3 | 70.960 |
| `MatMulV2_37` | 3 | 70.920 |
| `MatMulV2_108` | 3 | 70.920 |
| `MatMulV2_84` | 3 | 70.860 |
| `MatMulV2_6` | 3 | 70.820 |
| `MatMulV2_60` | 3 | 70.820 |
| `MatMulV2_102` | 3 | 70.780 |
| `MatMulV2_114` | 3 | 70.780 |
| `MatMulV2_36` | 3 | 70.660 |
| `MatMulV2_85` | 3 | 70.660 |
| `MatMulV2_109` | 3 | 70.660 |
| `MatMulV2_73` | 3 | 70.640 |
| `MatMulV2_42` | 3 | 70.600 |
| `MatMulV2_75` | 3 | 70.600 |
| `MatMulV2_150` | 3 | 70.600 |
| `MatMulV2_30` | 3 | 70.580 |
| `MatMulV2_18` | 3 | 70.560 |
| `MatMulV2_1` | 3 | 70.540 |
| `MatMulV2_49` | 3 | 70.520 |
| `MatMulV2_97` | 3 | 70.520 |
| `MatMulV2_99` | 3 | 70.520 |
| `MatMulV2_54` | 3 | 70.480 |
| `MatMulV2_96` | 3 | 70.480 |
| `MatMulV2_13` | 3 | 70.360 |
| `MatMulV2_48` | 3 | 70.360 |
| `MatMulV2_63` | 3 | 70.320 |
| `MatMulV2_24` | 3 | 70.300 |
| `MatMulV2_43` | 3 | 70.300 |
| `MatMulV2_50` | 3 | 70.240 |
| `MatMulV2_51` | 3 | 70.220 |
| `MatMulV2_2` | 3 | 70.180 |
| `MatMulV2_116` | 3 | 70.140 |
| `MatMulV2_91` | 3 | 70.120 |
| `MatMulV2_8` | 3 | 70.100 |
| `MatMulV2_72` | 3 | 70.040 |
| `MatMulV2_92` | 3 | 70.020 |
| `MatMulV2_38` | 3 | 70.000 |
| `MatMulV2_78` | 3 | 69.920 |
| `MatMulV2_39` | 3 | 69.900 |
| `MatMulV2_156` | 3 | 69.900 |
| `MatMulV2_56` | 3 | 69.820 |
| `MatMulV2_128` | 3 | 69.800 |
| `MatMulV2_55` | 3 | 69.760 |
| `MatMulV2_115` | 3 | 69.760 |
| `MatMulV2_45` | 3 | 69.740 |
| `MatMulV2_111` | 3 | 69.740 |
| `MatMulV2_122` | 3 | 69.720 |
| `MatMulV2_146` | 3 | 69.720 |
| `MatMulV2_20` | 3 | 69.700 |
| `MatMulV2_127` | 3 | 69.700 |
| `MatMulV2_159` | 3 | 69.700 |
| `MatMulV2_123` | 3 | 69.680 |
| `MatMulV2_74` | 3 | 69.660 |
| `MatMulV2_103` | 3 | 69.660 |
| `MatMulV2_129` | 3 | 69.660 |
| `MatMulV2_132` | 3 | 69.660 |
| `MatMulV2_62` | 3 | 69.640 |
| `MatMulV2_79` | 3 | 69.640 |
| `MatMulV2_81` | 3 | 69.640 |
| `MatMulV2_120` | 3 | 69.640 |
| `MatMulV2_105` | 3 | 69.600 |
| `MatMulV2_138` | 3 | 69.560 |
| `MatMulV2_93` | 3 | 69.540 |
| `MatMulV2_32` | 3 | 69.540 |
| `MatMulV2_126` | 3 | 69.520 |
| `MatMulV2_144` | 3 | 69.520 |
| `MatMulV2_44` | 3 | 69.500 |
| `MatMulV2_87` | 3 | 69.500 |
| `MatMulV2_3` | 3 | 69.480 |
| `MatMulV2_145` | 3 | 69.480 |
| `MatMulV2_104` | 3 | 69.460 |
| `MatMulV2_151` | 3 | 69.460 |
| `MatMulV2_33` | 3 | 69.440 |
| `MatMulV2_152` | 3 | 69.440 |
| `MatMulV2_153` | 3 | 69.440 |
| `MatMulV2_68` | 3 | 69.380 |
| `MatMulV2_19` | 3 | 69.380 |
| `MatMulV2_26` | 3 | 69.380 |
| `MatMulV2_110` | 3 | 69.380 |
| `MatMulV2_21` | 3 | 69.360 |
| `MatMulV2_134` | 3 | 69.360 |
| `MatMulV2_158` | 3 | 69.360 |
| `MatMulV2_157` | 3 | 69.340 |
| `MatMulV2_98` | 3 | 69.280 |
| `MatMulV2_139` | 3 | 69.280 |
| `MatMulV2_15` | 3 | 69.260 |
| `MatMulV2_133` | 3 | 69.260 |
| `MatMulV2_121` | 3 | 69.240 |
| `MatMulV2_14` | 3 | 69.240 |
| `MatMulV2_117` | 3 | 69.220 |
| `MatMulV2_86` | 3 | 69.200 |
| `MatMulV2_147` | 3 | 69.200 |
| `MatMulV2_135` | 3 | 69.180 |
| `MatMulV2_57` | 3 | 69.160 |
| `MatMulV2_69` | 3 | 69.140 |
| `MatMulV2_140` | 3 | 69.120 |
| `MatMulV2_31` | 3 | 69.100 |
| `MatMulV2_80` | 3 | 69.060 |
| `MatMulV2_9` | 3 | 69.020 |
| `MatMulV2_141` | 3 | 68.900 |
| `LayerNormV4_16_LayerNormV3/AddLayerNorm` | 3 | 52.260 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 51.740 |
| `LayerNormV4_40_LayerNormV3/AddLayerNorm` | 3 | 51.520 |
| `LayerNormV4_14_LayerNormV3/AddLayerNorm` | 3 | 51.380 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 51.300 |
| `LayerNormV4_42_LayerNormV3/AddLayerNorm` | 3 | 51.180 |
| `LayerNormV4_26_LayerNormV3/AddLayerNorm` | 3 | 50.240 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 50.000 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7570.580 |
| `MatMulV3 | "2048,1152;4352,1152;4352" -> "2048,4352" | ND;ND;ND -> ND` | 81 | 6616.080 |
| `MatMulV3 | "2048,4352;1152,4352;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 5612.600 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3103.720 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2718.920 |
| `AutomaticBufferFusionOp | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,1152" | ND;ND;ND -> ND` | 81 | 953.000 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 93.400 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 12.940 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 20845.660 |
| `ND` | 81 | 3103.720 |
| `ND;ND;ND;ND` | 162 | 2718.920 |
| `N/A` | 3 | 12.940 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_4_to_v3` | 0 | 85.780 |
| `MatMulV2_4_to_v3` | 0 | 85.660 |
| `MatMulV2_4_to_v3` | 0 | 85.080 |
| `MatMulV2_28_to_v3` | 0 | 83.920 |
| `MatMulV2_34_to_v3` | 0 | 83.720 |
| `MatMulV2_40_to_v3` | 0 | 83.580 |
| `MatMulV2_64_to_v3` | 0 | 83.560 |
| `MatMulV2_112_to_v3` | 0 | 83.340 |
| `MatMulV2_28_to_v3` | 0 | 82.900 |
| `MatMulV2_76_to_v3` | 0 | 82.800 |
| `MatMulV2_136_to_v3` | 0 | 82.680 |
| `MatMulV2_154_to_v3` | 0 | 82.660 |
| `MatMulV2_28_to_v3` | 0 | 82.520 |
| `MatMulV2_124_to_v3` | 0 | 82.420 |
| `MatMulV2_148_to_v3` | 0 | 82.140 |
| `MatMulV2_46_to_v3` | 0 | 82.120 |
| `MatMulV2_94_to_v3` | 0 | 82.100 |
| `MatMulV2_88_to_v3` | 0 | 82.080 |
| `MatMulV2_16_to_v3` | 0 | 82.000 |
| `MatMulV2_142_to_v3` | 0 | 82.000 |
| `MatMulV2_148_to_v3` | 0 | 81.980 |
| `MatMulV2_100_to_v3` | 0 | 81.920 |
| `MatMulV2_52_to_v3` | 0 | 81.900 |
| `MatMulV2_88_to_v3` | 0 | 81.900 |
| `MatMulV2_112_to_v3` | 0 | 81.900 |
| `MatMulV2_118_to_v3` | 0 | 81.900 |
| `MatMulV2_82_to_v3` | 0 | 81.860 |
| `MatMulV2_46_to_v3` | 0 | 81.800 |
| `MatMulV2_22_to_v3` | 0 | 81.780 |
| `MatMulV2_70_to_v3` | 0 | 81.740 |
| `MatMulV2_40_to_v3` | 0 | 81.720 |
| `MatMulV2_82_to_v3` | 0 | 81.700 |
| `MatMulV2_58_to_v3` | 0 | 81.660 |
| `MatMulV2_142_to_v3` | 0 | 81.600 |
| `MatMulV2_64_to_v3` | 0 | 81.580 |
| `MatMulV2_154_to_v3` | 0 | 81.520 |
| `MatMulV2_52_to_v3` | 0 | 81.500 |
| `MatMulV2_22_to_v3` | 0 | 81.460 |
| `MatMulV2_16_to_v3` | 0 | 81.420 |
| `MatMulV2_100_to_v3` | 0 | 81.420 |
| `MatMulV2_40_to_v3` | 0 | 81.380 |
| `MatMulV2_52_to_v3` | 0 | 81.360 |
| `MatMulV2_70_to_v3` | 0 | 81.340 |
| `MatMulV2_22_to_v3` | 0 | 81.220 |
| `MatMulV2_76_to_v3` | 0 | 81.220 |
| `MatMulV2_58_to_v3` | 0 | 81.200 |
| `MatMulV2_106_to_v3` | 0 | 81.180 |
| `MatMulV2_160_to_v3` | 0 | 81.180 |
| `MatMulV2_148_to_v3` | 0 | 81.140 |
| `MatMulV2_100_to_v3` | 0 | 81.120 |
| `MatMulV2_106_to_v3` | 0 | 81.120 |
| `MatMulV2_124_to_v3` | 0 | 81.100 |
| `MatMulV2_70_to_v3` | 0 | 81.100 |
| `MatMulV2_118_to_v3` | 0 | 81.100 |
| `MatMulV2_10_to_v3` | 0 | 81.080 |
| `MatMulV2_112_to_v3` | 0 | 81.080 |
| `MatMulV2_154_to_v3` | 0 | 81.080 |
| `MatMulV2_58_to_v3` | 0 | 81.060 |
| `MatMulV2_82_to_v3` | 0 | 81.040 |
| `MatMulV2_64_to_v3` | 0 | 81.040 |
| `MatMulV2_160_to_v3` | 0 | 81.040 |
| `MatMulV2_130_to_v3` | 0 | 81.020 |
| `MatMulV2_46_to_v3` | 0 | 81.000 |
| `MatMulV2_136_to_v3` | 0 | 80.980 |
| `MatMulV2_130_to_v3` | 0 | 80.960 |
| `MatMulV2_10_to_v3` | 0 | 80.900 |
| `MatMulV2_88_to_v3` | 0 | 80.860 |
| `MatMulV2_34_to_v3` | 0 | 80.840 |
| `MatMulV2_118_to_v3` | 0 | 80.820 |
| `MatMulV2_76_to_v3` | 0 | 80.800 |
| `MatMulV2_130_to_v3` | 0 | 80.760 |
| `MatMulV2_10_to_v3` | 0 | 80.760 |
| `MatMulV2_136_to_v3` | 0 | 80.740 |
| `MatMulV2_94_to_v3` | 0 | 80.680 |
| `MatMulV2_34_to_v3` | 0 | 80.620 |
| `MatMulV2_124_to_v3` | 0 | 80.580 |
| `MatMulV2_142_to_v3` | 0 | 80.560 |
| `MatMulV2_160_to_v3` | 0 | 80.500 |
| `MatMulV2_16_to_v3` | 0 | 80.440 |
| `MatMulV2_106_to_v3` | 0 | 80.420 |
| `MatMulV2_94_to_v3` | 0 | 80.340 |
| `MatMulV2_5_to_v3` | 0 | 73.880 |
| `MatMulV2_5_to_v3` | 0 | 72.920 |
| `MatMulV2_5_to_v3` | 0 | 72.820 |
| `MatMulV2_113_to_v3` | 0 | 71.100 |
| `MatMulV2_65_to_v3` | 0 | 71.060 |
| `MatMulV2_101_to_v3` | 0 | 70.540 |
| `MatMulV2_71_to_v3` | 0 | 70.380 |
| `MatMulV2_113_to_v3` | 0 | 70.200 |
| `MatMulV2_95_to_v3` | 0 | 70.080 |
| `MatMulV2_143_to_v3` | 0 | 69.920 |
| `MatMulV2_83_to_v3` | 0 | 69.900 |
| `MatMulV2_47_to_v3` | 0 | 69.840 |
| `MatMulV2_95_to_v3` | 0 | 69.840 |
| `MatMulV2_53_to_v3` | 0 | 69.820 |
| `MatMulV2_125_to_v3` | 0 | 69.740 |
| `MatMulV2_89_to_v3` | 0 | 69.720 |
| `MatMulV2_119_to_v3` | 0 | 69.700 |
| `MatMulV2_35_to_v3` | 0 | 69.680 |
| `MatMulV2_41_to_v3` | 0 | 69.660 |
| `MatMulV2_161_to_v3` | 0 | 69.640 |
| `MatMulV2_47_to_v3` | 0 | 69.520 |
| `MatMulV2_95_to_v3` | 0 | 69.500 |
| `MatMulV2_41_to_v3` | 0 | 69.480 |
| `MatMulV2_23_to_v3` | 0 | 69.400 |
| `MatMulV2_113_to_v3` | 0 | 69.380 |
| `MatMulV2_161_to_v3` | 0 | 69.380 |
| `MatMulV2_107_to_v3` | 0 | 69.360 |
| `MatMulV2_65_to_v3` | 0 | 69.320 |
| `MatMulV2_125_to_v3` | 0 | 69.320 |
| `MatMulV2_17_to_v3` | 0 | 69.280 |
| `MatMulV2_47_to_v3` | 0 | 69.280 |
| `MatMulV2_71_to_v3` | 0 | 69.260 |
| `MatMulV2_77_to_v3` | 0 | 69.240 |
| `MatMulV2_119_to_v3` | 0 | 69.220 |
| `MatMulV2_11_to_v3` | 0 | 69.200 |
| `MatMulV2_107_to_v3` | 0 | 69.200 |
| `MatMulV2_149_to_v3` | 0 | 69.180 |
| `MatMulV2_89_to_v3` | 0 | 69.160 |
| `MatMulV2_29_to_v3` | 0 | 69.160 |
| `MatMulV2_83_to_v3` | 0 | 69.160 |
| `MatMulV2_59_to_v3` | 0 | 69.140 |
| `MatMulV2_65_to_v3` | 0 | 69.140 |
| `MatMulV2_137_to_v3` | 0 | 69.080 |
| `MatMulV2_71_to_v3` | 0 | 69.080 |
| `MatMulV2_23_to_v3` | 0 | 69.080 |
| `MatMulV2_77_to_v3` | 0 | 69.060 |
| `MatMulV2_161_to_v3` | 0 | 69.040 |
| `MatMulV2_17_to_v3` | 0 | 69.040 |
| `MatMulV2_131_to_v3` | 0 | 69.040 |
| `MatMulV2_29_to_v3` | 0 | 69.020 |
| `MatMulV2_119_to_v3` | 0 | 69.020 |
| `MatMulV2_89_to_v3` | 0 | 69.000 |
| `MatMulV2_101_to_v3` | 0 | 69.000 |
| `MatMulV2_143_to_v3` | 0 | 68.980 |
| `MatMulV2_23_to_v3` | 0 | 68.940 |
| `MatMulV2_53_to_v3` | 0 | 68.920 |
| `MatMulV2_143_to_v3` | 0 | 68.880 |
| `MatMulV2_125_to_v3` | 0 | 68.840 |
| `MatMulV2_77_to_v3` | 0 | 68.820 |
| `MatMulV2_149_to_v3` | 0 | 68.780 |
| `MatMulV2_17_to_v3` | 0 | 68.760 |
| `MatMulV2_59_to_v3` | 0 | 68.760 |
| `MatMulV2_101_to_v3` | 0 | 68.700 |
| `MatMulV2_41_to_v3` | 0 | 68.700 |
| `MatMulV2_107_to_v3` | 0 | 68.680 |
| `MatMulV2_149_to_v3` | 0 | 68.640 |
| `MatMulV2_155_to_v3` | 0 | 68.620 |
| `MatMulV2_53_to_v3` | 0 | 68.560 |
| `MatMulV2_83_to_v3` | 0 | 68.460 |
| `MatMulV2_131_to_v3` | 0 | 68.400 |
| `MatMulV2_35_to_v3` | 0 | 68.360 |
| `MatMulV2_137_to_v3` | 0 | 68.360 |
| `MatMulV2_131_to_v3` | 0 | 68.340 |
| `MatMulV2_11_to_v3` | 0 | 68.340 |
| `MatMulV2_29_to_v3` | 0 | 68.320 |
| `MatMulV2_155_to_v3` | 0 | 68.220 |
| `MatMulV2_59_to_v3` | 0 | 68.200 |
| `MatMulV2_35_to_v3` | 0 | 68.140 |
| `MatMulV2_137_to_v3` | 0 | 68.080 |
| `MatMulV2_155_to_v3` | 0 | 67.900 |
| `MatMulV2_11_to_v3` | 0 | 67.720 |
| `Gelu_10` | 0 | 38.880 |
| `Gelu_10` | 0 | 38.740 |
| `Gelu` | 0 | 38.680 |
| `Gelu_20` | 0 | 38.660 |
| `Gelu_15` | 0 | 38.660 |
| `Gelu_20` | 0 | 38.660 |
| `Gelu_15` | 0 | 38.620 |
| `Gelu_5` | 0 | 38.620 |
| `Gelu` | 0 | 38.600 |
| `Gelu_10` | 0 | 38.600 |
| `Gelu_5` | 0 | 38.580 |
| `Gelu_5` | 0 | 38.580 |
| `Gelu_15` | 0 | 38.580 |
| `Gelu_20` | 0 | 38.540 |
| `Gelu` | 0 | 38.520 |
| `Gelu_23` | 0 | 38.400 |
| `Gelu_1` | 0 | 38.360 |
| `Gelu_4` | 0 | 38.360 |
| `Gelu_3` | 0 | 38.340 |
| `Gelu_13` | 0 | 38.340 |
| `Gelu_16` | 0 | 38.340 |
| `Gelu_21` | 0 | 38.340 |
| `Gelu_26` | 0 | 38.340 |
| `Gelu_6` | 0 | 38.320 |
| `Gelu_8` | 0 | 38.320 |
| `Gelu_14` | 0 | 38.320 |
| `Gelu_18` | 0 | 38.320 |
| `Gelu_24` | 0 | 38.320 |
| `Gelu_7` | 0 | 38.300 |
| `Gelu_9` | 0 | 38.300 |
| `Gelu_25` | 0 | 38.300 |
| `Gelu_2` | 0 | 38.280 |
| `Gelu_19` | 0 | 38.280 |
| `Gelu_12` | 0 | 38.260 |
| `Gelu_17` | 0 | 38.260 |
| `Gelu_1` | 0 | 38.240 |
| `Gelu_4` | 0 | 38.240 |
| `Gelu_25` | 0 | 38.240 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 11806.170 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step1` | 1 | 11017.270 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step2` | 1 | 9925.490 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step3` | 1 | 9921.090 |
| `TorchDynamo Cache Lookup` | 3 | 9068.980 |
| `Torch-Compiled Region: 0/0` | 3 | 4109.010 |
| `TorchNpuGraphBase::Run` | 3 | 2885.690 |
| `RefreshAtTensorFromGeTensor` | 3 | 1151.010 |
| `ExecuteGraph` | 3 | 593.750 |
| `aten::empty` | 3 | 543.320 |
| `AssembleInputs` | 3 | 437.870 |
| `aten::set_` | 3 | 308.600 |
| `AssembleOutputs` | 3 | 295.830 |
| `empty_tensor` | 3 | 276.710 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 66735.450 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 24444.390 |
| `launch` | 274 | 3582.310 |
| `InputCopy` | 3 | 225.890 |
| `ModelExecute` | 3 | 63.770 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 50.490 |
| `step_info` | 6 | 34.960 |
| `OutputCopy` | 3 | 1.120 |

