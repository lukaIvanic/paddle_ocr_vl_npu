# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/head80_weight_compiled_b356d13`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/head80_weight_compiled_b356d13/liteserver-c001-4_647103_20260729144150255_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `77239.440 us`
- `Free`: `3015.700 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4342.750 us`
- `Stage`: `80255.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 81 | 23079.520 |
| `MatMulV3` | 162 | 12800.660 |
| `MatMulV2` | 324 | 10093.400 |
| `StridedSliceD` | 324 | 8761.020 |
| `AddLayerNorm` | 162 | 4005.380 |
| `Transpose` | 324 | 3412.140 |
| `Mul` | 324 | 3221.760 |
| `Gelu` | 81 | 3150.600 |
| `ConcatV2D` | 243 | 2549.900 |
| `Add` | 162 | 1823.840 |
| `Cast` | 162 | 1671.160 |
| `Neg` | 162 | 1341.220 |
| `SplitVD` | 81 | 1229.120 |
| `LayerNormV3` | 3 | 84.800 |
| `Data` | 3 | 14.920 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_19` | 3 | 888.920 |
| `PromptFlashAttention_10` | 3 | 880.800 |
| `PromptFlashAttention_26` | 3 | 880.440 |
| `PromptFlashAttention` | 3 | 880.260 |
| `PromptFlashAttention_15` | 3 | 878.100 |
| `PromptFlashAttention_3` | 3 | 873.920 |
| `PromptFlashAttention_2` | 3 | 872.120 |
| `PromptFlashAttention_5` | 3 | 869.820 |
| `PromptFlashAttention_9` | 3 | 867.040 |
| `PromptFlashAttention_16` | 3 | 865.040 |
| `PromptFlashAttention_17` | 3 | 858.240 |
| `PromptFlashAttention_25` | 3 | 858.220 |
| `PromptFlashAttention_14` | 3 | 857.720 |
| `PromptFlashAttention_18` | 3 | 857.320 |
| `PromptFlashAttention_23` | 3 | 856.760 |
| `PromptFlashAttention_8` | 3 | 856.180 |
| `PromptFlashAttention_1` | 3 | 854.600 |
| `PromptFlashAttention_24` | 3 | 839.900 |
| `PromptFlashAttention_11` | 3 | 837.860 |
| `PromptFlashAttention_22` | 3 | 837.480 |
| `PromptFlashAttention_13` | 3 | 834.740 |
| `PromptFlashAttention_7` | 3 | 834.660 |
| `PromptFlashAttention_6` | 3 | 831.780 |
| `PromptFlashAttention_4` | 3 | 828.260 |
| `PromptFlashAttention_12` | 3 | 827.520 |
| `PromptFlashAttention_20` | 3 | 826.800 |
| `PromptFlashAttention_21` | 3 | 825.020 |
| `MatMulV2_112_to_v3` | 3 | 262.980 |
| `MatMulV2_22_to_v3` | 3 | 261.200 |
| `MatMulV2_28_to_v3` | 3 | 260.540 |
| `MatMulV2_76_to_v3` | 3 | 259.800 |
| `MatMulV2_40_to_v3` | 3 | 259.500 |
| `MatMulV2_124_to_v3` | 3 | 259.220 |
| `MatMulV2_106_to_v3` | 3 | 258.940 |
| `MatMulV2_46_to_v3` | 3 | 258.820 |
| `MatMulV2_88_to_v3` | 3 | 258.820 |
| `MatMulV2_160_to_v3` | 3 | 258.740 |
| `MatMulV2_82_to_v3` | 3 | 258.660 |
| `MatMulV2_10_to_v3` | 3 | 258.640 |
| `MatMulV2_52_to_v3` | 3 | 258.380 |
| `MatMulV2_130_to_v3` | 3 | 258.260 |
| `MatMulV2_100_to_v3` | 3 | 258.100 |
| `MatMulV2_118_to_v3` | 3 | 258.020 |
| `MatMulV2_136_to_v3` | 3 | 257.660 |
| `MatMulV2_64_to_v3` | 3 | 257.600 |
| `MatMulV2_154_to_v3` | 3 | 257.420 |
| `MatMulV2_148_to_v3` | 3 | 257.260 |
| `MatMulV2_58_to_v3` | 3 | 257.220 |
| `MatMulV2_16_to_v3` | 3 | 257.120 |
| `MatMulV2_70_to_v3` | 3 | 257.100 |
| `MatMulV2_4_to_v3` | 3 | 256.000 |
| `MatMulV2_34_to_v3` | 3 | 255.100 |
| `MatMulV2_94_to_v3` | 3 | 254.540 |
| `MatMulV2_142_to_v3` | 3 | 253.820 |
| `MatMulV2_149_to_v3` | 3 | 224.820 |
| `MatMulV2_5_to_v3` | 3 | 218.340 |
| `MatMulV2_137_to_v3` | 3 | 217.760 |
| `MatMulV2_155_to_v3` | 3 | 217.400 |
| `MatMulV2_143_to_v3` | 3 | 216.820 |
| `MatMulV2_119_to_v3` | 3 | 216.340 |
| `MatMulV2_17_to_v3` | 3 | 216.300 |
| `MatMulV2_83_to_v3` | 3 | 216.280 |
| `MatMulV2_131_to_v3` | 3 | 216.260 |
| `MatMulV2_89_to_v3` | 3 | 216.000 |
| `MatMulV2_35_to_v3` | 3 | 215.660 |
| `MatMulV2_65_to_v3` | 3 | 215.520 |
| `MatMulV2_107_to_v3` | 3 | 215.360 |
| `MatMulV2_95_to_v3` | 3 | 215.340 |
| `MatMulV2_161_to_v3` | 3 | 215.320 |
| `MatMulV2_101_to_v3` | 3 | 215.280 |
| `MatMulV2_29_to_v3` | 3 | 215.220 |
| `MatMulV2_41_to_v3` | 3 | 215.160 |
| `MatMulV2_47_to_v3` | 3 | 215.120 |
| `MatMulV2_53_to_v3` | 3 | 215.100 |
| `MatMulV2_59_to_v3` | 3 | 215.080 |
| `MatMulV2_77_to_v3` | 3 | 214.980 |
| `MatMulV2_11_to_v3` | 3 | 214.980 |
| `MatMulV2_23_to_v3` | 3 | 214.800 |
| `MatMulV2_113_to_v3` | 3 | 214.600 |
| `MatMulV2_125_to_v3` | 3 | 214.320 |
| `MatMulV2_71_to_v3` | 3 | 213.040 |
| `Gelu_21` | 3 | 132.560 |
| `Gelu_20` | 3 | 132.360 |
| `Gelu_22` | 3 | 116.260 |
| `Gelu_18` | 3 | 116.260 |
| `Gelu_10` | 3 | 116.200 |
| `Gelu` | 3 | 116.180 |
| `Gelu_5` | 3 | 116.140 |
| `Gelu_14` | 3 | 115.960 |
| `Gelu_16` | 3 | 115.500 |
| `Gelu_4` | 3 | 115.460 |
| `Gelu_7` | 3 | 115.440 |
| `Gelu_19` | 3 | 115.320 |
| `Gelu_25` | 3 | 115.260 |
| `Gelu_23` | 3 | 115.240 |
| `Gelu_8` | 3 | 115.240 |
| `Gelu_24` | 3 | 115.200 |
| `Gelu_12` | 3 | 115.180 |
| `Gelu_17` | 3 | 115.180 |
| `Gelu_15` | 3 | 115.160 |
| `Gelu_11` | 3 | 115.140 |
| `Gelu_3` | 3 | 115.120 |
| `Gelu_6` | 3 | 115.100 |
| `Gelu_2` | 3 | 115.060 |
| `Gelu_13` | 3 | 115.060 |
| `Gelu_1` | 3 | 115.040 |
| `Gelu_26` | 3 | 115.040 |
| `Gelu_9` | 3 | 114.940 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 112.680 |
| `MatMulV2_132` | 3 | 108.140 |
| `MatMulV2_36` | 3 | 104.400 |
| `MatMulV2_18` | 3 | 104.360 |
| `MatMulV2_102` | 3 | 104.180 |
| `MatMulV2_90` | 3 | 104.100 |
| `MatMulV2_60` | 3 | 104.060 |
| `MatMulV2_48` | 3 | 104.000 |
| `MatMulV2_156` | 3 | 103.980 |
| `MatMulV2_108` | 3 | 103.780 |
| `MatMulV2_96` | 3 | 103.760 |
| `MatMulV2_24` | 3 | 103.620 |
| `MatMulV2_42` | 3 | 103.620 |
| `MatMulV2_54` | 3 | 103.620 |
| `MatMulV2_6` | 3 | 103.560 |
| `MatMulV2_120` | 3 | 103.500 |
| `MatMulV2_150` | 3 | 103.340 |
| `MatMulV2_138` | 3 | 103.320 |
| `MatMulV2_126` | 3 | 103.200 |
| `MatMulV2` | 3 | 103.200 |
| `MatMulV2_144` | 3 | 103.140 |
| `MatMulV2_78` | 3 | 103.000 |
| `MatMulV2_30` | 3 | 102.900 |
| `MatMulV2_72` | 3 | 102.900 |
| `MatMulV2_12` | 3 | 102.780 |
| `MatMulV2_114` | 3 | 102.740 |
| `MatMulV2_84` | 3 | 102.720 |
| `MatMulV2_66` | 3 | 102.460 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 100.760 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 100.700 |
| `MatMulV2_14` | 3 | 100.440 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 100.320 |
| `StridedSliceV2_104` | 3 | 100.300 |
| `MatMulV2_19` | 3 | 100.180 |
| `MatMulV2_55` | 3 | 100.100 |
| `MatMulV2_32` | 3 | 100.080 |
| `MatMulV2_103` | 3 | 100.080 |
| `MatMulV2_115` | 3 | 100.080 |
| `MatMulV2_80` | 3 | 100.040 |
| `MatMulV2_140` | 3 | 99.980 |
| `MatMulV2_151` | 3 | 99.820 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 99.760 |
| `MatMulV2_85` | 3 | 99.640 |
| `MatMulV2_62` | 3 | 99.540 |
| `MatMulV2_133` | 3 | 99.520 |
| `MatMulV2_92` | 3 | 99.360 |
| `MatMulV2_158` | 3 | 99.360 |
| `StridedSliceV2_68` | 3 | 99.320 |
| `MatMulV2_2` | 3 | 99.260 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 99.260 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 99.180 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 99.060 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 99.020 |
| `MatMulV2_8` | 3 | 98.960 |
| `MatMulV2_104` | 3 | 98.960 |
| `MatMulV2_128` | 3 | 98.920 |
| `MatMulV2_44` | 3 | 98.880 |
| `MatMulV2_7` | 3 | 98.840 |
| `MatMulV2_79` | 3 | 98.780 |
| `MatMulV2_1` | 3 | 98.620 |
| `MatMulV2_20` | 3 | 98.540 |
| `MatMulV2_91` | 3 | 98.340 |
| `MatMulV2_26` | 3 | 98.320 |
| `MatMulV2_74` | 3 | 98.300 |
| `MatMulV2_56` | 3 | 98.300 |
| `MatMulV2_152` | 3 | 98.280 |
| `MatMulV2_145` | 3 | 98.260 |
| `MatMulV2_110` | 3 | 98.200 |
| `MatMulV2_67` | 3 | 98.200 |
| `MatMulV2_25` | 3 | 98.180 |
| `MatMulV2_38` | 3 | 98.180 |
| `MatMulV2_49` | 3 | 98.140 |
| `MatMulV2_68` | 3 | 98.120 |
| `MatMulV2_73` | 3 | 98.120 |
| `MatMulV2_43` | 3 | 98.000 |
| `MatMulV2_122` | 3 | 97.900 |
| `MatMulV2_157` | 3 | 97.900 |
| `MatMulV2_61` | 3 | 97.880 |
| `MatMulV2_13` | 3 | 97.860 |
| `MatMulV2_31` | 3 | 97.800 |
| `MatMulV2_37` | 3 | 97.720 |
| `MatMulV2_86` | 3 | 97.580 |
| `MatMulV2_97` | 3 | 97.540 |
| `MatMulV2_121` | 3 | 97.500 |
| `MatMulV2_127` | 3 | 97.380 |
| `MatMulV2_134` | 3 | 97.320 |
| `MatMulV2_109` | 3 | 97.300 |
| `MatMulV2_139` | 3 | 97.160 |
| `MatMulV2_50` | 3 | 97.100 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 97.100 |
| `MatMulV2_98` | 3 | 96.980 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 96.960 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 23079.520 |
| `StridedSliceD | "1,2048,16,80" -> "1,2048,16,40" | ND -> ND` | 324 | 8761.020 |
| `MatMulV2 | "2048,1152;1280,1152;1280" -> "2048,1280" | ND;ND;ND -> ND` | 243 | 8117.560 |
| `MatMulV3 | "2048,1152;4352,1152;4352" -> "2048,4352" | ND;ND;ND -> ND` | 81 | 6969.460 |
| `MatMulV3 | "2048,4352;1152,4352;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 5831.200 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4005.380 |
| `Mul | "1,2048,16,80;1,2048,1,80" -> "1,2048,16,80" | ND;ND -> ND` | 324 | 3221.760 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3150.600 |
| `Transpose | "2048,16,80;3" -> "16,2048,80" | ND;ND -> ND` | 243 | 2483.480 |
| `MatMulV2 | "2048,1280;1152,1280;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 1975.840 |
| `Add | "1,2048,16,80;1,2048,16,80" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1823.840 |
| `Cast | "1,2048,16,80" -> "1,2048,16,80" | ND -> ND` | 162 | 1671.160 |
| `ConcatV2D | "1,2048,16,40;1,2048,16,40" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1606.080 |
| `Neg | "1,2048,16,40" -> "1,2048,16,40" | ND -> ND` | 162 | 1341.220 |
| `SplitVD | "1,2048,3840" -> "1,2048,1280;1,2048,1280;1,2048,1280" | ND -> ND;ND;ND` | 81 | 1229.120 |
| `ConcatV2D | "1,2048,1280;1,2048,1280;1,2048,1280" -> "1,2048,3840" | ND;ND;ND -> ND` | 81 | 943.820 |
| `Transpose | "16,2048,80;3" -> "2048,16,80" | ND;ND -> ND` | 81 | 928.660 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 84.800 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.920 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 27084.900 |
| `ND;ND;ND` | 570 | 23922.680 |
| `ND` | 810 | 16153.120 |
| `ND;ND` | 972 | 10063.820 |
| `N/A` | 3 | 14.920 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_19` | 0 | 301.300 |
| `PromptFlashAttention_26` | 0 | 295.980 |
| `PromptFlashAttention_26` | 0 | 295.760 |
| `PromptFlashAttention_10` | 0 | 295.740 |
| `PromptFlashAttention_19` | 0 | 294.540 |
| `PromptFlashAttention` | 0 | 294.260 |
| `PromptFlashAttention_15` | 0 | 293.700 |
| `PromptFlashAttention` | 0 | 293.220 |
| `PromptFlashAttention_19` | 0 | 293.080 |
| `PromptFlashAttention_15` | 0 | 293.000 |
| `PromptFlashAttention_10` | 0 | 292.960 |
| `PromptFlashAttention` | 0 | 292.780 |
| `PromptFlashAttention_5` | 0 | 292.500 |
| `PromptFlashAttention_3` | 0 | 292.300 |
| `PromptFlashAttention_3` | 0 | 292.160 |
| `PromptFlashAttention_10` | 0 | 292.100 |
| `PromptFlashAttention_2` | 0 | 291.760 |
| `PromptFlashAttention_5` | 0 | 291.420 |
| `PromptFlashAttention_15` | 0 | 291.400 |
| `PromptFlashAttention_2` | 0 | 290.300 |
| `PromptFlashAttention_16` | 0 | 290.100 |
| `PromptFlashAttention_2` | 0 | 290.060 |
| `PromptFlashAttention_9` | 0 | 289.820 |
| `PromptFlashAttention_3` | 0 | 289.460 |
| `PromptFlashAttention_9` | 0 | 289.300 |
| `PromptFlashAttention_16` | 0 | 289.040 |
| `PromptFlashAttention_25` | 0 | 289.000 |
| `PromptFlashAttention_26` | 0 | 288.700 |
| `PromptFlashAttention_9` | 0 | 287.920 |
| `PromptFlashAttention_1` | 0 | 287.920 |
| `PromptFlashAttention_8` | 0 | 287.920 |
| `PromptFlashAttention_14` | 0 | 287.200 |
| `PromptFlashAttention_17` | 0 | 287.120 |
| `PromptFlashAttention_14` | 0 | 286.480 |
| `PromptFlashAttention_18` | 0 | 286.280 |
| `PromptFlashAttention_18` | 0 | 286.060 |
| `PromptFlashAttention_23` | 0 | 286.040 |
| `PromptFlashAttention_17` | 0 | 286.000 |
| `PromptFlashAttention_5` | 0 | 285.900 |
| `PromptFlashAttention_16` | 0 | 285.900 |
| `PromptFlashAttention_23` | 0 | 285.520 |
| `PromptFlashAttention_23` | 0 | 285.200 |
| `PromptFlashAttention_17` | 0 | 285.120 |
| `PromptFlashAttention_18` | 0 | 284.980 |
| `PromptFlashAttention_25` | 0 | 284.620 |
| `PromptFlashAttention_25` | 0 | 284.600 |
| `PromptFlashAttention_8` | 0 | 284.460 |
| `PromptFlashAttention_14` | 0 | 284.040 |
| `PromptFlashAttention_8` | 0 | 283.800 |
| `PromptFlashAttention_1` | 0 | 283.540 |
| `PromptFlashAttention_1` | 0 | 283.140 |
| `PromptFlashAttention_24` | 0 | 281.240 |
| `PromptFlashAttention_24` | 0 | 280.180 |
| `PromptFlashAttention_11` | 0 | 279.720 |
| `PromptFlashAttention_11` | 0 | 279.640 |
| `PromptFlashAttention_22` | 0 | 279.540 |
| `PromptFlashAttention_22` | 0 | 279.080 |
| `PromptFlashAttention_22` | 0 | 278.860 |
| `PromptFlashAttention_13` | 0 | 278.720 |
| `PromptFlashAttention_11` | 0 | 278.500 |
| `PromptFlashAttention_24` | 0 | 278.480 |
| `PromptFlashAttention_7` | 0 | 278.460 |
| `PromptFlashAttention_13` | 0 | 278.220 |
| `PromptFlashAttention_7` | 0 | 278.160 |
| `PromptFlashAttention_7` | 0 | 278.040 |
| `PromptFlashAttention_6` | 0 | 277.920 |
| `PromptFlashAttention_13` | 0 | 277.800 |
| `PromptFlashAttention_12` | 0 | 277.440 |
| `PromptFlashAttention_4` | 0 | 277.140 |
| `PromptFlashAttention_6` | 0 | 277.060 |
| `PromptFlashAttention_6` | 0 | 276.800 |
| `PromptFlashAttention_4` | 0 | 276.520 |
| `PromptFlashAttention_20` | 0 | 276.080 |
| `PromptFlashAttention_12` | 0 | 275.660 |
| `PromptFlashAttention_21` | 0 | 275.580 |
| `PromptFlashAttention_20` | 0 | 275.440 |
| `PromptFlashAttention_20` | 0 | 275.280 |
| `PromptFlashAttention_21` | 0 | 274.840 |
| `PromptFlashAttention_4` | 0 | 274.600 |
| `PromptFlashAttention_21` | 0 | 274.600 |
| `PromptFlashAttention_12` | 0 | 274.420 |
| `MatMulV2_112_to_v3` | 0 | 88.300 |
| `MatMulV2_112_to_v3` | 0 | 88.240 |
| `MatMulV2_40_to_v3` | 0 | 87.940 |
| `MatMulV2_106_to_v3` | 0 | 87.680 |
| `MatMulV2_76_to_v3` | 0 | 87.520 |
| `MatMulV2_160_to_v3` | 0 | 87.440 |
| `MatMulV2_10_to_v3` | 0 | 87.320 |
| `MatMulV2_22_to_v3` | 0 | 87.300 |
| `MatMulV2_82_to_v3` | 0 | 87.220 |
| `MatMulV2_22_to_v3` | 0 | 87.200 |
| `MatMulV2_100_to_v3` | 0 | 87.200 |
| `MatMulV2_28_to_v3` | 0 | 87.080 |
| `MatMulV2_130_to_v3` | 0 | 87.040 |
| `MatMulV2_64_to_v3` | 0 | 86.940 |
| `MatMulV2_118_to_v3` | 0 | 86.860 |
| `MatMulV2_154_to_v3` | 0 | 86.840 |
| `MatMulV2_28_to_v3` | 0 | 86.820 |
| `MatMulV2_88_to_v3` | 0 | 86.780 |
| `MatMulV2_124_to_v3` | 0 | 86.760 |
| `MatMulV2_88_to_v3` | 0 | 86.720 |
| `MatMulV2_22_to_v3` | 0 | 86.700 |
| `MatMulV2_28_to_v3` | 0 | 86.640 |
| `MatMulV2_52_to_v3` | 0 | 86.580 |
| `MatMulV2_46_to_v3` | 0 | 86.560 |
| `MatMulV2_124_to_v3` | 0 | 86.560 |
| `MatMulV2_76_to_v3` | 0 | 86.460 |
| `MatMulV2_112_to_v3` | 0 | 86.440 |
| `MatMulV2_106_to_v3` | 0 | 86.360 |
| `MatMulV2_148_to_v3` | 0 | 86.360 |
| `MatMulV2_46_to_v3` | 0 | 86.320 |
| `MatMulV2_136_to_v3` | 0 | 86.200 |
| `MatMulV2_82_to_v3` | 0 | 86.180 |
| `MatMulV2_16_to_v3` | 0 | 86.180 |
| `MatMulV2_118_to_v3` | 0 | 86.140 |
| `MatMulV2_40_to_v3` | 0 | 86.120 |
| `MatMulV2_70_to_v3` | 0 | 86.120 |
| `MatMulV2_58_to_v3` | 0 | 86.060 |
| `MatMulV2_136_to_v3` | 0 | 86.020 |
| `MatMulV2_52_to_v3` | 0 | 86.000 |
| `MatMulV2_58_to_v3` | 0 | 85.980 |
| `MatMulV2_46_to_v3` | 0 | 85.940 |
| `MatMulV2_124_to_v3` | 0 | 85.900 |
| `MatMulV2_76_to_v3` | 0 | 85.820 |
| `MatMulV2_52_to_v3` | 0 | 85.800 |
| `MatMulV2_4_to_v3` | 0 | 85.760 |
| `MatMulV2_160_to_v3` | 0 | 85.760 |
| `MatMulV2_10_to_v3` | 0 | 85.740 |
| `MatMulV2_100_to_v3` | 0 | 85.740 |
| `MatMulV2_4_to_v3` | 0 | 85.740 |
| `MatMulV2_130_to_v3` | 0 | 85.720 |
| `MatMulV2_148_to_v3` | 0 | 85.640 |
| `MatMulV2_10_to_v3` | 0 | 85.580 |
| `MatMulV2_160_to_v3` | 0 | 85.540 |
| `MatMulV2_16_to_v3` | 0 | 85.520 |
| `MatMulV2_64_to_v3` | 0 | 85.520 |
| `MatMulV2_70_to_v3` | 0 | 85.520 |
| `MatMulV2_154_to_v3` | 0 | 85.520 |
| `MatMulV2_130_to_v3` | 0 | 85.500 |
| `MatMulV2_70_to_v3` | 0 | 85.460 |
| `MatMulV2_40_to_v3` | 0 | 85.440 |
| `MatMulV2_136_to_v3` | 0 | 85.440 |
| `MatMulV2_16_to_v3` | 0 | 85.420 |
| `MatMulV2_34_to_v3` | 0 | 85.420 |
| `MatMulV2_88_to_v3` | 0 | 85.320 |
| `MatMulV2_148_to_v3` | 0 | 85.260 |
| `MatMulV2_82_to_v3` | 0 | 85.260 |
| `MatMulV2_58_to_v3` | 0 | 85.180 |
| `MatMulV2_100_to_v3` | 0 | 85.160 |
| `MatMulV2_64_to_v3` | 0 | 85.140 |
| `MatMulV2_142_to_v3` | 0 | 85.120 |
| `MatMulV2_154_to_v3` | 0 | 85.060 |
| `MatMulV2_94_to_v3` | 0 | 85.060 |
| `MatMulV2_118_to_v3` | 0 | 85.020 |
| `MatMulV2_34_to_v3` | 0 | 84.960 |
| `MatMulV2_106_to_v3` | 0 | 84.900 |
| `MatMulV2_94_to_v3` | 0 | 84.880 |
| `MatMulV2_34_to_v3` | 0 | 84.720 |
| `MatMulV2_94_to_v3` | 0 | 84.600 |
| `MatMulV2_142_to_v3` | 0 | 84.520 |
| `MatMulV2_4_to_v3` | 0 | 84.500 |
| `MatMulV2_142_to_v3` | 0 | 84.180 |
| `MatMulV2_149_to_v3` | 0 | 75.160 |
| `MatMulV2_149_to_v3` | 0 | 74.840 |
| `MatMulV2_149_to_v3` | 0 | 74.820 |
| `MatMulV2_155_to_v3` | 0 | 73.820 |
| `MatMulV2_5_to_v3` | 0 | 73.640 |
| `MatMulV2_119_to_v3` | 0 | 73.180 |
| `MatMulV2_137_to_v3` | 0 | 73.060 |
| `MatMulV2_35_to_v3` | 0 | 73.040 |
| `MatMulV2_47_to_v3` | 0 | 72.900 |
| `MatMulV2_143_to_v3` | 0 | 72.880 |
| `MatMulV2_95_to_v3` | 0 | 72.820 |
| `MatMulV2_101_to_v3` | 0 | 72.680 |
| `MatMulV2_89_to_v3` | 0 | 72.660 |
| `MatMulV2_17_to_v3` | 0 | 72.640 |
| `MatMulV2_107_to_v3` | 0 | 72.600 |
| `MatMulV2_5_to_v3` | 0 | 72.520 |
| `MatMulV2_137_to_v3` | 0 | 72.480 |
| `MatMulV2_131_to_v3` | 0 | 72.480 |
| `MatMulV2_83_to_v3` | 0 | 72.400 |
| `MatMulV2_83_to_v3` | 0 | 72.380 |
| `MatMulV2_35_to_v3` | 0 | 72.280 |
| `MatMulV2_65_to_v3` | 0 | 72.280 |
| `MatMulV2_77_to_v3` | 0 | 72.260 |
| `MatMulV2_137_to_v3` | 0 | 72.220 |
| `MatMulV2_5_to_v3` | 0 | 72.180 |
| `MatMulV2_131_to_v3` | 0 | 72.120 |
| `MatMulV2_155_to_v3` | 0 | 72.120 |
| `MatMulV2_11_to_v3` | 0 | 72.100 |
| `MatMulV2_59_to_v3` | 0 | 72.080 |
| `MatMulV2_53_to_v3` | 0 | 72.060 |
| `MatMulV2_65_to_v3` | 0 | 72.040 |
| `MatMulV2_161_to_v3` | 0 | 72.040 |
| `MatMulV2_41_to_v3` | 0 | 72.020 |
| `MatMulV2_143_to_v3` | 0 | 72.000 |
| `MatMulV2_17_to_v3` | 0 | 71.980 |
| `MatMulV2_107_to_v3` | 0 | 71.960 |
| `MatMulV2_119_to_v3` | 0 | 71.960 |
| `MatMulV2_29_to_v3` | 0 | 71.940 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 28785.080 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.torchair.active.step1` | 1 | 28166.250 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.torchair.active.step3` | 1 | 26992.580 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.torchair.active.step2` | 1 | 26954.850 |
| `TorchDynamo Cache Lookup` | 3 | 25911.970 |
| `Torch-Compiled Region: 0/0` | 3 | 4491.920 |
| `TorchNpuGraphBase::Run` | 3 | 2953.610 |
| `RefreshAtTensorFromGeTensor` | 3 | 1160.230 |
| `ExecuteGraph` | 3 | 585.760 |
| `aten::empty` | 3 | 582.340 |
| `AssembleInputs` | 3 | 480.650 |
| `AssembleOutputs` | 3 | 311.940 |
| `empty_tensor` | 3 | 305.020 |
| `aten::set_` | 3 | 279.650 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 190800.980 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 75282.920 |
| `launch` | 868 | 13147.730 |
| `InputCopy` | 3 | 206.490 |
| `ModelExecute` | 3 | 68.860 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 37.490 |
| `step_info` | 6 | 30.130 |
| `OutputCopy` | 3 | 1.470 |

