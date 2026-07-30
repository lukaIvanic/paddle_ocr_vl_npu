# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b1s2048_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b1s2048_pipe/liteserver-c001-4_763762_20260730140535719_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `78193.980 us`
- `Free`: `3258.460 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3913.250 us`
- `Stage`: `81452.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 24221.740 |
| `PromptFlashAttention` | 81 | 22770.740 |
| `StridedSliceD` | 324 | 8492.000 |
| `AddLayerNorm` | 162 | 4067.980 |
| `Transpose` | 324 | 3557.360 |
| `Mul` | 324 | 3171.000 |
| `Gelu` | 81 | 3035.760 |
| `ConcatV2D` | 243 | 2612.000 |
| `Add` | 162 | 1834.200 |
| `Cast` | 162 | 1698.900 |
| `Neg` | 162 | 1387.780 |
| `SplitVD` | 81 | 1240.640 |
| `LayerNormV3` | 3 | 88.820 |
| `Data` | 3 | 15.060 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 3 | 867.380 |
| `PromptFlashAttention_17` | 3 | 859.720 |
| `PromptFlashAttention_10` | 3 | 857.920 |
| `PromptFlashAttention_18` | 3 | 857.740 |
| `PromptFlashAttention_19` | 3 | 857.500 |
| `PromptFlashAttention_2` | 3 | 855.320 |
| `PromptFlashAttention_9` | 3 | 854.720 |
| `PromptFlashAttention_16` | 3 | 853.720 |
| `PromptFlashAttention_15` | 3 | 853.580 |
| `PromptFlashAttention_26` | 3 | 852.080 |
| `PromptFlashAttention_24` | 3 | 845.340 |
| `PromptFlashAttention_3` | 3 | 845.040 |
| `PromptFlashAttention_8` | 3 | 842.540 |
| `PromptFlashAttention_20` | 3 | 842.260 |
| `PromptFlashAttention_21` | 3 | 842.260 |
| `PromptFlashAttention_13` | 3 | 841.740 |
| `PromptFlashAttention_4` | 3 | 838.800 |
| `PromptFlashAttention_1` | 3 | 838.080 |
| `PromptFlashAttention_14` | 3 | 837.040 |
| `PromptFlashAttention_11` | 3 | 836.720 |
| `PromptFlashAttention_25` | 3 | 834.340 |
| `PromptFlashAttention_23` | 3 | 831.640 |
| `PromptFlashAttention_7` | 3 | 831.300 |
| `PromptFlashAttention_12` | 3 | 828.140 |
| `PromptFlashAttention_22` | 3 | 824.140 |
| `PromptFlashAttention_6` | 3 | 821.400 |
| `PromptFlashAttention_5` | 3 | 820.280 |
| `MatMulV2_149` | 3 | 284.580 |
| `MatMulV2_119` | 3 | 280.620 |
| `MatMulV2_113` | 3 | 280.460 |
| `MatMulV2_131` | 3 | 280.240 |
| `MatMulV2_71` | 3 | 279.100 |
| `MatMulV2_143` | 3 | 279.060 |
| `MatMulV2_35` | 3 | 278.760 |
| `MatMulV2_155` | 3 | 278.360 |
| `MatMulV2_47` | 3 | 278.340 |
| `MatMulV2_89` | 3 | 277.980 |
| `MatMulV2_101` | 3 | 277.640 |
| `MatMulV2_83` | 3 | 277.280 |
| `MatMulV2_107` | 3 | 277.280 |
| `MatMulV2_59` | 3 | 276.860 |
| `MatMulV2_137` | 3 | 275.120 |
| `MatMulV2_125` | 3 | 274.800 |
| `MatMulV2_11` | 3 | 274.700 |
| `MatMulV2_65` | 3 | 274.700 |
| `MatMulV2_77` | 3 | 274.440 |
| `MatMulV2_95` | 3 | 274.380 |
| `MatMulV2_23` | 3 | 273.760 |
| `MatMulV2_53` | 3 | 273.260 |
| `MatMulV2_161` | 3 | 271.180 |
| `MatMulV2_41` | 3 | 270.700 |
| `MatMulV2_5` | 3 | 269.800 |
| `MatMulV2_29` | 3 | 269.280 |
| `MatMulV2_17` | 3 | 258.880 |
| `MatMulV2_76` | 3 | 258.360 |
| `MatMulV2_118` | 3 | 257.880 |
| `MatMulV2_130` | 3 | 257.260 |
| `MatMulV2_148` | 3 | 256.920 |
| `MatMulV2_106` | 3 | 255.980 |
| `MatMulV2_124` | 3 | 255.880 |
| `MatMulV2_46` | 3 | 255.460 |
| `MatMulV2_52` | 3 | 255.460 |
| `MatMulV2_64` | 3 | 255.160 |
| `MatMulV2_154` | 3 | 254.960 |
| `MatMulV2_10` | 3 | 254.800 |
| `MatMulV2_112` | 3 | 254.760 |
| `MatMulV2_136` | 3 | 254.720 |
| `MatMulV2_70` | 3 | 254.720 |
| `MatMulV2_94` | 3 | 253.600 |
| `MatMulV2_142` | 3 | 253.560 |
| `MatMulV2_88` | 3 | 253.520 |
| `MatMulV2_100` | 3 | 253.180 |
| `MatMulV2_28` | 3 | 252.300 |
| `MatMulV2_22` | 3 | 252.280 |
| `MatMulV2_82` | 3 | 251.680 |
| `MatMulV2_58` | 3 | 251.200 |
| `MatMulV2_34` | 3 | 250.980 |
| `MatMulV2_40` | 3 | 250.800 |
| `MatMulV2_160` | 3 | 249.280 |
| `MatMulV2_4` | 3 | 248.600 |
| `MatMulV2_16` | 3 | 244.220 |
| `Gelu_23` | 3 | 125.300 |
| `MatMulV2_153` | 3 | 125.220 |
| `MatMulV2_63` | 3 | 123.260 |
| `MatMulV2_147` | 3 | 123.200 |
| `MatMulV2_39` | 3 | 122.720 |
| `MatMulV2_45` | 3 | 122.700 |
| `MatMulV2_33` | 3 | 122.320 |
| `MatMulV2_141` | 3 | 122.280 |
| `MatMulV2_3` | 3 | 121.820 |
| `MatMulV2_111` | 3 | 121.560 |
| `MatMulV2_87` | 3 | 120.640 |
| `MatMulV2_75` | 3 | 120.360 |
| `MatMulV2_15` | 3 | 118.720 |
| `MatMulV2_105` | 3 | 118.300 |
| `MatMulV2_123` | 3 | 117.960 |
| `MatMulV2_9` | 3 | 117.420 |
| `MatMulV2_69` | 3 | 117.260 |
| `MatMulV2_21` | 3 | 116.420 |
| `MatMulV2_81` | 3 | 116.080 |
| `MatMulV2_57` | 3 | 115.220 |
| `MatMulV2_135` | 3 | 114.780 |
| `MatMulV2_117` | 3 | 114.580 |
| `MatMulV2_159` | 3 | 113.920 |
| `Gelu_17` | 3 | 113.040 |
| `Gelu_1` | 3 | 112.940 |
| `Gelu_10` | 3 | 112.920 |
| `Gelu_14` | 3 | 112.840 |
| `Gelu_7` | 3 | 112.820 |
| `MatMulV2_51` | 3 | 112.220 |
| `MatMulV2_129` | 3 | 111.940 |
| `Gelu_19` | 3 | 111.920 |
| `Gelu_11` | 3 | 111.840 |
| `Gelu_16` | 3 | 111.840 |
| `Gelu_8` | 3 | 111.820 |
| `Gelu_20` | 3 | 111.800 |
| `Gelu_4` | 3 | 111.780 |
| `Gelu_22` | 3 | 111.780 |
| `Gelu_15` | 3 | 111.720 |
| `Gelu_21` | 3 | 111.720 |
| `Gelu_6` | 3 | 111.700 |
| `Gelu_18` | 3 | 111.700 |
| `Gelu_24` | 3 | 111.700 |
| `Gelu_5` | 3 | 111.680 |
| `Gelu_26` | 3 | 111.660 |
| `Gelu` | 3 | 111.660 |
| `Gelu_3` | 3 | 111.640 |
| `Gelu_12` | 3 | 111.640 |
| `Gelu_9` | 3 | 111.620 |
| `Gelu_2` | 3 | 111.580 |
| `Gelu_13` | 3 | 111.580 |
| `Gelu_25` | 3 | 111.520 |
| `MatMulV2_93` | 3 | 111.440 |
| `MatMulV2_99` | 3 | 110.500 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 110.360 |
| `MatMulV2_27` | 3 | 109.820 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 100.960 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 100.760 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 100.480 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 100.480 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 100.440 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 100.400 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 100.320 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 100.300 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 100.040 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 100.020 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 99.940 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 99.840 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 99.760 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 99.160 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 99.140 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 99.100 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 98.940 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 98.820 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 98.500 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 98.440 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 98.140 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 97.900 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 96.920 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 96.900 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 96.760 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 96.520 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 96.340 |
| `MatMulV2_42` | 3 | 92.180 |
| `MatMulV2_24` | 3 | 91.920 |
| `MatMulV2_150` | 3 | 90.880 |
| `MatMulV2_66` | 3 | 90.680 |
| `MatMulV2_30` | 3 | 90.560 |
| `MatMulV2_108` | 3 | 90.500 |
| `MatMulV2_102` | 3 | 90.340 |
| `MatMulV2_72` | 3 | 90.060 |
| `MatMulV2_126` | 3 | 89.900 |
| `MatMulV2_96` | 3 | 89.640 |
| `MatMulV2_60` | 3 | 89.560 |
| `MatMulV2_6` | 3 | 89.200 |
| `MatMulV2_54` | 3 | 89.040 |
| `MatMulV2_144` | 3 | 88.980 |
| `MatMulV2_48` | 3 | 88.960 |
| `MatMulV2_78` | 3 | 88.900 |
| `LayerNormV4_LayerNormV3` | 3 | 88.820 |
| `MatMulV2_120` | 3 | 88.740 |
| `MatMulV2_18` | 3 | 88.640 |
| `MatMulV2_84` | 3 | 88.560 |
| `MatMulV2_132` | 3 | 88.460 |
| `MatMulV2` | 3 | 88.260 |
| `MatMulV2_12` | 3 | 86.840 |
| `MatMulV2_36` | 3 | 86.080 |
| `MatMulV2_90` | 3 | 86.020 |
| `MatMulV2_156` | 3 | 85.900 |
| `MatMulV2_138` | 3 | 84.500 |
| `MatMulV2_114` | 3 | 84.340 |
| `MatMulV2_140` | 3 | 83.940 |
| `MatMulV2_104` | 3 | 82.960 |
| `MatMulV2_56` | 3 | 82.940 |
| `StridedSliceV2_80` | 3 | 82.920 |
| `MatMulV2_152` | 3 | 82.780 |
| `MatMulV2_68` | 3 | 82.480 |
| `MatMulV2_62` | 3 | 82.400 |
| `StridedSliceV2_4` | 3 | 82.380 |
| `MatMulV2_116` | 3 | 82.320 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22770.740 |
| `StridedSliceD | "1,2048,16,80" -> "1,2048,16,40" | ND -> ND` | 324 | 8492.000 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7441.560 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6847.520 |
| `MatMulV2 | "2048,1152;72,80,16,16;1280" -> "2048,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 6750.000 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4067.980 |
| `MatMulV2 | "2048,1280;80,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 3182.660 |
| `Mul | "1,2048,16,80;1,2048,1,80" -> "1,2048,16,80" | ND;ND -> ND` | 324 | 3171.000 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3035.760 |
| `Transpose | "2048,16,80;3" -> "16,2048,80" | ND;ND -> ND` | 243 | 2623.840 |
| `Add | "1,2048,16,80;1,2048,16,80" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1834.200 |
| `Cast | "1,2048,16,80" -> "1,2048,16,80" | ND -> ND` | 162 | 1698.900 |
| `ConcatV2D | "1,2048,16,40;1,2048,16,40" -> "1,2048,16,80" | ND;ND -> ND` | 162 | 1674.660 |
| `Neg | "1,2048,16,40" -> "1,2048,16,40" | ND -> ND` | 162 | 1387.780 |
| `SplitVD | "1,2048,3840" -> "1,2048,1280;1,2048,1280;1,2048,1280" | ND -> ND;ND;ND` | 81 | 1240.640 |
| `ConcatV2D | "1,2048,1280;1,2048,1280;1,2048,1280" -> "1,2048,3840" | ND;ND;ND -> ND` | 81 | 937.340 |
| `Transpose | "16,2048,80;3" -> "2048,16,80" | ND;ND -> ND` | 81 | 933.520 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 88.820 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.060 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 26838.720 |
| `ND;FRACTAL_NZ;ND` | 486 | 24221.740 |
| `ND` | 810 | 15855.080 |
| `ND;ND` | 972 | 10237.220 |
| `ND;ND;ND` | 84 | 1026.160 |
| `N/A` | 3 | 15.060 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 0 | 290.220 |
| `PromptFlashAttention` | 0 | 290.040 |
| `PromptFlashAttention_18` | 0 | 288.020 |
| `PromptFlashAttention_17` | 0 | 287.520 |
| `PromptFlashAttention_10` | 0 | 287.400 |
| `PromptFlashAttention_16` | 0 | 287.380 |
| `PromptFlashAttention_19` | 0 | 287.160 |
| `PromptFlashAttention` | 0 | 287.120 |
| `PromptFlashAttention_2` | 0 | 286.900 |
| `PromptFlashAttention_10` | 0 | 286.860 |
| `PromptFlashAttention_17` | 0 | 286.800 |
| `PromptFlashAttention_26` | 0 | 286.700 |
| `PromptFlashAttention_15` | 0 | 286.460 |
| `PromptFlashAttention_18` | 0 | 286.380 |
| `PromptFlashAttention_9` | 0 | 286.020 |
| `PromptFlashAttention_17` | 0 | 285.400 |
| `PromptFlashAttention_19` | 0 | 285.200 |
| `PromptFlashAttention_19` | 0 | 285.140 |
| `PromptFlashAttention_2` | 0 | 284.980 |
| `PromptFlashAttention_15` | 0 | 284.700 |
| `PromptFlashAttention_9` | 0 | 284.400 |
| `PromptFlashAttention_9` | 0 | 284.300 |
| `PromptFlashAttention_16` | 0 | 284.180 |
| `PromptFlashAttention_3` | 0 | 283.720 |
| `PromptFlashAttention_10` | 0 | 283.660 |
| `PromptFlashAttention_2` | 0 | 283.440 |
| `PromptFlashAttention_18` | 0 | 283.340 |
| `PromptFlashAttention_26` | 0 | 283.040 |
| `PromptFlashAttention_15` | 0 | 282.420 |
| `PromptFlashAttention_26` | 0 | 282.340 |
| `PromptFlashAttention_24` | 0 | 282.300 |
| `PromptFlashAttention_16` | 0 | 282.160 |
| `PromptFlashAttention_8` | 0 | 282.000 |
| `PromptFlashAttention_20` | 0 | 281.940 |
| `PromptFlashAttention_13` | 0 | 281.920 |
| `PromptFlashAttention_21` | 0 | 281.580 |
| `PromptFlashAttention_3` | 0 | 281.560 |
| `PromptFlashAttention_24` | 0 | 281.540 |
| `PromptFlashAttention_24` | 0 | 281.500 |
| `PromptFlashAttention_14` | 0 | 281.280 |
| `PromptFlashAttention_8` | 0 | 281.200 |
| `PromptFlashAttention_13` | 0 | 280.880 |
| `PromptFlashAttention_21` | 0 | 280.600 |
| `PromptFlashAttention_11` | 0 | 280.560 |
| `PromptFlashAttention_1` | 0 | 280.460 |
| `PromptFlashAttention_20` | 0 | 280.360 |
| `PromptFlashAttention_4` | 0 | 280.240 |
| `PromptFlashAttention_21` | 0 | 280.080 |
| `PromptFlashAttention_20` | 0 | 279.960 |
| `PromptFlashAttention_3` | 0 | 279.760 |
| `PromptFlashAttention_4` | 0 | 279.580 |
| `PromptFlashAttention_25` | 0 | 279.500 |
| `PromptFlashAttention_8` | 0 | 279.340 |
| `PromptFlashAttention_14` | 0 | 279.320 |
| `PromptFlashAttention_25` | 0 | 279.240 |
| `PromptFlashAttention_1` | 0 | 279.160 |
| `PromptFlashAttention_4` | 0 | 278.980 |
| `PromptFlashAttention_13` | 0 | 278.940 |
| `PromptFlashAttention_11` | 0 | 278.520 |
| `PromptFlashAttention_1` | 0 | 278.460 |
| `PromptFlashAttention_23` | 0 | 278.140 |
| `PromptFlashAttention_23` | 0 | 278.020 |
| `PromptFlashAttention_7` | 0 | 277.860 |
| `PromptFlashAttention_7` | 0 | 277.720 |
| `PromptFlashAttention_11` | 0 | 277.640 |
| `PromptFlashAttention_12` | 0 | 276.660 |
| `PromptFlashAttention_14` | 0 | 276.440 |
| `PromptFlashAttention_12` | 0 | 275.800 |
| `PromptFlashAttention_7` | 0 | 275.720 |
| `PromptFlashAttention_12` | 0 | 275.680 |
| `PromptFlashAttention_25` | 0 | 275.600 |
| `PromptFlashAttention_23` | 0 | 275.480 |
| `PromptFlashAttention_22` | 0 | 275.400 |
| `PromptFlashAttention_22` | 0 | 275.320 |
| `PromptFlashAttention_5` | 0 | 274.940 |
| `PromptFlashAttention_6` | 0 | 274.860 |
| `PromptFlashAttention_22` | 0 | 273.420 |
| `PromptFlashAttention_6` | 0 | 273.280 |
| `PromptFlashAttention_6` | 0 | 273.260 |
| `PromptFlashAttention_5` | 0 | 272.920 |
| `PromptFlashAttention_5` | 0 | 272.420 |
| `MatMulV2_149` | 0 | 96.080 |
| `MatMulV2_155` | 0 | 95.340 |
| `MatMulV2_149` | 0 | 94.640 |
| `MatMulV2_113` | 0 | 94.600 |
| `MatMulV2_23` | 0 | 94.560 |
| `MatMulV2_131` | 0 | 94.320 |
| `MatMulV2_11` | 0 | 94.320 |
| `MatMulV2_137` | 0 | 94.280 |
| `MatMulV2_119` | 0 | 94.240 |
| `MatMulV2_143` | 0 | 94.220 |
| `MatMulV2_143` | 0 | 94.140 |
| `MatMulV2_5` | 0 | 93.960 |
| `MatMulV2_113` | 0 | 93.900 |
| `MatMulV2_149` | 0 | 93.860 |
| `MatMulV2_119` | 0 | 93.800 |
| `MatMulV2_59` | 0 | 93.780 |
| `MatMulV2_107` | 0 | 93.740 |
| `MatMulV2_107` | 0 | 93.700 |
| `MatMulV2_35` | 0 | 93.620 |
| `MatMulV2_71` | 0 | 93.620 |
| `MatMulV2_65` | 0 | 93.580 |
| `MatMulV2_47` | 0 | 93.560 |
| `MatMulV2_71` | 0 | 93.520 |
| `MatMulV2_83` | 0 | 93.520 |
| `MatMulV2_155` | 0 | 93.500 |
| `MatMulV2_59` | 0 | 93.340 |
| `MatMulV2_77` | 0 | 93.340 |
| `MatMulV2_95` | 0 | 93.340 |
| `MatMulV2_131` | 0 | 93.340 |
| `MatMulV2_125` | 0 | 93.200 |
| `MatMulV2_65` | 0 | 93.160 |
| `MatMulV2_137` | 0 | 93.140 |
| `MatMulV2_41` | 0 | 93.140 |
| `MatMulV2_77` | 0 | 93.040 |
| `MatMulV2_125` | 0 | 93.040 |
| `MatMulV2_101` | 0 | 92.900 |
| `MatMulV2_89` | 0 | 92.840 |
| `MatMulV2_29` | 0 | 92.820 |
| `MatMulV2_35` | 0 | 92.780 |
| `MatMulV2_89` | 0 | 92.760 |
| `MatMulV2_101` | 0 | 92.640 |
| `MatMulV2_11` | 0 | 92.620 |
| `MatMulV2_53` | 0 | 92.580 |
| `MatMulV2_131` | 0 | 92.580 |
| `MatMulV2_119` | 0 | 92.580 |
| `MatMulV2_83` | 0 | 92.560 |
| `MatMulV2_47` | 0 | 92.500 |
| `MatMulV2_89` | 0 | 92.380 |
| `MatMulV2_35` | 0 | 92.360 |
| `MatMulV2_53` | 0 | 92.340 |
| `MatMulV2_47` | 0 | 92.280 |
| `MatMulV2_161` | 0 | 92.160 |
| `MatMulV2_23` | 0 | 92.120 |
| `MatMulV2_101` | 0 | 92.100 |
| `MatMulV2_71` | 0 | 91.960 |
| `MatMulV2_113` | 0 | 91.960 |
| `MatMulV2_161` | 0 | 91.780 |
| `MatMulV2_95` | 0 | 91.420 |
| `MatMulV2_83` | 0 | 91.200 |
| `MatMulV2_143` | 0 | 90.700 |
| `MatMulV2_5` | 0 | 89.860 |
| `MatMulV2_107` | 0 | 89.840 |
| `MatMulV2_59` | 0 | 89.740 |
| `MatMulV2_95` | 0 | 89.620 |
| `MatMulV2_155` | 0 | 89.520 |
| `MatMulV2_41` | 0 | 88.860 |
| `MatMulV2_41` | 0 | 88.700 |
| `MatMulV2_125` | 0 | 88.560 |
| `MatMulV2_29` | 0 | 88.500 |
| `MatMulV2_53` | 0 | 88.340 |
| `MatMulV2_77` | 0 | 88.060 |
| `MatMulV2_29` | 0 | 87.960 |
| `MatMulV2_65` | 0 | 87.960 |
| `MatMulV2_11` | 0 | 87.760 |
| `MatMulV2_137` | 0 | 87.700 |
| `MatMulV2_161` | 0 | 87.240 |
| `MatMulV2_17` | 0 | 87.100 |
| `MatMulV2_23` | 0 | 87.080 |
| `MatMulV2_118` | 0 | 86.520 |
| `MatMulV2_130` | 0 | 86.360 |
| `MatMulV2_112` | 0 | 86.320 |
| `MatMulV2_76` | 0 | 86.280 |
| `MatMulV2_52` | 0 | 86.280 |
| `MatMulV2_64` | 0 | 86.220 |
| `MatMulV2_136` | 0 | 86.180 |
| `MatMulV2_118` | 0 | 86.120 |
| `MatMulV2_148` | 0 | 86.120 |
| `MatMulV2_76` | 0 | 86.100 |
| `MatMulV2_17` | 0 | 86.060 |
| `MatMulV2_148` | 0 | 86.020 |
| `MatMulV2_5` | 0 | 85.980 |
| `MatMulV2_76` | 0 | 85.980 |
| `MatMulV2_124` | 0 | 85.920 |
| `MatMulV2_154` | 0 | 85.920 |
| `MatMulV2_10` | 0 | 85.840 |
| `MatMulV2_10` | 0 | 85.820 |
| `MatMulV2_17` | 0 | 85.720 |
| `MatMulV2_124` | 0 | 85.720 |
| `MatMulV2_46` | 0 | 85.640 |
| `MatMulV2_58` | 0 | 85.620 |
| `MatMulV2_88` | 0 | 85.600 |
| `MatMulV2_130` | 0 | 85.580 |
| `MatMulV2_106` | 0 | 85.500 |
| `MatMulV2_22` | 0 | 85.420 |
| `MatMulV2_142` | 0 | 85.400 |
| `MatMulV2_70` | 0 | 85.380 |
| `MatMulV2_106` | 0 | 85.380 |
| `MatMulV2_130` | 0 | 85.320 |
| `MatMulV2_88` | 0 | 85.260 |
| `MatMulV2_64` | 0 | 85.240 |
| `MatMulV2_118` | 0 | 85.240 |
| `MatMulV2_34` | 0 | 85.200 |
| `MatMulV2_94` | 0 | 85.180 |
| `MatMulV2_28` | 0 | 85.100 |
| `MatMulV2_46` | 0 | 85.100 |
| `MatMulV2_82` | 0 | 85.100 |
| `MatMulV2_106` | 0 | 85.100 |
| `MatMulV2_100` | 0 | 85.080 |
| `MatMulV2_28` | 0 | 85.060 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 29363.840 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step1` | 1 | 27777.030 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step2` | 1 | 27431.200 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step3` | 1 | 27363.330 |
| `TorchDynamo Cache Lookup` | 3 | 26206.250 |
| `Torch-Compiled Region: 0/0` | 3 | 4137.930 |
| `TorchNpuGraphBase::Run` | 3 | 2987.970 |
| `RefreshAtTensorFromGeTensor` | 3 | 1235.270 |
| `aten::empty` | 3 | 588.730 |
| `ExecuteGraph` | 3 | 571.890 |
| `AssembleInputs` | 3 | 426.180 |
| `aten::set_` | 3 | 325.430 |
| `AssembleOutputs` | 3 | 313.660 |
| `empty_tensor` | 3 | 289.580 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 213485.790 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 76277.470 |
| `launch` | 868 | 14788.850 |
| `InputCopy` | 3 | 201.900 |
| `ModelExecute` | 3 | 51.130 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 43.770 |
| `step_info` | 6 | 30.380 |
| `OutputCopy` | 3 | 1.170 |

