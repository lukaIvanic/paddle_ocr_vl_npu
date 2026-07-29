# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/rope_joint_manual_profile_e12cfe8`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/rope_joint_manual_profile_e12cfe8/liteserver-c001-4_669582_20260729161631177_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `72942.960 us`
- `Free`: `2428.220 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2536.500 us`
- `Stage`: `75371.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 81 | 22765.040 |
| `MatMulV3` | 162 | 12753.760 |
| `MatMulV2` | 324 | 10138.080 |
| `StridedSliceD` | 162 | 7445.160 |
| `AddLayerNorm` | 162 | 4104.440 |
| `Transpose` | 243 | 3709.580 |
| `Gelu` | 81 | 3122.720 |
| `Mul` | 162 | 2422.780 |
| `ConcatV2D` | 162 | 1912.420 |
| `Add` | 81 | 1655.640 |
| `Unpack` | 81 | 1038.900 |
| `Cast` | 81 | 986.240 |
| `Neg` | 81 | 784.000 |
| `LayerNormV3` | 3 | 90.500 |
| `Data` | 3 | 13.700 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 3 | 861.140 |
| `PromptFlashAttention_13` | 3 | 860.940 |
| `PromptFlashAttention_18` | 3 | 859.460 |
| `PromptFlashAttention_3` | 3 | 854.960 |
| `PromptFlashAttention_20` | 3 | 852.340 |
| `PromptFlashAttention_17` | 3 | 850.960 |
| `PromptFlashAttention_19` | 3 | 850.700 |
| `PromptFlashAttention_23` | 3 | 849.240 |
| `PromptFlashAttention_15` | 3 | 847.500 |
| `PromptFlashAttention_4` | 3 | 846.420 |
| `PromptFlashAttention_6` | 3 | 846.020 |
| `PromptFlashAttention_5` | 3 | 844.780 |
| `PromptFlashAttention_11` | 3 | 844.020 |
| `PromptFlashAttention_12` | 3 | 843.880 |
| `PromptFlashAttention_1` | 3 | 841.120 |
| `PromptFlashAttention_16` | 3 | 840.720 |
| `PromptFlashAttention_2` | 3 | 840.540 |
| `PromptFlashAttention_10` | 3 | 840.300 |
| `PromptFlashAttention_21` | 3 | 836.660 |
| `PromptFlashAttention_26` | 3 | 835.580 |
| `PromptFlashAttention_14` | 3 | 833.980 |
| `PromptFlashAttention_25` | 3 | 833.560 |
| `PromptFlashAttention_8` | 3 | 832.720 |
| `PromptFlashAttention_7` | 3 | 831.580 |
| `PromptFlashAttention_22` | 3 | 830.940 |
| `PromptFlashAttention_24` | 3 | 827.620 |
| `PromptFlashAttention_9` | 3 | 827.360 |
| `MatMulV2_64_to_v3` | 3 | 260.280 |
| `MatMulV2_100_to_v3` | 3 | 260.160 |
| `MatMulV2_154_to_v3` | 3 | 259.980 |
| `MatMulV2_160_to_v3` | 3 | 259.900 |
| `MatMulV2_82_to_v3` | 3 | 259.800 |
| `MatMulV2_106_to_v3` | 3 | 259.800 |
| `MatMulV2_58_to_v3` | 3 | 259.480 |
| `MatMulV2_22_to_v3` | 3 | 258.940 |
| `MatMulV2_4_to_v3` | 3 | 258.880 |
| `MatMulV2_118_to_v3` | 3 | 258.740 |
| `MatMulV2_136_to_v3` | 3 | 258.660 |
| `MatMulV2_76_to_v3` | 3 | 258.600 |
| `MatMulV2_40_to_v3` | 3 | 258.540 |
| `MatMulV2_148_to_v3` | 3 | 258.420 |
| `MatMulV2_142_to_v3` | 3 | 258.160 |
| `MatMulV2_124_to_v3` | 3 | 258.060 |
| `MatMulV2_70_to_v3` | 3 | 258.040 |
| `MatMulV2_28_to_v3` | 3 | 258.000 |
| `MatMulV2_94_to_v3` | 3 | 257.880 |
| `MatMulV2_52_to_v3` | 3 | 257.800 |
| `MatMulV2_130_to_v3` | 3 | 257.660 |
| `MatMulV2_88_to_v3` | 3 | 257.440 |
| `MatMulV2_16_to_v3` | 3 | 257.140 |
| `MatMulV2_34_to_v3` | 3 | 256.720 |
| `MatMulV2_112_to_v3` | 3 | 255.940 |
| `MatMulV2_10_to_v3` | 3 | 255.720 |
| `MatMulV2_46_to_v3` | 3 | 255.100 |
| `MatMulV2_5_to_v3` | 3 | 218.480 |
| `MatMulV2_17_to_v3` | 3 | 217.220 |
| `MatMulV2_29_to_v3` | 3 | 216.500 |
| `MatMulV2_149_to_v3` | 3 | 215.640 |
| `MatMulV2_41_to_v3` | 3 | 215.400 |
| `MatMulV2_71_to_v3` | 3 | 215.320 |
| `MatMulV2_95_to_v3` | 3 | 215.020 |
| `MatMulV2_11_to_v3` | 3 | 214.560 |
| `MatMulV2_101_to_v3` | 3 | 214.420 |
| `MatMulV2_161_to_v3` | 3 | 214.320 |
| `MatMulV2_35_to_v3` | 3 | 214.220 |
| `MatMulV2_83_to_v3` | 3 | 214.020 |
| `MatMulV2_143_to_v3` | 3 | 214.000 |
| `MatMulV2_125_to_v3` | 3 | 213.920 |
| `MatMulV2_77_to_v3` | 3 | 213.900 |
| `MatMulV2_131_to_v3` | 3 | 213.800 |
| `MatMulV2_119_to_v3` | 3 | 213.740 |
| `MatMulV2_113_to_v3` | 3 | 213.400 |
| `MatMulV2_23_to_v3` | 3 | 213.280 |
| `MatMulV2_65_to_v3` | 3 | 213.240 |
| `MatMulV2_53_to_v3` | 3 | 213.200 |
| `MatMulV2_137_to_v3` | 3 | 212.980 |
| `MatMulV2_155_to_v3` | 3 | 212.140 |
| `MatMulV2_47_to_v3` | 3 | 212.060 |
| `MatMulV2_59_to_v3` | 3 | 212.060 |
| `MatMulV2_89_to_v3` | 3 | 211.680 |
| `MatMulV2_107_to_v3` | 3 | 211.400 |
| `StridedSliceV2_19` | 3 | 140.280 |
| `StridedSliceV2_13` | 3 | 140.240 |
| `StridedSliceV2_49` | 3 | 139.980 |
| `StridedSliceV2_31` | 3 | 139.720 |
| `StridedSliceV2_25` | 3 | 139.700 |
| `StridedSliceV2_43` | 3 | 139.600 |
| `StridedSliceV2_23` | 3 | 139.520 |
| `StridedSliceV2_3` | 3 | 139.400 |
| `StridedSliceV2_53` | 3 | 139.380 |
| `StridedSliceV2_33` | 3 | 139.360 |
| `StridedSliceV2_12` | 3 | 139.220 |
| `StridedSliceV2_11` | 3 | 139.200 |
| `StridedSliceV2_51` | 3 | 139.180 |
| `StridedSliceV2_35` | 3 | 138.980 |
| `StridedSliceV2_15` | 3 | 138.960 |
| `StridedSliceV2_21` | 3 | 138.900 |
| `StridedSliceV2_41` | 3 | 138.900 |
| `StridedSliceV2_39` | 3 | 138.680 |
| `StridedSliceV2_5` | 3 | 138.640 |
| `StridedSliceV2_34` | 3 | 138.620 |
| `StridedSliceV2_24` | 3 | 138.580 |
| `StridedSliceV2_47` | 3 | 138.560 |
| `StridedSliceV2_44` | 3 | 138.540 |
| `StridedSliceV2_7` | 3 | 138.480 |
| `StridedSliceV2_42` | 3 | 138.460 |
| `StridedSliceV2_29` | 3 | 138.400 |
| `StridedSliceV2_32` | 3 | 138.400 |
| `StridedSliceV2_10` | 3 | 138.240 |
| `StridedSliceV2_46` | 3 | 138.120 |
| `StridedSliceV2_2` | 3 | 138.100 |
| `StridedSliceV2_26` | 3 | 138.060 |
| `StridedSliceV2_36` | 3 | 138.020 |
| `StridedSliceV2_52` | 3 | 137.960 |
| `StridedSliceV2_4` | 3 | 137.920 |
| `StridedSliceV2_18` | 3 | 137.880 |
| `StridedSliceV2_6` | 3 | 137.860 |
| `StridedSliceV2_28` | 3 | 137.860 |
| `StridedSliceV2_14` | 3 | 137.600 |
| `StridedSliceV2_40` | 3 | 137.480 |
| `StridedSliceV2_20` | 3 | 137.420 |
| `StridedSliceV2_50` | 3 | 137.080 |
| `StridedSliceV2_17` | 3 | 136.820 |
| `StridedSliceV2_27` | 3 | 136.640 |
| `StridedSliceV2_45` | 3 | 136.220 |
| `StridedSliceV2_48` | 3 | 136.220 |
| `StridedSliceV2_16` | 3 | 135.980 |
| `StridedSliceV2_30` | 3 | 135.680 |
| `StridedSliceV2_38` | 3 | 135.540 |
| `StridedSliceV2_9` | 3 | 135.520 |
| `StridedSliceV2_37` | 3 | 135.040 |
| `StridedSliceV2_1` | 3 | 134.840 |
| `StridedSliceV2_8` | 3 | 133.820 |
| `StridedSliceV2` | 3 | 133.700 |
| `StridedSliceV2_22` | 3 | 133.660 |
| `Gelu_25` | 3 | 116.480 |
| `Gelu_5` | 3 | 116.420 |
| `Gelu` | 3 | 116.220 |
| `Gelu_14` | 3 | 116.180 |
| `Gelu_20` | 3 | 116.060 |
| `Gelu_13` | 3 | 115.960 |
| `Gelu_11` | 3 | 115.880 |
| `Gelu_10` | 3 | 115.860 |
| `Gelu_21` | 3 | 115.740 |
| `Gelu_26` | 3 | 115.700 |
| `Gelu_1` | 3 | 115.660 |
| `Gelu_18` | 3 | 115.640 |
| `Gelu_3` | 3 | 115.600 |
| `Gelu_19` | 3 | 115.560 |
| `Gelu_9` | 3 | 115.540 |
| `Gelu_15` | 3 | 115.540 |
| `Gelu_22` | 3 | 115.440 |
| `Gelu_23` | 3 | 115.440 |
| `Gelu_4` | 3 | 115.420 |
| `Gelu_6` | 3 | 115.400 |
| `Gelu_7` | 3 | 115.400 |
| `Gelu_16` | 3 | 115.400 |
| `Gelu_2` | 3 | 115.300 |
| `Gelu_8` | 3 | 115.240 |
| `Gelu_24` | 3 | 115.220 |
| `Gelu_12` | 3 | 115.220 |
| `Gelu_17` | 3 | 115.200 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 110.300 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 107.420 |
| `MatMulV2_6` | 3 | 104.520 |
| `MatMulV2_12` | 3 | 104.340 |
| `MatMulV2_120` | 3 | 104.300 |
| `MatMulV2_54` | 3 | 104.260 |
| `MatMulV2_18` | 3 | 104.180 |
| `MatMulV2_102` | 3 | 104.100 |
| `MatMulV2_126` | 3 | 104.100 |
| `MatMulV2_36` | 3 | 104.000 |
| `MatMulV2_150` | 3 | 103.960 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 103.900 |
| `MatMulV2_90` | 3 | 103.580 |
| `MatMulV2_78` | 3 | 103.560 |
| `MatMulV2_144` | 3 | 103.560 |
| `MatMulV2_156` | 3 | 103.480 |
| `MatMulV2_60` | 3 | 103.400 |
| `MatMulV2_96` | 3 | 103.360 |
| `MatMulV2_42` | 3 | 103.320 |
| `MatMulV2_72` | 3 | 103.160 |
| `MatMulV2_84` | 3 | 103.160 |
| `MatMulV2_24` | 3 | 103.140 |
| `MatMulV2_132` | 3 | 103.080 |
| `MatMulV2_108` | 3 | 103.020 |
| `MatMulV2_48` | 3 | 102.960 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 102.860 |
| `MatMulV2` | 3 | 102.720 |
| `MatMulV2_114` | 3 | 102.700 |
| `MatMulV2_66` | 3 | 102.700 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 102.680 |
| `MatMulV2_30` | 3 | 102.600 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 102.600 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 102.500 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 102.320 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 102.300 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 102.180 |
| `MatMulV2_138` | 3 | 102.140 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 101.740 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22765.040 |
| `MatMulV2 | "2048,1152;1280,1152;1280" -> "2048,1280" | ND;ND;ND -> ND` | 243 | 8119.180 |
| `StridedSliceD | "1,2048,32,80" -> "1,2048,32,40" | ND -> ND` | 162 | 7445.160 |
| `MatMulV3 | "2048,1152;4352,1152;4352" -> "2048,4352" | ND;ND;ND -> ND` | 81 | 6973.840 |
| `MatMulV3 | "2048,4352;1152,4352;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 5779.920 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4104.440 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3122.720 |
| `Mul | "1,2048,32,80;1,2048,1,80" -> "1,2048,32,80" | ND;ND -> ND` | 162 | 2422.780 |
| `MatMulV2 | "2048,1280;1152,1280;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 2018.900 |
| `Add | "1,2048,32,80;1,2048,32,80" -> "1,2048,32,80" | ND;ND -> ND` | 81 | 1655.640 |
| `Transpose | "2048,2,16,80;4" -> "2,16,2048,80" | ND;ND -> ND` | 81 | 1410.480 |
| `Transpose | "2048,16,80;3" -> "16,2048,80" | ND;ND -> ND` | 81 | 1283.980 |
| `ConcatV2D | "1,2048,32,40;1,2048,32,40" -> "1,2048,32,80" | ND;ND -> ND` | 81 | 1141.100 |
| `Unpack | "2,1,16,2048,80" -> "1,16,2048,80;1,16,2048,80" | ND -> ND;ND` | 81 | 1038.900 |
| `Transpose | "16,2048,80;3" -> "2048,16,80" | ND;ND -> ND` | 81 | 1015.120 |
| `Cast | "1,2048,32,80" -> "1,2048,32,80" | ND -> ND` | 81 | 986.240 |
| `Neg | "1,2048,32,40" -> "1,2048,32,40" | ND -> ND` | 81 | 784.000 |
| `ConcatV2D | "1,2048,1280;1,2048,1280" -> "1,2048,2560" | ND;ND -> ND` | 81 | 771.320 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 90.500 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 13.700 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 26869.480 |
| `ND;ND;ND` | 489 | 22982.340 |
| `ND` | 486 | 13377.020 |
| `ND;ND` | 648 | 9700.420 |
| `N/A` | 3 | 13.700 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_18` | 0 | 287.720 |
| `PromptFlashAttention_13` | 0 | 287.660 |
| `PromptFlashAttention` | 0 | 287.240 |
| `PromptFlashAttention` | 0 | 287.220 |
| `PromptFlashAttention_13` | 0 | 286.900 |
| `PromptFlashAttention` | 0 | 286.680 |
| `PromptFlashAttention_13` | 0 | 286.380 |
| `PromptFlashAttention_18` | 0 | 286.120 |
| `PromptFlashAttention_18` | 0 | 285.620 |
| `PromptFlashAttention_20` | 0 | 285.220 |
| `PromptFlashAttention_3` | 0 | 285.200 |
| `PromptFlashAttention_3` | 0 | 284.880 |
| `PromptFlashAttention_3` | 0 | 284.880 |
| `PromptFlashAttention_17` | 0 | 284.820 |
| `PromptFlashAttention_19` | 0 | 284.140 |
| `PromptFlashAttention_20` | 0 | 283.940 |
| `PromptFlashAttention_17` | 0 | 283.840 |
| `PromptFlashAttention_23` | 0 | 283.700 |
| `PromptFlashAttention_19` | 0 | 283.440 |
| `PromptFlashAttention_20` | 0 | 283.180 |
| `PromptFlashAttention_19` | 0 | 283.120 |
| `PromptFlashAttention_4` | 0 | 282.920 |
| `PromptFlashAttention_6` | 0 | 282.880 |
| `PromptFlashAttention_23` | 0 | 282.880 |
| `PromptFlashAttention_23` | 0 | 282.660 |
| `PromptFlashAttention_15` | 0 | 282.660 |
| `PromptFlashAttention_5` | 0 | 282.580 |
| `PromptFlashAttention_15` | 0 | 282.480 |
| `PromptFlashAttention_15` | 0 | 282.360 |
| `PromptFlashAttention_17` | 0 | 282.300 |
| `PromptFlashAttention_4` | 0 | 282.260 |
| `PromptFlashAttention_11` | 0 | 282.200 |
| `PromptFlashAttention_12` | 0 | 282.120 |
| `PromptFlashAttention_5` | 0 | 281.660 |
| `PromptFlashAttention_6` | 0 | 281.660 |
| `PromptFlashAttention_12` | 0 | 281.640 |
| `PromptFlashAttention_6` | 0 | 281.480 |
| `PromptFlashAttention_4` | 0 | 281.240 |
| `PromptFlashAttention_16` | 0 | 281.220 |
| `PromptFlashAttention_2` | 0 | 281.160 |
| `PromptFlashAttention_11` | 0 | 281.140 |
| `PromptFlashAttention_1` | 0 | 281.080 |
| `PromptFlashAttention_1` | 0 | 280.760 |
| `PromptFlashAttention_11` | 0 | 280.680 |
| `PromptFlashAttention_5` | 0 | 280.540 |
| `PromptFlashAttention_10` | 0 | 280.420 |
| `PromptFlashAttention_14` | 0 | 280.260 |
| `PromptFlashAttention_12` | 0 | 280.120 |
| `PromptFlashAttention_10` | 0 | 280.120 |
| `PromptFlashAttention_16` | 0 | 279.820 |
| `PromptFlashAttention_10` | 0 | 279.760 |
| `PromptFlashAttention_2` | 0 | 279.700 |
| `PromptFlashAttention_2` | 0 | 279.680 |
| `PromptFlashAttention_16` | 0 | 279.680 |
| `PromptFlashAttention_1` | 0 | 279.280 |
| `PromptFlashAttention_26` | 0 | 279.180 |
| `PromptFlashAttention_21` | 0 | 279.120 |
| `PromptFlashAttention_21` | 0 | 278.800 |
| `PromptFlashAttention_21` | 0 | 278.740 |
| `PromptFlashAttention_26` | 0 | 278.660 |
| `PromptFlashAttention_25` | 0 | 278.020 |
| `PromptFlashAttention_25` | 0 | 278.020 |
| `PromptFlashAttention_22` | 0 | 278.000 |
| `PromptFlashAttention_8` | 0 | 277.980 |
| `PromptFlashAttention_7` | 0 | 277.940 |
| `PromptFlashAttention_7` | 0 | 277.880 |
| `PromptFlashAttention_26` | 0 | 277.740 |
| `PromptFlashAttention_14` | 0 | 277.620 |
| `PromptFlashAttention_25` | 0 | 277.520 |
| `PromptFlashAttention_8` | 0 | 277.400 |
| `PromptFlashAttention_22` | 0 | 277.400 |
| `PromptFlashAttention_8` | 0 | 277.340 |
| `PromptFlashAttention_9` | 0 | 276.860 |
| `PromptFlashAttention_9` | 0 | 276.680 |
| `PromptFlashAttention_24` | 0 | 276.560 |
| `PromptFlashAttention_14` | 0 | 276.100 |
| `PromptFlashAttention_24` | 0 | 276.000 |
| `PromptFlashAttention_7` | 0 | 275.760 |
| `PromptFlashAttention_22` | 0 | 275.540 |
| `PromptFlashAttention_24` | 0 | 275.060 |
| `PromptFlashAttention_9` | 0 | 273.820 |
| `MatMulV2_22_to_v3` | 0 | 88.460 |
| `MatMulV2_64_to_v3` | 0 | 88.400 |
| `MatMulV2_100_to_v3` | 0 | 87.440 |
| `MatMulV2_118_to_v3` | 0 | 87.420 |
| `MatMulV2_154_to_v3` | 0 | 87.340 |
| `MatMulV2_160_to_v3` | 0 | 87.300 |
| `MatMulV2_130_to_v3` | 0 | 87.180 |
| `MatMulV2_82_to_v3` | 0 | 87.120 |
| `MatMulV2_4_to_v3` | 0 | 87.100 |
| `MatMulV2_76_to_v3` | 0 | 87.080 |
| `MatMulV2_10_to_v3` | 0 | 87.060 |
| `MatMulV2_106_to_v3` | 0 | 87.040 |
| `MatMulV2_70_to_v3` | 0 | 86.940 |
| `MatMulV2_142_to_v3` | 0 | 86.920 |
| `MatMulV2_28_to_v3` | 0 | 86.840 |
| `MatMulV2_40_to_v3` | 0 | 86.760 |
| `MatMulV2_100_to_v3` | 0 | 86.700 |
| `MatMulV2_136_to_v3` | 0 | 86.620 |
| `MatMulV2_52_to_v3` | 0 | 86.600 |
| `MatMulV2_58_to_v3` | 0 | 86.600 |
| `MatMulV2_58_to_v3` | 0 | 86.480 |
| `MatMulV2_154_to_v3` | 0 | 86.480 |
| `MatMulV2_106_to_v3` | 0 | 86.460 |
| `MatMulV2_40_to_v3` | 0 | 86.460 |
| `MatMulV2_34_to_v3` | 0 | 86.440 |
| `MatMulV2_148_to_v3` | 0 | 86.420 |
| `MatMulV2_76_to_v3` | 0 | 86.420 |
| `MatMulV2_58_to_v3` | 0 | 86.400 |
| `MatMulV2_82_to_v3` | 0 | 86.360 |
| `MatMulV2_94_to_v3` | 0 | 86.360 |
| `MatMulV2_88_to_v3` | 0 | 86.320 |
| `MatMulV2_160_to_v3` | 0 | 86.320 |
| `MatMulV2_82_to_v3` | 0 | 86.320 |
| `MatMulV2_106_to_v3` | 0 | 86.300 |
| `MatMulV2_160_to_v3` | 0 | 86.280 |
| `MatMulV2_118_to_v3` | 0 | 86.260 |
| `MatMulV2_124_to_v3` | 0 | 86.240 |
| `MatMulV2_94_to_v3` | 0 | 86.220 |
| `MatMulV2_16_to_v3` | 0 | 86.220 |
| `MatMulV2_88_to_v3` | 0 | 86.220 |
| `MatMulV2_46_to_v3` | 0 | 86.200 |
| `MatMulV2_4_to_v3` | 0 | 86.180 |
| `MatMulV2_136_to_v3` | 0 | 86.160 |
| `MatMulV2_154_to_v3` | 0 | 86.160 |
| `MatMulV2_112_to_v3` | 0 | 86.080 |
| `MatMulV2_148_to_v3` | 0 | 86.060 |
| `MatMulV2_100_to_v3` | 0 | 86.020 |
| `MatMulV2_124_to_v3` | 0 | 85.980 |
| `MatMulV2_64_to_v3` | 0 | 85.980 |
| `MatMulV2_142_to_v3` | 0 | 85.980 |
| `MatMulV2_148_to_v3` | 0 | 85.940 |
| `MatMulV2_64_to_v3` | 0 | 85.900 |
| `MatMulV2_136_to_v3` | 0 | 85.880 |
| `MatMulV2_124_to_v3` | 0 | 85.840 |
| `MatMulV2_70_to_v3` | 0 | 85.840 |
| `MatMulV2_22_to_v3` | 0 | 85.760 |
| `MatMulV2_28_to_v3` | 0 | 85.760 |
| `MatMulV2_52_to_v3` | 0 | 85.620 |
| `MatMulV2_112_to_v3` | 0 | 85.600 |
| `MatMulV2_4_to_v3` | 0 | 85.600 |
| `MatMulV2_52_to_v3` | 0 | 85.580 |
| `MatMulV2_130_to_v3` | 0 | 85.540 |
| `MatMulV2_16_to_v3` | 0 | 85.480 |
| `MatMulV2_16_to_v3` | 0 | 85.440 |
| `MatMulV2_28_to_v3` | 0 | 85.400 |
| `MatMulV2_40_to_v3` | 0 | 85.320 |
| `MatMulV2_94_to_v3` | 0 | 85.300 |
| `MatMulV2_70_to_v3` | 0 | 85.260 |
| `MatMulV2_142_to_v3` | 0 | 85.260 |
| `MatMulV2_34_to_v3` | 0 | 85.200 |
| `MatMulV2_76_to_v3` | 0 | 85.100 |
| `MatMulV2_34_to_v3` | 0 | 85.080 |
| `MatMulV2_118_to_v3` | 0 | 85.060 |
| `MatMulV2_130_to_v3` | 0 | 84.940 |
| `MatMulV2_88_to_v3` | 0 | 84.900 |
| `MatMulV2_22_to_v3` | 0 | 84.720 |
| `MatMulV2_46_to_v3` | 0 | 84.720 |
| `MatMulV2_10_to_v3` | 0 | 84.460 |
| `MatMulV2_112_to_v3` | 0 | 84.260 |
| `MatMulV2_10_to_v3` | 0 | 84.200 |
| `MatMulV2_46_to_v3` | 0 | 84.180 |
| `MatMulV2_29_to_v3` | 0 | 74.060 |
| `MatMulV2_17_to_v3` | 0 | 73.500 |
| `MatMulV2_41_to_v3` | 0 | 73.360 |
| `MatMulV2_5_to_v3` | 0 | 73.300 |
| `MatMulV2_149_to_v3` | 0 | 72.840 |
| `MatMulV2_5_to_v3` | 0 | 72.660 |
| `MatMulV2_5_to_v3` | 0 | 72.520 |
| `MatMulV2_11_to_v3` | 0 | 72.220 |
| `MatMulV2_71_to_v3` | 0 | 72.120 |
| `MatMulV2_71_to_v3` | 0 | 72.100 |
| `MatMulV2_95_to_v3` | 0 | 72.100 |
| `MatMulV2_35_to_v3` | 0 | 72.020 |
| `MatMulV2_17_to_v3` | 0 | 72.020 |
| `MatMulV2_143_to_v3` | 0 | 72.020 |
| `MatMulV2_101_to_v3` | 0 | 71.820 |
| `MatMulV2_161_to_v3` | 0 | 71.760 |
| `MatMulV2_11_to_v3` | 0 | 71.720 |
| `MatMulV2_65_to_v3` | 0 | 71.700 |
| `MatMulV2_83_to_v3` | 0 | 71.700 |
| `MatMulV2_17_to_v3` | 0 | 71.700 |
| `MatMulV2_125_to_v3` | 0 | 71.680 |
| `MatMulV2_77_to_v3` | 0 | 71.660 |
| `MatMulV2_161_to_v3` | 0 | 71.600 |
| `MatMulV2_131_to_v3` | 0 | 71.560 |
| `MatMulV2_119_to_v3` | 0 | 71.520 |
| `MatMulV2_95_to_v3` | 0 | 71.500 |
| `MatMulV2_23_to_v3` | 0 | 71.460 |
| `MatMulV2_131_to_v3` | 0 | 71.460 |
| `MatMulV2_53_to_v3` | 0 | 71.440 |
| `MatMulV2_113_to_v3` | 0 | 71.440 |
| `MatMulV2_143_to_v3` | 0 | 71.440 |
| `MatMulV2_23_to_v3` | 0 | 71.420 |
| `MatMulV2_149_to_v3` | 0 | 71.420 |
| `MatMulV2_95_to_v3` | 0 | 71.420 |
| `MatMulV2_101_to_v3` | 0 | 71.400 |
| `MatMulV2_149_to_v3` | 0 | 71.380 |
| `MatMulV2_35_to_v3` | 0 | 71.360 |
| `MatMulV2_65_to_v3` | 0 | 71.360 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 26553.210 |
| `TorchNpuGraphBase::Run` | 3 | 25914.500 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.joint_manual.torchair.active.step1` | 1 | 25502.270 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.joint_manual.torchair.active.step2` | 1 | 25249.370 |
| `paddleocr_vl.vision_matmul_lab.B1.S2048.I4352.native.weights.joint_manual.torchair.active.step3` | 1 | 25203.680 |
| `AssembleInputs` | 3 | 24545.990 |
| `RefreshAtTensorFromGeTensor` | 3 | 1048.070 |
| `aten::empty` | 3 | 500.920 |
| `ExecuteGraph` | 3 | 457.290 |
| `aten::set_` | 3 | 276.440 |
| `AssembleOutputs` | 3 | 265.380 |
| `empty_tensor` | 3 | 247.210 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 164262.800 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 71465.360 |
| `launch` | 625 | 11485.410 |
| `InputCopy` | 3 | 151.370 |
| `ModelExecute` | 3 | 42.390 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 36.280 |
| `step_info` | 6 | 14.070 |
| `OutputCopy` | 3 | 1.020 |
