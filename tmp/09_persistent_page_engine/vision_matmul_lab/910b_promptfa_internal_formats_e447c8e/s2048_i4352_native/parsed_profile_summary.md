# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4352_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4352_native/liteserver-c001-4_636796_20260729135042626_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `90278.680 us`
- `Free`: `3475.920 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3351.000 us`
- `Stage`: `93754.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 81 | 22900.540 |
| `MatMulV3` | 162 | 12721.500 |
| `StridedSliceD` | 405 | 11588.140 |
| `Transpose` | 324 | 10906.320 |
| `MatMulV2` | 324 | 7916.020 |
| `PadV3` | 243 | 4333.900 |
| `AddLayerNorm` | 162 | 4156.040 |
| `ConcatV2D` | 243 | 3545.720 |
| `Gelu` | 81 | 3038.860 |
| `Mul` | 324 | 2740.160 |
| `Cast` | 162 | 1681.400 |
| `Add` | 162 | 1666.900 |
| `Neg` | 162 | 1455.540 |
| `SplitVD` | 81 | 1284.440 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 238.440 |
| `LayerNormV3` | 3 | 89.920 |
| `Data` | 3 | 14.840 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 3 | 879.940 |
| `PromptFlashAttention_9` | 3 | 873.540 |
| `PromptFlashAttention_16` | 3 | 873.320 |
| `PromptFlashAttention_24` | 3 | 871.660 |
| `PromptFlashAttention_8` | 3 | 862.420 |
| `PromptFlashAttention_23` | 3 | 861.720 |
| `PromptFlashAttention_14` | 3 | 853.300 |
| `PromptFlashAttention` | 3 | 853.060 |
| `PromptFlashAttention_12` | 3 | 849.720 |
| `PromptFlashAttention_26` | 3 | 849.720 |
| `PromptFlashAttention_25` | 3 | 847.560 |
| `PromptFlashAttention_2` | 3 | 845.820 |
| `PromptFlashAttention_1` | 3 | 845.680 |
| `PromptFlashAttention_13` | 3 | 844.760 |
| `PromptFlashAttention_22` | 3 | 844.620 |
| `PromptFlashAttention_7` | 3 | 843.440 |
| `PromptFlashAttention_11` | 3 | 840.920 |
| `PromptFlashAttention_21` | 3 | 839.480 |
| `PromptFlashAttention_6` | 3 | 838.940 |
| `PromptFlashAttention_19` | 3 | 837.860 |
| `PromptFlashAttention_17` | 3 | 837.640 |
| `PromptFlashAttention_5` | 3 | 837.540 |
| `PromptFlashAttention_18` | 3 | 836.160 |
| `PromptFlashAttention_4` | 3 | 834.540 |
| `PromptFlashAttention_20` | 3 | 834.020 |
| `PromptFlashAttention_3` | 3 | 832.520 |
| `PromptFlashAttention_10` | 3 | 830.640 |
| `MatMulV2_28_to_v3` | 3 | 268.900 |
| `MatMulV2_148_to_v3` | 3 | 259.620 |
| `MatMulV2_118_to_v3` | 3 | 257.940 |
| `MatMulV2_22_to_v3` | 3 | 257.820 |
| `MatMulV2_58_to_v3` | 3 | 257.140 |
| `MatMulV2_4_to_v3` | 3 | 256.520 |
| `MatMulV2_142_to_v3` | 3 | 256.060 |
| `MatMulV2_88_to_v3` | 3 | 255.880 |
| `MatMulV2_52_to_v3` | 3 | 255.800 |
| `MatMulV2_82_to_v3` | 3 | 255.460 |
| `MatMulV2_100_to_v3` | 3 | 255.040 |
| `MatMulV2_46_to_v3` | 3 | 254.600 |
| `MatMulV2_10_to_v3` | 3 | 254.020 |
| `MatMulV2_70_to_v3` | 3 | 253.860 |
| `MatMulV2_76_to_v3` | 3 | 253.020 |
| `MatMulV2_106_to_v3` | 3 | 252.920 |
| `MatMulV2_130_to_v3` | 3 | 252.860 |
| `MatMulV2_136_to_v3` | 3 | 252.620 |
| `MatMulV2_112_to_v3` | 3 | 252.480 |
| `MatMulV2_94_to_v3` | 3 | 252.200 |
| `MatMulV2_160_to_v3` | 3 | 252.140 |
| `MatMulV2_16_to_v3` | 3 | 251.880 |
| `MatMulV2_34_to_v3` | 3 | 251.540 |
| `MatMulV2_40_to_v3` | 3 | 251.480 |
| `MatMulV2_64_to_v3` | 3 | 251.000 |
| `MatMulV2_154_to_v3` | 3 | 251.000 |
| `MatMulV2_124_to_v3` | 3 | 249.160 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 238.440 |
| `MatMulV2_17_to_v3` | 3 | 218.060 |
| `MatMulV2_131_to_v3` | 3 | 217.820 |
| `MatMulV2_77_to_v3` | 3 | 217.780 |
| `MatMulV2_71_to_v3` | 3 | 217.700 |
| `MatMulV2_59_to_v3` | 3 | 217.600 |
| `MatMulV2_95_to_v3` | 3 | 217.500 |
| `MatMulV2_143_to_v3` | 3 | 217.380 |
| `MatMulV2_149_to_v3` | 3 | 217.320 |
| `MatMulV2_125_to_v3` | 3 | 217.160 |
| `MatMulV2_89_to_v3` | 3 | 217.020 |
| `MatMulV2_119_to_v3` | 3 | 216.600 |
| `MatMulV2_161_to_v3` | 3 | 216.600 |
| `MatMulV2_53_to_v3` | 3 | 216.500 |
| `MatMulV2_83_to_v3` | 3 | 216.440 |
| `MatMulV2_23_to_v3` | 3 | 216.340 |
| `MatMulV2_29_to_v3` | 3 | 216.340 |
| `MatMulV2_47_to_v3` | 3 | 216.260 |
| `MatMulV2_113_to_v3` | 3 | 216.140 |
| `MatMulV2_41_to_v3` | 3 | 216.060 |
| `MatMulV2_11_to_v3` | 3 | 216.020 |
| `MatMulV2_5_to_v3` | 3 | 215.980 |
| `MatMulV2_137_to_v3` | 3 | 215.940 |
| `MatMulV2_155_to_v3` | 3 | 215.940 |
| `MatMulV2_101_to_v3` | 3 | 215.880 |
| `MatMulV2_35_to_v3` | 3 | 215.860 |
| `MatMulV2_65_to_v3` | 3 | 215.460 |
| `MatMulV2_107_to_v3` | 3 | 214.840 |
| `Transpose_226` | 3 | 128.760 |
| `Transpose_156` | 3 | 118.000 |
| `Transpose_16` | 3 | 117.800 |
| `Transpose_96` | 3 | 117.700 |
| `Transpose_196` | 3 | 117.600 |
| `Transpose_266` | 3 | 117.360 |
| `Transpose_246` | 3 | 116.880 |
| `Transpose_236` | 3 | 116.840 |
| `Transpose_126` | 3 | 116.720 |
| `Transpose_166` | 3 | 116.700 |
| `Transpose_136` | 3 | 116.660 |
| `Transpose_106` | 3 | 116.560 |
| `Transpose_206` | 3 | 116.380 |
| `Transpose_36` | 3 | 116.340 |
| `Transpose_66` | 3 | 116.340 |
| `Transpose_46` | 3 | 116.300 |
| `Transpose_216` | 3 | 116.280 |
| `Transpose_56` | 3 | 116.240 |
| `Transpose_86` | 3 | 116.200 |
| `Transpose_146` | 3 | 116.060 |
| `Transpose_256` | 3 | 116.060 |
| `Transpose_176` | 3 | 116.040 |
| `Transpose_116` | 3 | 115.920 |
| `Transpose_186` | 3 | 115.480 |
| `Transpose_76` | 3 | 115.280 |
| `Transpose_26` | 3 | 115.140 |
| `Transpose_6` | 3 | 114.920 |
| `Gelu_19` | 3 | 113.480 |
| `Gelu_10` | 3 | 113.420 |
| `Gelu_14` | 3 | 113.400 |
| `Gelu_23` | 3 | 112.960 |
| `Gelu_24` | 3 | 112.840 |
| `Gelu_1` | 3 | 112.820 |
| `Gelu_2` | 3 | 112.740 |
| `Gelu_5` | 3 | 112.700 |
| `Gelu_6` | 3 | 112.580 |
| `Gelu_4` | 3 | 112.580 |
| `Gelu_20` | 3 | 112.480 |
| `Transpose_264` | 3 | 112.460 |
| `Gelu_26` | 3 | 112.460 |
| `Gelu_16` | 3 | 112.420 |
| `Gelu` | 3 | 112.400 |
| `Gelu_11` | 3 | 112.400 |
| `Gelu_25` | 3 | 112.320 |
| `Gelu_7` | 3 | 112.320 |
| `Gelu_8` | 3 | 112.320 |
| `Gelu_9` | 3 | 112.320 |
| `Gelu_15` | 3 | 112.300 |
| `Gelu_18` | 3 | 112.300 |
| `Gelu_13` | 3 | 112.300 |
| `Gelu_3` | 3 | 112.280 |
| `Gelu_21` | 3 | 112.240 |
| `Gelu_17` | 3 | 112.180 |
| `Gelu_22` | 3 | 112.180 |
| `Gelu_12` | 3 | 112.120 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 111.620 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 104.440 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 104.360 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 104.000 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 103.940 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 103.600 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 103.600 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 103.240 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 102.980 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 102.880 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 102.820 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 102.780 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 102.780 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 102.780 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 102.660 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 102.440 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 102.400 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 102.360 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 102.160 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 102.000 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 101.960 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 101.920 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 101.820 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 101.800 |
| `StridedSliceV2_18` | 3 | 101.560 |
| `StridedSliceV2_46` | 3 | 101.320 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 101.200 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 100.520 |
| `StridedSliceV2_16` | 3 | 100.400 |
| `StridedSliceV2_69` | 3 | 100.120 |
| `StridedSliceV2_9` | 3 | 99.920 |
| `StridedSliceV2_54` | 3 | 99.860 |
| `StridedSliceV2_84` | 3 | 99.740 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 99.460 |
| `StridedSliceV2_99` | 3 | 99.360 |
| `StridedSliceV2_109` | 3 | 99.100 |
| `StridedSliceV2_49` | 3 | 99.020 |
| `StridedSliceV2_105` | 3 | 98.640 |
| `StridedSliceV2_119` | 3 | 98.560 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 98.520 |
| `StridedSliceV2_24` | 3 | 98.440 |
| `StridedSliceV2_89` | 3 | 98.360 |
| `StridedSliceV2_39` | 3 | 98.260 |
| `Transpose_255` | 3 | 98.160 |
| `StridedSliceV2_104` | 3 | 97.960 |
| `StridedSliceV2_124` | 3 | 97.900 |
| `Transpose_165` | 3 | 97.860 |
| `Transpose_205` | 3 | 97.860 |
| `Transpose_45` | 3 | 97.800 |
| `Transpose_3` | 3 | 97.740 |
| `StridedSliceV2_19` | 3 | 97.720 |
| `Transpose_95` | 3 | 97.700 |
| `StridedSliceV2_29` | 3 | 97.680 |
| `StridedSliceV2_102` | 3 | 97.680 |
| `Transpose_265` | 3 | 97.600 |
| `Transpose_105` | 3 | 97.580 |
| `StridedSliceV2_14` | 3 | 97.560 |
| `Transpose_215` | 3 | 97.520 |
| `StridedSliceV2_64` | 3 | 97.380 |
| `StridedSliceV2_4` | 3 | 97.360 |
| `Transpose_145` | 3 | 97.360 |
| `Transpose_55` | 3 | 97.340 |
| `Transpose_245` | 3 | 97.220 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22900.540 |
| `StridedSliceD | "1,2048,16,72" -> "1,2048,16,36" | ND -> ND` | 324 | 8943.200 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7916.020 |
| `Transpose | "2048,16,72;3" -> "16,2048,72" | ND;ND -> ND` | 243 | 7749.760 |
| `MatMulV3 | "2048,1152;4352,1152;4352" -> "2048,4352" | ND;ND;ND -> ND` | 81 | 6872.960 |
| `MatMulV3 | "2048,4352;1152,4352;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 5848.540 |
| `PadV3 | "1,16,2048,72;8;" -> "1,16,2048,80" | ND;ND;ND -> ND` | 243 | 4333.900 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4156.040 |
| `Transpose | "16,2048,72;3" -> "2048,16,72" | ND;ND -> ND` | 81 | 3156.560 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3038.860 |
| `Mul | "1,2048,16,72;1,2048,1,72" -> "1,2048,16,72" | ND;ND -> ND` | 324 | 2740.160 |
| `ConcatV2D | "1,2048,16,36;1,2048,16,36" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 2652.180 |
| `StridedSliceD | "1,16,2048,80" -> "1,16,2048,72" | ND -> ND` | 81 | 2644.940 |
| `Cast | "1,2048,16,72" -> "1,2048,16,72" | ND -> ND` | 162 | 1681.400 |
| `Add | "1,2048,16,72;1,2048,16,72" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 1666.900 |
| `Neg | "1,2048,16,36" -> "1,2048,16,36" | ND -> ND` | 162 | 1455.540 |
| `SplitVD | "1,2048,3456" -> "1,2048,1152;1,2048,1152;1,2048,1152" | ND -> ND;ND;ND` | 81 | 1284.440 |
| `ConcatV2D | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,3456" | ND;ND;ND -> ND` | 81 | 893.540 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 238.440 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 89.920 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.840 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 27056.580 |
| `ND;ND;ND` | 813 | 25954.880 |
| `ND` | 891 | 19048.380 |
| `ND;ND` | 972 | 17965.560 |
| `N/A` | 246 | 253.280 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 0 | 294.560 |
| `PromptFlashAttention_15` | 0 | 292.960 |
| `PromptFlashAttention_9` | 0 | 292.560 |
| `PromptFlashAttention_15` | 0 | 292.420 |
| `PromptFlashAttention_16` | 0 | 292.180 |
| `PromptFlashAttention_16` | 0 | 291.960 |
| `PromptFlashAttention_24` | 0 | 291.220 |
| `PromptFlashAttention_9` | 0 | 290.920 |
| `PromptFlashAttention_24` | 0 | 290.480 |
| `PromptFlashAttention_9` | 0 | 290.060 |
| `PromptFlashAttention_24` | 0 | 289.960 |
| `PromptFlashAttention_16` | 0 | 289.180 |
| `PromptFlashAttention_8` | 0 | 288.320 |
| `PromptFlashAttention_23` | 0 | 287.640 |
| `PromptFlashAttention_23` | 0 | 287.560 |
| `PromptFlashAttention_8` | 0 | 287.380 |
| `PromptFlashAttention_8` | 0 | 286.720 |
| `PromptFlashAttention_23` | 0 | 286.520 |
| `PromptFlashAttention` | 0 | 286.220 |
| `PromptFlashAttention_12` | 0 | 285.400 |
| `PromptFlashAttention_14` | 0 | 285.340 |
| `PromptFlashAttention_14` | 0 | 285.060 |
| `PromptFlashAttention_12` | 0 | 284.920 |
| `PromptFlashAttention` | 0 | 284.100 |
| `PromptFlashAttention_26` | 0 | 283.480 |
| `PromptFlashAttention_2` | 0 | 283.440 |
| `PromptFlashAttention_13` | 0 | 283.340 |
| `PromptFlashAttention_26` | 0 | 283.260 |
| `PromptFlashAttention_1` | 0 | 283.080 |
| `PromptFlashAttention_25` | 0 | 283.080 |
| `PromptFlashAttention_26` | 0 | 282.980 |
| `PromptFlashAttention_14` | 0 | 282.900 |
| `PromptFlashAttention` | 0 | 282.740 |
| `PromptFlashAttention_25` | 0 | 282.520 |
| `PromptFlashAttention_22` | 0 | 282.240 |
| `PromptFlashAttention_25` | 0 | 281.960 |
| `PromptFlashAttention_13` | 0 | 281.800 |
| `PromptFlashAttention_7` | 0 | 281.800 |
| `PromptFlashAttention_7` | 0 | 281.700 |
| `PromptFlashAttention_1` | 0 | 281.620 |
| `PromptFlashAttention_11` | 0 | 281.540 |
| `PromptFlashAttention_2` | 0 | 281.540 |
| `PromptFlashAttention_22` | 0 | 281.200 |
| `PromptFlashAttention_22` | 0 | 281.180 |
| `PromptFlashAttention_6` | 0 | 281.060 |
| `PromptFlashAttention_1` | 0 | 280.980 |
| `PromptFlashAttention_2` | 0 | 280.840 |
| `PromptFlashAttention_17` | 0 | 280.700 |
| `PromptFlashAttention_21` | 0 | 280.620 |
| `PromptFlashAttention_11` | 0 | 280.360 |
| `PromptFlashAttention_21` | 0 | 280.340 |
| `PromptFlashAttention_5` | 0 | 280.040 |
| `PromptFlashAttention_19` | 0 | 279.980 |
| `PromptFlashAttention_7` | 0 | 279.940 |
| `PromptFlashAttention_13` | 0 | 279.620 |
| `PromptFlashAttention_12` | 0 | 279.400 |
| `PromptFlashAttention_18` | 0 | 279.300 |
| `PromptFlashAttention_19` | 0 | 279.280 |
| `PromptFlashAttention_6` | 0 | 279.280 |
| `PromptFlashAttention_17` | 0 | 279.160 |
| `PromptFlashAttention_5` | 0 | 279.100 |
| `PromptFlashAttention_11` | 0 | 279.020 |
| `PromptFlashAttention_20` | 0 | 278.720 |
| `PromptFlashAttention_18` | 0 | 278.640 |
| `PromptFlashAttention_6` | 0 | 278.600 |
| `PromptFlashAttention_19` | 0 | 278.600 |
| `PromptFlashAttention_21` | 0 | 278.520 |
| `PromptFlashAttention_5` | 0 | 278.400 |
| `PromptFlashAttention_4` | 0 | 278.360 |
| `PromptFlashAttention_18` | 0 | 278.220 |
| `PromptFlashAttention_4` | 0 | 278.100 |
| `PromptFlashAttention_4` | 0 | 278.080 |
| `PromptFlashAttention_20` | 0 | 278.060 |
| `PromptFlashAttention_3` | 0 | 278.040 |
| `PromptFlashAttention_3` | 0 | 277.940 |
| `PromptFlashAttention_17` | 0 | 277.780 |
| `PromptFlashAttention_10` | 0 | 277.620 |
| `PromptFlashAttention_10` | 0 | 277.500 |
| `PromptFlashAttention_20` | 0 | 277.240 |
| `PromptFlashAttention_3` | 0 | 276.540 |
| `PromptFlashAttention_10` | 0 | 275.520 |
| `MatMulV2_28_to_v3` | 0 | 90.260 |
| `MatMulV2_28_to_v3` | 0 | 89.400 |
| `MatMulV2_28_to_v3` | 0 | 89.240 |
| `MatMulV2_148_to_v3` | 0 | 87.540 |
| `MatMulV2_22_to_v3` | 0 | 87.060 |
| `MatMulV2_118_to_v3` | 0 | 86.860 |
| `MatMulV2_148_to_v3` | 0 | 86.700 |
| `MatMulV2_82_to_v3` | 0 | 86.620 |
| `MatMulV2_58_to_v3` | 0 | 86.360 |
| `MatMulV2_88_to_v3` | 0 | 86.160 |
| `MatMulV2_52_to_v3` | 0 | 86.120 |
| `MatMulV2_10_to_v3` | 0 | 86.080 |
| `MatMulV2_4_to_v3` | 0 | 86.040 |
| `MatMulV2_22_to_v3` | 0 | 86.040 |
| `MatMulV2_100_to_v3` | 0 | 85.800 |
| `MatMulV2_142_to_v3` | 0 | 85.780 |
| `MatMulV2_88_to_v3` | 0 | 85.640 |
| `MatMulV2_58_to_v3` | 0 | 85.640 |
| `MatMulV2_118_to_v3` | 0 | 85.560 |
| `MatMulV2_118_to_v3` | 0 | 85.520 |
| `MatMulV2_148_to_v3` | 0 | 85.380 |
| `MatMulV2_142_to_v3` | 0 | 85.360 |
| `MatMulV2_46_to_v3` | 0 | 85.320 |
| `MatMulV2_52_to_v3` | 0 | 85.300 |
| `MatMulV2_70_to_v3` | 0 | 85.260 |
| `MatMulV2_4_to_v3` | 0 | 85.260 |
| `MatMulV2_4_to_v3` | 0 | 85.220 |
| `MatMulV2_160_to_v3` | 0 | 85.160 |
| `MatMulV2_58_to_v3` | 0 | 85.140 |
| `MatMulV2_46_to_v3` | 0 | 84.980 |
| `MatMulV2_142_to_v3` | 0 | 84.920 |
| `MatMulV2_136_to_v3` | 0 | 84.860 |
| `MatMulV2_16_to_v3` | 0 | 84.780 |
| `MatMulV2_22_to_v3` | 0 | 84.720 |
| `MatMulV2_106_to_v3` | 0 | 84.720 |
| `MatMulV2_100_to_v3` | 0 | 84.700 |
| `MatMulV2_82_to_v3` | 0 | 84.680 |
| `MatMulV2_76_to_v3` | 0 | 84.620 |
| `MatMulV2_94_to_v3` | 0 | 84.540 |
| `MatMulV2_100_to_v3` | 0 | 84.540 |
| `MatMulV2_70_to_v3` | 0 | 84.540 |
| `MatMulV2_130_to_v3` | 0 | 84.500 |
| `MatMulV2_34_to_v3` | 0 | 84.500 |
| `MatMulV2_112_to_v3` | 0 | 84.500 |
| `MatMulV2_76_to_v3` | 0 | 84.480 |
| `MatMulV2_52_to_v3` | 0 | 84.380 |
| `MatMulV2_112_to_v3` | 0 | 84.340 |
| `MatMulV2_46_to_v3` | 0 | 84.300 |
| `MatMulV2_64_to_v3` | 0 | 84.280 |
| `MatMulV2_94_to_v3` | 0 | 84.220 |
| `MatMulV2_130_to_v3` | 0 | 84.220 |
| `MatMulV2_82_to_v3` | 0 | 84.160 |
| `MatMulV2_130_to_v3` | 0 | 84.140 |
| `MatMulV2_10_to_v3` | 0 | 84.120 |
| `MatMulV2_40_to_v3` | 0 | 84.120 |
| `MatMulV2_106_to_v3` | 0 | 84.100 |
| `MatMulV2_106_to_v3` | 0 | 84.100 |
| `MatMulV2_136_to_v3` | 0 | 84.080 |
| `MatMulV2_88_to_v3` | 0 | 84.080 |
| `MatMulV2_70_to_v3` | 0 | 84.060 |
| `MatMulV2_76_to_v3` | 0 | 83.920 |
| `MatMulV2_10_to_v3` | 0 | 83.820 |
| `MatMulV2_40_to_v3` | 0 | 83.720 |
| `MatMulV2_154_to_v3` | 0 | 83.720 |
| `MatMulV2_136_to_v3` | 0 | 83.680 |
| `MatMulV2_112_to_v3` | 0 | 83.640 |
| `MatMulV2_154_to_v3` | 0 | 83.640 |
| `MatMulV2_16_to_v3` | 0 | 83.640 |
| `MatMulV2_154_to_v3` | 0 | 83.640 |
| `MatMulV2_40_to_v3` | 0 | 83.640 |
| `MatMulV2_64_to_v3` | 0 | 83.600 |
| `MatMulV2_34_to_v3` | 0 | 83.560 |
| `MatMulV2_160_to_v3` | 0 | 83.540 |
| `MatMulV2_34_to_v3` | 0 | 83.480 |
| `MatMulV2_16_to_v3` | 0 | 83.460 |
| `MatMulV2_94_to_v3` | 0 | 83.440 |
| `MatMulV2_160_to_v3` | 0 | 83.440 |
| `MatMulV2_124_to_v3` | 0 | 83.180 |
| `MatMulV2_64_to_v3` | 0 | 83.120 |
| `MatMulV2_124_to_v3` | 0 | 83.080 |
| `MatMulV2_124_to_v3` | 0 | 82.900 |
| `MatMulV2_17_to_v3` | 0 | 73.820 |
| `MatMulV2_29_to_v3` | 0 | 73.820 |
| `MatMulV2_71_to_v3` | 0 | 73.320 |
| `MatMulV2_95_to_v3` | 0 | 73.300 |
| `MatMulV2_5_to_v3` | 0 | 73.280 |
| `MatMulV2_125_to_v3` | 0 | 73.020 |
| `MatMulV2_119_to_v3` | 0 | 73.000 |
| `MatMulV2_161_to_v3` | 0 | 72.980 |
| `MatMulV2_149_to_v3` | 0 | 72.900 |
| `MatMulV2_143_to_v3` | 0 | 72.840 |
| `MatMulV2_77_to_v3` | 0 | 72.820 |
| `MatMulV2_131_to_v3` | 0 | 72.760 |
| `MatMulV2_59_to_v3` | 0 | 72.740 |
| `MatMulV2_137_to_v3` | 0 | 72.680 |
| `MatMulV2_77_to_v3` | 0 | 72.640 |
| `MatMulV2_11_to_v3` | 0 | 72.640 |
| `MatMulV2_53_to_v3` | 0 | 72.580 |
| `MatMulV2_95_to_v3` | 0 | 72.580 |
| `MatMulV2_131_to_v3` | 0 | 72.580 |
| `MatMulV2_89_to_v3` | 0 | 72.560 |
| `MatMulV2_47_to_v3` | 0 | 72.540 |
| `MatMulV2_23_to_v3` | 0 | 72.500 |
| `MatMulV2_113_to_v3` | 0 | 72.480 |
| `MatMulV2_131_to_v3` | 0 | 72.480 |
| `MatMulV2_59_to_v3` | 0 | 72.440 |
| `MatMulV2_143_to_v3` | 0 | 72.420 |
| `MatMulV2_59_to_v3` | 0 | 72.420 |
| `MatMulV2_149_to_v3` | 0 | 72.400 |
| `MatMulV2_83_to_v3` | 0 | 72.400 |
| `MatMulV2_101_to_v3` | 0 | 72.400 |
| `MatMulV2_125_to_v3` | 0 | 72.380 |
| `MatMulV2_35_to_v3` | 0 | 72.380 |
| `MatMulV2_89_to_v3` | 0 | 72.360 |
| `MatMulV2_77_to_v3` | 0 | 72.320 |
| `MatMulV2_65_to_v3` | 0 | 72.240 |
| `MatMulV2_155_to_v3` | 0 | 72.240 |
| `MatMulV2_161_to_v3` | 0 | 72.240 |
| `MatMulV2_71_to_v3` | 0 | 72.200 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 33019.130 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step1` | 1 | 31810.640 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step3` | 1 | 31485.070 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.native.torchair.active.step2` | 1 | 31438.550 |
| `TorchDynamo Cache Lookup` | 3 | 30191.990 |
| `Torch-Compiled Region: 0/0` | 3 | 3621.000 |
| `TorchNpuGraphBase::Run` | 3 | 2662.560 |
| `RefreshAtTensorFromGeTensor` | 3 | 1121.910 |
| `aten::empty` | 3 | 550.930 |
| `ExecuteGraph` | 3 | 495.920 |
| `AssembleInputs` | 3 | 383.650 |
| `AssembleOutputs` | 3 | 284.120 |
| `aten::set_` | 3 | 274.320 |
| `empty_tensor` | 3 | 267.320 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 222375.530 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 89027.280 |
| `launch` | 976 | 18637.930 |
| `InputCopy` | 3 | 152.700 |
| `ModelExecute` | 3 | 51.280 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 27.520 |
| `step_info` | 6 | 16.280 |
| `OutputCopy` | 3 | 0.720 |

