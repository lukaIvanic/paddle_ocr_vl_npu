# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4304_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4304_native/liteserver-c001-4_633530_20260729134820534_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `94141.920 us`
- `Free`: `3770.360 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3492.750 us`
- `Stage`: `97912.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 81 | 22872.100 |
| `MatMulV3` | 162 | 16064.620 |
| `StridedSliceD` | 405 | 11402.160 |
| `Transpose` | 324 | 11212.840 |
| `MatMulV2` | 324 | 7787.820 |
| `AddLayerNorm` | 162 | 4629.420 |
| `PadV3` | 243 | 3979.260 |
| `ConcatV2D` | 243 | 3690.120 |
| `Gelu` | 81 | 3082.440 |
| `Mul` | 324 | 2826.400 |
| `Add` | 162 | 1803.420 |
| `Cast` | 162 | 1640.820 |
| `Neg` | 162 | 1514.520 |
| `SplitVD` | 81 | 1297.700 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 236.180 |
| `LayerNormV3` | 3 | 87.380 |
| `Data` | 3 | 14.720 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 3 | 878.940 |
| `PromptFlashAttention_9` | 3 | 873.680 |
| `PromptFlashAttention_16` | 3 | 866.520 |
| `PromptFlashAttention_8` | 3 | 865.200 |
| `PromptFlashAttention_26` | 3 | 865.040 |
| `PromptFlashAttention_24` | 3 | 861.800 |
| `PromptFlashAttention_23` | 3 | 858.940 |
| `PromptFlashAttention` | 3 | 851.620 |
| `PromptFlashAttention_13` | 3 | 851.100 |
| `PromptFlashAttention_14` | 3 | 850.660 |
| `PromptFlashAttention_7` | 3 | 848.040 |
| `PromptFlashAttention_1` | 3 | 847.780 |
| `PromptFlashAttention_22` | 3 | 845.220 |
| `PromptFlashAttention_12` | 3 | 844.900 |
| `PromptFlashAttention_25` | 3 | 844.780 |
| `PromptFlashAttention_21` | 3 | 840.500 |
| `PromptFlashAttention_18` | 3 | 839.720 |
| `PromptFlashAttention_6` | 3 | 837.920 |
| `PromptFlashAttention_10` | 3 | 837.400 |
| `PromptFlashAttention_11` | 3 | 836.620 |
| `PromptFlashAttention_4` | 3 | 835.740 |
| `PromptFlashAttention_2` | 3 | 833.920 |
| `PromptFlashAttention_3` | 3 | 833.900 |
| `PromptFlashAttention_17` | 3 | 833.560 |
| `PromptFlashAttention_5` | 3 | 833.220 |
| `PromptFlashAttention_19` | 3 | 828.000 |
| `PromptFlashAttention_20` | 3 | 827.380 |
| `MatMulV2_155_to_v3` | 3 | 306.120 |
| `MatMulV2_161_to_v3` | 3 | 305.660 |
| `MatMulV2_28_to_v3` | 3 | 304.320 |
| `MatMulV2_53_to_v3` | 3 | 303.780 |
| `MatMulV2_149_to_v3` | 3 | 303.660 |
| `MatMulV2_29_to_v3` | 3 | 301.580 |
| `MatMulV2_16_to_v3` | 3 | 301.200 |
| `MatMulV2_5_to_v3` | 3 | 300.920 |
| `MatMulV2_143_to_v3` | 3 | 300.440 |
| `MatMulV2_142_to_v3` | 3 | 299.920 |
| `MatMulV2_101_to_v3` | 3 | 299.500 |
| `MatMulV2_35_to_v3` | 3 | 299.440 |
| `MatMulV2_119_to_v3` | 3 | 299.440 |
| `MatMulV2_112_to_v3` | 3 | 299.380 |
| `MatMulV2_100_to_v3` | 3 | 299.280 |
| `MatMulV2_46_to_v3` | 3 | 299.220 |
| `MatMulV2_58_to_v3` | 3 | 299.140 |
| `MatMulV2_83_to_v3` | 3 | 299.120 |
| `MatMulV2_11_to_v3` | 3 | 298.880 |
| `MatMulV2_125_to_v3` | 3 | 298.860 |
| `MatMulV2_52_to_v3` | 3 | 298.780 |
| `MatMulV2_82_to_v3` | 3 | 297.860 |
| `MatMulV2_95_to_v3` | 3 | 297.820 |
| `MatMulV2_76_to_v3` | 3 | 297.700 |
| `MatMulV2_23_to_v3` | 3 | 297.680 |
| `MatMulV2_113_to_v3` | 3 | 297.600 |
| `MatMulV2_154_to_v3` | 3 | 297.580 |
| `MatMulV2_59_to_v3` | 3 | 297.480 |
| `MatMulV2_137_to_v3` | 3 | 297.480 |
| `MatMulV2_77_to_v3` | 3 | 297.440 |
| `MatMulV2_106_to_v3` | 3 | 297.440 |
| `MatMulV2_160_to_v3` | 3 | 297.360 |
| `MatMulV2_88_to_v3` | 3 | 297.060 |
| `MatMulV2_131_to_v3` | 3 | 296.820 |
| `MatMulV2_47_to_v3` | 3 | 296.760 |
| `MatMulV2_130_to_v3` | 3 | 296.680 |
| `MatMulV2_70_to_v3` | 3 | 296.600 |
| `MatMulV2_107_to_v3` | 3 | 296.140 |
| `MatMulV2_124_to_v3` | 3 | 296.080 |
| `MatMulV2_71_to_v3` | 3 | 295.780 |
| `MatMulV2_17_to_v3` | 3 | 295.760 |
| `MatMulV2_41_to_v3` | 3 | 295.620 |
| `MatMulV2_40_to_v3` | 3 | 295.540 |
| `MatMulV2_89_to_v3` | 3 | 295.360 |
| `MatMulV2_4_to_v3` | 3 | 295.220 |
| `MatMulV2_136_to_v3` | 3 | 294.580 |
| `MatMulV2_65_to_v3` | 3 | 294.540 |
| `MatMulV2_22_to_v3` | 3 | 294.060 |
| `MatMulV2_148_to_v3` | 3 | 291.660 |
| `MatMulV2_64_to_v3` | 3 | 291.540 |
| `MatMulV2_10_to_v3` | 3 | 290.320 |
| `MatMulV2_118_to_v3` | 3 | 289.580 |
| `MatMulV2_94_to_v3` | 3 | 288.960 |
| `MatMulV2_34_to_v3` | 3 | 287.880 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 236.180 |
| `Transpose_206` | 3 | 135.500 |
| `Transpose_146` | 3 | 134.700 |
| `Gelu_8` | 3 | 124.660 |
| `Gelu_2` | 3 | 124.340 |
| `Transpose_136` | 3 | 122.440 |
| `Transpose_26` | 3 | 122.080 |
| `Transpose_96` | 3 | 121.820 |
| `Transpose_76` | 3 | 121.780 |
| `Transpose_196` | 3 | 121.680 |
| `Transpose_126` | 3 | 121.000 |
| `Transpose_176` | 3 | 120.840 |
| `Transpose_246` | 3 | 120.680 |
| `Transpose_16` | 3 | 120.520 |
| `Transpose_86` | 3 | 120.520 |
| `Transpose_116` | 3 | 120.520 |
| `Transpose_166` | 3 | 120.480 |
| `Transpose_266` | 3 | 120.360 |
| `Transpose_46` | 3 | 120.320 |
| `Transpose_236` | 3 | 120.220 |
| `Transpose_216` | 3 | 120.120 |
| `Transpose_63` | 3 | 120.000 |
| `Transpose_56` | 3 | 119.940 |
| `Transpose_156` | 3 | 119.820 |
| `Transpose_256` | 3 | 119.660 |
| `Transpose_66` | 3 | 119.240 |
| `Transpose_36` | 3 | 119.220 |
| `Transpose_226` | 3 | 119.160 |
| `Transpose_186` | 3 | 118.400 |
| `Transpose_6` | 3 | 117.820 |
| `Transpose_106` | 3 | 117.540 |
| `Transpose_65` | 3 | 115.740 |
| `Transpose_3` | 3 | 115.580 |
| `Gelu_6` | 3 | 114.420 |
| `Gelu_18` | 3 | 114.280 |
| `Gelu_14` | 3 | 114.200 |
| `Gelu_10` | 3 | 114.100 |
| `Gelu_22` | 3 | 113.880 |
| `Gelu_26` | 3 | 113.800 |
| `Gelu_3` | 3 | 113.780 |
| `Gelu` | 3 | 113.700 |
| `Gelu_24` | 3 | 113.220 |
| `Gelu_9` | 3 | 113.160 |
| `Gelu_25` | 3 | 113.120 |
| `Gelu_12` | 3 | 113.080 |
| `Gelu_17` | 3 | 113.060 |
| `Gelu_5` | 3 | 113.040 |
| `Gelu_15` | 3 | 113.040 |
| `Gelu_20` | 3 | 113.020 |
| `Gelu_23` | 3 | 113.020 |
| `Gelu_11` | 3 | 113.000 |
| `Gelu_1` | 3 | 112.980 |
| `Gelu_7` | 3 | 112.980 |
| `Gelu_19` | 3 | 112.980 |
| `Gelu_16` | 3 | 112.960 |
| `Gelu_21` | 3 | 112.940 |
| `Gelu_4` | 3 | 112.900 |
| `Gelu_13` | 3 | 112.780 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 106.360 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 106.260 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 106.020 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 105.940 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 105.860 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 105.600 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 105.580 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 105.520 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 105.400 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 105.140 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 105.140 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 105.080 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 105.020 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 105.020 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 104.820 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 104.780 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 104.480 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 104.180 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 104.160 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 103.960 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 103.780 |
| `StridedSliceV2_119` | 3 | 103.600 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 103.560 |
| `StridedSliceV2_108` | 3 | 103.260 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 103.000 |
| `StridedSliceV2_78` | 3 | 102.760 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 102.120 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 101.700 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 101.300 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 100.340 |
| `Transpose_135` | 3 | 100.240 |
| `Transpose_125` | 3 | 99.960 |
| `Transpose_15` | 3 | 99.840 |
| `Transpose_175` | 3 | 99.540 |
| `Transpose_225` | 3 | 99.520 |
| `Transpose_75` | 3 | 99.440 |
| `Transpose_185` | 3 | 99.360 |
| `Transpose_235` | 3 | 99.260 |
| `Transpose_165` | 3 | 99.240 |
| `Transpose_25` | 3 | 99.220 |
| `Transpose_115` | 3 | 99.200 |
| `Transpose_95` | 3 | 99.080 |
| `Transpose_255` | 3 | 99.040 |
| `Transpose_245` | 3 | 99.020 |
| `Transpose_265` | 3 | 98.880 |
| `Transpose_105` | 3 | 98.820 |
| `Transpose_45` | 3 | 98.780 |
| `Transpose_205` | 3 | 98.780 |
| `Transpose_215` | 3 | 98.780 |
| `Transpose_85` | 3 | 98.760 |
| `Transpose_145` | 3 | 98.600 |
| `Transpose_195` | 3 | 98.460 |
| `Transpose_55` | 3 | 98.320 |
| `Transpose_35` | 3 | 98.260 |
| `Transpose_155` | 3 | 98.160 |
| `Transpose_223` | 3 | 98.040 |
| `Transpose_163` | 3 | 97.860 |
| `Transpose_64` | 3 | 97.800 |
| `Transpose_53` | 3 | 97.600 |
| `Transpose_114` | 3 | 97.580 |
| `Transpose_264` | 3 | 97.560 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22872.100 |
| `StridedSliceD | "1,2048,16,72" -> "1,2048,16,36" | ND -> ND` | 324 | 8897.420 |
| `MatMulV3 | "2048,4304;1152,4304;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 8069.680 |
| `MatMulV3 | "2048,1152;4304,1152;4304" -> "2048,4304" | ND;ND;ND -> ND` | 81 | 7994.940 |
| `Transpose | "2048,16,72;3" -> "16,2048,72" | ND;ND -> ND` | 243 | 7936.460 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7787.820 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4629.420 |
| `PadV3 | "1,16,2048,72;8;" -> "1,16,2048,80" | ND;ND;ND -> ND` | 243 | 3979.260 |
| `Transpose | "16,2048,72;3" -> "2048,16,72" | ND;ND -> ND` | 81 | 3276.380 |
| `Gelu | "1,2048,4304" -> "1,2048,4304" | ND -> ND` | 81 | 3082.440 |
| `Mul | "1,2048,16,72;1,2048,1,72" -> "1,2048,16,72" | ND;ND -> ND` | 324 | 2826.400 |
| `ConcatV2D | "1,2048,16,36;1,2048,16,36" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 2799.100 |
| `StridedSliceD | "1,16,2048,80" -> "1,16,2048,72" | ND -> ND` | 81 | 2504.740 |
| `Add | "1,2048,16,72;1,2048,16,72" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 1803.420 |
| `Cast | "1,2048,16,72" -> "1,2048,16,72" | ND -> ND` | 162 | 1640.820 |
| `Neg | "1,2048,16,36" -> "1,2048,16,36" | ND -> ND` | 162 | 1514.520 |
| `SplitVD | "1,2048,3456" -> "1,2048,1152;1,2048,1152;1,2048,1152" | ND -> ND;ND;ND` | 81 | 1297.700 |
| `ConcatV2D | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,3456" | ND;ND;ND -> ND` | 81 | 891.020 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 236.180 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 87.380 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.720 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 813 | 28810.100 |
| `ND;ND;ND;ND` | 243 | 27501.520 |
| `ND` | 891 | 18937.640 |
| `ND;ND` | 972 | 18641.760 |
| `N/A` | 246 | 250.900 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 0 | 293.800 |
| `PromptFlashAttention_15` | 0 | 293.360 |
| `PromptFlashAttention_9` | 0 | 292.940 |
| `PromptFlashAttention_9` | 0 | 292.260 |
| `PromptFlashAttention_15` | 0 | 291.780 |
| `PromptFlashAttention_16` | 0 | 290.900 |
| `PromptFlashAttention_26` | 0 | 289.700 |
| `PromptFlashAttention_8` | 0 | 289.000 |
| `PromptFlashAttention_16` | 0 | 288.820 |
| `PromptFlashAttention_23` | 0 | 288.720 |
| `PromptFlashAttention_9` | 0 | 288.480 |
| `PromptFlashAttention_26` | 0 | 288.360 |
| `PromptFlashAttention_8` | 0 | 288.220 |
| `PromptFlashAttention_8` | 0 | 287.980 |
| `PromptFlashAttention_24` | 0 | 287.820 |
| `PromptFlashAttention_24` | 0 | 287.740 |
| `PromptFlashAttention_26` | 0 | 286.980 |
| `PromptFlashAttention_16` | 0 | 286.800 |
| `PromptFlashAttention_24` | 0 | 286.240 |
| `PromptFlashAttention_23` | 0 | 285.580 |
| `PromptFlashAttention_13` | 0 | 285.400 |
| `PromptFlashAttention_14` | 0 | 285.060 |
| `PromptFlashAttention_23` | 0 | 284.640 |
| `PromptFlashAttention_13` | 0 | 284.400 |
| `PromptFlashAttention` | 0 | 284.040 |
| `PromptFlashAttention` | 0 | 283.800 |
| `PromptFlashAttention` | 0 | 283.780 |
| `PromptFlashAttention_25` | 0 | 283.720 |
| `PromptFlashAttention_1` | 0 | 283.500 |
| `PromptFlashAttention_14` | 0 | 283.380 |
| `PromptFlashAttention_7` | 0 | 283.340 |
| `PromptFlashAttention_22` | 0 | 282.640 |
| `PromptFlashAttention_6` | 0 | 282.620 |
| `PromptFlashAttention_7` | 0 | 282.560 |
| `PromptFlashAttention_12` | 0 | 282.480 |
| `PromptFlashAttention_1` | 0 | 282.440 |
| `PromptFlashAttention_14` | 0 | 282.220 |
| `PromptFlashAttention_7` | 0 | 282.140 |
| `PromptFlashAttention_1` | 0 | 281.840 |
| `PromptFlashAttention_22` | 0 | 281.660 |
| `PromptFlashAttention_18` | 0 | 281.480 |
| `PromptFlashAttention_13` | 0 | 281.300 |
| `PromptFlashAttention_12` | 0 | 281.260 |
| `PromptFlashAttention_21` | 0 | 281.200 |
| `PromptFlashAttention_12` | 0 | 281.160 |
| `PromptFlashAttention_25` | 0 | 281.140 |
| `PromptFlashAttention_22` | 0 | 280.920 |
| `PromptFlashAttention_10` | 0 | 280.820 |
| `PromptFlashAttention_21` | 0 | 280.320 |
| `PromptFlashAttention_3` | 0 | 279.980 |
| `PromptFlashAttention_25` | 0 | 279.920 |
| `PromptFlashAttention_5` | 0 | 279.500 |
| `PromptFlashAttention_18` | 0 | 279.340 |
| `PromptFlashAttention_17` | 0 | 279.100 |
| `PromptFlashAttention_10` | 0 | 279.080 |
| `PromptFlashAttention_11` | 0 | 279.040 |
| `PromptFlashAttention_4` | 0 | 279.020 |
| `PromptFlashAttention_11` | 0 | 279.020 |
| `PromptFlashAttention_21` | 0 | 278.980 |
| `PromptFlashAttention_18` | 0 | 278.900 |
| `PromptFlashAttention_2` | 0 | 278.740 |
| `PromptFlashAttention_17` | 0 | 278.700 |
| `PromptFlashAttention_4` | 0 | 278.580 |
| `PromptFlashAttention_11` | 0 | 278.560 |
| `PromptFlashAttention_6` | 0 | 278.420 |
| `PromptFlashAttention_5` | 0 | 278.240 |
| `PromptFlashAttention_4` | 0 | 278.140 |
| `PromptFlashAttention_3` | 0 | 277.740 |
| `PromptFlashAttention_2` | 0 | 277.600 |
| `PromptFlashAttention_2` | 0 | 277.580 |
| `PromptFlashAttention_10` | 0 | 277.500 |
| `PromptFlashAttention_19` | 0 | 277.020 |
| `PromptFlashAttention_6` | 0 | 276.880 |
| `PromptFlashAttention_20` | 0 | 276.780 |
| `PromptFlashAttention_19` | 0 | 276.300 |
| `PromptFlashAttention_3` | 0 | 276.180 |
| `PromptFlashAttention_20` | 0 | 276.120 |
| `PromptFlashAttention_17` | 0 | 275.760 |
| `PromptFlashAttention_5` | 0 | 275.480 |
| `PromptFlashAttention_19` | 0 | 274.680 |
| `PromptFlashAttention_20` | 0 | 274.480 |
| `MatMulV2_28_to_v3` | 0 | 102.360 |
| `MatMulV2_155_to_v3` | 0 | 102.160 |
| `MatMulV2_155_to_v3` | 0 | 102.140 |
| `MatMulV2_161_to_v3` | 0 | 101.940 |
| `MatMulV2_161_to_v3` | 0 | 101.860 |
| `MatMulV2_28_to_v3` | 0 | 101.860 |
| `MatMulV2_161_to_v3` | 0 | 101.860 |
| `MatMulV2_155_to_v3` | 0 | 101.820 |
| `MatMulV2_149_to_v3` | 0 | 101.640 |
| `MatMulV2_53_to_v3` | 0 | 101.540 |
| `MatMulV2_59_to_v3` | 0 | 101.420 |
| `MatMulV2_16_to_v3` | 0 | 101.380 |
| `MatMulV2_53_to_v3` | 0 | 101.280 |
| `MatMulV2_16_to_v3` | 0 | 101.100 |
| `MatMulV2_149_to_v3` | 0 | 101.060 |
| `MatMulV2_53_to_v3` | 0 | 100.960 |
| `MatMulV2_149_to_v3` | 0 | 100.960 |
| `MatMulV2_143_to_v3` | 0 | 100.920 |
| `MatMulV2_29_to_v3` | 0 | 100.740 |
| `MatMulV2_101_to_v3` | 0 | 100.640 |
| `MatMulV2_29_to_v3` | 0 | 100.540 |
| `MatMulV2_11_to_v3` | 0 | 100.520 |
| `MatMulV2_82_to_v3` | 0 | 100.520 |
| `MatMulV2_5_to_v3` | 0 | 100.500 |
| `MatMulV2_5_to_v3` | 0 | 100.480 |
| `MatMulV2_142_to_v3` | 0 | 100.440 |
| `MatMulV2_143_to_v3` | 0 | 100.380 |
| `MatMulV2_83_to_v3` | 0 | 100.380 |
| `MatMulV2_119_to_v3` | 0 | 100.300 |
| `MatMulV2_29_to_v3` | 0 | 100.300 |
| `MatMulV2_35_to_v3` | 0 | 100.300 |
| `MatMulV2_125_to_v3` | 0 | 100.280 |
| `MatMulV2_113_to_v3` | 0 | 100.260 |
| `MatMulV2_112_to_v3` | 0 | 100.180 |
| `MatMulV2_119_to_v3` | 0 | 100.140 |
| `MatMulV2_76_to_v3` | 0 | 100.120 |
| `MatMulV2_28_to_v3` | 0 | 100.100 |
| `MatMulV2_35_to_v3` | 0 | 100.040 |
| `MatMulV2_58_to_v3` | 0 | 100.020 |
| `MatMulV2_46_to_v3` | 0 | 100.000 |
| `MatMulV2_112_to_v3` | 0 | 99.980 |
| `MatMulV2_5_to_v3` | 0 | 99.940 |
| `MatMulV2_100_to_v3` | 0 | 99.920 |
| `MatMulV2_52_to_v3` | 0 | 99.900 |
| `MatMulV2_46_to_v3` | 0 | 99.880 |
| `MatMulV2_52_to_v3` | 0 | 99.860 |
| `MatMulV2_125_to_v3` | 0 | 99.860 |
| `MatMulV2_47_to_v3` | 0 | 99.800 |
| `MatMulV2_154_to_v3` | 0 | 99.780 |
| `MatMulV2_58_to_v3` | 0 | 99.760 |
| `MatMulV2_131_to_v3` | 0 | 99.760 |
| `MatMulV2_142_to_v3` | 0 | 99.740 |
| `MatMulV2_142_to_v3` | 0 | 99.740 |
| `MatMulV2_101_to_v3` | 0 | 99.720 |
| `MatMulV2_100_to_v3` | 0 | 99.700 |
| `MatMulV2_130_to_v3` | 0 | 99.700 |
| `MatMulV2_100_to_v3` | 0 | 99.660 |
| `MatMulV2_83_to_v3` | 0 | 99.660 |
| `MatMulV2_23_to_v3` | 0 | 99.600 |
| `MatMulV2_106_to_v3` | 0 | 99.560 |
| `MatMulV2_107_to_v3` | 0 | 99.540 |
| `MatMulV2_77_to_v3` | 0 | 99.520 |
| `MatMulV2_11_to_v3` | 0 | 99.480 |
| `MatMulV2_95_to_v3` | 0 | 99.460 |
| `MatMulV2_88_to_v3` | 0 | 99.420 |
| `MatMulV2_77_to_v3` | 0 | 99.400 |
| `MatMulV2_58_to_v3` | 0 | 99.360 |
| `MatMulV2_71_to_v3` | 0 | 99.360 |
| `MatMulV2_46_to_v3` | 0 | 99.340 |
| `MatMulV2_124_to_v3` | 0 | 99.260 |
| `MatMulV2_137_to_v3` | 0 | 99.260 |
| `MatMulV2_160_to_v3` | 0 | 99.260 |
| `MatMulV2_70_to_v3` | 0 | 99.240 |
| `MatMulV2_112_to_v3` | 0 | 99.220 |
| `MatMulV2_160_to_v3` | 0 | 99.200 |
| `MatMulV2_95_to_v3` | 0 | 99.180 |
| `MatMulV2_95_to_v3` | 0 | 99.180 |
| `MatMulV2_113_to_v3` | 0 | 99.180 |
| `MatMulV2_65_to_v3` | 0 | 99.160 |
| `MatMulV2_70_to_v3` | 0 | 99.160 |
| `MatMulV2_4_to_v3` | 0 | 99.160 |
| `MatMulV2_101_to_v3` | 0 | 99.140 |
| `MatMulV2_137_to_v3` | 0 | 99.140 |
| `MatMulV2_143_to_v3` | 0 | 99.140 |
| `MatMulV2_23_to_v3` | 0 | 99.140 |
| `MatMulV2_154_to_v3` | 0 | 99.120 |
| `MatMulV2_35_to_v3` | 0 | 99.100 |
| `MatMulV2_83_to_v3` | 0 | 99.080 |
| `MatMulV2_137_to_v3` | 0 | 99.080 |
| `MatMulV2_106_to_v3` | 0 | 99.060 |
| `MatMulV2_52_to_v3` | 0 | 99.020 |
| `MatMulV2_119_to_v3` | 0 | 99.000 |
| `MatMulV2_82_to_v3` | 0 | 98.960 |
| `MatMulV2_23_to_v3` | 0 | 98.940 |
| `MatMulV2_17_to_v3` | 0 | 98.920 |
| `MatMulV2_41_to_v3` | 0 | 98.920 |
| `MatMulV2_88_to_v3` | 0 | 98.900 |
| `MatMulV2_160_to_v3` | 0 | 98.900 |
| `MatMulV2_40_to_v3` | 0 | 98.880 |
| `MatMulV2_11_to_v3` | 0 | 98.880 |
| `MatMulV2_89_to_v3` | 0 | 98.840 |
| `MatMulV2_76_to_v3` | 0 | 98.820 |
| `MatMulV2_106_to_v3` | 0 | 98.820 |
| `MatMulV2_131_to_v3` | 0 | 98.800 |
| `MatMulV2_76_to_v3` | 0 | 98.760 |
| `MatMulV2_136_to_v3` | 0 | 98.740 |
| `MatMulV2_88_to_v3` | 0 | 98.740 |
| `MatMulV2_16_to_v3` | 0 | 98.720 |
| `MatMulV2_125_to_v3` | 0 | 98.720 |
| `MatMulV2_130_to_v3` | 0 | 98.720 |
| `MatMulV2_47_to_v3` | 0 | 98.680 |
| `MatMulV2_154_to_v3` | 0 | 98.680 |
| `MatMulV2_71_to_v3` | 0 | 98.660 |
| `MatMulV2_77_to_v3` | 0 | 98.520 |
| `MatMulV2_17_to_v3` | 0 | 98.500 |
| `MatMulV2_124_to_v3` | 0 | 98.480 |
| `MatMulV2_40_to_v3` | 0 | 98.460 |
| `MatMulV2_22_to_v3` | 0 | 98.440 |
| `MatMulV2_41_to_v3` | 0 | 98.400 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 34340.570 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step1` | 1 | 33176.930 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step2` | 1 | 32847.870 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.native.torchair.active.step3` | 1 | 32829.440 |
| `TorchDynamo Cache Lookup` | 3 | 31463.960 |
| `Torch-Compiled Region: 0/0` | 3 | 3709.630 |
| `TorchNpuGraphBase::Run` | 3 | 2740.680 |
| `RefreshAtTensorFromGeTensor` | 3 | 1137.320 |
| `aten::empty` | 3 | 553.120 |
| `ExecuteGraph` | 3 | 525.360 |
| `AssembleInputs` | 3 | 388.500 |
| `AssembleOutputs` | 3 | 300.720 |
| `aten::set_` | 3 | 278.960 |
| `empty_tensor` | 3 | 272.220 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 223438.600 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 93083.560 |
| `launch` | 976 | 19058.090 |
| `InputCopy` | 3 | 154.420 |
| `ModelExecute` | 3 | 49.840 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 28.570 |
| `step_info` | 6 | 14.640 |
| `OutputCopy` | 3 | 0.750 |

