# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4304_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4304_fractal_nz/liteserver-c001-4_634951_20260729134933925_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `93595.820 us`
- `Free`: `3495.920 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3362.250 us`
- `Stage`: `97091.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 23883.260 |
| `PromptFlashAttention` | 81 | 22844.740 |
| `StridedSliceD` | 405 | 11369.240 |
| `Transpose` | 324 | 11268.720 |
| `AddLayerNorm` | 162 | 4120.940 |
| `PadV3` | 243 | 4030.640 |
| `ConcatV2D` | 243 | 3624.680 |
| `Gelu` | 81 | 2995.720 |
| `Mul` | 324 | 2850.860 |
| `Add` | 162 | 1863.140 |
| `Cast` | 162 | 1669.180 |
| `Neg` | 162 | 1442.820 |
| `SplitVD` | 81 | 1295.060 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 233.300 |
| `LayerNormV3` | 3 | 88.660 |
| `Data` | 3 | 14.860 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_8` | 3 | 873.300 |
| `PromptFlashAttention_13` | 3 | 868.640 |
| `PromptFlashAttention_17` | 3 | 865.320 |
| `PromptFlashAttention_15` | 3 | 861.740 |
| `PromptFlashAttention_24` | 3 | 861.580 |
| `PromptFlashAttention_7` | 3 | 859.780 |
| `PromptFlashAttention_25` | 3 | 856.680 |
| `PromptFlashAttention_14` | 3 | 856.360 |
| `PromptFlashAttention_16` | 3 | 851.980 |
| `PromptFlashAttention` | 3 | 851.240 |
| `PromptFlashAttention_26` | 3 | 849.640 |
| `PromptFlashAttention_12` | 3 | 849.060 |
| `PromptFlashAttention_6` | 3 | 848.540 |
| `PromptFlashAttention_19` | 3 | 846.360 |
| `PromptFlashAttention_20` | 3 | 846.180 |
| `PromptFlashAttention_9` | 3 | 844.640 |
| `PromptFlashAttention_5` | 3 | 840.420 |
| `PromptFlashAttention_2` | 3 | 839.100 |
| `PromptFlashAttention_10` | 3 | 838.040 |
| `PromptFlashAttention_1` | 3 | 835.460 |
| `PromptFlashAttention_21` | 3 | 833.300 |
| `PromptFlashAttention_22` | 3 | 832.080 |
| `PromptFlashAttention_11` | 3 | 831.140 |
| `PromptFlashAttention_4` | 3 | 830.180 |
| `PromptFlashAttention_18` | 3 | 825.760 |
| `PromptFlashAttention_23` | 3 | 825.600 |
| `PromptFlashAttention_3` | 3 | 822.620 |
| `MatMulV2_107` | 3 | 312.000 |
| `MatMulV2_77` | 3 | 311.800 |
| `MatMulV2_95` | 3 | 311.680 |
| `MatMulV2_65` | 3 | 311.360 |
| `MatMulV2_119` | 3 | 311.180 |
| `MatMulV2_125` | 3 | 310.040 |
| `MatMulV2_131` | 3 | 309.940 |
| `MatMulV2_143` | 3 | 308.800 |
| `MatMulV2_83` | 3 | 308.420 |
| `MatMulV2_101` | 3 | 308.260 |
| `MatMulV2_53` | 3 | 307.820 |
| `MatMulV2_41` | 3 | 307.400 |
| `MatMulV2_59` | 3 | 307.320 |
| `MatMulV2_35` | 3 | 307.080 |
| `MatMulV2_89` | 3 | 306.980 |
| `MatMulV2_71` | 3 | 306.640 |
| `MatMulV2_23` | 3 | 306.360 |
| `MatMulV2_47` | 3 | 306.120 |
| `MatMulV2_11` | 3 | 305.160 |
| `MatMulV2_137` | 3 | 304.580 |
| `MatMulV2_155` | 3 | 304.500 |
| `MatMulV2_161` | 3 | 304.260 |
| `MatMulV2_149` | 3 | 301.700 |
| `MatMulV2_113` | 3 | 301.460 |
| `MatMulV2_17` | 3 | 300.900 |
| `MatMulV2_5` | 3 | 299.380 |
| `MatMulV2_29` | 3 | 298.020 |
| `MatMulV2_52` | 3 | 290.000 |
| `MatMulV2_100` | 3 | 284.920 |
| `MatMulV2_46` | 3 | 284.560 |
| `MatMulV2_130` | 3 | 284.440 |
| `MatMulV2_40` | 3 | 284.380 |
| `MatMulV2_76` | 3 | 284.160 |
| `MatMulV2_82` | 3 | 284.160 |
| `MatMulV2_106` | 3 | 283.940 |
| `MatMulV2_70` | 3 | 283.880 |
| `MatMulV2_148` | 3 | 283.800 |
| `MatMulV2_94` | 3 | 283.740 |
| `MatMulV2_64` | 3 | 283.460 |
| `MatMulV2_154` | 3 | 283.440 |
| `MatMulV2_124` | 3 | 283.360 |
| `MatMulV2_142` | 3 | 283.360 |
| `MatMulV2_34` | 3 | 283.300 |
| `MatMulV2_10` | 3 | 283.020 |
| `MatMulV2_160` | 3 | 282.980 |
| `MatMulV2_22` | 3 | 282.080 |
| `MatMulV2_118` | 3 | 281.960 |
| `MatMulV2_16` | 3 | 277.780 |
| `MatMulV2_58` | 3 | 274.020 |
| `MatMulV2_112` | 3 | 272.920 |
| `MatMulV2_136` | 3 | 272.100 |
| `MatMulV2_88` | 3 | 271.580 |
| `MatMulV2_4` | 3 | 270.920 |
| `MatMulV2_28` | 3 | 270.700 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 233.300 |
| `Transpose_255` | 3 | 124.440 |
| `Transpose_256` | 3 | 120.560 |
| `Transpose_266` | 3 | 120.500 |
| `Transpose_106` | 3 | 120.420 |
| `Transpose_156` | 3 | 120.280 |
| `Transpose_46` | 3 | 119.960 |
| `Transpose_76` | 3 | 119.760 |
| `Transpose_216` | 3 | 119.540 |
| `Transpose_16` | 3 | 119.520 |
| `Transpose_56` | 3 | 119.480 |
| `Transpose_226` | 3 | 119.460 |
| `Transpose_66` | 3 | 119.300 |
| `Transpose_176` | 3 | 119.280 |
| `Transpose_136` | 3 | 118.900 |
| `Transpose_196` | 3 | 118.740 |
| `Transpose_116` | 3 | 118.640 |
| `Transpose_186` | 3 | 118.620 |
| `Transpose_166` | 3 | 118.460 |
| `Transpose_206` | 3 | 118.400 |
| `Transpose_246` | 3 | 118.380 |
| `Transpose_146` | 3 | 118.340 |
| `Transpose_96` | 3 | 118.320 |
| `Transpose_26` | 3 | 118.000 |
| `Transpose_254` | 3 | 116.880 |
| `Transpose_6` | 3 | 116.760 |
| `Transpose_86` | 3 | 116.120 |
| `Transpose_253` | 3 | 116.040 |
| `Transpose_236` | 3 | 115.840 |
| `Transpose_126` | 3 | 115.740 |
| `Transpose_36` | 3 | 115.560 |
| `Gelu_22` | 3 | 112.040 |
| `Gelu_4` | 3 | 111.940 |
| `Gelu_9` | 3 | 111.800 |
| `Gelu` | 3 | 111.760 |
| `Gelu_18` | 3 | 111.740 |
| `Gelu_13` | 3 | 111.500 |
| `Gelu_14` | 3 | 111.000 |
| `Gelu_11` | 3 | 110.940 |
| `Gelu_5` | 3 | 110.800 |
| `Gelu_21` | 3 | 110.760 |
| `Gelu_25` | 3 | 110.760 |
| `Gelu_12` | 3 | 110.740 |
| `Gelu_15` | 3 | 110.740 |
| `Gelu_20` | 3 | 110.740 |
| `Gelu_3` | 3 | 110.720 |
| `Gelu_8` | 3 | 110.720 |
| `Gelu_24` | 3 | 110.700 |
| `Gelu_2` | 3 | 110.700 |
| `Gelu_23` | 3 | 110.680 |
| `Gelu_17` | 3 | 110.680 |
| `Gelu_1` | 3 | 110.660 |
| `Gelu_19` | 3 | 110.640 |
| `Gelu_6` | 3 | 110.620 |
| `Gelu_26` | 3 | 110.620 |
| `Gelu_7` | 3 | 110.600 |
| `Gelu_10` | 3 | 110.580 |
| `Gelu_16` | 3 | 110.540 |
| `StridedSliceV2_9` | 3 | 108.060 |
| `StridedSliceV2_39` | 3 | 105.080 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 103.980 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 103.920 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 103.840 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 103.460 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 103.340 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 103.220 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 102.680 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 102.440 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 102.320 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 102.260 |
| `Transpose_105` | 3 | 102.240 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 102.040 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 102.020 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 101.960 |
| `Transpose_265` | 3 | 101.960 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 101.860 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 101.760 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 101.660 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 101.640 |
| `Transpose_45` | 3 | 101.540 |
| `Transpose_95` | 3 | 101.440 |
| `Transpose_145` | 3 | 101.440 |
| `Transpose_35` | 3 | 101.300 |
| `Transpose_205` | 3 | 101.280 |
| `Transpose_175` | 3 | 101.160 |
| `Transpose_155` | 3 | 101.080 |
| `Transpose_215` | 3 | 101.060 |
| `Transpose_235` | 3 | 101.040 |
| `Transpose_15` | 3 | 100.980 |
| `Transpose_225` | 3 | 100.980 |
| `Transpose_55` | 3 | 100.900 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 100.880 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 100.840 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 100.840 |
| `Transpose_165` | 3 | 100.840 |
| `Transpose_85` | 3 | 100.820 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 100.780 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 100.780 |
| `Transpose_3` | 3 | 100.720 |
| `Transpose_195` | 3 | 100.720 |
| `Transpose_65` | 3 | 100.720 |
| `Transpose_75` | 3 | 100.680 |
| `Transpose_25` | 3 | 100.640 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 100.580 |
| `Transpose_135` | 3 | 100.520 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 100.460 |
| `Transpose_185` | 3 | 100.440 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 100.440 |
| `Transpose_115` | 3 | 100.360 |
| `Transpose_125` | 3 | 100.360 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 100.320 |
| `Transpose_245` | 3 | 100.180 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 99.900 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 99.860 |
| `Transpose_84` | 3 | 99.440 |
| `Transpose_244` | 3 | 99.280 |
| `Transpose_94` | 3 | 99.280 |
| `Transpose_33` | 3 | 99.080 |
| `Transpose_103` | 3 | 99.060 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22844.740 |
| `StridedSliceD | "1,2048,16,72" -> "1,2048,16,36" | ND -> ND` | 324 | 8829.920 |
| `MatMulV2 | "2048,4304;269,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 8279.160 |
| `Transpose | "2048,16,72;3" -> "16,2048,72" | ND;ND -> ND` | 243 | 8065.840 |
| `MatMulV2 | "2048,1152;72,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 8015.140 |
| `MatMulV2 | "2048,1152;72,269,16,16;4304" -> "2048,4304" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7588.960 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4120.940 |
| `PadV3 | "1,16,2048,72;8;" -> "1,16,2048,80" | ND;ND;ND -> ND` | 243 | 4030.640 |
| `Transpose | "16,2048,72;3" -> "2048,16,72" | ND;ND -> ND` | 81 | 3202.880 |
| `Gelu | "1,2048,4304" -> "1,2048,4304" | ND -> ND` | 81 | 2995.720 |
| `Mul | "1,2048,16,72;1,2048,1,72" -> "1,2048,16,72" | ND;ND -> ND` | 324 | 2850.860 |
| `ConcatV2D | "1,2048,16,36;1,2048,16,36" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 2723.140 |
| `StridedSliceD | "1,16,2048,80" -> "1,16,2048,72" | ND -> ND` | 81 | 2539.320 |
| `Add | "1,2048,16,72;1,2048,16,72" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 1863.140 |
| `Cast | "1,2048,16,72" -> "1,2048,16,72" | ND -> ND` | 162 | 1669.180 |
| `Neg | "1,2048,16,36" -> "1,2048,16,36" | ND -> ND` | 162 | 1442.820 |
| `SplitVD | "1,2048,3456" -> "1,2048,1152;1,2048,1152;1,2048,1152" | ND -> ND;ND;ND` | 81 | 1295.060 |
| `ConcatV2D | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,3456" | ND;ND;ND -> ND` | 81 | 901.540 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 233.300 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 88.660 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.860 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 26965.680 |
| `ND;FRACTAL_NZ;ND` | 486 | 23883.260 |
| `ND` | 891 | 18772.020 |
| `ND;ND` | 972 | 18705.860 |
| `ND;ND;ND` | 327 | 5020.840 |
| `N/A` | 246 | 248.160 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_8` | 0 | 291.880 |
| `PromptFlashAttention_13` | 0 | 291.440 |
| `PromptFlashAttention_8` | 0 | 290.940 |
| `PromptFlashAttention_8` | 0 | 290.480 |
| `PromptFlashAttention_17` | 0 | 289.520 |
| `PromptFlashAttention_17` | 0 | 289.280 |
| `PromptFlashAttention_13` | 0 | 289.240 |
| `PromptFlashAttention_15` | 0 | 288.080 |
| `PromptFlashAttention_13` | 0 | 287.960 |
| `PromptFlashAttention_24` | 0 | 287.860 |
| `PromptFlashAttention_15` | 0 | 287.780 |
| `PromptFlashAttention_7` | 0 | 287.440 |
| `PromptFlashAttention_24` | 0 | 287.100 |
| `PromptFlashAttention_24` | 0 | 286.620 |
| `PromptFlashAttention_17` | 0 | 286.520 |
| `PromptFlashAttention_7` | 0 | 286.300 |
| `PromptFlashAttention_25` | 0 | 286.160 |
| `PromptFlashAttention_7` | 0 | 286.040 |
| `PromptFlashAttention_15` | 0 | 285.880 |
| `PromptFlashAttention_14` | 0 | 285.840 |
| `PromptFlashAttention_14` | 0 | 285.500 |
| `PromptFlashAttention_25` | 0 | 285.380 |
| `PromptFlashAttention_16` | 0 | 285.360 |
| `PromptFlashAttention_25` | 0 | 285.140 |
| `PromptFlashAttention_14` | 0 | 285.020 |
| `PromptFlashAttention` | 0 | 284.720 |
| `PromptFlashAttention_26` | 0 | 284.080 |
| `PromptFlashAttention_6` | 0 | 283.680 |
| `PromptFlashAttention_12` | 0 | 283.660 |
| `PromptFlashAttention` | 0 | 283.580 |
| `PromptFlashAttention_16` | 0 | 283.320 |
| `PromptFlashAttention_16` | 0 | 283.300 |
| `PromptFlashAttention_12` | 0 | 283.300 |
| `PromptFlashAttention_26` | 0 | 283.240 |
| `PromptFlashAttention_20` | 0 | 283.200 |
| `PromptFlashAttention_20` | 0 | 283.140 |
| `PromptFlashAttention_6` | 0 | 283.100 |
| `PromptFlashAttention` | 0 | 282.940 |
| `PromptFlashAttention_19` | 0 | 282.540 |
| `PromptFlashAttention_9` | 0 | 282.400 |
| `PromptFlashAttention_26` | 0 | 282.320 |
| `PromptFlashAttention_19` | 0 | 282.180 |
| `PromptFlashAttention_12` | 0 | 282.100 |
| `PromptFlashAttention_5` | 0 | 282.080 |
| `PromptFlashAttention_6` | 0 | 281.760 |
| `PromptFlashAttention_19` | 0 | 281.640 |
| `PromptFlashAttention_9` | 0 | 281.320 |
| `PromptFlashAttention_9` | 0 | 280.920 |
| `PromptFlashAttention_2` | 0 | 280.560 |
| `PromptFlashAttention_5` | 0 | 280.240 |
| `PromptFlashAttention_21` | 0 | 280.120 |
| `PromptFlashAttention_10` | 0 | 280.020 |
| `PromptFlashAttention_20` | 0 | 279.840 |
| `PromptFlashAttention_2` | 0 | 279.760 |
| `PromptFlashAttention_1` | 0 | 279.420 |
| `PromptFlashAttention_10` | 0 | 279.180 |
| `PromptFlashAttention_10` | 0 | 278.840 |
| `PromptFlashAttention_2` | 0 | 278.780 |
| `PromptFlashAttention_1` | 0 | 278.620 |
| `PromptFlashAttention_22` | 0 | 278.500 |
| `PromptFlashAttention_22` | 0 | 278.100 |
| `PromptFlashAttention_5` | 0 | 278.100 |
| `PromptFlashAttention_11` | 0 | 277.960 |
| `PromptFlashAttention_11` | 0 | 277.620 |
| `PromptFlashAttention_1` | 0 | 277.420 |
| `PromptFlashAttention_4` | 0 | 277.380 |
| `PromptFlashAttention_4` | 0 | 277.220 |
| `PromptFlashAttention_23` | 0 | 277.180 |
| `PromptFlashAttention_21` | 0 | 276.760 |
| `PromptFlashAttention_21` | 0 | 276.420 |
| `PromptFlashAttention_4` | 0 | 275.580 |
| `PromptFlashAttention_11` | 0 | 275.560 |
| `PromptFlashAttention_22` | 0 | 275.480 |
| `PromptFlashAttention_18` | 0 | 275.400 |
| `PromptFlashAttention_18` | 0 | 275.380 |
| `PromptFlashAttention_23` | 0 | 275.220 |
| `PromptFlashAttention_18` | 0 | 274.980 |
| `PromptFlashAttention_3` | 0 | 274.780 |
| `PromptFlashAttention_3` | 0 | 274.060 |
| `PromptFlashAttention_3` | 0 | 273.780 |
| `PromptFlashAttention_23` | 0 | 273.200 |
| `MatMulV2_65` | 0 | 105.020 |
| `MatMulV2_95` | 0 | 104.780 |
| `MatMulV2_107` | 0 | 104.660 |
| `MatMulV2_77` | 0 | 104.600 |
| `MatMulV2_107` | 0 | 104.580 |
| `MatMulV2_131` | 0 | 104.480 |
| `MatMulV2_95` | 0 | 104.140 |
| `MatMulV2_77` | 0 | 104.140 |
| `MatMulV2_83` | 0 | 104.080 |
| `MatMulV2_143` | 0 | 104.000 |
| `MatMulV2_119` | 0 | 103.920 |
| `MatMulV2_125` | 0 | 103.820 |
| `MatMulV2_119` | 0 | 103.780 |
| `MatMulV2_53` | 0 | 103.500 |
| `MatMulV2_89` | 0 | 103.500 |
| `MatMulV2_119` | 0 | 103.480 |
| `MatMulV2_125` | 0 | 103.360 |
| `MatMulV2_59` | 0 | 103.320 |
| `MatMulV2_65` | 0 | 103.220 |
| `MatMulV2_53` | 0 | 103.200 |
| `MatMulV2_89` | 0 | 103.160 |
| `MatMulV2_11` | 0 | 103.120 |
| `MatMulV2_65` | 0 | 103.120 |
| `MatMulV2_143` | 0 | 103.100 |
| `MatMulV2_77` | 0 | 103.060 |
| `MatMulV2_47` | 0 | 103.000 |
| `MatMulV2_131` | 0 | 103.000 |
| `MatMulV2_101` | 0 | 102.960 |
| `MatMulV2_23` | 0 | 102.880 |
| `MatMulV2_125` | 0 | 102.860 |
| `MatMulV2_101` | 0 | 102.860 |
| `MatMulV2_35` | 0 | 102.780 |
| `MatMulV2_95` | 0 | 102.760 |
| `MatMulV2_107` | 0 | 102.760 |
| `MatMulV2_41` | 0 | 102.580 |
| `MatMulV2_83` | 0 | 102.540 |
| `MatMulV2_41` | 0 | 102.480 |
| `MatMulV2_71` | 0 | 102.480 |
| `MatMulV2_23` | 0 | 102.480 |
| `MatMulV2_131` | 0 | 102.460 |
| `MatMulV2_101` | 0 | 102.440 |
| `MatMulV2_59` | 0 | 102.380 |
| `MatMulV2_35` | 0 | 102.380 |
| `MatMulV2_155` | 0 | 102.340 |
| `MatMulV2_41` | 0 | 102.340 |
| `MatMulV2_71` | 0 | 102.320 |
| `MatMulV2_17` | 0 | 102.300 |
| `MatMulV2_137` | 0 | 102.240 |
| `MatMulV2_47` | 0 | 102.160 |
| `MatMulV2_35` | 0 | 101.920 |
| `MatMulV2_71` | 0 | 101.840 |
| `MatMulV2_155` | 0 | 101.820 |
| `MatMulV2_83` | 0 | 101.800 |
| `MatMulV2_143` | 0 | 101.700 |
| `MatMulV2_59` | 0 | 101.620 |
| `MatMulV2_11` | 0 | 101.580 |
| `MatMulV2_161` | 0 | 101.560 |
| `MatMulV2_161` | 0 | 101.460 |
| `MatMulV2_137` | 0 | 101.340 |
| `MatMulV2_161` | 0 | 101.240 |
| `MatMulV2_149` | 0 | 101.220 |
| `MatMulV2_149` | 0 | 101.200 |
| `MatMulV2_113` | 0 | 101.140 |
| `MatMulV2_53` | 0 | 101.120 |
| `MatMulV2_5` | 0 | 101.040 |
| `MatMulV2_23` | 0 | 101.000 |
| `MatMulV2_137` | 0 | 101.000 |
| `MatMulV2_47` | 0 | 100.960 |
| `MatMulV2_113` | 0 | 100.620 |
| `MatMulV2_11` | 0 | 100.460 |
| `MatMulV2_17` | 0 | 100.400 |
| `MatMulV2_155` | 0 | 100.340 |
| `MatMulV2_89` | 0 | 100.320 |
| `MatMulV2_29` | 0 | 99.740 |
| `MatMulV2_113` | 0 | 99.700 |
| `MatMulV2_29` | 0 | 99.320 |
| `MatMulV2_149` | 0 | 99.280 |
| `MatMulV2_5` | 0 | 99.200 |
| `MatMulV2_5` | 0 | 99.140 |
| `MatMulV2_29` | 0 | 98.960 |
| `MatMulV2_17` | 0 | 98.200 |
| `MatMulV2_52` | 0 | 96.900 |
| `MatMulV2_52` | 0 | 96.680 |
| `MatMulV2_52` | 0 | 96.420 |
| `MatMulV2_10` | 0 | 95.900 |
| `MatMulV2_76` | 0 | 95.440 |
| `MatMulV2_40` | 0 | 95.140 |
| `MatMulV2_100` | 0 | 95.100 |
| `MatMulV2_34` | 0 | 95.000 |
| `MatMulV2_64` | 0 | 94.980 |
| `MatMulV2_10` | 0 | 94.980 |
| `MatMulV2_94` | 0 | 94.960 |
| `MatMulV2_46` | 0 | 94.940 |
| `MatMulV2_100` | 0 | 94.920 |
| `MatMulV2_34` | 0 | 94.900 |
| `MatMulV2_130` | 0 | 94.900 |
| `MatMulV2_46` | 0 | 94.900 |
| `MatMulV2_100` | 0 | 94.900 |
| `MatMulV2_106` | 0 | 94.880 |
| `MatMulV2_82` | 0 | 94.860 |
| `MatMulV2_130` | 0 | 94.800 |
| `MatMulV2_106` | 0 | 94.780 |
| `MatMulV2_148` | 0 | 94.760 |
| `MatMulV2_40` | 0 | 94.760 |
| `MatMulV2_70` | 0 | 94.740 |
| `MatMulV2_130` | 0 | 94.740 |
| `MatMulV2_46` | 0 | 94.720 |
| `MatMulV2_22` | 0 | 94.720 |
| `MatMulV2_64` | 0 | 94.720 |
| `MatMulV2_148` | 0 | 94.700 |
| `MatMulV2_160` | 0 | 94.700 |
| `MatMulV2_82` | 0 | 94.680 |
| `MatMulV2_124` | 0 | 94.640 |
| `MatMulV2_142` | 0 | 94.620 |
| `MatMulV2_82` | 0 | 94.620 |
| `MatMulV2_70` | 0 | 94.600 |
| `MatMulV2_94` | 0 | 94.580 |
| `MatMulV2_118` | 0 | 94.580 |
| `MatMulV2_142` | 0 | 94.560 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 34059.510 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.fractal_nz.torchair.active.step1` | 1 | 32891.850 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.fractal_nz.torchair.active.step3` | 1 | 32557.260 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4304.fractal_nz.torchair.active.step2` | 1 | 32555.920 |
| `TorchDynamo Cache Lookup` | 3 | 31271.320 |
| `Torch-Compiled Region: 0/0` | 3 | 3595.730 |
| `TorchNpuGraphBase::Run` | 3 | 2620.750 |
| `RefreshAtTensorFromGeTensor` | 3 | 1115.060 |
| `aten::empty` | 3 | 543.720 |
| `ExecuteGraph` | 3 | 485.610 |
| `AssembleInputs` | 3 | 358.740 |
| `AssembleOutputs` | 3 | 285.490 |
| `aten::set_` | 3 | 272.430 |
| `empty_tensor` | 3 | 267.640 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 219726.950 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 92362.440 |
| `launch` | 976 | 18104.100 |
| `InputCopy` | 3 | 131.910 |
| `ModelExecute` | 3 | 58.890 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 25.970 |
| `step_info` | 6 | 25.630 |
| `OutputCopy` | 3 | 0.740 |

