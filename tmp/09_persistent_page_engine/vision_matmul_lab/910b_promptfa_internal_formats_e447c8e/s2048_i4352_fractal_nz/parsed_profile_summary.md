# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4352_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s2048_i4352_fractal_nz/liteserver-c001-4_638203_20260729135159389_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `91440.100 us`
- `Free`: `3514.200 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3349.000 us`
- `Stage`: `94954.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 81 | 22821.580 |
| `MatMulV2` | 486 | 21997.940 |
| `StridedSliceD` | 405 | 11405.600 |
| `Transpose` | 324 | 11049.580 |
| `PadV3` | 243 | 4230.560 |
| `AddLayerNorm` | 162 | 4097.340 |
| `ConcatV2D` | 243 | 3587.320 |
| `Gelu` | 81 | 3114.900 |
| `Mul` | 324 | 2682.380 |
| `Add` | 162 | 1716.660 |
| `Cast` | 162 | 1591.100 |
| `Neg` | 162 | 1505.140 |
| `SplitVD` | 81 | 1302.520 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 235.720 |
| `LayerNormV3` | 3 | 86.840 |
| `Data` | 3 | 14.920 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_17` | 3 | 875.220 |
| `PromptFlashAttention_13` | 3 | 862.560 |
| `PromptFlashAttention` | 3 | 859.720 |
| `PromptFlashAttention_15` | 3 | 857.540 |
| `PromptFlashAttention_24` | 3 | 856.280 |
| `PromptFlashAttention_25` | 3 | 856.100 |
| `PromptFlashAttention_8` | 3 | 853.820 |
| `PromptFlashAttention_7` | 3 | 853.740 |
| `PromptFlashAttention_14` | 3 | 852.900 |
| `PromptFlashAttention_20` | 3 | 852.340 |
| `PromptFlashAttention_26` | 3 | 850.500 |
| `PromptFlashAttention_16` | 3 | 849.460 |
| `PromptFlashAttention_12` | 3 | 848.660 |
| `PromptFlashAttention_6` | 3 | 847.820 |
| `PromptFlashAttention_5` | 3 | 847.560 |
| `PromptFlashAttention_9` | 3 | 844.300 |
| `PromptFlashAttention_11` | 3 | 837.740 |
| `PromptFlashAttention_19` | 3 | 837.140 |
| `PromptFlashAttention_22` | 3 | 835.320 |
| `PromptFlashAttention_1` | 3 | 834.240 |
| `PromptFlashAttention_10` | 3 | 833.660 |
| `PromptFlashAttention_4` | 3 | 833.080 |
| `PromptFlashAttention_21` | 3 | 829.880 |
| `PromptFlashAttention_3` | 3 | 829.140 |
| `PromptFlashAttention_18` | 3 | 829.040 |
| `PromptFlashAttention_2` | 3 | 828.040 |
| `PromptFlashAttention_23` | 3 | 825.780 |
| `MatMulV2_131` | 3 | 280.840 |
| `MatMulV2_53` | 3 | 280.640 |
| `MatMulV2_95` | 3 | 280.260 |
| `MatMulV2_47` | 3 | 280.200 |
| `MatMulV2_5` | 3 | 279.100 |
| `MatMulV2_149` | 3 | 278.660 |
| `MatMulV2_83` | 3 | 278.220 |
| `MatMulV2_41` | 3 | 277.800 |
| `MatMulV2_71` | 3 | 277.780 |
| `MatMulV2_107` | 3 | 277.440 |
| `MatMulV2_155` | 3 | 274.960 |
| `MatMulV2_35` | 3 | 273.840 |
| `MatMulV2_65` | 3 | 273.480 |
| `MatMulV2_77` | 3 | 272.480 |
| `MatMulV2_113` | 3 | 272.060 |
| `MatMulV2_11` | 3 | 271.720 |
| `MatMulV2_137` | 3 | 271.320 |
| `MatMulV2_161` | 3 | 271.300 |
| `MatMulV2_143` | 3 | 271.020 |
| `MatMulV2_125` | 3 | 270.900 |
| `MatMulV2_23` | 3 | 270.780 |
| `MatMulV2_17` | 3 | 270.320 |
| `MatMulV2_89` | 3 | 268.120 |
| `MatMulV2_101` | 3 | 267.560 |
| `MatMulV2_119` | 3 | 264.800 |
| `MatMulV2_59` | 3 | 264.740 |
| `MatMulV2_29` | 3 | 259.960 |
| `MatMulV2_40` | 3 | 251.480 |
| `MatMulV2_64` | 3 | 250.940 |
| `MatMulV2_112` | 3 | 250.440 |
| `MatMulV2_124` | 3 | 250.380 |
| `MatMulV2_70` | 3 | 249.220 |
| `MatMulV2_76` | 3 | 248.700 |
| `MatMulV2_22` | 3 | 248.200 |
| `MatMulV2_88` | 3 | 247.780 |
| `MatMulV2_148` | 3 | 247.320 |
| `MatMulV2_82` | 3 | 247.100 |
| `MatMulV2_136` | 3 | 247.020 |
| `MatMulV2_100` | 3 | 246.660 |
| `MatMulV2_94` | 3 | 245.680 |
| `MatMulV2_52` | 3 | 245.620 |
| `MatMulV2_160` | 3 | 244.980 |
| `MatMulV2_154` | 3 | 244.580 |
| `MatMulV2_130` | 3 | 243.780 |
| `MatMulV2_4` | 3 | 243.640 |
| `MatMulV2_46` | 3 | 243.580 |
| `MatMulV2_118` | 3 | 242.260 |
| `MatMulV2_58` | 3 | 241.820 |
| `MatMulV2_10` | 3 | 241.680 |
| `MatMulV2_16` | 3 | 241.480 |
| `MatMulV2_28` | 3 | 241.260 |
| `MatMulV2_142` | 3 | 241.100 |
| `MatMulV2_106` | 3 | 239.880 |
| `MatMulV2_34` | 3 | 239.140 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 235.720 |
| `Transpose_66` | 3 | 130.540 |
| `Transpose_6` | 3 | 128.840 |
| `Transpose_256` | 3 | 118.600 |
| `Transpose_26` | 3 | 118.540 |
| `Transpose_136` | 3 | 118.180 |
| `Transpose_186` | 3 | 118.180 |
| `Transpose_36` | 3 | 117.980 |
| `Transpose_246` | 3 | 117.880 |
| `Transpose_156` | 3 | 117.820 |
| `Transpose_16` | 3 | 117.680 |
| `Transpose_216` | 3 | 117.680 |
| `Transpose_86` | 3 | 117.660 |
| `Transpose_56` | 3 | 117.600 |
| `Transpose_176` | 3 | 117.460 |
| `Transpose_76` | 3 | 117.440 |
| `Transpose_226` | 3 | 117.120 |
| `Transpose_116` | 3 | 117.080 |
| `Transpose_126` | 3 | 116.980 |
| `Transpose_266` | 3 | 116.900 |
| `Transpose_96` | 3 | 116.640 |
| `Transpose_106` | 3 | 116.480 |
| `Transpose_206` | 3 | 116.380 |
| `Gelu_4` | 3 | 116.200 |
| `Gelu_19` | 3 | 116.180 |
| `Gelu_23` | 3 | 116.160 |
| `Transpose_166` | 3 | 116.080 |
| `Gelu` | 3 | 116.040 |
| `Gelu_9` | 3 | 115.940 |
| `Gelu_14` | 3 | 115.920 |
| `Gelu_24` | 3 | 115.500 |
| `Gelu_13` | 3 | 115.300 |
| `Gelu_20` | 3 | 115.300 |
| `Gelu_15` | 3 | 115.280 |
| `Gelu_10` | 3 | 115.260 |
| `Gelu_11` | 3 | 115.240 |
| `Gelu_17` | 3 | 115.220 |
| `Transpose_104` | 3 | 115.200 |
| `Gelu_3` | 3 | 115.180 |
| `Gelu_5` | 3 | 115.180 |
| `Gelu_21` | 3 | 115.180 |
| `Gelu_26` | 3 | 115.160 |
| `Transpose_146` | 3 | 115.140 |
| `Gelu_1` | 3 | 115.120 |
| `Gelu_18` | 3 | 115.120 |
| `Gelu_12` | 3 | 115.100 |
| `Gelu_8` | 3 | 115.100 |
| `Gelu_22` | 3 | 115.100 |
| `Gelu_6` | 3 | 115.080 |
| `Gelu_2` | 3 | 115.080 |
| `Gelu_7` | 3 | 115.060 |
| `Gelu_16` | 3 | 115.000 |
| `Gelu_25` | 3 | 114.900 |
| `Transpose_46` | 3 | 114.020 |
| `Transpose_196` | 3 | 113.840 |
| `Transpose_236` | 3 | 113.680 |
| `Transpose_55` | 3 | 112.640 |
| `Transpose_43` | 3 | 110.340 |
| `Transpose_44` | 3 | 110.340 |
| `Transpose_103` | 3 | 110.060 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 102.900 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 102.860 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 102.580 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 102.440 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 102.400 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 102.240 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 101.840 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 101.740 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 101.480 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 101.440 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 101.420 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 101.360 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 101.120 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 100.960 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 100.920 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 100.720 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 100.540 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 100.280 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 100.240 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 100.200 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 100.060 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 99.780 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 99.600 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 99.380 |
| `Transpose_245` | 3 | 99.220 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 99.180 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 99.160 |
| `Transpose_85` | 3 | 99.160 |
| `Transpose_255` | 3 | 99.020 |
| `Transpose_95` | 3 | 98.920 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 98.900 |
| `Transpose_195` | 3 | 98.560 |
| `Transpose_25` | 3 | 98.520 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 98.480 |
| `Transpose_35` | 3 | 98.380 |
| `Transpose_185` | 3 | 98.360 |
| `StridedSliceV2_29` | 3 | 98.360 |
| `Transpose_135` | 3 | 98.360 |
| `Transpose_205` | 3 | 98.340 |
| `StridedSliceV2_109` | 3 | 98.160 |
| `Transpose_45` | 3 | 98.140 |
| `Transpose_125` | 3 | 98.080 |
| `Transpose_145` | 3 | 98.060 |
| `Transpose_235` | 3 | 98.040 |
| `Transpose_165` | 3 | 97.960 |
| `Transpose_215` | 3 | 97.940 |
| `Transpose_3` | 3 | 97.920 |
| `Transpose_175` | 3 | 97.900 |
| `Transpose_75` | 3 | 97.860 |
| `StridedSliceV2_104` | 3 | 97.780 |
| `Transpose_105` | 3 | 97.720 |
| `Transpose_155` | 3 | 97.660 |
| `Transpose_65` | 3 | 97.540 |
| `StridedSliceV2_9` | 3 | 97.480 |
| `StridedSliceV2_69` | 3 | 97.480 |
| `Transpose_15` | 3 | 97.440 |
| `Transpose_225` | 3 | 97.420 |
| `Transpose_265` | 3 | 97.280 |
| `StridedSliceV2_39` | 3 | 97.100 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | ND;ND;ND;ND -> ND` | 81 | 22821.580 |
| `StridedSliceD | "1,2048,16,72" -> "1,2048,16,36" | ND -> ND` | 324 | 8811.860 |
| `MatMulV2 | "2048,1152;72,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 7991.920 |
| `Transpose | "2048,16,72;3" -> "16,2048,72" | ND;ND -> ND` | 243 | 7867.160 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7380.300 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6625.720 |
| `PadV3 | "1,16,2048,72;8;" -> "1,16,2048,80" | ND;ND;ND -> ND` | 243 | 4230.560 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4097.340 |
| `Transpose | "16,2048,72;3" -> "2048,16,72" | ND;ND -> ND` | 81 | 3182.420 |
| `Gelu | "1,2048,4352" -> "1,2048,4352" | ND -> ND` | 81 | 3114.900 |
| `Mul | "1,2048,16,72;1,2048,1,72" -> "1,2048,16,72" | ND;ND -> ND` | 324 | 2682.380 |
| `ConcatV2D | "1,2048,16,36;1,2048,16,36" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 2674.940 |
| `StridedSliceD | "1,16,2048,80" -> "1,16,2048,72" | ND -> ND` | 81 | 2593.740 |
| `Add | "1,2048,16,72;1,2048,16,72" -> "1,2048,16,72" | ND;ND -> ND` | 162 | 1716.660 |
| `Cast | "1,2048,16,72" -> "1,2048,16,72" | ND -> ND` | 162 | 1591.100 |
| `Neg | "1,2048,16,36" -> "1,2048,16,36" | ND -> ND` | 162 | 1505.140 |
| `SplitVD | "1,2048,3456" -> "1,2048,1152;1,2048,1152;1,2048,1152" | ND -> ND;ND;ND` | 81 | 1302.520 |
| `ConcatV2D | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,3456" | ND;ND;ND -> ND` | 81 | 912.380 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 235.720 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 3 | 86.840 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.920 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND;ND` | 243 | 26918.920 |
| `ND;FRACTAL_NZ;ND` | 486 | 21997.940 |
| `ND` | 891 | 18919.260 |
| `ND;ND` | 972 | 18123.560 |
| `ND;ND;ND` | 327 | 5229.780 |
| `N/A` | 246 | 250.640 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_17` | 0 | 295.060 |
| `PromptFlashAttention_17` | 0 | 290.320 |
| `PromptFlashAttention_17` | 0 | 289.840 |
| `PromptFlashAttention_13` | 0 | 288.800 |
| `PromptFlashAttention_13` | 0 | 287.280 |
| `PromptFlashAttention_25` | 0 | 287.140 |
| `PromptFlashAttention` | 0 | 287.040 |
| `PromptFlashAttention_15` | 0 | 286.760 |
| `PromptFlashAttention_24` | 0 | 286.700 |
| `PromptFlashAttention` | 0 | 286.520 |
| `PromptFlashAttention_13` | 0 | 286.480 |
| `PromptFlashAttention` | 0 | 286.160 |
| `PromptFlashAttention_15` | 0 | 285.660 |
| `PromptFlashAttention_8` | 0 | 285.560 |
| `PromptFlashAttention_14` | 0 | 285.560 |
| `PromptFlashAttention_24` | 0 | 285.480 |
| `PromptFlashAttention_8` | 0 | 285.300 |
| `PromptFlashAttention_7` | 0 | 285.180 |
| `PromptFlashAttention_15` | 0 | 285.120 |
| `PromptFlashAttention_7` | 0 | 284.920 |
| `PromptFlashAttention_25` | 0 | 284.780 |
| `PromptFlashAttention_20` | 0 | 284.740 |
| `PromptFlashAttention_20` | 0 | 284.500 |
| `PromptFlashAttention_14` | 0 | 284.440 |
| `PromptFlashAttention_26` | 0 | 284.280 |
| `PromptFlashAttention_5` | 0 | 284.220 |
| `PromptFlashAttention_25` | 0 | 284.180 |
| `PromptFlashAttention_24` | 0 | 284.100 |
| `PromptFlashAttention_16` | 0 | 283.880 |
| `PromptFlashAttention_12` | 0 | 283.860 |
| `PromptFlashAttention_7` | 0 | 283.640 |
| `PromptFlashAttention_12` | 0 | 283.560 |
| `PromptFlashAttention_16` | 0 | 283.540 |
| `PromptFlashAttention_26` | 0 | 283.180 |
| `PromptFlashAttention_20` | 0 | 283.100 |
| `PromptFlashAttention_26` | 0 | 283.040 |
| `PromptFlashAttention_6` | 0 | 282.960 |
| `PromptFlashAttention_8` | 0 | 282.960 |
| `PromptFlashAttention_14` | 0 | 282.900 |
| `PromptFlashAttention_9` | 0 | 282.760 |
| `PromptFlashAttention_5` | 0 | 282.480 |
| `PromptFlashAttention_6` | 0 | 282.460 |
| `PromptFlashAttention_6` | 0 | 282.400 |
| `PromptFlashAttention_16` | 0 | 282.040 |
| `PromptFlashAttention_9` | 0 | 281.580 |
| `PromptFlashAttention_12` | 0 | 281.240 |
| `PromptFlashAttention_19` | 0 | 281.140 |
| `PromptFlashAttention_5` | 0 | 280.860 |
| `PromptFlashAttention_9` | 0 | 279.960 |
| `PromptFlashAttention_11` | 0 | 279.860 |
| `PromptFlashAttention_11` | 0 | 279.400 |
| `PromptFlashAttention_1` | 0 | 279.300 |
| `PromptFlashAttention_22` | 0 | 279.280 |
| `PromptFlashAttention_1` | 0 | 279.220 |
| `PromptFlashAttention_10` | 0 | 279.080 |
| `PromptFlashAttention_3` | 0 | 278.500 |
| `PromptFlashAttention_11` | 0 | 278.480 |
| `PromptFlashAttention_19` | 0 | 278.440 |
| `PromptFlashAttention_4` | 0 | 278.400 |
| `PromptFlashAttention_22` | 0 | 278.240 |
| `PromptFlashAttention_21` | 0 | 277.960 |
| `PromptFlashAttention_22` | 0 | 277.800 |
| `PromptFlashAttention_19` | 0 | 277.560 |
| `PromptFlashAttention_10` | 0 | 277.540 |
| `PromptFlashAttention_4` | 0 | 277.500 |
| `PromptFlashAttention_2` | 0 | 277.480 |
| `PromptFlashAttention_4` | 0 | 277.180 |
| `PromptFlashAttention_10` | 0 | 277.040 |
| `PromptFlashAttention_18` | 0 | 276.880 |
| `PromptFlashAttention_2` | 0 | 276.520 |
| `PromptFlashAttention_21` | 0 | 276.160 |
| `PromptFlashAttention_18` | 0 | 276.120 |
| `PromptFlashAttention_18` | 0 | 276.040 |
| `PromptFlashAttention_23` | 0 | 275.980 |
| `PromptFlashAttention_21` | 0 | 275.760 |
| `PromptFlashAttention_1` | 0 | 275.720 |
| `PromptFlashAttention_3` | 0 | 275.540 |
| `PromptFlashAttention_23` | 0 | 275.160 |
| `PromptFlashAttention_3` | 0 | 275.100 |
| `PromptFlashAttention_23` | 0 | 274.640 |
| `PromptFlashAttention_2` | 0 | 274.040 |
| `MatMulV2_5` | 0 | 96.580 |
| `MatMulV2_95` | 0 | 95.620 |
| `MatMulV2_95` | 0 | 95.500 |
| `MatMulV2_131` | 0 | 94.880 |
| `MatMulV2_149` | 0 | 94.540 |
| `MatMulV2_41` | 0 | 94.080 |
| `MatMulV2_53` | 0 | 93.960 |
| `MatMulV2_47` | 0 | 93.880 |
| `MatMulV2_53` | 0 | 93.740 |
| `MatMulV2_47` | 0 | 93.680 |
| `MatMulV2_71` | 0 | 93.620 |
| `MatMulV2_35` | 0 | 93.600 |
| `MatMulV2_23` | 0 | 93.340 |
| `MatMulV2_83` | 0 | 93.300 |
| `MatMulV2_131` | 0 | 93.280 |
| `MatMulV2_11` | 0 | 92.960 |
| `MatMulV2_53` | 0 | 92.940 |
| `MatMulV2_83` | 0 | 92.860 |
| `MatMulV2_107` | 0 | 92.700 |
| `MatMulV2_131` | 0 | 92.680 |
| `MatMulV2_47` | 0 | 92.640 |
| `MatMulV2_71` | 0 | 92.640 |
| `MatMulV2_5` | 0 | 92.400 |
| `MatMulV2_107` | 0 | 92.380 |
| `MatMulV2_107` | 0 | 92.360 |
| `MatMulV2_155` | 0 | 92.340 |
| `MatMulV2_41` | 0 | 92.340 |
| `MatMulV2_11` | 0 | 92.280 |
| `MatMulV2_65` | 0 | 92.180 |
| `MatMulV2_149` | 0 | 92.120 |
| `MatMulV2_89` | 0 | 92.100 |
| `MatMulV2_83` | 0 | 92.060 |
| `MatMulV2_149` | 0 | 92.000 |
| `MatMulV2_23` | 0 | 91.880 |
| `MatMulV2_161` | 0 | 91.820 |
| `MatMulV2_71` | 0 | 91.520 |
| `MatMulV2_155` | 0 | 91.520 |
| `MatMulV2_35` | 0 | 91.480 |
| `MatMulV2_77` | 0 | 91.400 |
| `MatMulV2_41` | 0 | 91.380 |
| `MatMulV2_143` | 0 | 91.300 |
| `MatMulV2_113` | 0 | 91.280 |
| `MatMulV2_155` | 0 | 91.100 |
| `MatMulV2_65` | 0 | 91.060 |
| `MatMulV2_137` | 0 | 91.040 |
| `MatMulV2_137` | 0 | 90.860 |
| `MatMulV2_77` | 0 | 90.800 |
| `MatMulV2_17` | 0 | 90.720 |
| `MatMulV2_125` | 0 | 90.680 |
| `MatMulV2_113` | 0 | 90.660 |
| `MatMulV2_101` | 0 | 90.560 |
| `MatMulV2_17` | 0 | 90.320 |
| `MatMulV2_125` | 0 | 90.300 |
| `MatMulV2_77` | 0 | 90.280 |
| `MatMulV2_65` | 0 | 90.240 |
| `MatMulV2_5` | 0 | 90.120 |
| `MatMulV2_113` | 0 | 90.120 |
| `MatMulV2_143` | 0 | 90.000 |
| `MatMulV2_125` | 0 | 89.920 |
| `MatMulV2_161` | 0 | 89.880 |
| `MatMulV2_143` | 0 | 89.720 |
| `MatMulV2_161` | 0 | 89.600 |
| `MatMulV2_137` | 0 | 89.420 |
| `MatMulV2_17` | 0 | 89.280 |
| `MatMulV2_95` | 0 | 89.140 |
| `MatMulV2_119` | 0 | 89.040 |
| `MatMulV2_35` | 0 | 88.760 |
| `MatMulV2_101` | 0 | 88.740 |
| `MatMulV2_59` | 0 | 88.620 |
| `MatMulV2_101` | 0 | 88.260 |
| `MatMulV2_59` | 0 | 88.220 |
| `MatMulV2_89` | 0 | 88.140 |
| `MatMulV2_119` | 0 | 87.960 |
| `MatMulV2_59` | 0 | 87.900 |
| `MatMulV2_89` | 0 | 87.880 |
| `MatMulV2_119` | 0 | 87.800 |
| `MatMulV2_29` | 0 | 87.380 |
| `MatMulV2_29` | 0 | 86.960 |
| `MatMulV2_11` | 0 | 86.480 |
| `MatMulV2_29` | 0 | 85.620 |
| `MatMulV2_23` | 0 | 85.560 |
| `MatMulV2_22` | 0 | 85.400 |
| `MatMulV2_64` | 0 | 84.680 |
| `MatMulV2_76` | 0 | 84.140 |
| `MatMulV2_40` | 0 | 84.120 |
| `MatMulV2_124` | 0 | 83.760 |
| `MatMulV2_40` | 0 | 83.720 |
| `MatMulV2_94` | 0 | 83.680 |
| `MatMulV2_70` | 0 | 83.660 |
| `MatMulV2_40` | 0 | 83.640 |
| `MatMulV2_76` | 0 | 83.620 |
| `MatMulV2_148` | 0 | 83.540 |
| `MatMulV2_112` | 0 | 83.540 |
| `MatMulV2_112` | 0 | 83.540 |
| `MatMulV2_124` | 0 | 83.500 |
| `MatMulV2_64` | 0 | 83.440 |
| `MatMulV2_70` | 0 | 83.380 |
| `MatMulV2_112` | 0 | 83.360 |
| `MatMulV2_88` | 0 | 83.160 |
| `MatMulV2_124` | 0 | 83.120 |
| `MatMulV2_88` | 0 | 82.820 |
| `MatMulV2_136` | 0 | 82.820 |
| `MatMulV2_64` | 0 | 82.820 |
| `MatMulV2_82` | 0 | 82.820 |
| `MatMulV2_22` | 0 | 82.760 |
| `MatMulV2_100` | 0 | 82.720 |
| `MatMulV2_154` | 0 | 82.540 |
| `MatMulV2_154` | 0 | 82.480 |
| `MatMulV2_4` | 0 | 82.460 |
| `MatMulV2_82` | 0 | 82.380 |
| `MatMulV2_28` | 0 | 82.380 |
| `MatMulV2_100` | 0 | 82.240 |
| `MatMulV2_118` | 0 | 82.220 |
| `MatMulV2_136` | 0 | 82.220 |
| `MatMulV2_70` | 0 | 82.180 |
| `MatMulV2_130` | 0 | 82.160 |
| `MatMulV2_52` | 0 | 82.140 |
| `MatMulV2_160` | 0 | 82.080 |
| `MatMulV2_136` | 0 | 81.980 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 33364.210 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.fractal_nz.torchair.active.step1` | 1 | 32109.070 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.fractal_nz.torchair.active.step3` | 1 | 31866.010 |
| `paddleocr_vl.vision_matmul_lab.S2048.I4352.fractal_nz.torchair.active.step2` | 1 | 31860.890 |
| `TorchDynamo Cache Lookup` | 3 | 30510.480 |
| `Torch-Compiled Region: 0/0` | 3 | 3676.700 |
| `TorchNpuGraphBase::Run` | 3 | 2718.930 |
| `RefreshAtTensorFromGeTensor` | 3 | 1177.130 |
| `aten::empty` | 3 | 583.480 |
| `ExecuteGraph` | 3 | 487.350 |
| `AssembleInputs` | 3 | 384.190 |
| `empty_tensor` | 3 | 290.410 |
| `AssembleOutputs` | 3 | 287.790 |
| `aten::set_` | 3 | 280.060 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 221476.130 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 90167.410 |
| `launch` | 976 | 17977.150 |
| `InputCopy` | 3 | 148.760 |
| `ModelExecute` | 3 | 50.750 |
| `step_info` | 6 | 29.690 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 27.330 |
| `OutputCopy` | 3 | 0.730 |

