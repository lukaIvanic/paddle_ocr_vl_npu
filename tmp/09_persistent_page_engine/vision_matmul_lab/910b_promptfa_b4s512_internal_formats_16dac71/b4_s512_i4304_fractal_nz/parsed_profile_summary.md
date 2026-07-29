# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4304_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4304_fractal_nz/liteserver-c001-4_641525_20260729140043625_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `81318.860 us`
- `Free`: `3484.260 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3395.500 us`
- `Stage`: `84803.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 24073.680 |
| `Transpose` | 324 | 11739.280 |
| `StridedSliceD` | 405 | 11654.200 |
| `PromptFlashAttention` | 81 | 8589.260 |
| `PadV3` | 243 | 4475.340 |
| `AddLayerNorm` | 162 | 4004.060 |
| `ConcatV2D` | 243 | 3945.460 |
| `Gelu` | 81 | 3096.200 |
| `Mul` | 324 | 2919.660 |
| `Add` | 162 | 1977.540 |
| `Cast` | 162 | 1666.200 |
| `Neg` | 162 | 1488.960 |
| `SplitVD` | 81 | 1347.440 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 238.000 |
| `LayerNormV3` | 3 | 88.040 |
| `Data` | 3 | 15.540 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 3 | 338.780 |
| `PromptFlashAttention` | 3 | 336.600 |
| `PromptFlashAttention_9` | 3 | 335.880 |
| `PromptFlashAttention_25` | 3 | 334.540 |
| `PromptFlashAttention_13` | 3 | 334.420 |
| `PromptFlashAttention_5` | 3 | 330.480 |
| `PromptFlashAttention_8` | 3 | 329.680 |
| `PromptFlashAttention_16` | 3 | 323.560 |
| `PromptFlashAttention_23` | 3 | 321.040 |
| `MatMulV2_125` | 3 | 319.620 |
| `PromptFlashAttention_26` | 3 | 319.520 |
| `PromptFlashAttention_1` | 3 | 319.480 |
| `PromptFlashAttention_10` | 3 | 317.580 |
| `PromptFlashAttention_7` | 3 | 315.940 |
| `PromptFlashAttention_22` | 3 | 314.760 |
| `PromptFlashAttention_14` | 3 | 314.320 |
| `PromptFlashAttention_12` | 3 | 313.940 |
| `PromptFlashAttention_17` | 3 | 313.880 |
| `MatMulV2_83` | 3 | 313.440 |
| `MatMulV2_41` | 3 | 312.800 |
| `PromptFlashAttention_19` | 3 | 312.520 |
| `MatMulV2_119` | 3 | 312.140 |
| `PromptFlashAttention_6` | 3 | 311.940 |
| `MatMulV2_77` | 3 | 311.200 |
| `PromptFlashAttention_21` | 3 | 310.980 |
| `MatMulV2_155` | 3 | 310.920 |
| `MatMulV2_89` | 3 | 310.860 |
| `PromptFlashAttention_24` | 3 | 310.800 |
| `MatMulV2_23` | 3 | 310.540 |
| `MatMulV2_113` | 3 | 310.280 |
| `MatMulV2_161` | 3 | 310.260 |
| `MatMulV2_131` | 3 | 310.240 |
| `MatMulV2_107` | 3 | 309.520 |
| `MatMulV2_29` | 3 | 309.200 |
| `MatMulV2_137` | 3 | 308.860 |
| `MatMulV2_65` | 3 | 308.840 |
| `MatMulV2_71` | 3 | 308.800 |
| `MatMulV2_143` | 3 | 308.620 |
| `MatMulV2_17` | 3 | 308.340 |
| `MatMulV2_11` | 3 | 308.200 |
| `MatMulV2_95` | 3 | 308.200 |
| `MatMulV2_47` | 3 | 308.060 |
| `MatMulV2_101` | 3 | 307.880 |
| `PromptFlashAttention_11` | 3 | 307.620 |
| `MatMulV2_53` | 3 | 307.320 |
| `PromptFlashAttention_20` | 3 | 306.740 |
| `MatMulV2_5` | 3 | 306.640 |
| `PromptFlashAttention_3` | 3 | 306.380 |
| `MatMulV2_59` | 3 | 306.320 |
| `PromptFlashAttention_4` | 3 | 305.500 |
| `MatMulV2_35` | 3 | 304.580 |
| `MatMulV2_149` | 3 | 303.040 |
| `PromptFlashAttention_2` | 3 | 302.740 |
| `PromptFlashAttention_18` | 3 | 299.640 |
| `MatMulV2_64` | 3 | 282.520 |
| `MatMulV2_100` | 3 | 282.400 |
| `MatMulV2_58` | 3 | 282.220 |
| `MatMulV2_154` | 3 | 281.800 |
| `MatMulV2_136` | 3 | 280.620 |
| `MatMulV2_160` | 3 | 280.440 |
| `MatMulV2_112` | 3 | 280.400 |
| `MatMulV2_118` | 3 | 280.300 |
| `MatMulV2_148` | 3 | 280.300 |
| `MatMulV2_82` | 3 | 280.260 |
| `MatMulV2_94` | 3 | 280.240 |
| `MatMulV2_88` | 3 | 280.020 |
| `MatMulV2_130` | 3 | 279.700 |
| `MatMulV2_70` | 3 | 279.500 |
| `MatMulV2_124` | 3 | 279.380 |
| `MatMulV2_52` | 3 | 278.760 |
| `MatMulV2_46` | 3 | 278.640 |
| `MatMulV2_22` | 3 | 278.620 |
| `MatMulV2_34` | 3 | 277.580 |
| `MatMulV2_142` | 3 | 277.440 |
| `MatMulV2_16` | 3 | 277.160 |
| `MatMulV2_40` | 3 | 277.120 |
| `MatMulV2_10` | 3 | 276.900 |
| `MatMulV2_28` | 3 | 275.800 |
| `MatMulV2_76` | 3 | 275.800 |
| `MatMulV2_4` | 3 | 271.960 |
| `MatMulV2_106` | 3 | 270.900 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 238.000 |
| `Transpose_206` | 3 | 141.340 |
| `Transpose_146` | 3 | 140.260 |
| `Transpose_65` | 3 | 123.300 |
| `Transpose_216` | 3 | 122.520 |
| `Transpose_46` | 3 | 122.320 |
| `Transpose_56` | 3 | 122.300 |
| `Transpose_156` | 3 | 121.640 |
| `Transpose_96` | 3 | 121.620 |
| `Transpose_256` | 3 | 121.620 |
| `Transpose_246` | 3 | 121.560 |
| `Transpose_103` | 3 | 121.500 |
| `Transpose_86` | 3 | 121.360 |
| `Transpose_266` | 3 | 121.180 |
| `Transpose_106` | 3 | 121.100 |
| `Transpose_126` | 3 | 120.840 |
| `Transpose_236` | 3 | 120.740 |
| `Transpose_36` | 3 | 120.700 |
| `Transpose_226` | 3 | 120.660 |
| `Transpose_26` | 3 | 120.640 |
| `Transpose_196` | 3 | 120.640 |
| `Transpose_136` | 3 | 120.580 |
| `Transpose_16` | 3 | 120.560 |
| `Transpose_5` | 3 | 120.260 |
| `Transpose_166` | 3 | 120.240 |
| `Transpose_186` | 3 | 120.240 |
| `Transpose_104` | 3 | 120.140 |
| `Transpose_116` | 3 | 119.980 |
| `Transpose_6` | 3 | 119.840 |
| `Transpose_76` | 3 | 119.760 |
| `Transpose_66` | 3 | 119.720 |
| `Transpose_44` | 3 | 119.440 |
| `Transpose_176` | 3 | 119.080 |
| `Gelu_25` | 3 | 115.080 |
| `Gelu_22` | 3 | 115.060 |
| `Gelu` | 3 | 114.960 |
| `Gelu_4` | 3 | 114.940 |
| `Gelu_16` | 3 | 114.940 |
| `Gelu_26` | 3 | 114.900 |
| `Gelu_13` | 3 | 114.880 |
| `Gelu_23` | 3 | 114.840 |
| `Gelu_5` | 3 | 114.820 |
| `Gelu_10` | 3 | 114.820 |
| `Gelu_9` | 3 | 114.800 |
| `Gelu_18` | 3 | 114.800 |
| `Gelu_8` | 3 | 114.720 |
| `Gelu_14` | 3 | 114.720 |
| `Gelu_20` | 3 | 114.720 |
| `Gelu_2` | 3 | 114.700 |
| `Gelu_17` | 3 | 114.520 |
| `Gelu_19` | 3 | 114.500 |
| `Gelu_3` | 3 | 114.480 |
| `Gelu_1` | 3 | 114.460 |
| `Gelu_11` | 3 | 114.440 |
| `Gelu_7` | 3 | 114.440 |
| `Gelu_15` | 3 | 114.400 |
| `Gelu_24` | 3 | 114.380 |
| `Gelu_6` | 3 | 114.360 |
| `Gelu_12` | 3 | 114.340 |
| `Gelu_21` | 3 | 114.180 |
| `StridedSliceV2_119` | 3 | 112.020 |
| `Transpose_155` | 3 | 108.640 |
| `Transpose_45` | 3 | 108.560 |
| `Transpose_215` | 3 | 108.260 |
| `Transpose_205` | 3 | 108.260 |
| `Transpose_55` | 3 | 108.220 |
| `Transpose_105` | 3 | 108.140 |
| `Transpose_255` | 3 | 108.080 |
| `Transpose_35` | 3 | 108.060 |
| `Transpose_95` | 3 | 108.060 |
| `Transpose_145` | 3 | 108.020 |
| `Transpose_265` | 3 | 107.960 |
| `Transpose_165` | 3 | 107.780 |
| `Transpose_195` | 3 | 107.780 |
| `Transpose_245` | 3 | 107.720 |
| `Transpose_85` | 3 | 107.700 |
| `Transpose_115` | 3 | 107.620 |
| `Transpose_175` | 3 | 107.620 |
| `Transpose_225` | 3 | 107.580 |
| `Transpose_125` | 3 | 107.500 |
| `Transpose_25` | 3 | 107.460 |
| `Transpose_185` | 3 | 107.320 |
| `Transpose_135` | 3 | 107.240 |
| `Transpose_75` | 3 | 107.180 |
| `Transpose_15` | 3 | 107.000 |
| `Transpose_235` | 3 | 106.780 |
| `Transpose_233` | 3 | 104.800 |
| `Transpose_3` | 3 | 103.960 |
| `Transpose_53` | 3 | 103.060 |
| `Transpose_153` | 3 | 102.740 |
| `Transpose_33` | 3 | 102.680 |
| `Transpose_193` | 3 | 102.540 |
| `Transpose_43` | 3 | 102.380 |
| `Transpose_203` | 3 | 102.360 |
| `Transpose_133` | 3 | 102.300 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 102.220 |
| `Transpose_83` | 3 | 102.200 |
| `Transpose_213` | 3 | 102.200 |
| `Transpose_113` | 3 | 102.140 |
| `Transpose_243` | 3 | 102.040 |
| `Transpose_143` | 3 | 101.960 |
| `Transpose_93` | 3 | 101.960 |
| `Transpose_253` | 3 | 101.940 |
| `Transpose_13` | 3 | 101.900 |
| `Transpose_173` | 3 | 101.740 |
| `Transpose_73` | 3 | 101.600 |
| `Transpose_183` | 3 | 101.600 |
| `Transpose_163` | 3 | 101.500 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 101.480 |
| `Transpose_123` | 3 | 101.460 |
| `Transpose_263` | 3 | 101.340 |
| `Transpose_23` | 3 | 101.320 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 101.300 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 101.300 |
| `Transpose_134` | 3 | 101.060 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 101.040 |
| `Transpose_63` | 3 | 100.880 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 100.800 |
| `Transpose_144` | 3 | 100.800 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "4,512,16,72" -> "4,512,16,36" | ND -> ND` | 324 | 9012.360 |
| `PromptFlashAttention | "4,16,512,80;4,16,512,80;4,16,512,80;4,1,512,512" -> "4,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 81 | 8589.260 |
| `Transpose | "4,512,16,72;4" -> "4,16,512,72" | ND;ND -> ND` | 243 | 8436.240 |
| `MatMulV2 | "2048,4304;269,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 8354.720 |
| `MatMulV2 | "2048,1152;72,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 8192.180 |
| `MatMulV2 | "2048,1152;72,269,16,16;4304" -> "2048,4304" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7526.780 |
| `PadV3 | "4,16,512,72;8;" -> "4,16,512,80" | NCHW;NCHW;NCHW -> NCHW` | 243 | 4475.340 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4004.060 |
| `Transpose | "4,16,512,72;4" -> "4,512,16,72" | ND;ND -> ND` | 81 | 3303.040 |
| `Gelu | "4,512,4304" -> "4,512,4304" | ND -> ND` | 81 | 3096.200 |
| `ConcatV2D | "4,512,16,36;4,512,16,36" -> "4,512,16,72" | ND;ND -> ND` | 162 | 2933.140 |
| `Mul | "4,512,16,72;4,512,1,72" -> "4,512,16,72" | ND;ND -> ND` | 324 | 2919.660 |
| `StridedSliceD | "4,16,512,80" -> "4,16,512,72" | NCHW -> NCHW` | 81 | 2641.840 |
| `Add | "4,512,16,72;4,512,16,72" -> "4,512,16,72" | ND;ND -> ND` | 162 | 1977.540 |
| `Cast | "4,512,16,72" -> "4,512,16,72" | ND -> ND` | 162 | 1666.200 |
| `Neg | "4,512,16,36" -> "4,512,16,36" | ND -> ND` | 162 | 1488.960 |
| `SplitVD | "4,512,3456" -> "4,512,1152;4,512,1152;4,512,1152" | ND -> ND;ND;ND` | 81 | 1347.440 |
| `ConcatV2D | "4,512,1152;4,512,1152;4,512,1152" -> "4,512,3456" | ND;ND;ND -> ND` | 81 | 1012.320 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 238.000 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 88.040 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.540 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;FRACTAL_NZ;ND` | 486 | 24073.680 |
| `ND;ND` | 972 | 19569.620 |
| `ND` | 810 | 16611.160 |
| `NCHW;NCHW;NCHW;NCHW` | 81 | 8589.260 |
| `NCHW;NCHW;NCHW` | 243 | 4475.340 |
| `ND;ND;ND;ND` | 162 | 4004.060 |
| `NCHW` | 81 | 2641.840 |
| `ND;ND;ND` | 84 | 1100.360 |
| `N/A` | 246 | 253.540 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 0 | 113.560 |
| `PromptFlashAttention` | 0 | 113.140 |
| `PromptFlashAttention_25` | 0 | 112.820 |
| `PromptFlashAttention_15` | 0 | 112.800 |
| `PromptFlashAttention_15` | 0 | 112.420 |
| `PromptFlashAttention_13` | 0 | 112.420 |
| `PromptFlashAttention_9` | 0 | 112.240 |
| `PromptFlashAttention` | 0 | 112.180 |
| `PromptFlashAttention_9` | 0 | 112.060 |
| `PromptFlashAttention_13` | 0 | 111.920 |
| `PromptFlashAttention_9` | 0 | 111.580 |
| `PromptFlashAttention` | 0 | 111.280 |
| `PromptFlashAttention_5` | 0 | 111.140 |
| `PromptFlashAttention_25` | 0 | 111.080 |
| `PromptFlashAttention_5` | 0 | 110.820 |
| `PromptFlashAttention_25` | 0 | 110.640 |
| `PromptFlashAttention_8` | 0 | 110.440 |
| `PromptFlashAttention_8` | 0 | 110.440 |
| `PromptFlashAttention_13` | 0 | 110.080 |
| `PromptFlashAttention_16` | 0 | 109.000 |
| `PromptFlashAttention_8` | 0 | 108.800 |
| `PromptFlashAttention_5` | 0 | 108.520 |
| `PromptFlashAttention_16` | 0 | 107.900 |
| `PromptFlashAttention_10` | 0 | 107.660 |
| `PromptFlashAttention_23` | 0 | 107.520 |
| `MatMulV2_125` | 0 | 106.940 |
| `PromptFlashAttention_1` | 0 | 106.780 |
| `PromptFlashAttention_23` | 0 | 106.780 |
| `PromptFlashAttention_26` | 0 | 106.760 |
| `PromptFlashAttention_23` | 0 | 106.740 |
| `PromptFlashAttention_26` | 0 | 106.740 |
| `PromptFlashAttention_1` | 0 | 106.660 |
| `PromptFlashAttention_16` | 0 | 106.660 |
| `MatMulV2_125` | 0 | 106.620 |
| `MatMulV2_125` | 0 | 106.060 |
| `PromptFlashAttention_1` | 0 | 106.040 |
| `PromptFlashAttention_26` | 0 | 106.020 |
| `PromptFlashAttention_7` | 0 | 105.840 |
| `PromptFlashAttention_22` | 0 | 105.780 |
| `PromptFlashAttention_7` | 0 | 105.460 |
| `PromptFlashAttention_12` | 0 | 105.460 |
| `MatMulV2_155` | 0 | 105.380 |
| `PromptFlashAttention_22` | 0 | 105.320 |
| `PromptFlashAttention_14` | 0 | 105.300 |
| `MatMulV2_29` | 0 | 105.260 |
| `PromptFlashAttention_17` | 0 | 105.220 |
| `PromptFlashAttention_21` | 0 | 105.160 |
| `MatMulV2_29` | 0 | 105.100 |
| `PromptFlashAttention_10` | 0 | 105.080 |
| `MatMulV2_23` | 0 | 105.040 |
| `MatMulV2_83` | 0 | 104.860 |
| `PromptFlashAttention_10` | 0 | 104.840 |
| `MatMulV2_41` | 0 | 104.820 |
| `MatMulV2_119` | 0 | 104.780 |
| `MatMulV2_107` | 0 | 104.780 |
| `MatMulV2_77` | 0 | 104.760 |
| `PromptFlashAttention_14` | 0 | 104.740 |
| `MatMulV2_83` | 0 | 104.740 |
| `PromptFlashAttention_7` | 0 | 104.640 |
| `MatMulV2_65` | 0 | 104.600 |
| `MatMulV2_89` | 0 | 104.580 |
| `MatMulV2_161` | 0 | 104.500 |
| `PromptFlashAttention_19` | 0 | 104.480 |
| `PromptFlashAttention_17` | 0 | 104.380 |
| `PromptFlashAttention_12` | 0 | 104.300 |
| `PromptFlashAttention_14` | 0 | 104.280 |
| `PromptFlashAttention_17` | 0 | 104.280 |
| `PromptFlashAttention_6` | 0 | 104.200 |
| `PromptFlashAttention_12` | 0 | 104.180 |
| `PromptFlashAttention_6` | 0 | 104.180 |
| `PromptFlashAttention_19` | 0 | 104.100 |
| `PromptFlashAttention_24` | 0 | 104.100 |
| `MatMulV2_143` | 0 | 104.080 |
| `MatMulV2_41` | 0 | 104.040 |
| `MatMulV2_17` | 0 | 104.020 |
| `MatMulV2_155` | 0 | 104.020 |
| `MatMulV2_131` | 0 | 103.980 |
| `MatMulV2_41` | 0 | 103.940 |
| `PromptFlashAttention_19` | 0 | 103.940 |
| `MatMulV2_161` | 0 | 103.920 |
| `MatMulV2_119` | 0 | 103.900 |
| `MatMulV2_23` | 0 | 103.860 |
| `MatMulV2_83` | 0 | 103.840 |
| `MatMulV2_17` | 0 | 103.820 |
| `MatMulV2_59` | 0 | 103.780 |
| `MatMulV2_47` | 0 | 103.660 |
| `MatMulV2_113` | 0 | 103.660 |
| `PromptFlashAttention_22` | 0 | 103.660 |
| `MatMulV2_113` | 0 | 103.620 |
| `MatMulV2_65` | 0 | 103.620 |
| `PromptFlashAttention_24` | 0 | 103.600 |
| `MatMulV2_53` | 0 | 103.560 |
| `PromptFlashAttention_6` | 0 | 103.560 |
| `MatMulV2_71` | 0 | 103.520 |
| `MatMulV2_119` | 0 | 103.460 |
| `MatMulV2_131` | 0 | 103.420 |
| `MatMulV2_137` | 0 | 103.420 |
| `MatMulV2_77` | 0 | 103.320 |
| `PromptFlashAttention_20` | 0 | 103.300 |
| `MatMulV2_101` | 0 | 103.240 |
| `PromptFlashAttention_21` | 0 | 103.240 |
| `MatMulV2_137` | 0 | 103.220 |
| `MatMulV2_71` | 0 | 103.180 |
| `MatMulV2_89` | 0 | 103.180 |
| `MatMulV2_77` | 0 | 103.120 |
| `MatMulV2_89` | 0 | 103.100 |
| `PromptFlashAttention_24` | 0 | 103.100 |
| `MatMulV2_95` | 0 | 103.040 |
| `MatMulV2_11` | 0 | 103.020 |
| `MatMulV2_113` | 0 | 103.000 |
| `MatMulV2_35` | 0 | 102.960 |
| `PromptFlashAttention_11` | 0 | 102.940 |
| `MatMulV2_95` | 0 | 102.940 |
| `MatMulV2_11` | 0 | 102.920 |
| `PromptFlashAttention_11` | 0 | 102.900 |
| `MatMulV2_131` | 0 | 102.840 |
| `MatMulV2_101` | 0 | 102.760 |
| `MatMulV2_5` | 0 | 102.720 |
| `PromptFlashAttention_3` | 0 | 102.620 |
| `MatMulV2_107` | 0 | 102.620 |
| `PromptFlashAttention_21` | 0 | 102.580 |
| `MatMulV2_5` | 0 | 102.580 |
| `PromptFlashAttention_20` | 0 | 102.560 |
| `MatMulV2_53` | 0 | 102.560 |
| `MatMulV2_143` | 0 | 102.480 |
| `PromptFlashAttention_4` | 0 | 102.460 |
| `MatMulV2_11` | 0 | 102.260 |
| `MatMulV2_149` | 0 | 102.260 |
| `MatMulV2_47` | 0 | 102.220 |
| `MatMulV2_95` | 0 | 102.220 |
| `MatMulV2_137` | 0 | 102.220 |
| `MatMulV2_47` | 0 | 102.180 |
| `MatMulV2_107` | 0 | 102.120 |
| `MatMulV2_71` | 0 | 102.100 |
| `PromptFlashAttention_3` | 0 | 102.080 |
| `MatMulV2_59` | 0 | 102.060 |
| `MatMulV2_143` | 0 | 102.060 |
| `MatMulV2_101` | 0 | 101.880 |
| `MatMulV2_161` | 0 | 101.840 |
| `PromptFlashAttention_11` | 0 | 101.780 |
| `PromptFlashAttention_4` | 0 | 101.780 |
| `PromptFlashAttention_3` | 0 | 101.680 |
| `MatMulV2_23` | 0 | 101.640 |
| `MatMulV2_155` | 0 | 101.520 |
| `MatMulV2_149` | 0 | 101.460 |
| `MatMulV2_5` | 0 | 101.340 |
| `PromptFlashAttention_2` | 0 | 101.300 |
| `PromptFlashAttention_4` | 0 | 101.260 |
| `MatMulV2_53` | 0 | 101.200 |
| `MatMulV2_35` | 0 | 101.120 |
| `PromptFlashAttention_2` | 0 | 101.080 |
| `PromptFlashAttention_20` | 0 | 100.880 |
| `MatMulV2_65` | 0 | 100.620 |
| `MatMulV2_17` | 0 | 100.500 |
| `MatMulV2_35` | 0 | 100.500 |
| `MatMulV2_59` | 0 | 100.480 |
| `PromptFlashAttention_2` | 0 | 100.360 |
| `PromptFlashAttention_18` | 0 | 100.340 |
| `PromptFlashAttention_18` | 0 | 99.980 |
| `PromptFlashAttention_18` | 0 | 99.320 |
| `MatMulV2_149` | 0 | 99.320 |
| `MatMulV2_29` | 0 | 98.840 |
| `MatMulV2_154` | 0 | 94.620 |
| `MatMulV2_58` | 0 | 94.380 |
| `MatMulV2_100` | 0 | 94.360 |
| `MatMulV2_64` | 0 | 94.320 |
| `MatMulV2_64` | 0 | 94.160 |
| `MatMulV2_58` | 0 | 94.080 |
| `MatMulV2_100` | 0 | 94.060 |
| `MatMulV2_64` | 0 | 94.040 |
| `MatMulV2_100` | 0 | 93.980 |
| `MatMulV2_58` | 0 | 93.760 |
| `MatMulV2_94` | 0 | 93.760 |
| `MatMulV2_148` | 0 | 93.740 |
| `MatMulV2_118` | 0 | 93.720 |
| `MatMulV2_136` | 0 | 93.680 |
| `MatMulV2_154` | 0 | 93.680 |
| `MatMulV2_112` | 0 | 93.660 |
| `MatMulV2_46` | 0 | 93.620 |
| `MatMulV2_160` | 0 | 93.620 |
| `MatMulV2_136` | 0 | 93.600 |
| `MatMulV2_82` | 0 | 93.580 |
| `MatMulV2_112` | 0 | 93.580 |
| `MatMulV2_88` | 0 | 93.540 |
| `MatMulV2_160` | 0 | 93.520 |
| `MatMulV2_154` | 0 | 93.500 |
| `MatMulV2_130` | 0 | 93.460 |
| `MatMulV2_16` | 0 | 93.460 |
| `MatMulV2_82` | 0 | 93.460 |
| `MatMulV2_118` | 0 | 93.460 |
| `MatMulV2_40` | 0 | 93.440 |
| `MatMulV2_148` | 0 | 93.420 |
| `MatMulV2_28` | 0 | 93.340 |
| `MatMulV2_136` | 0 | 93.340 |
| `MatMulV2_124` | 0 | 93.320 |
| `MatMulV2_160` | 0 | 93.300 |
| `MatMulV2_130` | 0 | 93.300 |
| `MatMulV2_88` | 0 | 93.280 |
| `MatMulV2_16` | 0 | 93.280 |
| `MatMulV2_94` | 0 | 93.260 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 29939.540 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.fractal_nz.torchair.active.step1` | 1 | 28765.950 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.fractal_nz.torchair.active.step2` | 1 | 28492.480 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.fractal_nz.torchair.active.step3` | 1 | 28472.060 |
| `TorchDynamo Cache Lookup` | 3 | 27154.930 |
| `Torch-Compiled Region: 0/0` | 3 | 3624.530 |
| `TorchNpuGraphBase::Run` | 3 | 2659.810 |
| `RefreshAtTensorFromGeTensor` | 3 | 1135.790 |
| `aten::empty` | 3 | 544.480 |
| `ExecuteGraph` | 3 | 463.910 |
| `AssembleInputs` | 3 | 383.120 |
| `AssembleOutputs` | 3 | 289.670 |
| `aten::set_` | 3 | 274.440 |
| `empty_tensor` | 3 | 270.390 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 217652.300 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 80143.760 |
| `launch` | 976 | 17120.840 |
| `InputCopy` | 3 | 136.290 |
| `ModelExecute` | 3 | 50.370 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 33.920 |
| `step_info` | 6 | 32.170 |
| `OutputCopy` | 3 | 1.010 |

