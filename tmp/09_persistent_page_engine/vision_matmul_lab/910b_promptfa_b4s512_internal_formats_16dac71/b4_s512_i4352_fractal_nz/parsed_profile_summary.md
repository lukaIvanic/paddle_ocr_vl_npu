# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4352_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4352_fractal_nz/liteserver-c001-4_644528_20260729140259010_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `79663.200 us`
- `Free`: `3458.900 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3324.000 us`
- `Stage`: `83121.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 22038.160 |
| `StridedSliceD` | 405 | 11805.700 |
| `Transpose` | 324 | 11424.120 |
| `PromptFlashAttention` | 81 | 8581.860 |
| `PadV3` | 243 | 4921.760 |
| `AddLayerNorm` | 162 | 3990.560 |
| `ConcatV2D` | 243 | 3936.000 |
| `Gelu` | 81 | 3181.780 |
| `Mul` | 324 | 2950.860 |
| `Add` | 162 | 1981.420 |
| `Cast` | 162 | 1662.500 |
| `Neg` | 162 | 1494.220 |
| `SplitVD` | 81 | 1344.480 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 239.040 |
| `LayerNormV3` | 3 | 95.740 |
| `Data` | 3 | 15.000 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 3 | 342.780 |
| `PromptFlashAttention_25` | 3 | 336.960 |
| `PromptFlashAttention_9` | 3 | 334.120 |
| `PromptFlashAttention_8` | 3 | 330.180 |
| `PromptFlashAttention_15` | 3 | 328.860 |
| `PromptFlashAttention_13` | 3 | 325.660 |
| `PromptFlashAttention_24` | 3 | 324.060 |
| `PromptFlashAttention_16` | 3 | 322.620 |
| `PromptFlashAttention_1` | 3 | 321.480 |
| `PromptFlashAttention_23` | 3 | 319.240 |
| `PromptFlashAttention_26` | 3 | 319.180 |
| `PromptFlashAttention_11` | 3 | 318.320 |
| `PromptFlashAttention_14` | 3 | 317.940 |
| `PromptFlashAttention_6` | 3 | 317.400 |
| `PromptFlashAttention_7` | 3 | 313.880 |
| `PromptFlashAttention_5` | 3 | 313.480 |
| `PromptFlashAttention_12` | 3 | 313.180 |
| `PromptFlashAttention_10` | 3 | 312.760 |
| `PromptFlashAttention_21` | 3 | 312.300 |
| `PromptFlashAttention_17` | 3 | 312.020 |
| `PromptFlashAttention_20` | 3 | 310.780 |
| `PromptFlashAttention_3` | 3 | 309.560 |
| `PromptFlashAttention_4` | 3 | 307.500 |
| `PromptFlashAttention_22` | 3 | 307.200 |
| `PromptFlashAttention_19` | 3 | 306.000 |
| `PromptFlashAttention_2` | 3 | 304.340 |
| `PromptFlashAttention_18` | 3 | 300.060 |
| `MatMulV2_101` | 3 | 281.160 |
| `MatMulV2_137` | 3 | 280.340 |
| `MatMulV2_53` | 3 | 279.460 |
| `MatMulV2_47` | 3 | 278.680 |
| `MatMulV2_143` | 3 | 278.240 |
| `MatMulV2_41` | 3 | 277.300 |
| `MatMulV2_149` | 3 | 275.960 |
| `MatMulV2_107` | 3 | 274.540 |
| `MatMulV2_71` | 3 | 274.240 |
| `MatMulV2_125` | 3 | 274.100 |
| `MatMulV2_77` | 3 | 274.020 |
| `MatMulV2_89` | 3 | 273.540 |
| `MatMulV2_155` | 3 | 272.640 |
| `MatMulV2_83` | 3 | 272.400 |
| `MatMulV2_113` | 3 | 272.200 |
| `MatMulV2_95` | 3 | 271.540 |
| `MatMulV2_119` | 3 | 268.540 |
| `MatMulV2_131` | 3 | 267.480 |
| `MatMulV2_59` | 3 | 267.340 |
| `MatMulV2_161` | 3 | 266.720 |
| `MatMulV2_17` | 3 | 266.400 |
| `MatMulV2_65` | 3 | 266.380 |
| `MatMulV2_5` | 3 | 265.720 |
| `MatMulV2_11` | 3 | 265.620 |
| `MatMulV2_29` | 3 | 264.960 |
| `MatMulV2_23` | 3 | 263.940 |
| `MatMulV2_35` | 3 | 260.200 |
| `MatMulV2_118` | 3 | 246.020 |
| `MatMulV2_106` | 3 | 245.420 |
| `MatMulV2_124` | 3 | 244.060 |
| `MatMulV2_160` | 3 | 244.020 |
| `MatMulV2_70` | 3 | 243.500 |
| `MatMulV2_16` | 3 | 243.280 |
| `MatMulV2_22` | 3 | 242.300 |
| `MatMulV2_130` | 3 | 242.060 |
| `MatMulV2_58` | 3 | 241.720 |
| `MatMulV2_154` | 3 | 241.700 |
| `MatMulV2_82` | 3 | 241.560 |
| `MatMulV2_52` | 3 | 241.540 |
| `MatMulV2_46` | 3 | 241.420 |
| `MatMulV2_76` | 3 | 240.800 |
| `MatMulV2_112` | 3 | 240.700 |
| `MatMulV2_100` | 3 | 240.620 |
| `MatMulV2_28` | 3 | 240.600 |
| `MatMulV2_148` | 3 | 240.320 |
| `MatMulV2_34` | 3 | 240.260 |
| `MatMulV2_88` | 3 | 239.960 |
| `MatMulV2_94` | 3 | 239.460 |
| `MatMulV2_40` | 3 | 239.080 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 239.040 |
| `MatMulV2_64` | 3 | 238.980 |
| `MatMulV2_10` | 3 | 238.460 |
| `MatMulV2_136` | 3 | 238.340 |
| `MatMulV2_4` | 3 | 238.140 |
| `MatMulV2_142` | 3 | 238.120 |
| `Gelu_2` | 3 | 135.580 |
| `Transpose_226` | 3 | 133.480 |
| `Transpose_33` | 3 | 121.760 |
| `Transpose_93` | 3 | 121.240 |
| `Transpose_34` | 3 | 119.760 |
| `Transpose_186` | 3 | 117.900 |
| `Gelu_10` | 3 | 117.780 |
| `Gelu_26` | 3 | 117.780 |
| `Gelu` | 3 | 117.720 |
| `Transpose_26` | 3 | 117.680 |
| `Transpose_36` | 3 | 117.600 |
| `Gelu_14` | 3 | 117.580 |
| `Gelu_15` | 3 | 117.580 |
| `Gelu_5` | 3 | 117.480 |
| `Gelu_9` | 3 | 117.440 |
| `Transpose_176` | 3 | 117.400 |
| `Gelu_21` | 3 | 117.360 |
| `Gelu_3` | 3 | 117.280 |
| `Transpose_196` | 3 | 117.220 |
| `Gelu_19` | 3 | 117.160 |
| `Gelu_6` | 3 | 117.160 |
| `Transpose_66` | 3 | 117.140 |
| `Gelu_24` | 3 | 117.140 |
| `Gelu_18` | 3 | 117.060 |
| `Gelu_25` | 3 | 117.040 |
| `Transpose_236` | 3 | 117.020 |
| `Gelu_23` | 3 | 116.980 |
| `Gelu_22` | 3 | 116.960 |
| `Gelu_17` | 3 | 116.940 |
| `Transpose_146` | 3 | 116.920 |
| `Gelu_4` | 3 | 116.900 |
| `Gelu_11` | 3 | 116.900 |
| `Gelu_12` | 3 | 116.880 |
| `Gelu_20` | 3 | 116.880 |
| `Gelu_13` | 3 | 116.860 |
| `Gelu_16` | 3 | 116.860 |
| `Gelu_7` | 3 | 116.840 |
| `Gelu_1` | 3 | 116.820 |
| `Gelu_8` | 3 | 116.820 |
| `Transpose_126` | 3 | 116.640 |
| `Transpose_136` | 3 | 116.500 |
| `Transpose_76` | 3 | 116.380 |
| `Transpose_16` | 3 | 116.360 |
| `Transpose_106` | 3 | 116.300 |
| `Transpose_156` | 3 | 116.200 |
| `Transpose_266` | 3 | 116.160 |
| `Transpose_96` | 3 | 115.840 |
| `Transpose_256` | 3 | 115.760 |
| `Transpose_86` | 3 | 115.680 |
| `Transpose_6` | 3 | 115.620 |
| `Transpose_46` | 3 | 115.620 |
| `Transpose_246` | 3 | 115.560 |
| `Transpose_206` | 3 | 115.520 |
| `Transpose_166` | 3 | 115.320 |
| `Transpose_116` | 3 | 115.240 |
| `Transpose_216` | 3 | 114.980 |
| `Transpose_56` | 3 | 114.800 |
| `StridedSliceV2_29` | 3 | 105.640 |
| `StridedSliceV2_44` | 3 | 105.420 |
| `StridedSliceV2_19` | 3 | 105.340 |
| `StridedSliceV2_104` | 3 | 105.160 |
| `StridedSliceV2_124` | 3 | 105.060 |
| `StridedSliceV2_99` | 3 | 104.900 |
| `StridedSliceV2_94` | 3 | 104.860 |
| `StridedSliceV2_24` | 3 | 104.820 |
| `Transpose_135` | 3 | 104.720 |
| `StridedSliceV2_129` | 3 | 104.520 |
| `Transpose_35` | 3 | 104.500 |
| `StridedSliceV2_84` | 3 | 104.500 |
| `Transpose_195` | 3 | 104.500 |
| `StridedSliceV2_74` | 3 | 104.460 |
| `Transpose_235` | 3 | 104.340 |
| `StridedSliceV2_119` | 3 | 104.340 |
| `Transpose_75` | 3 | 104.320 |
| `Transpose_45` | 3 | 104.240 |
| `Transpose_255` | 3 | 104.100 |
| `StridedSliceV2_109` | 3 | 104.080 |
| `Transpose_85` | 3 | 104.060 |
| `Transpose_95` | 3 | 104.060 |
| `Transpose_185` | 3 | 104.060 |
| `Transpose_205` | 3 | 103.980 |
| `StridedSliceV2_14` | 3 | 103.960 |
| `StridedSliceV2_39` | 3 | 103.960 |
| `Transpose_123` | 3 | 103.920 |
| `StridedSliceV2_121` | 3 | 103.880 |
| `Transpose_245` | 3 | 103.860 |
| `Transpose_15` | 3 | 103.820 |
| `Transpose_25` | 3 | 103.740 |
| `Transpose_215` | 3 | 103.720 |
| `Transpose_55` | 3 | 103.700 |
| `Transpose_105` | 3 | 103.620 |
| `StridedSliceV2_49` | 3 | 103.600 |
| `Transpose_115` | 3 | 103.520 |
| `StridedSliceV2_118` | 3 | 103.500 |
| `Transpose_155` | 3 | 103.400 |
| `Transpose_145` | 3 | 103.360 |
| `StridedSliceV2_59` | 3 | 103.320 |
| `StridedSliceV2_69` | 3 | 103.300 |
| `Transpose_165` | 3 | 103.260 |
| `Transpose_265` | 3 | 103.260 |
| `Transpose_133` | 3 | 102.820 |
| `MatMulV2_39` | 3 | 102.800 |
| `StridedSliceV2_9` | 3 | 102.780 |
| `StridedSliceV2_134` | 3 | 102.600 |
| `Transpose_225` | 3 | 102.540 |
| `Transpose_63` | 3 | 102.420 |
| `Transpose_183` | 3 | 102.400 |
| `Transpose_65` | 3 | 102.320 |
| `Transpose_23` | 3 | 102.280 |
| `Transpose_193` | 3 | 102.280 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 102.260 |
| `Transpose_13` | 3 | 102.180 |
| `Transpose_233` | 3 | 102.160 |
| `Transpose_73` | 3 | 102.060 |
| `StridedSliceV2_54` | 3 | 102.060 |
| `StridedSliceV2_79` | 3 | 102.060 |
| `StridedSliceV2_34` | 3 | 101.900 |
| `Transpose_223` | 3 | 101.860 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "4,512,16,72" -> "4,512,16,36" | ND -> ND` | 324 | 9008.780 |
| `PromptFlashAttention | "4,16,512,80;4,16,512,80;4,16,512,80;4,1,512,512" -> "4,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 81 | 8581.860 |
| `Transpose | "4,512,16,72;4" -> "4,16,512,72" | ND;ND -> ND` | 243 | 8267.280 |
| `MatMulV2 | "2048,1152;72,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 8192.060 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7333.660 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6512.440 |
| `PadV3 | "4,16,512,72;8;" -> "4,16,512,80" | NCHW;NCHW;NCHW -> NCHW` | 243 | 4921.760 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 3990.560 |
| `Gelu | "4,512,4352" -> "4,512,4352" | ND -> ND` | 81 | 3181.780 |
| `Transpose | "4,16,512,72;4" -> "4,512,16,72" | ND;ND -> ND` | 81 | 3156.840 |
| `Mul | "4,512,16,72;4,512,1,72" -> "4,512,16,72" | ND;ND -> ND` | 324 | 2950.860 |
| `ConcatV2D | "4,512,16,36;4,512,16,36" -> "4,512,16,72" | ND;ND -> ND` | 162 | 2924.680 |
| `StridedSliceD | "4,16,512,80" -> "4,16,512,72" | NCHW -> NCHW` | 81 | 2796.920 |
| `Add | "4,512,16,72;4,512,16,72" -> "4,512,16,72" | ND;ND -> ND` | 162 | 1981.420 |
| `Cast | "4,512,16,72" -> "4,512,16,72" | ND -> ND` | 162 | 1662.500 |
| `Neg | "4,512,16,36" -> "4,512,16,36" | ND -> ND` | 162 | 1494.220 |
| `SplitVD | "4,512,3456" -> "4,512,1152;4,512,1152;4,512,1152" | ND -> ND;ND;ND` | 81 | 1344.480 |
| `ConcatV2D | "4,512,1152;4,512,1152;4,512,1152" -> "4,512,3456" | ND;ND;ND -> ND` | 81 | 1011.320 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 239.040 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 95.740 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.000 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;FRACTAL_NZ;ND` | 486 | 22038.160 |
| `ND;ND` | 972 | 19281.080 |
| `ND` | 810 | 16691.760 |
| `NCHW;NCHW;NCHW;NCHW` | 81 | 8581.860 |
| `NCHW;NCHW;NCHW` | 243 | 4921.760 |
| `ND;ND;ND;ND` | 162 | 3990.560 |
| `NCHW` | 81 | 2796.920 |
| `ND;ND;ND` | 84 | 1107.060 |
| `N/A` | 246 | 254.040 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 0 | 114.560 |
| `PromptFlashAttention` | 0 | 114.480 |
| `PromptFlashAttention` | 0 | 113.740 |
| `PromptFlashAttention_25` | 0 | 113.180 |
| `PromptFlashAttention_25` | 0 | 112.040 |
| `PromptFlashAttention_25` | 0 | 111.740 |
| `PromptFlashAttention_9` | 0 | 111.660 |
| `PromptFlashAttention_9` | 0 | 111.580 |
| `PromptFlashAttention_9` | 0 | 110.880 |
| `PromptFlashAttention_8` | 0 | 110.840 |
| `PromptFlashAttention_13` | 0 | 110.040 |
| `PromptFlashAttention_15` | 0 | 109.900 |
| `PromptFlashAttention_8` | 0 | 109.880 |
| `PromptFlashAttention_15` | 0 | 109.620 |
| `PromptFlashAttention_8` | 0 | 109.460 |
| `PromptFlashAttention_15` | 0 | 109.340 |
| `PromptFlashAttention_16` | 0 | 108.680 |
| `PromptFlashAttention_24` | 0 | 108.320 |
| `PromptFlashAttention_24` | 0 | 108.260 |
| `PromptFlashAttention_16` | 0 | 108.140 |
| `PromptFlashAttention_13` | 0 | 107.920 |
| `PromptFlashAttention_26` | 0 | 107.860 |
| `PromptFlashAttention_1` | 0 | 107.800 |
| `PromptFlashAttention_13` | 0 | 107.700 |
| `PromptFlashAttention_1` | 0 | 107.560 |
| `PromptFlashAttention_24` | 0 | 107.480 |
| `PromptFlashAttention_11` | 0 | 107.340 |
| `PromptFlashAttention_6` | 0 | 107.180 |
| `PromptFlashAttention_23` | 0 | 106.640 |
| `PromptFlashAttention_11` | 0 | 106.460 |
| `PromptFlashAttention_23` | 0 | 106.440 |
| `PromptFlashAttention_26` | 0 | 106.300 |
| `PromptFlashAttention_7` | 0 | 106.220 |
| `PromptFlashAttention_23` | 0 | 106.160 |
| `PromptFlashAttention_14` | 0 | 106.140 |
| `PromptFlashAttention_1` | 0 | 106.120 |
| `PromptFlashAttention_14` | 0 | 106.080 |
| `PromptFlashAttention_5` | 0 | 106.000 |
| `PromptFlashAttention_16` | 0 | 105.800 |
| `PromptFlashAttention_14` | 0 | 105.720 |
| `PromptFlashAttention_10` | 0 | 105.680 |
| `PromptFlashAttention_12` | 0 | 105.640 |
| `PromptFlashAttention_6` | 0 | 105.400 |
| `PromptFlashAttention_26` | 0 | 105.020 |
| `PromptFlashAttention_6` | 0 | 104.820 |
| `PromptFlashAttention_10` | 0 | 104.540 |
| `PromptFlashAttention_11` | 0 | 104.520 |
| `PromptFlashAttention_12` | 0 | 104.460 |
| `PromptFlashAttention_21` | 0 | 104.420 |
| `PromptFlashAttention_17` | 0 | 104.300 |
| `PromptFlashAttention_21` | 0 | 104.280 |
| `PromptFlashAttention_20` | 0 | 104.260 |
| `PromptFlashAttention_7` | 0 | 104.180 |
| `PromptFlashAttention_20` | 0 | 104.160 |
| `PromptFlashAttention_5` | 0 | 104.140 |
| `PromptFlashAttention_17` | 0 | 104.040 |
| `PromptFlashAttention_17` | 0 | 103.680 |
| `PromptFlashAttention_21` | 0 | 103.600 |
| `PromptFlashAttention_7` | 0 | 103.480 |
| `PromptFlashAttention_5` | 0 | 103.340 |
| `PromptFlashAttention_3` | 0 | 103.320 |
| `PromptFlashAttention_3` | 0 | 103.280 |
| `PromptFlashAttention_22` | 0 | 103.140 |
| `PromptFlashAttention_12` | 0 | 103.080 |
| `PromptFlashAttention_3` | 0 | 102.960 |
| `PromptFlashAttention_4` | 0 | 102.860 |
| `PromptFlashAttention_4` | 0 | 102.560 |
| `PromptFlashAttention_2` | 0 | 102.540 |
| `PromptFlashAttention_10` | 0 | 102.540 |
| `PromptFlashAttention_19` | 0 | 102.360 |
| `PromptFlashAttention_20` | 0 | 102.360 |
| `PromptFlashAttention_22` | 0 | 102.160 |
| `PromptFlashAttention_4` | 0 | 102.080 |
| `PromptFlashAttention_22` | 0 | 101.900 |
| `PromptFlashAttention_19` | 0 | 101.840 |
| `PromptFlashAttention_19` | 0 | 101.800 |
| `PromptFlashAttention_2` | 0 | 100.900 |
| `PromptFlashAttention_2` | 0 | 100.900 |
| `PromptFlashAttention_18` | 0 | 100.460 |
| `PromptFlashAttention_18` | 0 | 99.860 |
| `PromptFlashAttention_18` | 0 | 99.740 |
| `MatMulV2_95` | 0 | 95.380 |
| `MatMulV2_101` | 0 | 95.300 |
| `MatMulV2_53` | 0 | 94.420 |
| `MatMulV2_149` | 0 | 94.380 |
| `MatMulV2_137` | 0 | 93.580 |
| `MatMulV2_143` | 0 | 93.560 |
| `MatMulV2_137` | 0 | 93.540 |
| `MatMulV2_47` | 0 | 93.520 |
| `MatMulV2_41` | 0 | 93.480 |
| `MatMulV2_137` | 0 | 93.220 |
| `MatMulV2_53` | 0 | 93.120 |
| `MatMulV2_101` | 0 | 93.080 |
| `MatMulV2_149` | 0 | 93.020 |
| `MatMulV2_47` | 0 | 92.900 |
| `MatMulV2_89` | 0 | 92.780 |
| `MatMulV2_101` | 0 | 92.780 |
| `MatMulV2_5` | 0 | 92.700 |
| `MatMulV2_107` | 0 | 92.620 |
| `MatMulV2_89` | 0 | 92.560 |
| `MatMulV2_143` | 0 | 92.540 |
| `MatMulV2_47` | 0 | 92.260 |
| `MatMulV2_143` | 0 | 92.140 |
| `MatMulV2_41` | 0 | 92.120 |
| `MatMulV2_53` | 0 | 91.920 |
| `MatMulV2_65` | 0 | 91.880 |
| `MatMulV2_125` | 0 | 91.720 |
| `MatMulV2_125` | 0 | 91.720 |
| `MatMulV2_41` | 0 | 91.700 |
| `MatMulV2_131` | 0 | 91.580 |
| `MatMulV2_71` | 0 | 91.580 |
| `MatMulV2_113` | 0 | 91.560 |
| `MatMulV2_77` | 0 | 91.540 |
| `MatMulV2_107` | 0 | 91.500 |
| `MatMulV2_77` | 0 | 91.480 |
| `MatMulV2_83` | 0 | 91.480 |
| `MatMulV2_71` | 0 | 91.440 |
| `MatMulV2_119` | 0 | 91.420 |
| `MatMulV2_119` | 0 | 91.300 |
| `MatMulV2_161` | 0 | 91.280 |
| `MatMulV2_155` | 0 | 91.260 |
| `MatMulV2_71` | 0 | 91.220 |
| `MatMulV2_59` | 0 | 91.180 |
| `MatMulV2_59` | 0 | 91.040 |
| `MatMulV2_77` | 0 | 91.000 |
| `MatMulV2_155` | 0 | 90.880 |
| `MatMulV2_83` | 0 | 90.860 |
| `MatMulV2_35` | 0 | 90.700 |
| `MatMulV2_17` | 0 | 90.700 |
| `MatMulV2_125` | 0 | 90.660 |
| `MatMulV2_155` | 0 | 90.500 |
| `MatMulV2_29` | 0 | 90.480 |
| `MatMulV2_113` | 0 | 90.460 |
| `MatMulV2_107` | 0 | 90.420 |
| `MatMulV2_113` | 0 | 90.180 |
| `MatMulV2_11` | 0 | 90.160 |
| `MatMulV2_11` | 0 | 90.100 |
| `MatMulV2_83` | 0 | 90.060 |
| `MatMulV2_23` | 0 | 90.060 |
| `MatMulV2_29` | 0 | 90.040 |
| `MatMulV2_17` | 0 | 89.580 |
| `MatMulV2_5` | 0 | 89.520 |
| `MatMulV2_149` | 0 | 88.560 |
| `MatMulV2_131` | 0 | 88.500 |
| `MatMulV2_89` | 0 | 88.200 |
| `MatMulV2_95` | 0 | 88.100 |
| `MatMulV2_95` | 0 | 88.060 |
| `MatMulV2_161` | 0 | 87.840 |
| `MatMulV2_161` | 0 | 87.600 |
| `MatMulV2_65` | 0 | 87.460 |
| `MatMulV2_131` | 0 | 87.400 |
| `MatMulV2_23` | 0 | 87.380 |
| `MatMulV2_65` | 0 | 87.040 |
| `MatMulV2_23` | 0 | 86.500 |
| `MatMulV2_17` | 0 | 86.120 |
| `MatMulV2_35` | 0 | 86.020 |
| `MatMulV2_119` | 0 | 85.820 |
| `MatMulV2_11` | 0 | 85.360 |
| `MatMulV2_59` | 0 | 85.120 |
| `MatMulV2_29` | 0 | 84.440 |
| `MatMulV2_5` | 0 | 83.500 |
| `MatMulV2_35` | 0 | 83.480 |
| `MatMulV2_106` | 0 | 82.780 |
| `MatMulV2_16` | 0 | 82.620 |
| `MatMulV2_118` | 0 | 82.520 |
| `MatMulV2_118` | 0 | 82.420 |
| `MatMulV2_112` | 0 | 82.020 |
| `MatMulV2_22` | 0 | 81.880 |
| `MatMulV2_52` | 0 | 81.820 |
| `MatMulV2_70` | 0 | 81.800 |
| `MatMulV2_124` | 0 | 81.740 |
| `MatMulV2_106` | 0 | 81.580 |
| `MatMulV2_160` | 0 | 81.480 |
| `MatMulV2_160` | 0 | 81.460 |
| `MatMulV2_22` | 0 | 81.360 |
| `MatMulV2_58` | 0 | 81.320 |
| `MatMulV2_124` | 0 | 81.280 |
| `MatMulV2_28` | 0 | 81.280 |
| `MatMulV2_16` | 0 | 81.220 |
| `MatMulV2_34` | 0 | 81.120 |
| `MatMulV2_118` | 0 | 81.080 |
| `MatMulV2_160` | 0 | 81.080 |
| `MatMulV2_70` | 0 | 81.060 |
| `MatMulV2_106` | 0 | 81.060 |
| `MatMulV2_88` | 0 | 81.040 |
| `MatMulV2_124` | 0 | 81.040 |
| `MatMulV2_58` | 0 | 81.020 |
| `MatMulV2_46` | 0 | 81.000 |
| `MatMulV2_82` | 0 | 80.980 |
| `MatMulV2_154` | 0 | 80.960 |
| `MatMulV2_130` | 0 | 80.960 |
| `MatMulV2_136` | 0 | 80.900 |
| `MatMulV2_154` | 0 | 80.840 |
| `MatMulV2_100` | 0 | 80.800 |
| `MatMulV2_28` | 0 | 80.740 |
| `MatMulV2_46` | 0 | 80.640 |
| `MatMulV2_70` | 0 | 80.640 |
| `MatMulV2_130` | 0 | 80.600 |
| `MatMulV2_148` | 0 | 80.600 |
| `MatMulV2_82` | 0 | 80.580 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 29347.070 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.torchair.active.step1` | 1 | 28173.660 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.torchair.active.step2` | 1 | 27928.620 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.torchair.active.step3` | 1 | 27913.150 |
| `TorchDynamo Cache Lookup` | 3 | 26592.300 |
| `Torch-Compiled Region: 0/0` | 3 | 3552.820 |
| `TorchNpuGraphBase::Run` | 3 | 2610.470 |
| `RefreshAtTensorFromGeTensor` | 3 | 1112.780 |
| `aten::empty` | 3 | 542.770 |
| `ExecuteGraph` | 3 | 458.550 |
| `AssembleInputs` | 3 | 377.470 |
| `AssembleOutputs` | 3 | 287.090 |
| `aten::set_` | 3 | 273.170 |
| `empty_tensor` | 3 | 269.230 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 213260.950 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 78514.320 |
| `launch` | 976 | 17197.650 |
| `InputCopy` | 3 | 136.280 |
| `ModelExecute` | 3 | 41.970 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 23.430 |
| `step_info` | 6 | 12.780 |
| `OutputCopy` | 3 | 1.120 |

