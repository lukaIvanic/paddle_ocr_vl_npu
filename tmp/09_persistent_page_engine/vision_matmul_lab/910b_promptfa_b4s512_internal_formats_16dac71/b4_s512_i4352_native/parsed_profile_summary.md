# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4352_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4352_native/liteserver-c001-4_643006_20260729140152782_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `78221.360 us`
- `Free`: `3786.320 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3674.750 us`
- `Stage`: `82007.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 162 | 12762.360 |
| `StridedSliceD` | 405 | 11626.300 |
| `Transpose` | 324 | 11361.280 |
| `PromptFlashAttention` | 81 | 8562.360 |
| `MatMulV2` | 324 | 7870.920 |
| `PadV3` | 243 | 4923.480 |
| `AddLayerNorm` | 162 | 4107.740 |
| `ConcatV2D` | 243 | 3941.980 |
| `Gelu` | 81 | 3122.900 |
| `Mul` | 324 | 2941.960 |
| `Add` | 162 | 1969.520 |
| `Neg` | 162 | 1686.020 |
| `Cast` | 162 | 1665.860 |
| `SplitVD` | 81 | 1322.600 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 240.960 |
| `LayerNormV3` | 3 | 99.820 |
| `Data` | 3 | 15.300 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_14` | 3 | 335.880 |
| `PromptFlashAttention_25` | 3 | 335.200 |
| `PromptFlashAttention_24` | 3 | 331.640 |
| `PromptFlashAttention_1` | 3 | 331.500 |
| `PromptFlashAttention_8` | 3 | 328.520 |
| `PromptFlashAttention` | 3 | 327.220 |
| `PromptFlashAttention_15` | 3 | 327.160 |
| `PromptFlashAttention_26` | 3 | 325.760 |
| `PromptFlashAttention_7` | 3 | 324.680 |
| `PromptFlashAttention_17` | 3 | 322.520 |
| `PromptFlashAttention_23` | 3 | 321.580 |
| `PromptFlashAttention_9` | 3 | 321.240 |
| `PromptFlashAttention_13` | 3 | 320.780 |
| `PromptFlashAttention_19` | 3 | 319.600 |
| `PromptFlashAttention_16` | 3 | 318.340 |
| `PromptFlashAttention_12` | 3 | 314.320 |
| `PromptFlashAttention_6` | 3 | 313.560 |
| `PromptFlashAttention_18` | 3 | 310.380 |
| `PromptFlashAttention_3` | 3 | 307.360 |
| `PromptFlashAttention_22` | 3 | 306.700 |
| `PromptFlashAttention_11` | 3 | 306.560 |
| `PromptFlashAttention_21` | 3 | 305.020 |
| `PromptFlashAttention_10` | 3 | 304.260 |
| `PromptFlashAttention_20` | 3 | 303.020 |
| `PromptFlashAttention_2` | 3 | 301.740 |
| `PromptFlashAttention_4` | 3 | 299.400 |
| `PromptFlashAttention_5` | 3 | 298.420 |
| `MatMulV2_28_to_v3` | 3 | 270.580 |
| `MatMulV2_112_to_v3` | 3 | 261.160 |
| `MatMulV2_82_to_v3` | 3 | 259.840 |
| `MatMulV2_46_to_v3` | 3 | 259.800 |
| `MatMulV2_106_to_v3` | 3 | 258.460 |
| `MatMulV2_4_to_v3` | 3 | 256.760 |
| `MatMulV2_64_to_v3` | 3 | 256.280 |
| `MatMulV2_52_to_v3` | 3 | 256.120 |
| `MatMulV2_118_to_v3` | 3 | 255.900 |
| `MatMulV2_40_to_v3` | 3 | 255.300 |
| `MatMulV2_76_to_v3` | 3 | 255.280 |
| `MatMulV2_142_to_v3` | 3 | 254.740 |
| `MatMulV2_94_to_v3` | 3 | 254.680 |
| `MatMulV2_160_to_v3` | 3 | 254.480 |
| `MatMulV2_130_to_v3` | 3 | 254.380 |
| `MatMulV2_22_to_v3` | 3 | 254.300 |
| `MatMulV2_58_to_v3` | 3 | 254.260 |
| `MatMulV2_136_to_v3` | 3 | 254.020 |
| `MatMulV2_124_to_v3` | 3 | 253.880 |
| `MatMulV2_154_to_v3` | 3 | 253.620 |
| `MatMulV2_34_to_v3` | 3 | 253.020 |
| `MatMulV2_10_to_v3` | 3 | 252.320 |
| `MatMulV2_88_to_v3` | 3 | 252.080 |
| `MatMulV2_148_to_v3` | 3 | 251.880 |
| `MatMulV2_100_to_v3` | 3 | 251.760 |
| `MatMulV2_70_to_v3` | 3 | 251.720 |
| `MatMulV2_16_to_v3` | 3 | 251.660 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 240.960 |
| `MatMulV2_83_to_v3` | 3 | 220.160 |
| `MatMulV2_23_to_v3` | 3 | 219.000 |
| `MatMulV2_125_to_v3` | 3 | 218.340 |
| `MatMulV2_161_to_v3` | 3 | 218.300 |
| `MatMulV2_143_to_v3` | 3 | 218.260 |
| `MatMulV2_17_to_v3` | 3 | 217.880 |
| `MatMulV2_137_to_v3` | 3 | 217.880 |
| `MatMulV2_29_to_v3` | 3 | 217.800 |
| `MatMulV2_11_to_v3` | 3 | 217.740 |
| `MatMulV2_155_to_v3` | 3 | 217.580 |
| `MatMulV2_77_to_v3` | 3 | 217.300 |
| `MatMulV2_53_to_v3` | 3 | 217.280 |
| `MatMulV2_41_to_v3` | 3 | 217.260 |
| `MatMulV2_149_to_v3` | 3 | 217.260 |
| `MatMulV2_47_to_v3` | 3 | 217.120 |
| `MatMulV2_65_to_v3` | 3 | 217.060 |
| `MatMulV2_95_to_v3` | 3 | 216.860 |
| `MatMulV2_113_to_v3` | 3 | 216.820 |
| `MatMulV2_5_to_v3` | 3 | 216.680 |
| `MatMulV2_131_to_v3` | 3 | 216.540 |
| `MatMulV2_35_to_v3` | 3 | 216.540 |
| `MatMulV2_59_to_v3` | 3 | 216.480 |
| `MatMulV2_101_to_v3` | 3 | 216.360 |
| `MatMulV2_71_to_v3` | 3 | 215.860 |
| `MatMulV2_119_to_v3` | 3 | 215.640 |
| `MatMulV2_107_to_v3` | 3 | 215.120 |
| `MatMulV2_89_to_v3` | 3 | 214.960 |
| `Gelu_3` | 3 | 116.700 |
| `Transpose_243` | 3 | 116.340 |
| `Gelu_7` | 3 | 116.200 |
| `Gelu_23` | 3 | 116.200 |
| `Gelu_26` | 3 | 116.140 |
| `Gelu_20` | 3 | 115.960 |
| `Gelu` | 3 | 115.860 |
| `Gelu_15` | 3 | 115.860 |
| `Gelu_18` | 3 | 115.820 |
| `Gelu_11` | 3 | 115.780 |
| `Gelu_9` | 3 | 115.680 |
| `Gelu_13` | 3 | 115.680 |
| `Gelu_12` | 3 | 115.660 |
| `Gelu_22` | 3 | 115.640 |
| `Gelu_16` | 3 | 115.580 |
| `Gelu_4` | 3 | 115.560 |
| `Gelu_5` | 3 | 115.560 |
| `Gelu_6` | 3 | 115.520 |
| `Gelu_2` | 3 | 115.480 |
| `Gelu_14` | 3 | 115.420 |
| `Gelu_19` | 3 | 115.420 |
| `Gelu_10` | 3 | 115.420 |
| `Transpose_244` | 3 | 115.420 |
| `Gelu_24` | 3 | 115.340 |
| `Gelu_1` | 3 | 115.320 |
| `Gelu_25` | 3 | 115.300 |
| `Gelu_8` | 3 | 115.280 |
| `Gelu_21` | 3 | 115.260 |
| `Gelu_17` | 3 | 115.260 |
| `Transpose_16` | 3 | 113.940 |
| `Transpose_176` | 3 | 113.940 |
| `Transpose_186` | 3 | 113.820 |
| `Transpose_6` | 3 | 113.680 |
| `Transpose_56` | 3 | 113.620 |
| `Transpose_36` | 3 | 113.620 |
| `Transpose_266` | 3 | 113.600 |
| `Transpose_116` | 3 | 113.480 |
| `Transpose_206` | 3 | 113.440 |
| `Transpose_146` | 3 | 113.400 |
| `Transpose_236` | 3 | 113.280 |
| `Transpose_76` | 3 | 113.240 |
| `Transpose_106` | 3 | 112.940 |
| `Transpose_196` | 3 | 112.940 |
| `Transpose_136` | 3 | 112.880 |
| `Transpose_156` | 3 | 112.880 |
| `Transpose_256` | 3 | 112.860 |
| `Transpose_26` | 3 | 112.820 |
| `Transpose_126` | 3 | 112.680 |
| `Transpose_66` | 3 | 112.500 |
| `Transpose_216` | 3 | 112.420 |
| `Transpose_226` | 3 | 112.300 |
| `Transpose_166` | 3 | 112.140 |
| `Transpose_96` | 3 | 112.140 |
| `Transpose_246` | 3 | 112.120 |
| `Transpose_86` | 3 | 112.080 |
| `Transpose_46` | 3 | 111.360 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 107.580 |
| `Transpose_144` | 3 | 104.800 |
| `Transpose_94` | 3 | 104.720 |
| `Transpose_254` | 3 | 104.680 |
| `Transpose_44` | 3 | 104.620 |
| `Transpose_264` | 3 | 104.600 |
| `Transpose_194` | 3 | 104.520 |
| `Transpose_84` | 3 | 104.460 |
| `Transpose_133` | 3 | 104.460 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 104.440 |
| `Transpose_204` | 3 | 104.380 |
| `Transpose_34` | 3 | 104.360 |
| `Transpose_154` | 3 | 104.340 |
| `Transpose_104` | 3 | 104.280 |
| `Transpose_53` | 3 | 104.240 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 104.200 |
| `Transpose_114` | 3 | 104.140 |
| `Transpose_213` | 3 | 104.140 |
| `Transpose_163` | 3 | 104.080 |
| `Transpose_263` | 3 | 104.040 |
| `Transpose_214` | 3 | 104.020 |
| `Transpose_13` | 3 | 104.000 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 104.000 |
| `Transpose_124` | 3 | 103.960 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 103.940 |
| `Transpose_54` | 3 | 103.920 |
| `Transpose_134` | 3 | 103.920 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 103.880 |
| `Transpose_184` | 3 | 103.860 |
| `Transpose_253` | 3 | 103.840 |
| `Transpose_153` | 3 | 103.780 |
| `Transpose_173` | 3 | 103.780 |
| `Transpose_224` | 3 | 103.780 |
| `Transpose_233` | 3 | 103.760 |
| `Transpose_234` | 3 | 103.760 |
| `Transpose_43` | 3 | 103.740 |
| `Transpose_113` | 3 | 103.680 |
| `Transpose_64` | 3 | 103.660 |
| `Transpose_164` | 3 | 103.640 |
| `Transpose_14` | 3 | 103.520 |
| `Transpose_123` | 3 | 103.480 |
| `Transpose_223` | 3 | 103.420 |
| `Transpose_103` | 3 | 103.400 |
| `Transpose_203` | 3 | 103.400 |
| `Transpose_3` | 3 | 103.360 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 103.360 |
| `Transpose_24` | 3 | 103.300 |
| `Transpose_174` | 3 | 103.260 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 103.220 |
| `Transpose_63` | 3 | 103.220 |
| `Transpose_193` | 3 | 103.220 |
| `Transpose_73` | 3 | 103.160 |
| `Transpose_93` | 3 | 103.060 |
| `Transpose_23` | 3 | 103.000 |
| `Transpose_83` | 3 | 102.940 |
| `Transpose_74` | 3 | 102.900 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 102.900 |
| `Transpose_143` | 3 | 102.580 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 102.460 |
| `Transpose_33` | 3 | 101.580 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 101.440 |
| `Transpose_183` | 3 | 101.440 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "4,512,16,72" -> "4,512,16,36" | ND -> ND` | 324 | 8993.500 |
| `PromptFlashAttention | "4,16,512,80;4,16,512,80;4,16,512,80;4,1,512,512" -> "4,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 81 | 8562.360 |
| `Transpose | "4,512,16,72;4" -> "4,16,512,72" | ND;ND -> ND` | 243 | 8311.160 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7870.920 |
| `MatMulV3 | "2048,1152;4352,1152;4352" -> "2048,4352" | ND;ND;ND -> ND` | 81 | 6898.280 |
| `MatMulV3 | "2048,4352;1152,4352;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 5864.080 |
| `PadV3 | "4,16,512,72;8;" -> "4,16,512,80" | NCHW;NCHW;NCHW -> NCHW` | 243 | 4923.480 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4107.740 |
| `Gelu | "4,512,4352" -> "4,512,4352" | ND -> ND` | 81 | 3122.900 |
| `Transpose | "4,16,512,72;4" -> "4,512,16,72" | ND;ND -> ND` | 81 | 3050.120 |
| `Mul | "4,512,16,72;4,512,1,72" -> "4,512,16,72" | ND;ND -> ND` | 324 | 2941.960 |
| `ConcatV2D | "4,512,16,36;4,512,16,36" -> "4,512,16,72" | ND;ND -> ND` | 162 | 2927.380 |
| `StridedSliceD | "4,16,512,80" -> "4,16,512,72" | NCHW -> NCHW` | 81 | 2632.800 |
| `Add | "4,512,16,72;4,512,16,72" -> "4,512,16,72" | ND;ND -> ND` | 162 | 1969.520 |
| `Neg | "4,512,16,36" -> "4,512,16,36" | ND -> ND` | 162 | 1686.020 |
| `Cast | "4,512,16,72" -> "4,512,16,72" | ND -> ND` | 162 | 1665.860 |
| `SplitVD | "4,512,3456" -> "4,512,1152;4,512,1152;4,512,1152" | ND -> ND;ND;ND` | 81 | 1322.600 |
| `ConcatV2D | "4,512,1152;4,512,1152;4,512,1152" -> "4,512,3456" | ND;ND;ND -> ND` | 81 | 1014.600 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 240.960 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 99.820 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.300 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 21747.700 |
| `ND;ND` | 972 | 19200.140 |
| `ND` | 810 | 16790.880 |
| `NCHW;NCHW;NCHW;NCHW` | 81 | 8562.360 |
| `NCHW;NCHW;NCHW` | 243 | 4923.480 |
| `ND;ND;ND;ND` | 162 | 4107.740 |
| `NCHW` | 81 | 2632.800 |
| `N/A` | 246 | 256.260 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_14` | 0 | 112.580 |
| `PromptFlashAttention_25` | 0 | 112.520 |
| `PromptFlashAttention_14` | 0 | 112.460 |
| `PromptFlashAttention_25` | 0 | 111.920 |
| `PromptFlashAttention_1` | 0 | 111.880 |
| `PromptFlashAttention_8` | 0 | 111.700 |
| `PromptFlashAttention_24` | 0 | 111.260 |
| `PromptFlashAttention_14` | 0 | 110.840 |
| `PromptFlashAttention_25` | 0 | 110.760 |
| `PromptFlashAttention_24` | 0 | 110.340 |
| `PromptFlashAttention_1` | 0 | 110.200 |
| `PromptFlashAttention_15` | 0 | 110.140 |
| `PromptFlashAttention_24` | 0 | 110.040 |
| `PromptFlashAttention` | 0 | 109.900 |
| `PromptFlashAttention_8` | 0 | 109.860 |
| `PromptFlashAttention_26` | 0 | 109.640 |
| `PromptFlashAttention` | 0 | 109.480 |
| `PromptFlashAttention_1` | 0 | 109.420 |
| `PromptFlashAttention_19` | 0 | 109.180 |
| `PromptFlashAttention_23` | 0 | 109.080 |
| `PromptFlashAttention_7` | 0 | 108.760 |
| `PromptFlashAttention_26` | 0 | 108.660 |
| `PromptFlashAttention_15` | 0 | 108.560 |
| `PromptFlashAttention_15` | 0 | 108.460 |
| `PromptFlashAttention_7` | 0 | 108.420 |
| `PromptFlashAttention_9` | 0 | 108.280 |
| `PromptFlashAttention_17` | 0 | 108.180 |
| `PromptFlashAttention_16` | 0 | 108.060 |
| `PromptFlashAttention` | 0 | 107.840 |
| `PromptFlashAttention_17` | 0 | 107.820 |
| `PromptFlashAttention_23` | 0 | 107.660 |
| `PromptFlashAttention_13` | 0 | 107.520 |
| `PromptFlashAttention_7` | 0 | 107.500 |
| `PromptFlashAttention_26` | 0 | 107.460 |
| `PromptFlashAttention_9` | 0 | 107.240 |
| `PromptFlashAttention_8` | 0 | 106.960 |
| `PromptFlashAttention_13` | 0 | 106.840 |
| `PromptFlashAttention_17` | 0 | 106.520 |
| `PromptFlashAttention_13` | 0 | 106.420 |
| `PromptFlashAttention_19` | 0 | 106.120 |
| `PromptFlashAttention_12` | 0 | 106.100 |
| `PromptFlashAttention_16` | 0 | 105.960 |
| `PromptFlashAttention_9` | 0 | 105.720 |
| `PromptFlashAttention_23` | 0 | 104.840 |
| `PromptFlashAttention_6` | 0 | 104.660 |
| `PromptFlashAttention_6` | 0 | 104.540 |
| `PromptFlashAttention_6` | 0 | 104.360 |
| `PromptFlashAttention_16` | 0 | 104.320 |
| `PromptFlashAttention_19` | 0 | 104.300 |
| `PromptFlashAttention_12` | 0 | 104.200 |
| `PromptFlashAttention_18` | 0 | 104.180 |
| `PromptFlashAttention_22` | 0 | 104.100 |
| `PromptFlashAttention_12` | 0 | 104.020 |
| `PromptFlashAttention_18` | 0 | 103.960 |
| `PromptFlashAttention_3` | 0 | 103.900 |
| `PromptFlashAttention_11` | 0 | 103.240 |
| `PromptFlashAttention_3` | 0 | 103.060 |
| `PromptFlashAttention_21` | 0 | 102.940 |
| `PromptFlashAttention_18` | 0 | 102.240 |
| `PromptFlashAttention_11` | 0 | 101.840 |
| `PromptFlashAttention_5` | 0 | 101.840 |
| `PromptFlashAttention_20` | 0 | 101.800 |
| `PromptFlashAttention_10` | 0 | 101.800 |
| `PromptFlashAttention_22` | 0 | 101.500 |
| `PromptFlashAttention_2` | 0 | 101.480 |
| `PromptFlashAttention_11` | 0 | 101.480 |
| `PromptFlashAttention_21` | 0 | 101.340 |
| `PromptFlashAttention_10` | 0 | 101.320 |
| `PromptFlashAttention_2` | 0 | 101.240 |
| `PromptFlashAttention_10` | 0 | 101.140 |
| `PromptFlashAttention_22` | 0 | 101.100 |
| `PromptFlashAttention_20` | 0 | 101.000 |
| `PromptFlashAttention_21` | 0 | 100.740 |
| `PromptFlashAttention_3` | 0 | 100.400 |
| `PromptFlashAttention_20` | 0 | 100.220 |
| `PromptFlashAttention_4` | 0 | 100.100 |
| `PromptFlashAttention_4` | 0 | 99.860 |
| `PromptFlashAttention_4` | 0 | 99.440 |
| `PromptFlashAttention_2` | 0 | 99.020 |
| `PromptFlashAttention_5` | 0 | 98.920 |
| `PromptFlashAttention_5` | 0 | 97.660 |
| `MatMulV2_28_to_v3` | 0 | 92.040 |
| `MatMulV2_28_to_v3` | 0 | 89.660 |
| `MatMulV2_28_to_v3` | 0 | 88.880 |
| `MatMulV2_112_to_v3` | 0 | 88.620 |
| `MatMulV2_46_to_v3` | 0 | 87.740 |
| `MatMulV2_106_to_v3` | 0 | 87.260 |
| `MatMulV2_82_to_v3` | 0 | 87.100 |
| `MatMulV2_82_to_v3` | 0 | 86.700 |
| `MatMulV2_118_to_v3` | 0 | 86.580 |
| `MatMulV2_46_to_v3` | 0 | 86.560 |
| `MatMulV2_112_to_v3` | 0 | 86.500 |
| `MatMulV2_4_to_v3` | 0 | 86.240 |
| `MatMulV2_82_to_v3` | 0 | 86.040 |
| `MatMulV2_112_to_v3` | 0 | 86.040 |
| `MatMulV2_142_to_v3` | 0 | 86.020 |
| `MatMulV2_64_to_v3` | 0 | 85.920 |
| `MatMulV2_52_to_v3` | 0 | 85.820 |
| `MatMulV2_106_to_v3` | 0 | 85.800 |
| `MatMulV2_154_to_v3` | 0 | 85.760 |
| `MatMulV2_160_to_v3` | 0 | 85.740 |
| `MatMulV2_4_to_v3` | 0 | 85.700 |
| `MatMulV2_40_to_v3` | 0 | 85.620 |
| `MatMulV2_40_to_v3` | 0 | 85.560 |
| `MatMulV2_64_to_v3` | 0 | 85.560 |
| `MatMulV2_46_to_v3` | 0 | 85.500 |
| `MatMulV2_106_to_v3` | 0 | 85.400 |
| `MatMulV2_94_to_v3` | 0 | 85.360 |
| `MatMulV2_52_to_v3` | 0 | 85.260 |
| `MatMulV2_76_to_v3` | 0 | 85.220 |
| `MatMulV2_58_to_v3` | 0 | 85.140 |
| `MatMulV2_76_to_v3` | 0 | 85.140 |
| `MatMulV2_124_to_v3` | 0 | 85.080 |
| `MatMulV2_22_to_v3` | 0 | 85.080 |
| `MatMulV2_34_to_v3` | 0 | 85.060 |
| `MatMulV2_52_to_v3` | 0 | 85.040 |
| `MatMulV2_130_to_v3` | 0 | 85.040 |
| `MatMulV2_94_to_v3` | 0 | 85.020 |
| `MatMulV2_118_to_v3` | 0 | 85.000 |
| `MatMulV2_22_to_v3` | 0 | 85.000 |
| `MatMulV2_70_to_v3` | 0 | 85.000 |
| `MatMulV2_76_to_v3` | 0 | 84.920 |
| `MatMulV2_136_to_v3` | 0 | 84.880 |
| `MatMulV2_4_to_v3` | 0 | 84.820 |
| `MatMulV2_58_to_v3` | 0 | 84.820 |
| `MatMulV2_64_to_v3` | 0 | 84.800 |
| `MatMulV2_88_to_v3` | 0 | 84.780 |
| `MatMulV2_136_to_v3` | 0 | 84.740 |
| `MatMulV2_130_to_v3` | 0 | 84.680 |
| `MatMulV2_130_to_v3` | 0 | 84.660 |
| `MatMulV2_124_to_v3` | 0 | 84.640 |
| `MatMulV2_142_to_v3` | 0 | 84.600 |
| `MatMulV2_148_to_v3` | 0 | 84.540 |
| `MatMulV2_10_to_v3` | 0 | 84.400 |
| `MatMulV2_136_to_v3` | 0 | 84.400 |
| `MatMulV2_160_to_v3` | 0 | 84.400 |
| `MatMulV2_160_to_v3` | 0 | 84.340 |
| `MatMulV2_154_to_v3` | 0 | 84.320 |
| `MatMulV2_118_to_v3` | 0 | 84.320 |
| `MatMulV2_94_to_v3` | 0 | 84.300 |
| `MatMulV2_58_to_v3` | 0 | 84.300 |
| `MatMulV2_148_to_v3` | 0 | 84.260 |
| `MatMulV2_22_to_v3` | 0 | 84.220 |
| `MatMulV2_100_to_v3` | 0 | 84.220 |
| `MatMulV2_124_to_v3` | 0 | 84.160 |
| `MatMulV2_40_to_v3` | 0 | 84.120 |
| `MatMulV2_142_to_v3` | 0 | 84.120 |
| `MatMulV2_34_to_v3` | 0 | 84.060 |
| `MatMulV2_16_to_v3` | 0 | 84.060 |
| `MatMulV2_10_to_v3` | 0 | 84.040 |
| `MatMulV2_100_to_v3` | 0 | 84.000 |
| `MatMulV2_34_to_v3` | 0 | 83.900 |
| `MatMulV2_10_to_v3` | 0 | 83.880 |
| `MatMulV2_16_to_v3` | 0 | 83.880 |
| `MatMulV2_70_to_v3` | 0 | 83.840 |
| `MatMulV2_16_to_v3` | 0 | 83.720 |
| `MatMulV2_88_to_v3` | 0 | 83.720 |
| `MatMulV2_88_to_v3` | 0 | 83.580 |
| `MatMulV2_154_to_v3` | 0 | 83.540 |
| `MatMulV2_100_to_v3` | 0 | 83.540 |
| `MatMulV2_148_to_v3` | 0 | 83.080 |
| `MatMulV2_70_to_v3` | 0 | 82.880 |
| `MatMulV2_83_to_v3` | 0 | 74.180 |
| `MatMulV2_29_to_v3` | 0 | 73.820 |
| `MatMulV2_149_to_v3` | 0 | 73.740 |
| `MatMulV2_161_to_v3` | 0 | 73.620 |
| `MatMulV2_17_to_v3` | 0 | 73.560 |
| `MatMulV2_143_to_v3` | 0 | 73.500 |
| `MatMulV2_23_to_v3` | 0 | 73.400 |
| `MatMulV2_65_to_v3` | 0 | 73.340 |
| `MatMulV2_11_to_v3` | 0 | 73.280 |
| `MatMulV2_35_to_v3` | 0 | 73.100 |
| `MatMulV2_125_to_v3` | 0 | 73.060 |
| `MatMulV2_83_to_v3` | 0 | 73.020 |
| `MatMulV2_125_to_v3` | 0 | 73.020 |
| `MatMulV2_47_to_v3` | 0 | 72.960 |
| `MatMulV2_83_to_v3` | 0 | 72.960 |
| `MatMulV2_155_to_v3` | 0 | 72.940 |
| `MatMulV2_41_to_v3` | 0 | 72.940 |
| `MatMulV2_23_to_v3` | 0 | 72.880 |
| `MatMulV2_107_to_v3` | 0 | 72.880 |
| `MatMulV2_143_to_v3` | 0 | 72.820 |
| `MatMulV2_17_to_v3` | 0 | 72.780 |
| `MatMulV2_5_to_v3` | 0 | 72.740 |
| `MatMulV2_23_to_v3` | 0 | 72.720 |
| `MatMulV2_71_to_v3` | 0 | 72.700 |
| `MatMulV2_155_to_v3` | 0 | 72.700 |
| `MatMulV2_5_to_v3` | 0 | 72.680 |
| `MatMulV2_137_to_v3` | 0 | 72.660 |
| `MatMulV2_137_to_v3` | 0 | 72.640 |
| `MatMulV2_113_to_v3` | 0 | 72.640 |
| `MatMulV2_119_to_v3` | 0 | 72.620 |
| `MatMulV2_161_to_v3` | 0 | 72.580 |
| `MatMulV2_59_to_v3` | 0 | 72.580 |
| `MatMulV2_137_to_v3` | 0 | 72.580 |
| `MatMulV2_77_to_v3` | 0 | 72.560 |
| `MatMulV2_53_to_v3` | 0 | 72.540 |
| `MatMulV2_77_to_v3` | 0 | 72.540 |
| `MatMulV2_149_to_v3` | 0 | 72.540 |
| `MatMulV2_95_to_v3` | 0 | 72.520 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 29224.840 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.native.torchair.active.step1` | 1 | 27846.410 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.native.torchair.active.step2` | 1 | 27582.870 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.native.torchair.active.step3` | 1 | 27507.020 |
| `TorchDynamo Cache Lookup` | 3 | 26181.890 |
| `Torch-Compiled Region: 0/0` | 3 | 3882.660 |
| `TorchNpuGraphBase::Run` | 3 | 2855.040 |
| `RefreshAtTensorFromGeTensor` | 3 | 1153.320 |
| `aten::empty` | 3 | 560.740 |
| `ExecuteGraph` | 3 | 523.420 |
| `AssembleInputs` | 3 | 452.490 |
| `AssembleOutputs` | 3 | 323.530 |
| `aten::set_` | 3 | 285.600 |
| `empty_tensor` | 3 | 276.440 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 214717.940 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 76985.300 |
| `launch` | 976 | 16972.120 |
| `InputCopy` | 3 | 179.250 |
| `ModelExecute` | 3 | 44.170 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 23.840 |
| `step_info` | 6 | 14.120 |
| `OutputCopy` | 3 | 1.140 |

