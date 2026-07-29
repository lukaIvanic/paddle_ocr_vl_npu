# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4304_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_b4s512_internal_formats_16dac71/b4_s512_i4304_native/liteserver-c001-4_640118_20260729135936732_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `81645.560 us`
- `Free`: `3686.120 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3402.500 us`
- `Stage`: `85331.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 162 | 16075.060 |
| `Transpose` | 324 | 11831.740 |
| `StridedSliceD` | 405 | 11803.340 |
| `PromptFlashAttention` | 81 | 8580.200 |
| `MatMulV2` | 324 | 7865.540 |
| `AddLayerNorm` | 162 | 4526.780 |
| `PadV3` | 243 | 4487.380 |
| `ConcatV2D` | 243 | 3809.860 |
| `Gelu` | 81 | 3172.340 |
| `Mul` | 324 | 2809.960 |
| `Add` | 162 | 1843.020 |
| `Cast` | 162 | 1652.900 |
| `Neg` | 162 | 1524.300 |
| `SplitVD` | 81 | 1318.180 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 237.880 |
| `LayerNormV3` | 3 | 91.680 |
| `Data` | 3 | 15.400 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_25` | 3 | 339.000 |
| `PromptFlashAttention_14` | 3 | 336.760 |
| `PromptFlashAttention_15` | 3 | 334.420 |
| `PromptFlashAttention_6` | 3 | 334.320 |
| `PromptFlashAttention_24` | 3 | 332.600 |
| `PromptFlashAttention_12` | 3 | 327.120 |
| `PromptFlashAttention_13` | 3 | 325.860 |
| `PromptFlashAttention_7` | 3 | 324.660 |
| `PromptFlashAttention_23` | 3 | 322.100 |
| `PromptFlashAttention_1` | 3 | 321.780 |
| `PromptFlashAttention` | 3 | 320.940 |
| `PromptFlashAttention_9` | 3 | 320.120 |
| `PromptFlashAttention_16` | 3 | 318.640 |
| `PromptFlashAttention_26` | 3 | 317.540 |
| `PromptFlashAttention_17` | 3 | 315.380 |
| `PromptFlashAttention_8` | 3 | 314.260 |
| `PromptFlashAttention_19` | 3 | 312.740 |
| `PromptFlashAttention_11` | 3 | 310.920 |
| `PromptFlashAttention_18` | 3 | 310.800 |
| `PromptFlashAttention_21` | 3 | 308.920 |
| `PromptFlashAttention_3` | 3 | 308.680 |
| `PromptFlashAttention_20` | 3 | 308.240 |
| `PromptFlashAttention_2` | 3 | 308.160 |
| `MatMulV2_155_to_v3` | 3 | 308.000 |
| `MatMulV2_143_to_v3` | 3 | 307.340 |
| `PromptFlashAttention_22` | 3 | 306.860 |
| `MatMulV2_119_to_v3` | 3 | 305.040 |
| `MatMulV2_161_to_v3` | 3 | 304.920 |
| `MatMulV2_149_to_v3` | 3 | 304.600 |
| `PromptFlashAttention_10` | 3 | 304.340 |
| `MatMulV2_53_to_v3` | 3 | 301.680 |
| `MatMulV2_160_to_v3` | 3 | 301.400 |
| `MatMulV2_28_to_v3` | 3 | 301.280 |
| `MatMulV2_5_to_v3` | 3 | 300.100 |
| `MatMulV2_112_to_v3` | 3 | 300.060 |
| `MatMulV2_83_to_v3` | 3 | 300.000 |
| `MatMulV2_76_to_v3` | 3 | 299.880 |
| `MatMulV2_35_to_v3` | 3 | 299.840 |
| `MatMulV2_77_to_v3` | 3 | 299.460 |
| `MatMulV2_16_to_v3` | 3 | 299.380 |
| `MatMulV2_82_to_v3` | 3 | 299.300 |
| `MatMulV2_95_to_v3` | 3 | 299.280 |
| `MatMulV2_88_to_v3` | 3 | 299.160 |
| `MatMulV2_107_to_v3` | 3 | 299.040 |
| `MatMulV2_11_to_v3` | 3 | 299.000 |
| `MatMulV2_125_to_v3` | 3 | 299.000 |
| `MatMulV2_101_to_v3` | 3 | 298.800 |
| `MatMulV2_124_to_v3` | 3 | 298.740 |
| `MatMulV2_29_to_v3` | 3 | 298.460 |
| `MatMulV2_40_to_v3` | 3 | 298.080 |
| `PromptFlashAttention_4` | 3 | 298.020 |
| `MatMulV2_137_to_v3` | 3 | 297.540 |
| `MatMulV2_100_to_v3` | 3 | 297.400 |
| `MatMulV2_59_to_v3` | 3 | 297.400 |
| `MatMulV2_58_to_v3` | 3 | 297.340 |
| `MatMulV2_23_to_v3` | 3 | 297.220 |
| `MatMulV2_71_to_v3` | 3 | 297.120 |
| `MatMulV2_106_to_v3` | 3 | 297.120 |
| `PromptFlashAttention_5` | 3 | 297.020 |
| `MatMulV2_65_to_v3` | 3 | 296.860 |
| `MatMulV2_52_to_v3` | 3 | 296.640 |
| `MatMulV2_136_to_v3` | 3 | 296.580 |
| `MatMulV2_70_to_v3` | 3 | 296.320 |
| `MatMulV2_41_to_v3` | 3 | 296.280 |
| `MatMulV2_46_to_v3` | 3 | 296.280 |
| `MatMulV2_113_to_v3` | 3 | 295.980 |
| `MatMulV2_22_to_v3` | 3 | 295.920 |
| `MatMulV2_17_to_v3` | 3 | 295.880 |
| `MatMulV2_89_to_v3` | 3 | 295.760 |
| `MatMulV2_130_to_v3` | 3 | 295.760 |
| `MatMulV2_131_to_v3` | 3 | 295.740 |
| `MatMulV2_154_to_v3` | 3 | 295.620 |
| `MatMulV2_142_to_v3` | 3 | 295.140 |
| `MatMulV2_47_to_v3` | 3 | 294.680 |
| `MatMulV2_4_to_v3` | 3 | 293.480 |
| `MatMulV2_94_to_v3` | 3 | 291.360 |
| `MatMulV2_148_to_v3` | 3 | 290.000 |
| `MatMulV2_64_to_v3` | 3 | 289.780 |
| `MatMulV2_10_to_v3` | 3 | 289.420 |
| `MatMulV2_34_to_v3` | 3 | 289.320 |
| `MatMulV2_118_to_v3` | 3 | 289.280 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 237.880 |
| `Gelu_4` | 3 | 134.780 |
| `Gelu_10` | 3 | 134.380 |
| `Transpose_193` | 3 | 124.060 |
| `Transpose_133` | 3 | 123.980 |
| `Transpose_194` | 3 | 123.960 |
| `StridedSliceV2_84` | 3 | 123.660 |
| `Transpose_134` | 3 | 123.380 |
| `Transpose_25` | 3 | 120.700 |
| `Transpose_206` | 3 | 120.100 |
| `Transpose_36` | 3 | 119.680 |
| `Transpose_76` | 3 | 119.400 |
| `Transpose_116` | 3 | 119.260 |
| `Transpose_166` | 3 | 119.000 |
| `Transpose_266` | 3 | 118.840 |
| `Transpose_256` | 3 | 118.680 |
| `Transpose_96` | 3 | 118.660 |
| `Transpose_216` | 3 | 118.000 |
| `Transpose_156` | 3 | 117.880 |
| `Transpose_106` | 3 | 117.760 |
| `Transpose_196` | 3 | 117.740 |
| `Transpose_46` | 3 | 117.460 |
| `Transpose_16` | 3 | 117.360 |
| `Transpose_136` | 3 | 117.340 |
| `Transpose_56` | 3 | 117.320 |
| `Transpose_146` | 3 | 117.320 |
| `Gelu_25` | 3 | 117.280 |
| `Transpose_226` | 3 | 117.260 |
| `Transpose_246` | 3 | 117.160 |
| `Transpose_66` | 3 | 117.020 |
| `Gelu_3` | 3 | 117.020 |
| `Gelu_19` | 3 | 116.960 |
| `Transpose_186` | 3 | 116.960 |
| `Gelu_6` | 3 | 116.900 |
| `Transpose_236` | 3 | 116.860 |
| `Gelu_15` | 3 | 116.820 |
| `Transpose_26` | 3 | 116.800 |
| `Transpose_86` | 3 | 116.640 |
| `Transpose_176` | 3 | 116.620 |
| `Transpose_126` | 3 | 116.540 |
| `Transpose_6` | 3 | 116.320 |
| `Gelu_24` | 3 | 116.140 |
| `Gelu_20` | 3 | 116.100 |
| `Gelu_26` | 3 | 116.080 |
| `Gelu_8` | 3 | 116.060 |
| `Gelu_12` | 3 | 116.040 |
| `Gelu_2` | 3 | 116.020 |
| `Gelu_21` | 3 | 116.020 |
| `Gelu_14` | 3 | 115.980 |
| `Gelu_16` | 3 | 115.940 |
| `Gelu` | 3 | 115.920 |
| `Gelu_5` | 3 | 115.920 |
| `Gelu_11` | 3 | 115.880 |
| `Gelu_18` | 3 | 115.880 |
| `Gelu_22` | 3 | 115.820 |
| `Gelu_9` | 3 | 115.780 |
| `Gelu_23` | 3 | 115.780 |
| `Gelu_7` | 3 | 115.760 |
| `Gelu_17` | 3 | 115.740 |
| `Gelu_13` | 3 | 115.700 |
| `Gelu_1` | 3 | 115.640 |
| `Transpose_3` | 3 | 110.020 |
| `Transpose_243` | 3 | 109.840 |
| `Transpose_83` | 3 | 109.760 |
| `Transpose_93` | 3 | 109.700 |
| `Transpose_253` | 3 | 109.660 |
| `Transpose_244` | 3 | 109.520 |
| `Transpose_184` | 3 | 109.520 |
| `Transpose_24` | 3 | 109.440 |
| `Transpose_144` | 3 | 109.360 |
| `Transpose_43` | 3 | 109.280 |
| `Transpose_143` | 3 | 109.260 |
| `Transpose_123` | 3 | 109.260 |
| `Transpose_64` | 3 | 109.160 |
| `Transpose_173` | 3 | 109.140 |
| `Transpose_124` | 3 | 109.100 |
| `Transpose_13` | 3 | 109.100 |
| `Transpose_23` | 3 | 109.100 |
| `Transpose_94` | 3 | 109.060 |
| `Transpose_204` | 3 | 109.020 |
| `Transpose_224` | 3 | 108.980 |
| `Transpose_84` | 3 | 108.960 |
| `Transpose_44` | 3 | 108.900 |
| `Transpose_14` | 3 | 108.800 |
| `Transpose_104` | 3 | 108.800 |
| `Transpose_264` | 3 | 108.800 |
| `Transpose_53` | 3 | 108.740 |
| `Transpose_74` | 3 | 108.720 |
| `Transpose_213` | 3 | 108.680 |
| `Transpose_223` | 3 | 108.680 |
| `Transpose_234` | 3 | 108.660 |
| `Transpose_233` | 3 | 108.620 |
| `Transpose_164` | 3 | 108.560 |
| `Transpose_174` | 3 | 108.560 |
| `Transpose_154` | 3 | 108.520 |
| `Transpose_203` | 3 | 108.520 |
| `Transpose_114` | 3 | 108.500 |
| `Transpose_183` | 3 | 108.460 |
| `Transpose_34` | 3 | 108.400 |
| `Transpose_214` | 3 | 108.360 |
| `Transpose_254` | 3 | 108.340 |
| `Transpose_33` | 3 | 108.220 |
| `Transpose_63` | 3 | 108.040 |
| `Transpose_54` | 3 | 108.020 |
| `Transpose_73` | 3 | 108.020 |
| `Transpose_153` | 3 | 108.000 |
| `Transpose_103` | 3 | 107.920 |
| `Transpose_163` | 3 | 107.760 |
| `Transpose_113` | 3 | 107.360 |
| `Transpose_263` | 3 | 106.900 |
| `StridedSliceV2_134` | 3 | 106.200 |
| `StridedSliceV2_54` | 3 | 105.720 |
| `StridedSliceV2_19` | 3 | 104.900 |
| `StridedSliceV2_59` | 3 | 104.580 |
| `StridedSliceV2_79` | 3 | 104.120 |
| `StridedSliceV2_119` | 3 | 104.120 |
| `StridedSliceV2_44` | 3 | 104.040 |
| `StridedSliceV2_39` | 3 | 103.840 |
| `StridedSliceV2_14` | 3 | 103.700 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "4,512,16,72" -> "4,512,16,36" | ND -> ND` | 324 | 9010.120 |
| `Transpose | "4,512,16,72;4" -> "4,16,512,72" | ND;ND -> ND` | 243 | 8651.760 |
| `PromptFlashAttention | "4,16,512,80;4,16,512,80;4,16,512,80;4,1,512,512" -> "4,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 81 | 8580.200 |
| `MatMulV3 | "2048,4304;1152,4304;1152" -> "2048,1152" | ND;ND;ND -> ND` | 81 | 8085.020 |
| `MatMulV3 | "2048,1152;4304,1152;4304" -> "2048,4304" | ND;ND;ND -> ND` | 81 | 7990.040 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 324 | 7865.540 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 4526.780 |
| `PadV3 | "4,16,512,72;8;" -> "4,16,512,80" | NCHW;NCHW;NCHW -> NCHW` | 243 | 4487.380 |
| `Transpose | "4,16,512,72;4" -> "4,512,16,72" | ND;ND -> ND` | 81 | 3179.980 |
| `Gelu | "4,512,4304" -> "4,512,4304" | ND -> ND` | 81 | 3172.340 |
| `Mul | "4,512,16,72;4,512,1,72" -> "4,512,16,72" | ND;ND -> ND` | 324 | 2809.960 |
| `StridedSliceD | "4,16,512,80" -> "4,16,512,72" | NCHW -> NCHW` | 81 | 2793.220 |
| `ConcatV2D | "4,512,16,36;4,512,16,36" -> "4,512,16,72" | ND;ND -> ND` | 162 | 2765.480 |
| `Add | "4,512,16,72;4,512,16,72" -> "4,512,16,72" | ND;ND -> ND` | 162 | 1843.020 |
| `Cast | "4,512,16,72" -> "4,512,16,72" | ND -> ND` | 162 | 1652.900 |
| `Neg | "4,512,16,36" -> "4,512,16,36" | ND -> ND` | 162 | 1524.300 |
| `SplitVD | "4,512,3456" -> "4,512,1152;4,512,1152;4,512,1152" | ND -> ND;ND;ND` | 81 | 1318.180 |
| `ConcatV2D | "4,512,1152;4,512,1152;4,512,1152" -> "4,512,3456" | ND;ND;ND -> ND` | 81 | 1044.380 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 237.880 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 91.680 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.400 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 570 | 25076.660 |
| `ND;ND` | 972 | 19250.200 |
| `ND` | 810 | 16677.840 |
| `NCHW;NCHW;NCHW;NCHW` | 81 | 8580.200 |
| `ND;ND;ND;ND` | 162 | 4526.780 |
| `NCHW;NCHW;NCHW` | 243 | 4487.380 |
| `NCHW` | 81 | 2793.220 |
| `N/A` | 246 | 253.280 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_25` | 0 | 114.800 |
| `PromptFlashAttention_15` | 0 | 114.160 |
| `PromptFlashAttention_6` | 0 | 113.780 |
| `PromptFlashAttention_14` | 0 | 113.320 |
| `PromptFlashAttention_25` | 0 | 112.800 |
| `PromptFlashAttention_6` | 0 | 112.280 |
| `PromptFlashAttention_14` | 0 | 112.280 |
| `PromptFlashAttention_25` | 0 | 111.400 |
| `PromptFlashAttention_15` | 0 | 111.380 |
| `PromptFlashAttention_24` | 0 | 111.340 |
| `PromptFlashAttention_24` | 0 | 111.320 |
| `PromptFlashAttention_14` | 0 | 111.160 |
| `PromptFlashAttention_12` | 0 | 110.100 |
| `PromptFlashAttention_24` | 0 | 109.940 |
| `PromptFlashAttention_7` | 0 | 109.800 |
| `PromptFlashAttention_13` | 0 | 109.120 |
| `PromptFlashAttention_15` | 0 | 108.880 |
| `PromptFlashAttention_12` | 0 | 108.760 |
| `PromptFlashAttention_13` | 0 | 108.660 |
| `PromptFlashAttention_1` | 0 | 108.400 |
| `PromptFlashAttention_6` | 0 | 108.260 |
| `PromptFlashAttention_12` | 0 | 108.260 |
| `PromptFlashAttention_13` | 0 | 108.080 |
| `PromptFlashAttention_23` | 0 | 107.780 |
| `PromptFlashAttention_7` | 0 | 107.500 |
| `PromptFlashAttention_23` | 0 | 107.440 |
| `PromptFlashAttention_7` | 0 | 107.360 |
| `PromptFlashAttention_9` | 0 | 107.360 |
| `PromptFlashAttention_16` | 0 | 107.200 |
| `PromptFlashAttention` | 0 | 107.040 |
| `PromptFlashAttention` | 0 | 106.960 |
| `PromptFlashAttention` | 0 | 106.940 |
| `PromptFlashAttention_1` | 0 | 106.880 |
| `PromptFlashAttention_23` | 0 | 106.880 |
| `PromptFlashAttention_26` | 0 | 106.740 |
| `PromptFlashAttention_9` | 0 | 106.580 |
| `PromptFlashAttention_1` | 0 | 106.500 |
| `PromptFlashAttention_16` | 0 | 106.460 |
| `PromptFlashAttention_9` | 0 | 106.180 |
| `PromptFlashAttention_17` | 0 | 105.760 |
| `PromptFlashAttention_26` | 0 | 105.520 |
| `PromptFlashAttention_20` | 0 | 105.440 |
| `PromptFlashAttention_8` | 0 | 105.400 |
| `PromptFlashAttention_26` | 0 | 105.280 |
| `PromptFlashAttention_18` | 0 | 105.220 |
| `PromptFlashAttention_17` | 0 | 105.200 |
| `PromptFlashAttention_8` | 0 | 105.080 |
| `PromptFlashAttention_16` | 0 | 104.980 |
| `PromptFlashAttention_19` | 0 | 104.540 |
| `PromptFlashAttention_17` | 0 | 104.420 |
| `PromptFlashAttention_19` | 0 | 104.400 |
| `PromptFlashAttention_3` | 0 | 104.180 |
| `PromptFlashAttention_11` | 0 | 104.180 |
| `PromptFlashAttention_18` | 0 | 103.940 |
| `PromptFlashAttention_19` | 0 | 103.800 |
| `PromptFlashAttention_8` | 0 | 103.780 |
| `PromptFlashAttention_2` | 0 | 103.760 |
| `PromptFlashAttention_11` | 0 | 103.740 |
| `MatMulV2_143_to_v3` | 0 | 103.400 |
| `MatMulV2_155_to_v3` | 0 | 103.340 |
| `PromptFlashAttention_21` | 0 | 103.120 |
| `MatMulV2_155_to_v3` | 0 | 103.040 |
| `PromptFlashAttention_11` | 0 | 103.000 |
| `PromptFlashAttention_22` | 0 | 102.980 |
| `PromptFlashAttention_21` | 0 | 102.920 |
| `PromptFlashAttention_2` | 0 | 102.880 |
| `PromptFlashAttention_21` | 0 | 102.880 |
| `MatMulV2_83_to_v3` | 0 | 102.540 |
| `PromptFlashAttention_3` | 0 | 102.460 |
| `PromptFlashAttention_22` | 0 | 102.380 |
| `MatMulV2_119_to_v3` | 0 | 102.360 |
| `MatMulV2_161_to_v3` | 0 | 102.140 |
| `MatMulV2_143_to_v3` | 0 | 102.140 |
| `MatMulV2_161_to_v3` | 0 | 102.100 |
| `PromptFlashAttention_3` | 0 | 102.040 |
| `PromptFlashAttention_10` | 0 | 102.020 |
| `MatMulV2_149_to_v3` | 0 | 101.920 |
| `MatMulV2_143_to_v3` | 0 | 101.800 |
| `MatMulV2_95_to_v3` | 0 | 101.680 |
| `PromptFlashAttention_20` | 0 | 101.660 |
| `PromptFlashAttention_18` | 0 | 101.640 |
| `MatMulV2_107_to_v3` | 0 | 101.640 |
| `MatMulV2_155_to_v3` | 0 | 101.620 |
| `MatMulV2_119_to_v3` | 0 | 101.580 |
| `MatMulV2_149_to_v3` | 0 | 101.540 |
| `PromptFlashAttention_2` | 0 | 101.520 |
| `PromptFlashAttention_22` | 0 | 101.500 |
| `PromptFlashAttention_10` | 0 | 101.260 |
| `PromptFlashAttention_20` | 0 | 101.140 |
| `MatMulV2_149_to_v3` | 0 | 101.140 |
| `MatMulV2_119_to_v3` | 0 | 101.100 |
| `PromptFlashAttention_10` | 0 | 101.060 |
| `MatMulV2_28_to_v3` | 0 | 101.060 |
| `MatMulV2_160_to_v3` | 0 | 101.000 |
| `MatMulV2_53_to_v3` | 0 | 100.980 |
| `MatMulV2_112_to_v3` | 0 | 100.840 |
| `MatMulV2_53_to_v3` | 0 | 100.760 |
| `MatMulV2_161_to_v3` | 0 | 100.680 |
| `MatMulV2_28_to_v3` | 0 | 100.660 |
| `MatMulV2_112_to_v3` | 0 | 100.620 |
| `MatMulV2_35_to_v3` | 0 | 100.620 |
| `MatMulV2_76_to_v3` | 0 | 100.540 |
| `MatMulV2_106_to_v3` | 0 | 100.480 |
| `MatMulV2_88_to_v3` | 0 | 100.440 |
| `MatMulV2_5_to_v3` | 0 | 100.420 |
| `MatMulV2_5_to_v3` | 0 | 100.420 |
| `PromptFlashAttention_4` | 0 | 100.420 |
| `PromptFlashAttention_5` | 0 | 100.340 |
| `MatMulV2_160_to_v3` | 0 | 100.320 |
| `MatMulV2_124_to_v3` | 0 | 100.260 |
| `MatMulV2_11_to_v3` | 0 | 100.260 |
| `MatMulV2_77_to_v3` | 0 | 100.220 |
| `MatMulV2_101_to_v3` | 0 | 100.200 |
| `MatMulV2_35_to_v3` | 0 | 100.180 |
| `MatMulV2_40_to_v3` | 0 | 100.140 |
| `MatMulV2_137_to_v3` | 0 | 100.120 |
| `MatMulV2_77_to_v3` | 0 | 100.100 |
| `MatMulV2_16_to_v3` | 0 | 100.100 |
| `MatMulV2_82_to_v3` | 0 | 100.100 |
| `MatMulV2_160_to_v3` | 0 | 100.080 |
| `MatMulV2_125_to_v3` | 0 | 100.060 |
| `MatMulV2_136_to_v3` | 0 | 100.040 |
| `MatMulV2_82_to_v3` | 0 | 99.980 |
| `MatMulV2_53_to_v3` | 0 | 99.940 |
| `MatMulV2_95_to_v3` | 0 | 99.940 |
| `MatMulV2_76_to_v3` | 0 | 99.880 |
| `MatMulV2_29_to_v3` | 0 | 99.840 |
| `MatMulV2_58_to_v3` | 0 | 99.820 |
| `MatMulV2_16_to_v3` | 0 | 99.820 |
| `MatMulV2_11_to_v3` | 0 | 99.740 |
| `MatMulV2_29_to_v3` | 0 | 99.680 |
| `MatMulV2_59_to_v3` | 0 | 99.640 |
| `MatMulV2_28_to_v3` | 0 | 99.560 |
| `MatMulV2_40_to_v3` | 0 | 99.560 |
| `MatMulV2_88_to_v3` | 0 | 99.540 |
| `MatMulV2_46_to_v3` | 0 | 99.480 |
| `MatMulV2_125_to_v3` | 0 | 99.480 |
| `MatMulV2_100_to_v3` | 0 | 99.480 |
| `MatMulV2_16_to_v3` | 0 | 99.460 |
| `MatMulV2_125_to_v3` | 0 | 99.460 |
| `MatMulV2_76_to_v3` | 0 | 99.460 |
| `MatMulV2_70_to_v3` | 0 | 99.440 |
| `MatMulV2_71_to_v3` | 0 | 99.400 |
| `MatMulV2_154_to_v3` | 0 | 99.400 |
| `MatMulV2_23_to_v3` | 0 | 99.380 |
| `MatMulV2_101_to_v3` | 0 | 99.380 |
| `MatMulV2_100_to_v3` | 0 | 99.360 |
| `MatMulV2_52_to_v3` | 0 | 99.340 |
| `MatMulV2_65_to_v3` | 0 | 99.340 |
| `MatMulV2_17_to_v3` | 0 | 99.320 |
| `PromptFlashAttention_4` | 0 | 99.320 |
| `MatMulV2_130_to_v3` | 0 | 99.280 |
| `MatMulV2_23_to_v3` | 0 | 99.280 |
| `MatMulV2_124_to_v3` | 0 | 99.260 |
| `MatMulV2_5_to_v3` | 0 | 99.260 |
| `MatMulV2_82_to_v3` | 0 | 99.220 |
| `MatMulV2_101_to_v3` | 0 | 99.220 |
| `MatMulV2_124_to_v3` | 0 | 99.220 |
| `MatMulV2_88_to_v3` | 0 | 99.180 |
| `MatMulV2_46_to_v3` | 0 | 99.140 |
| `MatMulV2_77_to_v3` | 0 | 99.140 |
| `MatMulV2_52_to_v3` | 0 | 99.140 |
| `MatMulV2_113_to_v3` | 0 | 99.100 |
| `MatMulV2_107_to_v3` | 0 | 99.080 |
| `MatMulV2_35_to_v3` | 0 | 99.040 |
| `MatMulV2_65_to_v3` | 0 | 99.020 |
| `MatMulV2_11_to_v3` | 0 | 99.000 |
| `MatMulV2_59_to_v3` | 0 | 99.000 |
| `MatMulV2_137_to_v3` | 0 | 98.960 |
| `MatMulV2_29_to_v3` | 0 | 98.940 |
| `MatMulV2_71_to_v3` | 0 | 98.920 |
| `MatMulV2_83_to_v3` | 0 | 98.920 |
| `MatMulV2_58_to_v3` | 0 | 98.880 |
| `MatMulV2_41_to_v3` | 0 | 98.880 |
| `PromptFlashAttention_5` | 0 | 98.860 |
| `MatMulV2_22_to_v3` | 0 | 98.860 |
| `MatMulV2_131_to_v3` | 0 | 98.840 |
| `MatMulV2_89_to_v3` | 0 | 98.840 |
| `MatMulV2_113_to_v3` | 0 | 98.820 |
| `MatMulV2_71_to_v3` | 0 | 98.800 |
| `MatMulV2_41_to_v3` | 0 | 98.780 |
| `MatMulV2_130_to_v3` | 0 | 98.780 |
| `MatMulV2_59_to_v3` | 0 | 98.760 |
| `MatMulV2_142_to_v3` | 0 | 98.760 |
| `MatMulV2_22_to_v3` | 0 | 98.720 |
| `MatMulV2_131_to_v3` | 0 | 98.720 |
| `MatMulV2_89_to_v3` | 0 | 98.640 |
| `MatMulV2_58_to_v3` | 0 | 98.640 |
| `MatMulV2_41_to_v3` | 0 | 98.620 |
| `MatMulV2_112_to_v3` | 0 | 98.600 |
| `MatMulV2_47_to_v3` | 0 | 98.600 |
| `MatMulV2_23_to_v3` | 0 | 98.560 |
| `MatMulV2_100_to_v3` | 0 | 98.560 |
| `MatMulV2_83_to_v3` | 0 | 98.540 |
| `MatMulV2_17_to_v3` | 0 | 98.540 |
| `MatMulV2_65_to_v3` | 0 | 98.500 |
| `MatMulV2_70_to_v3` | 0 | 98.500 |
| `MatMulV2_106_to_v3` | 0 | 98.500 |
| `MatMulV2_137_to_v3` | 0 | 98.460 |
| `MatMulV2_136_to_v3` | 0 | 98.380 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 30082.570 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.native.torchair.active.step1` | 1 | 28998.440 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.native.torchair.active.step3` | 1 | 28663.560 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4304.native.torchair.active.step2` | 1 | 28634.960 |
| `TorchDynamo Cache Lookup` | 3 | 27290.360 |
| `Torch-Compiled Region: 0/0` | 3 | 3660.670 |
| `TorchNpuGraphBase::Run` | 3 | 2699.510 |
| `RefreshAtTensorFromGeTensor` | 3 | 1146.790 |
| `aten::empty` | 3 | 561.400 |
| `ExecuteGraph` | 3 | 463.560 |
| `AssembleInputs` | 3 | 393.700 |
| `AssembleOutputs` | 3 | 310.510 |
| `aten::set_` | 3 | 279.430 |
| `empty_tensor` | 3 | 278.270 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 216121.560 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 80620.520 |
| `launch` | 976 | 18000.970 |
| `InputCopy` | 3 | 131.810 |
| `ModelExecute` | 3 | 44.580 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 21.050 |
| `step_info` | 6 | 12.810 |
| `OutputCopy` | 3 | 1.430 |

