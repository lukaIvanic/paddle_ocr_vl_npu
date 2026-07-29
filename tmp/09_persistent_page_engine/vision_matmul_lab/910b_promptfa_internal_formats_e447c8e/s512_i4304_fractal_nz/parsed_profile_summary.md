# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4304_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4304_fractal_nz/liteserver-c001-4_628556_20260729134441517_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `42217.580 us`
- `Free`: `3483.820 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3357.250 us`
- `Stage`: `45701.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 9939.960 |
| `Transpose` | 324 | 6469.080 |
| `StridedSliceD` | 405 | 6180.620 |
| `PromptFlashAttention` | 81 | 5065.900 |
| `PadV3` | 243 | 2713.540 |
| `AddLayerNorm` | 162 | 2228.760 |
| `ConcatV2D` | 243 | 1873.520 |
| `Mul` | 324 | 1845.240 |
| `Add` | 162 | 1348.640 |
| `Neg` | 162 | 1273.300 |
| `Gelu` | 81 | 1269.900 |
| `Cast` | 162 | 1212.720 |
| `SplitVD` | 81 | 503.000 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 231.620 |
| `LayerNormV3` | 3 | 46.180 |
| `Data` | 3 | 15.600 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 231.620 |
| `PromptFlashAttention_13` | 3 | 212.220 |
| `PromptFlashAttention_14` | 3 | 207.720 |
| `PromptFlashAttention` | 3 | 202.380 |
| `PromptFlashAttention_24` | 3 | 200.120 |
| `PromptFlashAttention_9` | 3 | 200.080 |
| `PromptFlashAttention_17` | 3 | 198.560 |
| `PromptFlashAttention_16` | 3 | 196.120 |
| `PromptFlashAttention_12` | 3 | 193.160 |
| `PromptFlashAttention_7` | 3 | 191.460 |
| `PromptFlashAttention_25` | 3 | 190.500 |
| `PromptFlashAttention_15` | 3 | 188.280 |
| `PromptFlashAttention_6` | 3 | 187.780 |
| `PromptFlashAttention_8` | 3 | 187.680 |
| `PromptFlashAttention_20` | 3 | 187.540 |
| `PromptFlashAttention_1` | 3 | 187.460 |
| `PromptFlashAttention_26` | 3 | 186.680 |
| `PromptFlashAttention_4` | 3 | 183.060 |
| `PromptFlashAttention_19` | 3 | 181.620 |
| `PromptFlashAttention_10` | 3 | 180.700 |
| `PromptFlashAttention_5` | 3 | 179.220 |
| `PromptFlashAttention_22` | 3 | 177.520 |
| `PromptFlashAttention_18` | 3 | 177.340 |
| `PromptFlashAttention_11` | 3 | 175.660 |
| `PromptFlashAttention_23` | 3 | 175.220 |
| `PromptFlashAttention_3` | 3 | 174.840 |
| `PromptFlashAttention_21` | 3 | 174.780 |
| `PromptFlashAttention_2` | 3 | 168.200 |
| `MatMulV2_46` | 3 | 107.300 |
| `MatMulV2_137` | 3 | 106.920 |
| `MatMulV2_83` | 3 | 106.800 |
| `MatMulV2_119` | 3 | 106.560 |
| `MatMulV2_11` | 3 | 105.840 |
| `MatMulV2_155` | 3 | 105.660 |
| `MatMulV2_35` | 3 | 105.500 |
| `MatMulV2_125` | 3 | 105.100 |
| `MatMulV2_47` | 3 | 104.880 |
| `MatMulV2_94` | 3 | 104.760 |
| `MatMulV2_40` | 3 | 104.700 |
| `MatMulV2_71` | 3 | 104.560 |
| `MatMulV2_95` | 3 | 104.520 |
| `MatMulV2_136` | 3 | 104.420 |
| `MatMulV2_143` | 3 | 104.300 |
| `MatMulV2_113` | 3 | 104.280 |
| `MatMulV2_77` | 3 | 104.180 |
| `MatMulV2_59` | 3 | 103.980 |
| `MatMulV2_131` | 3 | 103.960 |
| `MatMulV2_5` | 3 | 103.840 |
| `MatMulV2_149` | 3 | 103.700 |
| `MatMulV2_101` | 3 | 103.640 |
| `MatMulV2_76` | 3 | 103.620 |
| `MatMulV2_161` | 3 | 103.340 |
| `MatMulV2_53` | 3 | 103.300 |
| `MatMulV2_4` | 3 | 103.200 |
| `MatMulV2_89` | 3 | 103.080 |
| `MatMulV2_23` | 3 | 103.020 |
| `MatMulV2_65` | 3 | 102.980 |
| `MatMulV2_142` | 3 | 102.820 |
| `MatMulV2_82` | 3 | 102.600 |
| `MatMulV2_34` | 3 | 102.480 |
| `MatMulV2_28` | 3 | 102.300 |
| `MatMulV2_107` | 3 | 102.300 |
| `MatMulV2_29` | 3 | 102.300 |
| `MatMulV2_154` | 3 | 102.160 |
| `MatMulV2_100` | 3 | 102.120 |
| `MatMulV2_52` | 3 | 101.880 |
| `MatMulV2_124` | 3 | 101.700 |
| `MatMulV2_58` | 3 | 101.480 |
| `MatMulV2_112` | 3 | 101.420 |
| `MatMulV2_70` | 3 | 101.320 |
| `MatMulV2_17` | 3 | 101.220 |
| `MatMulV2_130` | 3 | 101.020 |
| `MatMulV2_41` | 3 | 101.000 |
| `MatMulV2_10` | 3 | 100.740 |
| `MatMulV2_22` | 3 | 100.660 |
| `MatMulV2_160` | 3 | 100.200 |
| `MatMulV2_118` | 3 | 99.360 |
| `MatMulV2_88` | 3 | 98.940 |
| `MatMulV2_16` | 3 | 98.420 |
| `MatMulV2_148` | 3 | 98.360 |
| `MatMulV2_64` | 3 | 96.640 |
| `MatMulV2_106` | 3 | 96.620 |
| `Transpose_255` | 3 | 83.380 |
| `Transpose_254` | 3 | 78.920 |
| `Transpose_253` | 3 | 78.800 |
| `MatMulV2_147` | 3 | 69.760 |
| `MatMulV2` | 3 | 69.360 |
| `StridedSliceV2_39` | 3 | 69.320 |
| `Transpose_16` | 3 | 68.940 |
| `Transpose_76` | 3 | 68.760 |
| `MatMulV2_117` | 3 | 68.020 |
| `MatMulV2_99` | 3 | 68.000 |
| `StridedSliceV2_9` | 3 | 67.420 |
| `MatMulV2_135` | 3 | 66.900 |
| `MatMulV2_75` | 3 | 66.440 |
| `MatMulV2_93` | 3 | 65.580 |
| `MatMulV2_129` | 3 | 65.440 |
| `MatMulV2_105` | 3 | 64.200 |
| `MatMulV2_141` | 3 | 63.620 |
| `StridedSliceV2_131` | 3 | 63.580 |
| `MatMulV2_123` | 3 | 63.260 |
| `MatMulV2_63` | 3 | 63.120 |
| `MatMulV2_51` | 3 | 63.000 |
| `MatMulV2_81` | 3 | 62.920 |
| `Transpose_114` | 3 | 62.180 |
| `MatMulV2_57` | 3 | 62.080 |
| `Transpose_154` | 3 | 61.940 |
| `MatMulV2_69` | 3 | 61.920 |
| `Transpose_194` | 3 | 61.860 |
| `Transpose_63` | 3 | 61.800 |
| `Transpose_174` | 3 | 61.800 |
| `Transpose_134` | 3 | 61.780 |
| `Transpose_144` | 3 | 61.780 |
| `Transpose_94` | 3 | 61.740 |
| `Transpose_43` | 3 | 61.700 |
| `Transpose_34` | 3 | 61.680 |
| `MatMulV2_9` | 3 | 61.680 |
| `Transpose_223` | 3 | 61.660 |
| `Transpose_74` | 3 | 61.580 |
| `Transpose_53` | 3 | 61.520 |
| `Transpose_14` | 3 | 61.500 |
| `Transpose_203` | 3 | 61.480 |
| `Transpose_214` | 3 | 61.480 |
| `Transpose_85` | 3 | 61.440 |
| `Transpose_183` | 3 | 61.440 |
| `Transpose_264` | 3 | 61.420 |
| `Transpose_195` | 3 | 61.400 |
| `Transpose_243` | 3 | 61.400 |
| `Transpose_33` | 3 | 61.380 |
| `Transpose_54` | 3 | 61.360 |
| `Transpose_215` | 3 | 61.360 |
| `Transpose_3` | 3 | 61.340 |
| `Transpose_65` | 3 | 61.340 |
| `Transpose_75` | 3 | 61.340 |
| `Transpose_163` | 3 | 61.320 |
| `Transpose_234` | 3 | 61.320 |
| `Transpose_204` | 3 | 61.320 |
| `MatMulV2_87` | 3 | 61.300 |
| `Transpose_84` | 3 | 61.280 |
| `Transpose_23` | 3 | 61.260 |
| `Transpose_245` | 3 | 61.260 |
| `MatMulV2_21` | 3 | 61.220 |
| `Transpose_124` | 3 | 61.220 |
| `Transpose_193` | 3 | 61.140 |
| `Transpose_55` | 3 | 61.120 |
| `Transpose_123` | 3 | 61.120 |
| `Transpose_103` | 3 | 61.100 |
| `Transpose_265` | 3 | 61.100 |
| `Transpose_135` | 3 | 61.100 |
| `Transpose_143` | 3 | 61.100 |
| `Transpose_105` | 3 | 61.080 |
| `Transpose_164` | 3 | 61.080 |
| `Transpose_45` | 3 | 61.060 |
| `Transpose_155` | 3 | 61.060 |
| `Transpose_175` | 3 | 61.040 |
| `Transpose_104` | 3 | 61.020 |
| `Transpose_83` | 3 | 61.000 |
| `Transpose_173` | 3 | 61.000 |
| `Transpose_133` | 3 | 60.980 |
| `Transpose_184` | 3 | 60.940 |
| `Transpose_233` | 3 | 60.860 |
| `Transpose_205` | 3 | 60.840 |
| `Transpose_224` | 3 | 60.840 |
| `Transpose_95` | 3 | 60.820 |
| `Transpose_263` | 3 | 60.760 |
| `Transpose_35` | 3 | 60.720 |
| `Transpose_73` | 3 | 60.700 |
| `Transpose_125` | 3 | 60.700 |
| `Transpose_24` | 3 | 60.680 |
| `Transpose_153` | 3 | 60.680 |
| `Transpose_213` | 3 | 60.660 |
| `Transpose_13` | 3 | 60.660 |
| `Transpose_235` | 3 | 60.600 |
| `Transpose_15` | 3 | 60.600 |
| `Transpose_185` | 3 | 60.560 |
| `Transpose_244` | 3 | 60.480 |
| `Transpose_64` | 3 | 60.460 |
| `Transpose_44` | 3 | 60.460 |
| `MatMulV2_15` | 3 | 60.440 |
| `Transpose_145` | 3 | 60.360 |
| `StridedSliceV2_99` | 3 | 60.260 |
| `Transpose_225` | 3 | 60.180 |
| `Transpose_93` | 3 | 60.160 |
| `Transpose_115` | 3 | 60.120 |
| `Transpose_25` | 3 | 60.120 |
| `MatMulV2_45` | 3 | 60.040 |
| `Transpose_165` | 3 | 59.860 |
| `Transpose_113` | 3 | 59.840 |
| `StridedSliceV2_69` | 3 | 59.340 |
| `StridedSliceV2_109` | 3 | 59.180 |
| `MatMulV2_3` | 3 | 59.000 |
| `StridedSliceV2_119` | 3 | 58.840 |
| `StridedSliceV2_29` | 3 | 58.820 |
| `StridedSliceV2_59` | 3 | 58.480 |
| `MatMulV2_39` | 3 | 58.400 |
| `StridedSliceV2_19` | 3 | 58.320 |
| `StridedSliceV2_89` | 3 | 58.280 |
| `StridedSliceV2_79` | 3 | 58.220 |
| `StridedSliceV2_129` | 3 | 58.100 |
| `StridedSliceV2_49` | 3 | 58.020 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | ND;ND;ND;ND -> ND` | 81 | 5065.900 |
| `Transpose | "512,16,72;3" -> "16,512,72" | ND;ND -> ND` | 243 | 4962.400 |
| `StridedSliceD | "1,512,16,72" -> "1,512,16,36" | ND -> ND` | 324 | 4616.020 |
| `MatMulV2 | "512,1152;72,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 4387.960 |
| `MatMulV2 | "512,4304;269,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2810.760 |
| `MatMulV2 | "512,1152;72,269,16,16;4304" -> "512,4304" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2741.240 |
| `PadV3 | "1,16,512,72;8;" -> "1,16,512,80" | ND;ND;ND -> ND` | 243 | 2713.540 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2228.760 |
| `Mul | "1,512,16,72;1,512,1,72" -> "1,512,16,72" | ND;ND -> ND` | 324 | 1845.240 |
| `StridedSliceD | "1,16,512,80" -> "1,16,512,72" | ND -> ND` | 81 | 1564.600 |
| `Transpose | "16,512,72;3" -> "512,16,72" | ND;ND -> ND` | 81 | 1506.680 |
| `Add | "1,512,16,72;1,512,16,72" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1348.640 |
| `ConcatV2D | "1,512,16,36;1,512,16,36" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1316.000 |
| `Neg | "1,512,16,36" -> "1,512,16,36" | ND -> ND` | 162 | 1273.300 |
| `Gelu | "1,512,4304" -> "1,512,4304" | ND -> ND` | 81 | 1269.900 |
| `Cast | "1,512,16,72" -> "1,512,16,72" | ND -> ND` | 162 | 1212.720 |
| `ConcatV2D | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,3456" | ND;ND;ND -> ND` | 81 | 557.520 |
| `SplitVD | "1,512,3456" -> "1,512,1152;1,512,1152;1,512,1152" | ND -> ND;ND;ND` | 81 | 503.000 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 231.620 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 46.180 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.600 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND` | 972 | 10978.960 |
| `ND` | 891 | 10439.540 |
| `ND;FRACTAL_NZ;ND` | 486 | 9939.960 |
| `ND;ND;ND;ND` | 243 | 7294.660 |
| `ND;ND;ND` | 327 | 3317.240 |
| `N/A` | 246 | 247.220 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_13` | 0 | 71.000 |
| `PromptFlashAttention_13` | 0 | 70.820 |
| `PromptFlashAttention_13` | 0 | 70.400 |
| `PromptFlashAttention_14` | 0 | 69.700 |
| `PromptFlashAttention_14` | 0 | 69.160 |
| `PromptFlashAttention_14` | 0 | 68.860 |
| `PromptFlashAttention` | 0 | 67.840 |
| `PromptFlashAttention` | 0 | 67.360 |
| `PromptFlashAttention_9` | 0 | 67.360 |
| `PromptFlashAttention` | 0 | 67.180 |
| `PromptFlashAttention_24` | 0 | 67.140 |
| `PromptFlashAttention_17` | 0 | 66.940 |
| `PromptFlashAttention_24` | 0 | 66.540 |
| `PromptFlashAttention_24` | 0 | 66.440 |
| `PromptFlashAttention_9` | 0 | 66.380 |
| `PromptFlashAttention_9` | 0 | 66.340 |
| `PromptFlashAttention_17` | 0 | 65.980 |
| `PromptFlashAttention_16` | 0 | 65.880 |
| `PromptFlashAttention_17` | 0 | 65.640 |
| `PromptFlashAttention_16` | 0 | 65.160 |
| `PromptFlashAttention_16` | 0 | 65.080 |
| `PromptFlashAttention_25` | 0 | 64.640 |
| `PromptFlashAttention_12` | 0 | 64.560 |
| `PromptFlashAttention_12` | 0 | 64.560 |
| `PromptFlashAttention_7` | 0 | 64.200 |
| `PromptFlashAttention_12` | 0 | 64.040 |
| `PromptFlashAttention_7` | 0 | 63.700 |
| `PromptFlashAttention_7` | 0 | 63.560 |
| `PromptFlashAttention_20` | 0 | 63.380 |
| `PromptFlashAttention_6` | 0 | 63.240 |
| `PromptFlashAttention_25` | 0 | 63.240 |
| `PromptFlashAttention_15` | 0 | 63.100 |
| `PromptFlashAttention_6` | 0 | 63.100 |
| `PromptFlashAttention_8` | 0 | 62.900 |
| `PromptFlashAttention_15` | 0 | 62.900 |
| `PromptFlashAttention_8` | 0 | 62.880 |
| `PromptFlashAttention_1` | 0 | 62.740 |
| `PromptFlashAttention_26` | 0 | 62.620 |
| `PromptFlashAttention_25` | 0 | 62.620 |
| `PromptFlashAttention_1` | 0 | 62.540 |
| `PromptFlashAttention_15` | 0 | 62.280 |
| `PromptFlashAttention_1` | 0 | 62.180 |
| `PromptFlashAttention_20` | 0 | 62.180 |
| `PromptFlashAttention_26` | 0 | 62.080 |
| `PromptFlashAttention_4` | 0 | 62.060 |
| `PromptFlashAttention_26` | 0 | 61.980 |
| `PromptFlashAttention_20` | 0 | 61.980 |
| `PromptFlashAttention_8` | 0 | 61.900 |
| `PromptFlashAttention_6` | 0 | 61.440 |
| `PromptFlashAttention_4` | 0 | 61.320 |
| `PromptFlashAttention_19` | 0 | 60.920 |
| `PromptFlashAttention_10` | 0 | 60.720 |
| `PromptFlashAttention_5` | 0 | 60.520 |
| `PromptFlashAttention_19` | 0 | 60.480 |
| `PromptFlashAttention_19` | 0 | 60.220 |
| `PromptFlashAttention_18` | 0 | 60.180 |
| `PromptFlashAttention_10` | 0 | 60.140 |
| `PromptFlashAttention_10` | 0 | 59.840 |
| `PromptFlashAttention_22` | 0 | 59.800 |
| `PromptFlashAttention_4` | 0 | 59.680 |
| `PromptFlashAttention_11` | 0 | 59.640 |
| `PromptFlashAttention_23` | 0 | 59.600 |
| `PromptFlashAttention_5` | 0 | 59.540 |
| `PromptFlashAttention_5` | 0 | 59.160 |
| `PromptFlashAttention_22` | 0 | 59.140 |
| `PromptFlashAttention_3` | 0 | 58.760 |
| `PromptFlashAttention_21` | 0 | 58.640 |
| `PromptFlashAttention_18` | 0 | 58.600 |
| `PromptFlashAttention_22` | 0 | 58.580 |
| `PromptFlashAttention_18` | 0 | 58.560 |
| `PromptFlashAttention_21` | 0 | 58.520 |
| `PromptFlashAttention_3` | 0 | 58.160 |
| `PromptFlashAttention_23` | 0 | 58.100 |
| `PromptFlashAttention_11` | 0 | 58.020 |
| `PromptFlashAttention_11` | 0 | 58.000 |
| `PromptFlashAttention_3` | 0 | 57.920 |
| `PromptFlashAttention_21` | 0 | 57.620 |
| `PromptFlashAttention_23` | 0 | 57.520 |
| `PromptFlashAttention_2` | 0 | 56.840 |
| `PromptFlashAttention_2` | 0 | 56.800 |
| `PromptFlashAttention_2` | 0 | 54.560 |
| `MatMulV2_40` | 0 | 36.440 |
| `MatMulV2_46` | 0 | 36.420 |
| `MatMulV2_83` | 0 | 36.280 |
| `MatMulV2_137` | 0 | 36.180 |
| `MatMulV2_136` | 0 | 36.160 |
| `MatMulV2_137` | 0 | 36.100 |
| `MatMulV2_11` | 0 | 35.800 |
| `MatMulV2_46` | 0 | 35.760 |
| `MatMulV2_35` | 0 | 35.760 |
| `MatMulV2_119` | 0 | 35.760 |
| `MatMulV2_83` | 0 | 35.640 |
| `MatMulV2_59` | 0 | 35.620 |
| `MatMulV2_95` | 0 | 35.620 |
| `MatMulV2_119` | 0 | 35.620 |
| `MatMulV2_5` | 0 | 35.580 |
| `MatMulV2_113` | 0 | 35.560 |
| `MatMulV2_53` | 0 | 35.480 |
| `MatMulV2_11` | 0 | 35.460 |
| `MatMulV2_125` | 0 | 35.460 |
| `MatMulV2_71` | 0 | 35.460 |
| `MatMulV2_52` | 0 | 35.420 |
| `MatMulV2_94` | 0 | 35.360 |
| `MatMulV2_47` | 0 | 35.340 |
| `MatMulV2_155` | 0 | 35.340 |
| `MatMulV2_4` | 0 | 35.260 |
| `MatMulV2_119` | 0 | 35.180 |
| `MatMulV2_155` | 0 | 35.180 |
| `MatMulV2_155` | 0 | 35.140 |
| `MatMulV2_46` | 0 | 35.120 |
| `MatMulV2_143` | 0 | 35.080 |
| `MatMulV2_10` | 0 | 35.020 |
| `MatMulV2_95` | 0 | 35.020 |
| `MatMulV2_47` | 0 | 34.980 |
| `MatMulV2_35` | 0 | 34.960 |
| `MatMulV2_23` | 0 | 34.960 |
| `MatMulV2_125` | 0 | 34.960 |
| `MatMulV2_131` | 0 | 34.940 |
| `MatMulV2_161` | 0 | 34.920 |
| `MatMulV2_76` | 0 | 34.920 |
| `MatMulV2_76` | 0 | 34.900 |
| `MatMulV2_83` | 0 | 34.880 |
| `MatMulV2_94` | 0 | 34.880 |
| `MatMulV2_77` | 0 | 34.860 |
| `MatMulV2_35` | 0 | 34.780 |
| `MatMulV2_143` | 0 | 34.760 |
| `MatMulV2_149` | 0 | 34.760 |
| `MatMulV2_89` | 0 | 34.740 |
| `MatMulV2_101` | 0 | 34.720 |
| `MatMulV2_149` | 0 | 34.700 |
| `MatMulV2_77` | 0 | 34.700 |
| `MatMulV2_125` | 0 | 34.680 |
| `MatMulV2_23` | 0 | 34.680 |
| `MatMulV2_29` | 0 | 34.680 |
| `MatMulV2_101` | 0 | 34.680 |
| `MatMulV2_17` | 0 | 34.660 |
| `MatMulV2_137` | 0 | 34.640 |
| `MatMulV2_71` | 0 | 34.640 |
| `MatMulV2_77` | 0 | 34.620 |
| `MatMulV2_131` | 0 | 34.600 |
| `MatMulV2_11` | 0 | 34.580 |
| `MatMulV2_52` | 0 | 34.580 |
| `MatMulV2_107` | 0 | 34.580 |
| `MatMulV2_118` | 0 | 34.580 |
| `MatMulV2_47` | 0 | 34.560 |
| `MatMulV2_94` | 0 | 34.520 |
| `MatMulV2_142` | 0 | 34.520 |
| `MatMulV2_65` | 0 | 34.500 |
| `MatMulV2_65` | 0 | 34.500 |
| `MatMulV2_113` | 0 | 34.480 |
| `MatMulV2_5` | 0 | 34.460 |
| `MatMulV2_71` | 0 | 34.460 |
| `MatMulV2_143` | 0 | 34.460 |
| `MatMulV2_131` | 0 | 34.420 |
| `MatMulV2_136` | 0 | 34.380 |
| `MatMulV2_40` | 0 | 34.380 |
| `MatMulV2_59` | 0 | 34.380 |
| `MatMulV2_34` | 0 | 34.360 |
| `MatMulV2_161` | 0 | 34.340 |
| `MatMulV2_142` | 0 | 34.340 |
| `MatMulV2_58` | 0 | 34.320 |
| `MatMulV2_89` | 0 | 34.300 |
| `MatMulV2_154` | 0 | 34.280 |
| `MatMulV2_82` | 0 | 34.280 |
| `MatMulV2_28` | 0 | 34.280 |
| `MatMulV2_100` | 0 | 34.260 |
| `MatMulV2_29` | 0 | 34.260 |
| `MatMulV2_101` | 0 | 34.240 |
| `MatMulV2_149` | 0 | 34.240 |
| `MatMulV2_22` | 0 | 34.240 |
| `MatMulV2_82` | 0 | 34.240 |
| `MatMulV2_113` | 0 | 34.240 |
| `MatMulV2_4` | 0 | 34.220 |
| `MatMulV2_88` | 0 | 34.180 |
| `MatMulV2_70` | 0 | 34.140 |
| `MatMulV2_53` | 0 | 34.120 |
| `MatMulV2_107` | 0 | 34.100 |
| `MatMulV2_34` | 0 | 34.080 |
| `MatMulV2_82` | 0 | 34.080 |
| `MatMulV2_154` | 0 | 34.080 |
| `MatMulV2_161` | 0 | 34.080 |
| `MatMulV2_28` | 0 | 34.060 |
| `MatMulV2_89` | 0 | 34.040 |
| `MatMulV2_34` | 0 | 34.040 |
| `MatMulV2_65` | 0 | 33.980 |
| `MatMulV2_124` | 0 | 33.980 |
| `MatMulV2_59` | 0 | 33.980 |
| `MatMulV2_142` | 0 | 33.960 |
| `MatMulV2_28` | 0 | 33.960 |
| `MatMulV2_100` | 0 | 33.960 |
| `MatMulV2_124` | 0 | 33.960 |
| `MatMulV2_130` | 0 | 33.920 |
| `MatMulV2_160` | 0 | 33.920 |
| `MatMulV2_100` | 0 | 33.900 |
| `MatMulV2_112` | 0 | 33.900 |
| `MatMulV2_41` | 0 | 33.880 |
| `MatMulV2_95` | 0 | 33.880 |
| `MatMulV2_40` | 0 | 33.880 |
| `MatMulV2_136` | 0 | 33.880 |
| `MatMulV2_64` | 0 | 33.860 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 16903.930 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.fractal_nz.torchair.active.step1` | 1 | 15746.540 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.fractal_nz.torchair.active.step3` | 1 | 15424.220 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.fractal_nz.torchair.active.step2` | 1 | 15420.430 |
| `TorchDynamo Cache Lookup` | 3 | 14131.920 |
| `Torch-Compiled Region: 0/0` | 3 | 3592.110 |
| `TorchNpuGraphBase::Run` | 3 | 2625.310 |
| `RefreshAtTensorFromGeTensor` | 3 | 1133.740 |
| `aten::empty` | 3 | 545.540 |
| `ExecuteGraph` | 3 | 477.540 |
| `AssembleInputs` | 3 | 352.690 |
| `aten::set_` | 3 | 290.830 |
| `AssembleOutputs` | 3 | 286.110 |
| `empty_tensor` | 3 | 269.050 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 236026.850 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 40978.330 |
| `launch` | 976 | 18163.260 |
| `InputCopy` | 3 | 147.960 |
| `ModelExecute` | 3 | 47.710 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 27.060 |
| `step_info` | 6 | 26.610 |
| `OutputCopy` | 3 | 1.050 |

