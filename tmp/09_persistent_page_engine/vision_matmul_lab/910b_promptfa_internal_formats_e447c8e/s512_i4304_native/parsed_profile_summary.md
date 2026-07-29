# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4304_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4304_native/liteserver-c001-4_627143_20260729134329295_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `47102.340 us`
- `Free`: `3524.460 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3436.250 us`
- `Stage`: `50626.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 14834.320 |
| `Transpose` | 324 | 6263.820 |
| `StridedSliceD` | 405 | 5999.160 |
| `PromptFlashAttention` | 81 | 5116.660 |
| `PadV3` | 243 | 2773.500 |
| `AddLayerNorm` | 162 | 2283.460 |
| `ConcatV2D` | 243 | 1915.220 |
| `Mul` | 324 | 1894.000 |
| `Add` | 162 | 1409.000 |
| `Neg` | 162 | 1332.360 |
| `Cast` | 162 | 1241.040 |
| `Gelu` | 81 | 1240.060 |
| `SplitVD` | 81 | 511.040 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 231.160 |
| `LayerNormV3` | 3 | 41.880 |
| `Data` | 3 | 15.660 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_53` | 3 | 263.680 |
| `MatMulV2_71` | 3 | 256.300 |
| `MatMulV2_41` | 3 | 255.040 |
| `MatMulV2_161` | 3 | 254.720 |
| `MatMulV2_131` | 3 | 252.660 |
| `MatMulV2_29` | 3 | 251.960 |
| `MatMulV2_23` | 3 | 251.880 |
| `MatMulV2_17` | 3 | 251.800 |
| `MatMulV2_89` | 3 | 251.540 |
| `MatMulV2_77` | 3 | 250.940 |
| `MatMulV2_95` | 3 | 250.940 |
| `MatMulV2_137` | 3 | 250.080 |
| `MatMulV2_149` | 3 | 249.820 |
| `MatMulV2_113` | 3 | 249.660 |
| `MatMulV2_155` | 3 | 249.640 |
| `MatMulV2_125` | 3 | 249.480 |
| `MatMulV2_65` | 3 | 249.340 |
| `MatMulV2_119` | 3 | 249.220 |
| `MatMulV2_35` | 3 | 249.160 |
| `MatMulV2_143` | 3 | 249.120 |
| `MatMulV2_59` | 3 | 249.060 |
| `MatMulV2_47` | 3 | 248.920 |
| `MatMulV2_101` | 3 | 248.740 |
| `MatMulV2_5` | 3 | 248.400 |
| `MatMulV2_11` | 3 | 248.320 |
| `MatMulV2_83` | 3 | 248.220 |
| `MatMulV2_107` | 3 | 245.760 |
| `PromptFlashAttention_9` | 3 | 232.180 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 231.160 |
| `PromptFlashAttention_16` | 3 | 213.320 |
| `PromptFlashAttention_15` | 3 | 207.940 |
| `PromptFlashAttention_8` | 3 | 205.260 |
| `PromptFlashAttention_7` | 3 | 198.520 |
| `PromptFlashAttention` | 3 | 197.720 |
| `PromptFlashAttention_17` | 3 | 193.780 |
| `PromptFlashAttention_26` | 3 | 193.000 |
| `PromptFlashAttention_6` | 3 | 190.860 |
| `PromptFlashAttention_25` | 3 | 190.820 |
| `PromptFlashAttention_1` | 3 | 190.160 |
| `PromptFlashAttention_10` | 3 | 188.840 |
| `PromptFlashAttention_5` | 3 | 188.580 |
| `PromptFlashAttention_13` | 3 | 188.580 |
| `PromptFlashAttention_23` | 3 | 185.860 |
| `PromptFlashAttention_12` | 3 | 185.180 |
| `PromptFlashAttention_3` | 3 | 185.120 |
| `PromptFlashAttention_24` | 3 | 182.460 |
| `PromptFlashAttention_14` | 3 | 180.240 |
| `PromptFlashAttention_2` | 3 | 179.980 |
| `PromptFlashAttention_22` | 3 | 179.820 |
| `PromptFlashAttention_20` | 3 | 178.880 |
| `PromptFlashAttention_21` | 3 | 178.520 |
| `PromptFlashAttention_19` | 3 | 175.940 |
| `PromptFlashAttention_11` | 3 | 175.640 |
| `PromptFlashAttention_18` | 3 | 175.020 |
| `PromptFlashAttention_4` | 3 | 174.440 |
| `MatMulV2_94` | 3 | 136.680 |
| `MatMulV2_136` | 3 | 136.560 |
| `MatMulV2_40` | 3 | 131.860 |
| `MatMulV2_4` | 3 | 131.120 |
| `MatMulV2_10` | 3 | 129.800 |
| `MatMulV2_58` | 3 | 129.800 |
| `MatMulV2_70` | 3 | 129.400 |
| `MatMulV2_46` | 3 | 128.840 |
| `MatMulV2_154` | 3 | 128.680 |
| `MatMulV2_52` | 3 | 128.660 |
| `MatMulV2_28` | 3 | 128.580 |
| `MatMulV2_34` | 3 | 128.560 |
| `MatMulV2_118` | 3 | 128.520 |
| `MatMulV2_16` | 3 | 128.480 |
| `MatMulV2_64` | 3 | 128.420 |
| `MatMulV2_22` | 3 | 128.280 |
| `MatMulV2_100` | 3 | 128.200 |
| `MatMulV2_130` | 3 | 127.840 |
| `MatMulV2_82` | 3 | 127.740 |
| `MatMulV2_142` | 3 | 127.720 |
| `MatMulV2_76` | 3 | 127.700 |
| `MatMulV2_160` | 3 | 127.420 |
| `MatMulV2_148` | 3 | 127.120 |
| `MatMulV2_106` | 3 | 127.020 |
| `MatMulV2_88` | 3 | 126.960 |
| `MatMulV2_124` | 3 | 126.820 |
| `MatMulV2_112` | 3 | 126.420 |
| `Transpose_255` | 3 | 78.740 |
| `Transpose_254` | 3 | 76.540 |
| `Transpose_253` | 3 | 74.920 |
| `Transpose_76` | 3 | 68.560 |
| `Transpose_16` | 3 | 68.300 |
| `StridedSliceV2_39` | 3 | 67.440 |
| `StridedSliceV2_9` | 3 | 66.660 |
| `StridedSliceV2_131` | 3 | 60.460 |
| `Transpose_264` | 3 | 59.680 |
| `Transpose_84` | 3 | 59.540 |
| `Transpose_244` | 3 | 59.540 |
| `Transpose_64` | 3 | 59.460 |
| `Transpose_234` | 3 | 59.460 |
| `Transpose_104` | 3 | 59.440 |
| `Transpose_164` | 3 | 59.420 |
| `Transpose_133` | 3 | 59.400 |
| `Transpose_213` | 3 | 59.360 |
| `Transpose_114` | 3 | 59.340 |
| `Transpose_94` | 3 | 59.320 |
| `Transpose_163` | 3 | 59.320 |
| `Transpose_144` | 3 | 59.300 |
| `Transpose_204` | 3 | 59.300 |
| `Transpose_224` | 3 | 59.300 |
| `Transpose_44` | 3 | 59.260 |
| `Transpose_143` | 3 | 59.260 |
| `Transpose_184` | 3 | 59.260 |
| `Transpose_124` | 3 | 59.260 |
| `Transpose_233` | 3 | 59.200 |
| `Transpose_45` | 3 | 59.140 |
| `Transpose_205` | 3 | 59.140 |
| `Transpose_65` | 3 | 59.120 |
| `Transpose_25` | 3 | 59.060 |
| `Transpose_175` | 3 | 59.060 |
| `Transpose_215` | 3 | 59.060 |
| `Transpose_24` | 3 | 59.040 |
| `Transpose_33` | 3 | 59.040 |
| `Transpose_153` | 3 | 59.040 |
| `Transpose_93` | 3 | 59.040 |
| `Transpose_85` | 3 | 59.020 |
| `Transpose_74` | 3 | 58.980 |
| `Transpose_15` | 3 | 58.960 |
| `Transpose_3` | 3 | 58.940 |
| `Transpose_134` | 3 | 58.940 |
| `Transpose_145` | 3 | 58.920 |
| `Transpose_43` | 3 | 58.900 |
| `Transpose_113` | 3 | 58.900 |
| `Transpose_13` | 3 | 58.820 |
| `Transpose_195` | 3 | 58.820 |
| `Transpose_214` | 3 | 58.800 |
| `Transpose_173` | 3 | 58.780 |
| `Transpose_245` | 3 | 58.780 |
| `Transpose_185` | 3 | 58.760 |
| `Transpose_53` | 3 | 58.740 |
| `Transpose_183` | 3 | 58.720 |
| `Transpose_225` | 3 | 58.700 |
| `Transpose_54` | 3 | 58.700 |
| `Transpose_265` | 3 | 58.700 |
| `Transpose_103` | 3 | 58.680 |
| `Transpose_73` | 3 | 58.680 |
| `Transpose_95` | 3 | 58.660 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 58.620 |
| `Transpose_123` | 3 | 58.580 |
| `Transpose_155` | 3 | 58.560 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 58.560 |
| `Transpose_193` | 3 | 58.560 |
| `StridedSliceV2_89` | 3 | 58.520 |
| `Transpose_194` | 3 | 58.500 |
| `Transpose_115` | 3 | 58.480 |
| `Transpose_165` | 3 | 58.480 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 58.480 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 58.460 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 58.460 |
| `Transpose_34` | 3 | 58.440 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 58.440 |
| `StridedSliceV2_79` | 3 | 58.420 |
| `Transpose_105` | 3 | 58.400 |
| `Transpose_63` | 3 | 58.380 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 58.380 |
| `Transpose_135` | 3 | 58.380 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 58.360 |
| `Transpose_23` | 3 | 58.360 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 58.360 |
| `Transpose_154` | 3 | 58.360 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 58.360 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 58.340 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 58.300 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 58.300 |
| `Transpose_35` | 3 | 58.280 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 58.280 |
| `Transpose_55` | 3 | 58.260 |
| `Transpose_125` | 3 | 58.260 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 58.240 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 58.240 |
| `Transpose_14` | 3 | 58.240 |
| `Transpose_263` | 3 | 58.240 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 58.220 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 58.220 |
| `Transpose_174` | 3 | 58.180 |
| `StridedSliceV2_69` | 3 | 58.180 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 58.180 |
| `Transpose_83` | 3 | 58.140 |
| `Transpose_75` | 3 | 58.080 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 58.040 |
| `StridedSliceV2_59` | 3 | 58.020 |
| `Transpose_235` | 3 | 57.940 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 57.940 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 57.900 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 57.880 |
| `Transpose_243` | 3 | 57.880 |
| `Transpose_223` | 3 | 57.880 |
| `StridedSliceV2_49` | 3 | 57.840 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 57.840 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 57.820 |
| `StridedSliceV2_129` | 3 | 57.800 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 57.760 |
| `Transpose_203` | 3 | 57.700 |
| `StridedSliceV2_99` | 3 | 57.680 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 57.660 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "512,4304;1152,4304;1152" -> "512,1152" | ND;ND;ND -> ND` | 81 | 6774.400 |
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | ND;ND;ND;ND -> ND` | 81 | 5116.660 |
| `Transpose | "512,16,72;3" -> "16,512,72" | ND;ND -> ND` | 243 | 4779.080 |
| `MatMulV2 | "512,1152;1152,1152;1152" -> "512,1152" | ND;ND;ND -> ND` | 324 | 4576.720 |
| `StridedSliceD | "1,512,16,72" -> "1,512,16,36" | ND -> ND` | 324 | 4449.900 |
| `MatMulV2 | "512,1152;4304,1152;4304" -> "512,4304" | ND;ND;ND -> ND` | 81 | 3483.200 |
| `PadV3 | "1,16,512,72;8;" -> "1,16,512,80" | ND;ND;ND -> ND` | 243 | 2773.500 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2283.460 |
| `Mul | "1,512,16,72;1,512,1,72" -> "1,512,16,72" | ND;ND -> ND` | 324 | 1894.000 |
| `StridedSliceD | "1,16,512,80" -> "1,16,512,72" | ND -> ND` | 81 | 1549.260 |
| `Transpose | "16,512,72;3" -> "512,16,72" | ND;ND -> ND` | 81 | 1484.740 |
| `Add | "1,512,16,72;1,512,16,72" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1409.000 |
| `ConcatV2D | "1,512,16,36;1,512,16,36" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1345.960 |
| `Neg | "1,512,16,36" -> "1,512,16,36" | ND -> ND` | 162 | 1332.360 |
| `Cast | "1,512,16,72" -> "1,512,16,72" | ND -> ND` | 162 | 1241.040 |
| `Gelu | "1,512,4304" -> "1,512,4304" | ND -> ND` | 81 | 1240.060 |
| `ConcatV2D | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,3456" | ND;ND;ND -> ND` | 81 | 569.260 |
| `SplitVD | "1,512,3456" -> "1,512,1152;1,512,1152;1,512,1152" | ND -> ND;ND;ND` | 81 | 511.040 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 231.160 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 41.880 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.660 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 813 | 18218.960 |
| `ND;ND` | 972 | 10912.780 |
| `ND` | 891 | 10323.660 |
| `ND;ND;ND;ND` | 243 | 7400.120 |
| `N/A` | 246 | 246.820 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_53` | 0 | 89.080 |
| `MatMulV2_53` | 0 | 87.680 |
| `MatMulV2_41` | 0 | 86.980 |
| `MatMulV2_53` | 0 | 86.920 |
| `MatMulV2_23` | 0 | 86.480 |
| `MatMulV2_71` | 0 | 85.960 |
| `MatMulV2_131` | 0 | 85.920 |
| `MatMulV2_161` | 0 | 85.740 |
| `MatMulV2_71` | 0 | 85.240 |
| `MatMulV2_71` | 0 | 85.100 |
| `MatMulV2_17` | 0 | 85.080 |
| `MatMulV2_65` | 0 | 85.080 |
| `MatMulV2_89` | 0 | 85.080 |
| `MatMulV2_35` | 0 | 84.900 |
| `MatMulV2_113` | 0 | 84.840 |
| `MatMulV2_17` | 0 | 84.840 |
| `MatMulV2_161` | 0 | 84.700 |
| `MatMulV2_29` | 0 | 84.640 |
| `MatMulV2_5` | 0 | 84.620 |
| `MatMulV2_89` | 0 | 84.600 |
| `MatMulV2_137` | 0 | 84.560 |
| `MatMulV2_29` | 0 | 84.480 |
| `MatMulV2_131` | 0 | 84.380 |
| `MatMulV2_41` | 0 | 84.360 |
| `MatMulV2_101` | 0 | 84.300 |
| `MatMulV2_161` | 0 | 84.280 |
| `MatMulV2_23` | 0 | 84.200 |
| `MatMulV2_125` | 0 | 84.160 |
| `MatMulV2_35` | 0 | 84.120 |
| `MatMulV2_77` | 0 | 84.080 |
| `MatMulV2_137` | 0 | 84.060 |
| `MatMulV2_59` | 0 | 84.020 |
| `MatMulV2_77` | 0 | 84.020 |
| `MatMulV2_95` | 0 | 83.960 |
| `MatMulV2_95` | 0 | 83.860 |
| `MatMulV2_107` | 0 | 83.800 |
| `MatMulV2_11` | 0 | 83.780 |
| `MatMulV2_41` | 0 | 83.700 |
| `MatMulV2_149` | 0 | 83.700 |
| `MatMulV2_143` | 0 | 83.680 |
| `MatMulV2_5` | 0 | 83.600 |
| `MatMulV2_149` | 0 | 83.520 |
| `MatMulV2_119` | 0 | 83.500 |
| `MatMulV2_155` | 0 | 83.500 |
| `MatMulV2_47` | 0 | 83.440 |
| `MatMulV2_113` | 0 | 83.380 |
| `MatMulV2_83` | 0 | 83.360 |
| `MatMulV2_125` | 0 | 83.260 |
| `MatMulV2_119` | 0 | 83.260 |
| `MatMulV2_143` | 0 | 83.260 |
| `MatMulV2_155` | 0 | 83.220 |
| `MatMulV2_95` | 0 | 83.120 |
| `MatMulV2_47` | 0 | 82.920 |
| `MatMulV2_155` | 0 | 82.920 |
| `MatMulV2_29` | 0 | 82.840 |
| `MatMulV2_77` | 0 | 82.840 |
| `MatMulV2_59` | 0 | 82.800 |
| `MatMulV2_65` | 0 | 82.620 |
| `MatMulV2_149` | 0 | 82.600 |
| `MatMulV2_47` | 0 | 82.560 |
| `MatMulV2_83` | 0 | 82.460 |
| `MatMulV2_119` | 0 | 82.460 |
| `MatMulV2_83` | 0 | 82.400 |
| `MatMulV2_131` | 0 | 82.360 |
| `MatMulV2_101` | 0 | 82.340 |
| `MatMulV2_11` | 0 | 82.300 |
| `MatMulV2_11` | 0 | 82.240 |
| `MatMulV2_59` | 0 | 82.240 |
| `MatMulV2_143` | 0 | 82.180 |
| `MatMulV2_101` | 0 | 82.100 |
| `MatMulV2_125` | 0 | 82.060 |
| `MatMulV2_17` | 0 | 81.880 |
| `MatMulV2_89` | 0 | 81.860 |
| `MatMulV2_65` | 0 | 81.640 |
| `MatMulV2_137` | 0 | 81.460 |
| `MatMulV2_113` | 0 | 81.440 |
| `MatMulV2_107` | 0 | 81.420 |
| `MatMulV2_23` | 0 | 81.200 |
| `MatMulV2_107` | 0 | 80.540 |
| `MatMulV2_5` | 0 | 80.180 |
| `MatMulV2_35` | 0 | 80.140 |
| `PromptFlashAttention_9` | 0 | 77.960 |
| `PromptFlashAttention_9` | 0 | 77.200 |
| `PromptFlashAttention_9` | 0 | 77.020 |
| `PromptFlashAttention_16` | 0 | 71.640 |
| `PromptFlashAttention_16` | 0 | 70.840 |
| `PromptFlashAttention_16` | 0 | 70.840 |
| `PromptFlashAttention_15` | 0 | 69.580 |
| `PromptFlashAttention_15` | 0 | 69.360 |
| `PromptFlashAttention_15` | 0 | 69.000 |
| `PromptFlashAttention_8` | 0 | 68.680 |
| `PromptFlashAttention_8` | 0 | 68.420 |
| `PromptFlashAttention_8` | 0 | 68.160 |
| `PromptFlashAttention` | 0 | 67.100 |
| `PromptFlashAttention_7` | 0 | 67.060 |
| `PromptFlashAttention_7` | 0 | 66.240 |
| `PromptFlashAttention` | 0 | 65.980 |
| `PromptFlashAttention_7` | 0 | 65.220 |
| `PromptFlashAttention_17` | 0 | 65.080 |
| `PromptFlashAttention_6` | 0 | 64.680 |
| `PromptFlashAttention` | 0 | 64.640 |
| `PromptFlashAttention_17` | 0 | 64.620 |
| `PromptFlashAttention_26` | 0 | 64.520 |
| `PromptFlashAttention_1` | 0 | 64.420 |
| `PromptFlashAttention_26` | 0 | 64.360 |
| `PromptFlashAttention_26` | 0 | 64.120 |
| `PromptFlashAttention_17` | 0 | 64.080 |
| `PromptFlashAttention_25` | 0 | 64.000 |
| `PromptFlashAttention_6` | 0 | 63.540 |
| `PromptFlashAttention_25` | 0 | 63.480 |
| `PromptFlashAttention_10` | 0 | 63.460 |
| `PromptFlashAttention_25` | 0 | 63.340 |
| `PromptFlashAttention_13` | 0 | 63.100 |
| `PromptFlashAttention_12` | 0 | 63.000 |
| `PromptFlashAttention_1` | 0 | 63.000 |
| `PromptFlashAttention_13` | 0 | 62.980 |
| `PromptFlashAttention_5` | 0 | 62.960 |
| `PromptFlashAttention_5` | 0 | 62.920 |
| `PromptFlashAttention_10` | 0 | 62.880 |
| `PromptFlashAttention_1` | 0 | 62.740 |
| `PromptFlashAttention_5` | 0 | 62.700 |
| `PromptFlashAttention_6` | 0 | 62.640 |
| `PromptFlashAttention_13` | 0 | 62.500 |
| `PromptFlashAttention_10` | 0 | 62.500 |
| `PromptFlashAttention_3` | 0 | 62.460 |
| `PromptFlashAttention_12` | 0 | 62.420 |
| `PromptFlashAttention_23` | 0 | 62.180 |
| `PromptFlashAttention_23` | 0 | 61.880 |
| `PromptFlashAttention_23` | 0 | 61.800 |
| `PromptFlashAttention_3` | 0 | 61.340 |
| `PromptFlashAttention_3` | 0 | 61.320 |
| `PromptFlashAttention_14` | 0 | 61.060 |
| `PromptFlashAttention_24` | 0 | 60.880 |
| `PromptFlashAttention_24` | 0 | 60.840 |
| `PromptFlashAttention_24` | 0 | 60.740 |
| `PromptFlashAttention_2` | 0 | 60.560 |
| `PromptFlashAttention_22` | 0 | 60.220 |
| `PromptFlashAttention_20` | 0 | 60.100 |
| `PromptFlashAttention_21` | 0 | 60.080 |
| `PromptFlashAttention_22` | 0 | 60.020 |
| `PromptFlashAttention_14` | 0 | 60.000 |
| `PromptFlashAttention_2` | 0 | 59.940 |
| `PromptFlashAttention_12` | 0 | 59.760 |
| `PromptFlashAttention_20` | 0 | 59.660 |
| `PromptFlashAttention_22` | 0 | 59.580 |
| `PromptFlashAttention_2` | 0 | 59.480 |
| `PromptFlashAttention_21` | 0 | 59.240 |
| `PromptFlashAttention_11` | 0 | 59.220 |
| `PromptFlashAttention_21` | 0 | 59.200 |
| `PromptFlashAttention_14` | 0 | 59.180 |
| `PromptFlashAttention_19` | 0 | 59.140 |
| `PromptFlashAttention_20` | 0 | 59.120 |
| `PromptFlashAttention_11` | 0 | 58.780 |
| `PromptFlashAttention_19` | 0 | 58.720 |
| `PromptFlashAttention_18` | 0 | 58.580 |
| `PromptFlashAttention_18` | 0 | 58.440 |
| `PromptFlashAttention_4` | 0 | 58.340 |
| `PromptFlashAttention_4` | 0 | 58.120 |
| `PromptFlashAttention_19` | 0 | 58.080 |
| `PromptFlashAttention_18` | 0 | 58.000 |
| `PromptFlashAttention_4` | 0 | 57.980 |
| `PromptFlashAttention_11` | 0 | 57.640 |
| `MatMulV2_94` | 0 | 46.280 |
| `MatMulV2_136` | 0 | 45.620 |
| `MatMulV2_136` | 0 | 45.560 |
| `MatMulV2_94` | 0 | 45.420 |
| `MatMulV2_136` | 0 | 45.380 |
| `MatMulV2_94` | 0 | 44.980 |
| `MatMulV2_10` | 0 | 44.520 |
| `MatMulV2_40` | 0 | 44.420 |
| `MatMulV2_70` | 0 | 44.020 |
| `MatMulV2_40` | 0 | 43.900 |
| `MatMulV2_4` | 0 | 43.900 |
| `MatMulV2_16` | 0 | 43.820 |
| `MatMulV2_52` | 0 | 43.800 |
| `MatMulV2_4` | 0 | 43.800 |
| `MatMulV2_34` | 0 | 43.760 |
| `MatMulV2_58` | 0 | 43.740 |
| `MatMulV2_46` | 0 | 43.660 |
| `MatMulV2_40` | 0 | 43.540 |
| `MatMulV2_4` | 0 | 43.420 |
| `MatMulV2_154` | 0 | 43.420 |
| `MatMulV2_160` | 0 | 43.400 |
| `MatMulV2_64` | 0 | 43.280 |
| `MatMulV2_118` | 0 | 43.240 |
| `MatMulV2_28` | 0 | 43.140 |
| `MatMulV2_58` | 0 | 43.100 |
| `MatMulV2_70` | 0 | 43.080 |
| `MatMulV2_154` | 0 | 43.080 |
| `MatMulV2_142` | 0 | 43.040 |
| `MatMulV2_76` | 0 | 43.020 |
| `MatMulV2_22` | 0 | 42.980 |
| `MatMulV2_100` | 0 | 42.980 |
| `MatMulV2_58` | 0 | 42.960 |
| `MatMulV2_82` | 0 | 42.960 |
| `MatMulV2_106` | 0 | 42.920 |
| `MatMulV2_124` | 0 | 42.920 |
| `MatMulV2_52` | 0 | 42.900 |
| `MatMulV2_142` | 0 | 42.880 |
| `MatMulV2_100` | 0 | 42.840 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 18628.300 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step1` | 1 | 17487.640 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step3` | 1 | 17098.750 |
| `paddleocr_vl.vision_matmul_lab.S512.I4304.native.torchair.active.step2` | 1 | 17058.860 |
| `TorchDynamo Cache Lookup` | 3 | 15814.720 |
| `Torch-Compiled Region: 0/0` | 3 | 3654.340 |
| `TorchNpuGraphBase::Run` | 3 | 2660.750 |
| `RefreshAtTensorFromGeTensor` | 3 | 1122.600 |
| `aten::empty` | 3 | 540.850 |
| `ExecuteGraph` | 3 | 491.890 |
| `AssembleInputs` | 3 | 381.340 |
| `AssembleOutputs` | 3 | 282.640 |
| `aten::set_` | 3 | 272.900 |
| `empty_tensor` | 3 | 267.500 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 220069.610 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 45886.250 |
| `launch` | 976 | 18285.620 |
| `InputCopy` | 3 | 156.030 |
| `ModelExecute` | 3 | 47.740 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 29.410 |
| `step_info` | 6 | 15.280 |
| `OutputCopy` | 3 | 0.800 |

