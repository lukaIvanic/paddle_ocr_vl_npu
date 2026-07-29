# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4352_native`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4352_native/liteserver-c001-4_630391_20260729134552294_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `42197.320 us`
- `Free`: `3554.600 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3420.500 us`
- `Stage`: `45751.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 9977.820 |
| `StridedSliceD` | 405 | 6392.100 |
| `Transpose` | 324 | 6041.540 |
| `PromptFlashAttention` | 81 | 5171.520 |
| `PadV3` | 243 | 2952.940 |
| `AddLayerNorm` | 162 | 2285.320 |
| `ConcatV2D` | 243 | 1892.820 |
| `Mul` | 324 | 1711.380 |
| `Add` | 162 | 1273.220 |
| `Neg` | 162 | 1272.780 |
| `Gelu` | 81 | 1226.820 |
| `Cast` | 162 | 1208.380 |
| `SplitVD` | 81 | 496.500 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 233.680 |
| `LayerNormV3` | 3 | 44.740 |
| `Data` | 3 | 15.760 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 233.680 |
| `PromptFlashAttention_9` | 3 | 233.100 |
| `PromptFlashAttention_16` | 3 | 221.800 |
| `PromptFlashAttention_10` | 3 | 216.220 |
| `PromptFlashAttention_17` | 3 | 214.260 |
| `PromptFlashAttention_15` | 3 | 203.940 |
| `PromptFlashAttention_8` | 3 | 203.300 |
| `PromptFlashAttention_26` | 3 | 198.340 |
| `PromptFlashAttention_5` | 3 | 196.120 |
| `PromptFlashAttention_25` | 3 | 196.100 |
| `PromptFlashAttention_6` | 3 | 191.860 |
| `PromptFlashAttention_7` | 3 | 191.020 |
| `PromptFlashAttention` | 3 | 190.380 |
| `PromptFlashAttention_13` | 3 | 189.500 |
| `PromptFlashAttention_23` | 3 | 188.320 |
| `PromptFlashAttention_1` | 3 | 188.280 |
| `PromptFlashAttention_24` | 3 | 187.140 |
| `PromptFlashAttention_12` | 3 | 186.460 |
| `PromptFlashAttention_14` | 3 | 183.600 |
| `PromptFlashAttention_3` | 3 | 181.860 |
| `PromptFlashAttention_22` | 3 | 180.600 |
| `PromptFlashAttention_18` | 3 | 179.180 |
| `PromptFlashAttention_20` | 3 | 178.960 |
| `PromptFlashAttention_2` | 3 | 176.080 |
| `PromptFlashAttention_4` | 3 | 175.440 |
| `PromptFlashAttention_19` | 3 | 173.960 |
| `PromptFlashAttention_11` | 3 | 172.960 |
| `PromptFlashAttention_21` | 3 | 172.740 |
| `MatMulV2_58` | 3 | 109.280 |
| `MatMulV2_70` | 3 | 109.260 |
| `MatMulV2_52` | 3 | 109.200 |
| `MatMulV2_4` | 3 | 109.160 |
| `MatMulV2_148` | 3 | 108.880 |
| `MatMulV2_22` | 3 | 108.860 |
| `MatMulV2_28` | 3 | 108.480 |
| `MatMulV2_160` | 3 | 108.400 |
| `MatMulV2_142` | 3 | 108.220 |
| `MatMulV2_34` | 3 | 108.000 |
| `MatMulV2_118` | 3 | 107.920 |
| `MatMulV2_16` | 3 | 107.800 |
| `MatMulV2_40` | 3 | 107.660 |
| `MatMulV2_112` | 3 | 107.580 |
| `MatMulV2_82` | 3 | 107.500 |
| `MatMulV2_100` | 3 | 107.400 |
| `MatMulV2_154` | 3 | 107.360 |
| `MatMulV2_106` | 3 | 107.300 |
| `MatMulV2_124` | 3 | 107.160 |
| `MatMulV2_136` | 3 | 107.060 |
| `MatMulV2_46` | 3 | 107.040 |
| `MatMulV2_10` | 3 | 106.980 |
| `MatMulV2_64` | 3 | 106.740 |
| `MatMulV2_76` | 3 | 106.540 |
| `MatMulV2_94` | 3 | 106.040 |
| `MatMulV2_88` | 3 | 105.820 |
| `MatMulV2_130` | 3 | 105.400 |
| `MatMulV2_5` | 3 | 94.900 |
| `MatMulV2_59` | 3 | 94.640 |
| `MatMulV2_41` | 3 | 94.600 |
| `MatMulV2_23` | 3 | 94.520 |
| `MatMulV2_53` | 3 | 94.380 |
| `MatMulV2_65` | 3 | 94.200 |
| `MatMulV2_35` | 3 | 94.000 |
| `MatMulV2_47` | 3 | 93.720 |
| `MatMulV2_71` | 3 | 93.360 |
| `MatMulV2_29` | 3 | 93.220 |
| `MatMulV2_155` | 3 | 93.200 |
| `MatMulV2_11` | 3 | 93.100 |
| `MatMulV2_17` | 3 | 93.100 |
| `MatMulV2_101` | 3 | 92.640 |
| `MatMulV2_131` | 3 | 92.440 |
| `MatMulV2_143` | 3 | 92.360 |
| `MatMulV2_137` | 3 | 92.300 |
| `MatMulV2_107` | 3 | 92.180 |
| `MatMulV2_161` | 3 | 92.140 |
| `MatMulV2_77` | 3 | 91.960 |
| `MatMulV2_83` | 3 | 91.960 |
| `MatMulV2_149` | 3 | 91.940 |
| `MatMulV2_95` | 3 | 91.920 |
| `MatMulV2_113` | 3 | 91.580 |
| `MatMulV2_119` | 3 | 91.400 |
| `MatMulV2_89` | 3 | 91.340 |
| `MatMulV2_125` | 3 | 91.040 |
| `Transpose_104` | 3 | 74.700 |
| `Transpose_44` | 3 | 69.360 |
| `Transpose_55` | 3 | 69.220 |
| `Transpose_43` | 3 | 68.460 |
| `Transpose_103` | 3 | 68.440 |
| `StridedSliceV2_44` | 3 | 64.480 |
| `StridedSliceV2_29` | 3 | 63.860 |
| `StridedSliceV2_114` | 3 | 63.440 |
| `StridedSliceV2_59` | 3 | 63.400 |
| `StridedSliceV2_89` | 3 | 63.300 |
| `StridedSliceV2_99` | 3 | 63.260 |
| `StridedSliceV2_34` | 3 | 62.980 |
| `StridedSliceV2_69` | 3 | 62.920 |
| `StridedSliceV2_39` | 3 | 62.880 |
| `StridedSliceV2_49` | 3 | 62.880 |
| `StridedSliceV2_119` | 3 | 62.780 |
| `StridedSliceV2_14` | 3 | 62.720 |
| `StridedSliceV2_129` | 3 | 62.720 |
| `StridedSliceV2_4` | 3 | 62.540 |
| `StridedSliceV2_94` | 3 | 62.520 |
| `StridedSliceV2_19` | 3 | 62.480 |
| `StridedSliceV2_109` | 3 | 62.480 |
| `StridedSliceV2_24` | 3 | 62.440 |
| `StridedSliceV2_74` | 3 | 62.420 |
| `StridedSliceV2_104` | 3 | 62.380 |
| `StridedSliceV2_9` | 3 | 62.360 |
| `StridedSliceV2_79` | 3 | 62.240 |
| `StridedSliceV2_134` | 3 | 62.240 |
| `StridedSliceV2_54` | 3 | 62.200 |
| `StridedSliceV2_64` | 3 | 62.160 |
| `StridedSliceV2_124` | 3 | 62.000 |
| `StridedSliceV2_84` | 3 | 61.660 |
| `Transpose_156` | 3 | 59.000 |
| `Gelu_12` | 3 | 58.560 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 58.420 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 58.420 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 58.400 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 58.400 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 58.360 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 58.340 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 58.320 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 58.300 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 58.280 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 58.280 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 58.260 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 58.240 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 58.240 |
| `Transpose_26` | 3 | 58.220 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 58.220 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 58.200 |
| `Gelu_18` | 3 | 58.180 |
| `Transpose_216` | 3 | 58.180 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 58.160 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 58.160 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 58.140 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 58.140 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 58.120 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 58.080 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 58.080 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 58.080 |
| `Transpose_76` | 3 | 58.020 |
| `Transpose_176` | 3 | 58.020 |
| `Transpose_236` | 3 | 58.020 |
| `Transpose_56` | 3 | 58.000 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 57.920 |
| `Transpose_16` | 3 | 57.880 |
| `Transpose_196` | 3 | 57.760 |
| `Transpose_266` | 3 | 57.760 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 57.720 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 57.720 |
| `Transpose_36` | 3 | 57.700 |
| `Transpose_256` | 3 | 57.660 |
| `Transpose_226` | 3 | 57.600 |
| `Transpose_86` | 3 | 57.540 |
| `Transpose_6` | 3 | 57.540 |
| `Transpose_136` | 3 | 57.480 |
| `Transpose_66` | 3 | 57.420 |
| `Transpose_46` | 3 | 57.380 |
| `Transpose_146` | 3 | 57.300 |
| `Transpose_186` | 3 | 57.300 |
| `Transpose_96` | 3 | 57.280 |
| `Transpose_206` | 3 | 57.260 |
| `Transpose_166` | 3 | 57.200 |
| `Transpose_116` | 3 | 57.160 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 56.900 |
| `Transpose_246` | 3 | 56.880 |
| `Transpose_106` | 3 | 56.480 |
| `Transpose_145` | 3 | 56.120 |
| `Transpose_185` | 3 | 56.100 |
| `Transpose_265` | 3 | 55.980 |
| `Transpose_105` | 3 | 55.960 |
| `Transpose_126` | 3 | 55.920 |
| `Transpose_25` | 3 | 55.860 |
| `Transpose_95` | 3 | 55.860 |
| `Transpose_165` | 3 | 55.840 |
| `Transpose_15` | 3 | 55.820 |
| `Transpose_205` | 3 | 55.820 |
| `Transpose_125` | 3 | 55.820 |
| `Transpose_45` | 3 | 55.800 |
| `Transpose_195` | 3 | 55.720 |
| `Transpose_3` | 3 | 55.660 |
| `Transpose_175` | 3 | 55.660 |
| `Transpose_65` | 3 | 55.620 |
| `Transpose_135` | 3 | 55.480 |
| `Transpose_215` | 3 | 55.460 |
| `Transpose_115` | 3 | 55.440 |
| `Transpose_245` | 3 | 55.360 |
| `Transpose_35` | 3 | 55.340 |
| `Transpose_255` | 3 | 55.300 |
| `Transpose_235` | 3 | 55.200 |
| `Transpose_204` | 3 | 55.180 |
| `Transpose_24` | 3 | 55.160 |
| `Transpose_124` | 3 | 55.160 |
| `Transpose_225` | 3 | 55.140 |
| `Transpose_264` | 3 | 55.060 |
| `Transpose_244` | 3 | 55.060 |
| `Transpose_64` | 3 | 55.020 |
| `Transpose_75` | 3 | 55.020 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | ND;ND;ND;ND -> ND` | 81 | 5171.520 |
| `StridedSliceD | "1,512,16,72" -> "1,512,16,36" | ND -> ND` | 324 | 4698.360 |
| `MatMulV2 | "512,1152;1152,1152;1152" -> "512,1152" | ND;ND;ND -> ND` | 324 | 4562.640 |
| `Transpose | "512,16,72;3" -> "16,512,72" | ND;ND -> ND` | 243 | 4487.580 |
| `PadV3 | "1,16,512,72;8;" -> "1,16,512,80" | ND;ND;ND -> ND` | 243 | 2952.940 |
| `MatMulV2 | "512,1152;4352,1152;4352" -> "512,4352" | ND;ND;ND -> ND` | 81 | 2907.040 |
| `MatMulV2 | "512,4352;1152,4352;1152" -> "512,1152" | ND;ND;ND -> ND` | 81 | 2508.140 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2285.320 |
| `Mul | "1,512,16,72;1,512,1,72" -> "1,512,16,72" | ND;ND -> ND` | 324 | 1711.380 |
| `StridedSliceD | "1,16,512,80" -> "1,16,512,72" | ND -> ND` | 81 | 1693.740 |
| `Transpose | "16,512,72;3" -> "512,16,72" | ND;ND -> ND` | 81 | 1553.960 |
| `ConcatV2D | "1,512,16,36;1,512,16,36" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1322.920 |
| `Add | "1,512,16,72;1,512,16,72" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1273.220 |
| `Neg | "1,512,16,36" -> "1,512,16,36" | ND -> ND` | 162 | 1272.780 |
| `Gelu | "1,512,4352" -> "1,512,4352" | ND -> ND` | 81 | 1226.820 |
| `Cast | "1,512,16,72" -> "1,512,16,72" | ND -> ND` | 162 | 1208.380 |
| `ConcatV2D | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,3456" | ND;ND;ND -> ND` | 81 | 569.900 |
| `SplitVD | "1,512,3456" -> "1,512,1152;1,512,1152;1,512,1152" | ND -> ND;ND;ND` | 81 | 496.500 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 233.680 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 44.740 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.760 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 813 | 13545.400 |
| `ND` | 891 | 10596.580 |
| `ND;ND` | 972 | 10349.060 |
| `ND;ND;ND;ND` | 243 | 7456.840 |
| `N/A` | 246 | 249.440 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_9` | 0 | 78.160 |
| `PromptFlashAttention_9` | 0 | 77.800 |
| `PromptFlashAttention_9` | 0 | 77.140 |
| `PromptFlashAttention_16` | 0 | 74.900 |
| `PromptFlashAttention_16` | 0 | 73.560 |
| `PromptFlashAttention_16` | 0 | 73.340 |
| `PromptFlashAttention_17` | 0 | 73.020 |
| `PromptFlashAttention_10` | 0 | 72.880 |
| `PromptFlashAttention_10` | 0 | 71.880 |
| `PromptFlashAttention_10` | 0 | 71.460 |
| `PromptFlashAttention_17` | 0 | 70.860 |
| `PromptFlashAttention_17` | 0 | 70.380 |
| `PromptFlashAttention_15` | 0 | 69.220 |
| `PromptFlashAttention_8` | 0 | 67.800 |
| `PromptFlashAttention_8` | 0 | 67.760 |
| `PromptFlashAttention_8` | 0 | 67.740 |
| `PromptFlashAttention_15` | 0 | 67.560 |
| `PromptFlashAttention_15` | 0 | 67.160 |
| `PromptFlashAttention_5` | 0 | 66.540 |
| `PromptFlashAttention_26` | 0 | 66.240 |
| `PromptFlashAttention_5` | 0 | 66.180 |
| `PromptFlashAttention_26` | 0 | 66.080 |
| `PromptFlashAttention_26` | 0 | 66.020 |
| `PromptFlashAttention_25` | 0 | 65.760 |
| `PromptFlashAttention_25` | 0 | 65.580 |
| `PromptFlashAttention_25` | 0 | 64.760 |
| `PromptFlashAttention_6` | 0 | 64.400 |
| `PromptFlashAttention_7` | 0 | 64.320 |
| `PromptFlashAttention_6` | 0 | 63.960 |
| `PromptFlashAttention` | 0 | 63.940 |
| `PromptFlashAttention` | 0 | 63.640 |
| `PromptFlashAttention_13` | 0 | 63.580 |
| `PromptFlashAttention_6` | 0 | 63.500 |
| `PromptFlashAttention_7` | 0 | 63.500 |
| `PromptFlashAttention_13` | 0 | 63.400 |
| `PromptFlashAttention_5` | 0 | 63.400 |
| `PromptFlashAttention_23` | 0 | 63.220 |
| `PromptFlashAttention_7` | 0 | 63.200 |
| `PromptFlashAttention_24` | 0 | 63.180 |
| `PromptFlashAttention_1` | 0 | 63.080 |
| `PromptFlashAttention_1` | 0 | 63.000 |
| `PromptFlashAttention` | 0 | 62.800 |
| `PromptFlashAttention_23` | 0 | 62.720 |
| `PromptFlashAttention_12` | 0 | 62.580 |
| `PromptFlashAttention_13` | 0 | 62.520 |
| `PromptFlashAttention_23` | 0 | 62.380 |
| `PromptFlashAttention_1` | 0 | 62.200 |
| `PromptFlashAttention_24` | 0 | 62.200 |
| `PromptFlashAttention_12` | 0 | 62.140 |
| `PromptFlashAttention_24` | 0 | 61.760 |
| `PromptFlashAttention_12` | 0 | 61.740 |
| `PromptFlashAttention_14` | 0 | 61.620 |
| `PromptFlashAttention_14` | 0 | 61.260 |
| `PromptFlashAttention_3` | 0 | 60.940 |
| `PromptFlashAttention_14` | 0 | 60.720 |
| `PromptFlashAttention_18` | 0 | 60.660 |
| `PromptFlashAttention_20` | 0 | 60.620 |
| `PromptFlashAttention_3` | 0 | 60.460 |
| `PromptFlashAttention_3` | 0 | 60.460 |
| `PromptFlashAttention_22` | 0 | 60.260 |
| `PromptFlashAttention_22` | 0 | 60.220 |
| `PromptFlashAttention_22` | 0 | 60.120 |
| `PromptFlashAttention_20` | 0 | 59.560 |
| `PromptFlashAttention_18` | 0 | 59.480 |
| `PromptFlashAttention_18` | 0 | 59.040 |
| `PromptFlashAttention_4` | 0 | 58.960 |
| `PromptFlashAttention_19` | 0 | 58.860 |
| `PromptFlashAttention_2` | 0 | 58.780 |
| `PromptFlashAttention_20` | 0 | 58.780 |
| `PromptFlashAttention_2` | 0 | 58.660 |
| `PromptFlashAttention_2` | 0 | 58.640 |
| `PromptFlashAttention_21` | 0 | 58.420 |
| `PromptFlashAttention_4` | 0 | 58.420 |
| `PromptFlashAttention_11` | 0 | 58.180 |
| `PromptFlashAttention_4` | 0 | 58.060 |
| `PromptFlashAttention_21` | 0 | 58.000 |
| `PromptFlashAttention_19` | 0 | 57.700 |
| `PromptFlashAttention_11` | 0 | 57.540 |
| `PromptFlashAttention_19` | 0 | 57.400 |
| `PromptFlashAttention_11` | 0 | 57.240 |
| `PromptFlashAttention_21` | 0 | 56.320 |
| `MatMulV2_58` | 0 | 38.080 |
| `MatMulV2_4` | 0 | 37.480 |
| `MatMulV2_70` | 0 | 37.400 |
| `MatMulV2_28` | 0 | 37.100 |
| `MatMulV2_22` | 0 | 37.080 |
| `MatMulV2_40` | 0 | 36.600 |
| `MatMulV2_148` | 0 | 36.520 |
| `MatMulV2_64` | 0 | 36.500 |
| `MatMulV2_52` | 0 | 36.500 |
| `MatMulV2_10` | 0 | 36.440 |
| `MatMulV2_34` | 0 | 36.440 |
| `MatMulV2_52` | 0 | 36.420 |
| `MatMulV2_154` | 0 | 36.420 |
| `MatMulV2_160` | 0 | 36.400 |
| `MatMulV2_118` | 0 | 36.320 |
| `MatMulV2_160` | 0 | 36.320 |
| `MatMulV2_142` | 0 | 36.300 |
| `MatMulV2_52` | 0 | 36.280 |
| `MatMulV2_148` | 0 | 36.240 |
| `MatMulV2_82` | 0 | 36.160 |
| `MatMulV2_142` | 0 | 36.160 |
| `MatMulV2_118` | 0 | 36.160 |
| `MatMulV2_16` | 0 | 36.140 |
| `MatMulV2_148` | 0 | 36.120 |
| `MatMulV2_112` | 0 | 36.120 |
| `MatMulV2_136` | 0 | 36.120 |
| `MatMulV2_100` | 0 | 36.100 |
| `MatMulV2_70` | 0 | 36.080 |
| `MatMulV2_22` | 0 | 36.080 |
| `MatMulV2_124` | 0 | 36.060 |
| `MatMulV2_46` | 0 | 35.980 |
| `MatMulV2_4` | 0 | 35.980 |
| `MatMulV2_34` | 0 | 35.940 |
| `MatMulV2_40` | 0 | 35.920 |
| `MatMulV2_100` | 0 | 35.920 |
| `MatMulV2_106` | 0 | 35.900 |
| `MatMulV2_16` | 0 | 35.880 |
| `MatMulV2_106` | 0 | 35.880 |
| `MatMulV2_58` | 0 | 35.860 |
| `MatMulV2_46` | 0 | 35.840 |
| `MatMulV2_82` | 0 | 35.800 |
| `MatMulV2_112` | 0 | 35.780 |
| `MatMulV2_16` | 0 | 35.780 |
| `MatMulV2_70` | 0 | 35.780 |
| `MatMulV2_142` | 0 | 35.760 |
| `MatMulV2_4` | 0 | 35.700 |
| `MatMulV2_22` | 0 | 35.700 |
| `MatMulV2_28` | 0 | 35.700 |
| `MatMulV2_112` | 0 | 35.680 |
| `MatMulV2_28` | 0 | 35.680 |
| `MatMulV2_160` | 0 | 35.680 |
| `MatMulV2_76` | 0 | 35.620 |
| `MatMulV2_34` | 0 | 35.620 |
| `MatMulV2_76` | 0 | 35.600 |
| `MatMulV2_124` | 0 | 35.560 |
| `MatMulV2_82` | 0 | 35.540 |
| `MatMulV2_124` | 0 | 35.540 |
| `MatMulV2_136` | 0 | 35.540 |
| `MatMulV2_106` | 0 | 35.520 |
| `MatMulV2_154` | 0 | 35.520 |
| `MatMulV2_10` | 0 | 35.500 |
| `MatMulV2_88` | 0 | 35.500 |
| `MatMulV2_94` | 0 | 35.480 |
| `MatMulV2_94` | 0 | 35.460 |
| `MatMulV2_118` | 0 | 35.440 |
| `MatMulV2_154` | 0 | 35.420 |
| `MatMulV2_136` | 0 | 35.400 |
| `MatMulV2_130` | 0 | 35.400 |
| `MatMulV2_100` | 0 | 35.380 |
| `MatMulV2_58` | 0 | 35.340 |
| `MatMulV2_76` | 0 | 35.320 |
| `MatMulV2_88` | 0 | 35.240 |
| `MatMulV2_64` | 0 | 35.240 |
| `MatMulV2_46` | 0 | 35.220 |
| `MatMulV2_40` | 0 | 35.140 |
| `MatMulV2_94` | 0 | 35.100 |
| `MatMulV2_88` | 0 | 35.080 |
| `MatMulV2_10` | 0 | 35.040 |
| `MatMulV2_130` | 0 | 35.040 |
| `MatMulV2_64` | 0 | 35.000 |
| `MatMulV2_130` | 0 | 34.960 |
| `MatMulV2_65` | 0 | 33.120 |
| `MatMulV2_41` | 0 | 32.980 |
| `MatMulV2_5` | 0 | 32.820 |
| `MatMulV2_59` | 0 | 32.660 |
| `MatMulV2_23` | 0 | 32.560 |
| `MatMulV2_53` | 0 | 32.360 |
| `MatMulV2_11` | 0 | 32.220 |
| `MatMulV2_17` | 0 | 32.140 |
| `MatMulV2_47` | 0 | 31.860 |
| `MatMulV2_29` | 0 | 31.800 |
| `MatMulV2_71` | 0 | 31.800 |
| `MatMulV2_35` | 0 | 31.660 |
| `MatMulV2_23` | 0 | 31.420 |
| `MatMulV2_107` | 0 | 31.340 |
| `MatMulV2_155` | 0 | 31.340 |
| `MatMulV2_47` | 0 | 31.240 |
| `MatMulV2_35` | 0 | 31.220 |
| `MatMulV2_53` | 0 | 31.220 |
| `MatMulV2_101` | 0 | 31.200 |
| `MatMulV2_143` | 0 | 31.180 |
| `MatMulV2_155` | 0 | 31.120 |
| `MatMulV2_5` | 0 | 31.120 |
| `MatMulV2_35` | 0 | 31.120 |
| `MatMulV2_83` | 0 | 31.100 |
| `MatMulV2_59` | 0 | 31.040 |
| `MatMulV2_41` | 0 | 30.980 |
| `MatMulV2_5` | 0 | 30.960 |
| `MatMulV2_59` | 0 | 30.940 |
| `MatMulV2_131` | 0 | 30.940 |
| `MatMulV2_71` | 0 | 30.940 |
| `MatMulV2_29` | 0 | 30.880 |
| `MatMulV2_137` | 0 | 30.860 |
| `MatMulV2_137` | 0 | 30.840 |
| `MatMulV2_77` | 0 | 30.840 |
| `MatMulV2_77` | 0 | 30.800 |
| `MatMulV2_53` | 0 | 30.800 |
| `MatMulV2_131` | 0 | 30.800 |
| `MatMulV2_95` | 0 | 30.780 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 17019.570 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step1` | 1 | 15825.840 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step3` | 1 | 15489.080 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.native.torchair.active.step2` | 1 | 15422.830 |
| `TorchDynamo Cache Lookup` | 3 | 14186.650 |
| `Torch-Compiled Region: 0/0` | 3 | 3657.260 |
| `TorchNpuGraphBase::Run` | 3 | 2685.840 |
| `RefreshAtTensorFromGeTensor` | 3 | 1139.820 |
| `aten::empty` | 3 | 552.980 |
| `ExecuteGraph` | 3 | 489.170 |
| `AssembleInputs` | 3 | 390.910 |
| `AssembleOutputs` | 3 | 286.310 |
| `aten::set_` | 3 | 279.490 |
| `empty_tensor` | 3 | 273.390 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 221529.160 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 40979.590 |
| `launch` | 976 | 18642.730 |
| `InputCopy` | 3 | 149.490 |
| `ModelExecute` | 3 | 51.990 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 26.470 |
| `step_info` | 6 | 15.300 |
| `OutputCopy` | 3 | 0.720 |

