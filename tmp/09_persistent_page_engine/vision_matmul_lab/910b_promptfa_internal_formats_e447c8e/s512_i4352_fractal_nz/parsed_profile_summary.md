# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4352_fractal_nz`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/910b_promptfa_internal_formats_e447c8e/s512_i4352_fractal_nz/liteserver-c001-4_631790_20260729134702658_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `40635.980 us`
- `Free`: `3471.420 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3340.250 us`
- `Stage`: `44107.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 8526.400 |
| `StridedSliceD` | 405 | 6163.300 |
| `Transpose` | 324 | 6096.020 |
| `PromptFlashAttention` | 81 | 5108.800 |
| `PadV3` | 243 | 3073.880 |
| `AddLayerNorm` | 162 | 2238.400 |
| `ConcatV2D` | 243 | 1917.220 |
| `Mul` | 324 | 1680.980 |
| `Neg` | 162 | 1333.140 |
| `Add` | 162 | 1242.940 |
| `Cast` | 162 | 1238.300 |
| `Gelu` | 81 | 1214.840 |
| `SplitVD` | 81 | 505.540 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 234.280 |
| `LayerNormV3` | 3 | 46.160 |
| `Data` | 3 | 15.780 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 243 | 234.280 |
| `PromptFlashAttention_13` | 3 | 212.440 |
| `PromptFlashAttention_14` | 3 | 205.620 |
| `PromptFlashAttention_9` | 3 | 202.460 |
| `PromptFlashAttention` | 3 | 201.960 |
| `PromptFlashAttention_24` | 3 | 201.940 |
| `PromptFlashAttention_17` | 3 | 200.520 |
| `PromptFlashAttention_16` | 3 | 199.060 |
| `PromptFlashAttention_7` | 3 | 193.900 |
| `PromptFlashAttention_15` | 3 | 193.800 |
| `PromptFlashAttention_12` | 3 | 193.580 |
| `PromptFlashAttention_8` | 3 | 193.240 |
| `PromptFlashAttention_25` | 3 | 192.000 |
| `PromptFlashAttention_20` | 3 | 190.000 |
| `PromptFlashAttention_1` | 3 | 187.980 |
| `PromptFlashAttention_26` | 3 | 186.580 |
| `PromptFlashAttention_6` | 3 | 186.260 |
| `PromptFlashAttention_5` | 3 | 182.700 |
| `PromptFlashAttention_4` | 3 | 182.100 |
| `PromptFlashAttention_19` | 3 | 181.980 |
| `PromptFlashAttention_18` | 3 | 180.960 |
| `PromptFlashAttention_10` | 3 | 179.780 |
| `PromptFlashAttention_11` | 3 | 179.620 |
| `PromptFlashAttention_22` | 3 | 179.300 |
| `PromptFlashAttention_21` | 3 | 177.800 |
| `PromptFlashAttention_23` | 3 | 174.960 |
| `PromptFlashAttention_2` | 3 | 174.620 |
| `PromptFlashAttention_3` | 3 | 173.640 |
| `MatMulV2_5` | 3 | 91.060 |
| `MatMulV2_47` | 3 | 88.820 |
| `MatMulV2_41` | 3 | 86.740 |
| `MatMulV2_53` | 3 | 85.200 |
| `MatMulV2_89` | 3 | 85.060 |
| `MatMulV2_101` | 3 | 84.920 |
| `MatMulV2_95` | 3 | 84.600 |
| `MatMulV2_71` | 3 | 84.340 |
| `MatMulV2_29` | 3 | 84.300 |
| `MatMulV2_143` | 3 | 84.300 |
| `MatMulV2_131` | 3 | 84.280 |
| `MatMulV2_17` | 3 | 84.240 |
| `MatMulV2_155` | 3 | 84.220 |
| `MatMulV2_161` | 3 | 84.200 |
| `MatMulV2_4` | 3 | 84.120 |
| `MatMulV2_11` | 3 | 84.020 |
| `MatMulV2_65` | 3 | 83.940 |
| `MatMulV2_125` | 3 | 83.880 |
| `MatMulV2_113` | 3 | 83.320 |
| `MatMulV2_119` | 3 | 82.820 |
| `MatMulV2_136` | 3 | 82.460 |
| `MatMulV2_100` | 3 | 82.440 |
| `MatMulV2_106` | 3 | 82.420 |
| `MatMulV2_77` | 3 | 82.320 |
| `MatMulV2_35` | 3 | 82.160 |
| `MatMulV2_149` | 3 | 82.160 |
| `MatMulV2_76` | 3 | 82.020 |
| `MatMulV2_82` | 3 | 81.940 |
| `MatMulV2_94` | 3 | 81.760 |
| `MatMulV2_112` | 3 | 81.680 |
| `MatMulV2_107` | 3 | 81.600 |
| `MatMulV2_64` | 3 | 81.320 |
| `MatMulV2_142` | 3 | 81.320 |
| `MatMulV2_10` | 3 | 81.260 |
| `MatMulV2_130` | 3 | 81.060 |
| `MatMulV2_137` | 3 | 80.960 |
| `MatMulV2_59` | 3 | 80.880 |
| `MatMulV2_83` | 3 | 80.780 |
| `MatMulV2_58` | 3 | 80.740 |
| `MatMulV2_22` | 3 | 80.620 |
| `MatMulV2_40` | 3 | 80.140 |
| `MatMulV2_23` | 3 | 80.100 |
| `MatMulV2_148` | 3 | 80.080 |
| `MatMulV2_124` | 3 | 79.880 |
| `MatMulV2_160` | 3 | 79.840 |
| `MatMulV2_34` | 3 | 79.820 |
| `MatMulV2_88` | 3 | 79.320 |
| `MatMulV2_46` | 3 | 79.320 |
| `MatMulV2_16` | 3 | 79.260 |
| `MatMulV2_118` | 3 | 78.380 |
| `MatMulV2_70` | 3 | 78.020 |
| `MatMulV2_52` | 3 | 77.760 |
| `MatMulV2_28` | 3 | 76.920 |
| `MatMulV2_154` | 3 | 76.620 |
| `Transpose_104` | 3 | 76.240 |
| `Transpose_55` | 3 | 71.000 |
| `Transpose_43` | 3 | 70.980 |
| `Transpose_103` | 3 | 70.760 |
| `Transpose_44` | 3 | 70.440 |
| `MatMulV2_3` | 3 | 67.100 |
| `MatMulV2` | 3 | 62.140 |
| `StridedSliceV2_39` | 3 | 62.060 |
| `StridedSliceV2_74` | 3 | 62.040 |
| `StridedSliceV2_59` | 3 | 62.000 |
| `StridedSliceV2_99` | 3 | 61.880 |
| `StridedSliceV2_69` | 3 | 61.820 |
| `MatMulV2_123` | 3 | 61.740 |
| `StridedSliceV2_29` | 3 | 61.720 |
| `StridedSliceV2_14` | 3 | 61.660 |
| `StridedSliceV2_4` | 3 | 61.580 |
| `StridedSliceV2_89` | 3 | 61.540 |
| `StridedSliceV2_109` | 3 | 61.460 |
| `StridedSliceV2_9` | 3 | 61.420 |
| `StridedSliceV2_54` | 3 | 61.380 |
| `StridedSliceV2_49` | 3 | 61.340 |
| `StridedSliceV2_79` | 3 | 61.260 |
| `StridedSliceV2_94` | 3 | 61.220 |
| `StridedSliceV2_19` | 3 | 61.160 |
| `StridedSliceV2_129` | 3 | 61.100 |
| `StridedSliceV2_134` | 3 | 61.080 |
| `StridedSliceV2_24` | 3 | 61.080 |
| `StridedSliceV2_64` | 3 | 61.000 |
| `StridedSliceV2_44` | 3 | 60.840 |
| `StridedSliceV2_104` | 3 | 60.780 |
| `StridedSliceV2_114` | 3 | 60.740 |
| `StridedSliceV2_119` | 3 | 60.640 |
| `StridedSliceV2_124` | 3 | 60.640 |
| `StridedSliceV2_34` | 3 | 60.540 |
| `StridedSliceV2_84` | 3 | 59.980 |
| `MatMulV2_105` | 3 | 59.580 |
| `MatMulV2_63` | 3 | 59.520 |
| `MatMulV2_117` | 3 | 58.580 |
| `Gelu_18` | 3 | 58.320 |
| `Gelu_12` | 3 | 58.220 |
| `MatMulV2_75` | 3 | 57.820 |
| `Transpose_175` | 3 | 57.140 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 57.040 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 57.020 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 56.980 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 56.940 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 56.920 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 56.920 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 56.900 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 56.900 |
| `Transpose_226` | 3 | 56.900 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 56.900 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 56.900 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 56.880 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 56.880 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 56.880 |
| `Transpose_15` | 3 | 56.860 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 56.860 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 56.840 |
| `Transpose_6` | 3 | 56.840 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 56.820 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 56.780 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 56.780 |
| `Transpose_95` | 3 | 56.760 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 56.740 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 56.740 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 56.720 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 56.680 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 56.660 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 56.640 |
| `Transpose_105` | 3 | 56.640 |
| `Transpose_135` | 3 | 56.640 |
| `Transpose_146` | 3 | 56.640 |
| `Transpose_266` | 3 | 56.640 |
| `Transpose_166` | 3 | 56.620 |
| `Transpose_195` | 3 | 56.620 |
| `Transpose_25` | 3 | 56.620 |
| `Transpose_26` | 3 | 56.600 |
| `Transpose_106` | 3 | 56.600 |
| `Transpose_176` | 3 | 56.580 |
| `Transpose_246` | 3 | 56.580 |
| `MatMulV2_9` | 3 | 56.560 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 56.560 |
| `Transpose_75` | 3 | 56.540 |
| `Transpose_155` | 3 | 56.540 |
| `Transpose_255` | 3 | 56.520 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 56.480 |
| `Transpose_125` | 3 | 56.480 |
| `Transpose_174` | 3 | 56.480 |
| `Transpose_86` | 3 | 56.480 |
| `Transpose_46` | 3 | 56.460 |
| `Transpose_265` | 3 | 56.420 |
| `Transpose_115` | 3 | 56.400 |
| `Transpose_156` | 3 | 56.400 |
| `Transpose_145` | 3 | 56.380 |
| `Transpose_56` | 3 | 56.380 |
| `Transpose_3` | 3 | 56.360 |
| `Transpose_66` | 3 | 56.360 |
| `Transpose_34` | 3 | 56.340 |
| `Transpose_214` | 3 | 56.340 |
| `Transpose_35` | 3 | 56.320 |
| `Transpose_154` | 3 | 56.320 |
| `Transpose_74` | 3 | 56.300 |
| `Transpose_254` | 3 | 56.300 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 56.280 |
| `Transpose_186` | 3 | 56.280 |
| `Transpose_206` | 3 | 56.260 |
| `Transpose_194` | 3 | 56.240 |
| `Transpose_114` | 3 | 56.240 |
| `Transpose_215` | 3 | 56.200 |
| `Transpose_85` | 3 | 56.180 |
| `Transpose_236` | 3 | 56.160 |
| `Transpose_36` | 3 | 56.160 |
| `Transpose_94` | 3 | 56.160 |
| `Transpose_126` | 3 | 56.160 |
| `Transpose_234` | 3 | 56.140 |
| `Transpose_245` | 3 | 56.100 |
| `Transpose_185` | 3 | 56.080 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | ND;ND;ND;ND -> ND` | 81 | 5108.800 |
| `Transpose | "512,16,72;3" -> "16,512,72" | ND;ND -> ND` | 243 | 4578.080 |
| `StridedSliceD | "1,512,16,72" -> "1,512,16,36" | ND -> ND` | 324 | 4509.340 |
| `MatMulV2 | "512,1152;72,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 324 | 4090.660 |
| `PadV3 | "1,16,512,72;8;" -> "1,16,512,80" | ND;ND;ND -> ND` | 243 | 3073.880 |
| `MatMulV2 | "512,4352;272,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2265.220 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2238.400 |
| `MatMulV2 | "512,1152;72,272,16,16;4352" -> "512,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2170.520 |
| `Mul | "1,512,16,72;1,512,1,72" -> "1,512,16,72" | ND;ND -> ND` | 324 | 1680.980 |
| `StridedSliceD | "1,16,512,80" -> "1,16,512,72" | ND -> ND` | 81 | 1653.960 |
| `Transpose | "16,512,72;3" -> "512,16,72" | ND;ND -> ND` | 81 | 1517.940 |
| `ConcatV2D | "1,512,16,36;1,512,16,36" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1358.220 |
| `Neg | "1,512,16,36" -> "1,512,16,36" | ND -> ND` | 162 | 1333.140 |
| `Add | "1,512,16,72;1,512,16,72" -> "1,512,16,72" | ND;ND -> ND` | 162 | 1242.940 |
| `Cast | "1,512,16,72" -> "1,512,16,72" | ND -> ND` | 162 | 1238.300 |
| `Gelu | "1,512,4352" -> "1,512,4352" | ND -> ND` | 81 | 1214.840 |
| `ConcatV2D | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,3456" | ND;ND;ND -> ND` | 81 | 559.000 |
| `SplitVD | "1,512,3456" -> "1,512,1152;1,512,1152;1,512,1152" | ND -> ND;ND;ND` | 81 | 505.540 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 243 | 234.280 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 46.160 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 15.780 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND` | 891 | 10455.120 |
| `ND;ND` | 972 | 10378.160 |
| `ND;FRACTAL_NZ;ND` | 486 | 8526.400 |
| `ND;ND;ND;ND` | 243 | 7347.200 |
| `ND;ND;ND` | 327 | 3679.040 |
| `N/A` | 246 | 250.060 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_13` | 0 | 72.260 |
| `PromptFlashAttention_13` | 0 | 70.100 |
| `PromptFlashAttention_13` | 0 | 70.080 |
| `PromptFlashAttention_14` | 0 | 69.140 |
| `PromptFlashAttention_14` | 0 | 68.900 |
| `PromptFlashAttention_24` | 0 | 68.300 |
| `PromptFlashAttention_9` | 0 | 67.900 |
| `PromptFlashAttention` | 0 | 67.800 |
| `PromptFlashAttention_16` | 0 | 67.740 |
| `PromptFlashAttention_9` | 0 | 67.640 |
| `PromptFlashAttention` | 0 | 67.580 |
| `PromptFlashAttention_14` | 0 | 67.580 |
| `PromptFlashAttention_24` | 0 | 67.000 |
| `PromptFlashAttention_17` | 0 | 66.980 |
| `PromptFlashAttention_17` | 0 | 66.960 |
| `PromptFlashAttention_9` | 0 | 66.920 |
| `PromptFlashAttention_24` | 0 | 66.640 |
| `PromptFlashAttention` | 0 | 66.580 |
| `PromptFlashAttention_17` | 0 | 66.580 |
| `PromptFlashAttention_16` | 0 | 66.480 |
| `PromptFlashAttention_7` | 0 | 66.380 |
| `PromptFlashAttention_15` | 0 | 66.280 |
| `PromptFlashAttention_8` | 0 | 65.140 |
| `PromptFlashAttention_25` | 0 | 64.960 |
| `PromptFlashAttention_20` | 0 | 64.900 |
| `PromptFlashAttention_16` | 0 | 64.840 |
| `PromptFlashAttention_12` | 0 | 64.800 |
| `PromptFlashAttention_12` | 0 | 64.620 |
| `PromptFlashAttention_1` | 0 | 64.260 |
| `PromptFlashAttention_7` | 0 | 64.200 |
| `PromptFlashAttention_12` | 0 | 64.160 |
| `PromptFlashAttention_8` | 0 | 64.120 |
| `PromptFlashAttention_8` | 0 | 63.980 |
| `PromptFlashAttention_15` | 0 | 63.800 |
| `PromptFlashAttention_25` | 0 | 63.740 |
| `PromptFlashAttention_15` | 0 | 63.720 |
| `PromptFlashAttention_7` | 0 | 63.320 |
| `PromptFlashAttention_25` | 0 | 63.300 |
| `PromptFlashAttention_20` | 0 | 62.580 |
| `PromptFlashAttention_20` | 0 | 62.520 |
| `PromptFlashAttention_26` | 0 | 62.440 |
| `PromptFlashAttention_6` | 0 | 62.400 |
| `PromptFlashAttention_26` | 0 | 62.200 |
| `PromptFlashAttention_1` | 0 | 62.160 |
| `PromptFlashAttention_6` | 0 | 61.940 |
| `PromptFlashAttention_26` | 0 | 61.940 |
| `PromptFlashAttention_6` | 0 | 61.920 |
| `PromptFlashAttention_18` | 0 | 61.760 |
| `PromptFlashAttention_1` | 0 | 61.560 |
| `PromptFlashAttention_5` | 0 | 61.540 |
| `PromptFlashAttention_11` | 0 | 61.500 |
| `PromptFlashAttention_4` | 0 | 61.300 |
| `PromptFlashAttention_5` | 0 | 61.100 |
| `PromptFlashAttention_2` | 0 | 61.100 |
| `PromptFlashAttention_4` | 0 | 60.860 |
| `PromptFlashAttention_21` | 0 | 60.700 |
| `PromptFlashAttention_19` | 0 | 60.700 |
| `PromptFlashAttention_19` | 0 | 60.660 |
| `PromptFlashAttention_19` | 0 | 60.620 |
| `PromptFlashAttention_22` | 0 | 60.380 |
| `PromptFlashAttention_10` | 0 | 60.140 |
| `PromptFlashAttention_10` | 0 | 60.100 |
| `PromptFlashAttention_5` | 0 | 60.060 |
| `PromptFlashAttention_4` | 0 | 59.940 |
| `PromptFlashAttention_22` | 0 | 59.740 |
| `PromptFlashAttention_18` | 0 | 59.600 |
| `PromptFlashAttention_18` | 0 | 59.600 |
| `PromptFlashAttention_10` | 0 | 59.540 |
| `PromptFlashAttention_11` | 0 | 59.200 |
| `PromptFlashAttention_22` | 0 | 59.180 |
| `PromptFlashAttention_21` | 0 | 59.060 |
| `PromptFlashAttention_11` | 0 | 58.920 |
| `PromptFlashAttention_23` | 0 | 58.520 |
| `PromptFlashAttention_3` | 0 | 58.360 |
| `PromptFlashAttention_23` | 0 | 58.220 |
| `PromptFlashAttention_23` | 0 | 58.220 |
| `PromptFlashAttention_21` | 0 | 58.040 |
| `PromptFlashAttention_3` | 0 | 57.920 |
| `PromptFlashAttention_3` | 0 | 57.360 |
| `PromptFlashAttention_2` | 0 | 56.960 |
| `PromptFlashAttention_2` | 0 | 56.560 |
| `MatMulV2_5` | 0 | 31.200 |
| `MatMulV2_5` | 0 | 30.940 |
| `MatMulV2_47` | 0 | 30.300 |
| `MatMulV2_47` | 0 | 30.160 |
| `MatMulV2_41` | 0 | 29.380 |
| `MatMulV2_5` | 0 | 28.920 |
| `MatMulV2_4` | 0 | 28.780 |
| `MatMulV2_41` | 0 | 28.740 |
| `MatMulV2_143` | 0 | 28.680 |
| `MatMulV2_53` | 0 | 28.660 |
| `MatMulV2_41` | 0 | 28.620 |
| `MatMulV2_101` | 0 | 28.600 |
| `MatMulV2_119` | 0 | 28.600 |
| `MatMulV2_89` | 0 | 28.560 |
| `MatMulV2_29` | 0 | 28.380 |
| `MatMulV2_95` | 0 | 28.380 |
| `MatMulV2_47` | 0 | 28.360 |
| `MatMulV2_107` | 0 | 28.360 |
| `MatMulV2_53` | 0 | 28.340 |
| `MatMulV2_71` | 0 | 28.320 |
| `MatMulV2_89` | 0 | 28.280 |
| `MatMulV2_17` | 0 | 28.260 |
| `MatMulV2_29` | 0 | 28.260 |
| `MatMulV2_131` | 0 | 28.240 |
| `MatMulV2_89` | 0 | 28.220 |
| `MatMulV2_95` | 0 | 28.220 |
| `MatMulV2_137` | 0 | 28.220 |
| `MatMulV2_53` | 0 | 28.200 |
| `MatMulV2_161` | 0 | 28.200 |
| `MatMulV2_11` | 0 | 28.180 |
| `MatMulV2_149` | 0 | 28.180 |
| `MatMulV2_155` | 0 | 28.180 |
| `MatMulV2_71` | 0 | 28.160 |
| `MatMulV2_101` | 0 | 28.160 |
| `MatMulV2_101` | 0 | 28.160 |
| `MatMulV2_131` | 0 | 28.140 |
| `MatMulV2_125` | 0 | 28.140 |
| `MatMulV2_11` | 0 | 28.120 |
| `MatMulV2_17` | 0 | 28.120 |
| `MatMulV2_149` | 0 | 28.100 |
| `MatMulV2_59` | 0 | 28.100 |
| `MatMulV2_155` | 0 | 28.100 |
| `MatMulV2_65` | 0 | 28.080 |
| `MatMulV2_65` | 0 | 28.080 |
| `MatMulV2_161` | 0 | 28.060 |
| `MatMulV2_113` | 0 | 28.040 |
| `MatMulV2_35` | 0 | 28.040 |
| `MatMulV2_125` | 0 | 28.020 |
| `MatMulV2_143` | 0 | 28.000 |
| `MatMulV2_95` | 0 | 28.000 |
| `MatMulV2_155` | 0 | 27.940 |
| `MatMulV2_161` | 0 | 27.940 |
| `MatMulV2_100` | 0 | 27.940 |
| `MatMulV2_94` | 0 | 27.900 |
| `MatMulV2_131` | 0 | 27.900 |
| `MatMulV2_17` | 0 | 27.860 |
| `MatMulV2_71` | 0 | 27.860 |
| `MatMulV2_4` | 0 | 27.820 |
| `MatMulV2_23` | 0 | 27.780 |
| `MatMulV2_65` | 0 | 27.780 |
| `MatMulV2_77` | 0 | 27.760 |
| `MatMulV2_113` | 0 | 27.760 |
| `MatMulV2_64` | 0 | 27.740 |
| `MatMulV2_125` | 0 | 27.720 |
| `MatMulV2_11` | 0 | 27.720 |
| `MatMulV2_106` | 0 | 27.720 |
| `MatMulV2_29` | 0 | 27.660 |
| `MatMulV2_83` | 0 | 27.660 |
| `MatMulV2_58` | 0 | 27.620 |
| `MatMulV2_143` | 0 | 27.620 |
| `MatMulV2_136` | 0 | 27.600 |
| `MatMulV2_76` | 0 | 27.560 |
| `MatMulV2_136` | 0 | 27.520 |
| `MatMulV2_4` | 0 | 27.520 |
| `MatMulV2_113` | 0 | 27.520 |
| `MatMulV2_119` | 0 | 27.460 |
| `MatMulV2_106` | 0 | 27.440 |
| `MatMulV2_35` | 0 | 27.440 |
| `MatMulV2_58` | 0 | 27.440 |
| `MatMulV2_82` | 0 | 27.400 |
| `MatMulV2_112` | 0 | 27.380 |
| `MatMulV2_10` | 0 | 27.340 |
| `MatMulV2_136` | 0 | 27.340 |
| `MatMulV2_124` | 0 | 27.340 |
| `MatMulV2_77` | 0 | 27.320 |
| `MatMulV2_82` | 0 | 27.320 |
| `MatMulV2_83` | 0 | 27.320 |
| `MatMulV2_130` | 0 | 27.300 |
| `MatMulV2_76` | 0 | 27.260 |
| `MatMulV2_112` | 0 | 27.260 |
| `MatMulV2_100` | 0 | 27.260 |
| `MatMulV2_106` | 0 | 27.260 |
| `MatMulV2_100` | 0 | 27.240 |
| `MatMulV2_118` | 0 | 27.240 |
| `MatMulV2_77` | 0 | 27.240 |
| `MatMulV2_82` | 0 | 27.220 |
| `MatMulV2_142` | 0 | 27.220 |
| `MatMulV2_76` | 0 | 27.200 |
| `MatMulV2_70` | 0 | 27.180 |
| `MatMulV2_40` | 0 | 27.140 |
| `MatMulV2_10` | 0 | 27.080 |
| `MatMulV2_142` | 0 | 27.060 |
| `MatMulV2_88` | 0 | 27.060 |
| `MatMulV2_112` | 0 | 27.040 |
| `MatMulV2_142` | 0 | 27.040 |
| `MatMulV2_22` | 0 | 27.000 |
| `MatMulV2_94` | 0 | 26.980 |
| `MatMulV2_34` | 0 | 26.980 |
| `MatMulV2_160` | 0 | 26.940 |
| `MatMulV2_64` | 0 | 26.880 |
| `MatMulV2_160` | 0 | 26.880 |
| `MatMulV2_130` | 0 | 26.880 |
| `MatMulV2_94` | 0 | 26.880 |
| `MatMulV2_107` | 0 | 26.880 |
| `MatMulV2_130` | 0 | 26.880 |
| `MatMulV2_10` | 0 | 26.840 |
| `MatMulV2_88` | 0 | 26.840 |
| `MatMulV2_22` | 0 | 26.840 |
| `MatMulV2_124` | 0 | 26.820 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 16388.250 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.fractal_nz.torchair.active.step1` | 1 | 15253.480 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.fractal_nz.torchair.active.step2` | 1 | 14891.930 |
| `paddleocr_vl.vision_matmul_lab.S512.I4352.fractal_nz.torchair.active.step3` | 1 | 14883.750 |
| `TorchDynamo Cache Lookup` | 3 | 13621.770 |
| `Torch-Compiled Region: 0/0` | 3 | 3581.220 |
| `TorchNpuGraphBase::Run` | 3 | 2638.960 |
| `RefreshAtTensorFromGeTensor` | 3 | 1138.620 |
| `aten::empty` | 3 | 546.640 |
| `ExecuteGraph` | 3 | 449.040 |
| `AssembleInputs` | 3 | 381.760 |
| `aten::set_` | 3 | 290.540 |
| `AssembleOutputs` | 3 | 276.220 |
| `empty_tensor` | 3 | 272.360 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 211686.490 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 39415.480 |
| `launch` | 976 | 16268.660 |
| `InputCopy` | 3 | 129.600 |
| `ModelExecute` | 3 | 41.760 |
| `step_info` | 6 | 23.860 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 21.790 |
| `OutputCopy` | 3 | 0.930 |

