# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b4s512_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b4s512_pipe/liteserver-c001-4_759972_20260730140153552_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `67059.520 us`
- `Free`: `3048.420 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3661.750 us`
- `Stage`: `70107.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 24107.880 |
| `StridedSliceD` | 324 | 8853.080 |
| `PromptFlashAttention` | 81 | 8614.140 |
| `Transpose` | 324 | 5431.840 |
| `AddLayerNorm` | 162 | 3982.460 |
| `Mul` | 324 | 3331.860 |
| `Gelu` | 81 | 3106.920 |
| `ConcatV2D` | 243 | 2745.340 |
| `Add` | 162 | 1990.540 |
| `Cast` | 162 | 1879.000 |
| `Neg` | 162 | 1636.940 |
| `SplitVD` | 81 | 1267.360 |
| `LayerNormV3` | 3 | 97.760 |
| `Data` | 3 | 14.400 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_9` | 3 | 337.400 |
| `PromptFlashAttention_16` | 3 | 336.960 |
| `PromptFlashAttention_25` | 3 | 336.820 |
| `PromptFlashAttention_15` | 3 | 336.200 |
| `PromptFlashAttention_1` | 3 | 335.020 |
| `PromptFlashAttention` | 3 | 334.660 |
| `PromptFlashAttention_10` | 3 | 332.920 |
| `PromptFlashAttention_2` | 3 | 331.700 |
| `PromptFlashAttention_26` | 3 | 329.020 |
| `PromptFlashAttention_17` | 3 | 327.920 |
| `PromptFlashAttention_11` | 3 | 324.300 |
| `PromptFlashAttention_21` | 3 | 319.680 |
| `PromptFlashAttention_18` | 3 | 318.780 |
| `PromptFlashAttention_3` | 3 | 317.560 |
| `PromptFlashAttention_4` | 3 | 314.100 |
| `PromptFlashAttention_20` | 3 | 313.420 |
| `PromptFlashAttention_19` | 3 | 311.540 |
| `PromptFlashAttention_5` | 3 | 309.340 |
| `PromptFlashAttention_23` | 3 | 309.020 |
| `PromptFlashAttention_12` | 3 | 307.500 |
| `PromptFlashAttention_14` | 3 | 307.180 |
| `PromptFlashAttention_8` | 3 | 305.380 |
| `PromptFlashAttention_24` | 3 | 305.120 |
| `PromptFlashAttention_13` | 3 | 304.700 |
| `PromptFlashAttention_6` | 3 | 304.480 |
| `PromptFlashAttention_22` | 3 | 302.760 |
| `PromptFlashAttention_7` | 3 | 300.660 |
| `MatMulV2_149` | 3 | 281.020 |
| `MatMulV2_113` | 3 | 277.940 |
| `MatMulV2_65` | 3 | 277.840 |
| `MatMulV2_155` | 3 | 277.620 |
| `MatMulV2_53` | 3 | 277.240 |
| `MatMulV2_125` | 3 | 277.160 |
| `MatMulV2_89` | 3 | 277.000 |
| `MatMulV2_95` | 3 | 276.980 |
| `MatMulV2_101` | 3 | 276.880 |
| `MatMulV2_143` | 3 | 276.880 |
| `MatMulV2_77` | 3 | 276.820 |
| `MatMulV2_131` | 3 | 276.180 |
| `MatMulV2_83` | 3 | 275.780 |
| `MatMulV2_119` | 3 | 275.740 |
| `MatMulV2_59` | 3 | 275.320 |
| `MatMulV2_41` | 3 | 274.440 |
| `MatMulV2_35` | 3 | 271.660 |
| `MatMulV2_11` | 3 | 269.600 |
| `MatMulV2_29` | 3 | 269.600 |
| `MatMulV2_23` | 3 | 269.300 |
| `MatMulV2_5` | 3 | 269.160 |
| `MatMulV2_107` | 3 | 265.280 |
| `MatMulV2_71` | 3 | 264.520 |
| `MatMulV2_47` | 3 | 264.200 |
| `MatMulV2_161` | 3 | 263.880 |
| `MatMulV2_17` | 3 | 261.300 |
| `MatMulV2_137` | 3 | 260.600 |
| `MatMulV2_142` | 3 | 259.440 |
| `MatMulV2_70` | 3 | 259.200 |
| `MatMulV2_118` | 3 | 259.020 |
| `MatMulV2_148` | 3 | 258.880 |
| `MatMulV2_112` | 3 | 258.740 |
| `MatMulV2_82` | 3 | 258.680 |
| `MatMulV2_154` | 3 | 258.320 |
| `MatMulV2_4` | 3 | 257.860 |
| `MatMulV2_94` | 3 | 257.220 |
| `MatMulV2_76` | 3 | 257.180 |
| `MatMulV2_100` | 3 | 257.040 |
| `MatMulV2_52` | 3 | 256.460 |
| `MatMulV2_136` | 3 | 256.200 |
| `MatMulV2_58` | 3 | 255.880 |
| `MatMulV2_124` | 3 | 255.380 |
| `MatMulV2_34` | 3 | 255.180 |
| `MatMulV2_106` | 3 | 254.880 |
| `MatMulV2_46` | 3 | 254.840 |
| `MatMulV2_28` | 3 | 254.680 |
| `MatMulV2_88` | 3 | 254.560 |
| `MatMulV2_40` | 3 | 253.700 |
| `MatMulV2_64` | 3 | 253.700 |
| `MatMulV2_130` | 3 | 253.440 |
| `MatMulV2_10` | 3 | 251.840 |
| `MatMulV2_16` | 3 | 251.140 |
| `MatMulV2_160` | 3 | 249.540 |
| `MatMulV2_22` | 3 | 248.120 |
| `MatMulV2_21` | 3 | 119.420 |
| `MatMulV2_63` | 3 | 117.580 |
| `MatMulV2_45` | 3 | 117.440 |
| `MatMulV2_129` | 3 | 117.360 |
| `MatMulV2_159` | 3 | 117.220 |
| `MatMulV2_135` | 3 | 117.100 |
| `MatMulV2_69` | 3 | 116.920 |
| `MatMulV2_99` | 3 | 116.880 |
| `MatMulV2_93` | 3 | 116.680 |
| `MatMulV2_15` | 3 | 116.620 |
| `MatMulV2_87` | 3 | 116.440 |
| `MatMulV2_51` | 3 | 116.420 |
| `MatMulV2_141` | 3 | 116.260 |
| `Gelu_11` | 3 | 116.080 |
| `Gelu_2` | 3 | 116.060 |
| `Gelu_26` | 3 | 116.060 |
| `Gelu_17` | 3 | 115.940 |
| `Gelu_7` | 3 | 115.840 |
| `MatMulV2_123` | 3 | 115.740 |
| `Gelu_22` | 3 | 115.740 |
| `MatMulV2_33` | 3 | 115.680 |
| `MatMulV2_75` | 3 | 115.660 |
| `MatMulV2_153` | 3 | 115.380 |
| `MatMulV2_111` | 3 | 115.300 |
| `Gelu_5` | 3 | 115.020 |
| `Gelu_21` | 3 | 114.980 |
| `Gelu_20` | 3 | 114.960 |
| `Gelu_8` | 3 | 114.920 |
| `Gelu_9` | 3 | 114.920 |
| `Gelu_25` | 3 | 114.900 |
| `Gelu_13` | 3 | 114.900 |
| `Gelu_1` | 3 | 114.840 |
| `Gelu_3` | 3 | 114.840 |
| `Gelu_23` | 3 | 114.820 |
| `Gelu_10` | 3 | 114.800 |
| `Gelu_19` | 3 | 114.800 |
| `Gelu_15` | 3 | 114.780 |
| `Gelu_12` | 3 | 114.760 |
| `Gelu_14` | 3 | 114.760 |
| `Gelu_18` | 3 | 114.760 |
| `Gelu_24` | 3 | 114.740 |
| `Gelu` | 3 | 114.720 |
| `Gelu_6` | 3 | 114.680 |
| `Gelu_16` | 3 | 114.680 |
| `Gelu_4` | 3 | 114.620 |
| `MatMulV2_9` | 3 | 112.440 |
| `MatMulV2_39` | 3 | 112.260 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 106.560 |
| `MatMulV2_117` | 3 | 105.720 |
| `MatMulV2_3` | 3 | 105.140 |
| `MatMulV2_147` | 3 | 104.880 |
| `MatMulV2_57` | 3 | 104.140 |
| `MatMulV2_105` | 3 | 102.640 |
| `StridedSliceV2_100` | 3 | 102.380 |
| `MatMulV2_27` | 3 | 101.320 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 100.980 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 100.780 |
| `MatMulV2_81` | 3 | 100.740 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 100.600 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 100.300 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 99.400 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 99.000 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 98.780 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 98.500 |
| `StridedSliceV2_98` | 3 | 98.360 |
| `StridedSliceV2_96` | 3 | 98.300 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 97.860 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 97.780 |
| `LayerNormV4_LayerNormV3` | 3 | 97.760 |
| `StridedSliceV2_65` | 3 | 97.460 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 97.420 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 97.360 |
| `StridedSliceV2_27` | 3 | 97.100 |
| `StridedSliceV2_63` | 3 | 96.980 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 96.860 |
| `MatMulV2_156` | 3 | 96.840 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 96.660 |
| `StridedSliceV2_31` | 3 | 96.500 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 96.320 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 96.220 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 96.160 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 96.080 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 96.000 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 96.000 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 95.700 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 95.680 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 95.320 |
| `StridedSliceV2_29` | 3 | 95.200 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 93.880 |
| `MatMulV2_60` | 3 | 92.820 |
| `MatMulV2_132` | 3 | 92.820 |
| `MatMulV2_96` | 3 | 92.380 |
| `MatMulV2_102` | 3 | 92.140 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 92.120 |
| `MatMulV2_108` | 3 | 91.980 |
| `MatMulV2_138` | 3 | 91.700 |
| `MatMulV2_48` | 3 | 91.560 |
| `MatMulV2_126` | 3 | 91.520 |
| `MatMulV2_72` | 3 | 91.420 |
| `MatMulV2_84` | 3 | 91.320 |
| `MatMulV2_36` | 3 | 91.300 |
| `MatMulV2` | 3 | 91.280 |
| `MatMulV2_66` | 3 | 91.280 |
| `MatMulV2_120` | 3 | 91.240 |
| `MatMulV2_18` | 3 | 91.000 |
| `MatMulV2_24` | 3 | 90.940 |
| `MatMulV2_30` | 3 | 90.920 |
| `MatMulV2_78` | 3 | 90.920 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 90.920 |
| `MatMulV2_42` | 3 | 90.580 |
| `MatMulV2_150` | 3 | 90.560 |
| `MatMulV2_90` | 3 | 90.160 |
| `MatMulV2_12` | 3 | 90.000 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 89.980 |
| `MatMulV2_54` | 3 | 88.860 |
| `MatMulV2_114` | 3 | 88.560 |
| `MatMulV2_144` | 3 | 87.840 |
| `MatMulV2_6` | 3 | 87.640 |
| `StridedSliceV2_20` | 3 | 83.960 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `StridedSliceD | "4,512,16,80" -> "4,512,16,40" | ND -> ND` | 324 | 8853.080 |
| `PromptFlashAttention | "4,16,512,80;4,16,512,80;4,16,512,80;4,1,512,512" -> "4,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 81 | 8614.140 |
| `MatMulV2 | "2048,4352;272,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 7359.940 |
| `MatMulV2 | "2048,1152;72,272,16,16;4352" -> "2048,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 6901.120 |
| `MatMulV2 | "2048,1152;72,80,16,16;1280" -> "2048,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 6797.440 |
| `AddLayerNorm | "4,512,1152;4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1;4,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 3982.460 |
| `Transpose | "4,512,16,80;4" -> "4,16,512,80" | ND;ND -> ND` | 243 | 3661.240 |
| `Mul | "4,512,16,80;4,512,1,80" -> "4,512,16,80" | ND;ND -> ND` | 324 | 3331.860 |
| `Gelu | "4,512,4352" -> "4,512,4352" | ND -> ND` | 81 | 3106.920 |
| `MatMulV2 | "2048,1280;80,72,16,16;1152" -> "2048,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 3049.380 |
| `Add | "4,512,16,80;4,512,16,80" -> "4,512,16,80" | ND;ND -> ND` | 162 | 1990.540 |
| `Cast | "4,512,16,80" -> "4,512,16,80" | ND -> ND` | 162 | 1879.000 |
| `Transpose | "4,16,512,80;4" -> "4,512,16,80" | ND;ND -> ND` | 81 | 1770.600 |
| `ConcatV2D | "4,512,16,40;4,512,16,40" -> "4,512,16,80" | ND;ND -> ND` | 162 | 1728.360 |
| `Neg | "4,512,16,40" -> "4,512,16,40" | ND -> ND` | 162 | 1636.940 |
| `SplitVD | "4,512,3840" -> "4,512,1280;4,512,1280;4,512,1280" | ND -> ND;ND;ND` | 81 | 1267.360 |
| `ConcatV2D | "4,512,1280;4,512,1280;4,512,1280" -> "4,512,3840" | ND;ND;ND -> ND` | 81 | 1016.980 |
| `LayerNormV3 | "4,512,1152;1152;1152" -> "4,512,1152;4,512,1;4,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 97.760 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.400 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;FRACTAL_NZ;ND` | 486 | 24107.880 |
| `ND` | 810 | 16743.300 |
| `ND;ND` | 972 | 12482.600 |
| `NCHW;NCHW;NCHW;NCHW` | 81 | 8614.140 |
| `ND;ND;ND;ND` | 162 | 3982.460 |
| `ND;ND;ND` | 84 | 1114.740 |
| `N/A` | 3 | 14.400 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_9` | 0 | 116.100 |
| `PromptFlashAttention_16` | 0 | 116.000 |
| `PromptFlashAttention_16` | 0 | 113.500 |
| `PromptFlashAttention_9` | 0 | 112.980 |
| `PromptFlashAttention_15` | 0 | 112.900 |
| `PromptFlashAttention_15` | 0 | 112.800 |
| `PromptFlashAttention_25` | 0 | 112.780 |
| `PromptFlashAttention_1` | 0 | 112.580 |
| `PromptFlashAttention_25` | 0 | 112.320 |
| `PromptFlashAttention` | 0 | 112.300 |
| `PromptFlashAttention_10` | 0 | 112.140 |
| `PromptFlashAttention_1` | 0 | 111.960 |
| `PromptFlashAttention_26` | 0 | 111.900 |
| `PromptFlashAttention_25` | 0 | 111.720 |
| `PromptFlashAttention` | 0 | 111.380 |
| `PromptFlashAttention` | 0 | 110.980 |
| `PromptFlashAttention_2` | 0 | 110.980 |
| `PromptFlashAttention_10` | 0 | 110.880 |
| `PromptFlashAttention_2` | 0 | 110.560 |
| `PromptFlashAttention_15` | 0 | 110.500 |
| `PromptFlashAttention_1` | 0 | 110.480 |
| `PromptFlashAttention_2` | 0 | 110.160 |
| `PromptFlashAttention_17` | 0 | 110.040 |
| `PromptFlashAttention_10` | 0 | 109.900 |
| `PromptFlashAttention_26` | 0 | 109.840 |
| `PromptFlashAttention_17` | 0 | 109.460 |
| `PromptFlashAttention_11` | 0 | 109.420 |
| `PromptFlashAttention_17` | 0 | 108.420 |
| `PromptFlashAttention_9` | 0 | 108.320 |
| `PromptFlashAttention_21` | 0 | 107.900 |
| `PromptFlashAttention_11` | 0 | 107.540 |
| `PromptFlashAttention_16` | 0 | 107.460 |
| `PromptFlashAttention_11` | 0 | 107.340 |
| `PromptFlashAttention_26` | 0 | 107.280 |
| `PromptFlashAttention_18` | 0 | 106.920 |
| `PromptFlashAttention_3` | 0 | 106.780 |
| `PromptFlashAttention_18` | 0 | 106.660 |
| `PromptFlashAttention_20` | 0 | 106.100 |
| `PromptFlashAttention_21` | 0 | 106.060 |
| `PromptFlashAttention_21` | 0 | 105.720 |
| `PromptFlashAttention_4` | 0 | 105.540 |
| `PromptFlashAttention_3` | 0 | 105.440 |
| `PromptFlashAttention_4` | 0 | 105.400 |
| `PromptFlashAttention_3` | 0 | 105.340 |
| `PromptFlashAttention_18` | 0 | 105.200 |
| `PromptFlashAttention_5` | 0 | 105.080 |
| `PromptFlashAttention_19` | 0 | 104.780 |
| `PromptFlashAttention_19` | 0 | 103.980 |
| `PromptFlashAttention_23` | 0 | 103.880 |
| `PromptFlashAttention_20` | 0 | 103.860 |
| `PromptFlashAttention_14` | 0 | 103.820 |
| `PromptFlashAttention_12` | 0 | 103.560 |
| `PromptFlashAttention_20` | 0 | 103.460 |
| `PromptFlashAttention_4` | 0 | 103.160 |
| `PromptFlashAttention_5` | 0 | 103.020 |
| `PromptFlashAttention_24` | 0 | 102.960 |
| `PromptFlashAttention_19` | 0 | 102.780 |
| `PromptFlashAttention_23` | 0 | 102.760 |
| `PromptFlashAttention_13` | 0 | 102.400 |
| `PromptFlashAttention_23` | 0 | 102.380 |
| `PromptFlashAttention_12` | 0 | 102.240 |
| `PromptFlashAttention_6` | 0 | 102.040 |
| `PromptFlashAttention_8` | 0 | 101.980 |
| `PromptFlashAttention_6` | 0 | 101.960 |
| `PromptFlashAttention_8` | 0 | 101.900 |
| `PromptFlashAttention_14` | 0 | 101.720 |
| `PromptFlashAttention_12` | 0 | 101.700 |
| `PromptFlashAttention_14` | 0 | 101.640 |
| `PromptFlashAttention_13` | 0 | 101.640 |
| `PromptFlashAttention_24` | 0 | 101.520 |
| `PromptFlashAttention_8` | 0 | 101.500 |
| `PromptFlashAttention_22` | 0 | 101.480 |
| `PromptFlashAttention_5` | 0 | 101.240 |
| `PromptFlashAttention_22` | 0 | 101.040 |
| `PromptFlashAttention_7` | 0 | 100.900 |
| `PromptFlashAttention_13` | 0 | 100.660 |
| `PromptFlashAttention_24` | 0 | 100.640 |
| `PromptFlashAttention_6` | 0 | 100.480 |
| `PromptFlashAttention_7` | 0 | 100.240 |
| `PromptFlashAttention_22` | 0 | 100.240 |
| `PromptFlashAttention_7` | 0 | 99.520 |
| `MatMulV2_149` | 0 | 94.200 |
| `MatMulV2_149` | 0 | 93.900 |
| `MatMulV2_5` | 0 | 93.640 |
| `MatMulV2_65` | 0 | 93.340 |
| `MatMulV2_95` | 0 | 93.240 |
| `MatMulV2_53` | 0 | 93.200 |
| `MatMulV2_35` | 0 | 93.160 |
| `MatMulV2_113` | 0 | 93.120 |
| `MatMulV2_11` | 0 | 93.080 |
| `MatMulV2_113` | 0 | 92.980 |
| `MatMulV2_125` | 0 | 92.960 |
| `MatMulV2_149` | 0 | 92.920 |
| `MatMulV2_89` | 0 | 92.900 |
| `MatMulV2_155` | 0 | 92.900 |
| `MatMulV2_65` | 0 | 92.880 |
| `MatMulV2_53` | 0 | 92.780 |
| `MatMulV2_101` | 0 | 92.720 |
| `MatMulV2_23` | 0 | 92.720 |
| `MatMulV2_101` | 0 | 92.680 |
| `MatMulV2_83` | 0 | 92.620 |
| `MatMulV2_143` | 0 | 92.420 |
| `MatMulV2_77` | 0 | 92.400 |
| `MatMulV2_143` | 0 | 92.400 |
| `MatMulV2_155` | 0 | 92.400 |
| `MatMulV2_131` | 0 | 92.320 |
| `MatMulV2_155` | 0 | 92.320 |
| `MatMulV2_11` | 0 | 92.260 |
| `MatMulV2_77` | 0 | 92.220 |
| `MatMulV2_119` | 0 | 92.220 |
| `MatMulV2_77` | 0 | 92.200 |
| `MatMulV2_125` | 0 | 92.180 |
| `MatMulV2_95` | 0 | 92.120 |
| `MatMulV2_89` | 0 | 92.060 |
| `MatMulV2_143` | 0 | 92.060 |
| `MatMulV2_41` | 0 | 92.040 |
| `MatMulV2_89` | 0 | 92.040 |
| `MatMulV2_125` | 0 | 92.020 |
| `MatMulV2_23` | 0 | 92.000 |
| `MatMulV2_131` | 0 | 91.980 |
| `MatMulV2_119` | 0 | 91.940 |
| `MatMulV2_59` | 0 | 91.900 |
| `MatMulV2_131` | 0 | 91.880 |
| `MatMulV2_29` | 0 | 91.880 |
| `MatMulV2_113` | 0 | 91.840 |
| `MatMulV2_29` | 0 | 91.840 |
| `MatMulV2_59` | 0 | 91.780 |
| `MatMulV2_59` | 0 | 91.640 |
| `MatMulV2_65` | 0 | 91.620 |
| `MatMulV2_83` | 0 | 91.620 |
| `MatMulV2_95` | 0 | 91.620 |
| `MatMulV2_119` | 0 | 91.580 |
| `MatMulV2_83` | 0 | 91.540 |
| `MatMulV2_41` | 0 | 91.500 |
| `MatMulV2_5` | 0 | 91.480 |
| `MatMulV2_101` | 0 | 91.480 |
| `MatMulV2_53` | 0 | 91.260 |
| `MatMulV2_35` | 0 | 91.060 |
| `MatMulV2_41` | 0 | 90.900 |
| `MatMulV2_107` | 0 | 89.020 |
| `MatMulV2_17` | 0 | 88.960 |
| `MatMulV2_17` | 0 | 88.720 |
| `MatMulV2_161` | 0 | 88.660 |
| `MatMulV2_71` | 0 | 88.540 |
| `MatMulV2_47` | 0 | 88.400 |
| `MatMulV2_137` | 0 | 88.380 |
| `MatMulV2_161` | 0 | 88.240 |
| `MatMulV2_137` | 0 | 88.180 |
| `MatMulV2_107` | 0 | 88.160 |
| `MatMulV2_71` | 0 | 88.120 |
| `MatMulV2_107` | 0 | 88.100 |
| `MatMulV2_47` | 0 | 88.000 |
| `MatMulV2_71` | 0 | 87.860 |
| `MatMulV2_47` | 0 | 87.800 |
| `MatMulV2_35` | 0 | 87.440 |
| `MatMulV2_28` | 0 | 87.440 |
| `MatMulV2_154` | 0 | 87.400 |
| `MatMulV2_70` | 0 | 87.360 |
| `MatMulV2_118` | 0 | 87.220 |
| `MatMulV2_94` | 0 | 87.200 |
| `MatMulV2_34` | 0 | 87.120 |
| `MatMulV2_34` | 0 | 87.100 |
| `MatMulV2_70` | 0 | 87.040 |
| `MatMulV2_161` | 0 | 86.980 |
| `MatMulV2_148` | 0 | 86.880 |
| `MatMulV2_40` | 0 | 86.800 |
| `MatMulV2_82` | 0 | 86.800 |
| `MatMulV2_94` | 0 | 86.800 |
| `MatMulV2_154` | 0 | 86.720 |
| `MatMulV2_112` | 0 | 86.720 |
| `MatMulV2_142` | 0 | 86.660 |
| `MatMulV2_46` | 0 | 86.660 |
| `MatMulV2_4` | 0 | 86.640 |
| `MatMulV2_112` | 0 | 86.600 |
| `MatMulV2_142` | 0 | 86.560 |
| `MatMulV2_10` | 0 | 86.440 |
| `MatMulV2_40` | 0 | 86.440 |
| `MatMulV2_124` | 0 | 86.360 |
| `MatMulV2_58` | 0 | 86.280 |
| `MatMulV2_142` | 0 | 86.220 |
| `MatMulV2_136` | 0 | 86.180 |
| `MatMulV2_100` | 0 | 86.160 |
| `MatMulV2_82` | 0 | 86.160 |
| `MatMulV2_10` | 0 | 86.140 |
| `MatMulV2_58` | 0 | 86.120 |
| `MatMulV2_118` | 0 | 86.120 |
| `MatMulV2_88` | 0 | 86.080 |
| `MatMulV2_148` | 0 | 86.080 |
| `MatMulV2_100` | 0 | 86.060 |
| `MatMulV2_88` | 0 | 86.040 |
| `MatMulV2_76` | 0 | 85.960 |
| `MatMulV2_148` | 0 | 85.920 |
| `MatMulV2_124` | 0 | 85.900 |
| `MatMulV2_16` | 0 | 85.900 |
| `MatMulV2_29` | 0 | 85.880 |
| `MatMulV2_76` | 0 | 85.880 |
| `MatMulV2_64` | 0 | 85.820 |
| `MatMulV2_4` | 0 | 85.800 |
| `MatMulV2_136` | 0 | 85.780 |
| `MatMulV2_46` | 0 | 85.760 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 25416.390 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step1` | 1 | 23933.500 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step3` | 1 | 23610.570 |
| `paddleocr_vl.vision_matmul_lab.B4.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step2` | 1 | 23572.690 |
| `TorchDynamo Cache Lookup` | 3 | 22490.370 |
| `Torch-Compiled Region: 0/0` | 3 | 3844.410 |
| `TorchNpuGraphBase::Run` | 3 | 2854.010 |
| `RefreshAtTensorFromGeTensor` | 3 | 1225.020 |
| `aten::empty` | 3 | 600.890 |
| `ExecuteGraph` | 3 | 471.690 |
| `AssembleInputs` | 3 | 441.220 |
| `AssembleOutputs` | 3 | 312.880 |
| `aten::set_` | 3 | 299.370 |
| `empty_tensor` | 3 | 295.400 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 180308.160 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 65246.650 |
| `launch` | 868 | 11643.960 |
| `InputCopy` | 3 | 124.440 |
| `ModelExecute` | 3 | 43.330 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 22.650 |
| `step_info` | 6 | 12.490 |
| `OutputCopy` | 3 | 0.990 |

