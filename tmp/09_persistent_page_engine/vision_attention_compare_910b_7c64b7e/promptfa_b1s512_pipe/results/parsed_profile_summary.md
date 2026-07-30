# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b1s512_pipe`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_vision_matmul_profiles/vision_attention_compare_910b_7c64b7e/promptfa_b1s512_pipe/liteserver-c001-4_755619_20260730135724853_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `33706.840 us`
- `Free`: `3026.560 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3599.000 us`
- `Stage`: `36733.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 486 | 10013.040 |
| `PromptFlashAttention` | 81 | 5031.280 |
| `StridedSliceD` | 324 | 4438.740 |
| `Transpose` | 324 | 2922.640 |
| `AddLayerNorm` | 162 | 2236.920 |
| `Mul` | 324 | 1839.580 |
| `ConcatV2D` | 243 | 1565.960 |
| `Cast` | 162 | 1385.620 |
| `Add` | 162 | 1267.040 |
| `Gelu` | 81 | 1230.120 |
| `Neg` | 162 | 1211.800 |
| `SplitVD` | 81 | 508.460 |
| `LayerNormV3` | 3 | 41.240 |
| `Data` | 3 | 14.400 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 3 | 208.820 |
| `PromptFlashAttention_26` | 3 | 206.020 |
| `PromptFlashAttention_16` | 3 | 200.620 |
| `PromptFlashAttention_19` | 3 | 199.660 |
| `PromptFlashAttention_2` | 3 | 198.220 |
| `PromptFlashAttention` | 3 | 196.840 |
| `PromptFlashAttention_10` | 3 | 193.500 |
| `PromptFlashAttention_17` | 3 | 192.800 |
| `PromptFlashAttention_9` | 3 | 191.980 |
| `PromptFlashAttention_14` | 3 | 189.060 |
| `PromptFlashAttention_24` | 3 | 188.960 |
| `PromptFlashAttention_3` | 3 | 187.820 |
| `PromptFlashAttention_11` | 3 | 187.020 |
| `PromptFlashAttention_1` | 3 | 185.920 |
| `PromptFlashAttention_23` | 3 | 185.900 |
| `PromptFlashAttention_18` | 3 | 185.180 |
| `PromptFlashAttention_7` | 3 | 180.120 |
| `PromptFlashAttention_4` | 3 | 177.880 |
| `PromptFlashAttention_8` | 3 | 177.860 |
| `PromptFlashAttention_20` | 3 | 176.860 |
| `PromptFlashAttention_25` | 3 | 175.700 |
| `PromptFlashAttention_5` | 3 | 175.600 |
| `PromptFlashAttention_13` | 3 | 174.900 |
| `PromptFlashAttention_22` | 3 | 174.640 |
| `PromptFlashAttention_6` | 3 | 173.660 |
| `PromptFlashAttention_12` | 3 | 173.580 |
| `PromptFlashAttention_21` | 3 | 172.160 |
| `MatMulV2_23` | 3 | 86.340 |
| `MatMulV2_4` | 3 | 85.940 |
| `MatMulV2_142` | 3 | 85.420 |
| `MatMulV2_137` | 3 | 84.340 |
| `MatMulV2_41` | 3 | 83.860 |
| `MatMulV2_59` | 3 | 83.740 |
| `MatMulV2_47` | 3 | 83.540 |
| `MatMulV2_95` | 3 | 83.340 |
| `MatMulV2_148` | 3 | 83.300 |
| `MatMulV2_5` | 3 | 83.280 |
| `MatMulV2_130` | 3 | 83.280 |
| `MatMulV2_83` | 3 | 82.860 |
| `MatMulV2_124` | 3 | 82.720 |
| `MatMulV2_40` | 3 | 82.680 |
| `MatMulV2_71` | 3 | 82.680 |
| `MatMulV2_65` | 3 | 82.580 |
| `MatMulV2_88` | 3 | 82.540 |
| `MatMulV2_16` | 3 | 82.440 |
| `MatMulV2_161` | 3 | 82.440 |
| `MatMulV2_155` | 3 | 82.420 |
| `MatMulV2_119` | 3 | 82.320 |
| `MatMulV2_125` | 3 | 82.300 |
| `MatMulV2_10` | 3 | 82.000 |
| `MatMulV2_35` | 3 | 81.960 |
| `MatMulV2_154` | 3 | 81.880 |
| `MatMulV2_64` | 3 | 81.860 |
| `MatMulV2_136` | 3 | 81.800 |
| `MatMulV2_101` | 3 | 81.760 |
| `MatMulV2_70` | 3 | 81.740 |
| `MatMulV2_89` | 3 | 81.660 |
| `MatMulV2_76` | 3 | 81.500 |
| `MatMulV2_160` | 3 | 81.480 |
| `MatMulV2_52` | 3 | 81.420 |
| `MatMulV2_11` | 3 | 81.320 |
| `MatMulV2_149` | 3 | 81.040 |
| `MatMulV2_46` | 3 | 81.020 |
| `MatMulV2_143` | 3 | 81.000 |
| `MatMulV2_29` | 3 | 80.960 |
| `MatMulV2_131` | 3 | 80.780 |
| `MatMulV2_113` | 3 | 80.680 |
| `MatMulV2_106` | 3 | 80.640 |
| `MatMulV2_107` | 3 | 80.580 |
| `MatMulV2_17` | 3 | 80.580 |
| `MatMulV2_94` | 3 | 80.560 |
| `MatMulV2_77` | 3 | 80.320 |
| `MatMulV2_53` | 3 | 80.300 |
| `MatMulV2_118` | 3 | 80.220 |
| `MatMulV2_34` | 3 | 79.580 |
| `MatMulV2_82` | 3 | 79.500 |
| `MatMulV2_112` | 3 | 79.180 |
| `MatMulV2_58` | 3 | 79.020 |
| `MatMulV2_100` | 3 | 78.860 |
| `MatMulV2_28` | 3 | 78.300 |
| `MatMulV2_22` | 3 | 78.260 |
| `MatMulV2_144` | 3 | 73.420 |
| `MatMulV2_72` | 3 | 73.280 |
| `MatMulV2_114` | 3 | 73.280 |
| `MatMulV2_120` | 3 | 73.260 |
| `MatMulV2_54` | 3 | 73.260 |
| `MatMulV2_126` | 3 | 73.180 |
| `MatMulV2_48` | 3 | 73.000 |
| `MatMulV2_108` | 3 | 72.980 |
| `MatMulV2_90` | 3 | 72.940 |
| `MatMulV2_6` | 3 | 72.800 |
| `MatMulV2_156` | 3 | 72.740 |
| `MatMulV2_102` | 3 | 72.680 |
| `MatMulV2_78` | 3 | 72.660 |
| `MatMulV2_36` | 3 | 72.360 |
| `MatMulV2_18` | 3 | 72.320 |
| `MatMulV2_30` | 3 | 72.180 |
| `MatMulV2_150` | 3 | 72.140 |
| `MatMulV2_132` | 3 | 72.060 |
| `MatMulV2_66` | 3 | 71.540 |
| `MatMulV2_84` | 3 | 70.800 |
| `MatMulV2_24` | 3 | 70.120 |
| `MatMulV2_135` | 3 | 69.400 |
| `MatMulV2_45` | 3 | 69.260 |
| `MatMulV2_42` | 3 | 68.540 |
| `LayerNormV4_9_LayerNormV3/AddLayerNorm` | 3 | 68.400 |
| `MatMulV2_51` | 3 | 68.200 |
| `MatMulV2_153` | 3 | 68.020 |
| `MatMulV2_39` | 3 | 68.000 |
| `MatMulV2` | 3 | 67.980 |
| `MatMulV2_33` | 3 | 67.840 |
| `MatMulV2_75` | 3 | 67.480 |
| `MatMulV2_105` | 3 | 67.480 |
| `MatMulV2_81` | 3 | 67.120 |
| `MatMulV2_129` | 3 | 66.920 |
| `MatMulV2_141` | 3 | 66.680 |
| `MatMulV2_99` | 3 | 64.980 |
| `MatMulV2_111` | 3 | 64.280 |
| `MatMulV2_159` | 3 | 64.280 |
| `MatMulV2_147` | 3 | 64.220 |
| `MatMulV2_9` | 3 | 63.840 |
| `MatMulV2_123` | 3 | 63.720 |
| `MatMulV2_93` | 3 | 63.200 |
| `MatMulV2_117` | 3 | 63.100 |
| `MatMulV2_96` | 3 | 62.380 |
| `MatMulV2_27` | 3 | 62.340 |
| `MatMulV2_15` | 3 | 62.220 |
| `MatMulV2_69` | 3 | 61.300 |
| `MatMulV2_60` | 3 | 60.860 |
| `MatMulV2_138` | 3 | 60.760 |
| `MatMulV2_57` | 3 | 60.580 |
| `MatMulV2_21` | 3 | 60.200 |
| `MatMulV2_12` | 3 | 59.660 |
| `Gelu_5` | 3 | 59.220 |
| `Gelu_14` | 3 | 58.840 |
| `MatMulV2_87` | 3 | 58.400 |
| `LayerNormV4_53_LayerNormV3/AddLayerNorm` | 3 | 56.760 |
| `LayerNormV4_1_LayerNormV3/AddLayerNorm` | 3 | 56.680 |
| `LayerNormV4_25_LayerNormV3/AddLayerNorm` | 3 | 56.660 |
| `LayerNormV4_13_LayerNormV3/AddLayerNorm` | 3 | 56.640 |
| `LayerNormV4_15_LayerNormV3/AddLayerNorm` | 3 | 56.440 |
| `LayerNormV4_3_LayerNormV3/AddLayerNorm` | 3 | 56.400 |
| `LayerNormV4_37_LayerNormV3/AddLayerNorm` | 3 | 56.380 |
| `LayerNormV4_27_LayerNormV3/AddLayerNorm` | 3 | 56.300 |
| `LayerNormV4_23_LayerNormV3/AddLayerNorm` | 3 | 56.260 |
| `LayerNormV4_47_LayerNormV3/AddLayerNorm` | 3 | 56.260 |
| `LayerNormV4_21_LayerNormV3/AddLayerNorm` | 3 | 56.180 |
| `LayerNormV4_41_LayerNormV3/AddLayerNorm` | 3 | 56.140 |
| `LayerNormV4_31_LayerNormV3/AddLayerNorm` | 3 | 56.060 |
| `LayerNormV4_43_LayerNormV3/AddLayerNorm` | 3 | 56.020 |
| `LayerNormV4_7_LayerNormV3/AddLayerNorm` | 3 | 56.020 |
| `LayerNormV4_33_LayerNormV3/AddLayerNorm` | 3 | 56.020 |
| `LayerNormV4_29_LayerNormV3/AddLayerNorm` | 3 | 56.000 |
| `LayerNormV4_39_LayerNormV3/AddLayerNorm` | 3 | 55.980 |
| `LayerNormV4_49_LayerNormV3/AddLayerNorm` | 3 | 55.960 |
| `LayerNormV4_17_LayerNormV3/AddLayerNorm` | 3 | 55.940 |
| `LayerNormV4_45_LayerNormV3/AddLayerNorm` | 3 | 55.940 |
| `LayerNormV4_5_LayerNormV3/AddLayerNorm` | 3 | 55.920 |
| `LayerNormV4_19_LayerNormV3/AddLayerNorm` | 3 | 55.920 |
| `LayerNormV4_51_LayerNormV3/AddLayerNorm` | 3 | 55.900 |
| `LayerNormV4_35_LayerNormV3/AddLayerNorm` | 3 | 55.900 |
| `LayerNormV4_11_LayerNormV3/AddLayerNorm` | 3 | 54.920 |
| `MatMulV2_63` | 3 | 54.820 |
| `LayerNormV4_54_LayerNormV3/AddLayerNorm` | 3 | 51.040 |
| `MatMulV2_3` | 3 | 50.560 |
| `StridedSliceV2_89` | 3 | 46.200 |
| `StridedSliceV2_21` | 3 | 46.140 |
| `StridedSliceV2_53` | 3 | 46.100 |
| `StridedSliceV2_93` | 3 | 46.100 |
| `StridedSliceV2_85` | 3 | 46.080 |
| `StridedSliceV2_45` | 3 | 45.920 |
| `StridedSliceV2_69` | 3 | 45.880 |
| `StridedSliceV2_61` | 3 | 45.860 |
| `StridedSliceV2_25` | 3 | 45.780 |
| `StridedSliceV2_33` | 3 | 45.760 |
| `StridedSliceV2_49` | 3 | 45.760 |
| `StridedSliceV2_65` | 3 | 45.720 |
| `StridedSliceV2_29` | 3 | 45.700 |
| `StridedSliceV2_9` | 3 | 45.480 |
| `StridedSliceV2_5` | 3 | 45.180 |
| `StridedSliceV2_17` | 3 | 45.140 |
| `StridedSliceV2_97` | 3 | 45.100 |
| `StridedSliceV2_73` | 3 | 45.020 |
| `StridedSliceV2_101` | 3 | 44.980 |
| `StridedSliceV2_41` | 3 | 44.960 |
| `StridedSliceV2_81` | 3 | 44.920 |
| `Gelu_12` | 3 | 44.840 |
| `StridedSliceV2_77` | 3 | 44.820 |
| `Gelu_2` | 3 | 44.800 |
| `Gelu_21` | 3 | 44.780 |
| `Gelu_8` | 3 | 44.740 |
| `Gelu_18` | 3 | 44.720 |
| `Gelu_16` | 3 | 44.700 |
| `Gelu_24` | 3 | 44.700 |
| `Gelu_10` | 3 | 44.660 |
| `Gelu_4` | 3 | 44.640 |
| `Gelu_23` | 3 | 44.560 |
| `Gelu_17` | 3 | 44.520 |
| `StridedSliceV2_13` | 3 | 44.480 |
| `StridedSliceV2_105` | 3 | 44.480 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | ND;ND;ND;ND -> ND` | 81 | 5031.280 |
| `StridedSliceD | "1,512,16,80" -> "1,512,16,40" | ND -> ND` | 324 | 4438.740 |
| `MatMulV2 | "512,1152;72,80,16,16;1280" -> "512,1280" | ND;FRACTAL_NZ;ND -> ND` | 243 | 3868.480 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 162 | 2236.920 |
| `MatMulV2 | "512,4352;272,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2218.980 |
| `Transpose | "512,16,80;3" -> "16,512,80" | ND;ND -> ND` | 243 | 2198.500 |
| `MatMulV2 | "512,1152;72,272,16,16;4352" -> "512,4352" | ND;FRACTAL_NZ;ND -> ND` | 81 | 2197.140 |
| `Mul | "1,512,16,80;1,512,1,80" -> "1,512,16,80" | ND;ND -> ND` | 324 | 1839.580 |
| `MatMulV2 | "512,1280;80,72,16,16;1152" -> "512,1152" | ND;FRACTAL_NZ;ND -> ND` | 81 | 1728.440 |
| `Cast | "1,512,16,80" -> "1,512,16,80" | ND -> ND` | 162 | 1385.620 |
| `Add | "1,512,16,80;1,512,16,80" -> "1,512,16,80" | ND;ND -> ND` | 162 | 1267.040 |
| `Gelu | "1,512,4352" -> "1,512,4352" | ND -> ND` | 81 | 1230.120 |
| `Neg | "1,512,16,40" -> "1,512,16,40" | ND -> ND` | 162 | 1211.800 |
| `ConcatV2D | "1,512,16,40;1,512,16,40" -> "1,512,16,80" | ND;ND -> ND` | 162 | 951.760 |
| `Transpose | "16,512,80;3" -> "512,16,80" | ND;ND -> ND` | 81 | 724.140 |
| `ConcatV2D | "1,512,1280;1,512,1280;1,512,1280" -> "1,512,3840" | ND;ND;ND -> ND` | 81 | 614.200 |
| `SplitVD | "1,512,3840" -> "1,512,1280;1,512,1280;1,512,1280" | ND -> ND;ND;ND` | 81 | 508.460 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 3 | 41.240 |
| `Data | N/A -> N/A | N/A -> N/A` | 3 | 14.400 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;FRACTAL_NZ;ND` | 486 | 10013.040 |
| `ND` | 810 | 8774.740 |
| `ND;ND;ND;ND` | 243 | 7268.200 |
| `ND;ND` | 972 | 6981.020 |
| `ND;ND;ND` | 84 | 655.440 |
| `N/A` | 3 | 14.400 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_15` | 0 | 69.640 |
| `PromptFlashAttention_15` | 0 | 69.640 |
| `PromptFlashAttention_15` | 0 | 69.540 |
| `PromptFlashAttention_26` | 0 | 69.060 |
| `PromptFlashAttention_26` | 0 | 68.700 |
| `PromptFlashAttention_26` | 0 | 68.260 |
| `PromptFlashAttention_16` | 0 | 67.480 |
| `PromptFlashAttention_19` | 0 | 67.180 |
| `PromptFlashAttention_16` | 0 | 67.120 |
| `PromptFlashAttention_19` | 0 | 66.740 |
| `PromptFlashAttention_2` | 0 | 66.680 |
| `PromptFlashAttention` | 0 | 66.440 |
| `PromptFlashAttention_2` | 0 | 66.100 |
| `PromptFlashAttention_16` | 0 | 66.020 |
| `PromptFlashAttention_17` | 0 | 65.860 |
| `PromptFlashAttention_19` | 0 | 65.740 |
| `PromptFlashAttention_2` | 0 | 65.440 |
| `PromptFlashAttention` | 0 | 65.440 |
| `PromptFlashAttention_24` | 0 | 65.060 |
| `PromptFlashAttention` | 0 | 64.960 |
| `PromptFlashAttention_10` | 0 | 64.940 |
| `PromptFlashAttention_18` | 0 | 64.740 |
| `PromptFlashAttention_9` | 0 | 64.480 |
| `PromptFlashAttention_10` | 0 | 64.280 |
| `PromptFlashAttention_10` | 0 | 64.280 |
| `PromptFlashAttention_14` | 0 | 64.220 |
| `PromptFlashAttention_1` | 0 | 64.000 |
| `PromptFlashAttention_9` | 0 | 63.960 |
| `PromptFlashAttention_11` | 0 | 63.840 |
| `PromptFlashAttention_17` | 0 | 63.600 |
| `PromptFlashAttention_3` | 0 | 63.560 |
| `PromptFlashAttention_9` | 0 | 63.540 |
| `PromptFlashAttention_17` | 0 | 63.340 |
| `PromptFlashAttention_23` | 0 | 62.600 |
| `PromptFlashAttention_14` | 0 | 62.520 |
| `PromptFlashAttention_14` | 0 | 62.320 |
| `PromptFlashAttention_3` | 0 | 62.280 |
| `PromptFlashAttention_24` | 0 | 62.180 |
| `PromptFlashAttention_23` | 0 | 62.120 |
| `PromptFlashAttention_11` | 0 | 62.020 |
| `PromptFlashAttention_3` | 0 | 61.980 |
| `PromptFlashAttention_24` | 0 | 61.720 |
| `PromptFlashAttention_1` | 0 | 61.200 |
| `PromptFlashAttention_23` | 0 | 61.180 |
| `PromptFlashAttention_11` | 0 | 61.160 |
| `PromptFlashAttention_18` | 0 | 60.800 |
| `PromptFlashAttention_1` | 0 | 60.720 |
| `PromptFlashAttention_7` | 0 | 60.500 |
| `PromptFlashAttention_4` | 0 | 60.360 |
| `PromptFlashAttention_7` | 0 | 60.080 |
| `PromptFlashAttention_8` | 0 | 59.840 |
| `PromptFlashAttention_5` | 0 | 59.640 |
| `PromptFlashAttention_18` | 0 | 59.640 |
| `PromptFlashAttention_7` | 0 | 59.540 |
| `PromptFlashAttention_25` | 0 | 59.340 |
| `PromptFlashAttention_13` | 0 | 59.280 |
| `PromptFlashAttention_22` | 0 | 59.160 |
| `PromptFlashAttention_20` | 0 | 59.120 |
| `PromptFlashAttention_8` | 0 | 59.100 |
| `PromptFlashAttention_21` | 0 | 59.100 |
| `PromptFlashAttention_20` | 0 | 59.020 |
| `PromptFlashAttention_4` | 0 | 58.980 |
| `PromptFlashAttention_8` | 0 | 58.920 |
| `PromptFlashAttention_12` | 0 | 58.800 |
| `PromptFlashAttention_20` | 0 | 58.720 |
| `PromptFlashAttention_22` | 0 | 58.700 |
| `PromptFlashAttention_4` | 0 | 58.540 |
| `PromptFlashAttention_25` | 0 | 58.440 |
| `PromptFlashAttention_5` | 0 | 58.360 |
| `PromptFlashAttention_6` | 0 | 58.280 |
| `PromptFlashAttention_13` | 0 | 58.000 |
| `PromptFlashAttention_25` | 0 | 57.920 |
| `PromptFlashAttention_6` | 0 | 57.720 |
| `PromptFlashAttention_6` | 0 | 57.660 |
| `PromptFlashAttention_12` | 0 | 57.620 |
| `PromptFlashAttention_13` | 0 | 57.620 |
| `PromptFlashAttention_5` | 0 | 57.600 |
| `PromptFlashAttention_21` | 0 | 57.560 |
| `PromptFlashAttention_12` | 0 | 57.160 |
| `PromptFlashAttention_22` | 0 | 56.780 |
| `PromptFlashAttention_21` | 0 | 55.500 |
| `MatMulV2_23` | 0 | 29.460 |
| `MatMulV2_4` | 0 | 29.160 |
| `MatMulV2_23` | 0 | 28.980 |
| `MatMulV2_142` | 0 | 28.940 |
| `MatMulV2_4` | 0 | 28.860 |
| `MatMulV2_142` | 0 | 28.740 |
| `MatMulV2_137` | 0 | 28.640 |
| `MatMulV2_11` | 0 | 28.600 |
| `MatMulV2_124` | 0 | 28.580 |
| `MatMulV2_47` | 0 | 28.560 |
| `MatMulV2_148` | 0 | 28.440 |
| `MatMulV2_47` | 0 | 28.320 |
| `MatMulV2_10` | 0 | 28.240 |
| `MatMulV2_41` | 0 | 28.160 |
| `MatMulV2_83` | 0 | 28.140 |
| `MatMulV2_160` | 0 | 28.140 |
| `MatMulV2_59` | 0 | 28.140 |
| `MatMulV2_59` | 0 | 28.100 |
| `MatMulV2_65` | 0 | 28.080 |
| `MatMulV2_88` | 0 | 28.060 |
| `MatMulV2_101` | 0 | 28.020 |
| `MatMulV2_137` | 0 | 28.020 |
| `MatMulV2_5` | 0 | 28.000 |
| `MatMulV2_130` | 0 | 27.980 |
| `MatMulV2_41` | 0 | 27.960 |
| `MatMulV2_53` | 0 | 27.960 |
| `MatMulV2_118` | 0 | 27.940 |
| `MatMulV2_95` | 0 | 27.940 |
| `MatMulV2_143` | 0 | 27.940 |
| `MatMulV2_4` | 0 | 27.920 |
| `MatMulV2_23` | 0 | 27.900 |
| `MatMulV2_64` | 0 | 27.900 |
| `MatMulV2_154` | 0 | 27.880 |
| `MatMulV2_29` | 0 | 27.880 |
| `MatMulV2_65` | 0 | 27.820 |
| `MatMulV2_89` | 0 | 27.820 |
| `MatMulV2_95` | 0 | 27.800 |
| `MatMulV2_16` | 0 | 27.780 |
| `MatMulV2_40` | 0 | 27.780 |
| `MatMulV2_124` | 0 | 27.780 |
| `MatMulV2_70` | 0 | 27.780 |
| `MatMulV2_35` | 0 | 27.760 |
| `MatMulV2_70` | 0 | 27.760 |
| `MatMulV2_101` | 0 | 27.760 |
| `MatMulV2_142` | 0 | 27.740 |
| `MatMulV2_155` | 0 | 27.740 |
| `MatMulV2_41` | 0 | 27.740 |
| `MatMulV2_130` | 0 | 27.740 |
| `MatMulV2_148` | 0 | 27.740 |
| `MatMulV2_154` | 0 | 27.740 |
| `MatMulV2_149` | 0 | 27.720 |
| `MatMulV2_5` | 0 | 27.720 |
| `MatMulV2_46` | 0 | 27.700 |
| `MatMulV2_125` | 0 | 27.700 |
| `MatMulV2_137` | 0 | 27.680 |
| `MatMulV2_88` | 0 | 27.680 |
| `MatMulV2_161` | 0 | 27.680 |
| `MatMulV2_64` | 0 | 27.660 |
| `MatMulV2_161` | 0 | 27.660 |
| `MatMulV2_71` | 0 | 27.640 |
| `MatMulV2_155` | 0 | 27.620 |
| `MatMulV2_71` | 0 | 27.620 |
| `MatMulV2_119` | 0 | 27.620 |
| `MatMulV2_131` | 0 | 27.620 |
| `MatMulV2_95` | 0 | 27.600 |
| `MatMulV2_17` | 0 | 27.600 |
| `MatMulV2_77` | 0 | 27.600 |
| `MatMulV2_83` | 0 | 27.600 |
| `MatMulV2_100` | 0 | 27.600 |
| `MatMulV2_119` | 0 | 27.600 |
| `MatMulV2_136` | 0 | 27.600 |
| `MatMulV2_5` | 0 | 27.560 |
| `MatMulV2_52` | 0 | 27.560 |
| `MatMulV2_130` | 0 | 27.560 |
| `MatMulV2_107` | 0 | 27.540 |
| `MatMulV2_143` | 0 | 27.540 |
| `MatMulV2_40` | 0 | 27.540 |
| `MatMulV2_58` | 0 | 27.540 |
| `MatMulV2_76` | 0 | 27.540 |
| `MatMulV2_59` | 0 | 27.500 |
| `MatMulV2_113` | 0 | 27.500 |
| `MatMulV2_149` | 0 | 27.500 |
| `MatMulV2_131` | 0 | 27.480 |
| `MatMulV2_89` | 0 | 27.480 |
| `MatMulV2_107` | 0 | 27.440 |
| `MatMulV2_71` | 0 | 27.420 |
| `MatMulV2_136` | 0 | 27.380 |
| `MatMulV2_125` | 0 | 27.360 |
| `MatMulV2_35` | 0 | 27.360 |
| `MatMulV2_16` | 0 | 27.360 |
| `MatMulV2_40` | 0 | 27.360 |
| `MatMulV2_94` | 0 | 27.360 |
| `MatMulV2_17` | 0 | 27.340 |
| `MatMulV2_113` | 0 | 27.320 |
| `MatMulV2_16` | 0 | 27.300 |
| `MatMulV2_160` | 0 | 27.260 |
| `MatMulV2_125` | 0 | 27.240 |
| `MatMulV2_76` | 0 | 27.160 |
| `MatMulV2_106` | 0 | 27.160 |
| `MatMulV2_28` | 0 | 27.140 |
| `MatMulV2_148` | 0 | 27.120 |
| `MatMulV2_10` | 0 | 27.120 |
| `MatMulV2_11` | 0 | 27.120 |
| `MatMulV2_46` | 0 | 27.120 |
| `MatMulV2_83` | 0 | 27.120 |
| `MatMulV2_119` | 0 | 27.100 |
| `MatMulV2_161` | 0 | 27.100 |
| `MatMulV2_155` | 0 | 27.060 |
| `MatMulV2_82` | 0 | 27.020 |
| `MatMulV2_94` | 0 | 26.980 |
| `MatMulV2_52` | 0 | 26.980 |
| `MatMulV2_52` | 0 | 26.880 |
| `MatMulV2_77` | 0 | 26.860 |
| `MatMulV2_35` | 0 | 26.840 |
| `MatMulV2_106` | 0 | 26.840 |
| `MatMulV2_136` | 0 | 26.820 |
| `MatMulV2_88` | 0 | 26.800 |
| `MatMulV2_76` | 0 | 26.800 |
| `MatMulV2_22` | 0 | 26.800 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `cache_compiler inference` | 3 | 14326.310 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step1` | 1 | 12871.860 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step2` | 1 | 12455.430 |
| `paddleocr_vl.vision_matmul_lab.B1.S512.I4352.fractal_nz.weights.prompt_flash_attention.separate_manual.torchair.active.step3` | 1 | 12437.060 |
| `TorchDynamo Cache Lookup` | 3 | 11397.160 |
| `Torch-Compiled Region: 0/0` | 3 | 3840.200 |
| `TorchNpuGraphBase::Run` | 3 | 2820.160 |
| `RefreshAtTensorFromGeTensor` | 3 | 1192.870 |
| `aten::empty` | 3 | 589.040 |
| `ExecuteGraph` | 3 | 519.930 |
| `AssembleInputs` | 3 | 405.970 |
| `AssembleOutputs` | 3 | 305.480 |
| `aten::set_` | 3 | 290.570 |
| `empty_tensor` | 3 | 287.830 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 198519.260 |
| `aclrtSynchronizeDeviceWithTimeout` | 4 | 31929.750 |
| `launch` | 868 | 13881.770 |
| `InputCopy` | 3 | 153.310 |
| `ModelExecute` | 3 | 50.260 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 26.980 |
| `step_info` | 6 | 13.900 |
| `OutputCopy` | 3 | 0.910 |

