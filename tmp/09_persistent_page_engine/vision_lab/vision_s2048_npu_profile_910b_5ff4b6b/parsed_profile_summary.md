# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_profiles/vision_s2048_npu_profile_910b_5ff4b6b`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_profiles/vision_s2048_npu_profile_910b_5ff4b6b/liteserver-c001-4_614469_20260729105827092_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `156338.220 us`
- `Free`: `10797.080 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `6505.000 us`
- `Stage`: `167135.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention` | 135 | 38198.040 |
| `MatMulV3` | 270 | 26557.440 |
| `StridedSliceD` | 675 | 18688.980 |
| `Transpose` | 540 | 18630.580 |
| `MatMulV2` | 540 | 13030.280 |
| `AddLayerNorm` | 270 | 7758.640 |
| `PadV3` | 405 | 6645.260 |
| `ConcatV2D` | 405 | 6003.500 |
| `Gelu` | 135 | 5148.320 |
| `Mul` | 540 | 4762.380 |
| `Add` | 270 | 3109.080 |
| `Cast` | 270 | 2755.360 |
| `Neg` | 270 | 2272.400 |
| `SplitVD` | 135 | 2210.200 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 405 | 397.480 |
| `LayerNormV3` | 5 | 145.200 |
| `Data` | 5 | 25.080 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_26` | 5 | 1453.460 |
| `PromptFlashAttention_15` | 5 | 1444.640 |
| `PromptFlashAttention_25` | 5 | 1436.240 |
| `PromptFlashAttention_9` | 5 | 1434.880 |
| `PromptFlashAttention_7` | 5 | 1432.780 |
| `PromptFlashAttention_6` | 5 | 1432.780 |
| `PromptFlashAttention_14` | 5 | 1431.420 |
| `PromptFlashAttention_19` | 5 | 1429.600 |
| `PromptFlashAttention_1` | 5 | 1428.080 |
| `PromptFlashAttention_12` | 5 | 1424.760 |
| `PromptFlashAttention_24` | 5 | 1422.880 |
| `PromptFlashAttention` | 5 | 1422.720 |
| `PromptFlashAttention_8` | 5 | 1422.500 |
| `PromptFlashAttention_16` | 5 | 1418.420 |
| `PromptFlashAttention_13` | 5 | 1415.140 |
| `PromptFlashAttention_17` | 5 | 1413.460 |
| `PromptFlashAttention_18` | 5 | 1405.160 |
| `PromptFlashAttention_11` | 5 | 1404.080 |
| `PromptFlashAttention_23` | 5 | 1403.920 |
| `PromptFlashAttention_21` | 5 | 1402.580 |
| `PromptFlashAttention_20` | 5 | 1399.660 |
| `PromptFlashAttention_2` | 5 | 1397.860 |
| `PromptFlashAttention_10` | 5 | 1386.220 |
| `PromptFlashAttention_22` | 5 | 1385.320 |
| `PromptFlashAttention_4` | 5 | 1385.220 |
| `PromptFlashAttention_3` | 5 | 1383.440 |
| `PromptFlashAttention_5` | 5 | 1380.820 |
| `MatMulV2_52_to_v3` | 5 | 511.760 |
| `MatMulV2_118_to_v3` | 5 | 500.760 |
| `MatMulV2_16_to_v3` | 5 | 497.780 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention | "1,16,2048,80;1,16,2048,80;1,16,2048,80;1,1,2048,2048" -> "1,16,2048,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 135 | 38198.040 |
| `StridedSliceD | "1,2048,16,72" -> "1,2048,16,36" | ND -> ND` | 540 | 14474.160 |
| `Transpose | "2048,16,72;3" -> "16,2048,72" | ND;ND -> ND` | 405 | 13483.920 |
| `MatMulV3 | "2048,1152;4304,1152;4304" -> "2048,4304" | ND;ND;ND -> ND` | 135 | 13328.240 |
| `MatMulV3 | "2048,4304;1152,4304;1152" -> "2048,1152" | ND;ND;ND -> ND` | 135 | 13229.200 |
| `MatMulV2 | "2048,1152;1152,1152;1152" -> "2048,1152" | ND;ND;ND -> ND` | 540 | 13030.280 |
| `AddLayerNorm | "1,2048,1152;1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1;1,2048,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 270 | 7758.640 |
| `PadV3 | "1,16,2048,72;8;" -> "1,16,2048,80" | NCHW;NCHW;NCHW -> NCHW` | 405 | 6645.260 |
| `Gelu | "1,2048,4304" -> "1,2048,4304" | ND -> ND` | 135 | 5148.320 |
| `Transpose | "16,2048,72;3" -> "2048,16,72" | ND;ND -> ND` | 135 | 5146.660 |
| `Mul | "1,2048,16,72;1,2048,1,72" -> "1,2048,16,72" | ND;ND -> ND` | 540 | 4762.380 |
| `ConcatV2D | "1,2048,16,36;1,2048,16,36" -> "1,2048,16,72" | ND;ND -> ND` | 270 | 4461.100 |
| `StridedSliceD | "1,16,2048,80" -> "1,16,2048,72" | NCHW -> NCHW` | 135 | 4214.820 |
| `Add | "1,2048,16,72;1,2048,16,72" -> "1,2048,16,72" | ND;ND -> ND` | 270 | 3109.080 |
| `Cast | "1,2048,16,72" -> "1,2048,16,72" | ND -> ND` | 270 | 2755.360 |
| `Neg | "1,2048,16,36" -> "1,2048,16,36" | ND -> ND` | 270 | 2272.400 |
| `SplitVD | "1,2048,3456" -> "1,2048,1152;1,2048,1152;1,2048,1152" | ND -> ND;ND;ND` | 135 | 2210.200 |
| `ConcatV2D | "1,2048,1152;1,2048,1152;1,2048,1152" -> "1,2048,3456" | ND;ND;ND -> ND` | 135 | 1542.400 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 405 | 397.480 |
| `LayerNormV3 | "1,2048,1152;1152;1152" -> "1,2048,1152;1,2048,1;1,2048,1" | ND;ND;ND -> ND;ND;ND` | 5 | 145.200 |
| `Data | N/A -> N/A | N/A -> N/A` | 5 | 25.080 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 950 | 41275.320 |
| `NCHW;NCHW;NCHW;NCHW` | 135 | 38198.040 |
| `ND;ND` | 1620 | 30963.140 |
| `ND` | 1350 | 26860.440 |
| `ND;ND;ND;ND` | 270 | 7758.640 |
| `NCHW;NCHW;NCHW` | 405 | 6645.260 |
| `NCHW` | 135 | 4214.820 |
| `N/A` | 410 | 422.560 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `PromptFlashAttention_26` | 0 | 294.440 |
| `PromptFlashAttention_26` | 0 | 292.940 |
| `PromptFlashAttention_26` | 0 | 292.400 |
| `PromptFlashAttention_15` | 0 | 290.660 |
| `PromptFlashAttention_15` | 0 | 290.540 |
| `PromptFlashAttention_15` | 0 | 290.280 |
| `PromptFlashAttention_9` | 0 | 290.220 |
| `PromptFlashAttention_25` | 0 | 289.080 |
| `PromptFlashAttention_7` | 0 | 288.880 |
| `PromptFlashAttention_7` | 0 | 288.620 |
| `PromptFlashAttention_25` | 0 | 288.440 |
| `PromptFlashAttention_9` | 0 | 287.860 |
| `PromptFlashAttention_9` | 0 | 287.860 |
| `PromptFlashAttention_15` | 0 | 287.820 |
| `PromptFlashAttention_25` | 0 | 287.360 |
| `PromptFlashAttention_6` | 0 | 287.300 |
| `PromptFlashAttention_26` | 0 | 287.040 |
| `PromptFlashAttention_1` | 0 | 286.880 |
| `PromptFlashAttention_14` | 0 | 286.740 |
| `PromptFlashAttention_26` | 0 | 286.640 |
| `PromptFlashAttention_19` | 0 | 286.620 |
| `PromptFlashAttention_6` | 0 | 286.560 |
| `PromptFlashAttention_6` | 0 | 286.480 |
| `PromptFlashAttention_13` | 0 | 286.460 |
| `PromptFlashAttention_24` | 0 | 286.400 |
| `PromptFlashAttention_19` | 0 | 286.360 |
| `PromptFlashAttention_6` | 0 | 286.220 |
| `PromptFlashAttention_6` | 0 | 286.220 |
| `PromptFlashAttention_14` | 0 | 286.220 |
| `PromptFlashAttention_14` | 0 | 286.180 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `Event::synchronize` | 5 | 154353.930 |
| `paddleocr_vl.vision_prefill.B1.S2048.active.step0` | 1 | 33804.180 |
| `paddleocr_vl.vision_prefill.B1.S2048.active.step1` | 1 | 33414.010 |
| `paddleocr_vl.vision_prefill.B1.S2048.active.step4` | 1 | 33411.550 |
| `paddleocr_vl.vision_prefill.B1.S2048.active.step3` | 1 | 33404.770 |
| `paddleocr_vl.vision_prefill.B1.S2048.active.step2` | 1 | 33392.260 |
| `cache_compiler inference` | 5 | 5913.360 |
| `TorchNpuGraphBase::Run` | 5 | 4034.210 |
| `Event::record` | 10 | 3419.840 |
| `RefreshAtTensorFromGeTensor` | 5 | 1653.810 |
| `Event::elapsed_time` | 10 | 1424.250 |
| `aten::empty` | 5 | 814.510 |
| `ExecuteGraph` | 5 | 768.630 |
| `AssembleInputs` | 5 | 569.000 |
| `AssembleOutputs` | 5 | 456.860 |
| `empty_tensor` | 5 | 397.430 |
| `aten::set_` | 5 | 393.790 |
| `record_event` | 10 | 368.040 |
| `destroy_event` | 10 | 254.400 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 231580.840 |
| `aclrtSynchronizeEvent` | 25 | 153616.420 |
| `launch` | 978 | 18973.970 |
| `InputCopy` | 5 | 265.160 |
| `aclrtRecordEvent` | 10 | 135.200 |
| `ModelExecute` | 5 | 85.560 |
| `aclrtLaunchKernelWithHostArgs` | 5 | 61.740 |
| `aclrtCreateEventExWithFlag` | 10 | 48.890 |
| `step_info` | 10 | 37.640 |
| `aclrtDestroyEvent` | 10 | 24.220 |
| `aclrtSynchronizeDeviceWithTimeout` | 1 | 22.870 |
| `OutputCopy` | 5 | 1.670 |
