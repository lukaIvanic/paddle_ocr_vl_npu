# Static Visual Batched Encoder Profile

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_profiles/vision_s512_npu_profile_910b_7d1f778`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/09_persistent_page_engine_profiles/vision_s512_npu_profile_910b_7d1f778/liteserver-c001-4_612099_20260729104014408_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `77915.620 us`
- `Free`: `11569.320 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `7481.750 us`
- `Stage`: `89485.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV2` | 810 | 24806.520 |
| `StridedSliceD` | 675 | 10111.180 |
| `Transpose` | 540 | 10039.020 |
| `PromptFlashAttention` | 135 | 8549.100 |
| `PadV3` | 405 | 4613.360 |
| `AddLayerNorm` | 270 | 3657.700 |
| `ConcatV2D` | 405 | 3155.120 |
| `Mul` | 540 | 2885.740 |
| `Add` | 270 | 2365.200 |
| `Cast` | 270 | 2302.880 |
| `Gelu` | 135 | 2119.460 |
| `Neg` | 270 | 1941.300 |
| `SplitVD` | 135 | 880.680 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 405 | 392.320 |
| `LayerNormV3` | 5 | 69.700 |
| `Data` | 5 | 26.340 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_89` | 5 | 432.200 |
| `MatMulV2_131` | 5 | 430.200 |
| `MatMulV2_17` | 5 | 423.540 |
| `MatMulV2_71` | 5 | 423.520 |
| `MatMulV2_77` | 5 | 422.980 |
| `MatMulV2_5` | 5 | 422.740 |
| `MatMulV2_47` | 5 | 422.560 |
| `MatMulV2_161` | 5 | 422.020 |
| `MatMulV2_155` | 5 | 421.740 |
| `MatMulV2_143` | 5 | 420.040 |
| `MatMulV2_35` | 5 | 419.460 |
| `MatMulV2_113` | 5 | 419.300 |
| `MatMulV2_95` | 5 | 418.620 |
| `MatMulV2_23` | 5 | 418.520 |
| `MatMulV2_53` | 5 | 418.320 |
| `MatMulV2_149` | 5 | 418.060 |
| `MatMulV2_29` | 5 | 417.720 |
| `MatMulV2_65` | 5 | 417.680 |
| `MatMulV2_83` | 5 | 417.340 |
| `MatMulV2_101` | 5 | 416.740 |
| `MatMulV2_107` | 5 | 416.720 |
| `MatMulV2_41` | 5 | 416.520 |
| `MatMulV2_59` | 5 | 415.020 |
| `MatMulV2_137` | 5 | 413.100 |
| `MatMulV2_125` | 5 | 413.080 |
| `MatMulV2_119` | 5 | 413.040 |
| `MatMulV2_11` | 5 | 412.160 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0` | 405 | 392.320 |
| `PromptFlashAttention_17` | 5 | 346.180 |
| `PromptFlashAttention_24` | 5 | 341.220 |

### Shape/Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "512,4304;1152,4304;1152" -> "512,1152" | ND;ND;ND -> ND` | 135 | 11322.940 |
| `PromptFlashAttention | "1,16,512,80;1,16,512,80;1,16,512,80;1,1,512,512" -> "1,16,512,80" | NCHW;NCHW;NCHW;NCHW -> NCHW` | 135 | 8549.100 |
| `MatMulV2 | "512,1152;1152,1152;1152" -> "512,1152" | ND;ND;ND -> ND` | 540 | 7627.180 |
| `StridedSliceD | "1,512,16,72" -> "1,512,16,36" | ND -> ND` | 540 | 7482.940 |
| `Transpose | "512,16,72;3" -> "16,512,72" | ND;ND -> ND` | 405 | 7436.380 |
| `MatMulV2 | "512,1152;4304,1152;4304" -> "512,4304" | ND;ND;ND -> ND` | 135 | 5856.400 |
| `PadV3 | "1,16,512,72;8;" -> "1,16,512,80" | NCHW;NCHW;NCHW -> NCHW` | 405 | 4613.360 |
| `AddLayerNorm | "1,512,1152;1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1;1,512,1152" | ND;ND;ND;ND -> ND;ND;ND;ND` | 270 | 3657.700 |
| `Mul | "1,512,16,72;1,512,1,72" -> "1,512,16,72" | ND;ND -> ND` | 540 | 2885.740 |
| `StridedSliceD | "1,16,512,80" -> "1,16,512,72" | NCHW -> NCHW` | 135 | 2628.240 |
| `Transpose | "16,512,72;3" -> "512,16,72" | ND;ND -> ND` | 135 | 2602.640 |
| `Add | "1,512,16,72;1,512,16,72" -> "1,512,16,72" | ND;ND -> ND` | 270 | 2365.200 |
| `Cast | "1,512,16,72" -> "1,512,16,72" | ND -> ND` | 270 | 2302.880 |
| `ConcatV2D | "1,512,16,36;1,512,16,36" -> "1,512,16,72" | ND;ND -> ND` | 270 | 2152.760 |
| `Gelu | "1,512,4304" -> "1,512,4304" | ND -> ND` | 135 | 2119.460 |
| `Neg | "1,512,16,36" -> "1,512,16,36" | ND -> ND` | 270 | 1941.300 |
| `ConcatV2D | "1,512,1152;1,512,1152;1,512,1152" -> "1,512,3456" | ND;ND;ND -> ND` | 135 | 1002.360 |
| `SplitVD | "1,512,3456" -> "1,512,1152;1,512,1152;1,512,1152" | ND -> ND;ND;ND` | 135 | 880.680 |
| `te_memset_c382be8ff0fa928219b781ee53a60023bdbd28457e0da7af735c6c763c658e74__kernel0 | N/A -> N/A | N/A -> N/A` | 405 | 392.320 |
| `LayerNormV3 | "1,512,1152;1152;1152" -> "1,512,1152;1,512,1;1,512,1" | ND;ND;ND -> ND;ND;ND` | 5 | 69.700 |
| `Data | N/A -> N/A | N/A -> N/A` | 5 | 26.340 |

### Input Formats
| name | count | total_us |
|---|---:|---:|
| `ND;ND;ND` | 950 | 25878.580 |
| `ND;ND` | 1620 | 17442.720 |
| `ND` | 1350 | 14727.260 |
| `NCHW;NCHW;NCHW;NCHW` | 135 | 8549.100 |
| `NCHW;NCHW;NCHW` | 405 | 4613.360 |
| `ND;ND;ND;ND` | 270 | 3657.700 |
| `NCHW` | 135 | 2628.240 |
| `N/A` | 410 | 418.660 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMulV2_131` | 0 | 87.820 |
| `MatMulV2_131` | 0 | 87.760 |
| `MatMulV2_89` | 0 | 87.180 |
| `MatMulV2_89` | 0 | 87.180 |
| `MatMulV2_89` | 0 | 87.060 |
| `MatMulV2_71` | 0 | 86.760 |
| `MatMulV2_155` | 0 | 86.540 |
| `MatMulV2_161` | 0 | 86.420 |
| `MatMulV2_71` | 0 | 86.360 |
| `MatMulV2_89` | 0 | 86.340 |
| `MatMulV2_125` | 0 | 86.280 |
| `MatMulV2_47` | 0 | 85.880 |
| `MatMulV2_17` | 0 | 85.780 |
| `MatMulV2_53` | 0 | 85.680 |
| `MatMulV2_77` | 0 | 85.660 |
| `MatMulV2_17` | 0 | 85.660 |
| `MatMulV2_17` | 0 | 85.440 |
| `MatMulV2_5` | 0 | 85.360 |
| `MatMulV2_35` | 0 | 85.280 |
| `MatMulV2_47` | 0 | 85.280 |
| `MatMulV2_143` | 0 | 85.240 |
| `MatMulV2_5` | 0 | 85.220 |
| `MatMulV2_65` | 0 | 85.160 |
| `MatMulV2_131` | 0 | 85.160 |
| `MatMulV2_77` | 0 | 85.120 |
| `MatMulV2_113` | 0 | 85.120 |
| `MatMulV2_23` | 0 | 85.060 |
| `MatMulV2_113` | 0 | 85.060 |
| `MatMulV2_29` | 0 | 85.000 |
| `MatMulV2_143` | 0 | 84.980 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `Event::synchronize` | 5 | 75530.030 |
| `paddleocr_vl.vision_prefill.B1.S512.active.step0` | 1 | 18362.790 |
| `paddleocr_vl.vision_prefill.B1.S512.active.step1` | 1 | 17983.630 |
| `paddleocr_vl.vision_prefill.B1.S512.active.step3` | 1 | 17869.380 |
| `paddleocr_vl.vision_prefill.B1.S512.active.step2` | 1 | 17823.540 |
| `paddleocr_vl.vision_prefill.B1.S512.active.step4` | 1 | 17821.640 |
| `cache_compiler inference` | 5 | 6819.880 |
| `TorchNpuGraphBase::Run` | 5 | 4464.370 |
| `Event::record` | 10 | 3628.330 |
| `RefreshAtTensorFromGeTensor` | 5 | 1699.660 |
| `Event::elapsed_time` | 10 | 1443.180 |
| `ExecuteGraph` | 5 | 922.150 |
| `aten::empty` | 5 | 831.590 |
| `AssembleInputs` | 5 | 741.600 |
| `AssembleOutputs` | 5 | 445.610 |
| `aten::set_` | 5 | 413.540 |
| `empty_tensor` | 5 | 411.660 |
| `record_event` | 10 | 392.500 |
| `destroy_event` | 10 | 299.730 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 237979.120 |
| `aclrtSynchronizeEvent` | 25 | 74786.930 |
| `launch` | 978 | 19549.290 |
| `InputCopy` | 5 | 361.540 |
| `aclrtRecordEvent` | 10 | 185.300 |
| `ModelExecute` | 5 | 103.460 |
| `aclrtLaunchKernelWithHostArgs` | 5 | 68.020 |
| `aclrtCreateEventExWithFlag` | 10 | 64.300 |
| `step_info` | 10 | 39.970 |
| `aclrtDestroyEvent` | 10 | 28.650 |
| `aclrtSynchronizeDeviceWithTimeout` | 1 | 17.210 |
| `OutputCopy` | 5 | 3.110 |
