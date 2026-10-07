# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_profile_scaling_b1_b16_b32_b64_k4096_b0f0cde_20260810/b1/profile_increfa_pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_profile_scaling_b1_b16_b32_b64_k4096_b0f0cde_20260810/b1/profile_increfa_pipe/liteserver-c001-4_3296459_20260810173732293_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `1879.540 us`
- `Free`: `95.960 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2812.750 us`
- `Stage`: `1975.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 146 | 1911.480 |
| `IncreFlashAttention` | 48 | 920.680 |
| `MatMulV2` | 48 | 266.080 |
| `InplaceAddRmsNorm` | 96 | 163.280 |
| `Scatter` | 96 | 121.360 |
| `ApplyRotaryPosEmb` | 48 | 119.600 |
| `SplitVD` | 100 | 119.420 |
| `AutomaticBufferFusionOp` | 50 | 89.840 |
| `GatherV2` | 28 | 67.360 |
| `Less` | 48 | 65.440 |
| `ArgMaxV2` | 2 | 38.180 |
| `Cast` | 8 | 29.020 |
| `Range` | 2 | 19.840 |
| `ConcatV2D` | 6 | 11.800 |
| `RmsNorm` | 2 | 11.180 |
| `Data` | 2 | 10.500 |
| `Mul` | 6 | 8.160 |
| `Add` | 4 | 8.000 |
| `LessEqual` | 2 | 7.880 |
| `Cos` | 2 | 3.540 |
| `Sin` | 2 | 3.520 |
| `BroadcastTo` | 2 | 2.420 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 2 | 410.260 |
| `IncreFlashAttention` | 2 | 65.300 |
| `MatMul_19` | 2 | 44.080 |
| `MatMul_16` | 2 | 43.800 |
| `IncreFlashAttention_9` | 2 | 42.860 |
| `IncreFlashAttention_10` | 2 | 40.560 |
| `IncreFlashAttention_1` | 2 | 40.500 |
| `IncreFlashAttention_2` | 2 | 40.380 |
| `IncreFlashAttention_18` | 2 | 40.300 |
| `IncreFlashAttention_16` | 2 | 39.660 |
| `IncreFlashAttention_11` | 2 | 39.400 |
| `IncreFlashAttention_13` | 2 | 39.120 |
| `IncreFlashAttention_8` | 2 | 38.880 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 38.180 |
| `IncreFlashAttention_7` | 2 | 37.260 |
| `IncreFlashAttention_19` | 2 | 37.260 |
| `IncreFlashAttention_21` | 2 | 36.980 |
| `IncreFlashAttention_15` | 2 | 36.760 |
| `IncreFlashAttention_12` | 2 | 36.700 |
| `IncreFlashAttention_20` | 2 | 36.660 |
| `MatMul_1` | 2 | 35.540 |
| `IncreFlashAttention_17` | 2 | 35.500 |
| `IncreFlashAttention_5` | 2 | 35.480 |
| `IncreFlashAttention_6` | 2 | 35.240 |
| `IncreFlashAttention_3` | 2 | 34.960 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 2 | 410.260 |
| `MatMul_19` | 2 | 44.080 |
| `MatMul_16` | 2 | 43.800 |
| `MatMul_1` | 2 | 35.540 |
| `MatMul_67` | 2 | 33.220 |
| `MatMul_58` | 2 | 33.060 |
| `MatMul_22` | 2 | 32.820 |
| `MatMul_4` | 2 | 32.060 |
| `MatMul_13` | 2 | 31.920 |
| `MatMul_70` | 2 | 31.780 |
| `MatMul_25` | 2 | 31.660 |
| `MatMul_7` | 2 | 31.640 |
| `MatMul_46` | 2 | 31.640 |
| `MatMul_64` | 2 | 31.500 |
| `MatMul_28` | 2 | 31.480 |
| `MatMul_52` | 2 | 31.080 |
| `MatMul_61` | 2 | 31.060 |
| `MatMul_10` | 2 | 30.980 |
| `MatMul_37` | 2 | 30.920 |
| `MatMul_31` | 2 | 30.780 |
| `MatMul_40` | 2 | 30.320 |
| `MatMul_34` | 2 | 29.920 |
| `MatMul_49` | 2 | 29.820 |
| `MatMul_55` | 2 | 29.580 |
| `MatMul_43` | 2 | 29.440 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,896;56,608,16,16" -> "1,9728" | ND;FRACTAL_NZ -> ND` | 48 | 780.100 |
| `MatMul | "1,4864;304,56,16,16" -> "1,896" | ND;FRACTAL_NZ -> ND` | 48 | 487.620 |
| `MatMul | "1,896;56,9496,16,16" -> "1,151936" | ND;FRACTAL_NZ -> ND` | 2 | 410.260 |
| `MatMulV2 | "1,896;56,72,16,16;1152" -> "1,1152" | ND;FRACTAL_NZ;ND -> ND` | 48 | 266.080 |
| `MatMul | "1,896;56,56,16,16" -> "1,896" | ND;FRACTAL_NZ -> ND` | 48 | 233.500 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 1 | 230.740 |
| `MatMul_72` | 1 | 179.520 |
| `IncreFlashAttention` | 1 | 33.400 |
| `IncreFlashAttention` | 1 | 31.900 |
| `MatMul_16` | 1 | 31.880 |
| `MatMul_19` | 1 | 29.920 |
| `IncreFlashAttention_9` | 1 | 21.560 |
| `IncreFlashAttention_9` | 1 | 21.300 |
| `IncreFlashAttention_1` | 1 | 21.020 |
| `IncreFlashAttention_2` | 1 | 20.620 |
| `IncreFlashAttention_10` | 1 | 20.420 |
| `IncreFlashAttention_18` | 1 | 20.300 |
| `IncreFlashAttention_10` | 1 | 20.140 |
| `IncreFlashAttention_16` | 1 | 20.000 |
| `IncreFlashAttention_18` | 1 | 20.000 |
| `MatMul_13` | 1 | 19.880 |
| `IncreFlashAttention_11` | 1 | 19.800 |
| `IncreFlashAttention_2` | 1 | 19.760 |
| `IncreFlashAttention_13` | 1 | 19.680 |
| `IncreFlashAttention_16` | 1 | 19.660 |
| `IncreFlashAttention_11` | 1 | 19.600 |
| `MatMul_17` | 1 | 19.580 |
| `IncreFlashAttention_8` | 1 | 19.560 |
| `MatMul_58` | 1 | 19.540 |
| `IncreFlashAttention_1` | 1 | 19.480 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.text_decode.increfa` | 1 | 1573.260 |
| `cache_compiler inference` | 2 | 987.380 |
| `TorchNpuGraphBase::Run` | 2 | 506.190 |
| `ExecuteGraph` | 2 | 226.390 |
| `aten::argmax` | 4 | 166.860 |
| `aten::to` | 2 | 142.370 |
| `aten::_to_copy` | 2 | 133.060 |
| `RefreshAtTensorFromGeTensor` | 2 | 96.060 |
| `aten::copy_` | 2 | 95.300 |
| `aten::empty` | 6 | 82.000 |
| `AssembleInputs` | 2 | 73.790 |
| `empty_tensor` | 6 | 53.170 |
| `aten::add_` | 2 | 52.260 |
| `aclnnArgMax` | 2 | 40.800 |
| `aclnnInplaceCopy` | 2 | 36.730 |
| `aten::select` | 2 | 29.130 |
| `aten::set_` | 2 | 25.350 |
| `AssembleOutputs` | 2 | 18.590 |
| `aten::as_strided` | 2 | 13.510 |
| `aclnnInplaceAdds` | 2 | 13.390 |
| `aten::reshape` | 2 | 12.680 |
| `aten::item` | 2 | 8.230 |
| `aten::view` | 2 | 7.890 |
| `aten::_local_scalar_dense` | 2 | 3.810 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 73973.620 |
| `launch` | 379 | 5902.710 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 3079.080 |
| `InputCopy` | 2 | 151.990 |
| `aclrtLaunchKernelWithHostArgs` | 10 | 104.190 |
| `aclnnInplaceCopy` | 2 | 56.860 |
| `ModelExecute` | 2 | 35.410 |
| `aclnnArgMax` | 2 | 31.030 |
| `aclnnInplaceAdds` | 2 | 19.000 |
| `step_info` | 4 | 13.750 |
| `aclrtGetStreamAttribute` | 6 | 3.820 |
| `OutputCopy` | 2 | 1.170 |


### Trace Events
| name | count | total_us |
|---|---:|---:|
| `Model@ModelLoad` | 1 | 73973.620 |
| `Node@launch` | 379 | 5902.710 |
| `ProfilerStep#0` | 1 | 4858.150 |
| `NOTIFY_WAIT` | 2 | 4026.980 |
| `Computing` | 748 | 3998.580 |
| `AscendCL@aclrtSynchronizeDeviceWithTimeout` | 2 | 3079.080 |
| `Iteration 1` | 1 | 2204.440 |
| `Iteration 2` | 1 | 1888.020 |
| `mineru.text_decode.increfa` | 1 | 1573.260 |
| `cache_compiler inference` | 2 | 987.380 |
| `TorchNpuGraphBase::Run` | 2 | 506.190 |
| `MatMul_72` | 2 | 410.260 |
| `ExecuteGraph` | 2 | 226.390 |
| `Free` | 748 | 218.020 |
| `aten::argmax` | 4 | 166.860 |
| `Model@InputCopy` | 2 | 151.990 |
| `aten::to` | 2 | 142.370 |
| `aten::_to_copy` | 2 | 133.060 |
| `AscendCL@aclrtLaunchKernelWithHostArgs` | 10 | 104.190 |
| `RefreshAtTensorFromGeTensor` | 2 | 96.060 |
| `aten::copy_` | 2 | 95.300 |
| `Dequeue@aclnnInplaceCopy` | 2 | 92.310 |
| `aten::empty` | 6 | 82.000 |
| `AssembleInputs` | 2 | 73.790 |
| `IncreFlashAttention` | 2 | 65.300 |

