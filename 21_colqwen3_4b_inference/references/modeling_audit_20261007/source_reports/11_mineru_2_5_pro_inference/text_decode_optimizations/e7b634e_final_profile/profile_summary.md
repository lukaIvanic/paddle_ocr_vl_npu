# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_optimizations/e7b634e_final_profile/profile_increfa_pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_optimizations/e7b634e_final_profile/profile_increfa_pipe/liteserver-c001-4_3001424_20260805183720182_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `1950.700 us`
- `Free`: `95.800 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3245.750 us`
- `Stage`: `2046.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 146 | 1896.340 |
| `IncreFlashAttention` | 48 | 906.860 |
| `RotaryMul` | 96 | 317.680 |
| `MatMulV2` | 48 | 251.260 |
| `InplaceAddRmsNorm` | 96 | 164.980 |
| `Scatter` | 96 | 122.100 |
| `SplitVD` | 100 | 118.640 |
| `AutomaticBufferFusionOp` | 50 | 94.060 |
| `Less` | 48 | 67.860 |
| `GatherV2` | 28 | 63.940 |
| `ArgMaxV2` | 2 | 38.260 |
| `Cast` | 8 | 27.800 |
| `Range` | 2 | 18.360 |
| `ConcatV2D` | 6 | 10.760 |
| `RmsNorm` | 2 | 10.360 |
| `Data` | 2 | 9.960 |
| `Add` | 4 | 8.520 |
| `Mul` | 6 | 7.600 |
| `LessEqual` | 2 | 7.440 |
| `Sin` | 2 | 3.700 |
| `Cos` | 2 | 3.560 |
| `BroadcastTo` | 2 | 2.520 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 2 | 408.660 |
| `IncreFlashAttention` | 2 | 63.020 |
| `MatMul_19` | 2 | 43.280 |
| `MatMul_16` | 2 | 42.700 |
| `IncreFlashAttention_22` | 2 | 42.660 |
| `IncreFlashAttention_15` | 2 | 40.740 |
| `IncreFlashAttention_9` | 2 | 40.420 |
| `IncreFlashAttention_1` | 2 | 39.400 |
| `IncreFlashAttention_21` | 2 | 38.380 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 38.260 |
| `IncreFlashAttention_12` | 2 | 37.780 |
| `IncreFlashAttention_23` | 2 | 37.640 |
| `IncreFlashAttention_6` | 2 | 37.500 |
| `IncreFlashAttention_5` | 2 | 37.300 |
| `IncreFlashAttention_18` | 2 | 37.240 |
| `IncreFlashAttention_10` | 2 | 36.680 |
| `IncreFlashAttention_11` | 2 | 35.780 |
| `IncreFlashAttention_4` | 2 | 35.760 |
| `IncreFlashAttention_20` | 2 | 35.520 |
| `IncreFlashAttention_7` | 2 | 35.000 |
| `IncreFlashAttention_17` | 2 | 34.940 |
| `IncreFlashAttention_19` | 2 | 34.860 |
| `MatMul_1` | 2 | 34.700 |
| `IncreFlashAttention_16` | 2 | 34.620 |
| `IncreFlashAttention_3` | 2 | 34.520 |
| `IncreFlashAttention_13` | 2 | 34.520 |
| `IncreFlashAttention_8` | 2 | 34.440 |
| `IncreFlashAttention_14` | 2 | 34.200 |
| `IncreFlashAttention_2` | 2 | 33.940 |
| `MatMul_13` | 2 | 33.740 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 2 | 408.660 |
| `MatMul_19` | 2 | 43.280 |
| `MatMul_16` | 2 | 42.700 |
| `MatMul_1` | 2 | 34.700 |
| `MatMul_13` | 2 | 33.740 |
| `MatMul_58` | 2 | 33.160 |
| `MatMul_10` | 2 | 32.940 |
| `MatMul_67` | 2 | 32.700 |
| `MatMul_31` | 2 | 31.880 |
| `MatMul_28` | 2 | 31.860 |
| `MatMul_70` | 2 | 31.600 |
| `MatMul_52` | 2 | 31.480 |
| `MatMul_4` | 2 | 31.180 |
| `MatMul_64` | 2 | 30.740 |
| `MatMul_7` | 2 | 30.740 |
| `MatMul_61` | 2 | 30.720 |
| `MatMul_37` | 2 | 30.660 |
| `MatMul_22` | 2 | 30.540 |
| `MatMul_34` | 2 | 30.460 |
| `MatMul_25` | 2 | 30.340 |
| `MatMul_49` | 2 | 30.100 |
| `MatMul_55` | 2 | 29.920 |
| `MatMul_43` | 2 | 29.560 |
| `MatMul_40` | 2 | 29.560 |
| `MatMul_46` | 2 | 29.520 |
| `MatMul_17` | 2 | 28.180 |
| `MatMul_14` | 2 | 24.400 |
| `MatMul_20` | 2 | 24.120 |
| `MatMul_2` | 2 | 21.400 |
| `MatMul_5` | 2 | 21.360 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,896;56,608,16,16" -> "1,9728" | ND;FRACTAL_NZ -> ND` | 48 | 774.080 |
| `MatMul | "1,4864;304,56,16,16" -> "1,896" | ND;FRACTAL_NZ -> ND` | 48 | 489.240 |
| `MatMul | "1,896;56,9496,16,16" -> "1,151936" | ND;FRACTAL_NZ -> ND` | 2 | 408.660 |
| `MatMulV2 | "1,896;56,72,16,16;1152" -> "1,1152" | ND;FRACTAL_NZ;ND -> ND` | 48 | 251.260 |
| `MatMul | "1,896;56,56,16,16" -> "1,896" | ND;FRACTAL_NZ -> ND` | 48 | 224.360 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_72` | 1 | 230.600 |
| `MatMul_72` | 1 | 178.060 |
| `IncreFlashAttention` | 1 | 32.120 |
| `IncreFlashAttention` | 1 | 30.900 |
| `MatMul_16` | 1 | 30.540 |
| `MatMul_19` | 1 | 30.000 |
| `MatMul_13` | 1 | 21.860 |
| `IncreFlashAttention_22` | 1 | 21.400 |
| `IncreFlashAttention_22` | 1 | 21.260 |
| `IncreFlashAttention_9` | 1 | 20.460 |
| `IncreFlashAttention_15` | 1 | 20.460 |
| `IncreFlashAttention_15` | 1 | 20.280 |
| `IncreFlashAttention_1` | 1 | 20.060 |
| `MatMul_31` | 1 | 19.960 |
| `IncreFlashAttention_9` | 1 | 19.960 |
| `MatMul_52` | 1 | 19.660 |
| `IncreFlashAttention_1` | 1 | 19.340 |
| `IncreFlashAttention_21` | 1 | 19.300 |
| `MatMul_70` | 1 | 19.300 |
| `MatMul_17` | 1 | 19.260 |
| `MatMul_58` | 1 | 19.200 |
| `MatMul_1` | 1 | 19.100 |
| `IncreFlashAttention_6` | 1 | 19.080 |
| `IncreFlashAttention_12` | 1 | 19.080 |
| `IncreFlashAttention_21` | 1 | 19.080 |
| `IncreFlashAttention_23` | 1 | 18.980 |
| `MatMul_22` | 1 | 18.860 |
| `IncreFlashAttention_18` | 1 | 18.740 |
| `MatMul_37` | 1 | 18.720 |
| `IncreFlashAttention_12` | 1 | 18.700 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.text_decode.increfa` | 1 | 2212.250 |
| `cache_compiler inference` | 2 | 1429.230 |
| `TorchNpuGraphBase::Run` | 2 | 680.970 |
| `ExecuteGraph` | 2 | 276.640 |
| `aten::argmax` | 4 | 230.920 |
| `aten::to` | 2 | 199.370 |
| `aten::_to_copy` | 2 | 186.860 |
| `aten::empty` | 6 | 143.950 |
| `RefreshAtTensorFromGeTensor` | 2 | 141.350 |
| `AssembleInputs` | 2 | 130.350 |
| `aten::copy_` | 2 | 108.740 |
| `empty_tensor` | 6 | 79.610 |
| `aten::add_` | 2 | 66.810 |
| `aten::select` | 2 | 42.460 |
| `aclnnArgMax` | 2 | 40.720 |
| `aclnnInplaceCopy` | 2 | 36.460 |
| `aten::set_` | 2 | 35.570 |
| `AssembleOutputs` | 2 | 28.710 |
| `aten::reshape` | 2 | 19.760 |
| `aten::as_strided` | 2 | 18.040 |
| `aclnnInplaceAdds` | 2 | 14.740 |
| `aten::item` | 2 | 13.190 |
| `aten::view` | 2 | 11.340 |
| `aten::_local_scalar_dense` | 2 | 6.210 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 75702.680 |
| `launch` | 403 | 6250.430 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 2881.030 |
| `InputCopy` | 2 | 184.190 |
| `aclrtLaunchKernelWithHostArgs` | 10 | 142.820 |
| `aclnnInplaceCopy` | 2 | 109.400 |
| `aclnnArgMax` | 2 | 61.440 |
| `ModelExecute` | 2 | 48.730 |
| `aclnnInplaceAdds` | 2 | 22.440 |
| `step_info` | 4 | 15.520 |
| `aclrtGetStreamAttribute` | 6 | 6.580 |
| `OutputCopy` | 2 | 1.320 |

