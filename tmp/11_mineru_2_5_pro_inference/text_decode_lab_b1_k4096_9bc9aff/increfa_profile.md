# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab_b1_k4096_9bc9aff/profile_increfa_pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab_b1_k4096_9bc9aff/profile_increfa_pipe/liteserver-c001-4_2895975_20260805180213858_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `3262.740 us`
- `Free`: `118.920 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4698.750 us`
- `Stage`: `3381.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 194 | 2260.840 |
| `IncreFlashAttention` | 48 | 912.980 |
| `GatherV2` | 580 | 703.700 |
| `MatMulV2` | 144 | 479.460 |
| `StridedSliceD` | 192 | 389.920 |
| `ConcatV2D` | 194 | 370.860 |
| `AutomaticBufferFusionOp` | 244 | 308.780 |
| `SplitVD` | 96 | 240.760 |
| `Mul` | 202 | 234.680 |
| `Cast` | 106 | 129.540 |
| `Square` | 98 | 115.460 |
| `Scatter` | 96 | 107.040 |
| `Neg` | 96 | 104.740 |
| `Less` | 48 | 63.400 |
| `ArgMaxV2` | 2 | 37.860 |
| `Range` | 2 | 17.360 |
| `Data` | 2 | 10.300 |
| `Add` | 4 | 8.680 |
| `LessEqual` | 2 | 7.200 |
| `Sin` | 2 | 3.440 |
| `Cos` | 2 | 3.420 |
| `BroadcastTo` | 2 | 2.520 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 416.140 |
| `IncreFlashAttention` | 2 | 59.780 |
| `IncreFlashAttention_6` | 2 | 42.540 |
| `IncreFlashAttention_5` | 2 | 40.740 |
| `IncreFlashAttention_10` | 2 | 40.360 |
| `IncreFlashAttention_1` | 2 | 40.080 |
| `IncreFlashAttention_9` | 2 | 39.920 |
| `IncreFlashAttention_21` | 2 | 39.640 |
| `IncreFlashAttention_12` | 2 | 39.180 |
| `IncreFlashAttention_2` | 2 | 39.160 |
| `IncreFlashAttention_4` | 2 | 37.980 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 37.860 |
| `IncreFlashAttention_13` | 2 | 37.820 |
| `IncreFlashAttention_18` | 2 | 37.280 |
| `IncreFlashAttention_14` | 2 | 37.220 |
| `IncreFlashAttention_22` | 2 | 36.440 |
| `IncreFlashAttention_11` | 2 | 36.420 |
| `IncreFlashAttention_8` | 2 | 36.200 |
| `IncreFlashAttention_17` | 2 | 36.100 |
| `IncreFlashAttention_16` | 2 | 36.040 |
| `IncreFlashAttention_7` | 2 | 35.060 |
| `IncreFlashAttention_23` | 2 | 34.320 |
| `IncreFlashAttention_19` | 2 | 34.040 |
| `IncreFlashAttention_3` | 2 | 33.740 |
| `IncreFlashAttention_15` | 2 | 32.880 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 416.140 |
| `MatMul_23` | 2 | 31.600 |
| `MatMul_25` | 2 | 27.940 |
| `MatMul_21` | 2 | 27.560 |
| `MatMul_3` | 2 | 27.540 |
| `MatMul_19` | 2 | 27.540 |
| `MatMul_22` | 2 | 27.500 |
| `MatMul_27` | 2 | 26.360 |
| `MatMul_26` | 2 | 25.940 |
| `MatMul_39` | 2 | 25.020 |
| `MatMul_11` | 2 | 24.880 |
| `MatMul_7` | 2 | 24.760 |
| `MatMul_71` | 2 | 24.760 |
| `MatMul_67` | 2 | 24.700 |
| `MatMul_55` | 2 | 24.700 |
| `MatMul_87` | 2 | 24.680 |
| `MatMul_43` | 2 | 24.660 |
| `MatMul_91` | 2 | 24.620 |
| `MatMul_31` | 2 | 24.560 |
| `MatMul_75` | 2 | 24.560 |
| `MatMul_59` | 2 | 24.400 |
| `MatMul_95` | 2 | 24.320 |
| `MatMul_47` | 2 | 24.080 |
| `MatMul_15` | 2 | 24.040 |
| `MatMul_1` | 2 | 23.860 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,896;4864,896" -> "1,4864" | ND;ND -> ND` | 96 | 1018.280 |
| `MatMul | "1,4864;896,4864;1,1,896" -> "1,896" | ND;ND;ND -> ND` | 48 | 599.360 |
| `MatMul | "1,896;151936,896" -> "1,151936" | ND;ND -> ND` | 2 | 416.140 |
| `MatMulV2 | "1,896;896,896;896" -> "1,896" | ND;ND;ND -> ND` | 48 | 263.740 |
| `MatMul | "1,896;896,896;1,1,896" -> "1,896" | ND;ND;ND -> ND` | 48 | 227.060 |
| `MatMulV2 | "1,896;128,896;128" -> "1,128" | ND;ND;ND -> ND` | 96 | 215.720 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 1 | 208.800 |
| `MatMul_96` | 1 | 207.340 |
| `IncreFlashAttention` | 1 | 30.160 |
| `IncreFlashAttention` | 1 | 29.620 |
| `IncreFlashAttention_6` | 1 | 21.300 |
| `IncreFlashAttention_6` | 1 | 21.240 |
| `IncreFlashAttention_5` | 1 | 20.440 |
| `IncreFlashAttention_5` | 1 | 20.300 |
| `IncreFlashAttention_1` | 1 | 20.260 |
| `IncreFlashAttention_10` | 1 | 20.220 |
| `IncreFlashAttention_10` | 1 | 20.140 |
| `IncreFlashAttention_9` | 1 | 20.060 |
| `IncreFlashAttention_2` | 1 | 19.940 |
| `IncreFlashAttention_9` | 1 | 19.860 |
| `IncreFlashAttention_21` | 1 | 19.860 |
| `IncreFlashAttention_1` | 1 | 19.820 |
| `IncreFlashAttention_21` | 1 | 19.780 |
| `IncreFlashAttention_12` | 1 | 19.640 |
| `IncreFlashAttention_4` | 1 | 19.640 |
| `IncreFlashAttention_12` | 1 | 19.540 |
| `IncreFlashAttention_2` | 1 | 19.220 |
| `MatMul_23` | 1 | 19.220 |
| `IncreFlashAttention_13` | 1 | 18.960 |
| `IncreFlashAttention_13` | 1 | 18.860 |
| `IncreFlashAttention_14` | 1 | 18.660 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.text_decode.increfa` | 1 | 3157.930 |
| `cache_compiler inference` | 2 | 2389.630 |
| `Torch-Compiled Region: 0/1` | 2 | 1107.230 |
| `TorchNpuGraphBase::Run` | 2 | 616.020 |
| `TorchDynamo Cache Lookup` | 2 | 524.370 |
| `ExecuteGraph` | 2 | 250.970 |
| `aten::argmax` | 4 | 226.230 |
| `aten::to` | 2 | 206.830 |
| `aten::_to_copy` | 2 | 194.240 |
| `aten::empty` | 6 | 153.900 |
| `RefreshAtTensorFromGeTensor` | 2 | 124.930 |
| `aten::copy_` | 2 | 120.470 |
| `AssembleInputs` | 2 | 100.430 |
| `empty_tensor` | 6 | 98.300 |
| `aten::add_` | 2 | 64.880 |
| `aten::select` | 2 | 51.590 |
| `aclnnInplaceCopy` | 2 | 42.080 |
| `aclnnArgMax` | 2 | 40.260 |
| `AssembleOutputs` | 2 | 31.790 |
| `aten::as_strided` | 2 | 24.010 |
| `aten::set_` | 2 | 23.660 |
| `aten::reshape` | 2 | 19.050 |
| `aten::item` | 2 | 15.220 |
| `aclnnInplaceAdds` | 2 | 12.420 |
| `aten::view` | 2 | 11.750 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 2 | 469771.390 |
| `launch` | 2572 | 37722.040 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 4706.750 |
| `InputCopy` | 2 | 169.610 |
| `aclrtLaunchKernelWithHostArgs` | 10 | 125.090 |
| `aclnnInplaceCopy` | 2 | 114.280 |
| `aclnnArgMax` | 2 | 49.220 |
| `ModelExecute` | 2 | 36.310 |
| `aclnnInplaceAdds` | 2 | 21.050 |
| `step_info` | 4 | 13.360 |
| `aclrtGetStreamAttribute` | 6 | 7.120 |
| `OutputCopy` | 2 | 1.310 |

