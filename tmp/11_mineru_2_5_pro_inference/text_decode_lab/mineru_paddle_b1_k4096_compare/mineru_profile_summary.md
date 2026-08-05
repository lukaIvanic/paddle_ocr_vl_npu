# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab/mineru_paddle_b1_k4096_compare/profile_increfa_pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab/mineru_paddle_b1_k4096_compare/profile_increfa_pipe/liteserver-c001-4_2912362_20260805181545932_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `2678.340 us`
- `Free`: `110.300 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4285.250 us`
- `Stage`: `2788.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 194 | 1972.280 |
| `IncreFlashAttention` | 48 | 892.300 |
| `GatherV2` | 580 | 703.380 |
| `MatMulV2` | 144 | 397.680 |
| `RotaryMul` | 96 | 350.400 |
| `SplitVD` | 96 | 241.980 |
| `Mul` | 202 | 235.860 |
| `AutomaticBufferFusionOp` | 148 | 196.620 |
| `ConcatV2D` | 98 | 134.220 |
| `Cast` | 106 | 133.680 |
| `Scatter` | 96 | 122.860 |
| `Square` | 98 | 117.020 |
| `Less` | 48 | 63.080 |
| `ArgMaxV2` | 2 | 39.120 |
| `Range` | 2 | 20.320 |
| `Data` | 2 | 11.380 |
| `Add` | 4 | 8.880 |
| `LessEqual` | 2 | 7.620 |
| `Cos` | 2 | 3.500 |
| `Sin` | 2 | 3.420 |
| `BroadcastTo` | 2 | 2.540 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 407.540 |
| `IncreFlashAttention` | 2 | 52.380 |
| `IncreFlashAttention_18` | 2 | 43.440 |
| `IncreFlashAttention_1` | 2 | 42.380 |
| `IncreFlashAttention_21` | 2 | 40.680 |
| `IncreFlashAttention_6` | 2 | 40.320 |
| `IncreFlashAttention_13` | 2 | 39.320 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 39.120 |
| `IncreFlashAttention_2` | 2 | 38.840 |
| `IncreFlashAttention_5` | 2 | 37.800 |
| `IncreFlashAttention_17` | 2 | 37.260 |
| `IncreFlashAttention_10` | 2 | 37.200 |
| `IncreFlashAttention_23` | 2 | 36.980 |
| `IncreFlashAttention_14` | 2 | 36.900 |
| `IncreFlashAttention_20` | 2 | 36.880 |
| `IncreFlashAttention_8` | 2 | 36.120 |
| `IncreFlashAttention_22` | 2 | 35.720 |
| `IncreFlashAttention_9` | 2 | 34.740 |
| `IncreFlashAttention_7` | 2 | 34.620 |
| `IncreFlashAttention_16` | 2 | 34.100 |
| `IncreFlashAttention_4` | 2 | 33.860 |
| `IncreFlashAttention_19` | 2 | 33.220 |
| `IncreFlashAttention_11` | 2 | 32.760 |
| `IncreFlashAttention_15` | 2 | 32.720 |
| `IncreFlashAttention_3` | 2 | 32.340 |
| `IncreFlashAttention_12` | 2 | 31.720 |
| `MatMul_23` | 2 | 26.460 |
| `MatMul_21` | 2 | 25.740 |
| `MatMul_26` | 2 | 25.020 |
| `MatMul_25` | 2 | 23.960 |
| `MatMul_22` | 2 | 23.600 |
| `MatMul_19` | 2 | 21.700 |
| `MatMul_27` | 2 | 21.640 |
| `MatMul_5` | 2 | 21.600 |
| `MatMul_1` | 2 | 21.540 |
| `MatMul_3` | 2 | 21.260 |
| `MatMul_2` | 2 | 20.660 |
| `MatMul_55` | 2 | 20.580 |
| `MatMul_94` | 2 | 20.340 |
| `Range` | 2 | 20.320 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 407.540 |
| `MatMul_23` | 2 | 26.460 |
| `MatMul_21` | 2 | 25.740 |
| `MatMul_26` | 2 | 25.020 |
| `MatMul_25` | 2 | 23.960 |
| `MatMul_22` | 2 | 23.600 |
| `MatMul_19` | 2 | 21.700 |
| `MatMul_27` | 2 | 21.640 |
| `MatMul_5` | 2 | 21.600 |
| `MatMul_1` | 2 | 21.540 |
| `MatMul_3` | 2 | 21.260 |
| `MatMul_2` | 2 | 20.660 |
| `MatMul_55` | 2 | 20.580 |
| `MatMul_94` | 2 | 20.340 |
| `MatMul_78` | 2 | 20.320 |
| `MatMul_29` | 2 | 20.160 |
| `MatMul_15` | 2 | 20.120 |
| `MatMul_30` | 2 | 20.000 |
| `MatMul_10` | 2 | 19.920 |
| `MatMul_45` | 2 | 19.720 |
| `MatMul_95` | 2 | 19.660 |
| `MatMul_87` | 2 | 19.660 |
| `MatMul_37` | 2 | 19.560 |
| `MatMul_61` | 2 | 19.560 |
| `MatMul_74` | 2 | 19.400 |
| `MatMul_91` | 2 | 19.240 |
| `MatMul_85` | 2 | 19.220 |
| `MatMul_7` | 2 | 19.140 |
| `MatMul_54` | 2 | 19.060 |
| `MatMul_90` | 2 | 19.040 |
| `MatMul_69` | 2 | 18.980 |
| `MatMul_17` | 2 | 18.960 |
| `MatMul_39` | 2 | 18.860 |
| `MatMul_71` | 2 | 18.840 |
| `MatMul_11` | 2 | 18.820 |
| `MatMul_63` | 2 | 18.820 |
| `MatMul_59` | 2 | 18.740 |
| `MatMul_43` | 2 | 18.640 |
| `MatMul_18` | 2 | 18.580 |
| `MatMul_31` | 2 | 18.540 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,896;56,304,16,16" -> "1,4864" | ND;FRACTAL_NZ -> ND` | 96 | 916.660 |
| `MatMul | "1,4864;304,56,16,16;1,1,896" -> "1,896" | ND;FRACTAL_NZ;ND -> ND` | 48 | 466.600 |
| `MatMul | "1,896;56,9496,16,16" -> "1,151936" | ND;FRACTAL_NZ -> ND` | 2 | 407.540 |
| `MatMulV2 | "1,896;56,8,16,16;128" -> "1,128" | ND;FRACTAL_NZ;ND -> ND` | 96 | 210.060 |
| `MatMulV2 | "1,896;56,56,16,16;896" -> "1,896" | ND;FRACTAL_NZ;ND -> ND` | 48 | 187.620 |
| `MatMul | "1,896;56,56,16,16;1,1,896" -> "1,896" | ND;FRACTAL_NZ;ND -> ND` | 48 | 181.480 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 1 | 227.220 |
| `MatMul_96` | 1 | 180.320 |
| `IncreFlashAttention` | 1 | 27.780 |
| `IncreFlashAttention` | 1 | 24.600 |
| `IncreFlashAttention_18` | 1 | 21.920 |
| `IncreFlashAttention_18` | 1 | 21.520 |
| `IncreFlashAttention_1` | 1 | 21.480 |
| `IncreFlashAttention_1` | 1 | 20.900 |
| `IncreFlashAttention_21` | 1 | 20.440 |
| `IncreFlashAttention_6` | 1 | 20.360 |
| `IncreFlashAttention_21` | 1 | 20.240 |
| `IncreFlashAttention_6` | 1 | 19.960 |
| `IncreFlashAttention_2` | 1 | 19.920 |
| `IncreFlashAttention_13` | 1 | 19.760 |
| `IncreFlashAttention_13` | 1 | 19.560 |
| `IncreFlashAttention_5` | 1 | 19.140 |
| `IncreFlashAttention_17` | 1 | 18.940 |
| `IncreFlashAttention_2` | 1 | 18.920 |
| `IncreFlashAttention_20` | 1 | 18.820 |
| `IncreFlashAttention_14` | 1 | 18.700 |
| `IncreFlashAttention_5` | 1 | 18.660 |
| `IncreFlashAttention_10` | 1 | 18.660 |
| `IncreFlashAttention_23` | 1 | 18.660 |
| `IncreFlashAttention_10` | 1 | 18.540 |
| `IncreFlashAttention_8` | 1 | 18.320 |
| `IncreFlashAttention_23` | 1 | 18.320 |
| `IncreFlashAttention_17` | 1 | 18.320 |
| `IncreFlashAttention_14` | 1 | 18.200 |
| `MatMul_21` | 1 | 18.180 |
| `MatMul_23` | 1 | 18.120 |
| `IncreFlashAttention_20` | 1 | 18.060 |
| `MatMul_26` | 1 | 17.880 |
| `IncreFlashAttention_22` | 1 | 17.880 |
| `IncreFlashAttention_7` | 1 | 17.860 |
| `IncreFlashAttention_22` | 1 | 17.840 |
| `IncreFlashAttention_8` | 1 | 17.800 |
| `IncreFlashAttention_9` | 1 | 17.460 |
| `IncreFlashAttention_9` | 1 | 17.280 |
| `IncreFlashAttention_16` | 1 | 17.200 |
| `IncreFlashAttention_4` | 1 | 17.160 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.text_decode.increfa` | 1 | 2663.490 |
| `cache_compiler inference` | 2 | 1974.200 |
| `Torch-Compiled Region: 0/0` | 2 | 1099.840 |
| `TorchNpuGraphBase::Run` | 2 | 617.340 |
| `TorchDynamo Cache Lookup` | 2 | 450.280 |
| `ExecuteGraph` | 2 | 268.640 |
| `aten::argmax` | 4 | 215.920 |
| `aten::to` | 2 | 172.290 |
| `aten::_to_copy` | 2 | 161.330 |
| `RefreshAtTensorFromGeTensor` | 2 | 127.500 |
| `aten::empty` | 6 | 119.010 |
| `aten::copy_` | 2 | 109.020 |
| `AssembleInputs` | 2 | 90.050 |
| `empty_tensor` | 6 | 77.270 |
| `aten::add_` | 2 | 61.080 |
| `aten::select` | 2 | 45.160 |
| `aclnnArgMax` | 2 | 42.960 |
| `aten::set_` | 2 | 36.020 |
| `aclnnInplaceCopy` | 2 | 35.010 |
| `AssembleOutputs` | 2 | 29.430 |
| `aten::as_strided` | 2 | 20.860 |
| `aten::reshape` | 2 | 18.240 |
| `aclnnInplaceAdds` | 2 | 13.440 |
| `aten::view` | 2 | 11.660 |
| `aten::item` | 2 | 11.310 |
| `aten::_local_scalar_dense` | 2 | 5.420 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 181213.590 |
| `launch` | 991 | 14021.760 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 4243.340 |
| `InputCopy` | 2 | 193.310 |
| `aclrtLaunchKernelWithHostArgs` | 10 | 131.030 |
| `aclnnInplaceCopy` | 2 | 83.390 |
| `aclnnArgMax` | 2 | 41.740 |
| `aclnnInplaceAdds` | 2 | 34.450 |
| `ModelExecute` | 2 | 33.430 |
| `step_info` | 4 | 13.080 |
| `aclrtGetStreamAttribute` | 6 | 3.130 |
| `OutputCopy` | 2 | 1.130 |


### Trace Events
| name | count | total_us |
|---|---:|---:|
| `Model@ModelLoad` | 1 | 181213.590 |
| `Node@launch` | 991 | 14021.760 |
| `ProfilerStep#0` | 1 | 7154.080 |
| `NOTIFY_WAIT` | 2 | 5707.400 |
| `Computing` | 1972 | 5658.140 |
| `AscendCL@aclrtSynchronizeDeviceWithTimeout` | 2 | 4243.340 |
| `Iteration 1` | 1 | 3077.420 |
| `Iteration 2` | 1 | 2699.720 |
| `mineru.text_decode.increfa` | 1 | 2663.490 |
| `cache_compiler inference` | 2 | 1974.200 |
| `Torch-Compiled Region: 0/0` | 2 | 1099.840 |
| `TorchNpuGraphBase::Run` | 2 | 617.340 |
| `TorchDynamo Cache Lookup` | 2 | 450.280 |
| `MatMul_96` | 2 | 407.540 |
| `ExecuteGraph` | 2 | 268.640 |
| `Free` | 1972 | 238.300 |
| `aten::argmax` | 4 | 215.920 |
| `Model@InputCopy` | 2 | 193.310 |
| `aten::to` | 2 | 172.290 |
| `aten::_to_copy` | 2 | 161.330 |
| `Dequeue@aclnnInplaceCopy` | 2 | 147.110 |
| `AscendCL@aclrtLaunchKernelWithHostArgs` | 10 | 131.030 |
| `RefreshAtTensorFromGeTensor` | 2 | 127.500 |
| `aten::empty` | 6 | 119.010 |
| `aten::copy_` | 2 | 109.020 |
| `AssembleInputs` | 2 | 90.050 |
| `AscendCL@aclnnInplaceCopy` | 2 | 83.390 |
| `empty_tensor` | 6 | 77.270 |
| `aten::add_` | 2 | 61.080 |
| `IncreFlashAttention` | 2 | 52.380 |
| `Dequeue@aclnnArgMax` | 2 | 51.730 |
| `aten::select` | 2 | 45.160 |
| `Dequeue@aclnnInplaceAdds` | 2 | 43.780 |
| `IncreFlashAttention_18` | 2 | 43.440 |
| `IncreFlashAttention_1` | 2 | 42.380 |
| `AscendCL@aclnnArgMax` | 2 | 41.740 |
| `IncreFlashAttention_21` | 2 | 40.680 |
| `IncreFlashAttention_6` | 2 | 40.320 |
| `IncreFlashAttention_13` | 2 | 39.320 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 39.120 |

