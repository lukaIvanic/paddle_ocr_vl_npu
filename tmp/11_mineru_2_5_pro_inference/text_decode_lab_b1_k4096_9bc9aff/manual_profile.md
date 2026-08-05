# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab_b1_k4096_9bc9aff/profile_manual_pipe`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/text_decode_lab_b1_k4096_9bc9aff/profile_manual_pipe/liteserver-c001-4_2895975_20260805180111387_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `4573.620 us`
- `Free`: `71.100 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `5960.750 us`
- `Stage`: `4644.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 194 | 2432.600 |
| `TransData` | 144 | 927.200 |
| `BroadcastTo` | 98 | 726.320 |
| `GatherV2` | 580 | 713.780 |
| `BatchMatMul` | 96 | 535.860 |
| `Cast` | 154 | 506.020 |
| `StridedSliceD` | 192 | 461.240 |
| `MatMulV2` | 144 | 460.980 |
| `ConcatV2D` | 194 | 369.260 |
| `AutomaticBufferFusionOp` | 244 | 316.000 |
| `SplitVD` | 96 | 283.500 |
| `Add` | 52 | 265.900 |
| `Mul` | 202 | 236.820 |
| `Muls` | 48 | 222.080 |
| `SoftmaxV2` | 48 | 181.340 |
| `Scatter` | 96 | 118.540 |
| `Square` | 98 | 116.140 |
| `Neg` | 96 | 104.580 |
| `ArgMaxV2` | 2 | 38.260 |
| `Range` | 2 | 17.320 |
| `Data` | 2 | 12.360 |
| `LessEqual` | 2 | 7.280 |
| `Cos` | 2 | 3.620 |
| `Sin` | 2 | 3.480 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 475.980 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 38.260 |
| `MatMul_19` | 2 | 33.960 |
| `MatMul_21` | 2 | 32.500 |
| `MatMul_23` | 2 | 30.860 |
| `MatMul_18` | 2 | 30.040 |
| `MatMul_22` | 2 | 29.860 |
| `MatMul_3` | 2 | 29.540 |
| `MatMul_17` | 2 | 26.720 |
| `MatMul_87` | 2 | 26.180 |
| `MatMul_59` | 2 | 25.960 |
| `MatMul_67` | 2 | 25.920 |
| `MatMul_7` | 2 | 25.840 |
| `MatMul_91` | 2 | 25.780 |
| `MatMul_43` | 2 | 25.720 |
| `MatMul_15` | 2 | 25.700 |
| `MatMul_51` | 2 | 25.640 |
| `MatMul_55` | 2 | 25.600 |
| `MatMul_71` | 2 | 25.540 |
| `MatMul_79` | 2 | 25.540 |
| `MatMul_39` | 2 | 25.500 |
| `MatMul_47` | 2 | 25.500 |
| `MatMul_31` | 2 | 25.480 |
| `MatMul_75` | 2 | 25.480 |
| `MatMul_11` | 2 | 25.400 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 2 | 475.980 |
| `MatMul_19` | 2 | 33.960 |
| `MatMul_21` | 2 | 32.500 |
| `MatMul_23` | 2 | 30.860 |
| `MatMul_18` | 2 | 30.040 |
| `MatMul_22` | 2 | 29.860 |
| `MatMul_3` | 2 | 29.540 |
| `MatMul_17` | 2 | 26.720 |
| `MatMul_87` | 2 | 26.180 |
| `MatMul_59` | 2 | 25.960 |
| `MatMul_67` | 2 | 25.920 |
| `MatMul_7` | 2 | 25.840 |
| `MatMul_91` | 2 | 25.780 |
| `MatMul_43` | 2 | 25.720 |
| `MatMul_15` | 2 | 25.700 |
| `MatMul_51` | 2 | 25.640 |
| `MatMul_55` | 2 | 25.600 |
| `MatMul_71` | 2 | 25.540 |
| `MatMul_79` | 2 | 25.540 |
| `MatMul_39` | 2 | 25.500 |
| `MatMul_47` | 2 | 25.500 |
| `MatMul_31` | 2 | 25.480 |
| `MatMul_75` | 2 | 25.480 |
| `MatMul_11` | 2 | 25.400 |
| `MatMul_83` | 2 | 25.400 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,896;4864,896" -> "1,4864" | ND;ND -> ND` | 96 | 1108.580 |
| `MatMul | "1,4864;896,4864;1,1,896" -> "1,896" | ND;ND;ND -> ND` | 48 | 631.740 |
| `MatMul | "1,896;151936,896" -> "1,151936" | ND;ND -> ND` | 2 | 475.980 |
| `BatchMatMul | "14,1,4096;14,4,256,16,16" -> "14,1,64" | ND;FRACTAL_NZ -> ND` | 48 | 321.360 |
| `MatMulV2 | "1,896;896,896;896" -> "1,896" | ND;ND;ND -> ND` | 48 | 244.440 |
| `MatMulV2 | "1,896;128,896;128" -> "1,128" | ND;ND;ND -> ND` | 96 | 216.540 |
| `MatMul | "1,896;896,896;1,1,896" -> "1,896" | ND;ND;ND -> ND` | 48 | 216.300 |
| `BatchMatMul | "14,4,1,16,16;14,4,256,16,16" -> "14,1,4096" | FRACTAL_NZ;FRACTAL_NZ -> ND` | 48 | 214.500 |

### TransData Names
| name | count | total_us |
|---|---:|---:|
| `trans_TransData_69` | 2 | 23.140 |
| `trans_TransData_2` | 2 | 23.000 |
| `trans_TransData_42` | 2 | 22.820 |
| `trans_TransData_3` | 2 | 22.300 |
| `trans_TransData_39` | 2 | 21.140 |
| `trans_TransData_27` | 2 | 20.780 |
| `trans_TransData_51` | 2 | 20.760 |
| `trans_TransData_15` | 2 | 20.720 |
| `trans_TransData_63` | 2 | 20.680 |
| `trans_TransData_45` | 2 | 20.640 |
| `trans_TransData_6` | 2 | 20.620 |
| `trans_TransData_9` | 2 | 20.520 |
| `trans_TransData_48` | 2 | 20.520 |
| `trans_TransData_33` | 2 | 20.420 |
| `trans_TransData_54` | 2 | 20.420 |
| `trans_TransData_57` | 2 | 20.420 |
| `trans_TransData_18` | 2 | 20.400 |
| `trans_TransData_21` | 2 | 20.400 |
| `trans_TransData_66` | 2 | 20.400 |
| `trans_TransData_24` | 2 | 20.380 |
| `trans_TransData_12` | 2 | 20.340 |
| `trans_TransData_36` | 2 | 20.340 |
| `trans_TransData_30` | 2 | 20.280 |
| `trans_TransData_60` | 2 | 20.260 |
| `trans_TransData_72` | 2 | 20.120 |

### TransData Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `TransData | "14,4096,64" -> "14,4,256,16,16" | ND -> FRACTAL_NZ` | 96 | 794.560 |
| `TransData | "14,1,64" -> "14,4,1,16,16" | ND -> FRACTAL_NZ` | 48 | 132.640 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_96` | 1 | 238.140 |
| `MatMul_96` | 1 | 237.840 |
| `MatMul_19` | 1 | 21.360 |
| `MatMul_21` | 1 | 20.240 |
| `MatMul_18` | 1 | 19.020 |
| `MatMul_22` | 1 | 18.280 |
| `MatMul_23` | 1 | 18.220 |
| `MatMul_17` | 1 | 15.420 |
| `MatMul_3` | 1 | 14.960 |
| `MatMul_3` | 1 | 14.580 |
| `MatMul_67` | 1 | 13.200 |
| `MatMul_7` | 1 | 13.180 |
| `MatMul_59` | 1 | 13.160 |
| `MatMul_87` | 1 | 13.140 |
| `MatMul_91` | 1 | 13.100 |
| `MatMul_87` | 1 | 13.040 |
| `MatMul_79` | 1 | 13.020 |
| `MatMul_55` | 1 | 13.000 |
| `MatMul_51` | 1 | 12.920 |
| `MatMul_15` | 1 | 12.900 |
| `MatMul_31` | 1 | 12.880 |
| `MatMul_43` | 1 | 12.880 |
| `MatMul_11` | 1 | 12.840 |
| `MatMul_43` | 1 | 12.840 |
| `MatMul_75` | 1 | 12.840 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `mineru.text_decode.manual` | 1 | 2886.890 |
| `cache_compiler inference` | 2 | 2210.740 |
| `Torch-Compiled Region: 0/0` | 2 | 1287.940 |
| `TorchNpuGraphBase::Run` | 2 | 784.300 |
| `TorchDynamo Cache Lookup` | 2 | 497.430 |
| `ExecuteGraph` | 2 | 354.140 |
| `aten::argmax` | 4 | 208.580 |
| `aten::to` | 2 | 172.630 |
| `RefreshAtTensorFromGeTensor` | 2 | 165.590 |
| `aten::_to_copy` | 2 | 160.090 |
| `AssembleInputs` | 2 | 129.490 |
| `aten::empty` | 6 | 122.120 |
| `aten::copy_` | 2 | 103.320 |
| `empty_tensor` | 6 | 80.090 |
| `aten::set_` | 2 | 57.690 |
| `aten::add_` | 2 | 57.610 |
| `aten::select` | 2 | 44.420 |
| `aclnnArgMax` | 2 | 40.920 |
| `AssembleOutputs` | 2 | 29.680 |
| `aclnnInplaceCopy` | 2 | 29.110 |
| `aten::as_strided` | 2 | 19.420 |
| `aten::reshape` | 2 | 19.080 |
| `aten::item` | 2 | 12.990 |
| `aten::view` | 2 | 12.020 |
| `aclnnInplaceAdds` | 2 | 9.740 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 242803.080 |
| `launch` | 1399 | 18604.060 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 7531.370 |
| `InputCopy` | 2 | 218.990 |
| `aclrtLaunchKernelWithHostArgs` | 10 | 115.880 |
| `aclnnInplaceCopy` | 2 | 100.740 |
| `ModelExecute` | 2 | 88.780 |
| `step_info` | 4 | 58.020 |
| `aclnnArgMax` | 2 | 52.600 |
| `aclnnInplaceAdds` | 2 | 19.510 |
| `aclrtGetStreamAttribute` | 6 | 5.800 |
| `OutputCopy` | 2 | 1.350 |

