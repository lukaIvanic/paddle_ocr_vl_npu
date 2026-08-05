# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/text_decode_lab/mineru_paddle_b1_k4096_compare/paddle_combined_profile/b1_k4096_p2048_pipe_20260805_182126/liteserver-c001-4_2918463_20260805182126154_ascend_pt`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/text_decode_lab/mineru_paddle_b1_k4096_compare/paddle_combined_profile/b1_k4096_p2048_pipe_20260805_182126/liteserver-c001-4_2918463_20260805182126154_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `2890.240 us`
- `Free`: `3190.300 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `1641.750 us`
- `Stage`: `6080.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 182 | 1522.340 |
| `IncreFlashAttention` | 36 | 755.480 |
| `InplaceAddRmsNorm` | 72 | 116.680 |
| `ApplyRotaryPosEmb` | 36 | 84.820 |
| `Scatter` | 72 | 82.520 |
| `AutomaticBufferFusionOp` | 36 | 55.740 |
| `GatherV2` | 26 | 54.560 |
| `SplitVD` | 40 | 54.540 |
| `Cast` | 8 | 28.900 |
| `ArgMaxV2` | 2 | 28.200 |
| `Range` | 2 | 20.920 |
| `ConcatV2D` | 6 | 13.220 |
| `RmsNorm` | 2 | 10.740 |
| `SelectV2` | 4 | 9.100 |
| `Add` | 4 | 8.680 |
| `Data` | 2 | 8.240 |
| `Greater` | 2 | 7.460 |
| `Mul` | 6 | 6.940 |
| `ZerosLike` | 2 | 5.360 |
| `BroadcastTo` | 4 | 4.900 |
| `Fill` | 2 | 3.740 |
| `Cos` | 2 | 3.660 |
| `Sin` | 2 | 3.500 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 2 | 270.580 |
| `IncreFlashAttention` | 2 | 57.140 |
| `IncreFlashAttention_2` | 2 | 46.140 |
| `IncreFlashAttention_1` | 2 | 45.840 |
| `IncreFlashAttention_13` | 2 | 45.040 |
| `IncreFlashAttention_3` | 2 | 43.280 |
| `IncreFlashAttention_5` | 2 | 43.260 |
| `IncreFlashAttention_10` | 2 | 42.400 |
| `IncreFlashAttention_11` | 2 | 42.220 |
| `IncreFlashAttention_8` | 2 | 41.900 |
| `IncreFlashAttention_4` | 2 | 41.860 |
| `IncreFlashAttention_7` | 2 | 40.460 |
| `IncreFlashAttention_6` | 2 | 39.820 |
| `IncreFlashAttention_12` | 2 | 39.540 |
| `IncreFlashAttention_16` | 2 | 39.140 |
| `IncreFlashAttention_9` | 2 | 38.160 |
| `IncreFlashAttention_15` | 2 | 37.980 |
| `IncreFlashAttention_14` | 2 | 36.180 |
| `IncreFlashAttention_17` | 2 | 35.120 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 28.200 |
| `Range` | 2 | 20.920 |
| `MatMul_84` | 2 | 19.140 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 2 | 18.360 |
| `MatMul_62` | 2 | 18.080 |
| `GatherV2` | 2 | 17.860 |
| `MatMul_4` | 2 | 17.740 |
| `MatMul_29` | 2 | 17.360 |
| `MatMul_89` | 2 | 17.280 |
| `MatMul_32` | 2 | 17.200 |
| `MatMul_19` | 2 | 17.040 |
| `MatMul_49` | 2 | 16.980 |
| `MatMul_59` | 2 | 16.840 |
| `MatMul_24` | 2 | 16.540 |
| `MatMul_54` | 2 | 16.520 |
| `MatMul_79` | 2 | 16.480 |
| `MatMul_37` | 2 | 16.460 |
| `MatMul_80` | 2 | 16.460 |
| `MatMul_34` | 2 | 16.440 |
| `MatMul_9` | 2 | 16.400 |
| `MatMul_14` | 2 | 16.400 |
| `MatMul_64` | 2 | 16.380 |
| `MatMul_12` | 2 | 16.280 |
| `MatMul_17` | 2 | 16.240 |
| `MatMul_69` | 2 | 16.160 |
| `MatMul_39` | 2 | 16.140 |
| `MatMul_44` | 2 | 16.140 |
| `MatMul_30` | 2 | 16.080 |
| `MatMul_2` | 2 | 16.060 |
| `MatMul_22` | 2 | 15.960 |
| `MatMul_72` | 2 | 15.960 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 2 | 270.580 |
| `MatMul_84` | 2 | 19.140 |
| `MatMul_62` | 2 | 18.080 |
| `MatMul_4` | 2 | 17.740 |
| `MatMul_29` | 2 | 17.360 |
| `MatMul_89` | 2 | 17.280 |
| `MatMul_32` | 2 | 17.200 |
| `MatMul_19` | 2 | 17.040 |
| `MatMul_49` | 2 | 16.980 |
| `MatMul_59` | 2 | 16.840 |
| `MatMul_24` | 2 | 16.540 |
| `MatMul_54` | 2 | 16.520 |
| `MatMul_79` | 2 | 16.480 |
| `MatMul_37` | 2 | 16.460 |
| `MatMul_80` | 2 | 16.460 |
| `MatMul_34` | 2 | 16.440 |
| `MatMul_9` | 2 | 16.400 |
| `MatMul_14` | 2 | 16.400 |
| `MatMul_64` | 2 | 16.380 |
| `MatMul_12` | 2 | 16.280 |
| `MatMul_17` | 2 | 16.240 |
| `MatMul_69` | 2 | 16.160 |
| `MatMul_39` | 2 | 16.140 |
| `MatMul_44` | 2 | 16.140 |
| `MatMul_30` | 2 | 16.080 |
| `MatMul_2` | 2 | 16.060 |
| `MatMul_22` | 2 | 15.960 |
| `MatMul_72` | 2 | 15.960 |
| `MatMul_47` | 2 | 15.920 |
| `MatMul_74` | 2 | 15.920 |
| `MatMul_27` | 2 | 15.920 |
| `MatMul_77` | 2 | 15.860 |
| `MatMul_82` | 2 | 15.860 |
| `MatMul_87` | 2 | 15.860 |
| `MatMul_52` | 2 | 15.820 |
| `MatMul_67` | 2 | 15.760 |
| `MatMul_57` | 2 | 15.680 |
| `MatMul_7` | 2 | 15.620 |
| `MatMul_42` | 2 | 15.620 |
| `MatMul_58` | 2 | 15.560 |
| `MatMul_1` | 2 | 15.460 |
| `MatMul_20` | 2 | 15.460 |
| `MatMul_35` | 2 | 15.440 |
| `MatMul_15` | 2 | 15.320 |
| `MatMul_25` | 2 | 15.300 |
| `MatMul_50` | 2 | 15.240 |
| `MatMul_70` | 2 | 15.160 |
| `MatMul_45` | 2 | 15.140 |
| `MatMul_10` | 2 | 15.120 |
| `MatMul_55` | 2 | 15.100 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,1024;64,192,16,16" -> "1,3072" | ND;FRACTAL_NZ -> ND` | 72 | 500.220 |
| `MatMul | "1,3072;192,64,16,16" -> "1,1024" | ND;FRACTAL_NZ -> ND` | 36 | 301.900 |
| `MatMul | "1,1024;64,160,16,16" -> "1,2560" | ND;FRACTAL_NZ -> ND` | 36 | 273.560 |
| `MatMul | "1,1024;64,6464,16,16" -> "1,103424" | ND;FRACTAL_NZ -> ND` | 2 | 270.580 |
| `MatMul | "1,2048;128,64,16,16" -> "1,1024" | ND;FRACTAL_NZ -> ND` | 36 | 176.080 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 1 | 136.160 |
| `MatMul_90` | 1 | 134.420 |
| `IncreFlashAttention` | 1 | 29.040 |
| `IncreFlashAttention` | 1 | 28.100 |
| `IncreFlashAttention_1` | 1 | 23.360 |
| `IncreFlashAttention_2` | 1 | 23.260 |
| `IncreFlashAttention_13` | 1 | 22.940 |
| `IncreFlashAttention_2` | 1 | 22.880 |
| `IncreFlashAttention_1` | 1 | 22.480 |
| `IncreFlashAttention_13` | 1 | 22.100 |
| `IncreFlashAttention_3` | 1 | 21.760 |
| `IncreFlashAttention_5` | 1 | 21.640 |
| `IncreFlashAttention_5` | 1 | 21.620 |
| `IncreFlashAttention_3` | 1 | 21.520 |
| `IncreFlashAttention_10` | 1 | 21.280 |
| `IncreFlashAttention_11` | 1 | 21.140 |
| `IncreFlashAttention_10` | 1 | 21.120 |
| `IncreFlashAttention_11` | 1 | 21.080 |
| `IncreFlashAttention_4` | 1 | 21.060 |
| `IncreFlashAttention_8` | 1 | 20.980 |
| `IncreFlashAttention_8` | 1 | 20.920 |
| `IncreFlashAttention_4` | 1 | 20.800 |
| `IncreFlashAttention_7` | 1 | 20.320 |
| `IncreFlashAttention_7` | 1 | 20.140 |
| `IncreFlashAttention_6` | 1 | 20.040 |
| `IncreFlashAttention_12` | 1 | 19.800 |
| `IncreFlashAttention_6` | 1 | 19.780 |
| `IncreFlashAttention_16` | 1 | 19.780 |
| `IncreFlashAttention_12` | 1 | 19.740 |
| `IncreFlashAttention_9` | 1 | 19.380 |
| `IncreFlashAttention_16` | 1 | 19.360 |
| `IncreFlashAttention_15` | 1 | 19.060 |
| `IncreFlashAttention_15` | 1 | 18.920 |
| `IncreFlashAttention_9` | 1 | 18.780 |
| `IncreFlashAttention_14` | 1 | 18.400 |
| `IncreFlashAttention_14` | 1 | 17.780 |
| `IncreFlashAttention_17` | 1 | 17.560 |
| `IncreFlashAttention_17` | 1 | 17.560 |
| `MatMul_84` | 1 | 9.580 |
| `MatMul_84` | 1 | 9.560 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 1 | 9.260 |
| `MatMul_32` | 1 | 9.140 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 1 | 9.100 |
| `MatMul_62` | 1 | 9.080 |
| `MatMul_62` | 1 | 9.000 |
| `MatMul_4` | 1 | 8.940 |
| `MatMul_89` | 1 | 8.840 |
| `MatMul_4` | 1 | 8.800 |
| `MatMul_29` | 1 | 8.780 |
| `MatMul_29` | 1 | 8.580 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `paddleocr_vl.compiled_decode_profile` | 1 | 6994.170 |
| `paddleocr_vl.compiled_decode_step` | 2 | 6810.120 |
| `cache_compiler inference` | 2 | 2041.700 |
| `Event::record` | 4 | 1561.260 |
| `TorchNpuGraphBase::Run` | 2 | 1287.260 |
| `aten::argmax` | 4 | 632.820 |
| `RefreshAtTensorFromGeTensor` | 2 | 522.690 |
| `empty_tensor` | 16 | 453.660 |
| `aten::empty` | 6 | 437.400 |
| `aten::to` | 2 | 371.430 |
| `aten::_to_copy` | 2 | 324.530 |
| `ExecuteGraph` | 2 | 291.920 |
| `aten::full_like` | 2 | 291.180 |
| `aten::where` | 4 | 289.430 |
| `aten::add` | 2 | 229.390 |
| `aten::zeros_like` | 2 | 221.370 |
| `aten::empty_like` | 4 | 184.810 |
| `record_event` | 4 | 184.350 |
| `aten::copy_` | 2 | 157.440 |
| `AssembleInputs` | 2 | 148.650 |
| `aten::fill_` | 2 | 135.730 |
| `aten::set_` | 2 | 127.980 |
| `AssembleOutputs` | 2 | 120.950 |
| `aten::select` | 2 | 120.800 |
| `aten::view` | 4 | 100.540 |
| `aten::zero_` | 2 | 90.340 |
| `aclnnSWhere` | 4 | 84.660 |
| `aten::reshape` | 2 | 84.190 |
| `aten::item` | 2 | 69.110 |
| `aclnnInplaceFillScalar` | 2 | 62.850 |
| `aclnnInplaceCopy` | 2 | 57.740 |
| `aten::as_strided` | 2 | 54.730 |
| `aclnnArgMax` | 2 | 50.770 |
| `aclnnAdds` | 2 | 49.650 |
| `aclnnInplaceZero` | 2 | 41.310 |
| `aten::_local_scalar_dense` | 2 | 33.490 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 51796.080 |
| `launch` | 284 | 4061.660 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 176.680 |
| `aclrtLaunchKernelWithHostArgs` | 18 | 138.880 |
| `InputCopy` | 2 | 114.480 |
| `aclrtRecordEvent` | 4 | 68.990 |
| `aclnnInplaceCopy` | 2 | 56.020 |
| `aclnnArgMax` | 2 | 46.260 |
| `ModelExecute` | 2 | 43.700 |
| `aclnnSWhere` | 4 | 39.630 |
| `aclnnInplaceFillScalar` | 2 | 35.340 |
| `aclnnInplaceZero` | 2 | 27.870 |
| `aclrtCreateEventExWithFlag` | 4 | 25.840 |
| `aclnnAdds` | 2 | 20.460 |
| `step_info` | 4 | 12.440 |
| `aclrtGetStreamAttribute` | 14 | 7.780 |
| `OutputCopy` | 2 | 1.440 |


### Trace Events
| name | count | total_us |
|---|---:|---:|
| `Model@ModelLoad` | 1 | 51796.080 |
| `text_decode_lab.py(1530): <module>` | 1 | 7730.670 |
| `torch/utils/_contextlib.py(124): decorate_context` | 1 | 7729.820 |
| `text_decode_lab.py(1479): main` | 1 | 7727.500 |
| `text_decode_lab.py(596): torch_profile` | 1 | 7724.050 |
| `ProfilerStep#0` | 1 | 7480.020 |
| `paddleocr_vl.compiled_decode_profile` | 1 | 6994.170 |
| `paddleocr_vl.compiled_decode_step` | 2 | 6810.120 |
| `paddleocr_vl/serving/continuous_decode.py(470): step` | 2 | 6417.890 |
| `paddleocr_vl/serving/continuous_decode.py(242): _measure_enqueue` | 2 | 5000.070 |
| `Node@launch` | 284 | 4061.660 |
| `Free` | 550 | 3190.300 |
| `paddleocr_vl/serving/continuous_decode.py(501): execute` | 2 | 3138.030 |
| `Computing` | 550 | 2890.240 |
| `NOTIFY_WAIT` | 2 | 2868.640 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(595): __call__` | 2 | 2102.650 |
| `cache_compiler inference` | 2 | 2041.700 |
| `torch_npu/profiler/profiler_interface.py(44): wrapper` | 4 | 1727.600 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(317): compiled_method` | 2 | 1717.950 |
| `paddleocr_vl/model/text_decode.py(994): forward` | 2 | 1689.620 |
| `Event::record` | 4 | 1561.260 |
| `Iteration 1` | 1 | 1483.260 |
| `Iteration 2` | 1 | 1465.580 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(295): compiled_fn` | 2 | 1391.960 |
| `<string>(51): kernel` | 2 | 1350.250 |
| `torch_npu/dynamo/torchair/ge/_ge_graph.py(802): run` | 2 | 1334.570 |
| `torch_npu/dynamo/torchair/_utils/error_code.py(41): wapper` | 2 | 1328.770 |
| `torch_npu/dynamo/torchair/core/_backend.py(137): run` | 2 | 1322.670 |
| `<built-in method run of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi` | 2 | 1312.530 |
| `TorchNpuGraphBase::Run` | 2 | 1287.260 |
| `torch/autograd/profiler.py(806): __exit__` | 10 | 1151.460 |
| `torch/utils/_pytree.py(1462): tree_iter` | 40 | 1062.450 |
| `torch/_ops.py(1031): __call__` | 10 | 1021.090 |
| `torch_npu/npu/streams.py(143): record` | 4 | 927.940 |
| `torch/_ops.py(1081): _must_dispatch_in_python` | 10 | 837.180 |
| `torch/utils/_pytree.py(1784): tree_any` | 10 | 813.400 |
| `<built-in function any>` | 10 | 766.040 |
| `torch/autograd/profiler.py(800): __enter__` | 10 | 723.960 |
| `torch/_ops.py(1197): __call__` | 10 | 674.770 |
| `torch_npu/npu/utils.py(260): current_stream` | 4 | 644.730 |
| `aten::argmax` | 4 | 632.820 |
| `torch_npu/profiler/analysis/prof_common_func/_utils.py(27): wrapper` | 2 | 631.710 |
| `<built-in method _record_function_enter_new of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx` | 10 | 621.710 |
| `RefreshAtTensorFromGeTensor` | 2 | 522.690 |
| `torch_npu/npu/utils.py(129): _get_device_index` | 9 | 508.620 |
| `empty_tensor` | 16 | 453.660 |
| `aten::empty` | 6 | 437.400 |
| `torch/utils/_pytree.py(1048): _get_node_type` | 70 | 427.870 |
| `<built-in method argmax of type object at 0xffffa9efbe88>` | 2 | 425.300 |
| `torch/_utils.py(837): _get_device_index` | 9 | 420.160 |

