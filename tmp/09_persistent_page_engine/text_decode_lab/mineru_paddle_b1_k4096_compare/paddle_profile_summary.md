# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/text_decode_lab/mineru_paddle_b1_k4096_compare/paddle_profile/b1_k4096_p2048_pipe_20260805_181400/liteserver-c001-4_2910859_20260805181400399_ascend_pt`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/text_decode_lab/mineru_paddle_b1_k4096_compare/paddle_profile/b1_k4096_p2048_pipe_20260805_181400/liteserver-c001-4_2910859_20260805181400399_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `3245.680 us`
- `Free`: `2806.860 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `1889.750 us`
- `Stage`: `6052.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMul` | 182 | 1758.160 |
| `IncreFlashAttention` | 36 | 783.220 |
| `InplaceAddRmsNorm` | 72 | 118.320 |
| `Scatter` | 72 | 95.800 |
| `ApplyRotaryPosEmb` | 36 | 91.760 |
| `AutomaticBufferFusionOp` | 38 | 66.840 |
| `GatherV2` | 26 | 60.880 |
| `SplitVD` | 40 | 56.620 |
| `Range` | 4 | 37.180 |
| `Cast` | 10 | 32.800 |
| `ArgMaxV2` | 2 | 27.620 |
| `BroadcastTo` | 6 | 17.180 |
| `ConcatV2D` | 6 | 13.640 |
| `MaskedFill` | 2 | 13.200 |
| `RmsNorm` | 2 | 10.860 |
| `Data` | 2 | 9.420 |
| `SelectV2` | 4 | 8.480 |
| `Add` | 4 | 8.400 |
| `Mul` | 6 | 7.600 |
| `Greater` | 2 | 7.300 |
| `Less` | 2 | 6.180 |
| `Cos` | 2 | 3.820 |
| `Sin` | 2 | 3.800 |
| `ZerosLike` | 2 | 3.340 |
| `Fill` | 2 | 3.260 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 2 | 341.400 |
| `IncreFlashAttention` | 2 | 65.140 |
| `IncreFlashAttention_16` | 2 | 44.840 |
| `IncreFlashAttention_4` | 2 | 44.680 |
| `IncreFlashAttention_9` | 2 | 44.400 |
| `IncreFlashAttention_15` | 2 | 43.900 |
| `IncreFlashAttention_1` | 2 | 43.360 |
| `IncreFlashAttention_6` | 2 | 43.240 |
| `IncreFlashAttention_7` | 2 | 43.020 |
| `IncreFlashAttention_11` | 2 | 42.960 |
| `IncreFlashAttention_12` | 2 | 42.820 |
| `IncreFlashAttention_5` | 2 | 42.060 |
| `IncreFlashAttention_17` | 2 | 42.020 |
| `IncreFlashAttention_14` | 2 | 41.680 |
| `IncreFlashAttention_8` | 2 | 41.000 |
| `IncreFlashAttention_10` | 2 | 40.020 |
| `IncreFlashAttention_3` | 2 | 40.000 |
| `IncreFlashAttention_2` | 2 | 39.400 |
| `IncreFlashAttention_13` | 2 | 38.680 |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | 27.620 |
| `GatherV2` | 2 | 23.280 |
| `MatMul_29` | 2 | 20.980 |
| `MatMul_19` | 2 | 20.880 |
| `MatMul_22` | 2 | 20.160 |
| `MatMul_2` | 2 | 19.980 |
| `MatMul_49` | 2 | 19.940 |
| `MatMul_17` | 2 | 19.520 |
| `MatMul_9` | 2 | 19.420 |
| `Range` | 2 | 19.340 |
| `MatMul_4` | 2 | 19.200 |
| `MatMul` | 2 | 19.140 |
| `MatMul_27` | 2 | 19.080 |
| `MatMul_39` | 2 | 18.980 |
| `MatMul_59` | 2 | 18.860 |
| `MatMul_54` | 2 | 18.840 |
| `MatMul_74` | 2 | 18.620 |
| `MatMul_37` | 2 | 18.540 |
| `MatMul_14` | 2 | 18.480 |
| `MatMul_34` | 2 | 18.220 |
| `MatMul_82` | 2 | 18.220 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 2 | 341.400 |
| `MatMul_29` | 2 | 20.980 |
| `MatMul_19` | 2 | 20.880 |
| `MatMul_22` | 2 | 20.160 |
| `MatMul_2` | 2 | 19.980 |
| `MatMul_49` | 2 | 19.940 |
| `MatMul_17` | 2 | 19.520 |
| `MatMul_9` | 2 | 19.420 |
| `MatMul_4` | 2 | 19.200 |
| `MatMul` | 2 | 19.140 |
| `MatMul_27` | 2 | 19.080 |
| `MatMul_39` | 2 | 18.980 |
| `MatMul_59` | 2 | 18.860 |
| `MatMul_54` | 2 | 18.840 |
| `MatMul_74` | 2 | 18.620 |
| `MatMul_37` | 2 | 18.540 |
| `MatMul_14` | 2 | 18.480 |
| `MatMul_34` | 2 | 18.220 |
| `MatMul_82` | 2 | 18.220 |
| `MatMul_20` | 2 | 17.980 |
| `MatMul_12` | 2 | 17.960 |
| `MatMul_42` | 2 | 17.840 |
| `MatMul_30` | 2 | 17.800 |
| `MatMul_32` | 2 | 17.800 |
| `MatMul_44` | 2 | 17.780 |
| `MatMul_87` | 2 | 17.740 |
| `MatMul_84` | 2 | 17.540 |
| `MatMul_64` | 2 | 17.380 |
| `MatMul_24` | 2 | 17.360 |
| `MatMul_47` | 2 | 17.260 |
| `MatMul_62` | 2 | 17.220 |
| `MatMul_77` | 2 | 17.140 |
| `MatMul_57` | 2 | 16.900 |
| `MatMul_89` | 2 | 16.880 |
| `MatMul_50` | 2 | 16.860 |
| `MatMul_72` | 2 | 16.820 |
| `MatMul_6` | 2 | 16.800 |
| `MatMul_52` | 2 | 16.720 |
| `MatMul_7` | 2 | 16.660 |
| `MatMul_67` | 2 | 16.640 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1,1024;64,192,16,16" -> "1,3072" | ND;FRACTAL_NZ -> ND` | 72 | 576.880 |
| `MatMul | "1,1024;64,6464,16,16" -> "1,103424" | ND;FRACTAL_NZ -> ND` | 2 | 341.400 |
| `MatMul | "1,3072;192,64,16,16" -> "1,1024" | ND;FRACTAL_NZ -> ND` | 36 | 332.460 |
| `MatMul | "1,1024;64,160,16,16" -> "1,2560" | ND;FRACTAL_NZ -> ND` | 36 | 288.880 |
| `MatMul | "1,2048;128,64,16,16" -> "1,1024" | ND;FRACTAL_NZ -> ND` | 36 | 218.540 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_90` | 1 | 176.440 |
| `MatMul_90` | 1 | 164.960 |
| `IncreFlashAttention` | 1 | 34.020 |
| `IncreFlashAttention` | 1 | 31.120 |
| `IncreFlashAttention_6` | 1 | 23.100 |
| `IncreFlashAttention_16` | 1 | 22.500 |
| `IncreFlashAttention_4` | 1 | 22.400 |
| `IncreFlashAttention_16` | 1 | 22.340 |
| `IncreFlashAttention_9` | 1 | 22.280 |
| `IncreFlashAttention_4` | 1 | 22.280 |
| `IncreFlashAttention_9` | 1 | 22.120 |
| `IncreFlashAttention_15` | 1 | 21.980 |
| `IncreFlashAttention_7` | 1 | 21.960 |
| `IncreFlashAttention_15` | 1 | 21.920 |
| `IncreFlashAttention_1` | 1 | 21.720 |
| `IncreFlashAttention_11` | 1 | 21.680 |
| `IncreFlashAttention_1` | 1 | 21.640 |
| `IncreFlashAttention_12` | 1 | 21.500 |
| `IncreFlashAttention_12` | 1 | 21.320 |
| `IncreFlashAttention_11` | 1 | 21.280 |
| `IncreFlashAttention_5` | 1 | 21.260 |
| `IncreFlashAttention_7` | 1 | 21.060 |
| `IncreFlashAttention_17` | 1 | 21.040 |
| `IncreFlashAttention_17` | 1 | 20.980 |
| `IncreFlashAttention_14` | 1 | 20.860 |
| `IncreFlashAttention_14` | 1 | 20.820 |
| `IncreFlashAttention_5` | 1 | 20.800 |
| `IncreFlashAttention_8` | 1 | 20.520 |
| `IncreFlashAttention_8` | 1 | 20.480 |
| `IncreFlashAttention_10` | 1 | 20.160 |
| `IncreFlashAttention_6` | 1 | 20.140 |
| `IncreFlashAttention_3` | 1 | 20.080 |
| `IncreFlashAttention_3` | 1 | 19.920 |
| `IncreFlashAttention_10` | 1 | 19.860 |
| `IncreFlashAttention_2` | 1 | 19.840 |
| `IncreFlashAttention_2` | 1 | 19.560 |
| `IncreFlashAttention_13` | 1 | 19.380 |
| `IncreFlashAttention_13` | 1 | 19.300 |
| `MatMul_49` | 1 | 11.640 |
| `MatMul_54` | 1 | 10.780 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `paddleocr_vl.compiled_decode_profile` | 1 | 6931.390 |
| `paddleocr_vl.compiled_decode_step` | 2 | 6746.700 |
| `cache_compiler inference` | 2 | 2074.230 |
| `Event::record` | 4 | 1526.250 |
| `TorchNpuGraphBase::Run` | 2 | 1307.600 |
| `aten::argmax` | 4 | 626.010 |
| `RefreshAtTensorFromGeTensor` | 2 | 508.060 |
| `empty_tensor` | 16 | 438.990 |
| `aten::empty` | 6 | 425.100 |
| `aten::to` | 2 | 371.630 |
| `ExecuteGraph` | 2 | 328.380 |
| `aten::_to_copy` | 2 | 324.840 |
| `aten::where` | 4 | 297.790 |
| `aten::full_like` | 2 | 277.190 |
| `aten::add` | 2 | 228.470 |
| `aten::zeros_like` | 2 | 212.610 |
| `aten::empty_like` | 4 | 185.860 |
| `aten::copy_` | 2 | 175.620 |
| `record_event` | 4 | 168.250 |
| `AssembleInputs` | 2 | 147.070 |
| `aten::set_` | 2 | 126.490 |
| `AssembleOutputs` | 2 | 125.450 |
| `aten::select` | 2 | 116.980 |
| `aten::fill_` | 2 | 110.840 |
| `aten::view` | 4 | 107.350 |
| `aclnnSWhere` | 4 | 95.260 |
| `aten::zero_` | 2 | 88.710 |
| `aten::reshape` | 2 | 84.460 |
| `aten::item` | 2 | 72.270 |
| `aclnnInplaceCopy` | 2 | 65.500 |
| `aclnnArgMax` | 2 | 58.400 |
| `aten::as_strided` | 2 | 52.710 |
| `aclnnInplaceFillScalar` | 2 | 47.380 |
| `aclnnAdds` | 2 | 41.180 |
| `aclnnInplaceZero` | 2 | 39.990 |
| `aten::_local_scalar_dense` | 2 | 37.170 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 54031.560 |
| `launch` | 290 | 4287.520 |
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 183.520 |
| `InputCopy` | 2 | 151.020 |
| `aclrtLaunchKernelWithHostArgs` | 18 | 147.720 |
| `aclrtRecordEvent` | 4 | 74.620 |
| `aclnnInplaceCopy` | 2 | 68.560 |
| `aclnnSWhere` | 4 | 51.580 |
| `ModelExecute` | 2 | 46.210 |
| `aclnnArgMax` | 2 | 38.210 |
| `aclnnAdds` | 2 | 36.930 |
| `aclnnInplaceFillScalar` | 2 | 27.520 |
| `aclrtCreateEventExWithFlag` | 4 | 25.350 |
| `step_info` | 4 | 23.610 |
| `aclnnInplaceZero` | 2 | 21.020 |
| `aclrtGetStreamAttribute` | 14 | 8.040 |
| `OutputCopy` | 2 | 0.730 |


### Trace Events
| name | count | total_us |
|---|---:|---:|
| `Model@ModelLoad` | 1 | 54031.560 |
| `text_decode_lab.py(1530): <module>` | 1 | 7964.290 |
| `torch/utils/_contextlib.py(124): decorate_context` | 1 | 7963.400 |
| `text_decode_lab.py(1479): main` | 1 | 7961.390 |
| `text_decode_lab.py(596): torch_profile` | 1 | 7958.040 |
| `ProfilerStep#0` | 1 | 7668.830 |
| `paddleocr_vl.compiled_decode_profile` | 1 | 6931.390 |
| `paddleocr_vl.compiled_decode_step` | 2 | 6746.700 |
| `paddleocr_vl/serving/continuous_decode.py(470): step` | 2 | 6382.430 |
| `paddleocr_vl/serving/continuous_decode.py(242): _measure_enqueue` | 2 | 4974.580 |
| `Node@launch` | 290 | 4287.520 |
| `Computing` | 562 | 3245.680 |
| `NOTIFY_WAIT` | 2 | 3230.380 |
| `paddleocr_vl/serving/continuous_decode.py(501): execute` | 2 | 3162.860 |
| `Free` | 562 | 2806.860 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(595): __call__` | 2 | 2136.110 |
| `cache_compiler inference` | 2 | 2074.230 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(317): compiled_method` | 2 | 1754.450 |
| `paddleocr_vl/model/text_decode.py(994): forward` | 2 | 1725.970 |
| `Iteration 1` | 1 | 1720.920 |
| `torch_npu/profiler/profiler_interface.py(44): wrapper` | 4 | 1678.160 |
| `Iteration 2` | 1 | 1594.820 |
| `Event::record` | 4 | 1526.250 |
| `torch_npu/dynamo/torchair/inference/_cache_compiler.py(295): compiled_fn` | 2 | 1410.790 |
| `<string>(51): kernel` | 2 | 1369.980 |
| `torch_npu/dynamo/torchair/ge/_ge_graph.py(802): run` | 2 | 1354.700 |
| `torch_npu/dynamo/torchair/_utils/error_code.py(41): wapper` | 2 | 1349.580 |
| `torch_npu/dynamo/torchair/core/_backend.py(137): run` | 2 | 1343.540 |
| `<built-in method run of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi` | 2 | 1333.010 |
| `TorchNpuGraphBase::Run` | 2 | 1307.600 |
| `torch_npu/profiler/analysis/prof_common_func/_utils.py(27): wrapper` | 2 | 1189.290 |
| `torch/autograd/profiler.py(806): __exit__` | 10 | 1101.910 |
| `torch/utils/_pytree.py(1462): tree_iter` | 40 | 982.280 |
| `torch/_ops.py(1031): __call__` | 10 | 973.460 |
| `torch/autograd/profiler.py(800): __enter__` | 10 | 939.700 |
| `torch_npu/npu/streams.py(143): record` | 4 | 909.560 |
| `torch/_ops.py(1197): __call__` | 10 | 893.650 |
| `<built-in method _record_function_enter_new of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx` | 10 | 840.110 |
| `torch/_ops.py(1081): _must_dispatch_in_python` | 10 | 792.940 |
| `torch/utils/_pytree.py(1784): tree_any` | 10 | 770.600 |

