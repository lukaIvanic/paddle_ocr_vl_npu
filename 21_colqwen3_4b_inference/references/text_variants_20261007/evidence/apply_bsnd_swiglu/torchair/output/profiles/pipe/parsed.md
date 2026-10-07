# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd_swiglu/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd_swiglu/torchair/output/profiles/pipe/raw/liteserver-c001-4_103753_20261007122530448_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `175713.118 us`
- `Free`: `3441.061 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4440.500 us`
- `Stage`: `179154.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56164.553 |
| `MatMul` | 108 | 49881.437 |
| `PromptFlashAttention` | 108 | 35516.804 |
| `Mul` | 870 | 10744.030 |
| `Square` | 435 | 6932.649 |
| `SwiGlu` | 108 | 4907.575 |
| `Add` | 225 | 3196.811 |
| `ApplyRotaryPosEmb` | 108 | 2569.372 |
| `Cast` | 219 | 2564.645 |
| `SplitVD` | 108 | 1696.800 |
| `AutomaticBufferFusionOp` | 435 | 1522.608 |
| `Data` | 3 | 16.740 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_86` | 3 | 1395.188 |
| `MatMul_122` | 3 | 1393.829 |
| `MatMul_118` | 3 | 1392.967 |
| `MatMul_2` | 3 | 1387.149 |
| `MatMul_10` | 3 | 1386.988 |
| `MatMul_78` | 3 | 1386.067 |
| `MatMul_98` | 3 | 1385.728 |
| `MatMul_58` | 3 | 1385.588 |
| `MatMul_62` | 3 | 1385.528 |
| `MatMul_38` | 3 | 1385.427 |
| `MatMul_134` | 3 | 1385.387 |
| `MatMul_30` | 3 | 1385.247 |
| `MatMul_130` | 3 | 1385.247 |
| `MatMul_74` | 3 | 1385.088 |
| `MatMul_102` | 3 | 1385.067 |
| `MatMul_126` | 3 | 1385.047 |
| `MatMul_90` | 3 | 1385.028 |
| `MatMul_70` | 3 | 1384.968 |
| `MatMul_114` | 3 | 1384.888 |
| `MatMul_54` | 3 | 1384.847 |
| `MatMul_18` | 3 | 1384.827 |
| `MatMul_42` | 3 | 1384.668 |
| `MatMul_26` | 3 | 1384.647 |
| `MatMul_34` | 3 | 1384.569 |
| `MatMul_110` | 3 | 1384.548 |
| `MatMul_50` | 3 | 1384.528 |
| `MatMul_46` | 3 | 1384.447 |
| `MatMul_106` | 3 | 1384.349 |
| `MatMul_14` | 3 | 1384.007 |
| `MatMul_66` | 3 | 1383.927 |
| `MatMul_138` | 3 | 1383.888 |
| `MatMul_94` | 3 | 1383.848 |
| `MatMul_82` | 3 | 1383.808 |
| `MatMul_22` | 3 | 1383.667 |
| `MatMul_6` | 3 | 1383.448 |
| `MatMul_142` | 3 | 1382.988 |
| `PromptFlashAttention_19` | 3 | 1027.279 |
| `PromptFlashAttention_20` | 3 | 1018.561 |
| `PromptFlashAttention_33` | 3 | 1017.859 |
| `PromptFlashAttention_31` | 3 | 1016.561 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_86` | 3 | 1395.188 |
| `MatMul_122` | 3 | 1393.829 |
| `MatMul_118` | 3 | 1392.967 |
| `MatMul_2` | 3 | 1387.149 |
| `MatMul_10` | 3 | 1386.988 |
| `MatMul_78` | 3 | 1386.067 |
| `MatMul_98` | 3 | 1385.728 |
| `MatMul_58` | 3 | 1385.588 |
| `MatMul_62` | 3 | 1385.528 |
| `MatMul_38` | 3 | 1385.427 |
| `MatMul_134` | 3 | 1385.387 |
| `MatMul_30` | 3 | 1385.247 |
| `MatMul_130` | 3 | 1385.247 |
| `MatMul_74` | 3 | 1385.088 |
| `MatMul_102` | 3 | 1385.067 |
| `MatMul_126` | 3 | 1385.047 |
| `MatMul_90` | 3 | 1385.028 |
| `MatMul_70` | 3 | 1384.968 |
| `MatMul_114` | 3 | 1384.888 |
| `MatMul_54` | 3 | 1384.847 |
| `MatMul_18` | 3 | 1384.827 |
| `MatMul_42` | 3 | 1384.668 |
| `MatMul_26` | 3 | 1384.647 |
| `MatMul_34` | 3 | 1384.569 |
| `MatMul_110` | 3 | 1384.548 |
| `MatMul_50` | 3 | 1384.528 |
| `MatMul_46` | 3 | 1384.447 |
| `MatMul_106` | 3 | 1384.349 |
| `MatMul_14` | 3 | 1384.007 |
| `MatMul_66` | 3 | 1383.927 |
| `MatMul_138` | 3 | 1383.888 |
| `MatMul_94` | 3 | 1383.848 |
| `MatMul_82` | 3 | 1383.808 |
| `MatMul_22` | 3 | 1383.667 |
| `MatMul_6` | 3 | 1383.448 |
| `MatMul_142` | 3 | 1382.988 |
| `MatMul_51_to_v3` | 3 | 793.176 |
| `MatMul_15_to_v3` | 3 | 791.356 |
| `MatMul_7_to_v3` | 3 | 790.875 |
| `MatMul_59_to_v3` | 3 | 790.036 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49881.437 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28177.283 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15822.307 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12164.963 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_86` | 1 | 466.949 |
| `MatMul_122` | 1 | 466.050 |
| `MatMul_118` | 1 | 465.329 |
| `MatMul_10` | 1 | 464.190 |
| `MatMul_122` | 1 | 464.170 |
| `MatMul_86` | 1 | 464.129 |
| `MatMul_86` | 1 | 464.110 |
| `MatMul_118` | 1 | 463.949 |
| `MatMul_2` | 1 | 463.889 |
| `MatMul_118` | 1 | 463.689 |
| `MatMul_122` | 1 | 463.609 |
| `MatMul_78` | 1 | 463.489 |
| `MatMul_114` | 1 | 463.349 |
| `MatMul_98` | 1 | 463.250 |
| `MatMul_58` | 1 | 462.950 |
| `MatMul_110` | 1 | 462.890 |
| `MatMul_130` | 1 | 462.849 |
| `MatMul_90` | 1 | 462.770 |
| `MatMul_30` | 1 | 462.709 |
| `MatMul_54` | 1 | 462.649 |
| `MatMul_102` | 1 | 462.649 |
| `MatMul_134` | 1 | 462.509 |
| `MatMul_70` | 1 | 462.449 |
| `MatMul_66` | 1 | 462.389 |
| `MatMul_38` | 1 | 462.369 |
| `MatMul_50` | 1 | 462.329 |
| `MatMul_138` | 1 | 462.329 |
| `MatMul_74` | 1 | 462.309 |
| `MatMul_62` | 1 | 462.249 |
| `MatMul_106` | 1 | 462.249 |
| `MatMul_46` | 1 | 462.229 |
| `MatMul_82` | 1 | 462.209 |
| `MatMul_26` | 1 | 462.129 |
| `MatMul_62` | 1 | 462.030 |
| `MatMul_18` | 1 | 461.989 |
| `MatMul_34` | 1 | 461.970 |
| `MatMul_10` | 1 | 461.969 |
| `MatMul_18` | 1 | 461.889 |
| `MatMul_42` | 1 | 461.869 |
| `MatMul_38` | 1 | 461.869 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 61772.382 |
| `cache_compiler inference` | 3 | 61481.672 |
| `colqwen.torchair.text_forward.step0` | 1 | 60520.940 |
| `colqwen.torchair.text_forward.step1` | 1 | 59878.310 |
| `colqwen.torchair.text_forward.step2` | 1 | 59739.610 |
| `TorchDynamo Cache Lookup` | 3 | 58822.501 |
| `Torch-Compiled Region: 0/0` | 3 | 3614.430 |
| `TorchNpuGraphBase::Run` | 3 | 2500.540 |
| `RefreshAtTensorFromGeTensor` | 3 | 1022.350 |
| `aten::empty` | 3 | 488.020 |
| `ExecuteGraph` | 3 | 482.330 |
| `AssembleInputs` | 3 | 340.940 |
| `AssembleOutputs` | 3 | 286.950 |
| `aten::set_` | 3 | 256.110 |
| `empty_tensor` | 3 | 243.150 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 186196.730 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 173968.360 |
| `launch` | 1019 | 14006.170 |
| `InputCopy` | 3 | 163.020 |
| `ModelExecute` | 3 | 52.260 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 34.980 |
| `step_info` | 6 | 17.510 |
| `OutputCopy` | 3 | 1.080 |

