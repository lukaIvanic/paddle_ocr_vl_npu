# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/bsnd/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/bsnd/torchair/output/profiles/pipe/raw/liteserver-c001-4_93308_20261007121447147_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `185923.034 us`
- `Free`: `3918.219 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `5274.750 us`
- `Stage`: `189841.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56658.112 |
| `MatMul` | 108 | 49843.199 |
| `PromptFlashAttention` | 108 | 36093.100 |
| `Mul` | 870 | 11008.104 |
| `AutomaticBufferFusionOp` | 759 | 9237.404 |
| `SplitVD` | 432 | 7235.014 |
| `Square` | 435 | 6720.193 |
| `Add` | 225 | 3050.187 |
| `Cast` | 219 | 2445.990 |
| `Neg` | 216 | 1815.356 |
| `ConcatV2D` | 216 | 1801.276 |
| `Data` | 3 | 16.220 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_42` | 3 | 1387.908 |
| `MatMul_46` | 3 | 1386.489 |
| `MatMul_106` | 3 | 1386.247 |
| `MatMul_110` | 3 | 1386.168 |
| `MatMul_134` | 3 | 1386.128 |
| `MatMul_114` | 3 | 1386.048 |
| `MatMul_94` | 3 | 1385.728 |
| `MatMul_90` | 3 | 1385.648 |
| `MatMul_122` | 3 | 1385.388 |
| `MatMul_6` | 3 | 1385.348 |
| `MatMul_126` | 3 | 1385.048 |
| `MatMul_58` | 3 | 1385.008 |
| `MatMul_102` | 3 | 1384.948 |
| `MatMul_14` | 3 | 1384.888 |
| `MatMul_130` | 3 | 1384.848 |
| `MatMul_34` | 3 | 1384.747 |
| `MatMul_138` | 3 | 1384.727 |
| `MatMul_10` | 3 | 1384.627 |
| `MatMul_26` | 3 | 1384.608 |
| `MatMul_2` | 3 | 1384.607 |
| `MatMul_22` | 3 | 1384.467 |
| `MatMul_18` | 3 | 1383.867 |
| `MatMul_50` | 3 | 1383.748 |
| `MatMul_78` | 3 | 1383.748 |
| `MatMul_62` | 3 | 1383.688 |
| `MatMul_118` | 3 | 1383.668 |
| `MatMul_66` | 3 | 1383.567 |
| `MatMul_38` | 3 | 1383.528 |
| `MatMul_74` | 3 | 1383.468 |
| `MatMul_54` | 3 | 1383.447 |
| `MatMul_142` | 3 | 1383.447 |
| `MatMul_82` | 3 | 1383.109 |
| `MatMul_86` | 3 | 1383.048 |
| `MatMul_70` | 3 | 1382.708 |
| `MatMul_98` | 3 | 1382.368 |
| `MatMul_30` | 3 | 1382.167 |
| `PromptFlashAttention_30` | 3 | 1039.601 |
| `PromptFlashAttention_18` | 3 | 1029.481 |
| `PromptFlashAttention_17` | 3 | 1023.061 |
| `PromptFlashAttention_19` | 3 | 1020.661 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_42` | 3 | 1387.908 |
| `MatMul_46` | 3 | 1386.489 |
| `MatMul_106` | 3 | 1386.247 |
| `MatMul_110` | 3 | 1386.168 |
| `MatMul_134` | 3 | 1386.128 |
| `MatMul_114` | 3 | 1386.048 |
| `MatMul_94` | 3 | 1385.728 |
| `MatMul_90` | 3 | 1385.648 |
| `MatMul_122` | 3 | 1385.388 |
| `MatMul_6` | 3 | 1385.348 |
| `MatMul_126` | 3 | 1385.048 |
| `MatMul_58` | 3 | 1385.008 |
| `MatMul_102` | 3 | 1384.948 |
| `MatMul_14` | 3 | 1384.888 |
| `MatMul_130` | 3 | 1384.848 |
| `MatMul_34` | 3 | 1384.747 |
| `MatMul_138` | 3 | 1384.727 |
| `MatMul_10` | 3 | 1384.627 |
| `MatMul_26` | 3 | 1384.608 |
| `MatMul_2` | 3 | 1384.607 |
| `MatMul_22` | 3 | 1384.467 |
| `MatMul_18` | 3 | 1383.867 |
| `MatMul_50` | 3 | 1383.748 |
| `MatMul_78` | 3 | 1383.748 |
| `MatMul_62` | 3 | 1383.688 |
| `MatMul_118` | 3 | 1383.668 |
| `MatMul_66` | 3 | 1383.567 |
| `MatMul_38` | 3 | 1383.528 |
| `MatMul_74` | 3 | 1383.468 |
| `MatMul_54` | 3 | 1383.447 |
| `MatMul_142` | 3 | 1383.447 |
| `MatMul_82` | 3 | 1383.109 |
| `MatMul_86` | 3 | 1383.048 |
| `MatMul_70` | 3 | 1382.708 |
| `MatMul_98` | 3 | 1382.368 |
| `MatMul_30` | 3 | 1382.167 |
| `MatMul_7_to_v3` | 3 | 793.297 |
| `MatMul_131_to_v3` | 3 | 789.356 |
| `MatMul_31_to_v3` | 3 | 788.936 |
| `MatMul_103_to_v3` | 3 | 787.956 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49843.199 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28145.965 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15876.378 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12635.769 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_42` | 1 | 464.230 |
| `MatMul_134` | 1 | 463.610 |
| `MatMul_6` | 1 | 463.029 |
| `MatMul_114` | 1 | 463.009 |
| `MatMul_106` | 1 | 462.969 |
| `MatMul_110` | 1 | 462.969 |
| `MatMul_34` | 1 | 462.929 |
| `MatMul_2` | 1 | 462.889 |
| `MatMul_102` | 1 | 462.889 |
| `MatMul_46` | 1 | 462.750 |
| `MatMul_78` | 1 | 462.730 |
| `MatMul_118` | 1 | 462.610 |
| `MatMul_126` | 1 | 462.550 |
| `MatMul_94` | 1 | 462.509 |
| `MatMul_38` | 1 | 462.449 |
| `MatMul_46` | 1 | 462.389 |
| `MatMul_90` | 1 | 462.309 |
| `MatMul_138` | 1 | 462.309 |
| `MatMul_106` | 1 | 462.309 |
| `MatMul_142` | 1 | 462.289 |
| `MatMul_66` | 1 | 462.249 |
| `MatMul_74` | 1 | 462.249 |
| `MatMul_122` | 1 | 462.229 |
| `MatMul_130` | 1 | 462.089 |
| `MatMul_42` | 1 | 462.069 |
| `MatMul_58` | 1 | 462.029 |
| `MatMul_14` | 1 | 462.029 |
| `MatMul_90` | 1 | 461.990 |
| `MatMul_62` | 1 | 461.950 |
| `MatMul_10` | 1 | 461.949 |
| `MatMul_70` | 1 | 461.930 |
| `MatMul_94` | 1 | 461.910 |
| `MatMul_50` | 1 | 461.849 |
| `MatMul_26` | 1 | 461.770 |
| `MatMul_110` | 1 | 461.750 |
| `MatMul_58` | 1 | 461.749 |
| `MatMul_54` | 1 | 461.729 |
| `MatMul_98` | 1 | 461.690 |
| `MatMul_130` | 1 | 461.650 |
| `MatMul_10` | 1 | 461.649 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 65524.330 |
| `cache_compiler inference` | 3 | 65234.250 |
| `colqwen.torchair.text_forward.step0` | 1 | 64535.290 |
| `colqwen.torchair.text_forward.step1` | 1 | 63402.580 |
| `colqwen.torchair.text_forward.step2` | 1 | 63355.810 |
| `TorchDynamo Cache Lookup` | 3 | 62268.840 |
| `Torch-Compiled Region: 0/0` | 3 | 4414.370 |
| `TorchNpuGraphBase::Run` | 3 | 2803.880 |
| `RefreshAtTensorFromGeTensor` | 3 | 1159.260 |
| `ExecuteGraph` | 3 | 579.030 |
| `aten::empty` | 3 | 524.870 |
| `AssembleInputs` | 3 | 335.700 |
| `AssembleOutputs` | 3 | 308.080 |
| `aten::set_` | 3 | 266.970 |
| `empty_tensor` | 3 | 261.800 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 243763.940 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 183966.890 |
| `launch` | 1307 | 19501.940 |
| `InputCopy` | 3 | 224.510 |
| `ModelExecute` | 3 | 64.190 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 62.500 |
| `step_info` | 6 | 29.530 |
| `OutputCopy` | 3 | 2.850 |

