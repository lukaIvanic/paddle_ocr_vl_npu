# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd/torchair/output/profiles/pipe/raw/liteserver-c001-4_100515_20261007122112081_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `178151.561 us`
- `Free`: `5146.581 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `6778.000 us`
- `Stage`: `183298.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56677.364 |
| `MatMul` | 108 | 49892.873 |
| `PromptFlashAttention` | 108 | 35996.516 |
| `Mul` | 870 | 9778.875 |
| `Square` | 435 | 6339.458 |
| `AutomaticBufferFusionOp` | 543 | 6121.398 |
| `SplitVD` | 216 | 5522.887 |
| `Add` | 225 | 2998.451 |
| `ApplyRotaryPosEmb` | 108 | 2496.996 |
| `Cast` | 219 | 2312.381 |
| `Data` | 3 | 15.320 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_114` | 3 | 1396.727 |
| `MatMul_78` | 3 | 1395.608 |
| `MatMul_42` | 3 | 1388.647 |
| `MatMul_106` | 3 | 1387.388 |
| `MatMul_102` | 3 | 1386.909 |
| `MatMul_110` | 3 | 1386.847 |
| `MatMul_10` | 3 | 1386.727 |
| `MatMul_46` | 3 | 1386.647 |
| `MatMul_134` | 3 | 1386.567 |
| `MatMul_130` | 3 | 1386.448 |
| `MatMul_122` | 3 | 1386.227 |
| `MatMul_94` | 3 | 1386.047 |
| `MatMul_38` | 3 | 1385.747 |
| `MatMul_90` | 3 | 1385.707 |
| `MatMul_50` | 3 | 1385.629 |
| `MatMul_58` | 3 | 1385.569 |
| `MatMul_18` | 3 | 1385.548 |
| `MatMul_22` | 3 | 1385.408 |
| `MatMul_14` | 3 | 1385.388 |
| `MatMul_138` | 3 | 1385.288 |
| `MatMul_62` | 3 | 1385.207 |
| `MatMul_126` | 3 | 1385.167 |
| `MatMul_98` | 3 | 1385.007 |
| `MatMul_2` | 3 | 1384.667 |
| `MatMul_118` | 3 | 1384.627 |
| `MatMul_54` | 3 | 1384.608 |
| `MatMul_82` | 3 | 1384.487 |
| `MatMul_66` | 3 | 1384.467 |
| `MatMul_142` | 3 | 1384.387 |
| `MatMul_86` | 3 | 1384.247 |
| `MatMul_26` | 3 | 1384.208 |
| `MatMul_6` | 3 | 1383.768 |
| `MatMul_70` | 3 | 1383.649 |
| `MatMul_34` | 3 | 1383.207 |
| `MatMul_30` | 3 | 1383.150 |
| `MatMul_74` | 3 | 1382.947 |
| `PromptFlashAttention_18` | 3 | 1047.581 |
| `PromptFlashAttention_19` | 3 | 1044.381 |
| `PromptFlashAttention_7` | 3 | 1035.761 |
| `PromptFlashAttention_34` | 3 | 1033.021 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_114` | 3 | 1396.727 |
| `MatMul_78` | 3 | 1395.608 |
| `MatMul_42` | 3 | 1388.647 |
| `MatMul_106` | 3 | 1387.388 |
| `MatMul_102` | 3 | 1386.909 |
| `MatMul_110` | 3 | 1386.847 |
| `MatMul_10` | 3 | 1386.727 |
| `MatMul_46` | 3 | 1386.647 |
| `MatMul_134` | 3 | 1386.567 |
| `MatMul_130` | 3 | 1386.448 |
| `MatMul_122` | 3 | 1386.227 |
| `MatMul_94` | 3 | 1386.047 |
| `MatMul_38` | 3 | 1385.747 |
| `MatMul_90` | 3 | 1385.707 |
| `MatMul_50` | 3 | 1385.629 |
| `MatMul_58` | 3 | 1385.569 |
| `MatMul_18` | 3 | 1385.548 |
| `MatMul_22` | 3 | 1385.408 |
| `MatMul_14` | 3 | 1385.388 |
| `MatMul_138` | 3 | 1385.288 |
| `MatMul_62` | 3 | 1385.207 |
| `MatMul_126` | 3 | 1385.167 |
| `MatMul_98` | 3 | 1385.007 |
| `MatMul_2` | 3 | 1384.667 |
| `MatMul_118` | 3 | 1384.627 |
| `MatMul_54` | 3 | 1384.608 |
| `MatMul_82` | 3 | 1384.487 |
| `MatMul_66` | 3 | 1384.467 |
| `MatMul_142` | 3 | 1384.387 |
| `MatMul_86` | 3 | 1384.247 |
| `MatMul_26` | 3 | 1384.208 |
| `MatMul_6` | 3 | 1383.768 |
| `MatMul_70` | 3 | 1383.649 |
| `MatMul_34` | 3 | 1383.207 |
| `MatMul_30` | 3 | 1383.150 |
| `MatMul_74` | 3 | 1382.947 |
| `MatMul_47_to_v3` | 3 | 789.336 |
| `MatMul_87_to_v3` | 3 | 786.575 |
| `MatMul_75_to_v3` | 3 | 786.515 |
| `MatMul_59_to_v3` | 3 | 786.496 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49892.873 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28108.387 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15900.898 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12668.079 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_114` | 1 | 467.349 |
| `MatMul_78` | 1 | 466.990 |
| `MatMul_114` | 1 | 465.309 |
| `MatMul_78` | 1 | 464.589 |
| `MatMul_114` | 1 | 464.069 |
| `MatMul_78` | 1 | 464.029 |
| `MatMul_42` | 1 | 463.549 |
| `MatMul_102` | 1 | 463.270 |
| `MatMul_42` | 1 | 463.209 |
| `MatMul_50` | 1 | 463.190 |
| `MatMul_110` | 1 | 463.109 |
| `MatMul_106` | 1 | 463.090 |
| `MatMul_94` | 1 | 462.969 |
| `MatMul_38` | 1 | 462.869 |
| `MatMul_62` | 1 | 462.829 |
| `MatMul_134` | 1 | 462.769 |
| `MatMul_98` | 1 | 462.749 |
| `MatMul_130` | 1 | 462.749 |
| `MatMul_46` | 1 | 462.689 |
| `MatMul_118` | 1 | 462.609 |
| `MatMul_10` | 1 | 462.509 |
| `MatMul_138` | 1 | 462.450 |
| `MatMul_82` | 1 | 462.409 |
| `MatMul_126` | 1 | 462.409 |
| `MatMul_54` | 1 | 462.330 |
| `MatMul_106` | 1 | 462.289 |
| `MatMul_10` | 1 | 462.249 |
| `MatMul_142` | 1 | 462.249 |
| `MatMul_34` | 1 | 462.209 |
| `MatMul_46` | 1 | 462.209 |
| `MatMul_58` | 1 | 462.190 |
| `MatMul_18` | 1 | 462.170 |
| `MatMul_66` | 1 | 462.149 |
| `MatMul_2` | 1 | 462.109 |
| `MatMul_122` | 1 | 462.109 |
| `MatMul_14` | 1 | 462.090 |
| `MatMul_90` | 1 | 462.089 |
| `MatMul_122` | 1 | 462.069 |
| `MatMul_122` | 1 | 462.049 |
| `MatMul_22` | 1 | 462.009 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 63776.908 |
| `cache_compiler inference` | 3 | 63473.708 |
| `colqwen.torchair.text_forward.step0` | 1 | 62305.770 |
| `colqwen.torchair.text_forward.step1` | 1 | 61591.340 |
| `colqwen.torchair.text_forward.step2` | 1 | 60759.280 |
| `TorchDynamo Cache Lookup` | 3 | 59787.188 |
| `Torch-Compiled Region: 0/0` | 3 | 5257.290 |
| `TorchNpuGraphBase::Run` | 3 | 3222.110 |
| `RefreshAtTensorFromGeTensor` | 3 | 1146.350 |
| `ExecuteGraph` | 3 | 653.710 |
| `AssembleInputs` | 3 | 589.850 |
| `aten::empty` | 3 | 549.840 |
| `AssembleOutputs` | 3 | 345.540 |
| `aten::set_` | 3 | 276.860 |
| `empty_tensor` | 3 | 272.510 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 194704.660 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 176153.480 |
| `launch` | 1055 | 14982.800 |
| `InputCopy` | 3 | 282.450 |
| `ModelExecute` | 3 | 73.650 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 52.840 |
| `step_info` | 6 | 41.240 |
| `OutputCopy` | 3 | 1.820 |

