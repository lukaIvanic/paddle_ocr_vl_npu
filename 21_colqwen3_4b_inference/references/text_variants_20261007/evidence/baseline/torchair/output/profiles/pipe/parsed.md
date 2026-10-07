# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline/torchair/output/profiles/pipe/raw/liteserver-c001-4_91490_20261007121237723_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `185377.165 us`
- `Free`: `2981.418 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3420.500 us`
- `Stage`: `188359.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56576.273 |
| `MatMul` | 108 | 49831.772 |
| `PromptFlashAttention` | 108 | 32540.849 |
| `Mul` | 870 | 9909.823 |
| `AutomaticBufferFusionOp` | 759 | 8948.358 |
| `SplitVD` | 432 | 7195.742 |
| `Square` | 435 | 6280.904 |
| `Transpose` | 432 | 5437.621 |
| `Add` | 225 | 2790.254 |
| `Cast` | 219 | 2315.810 |
| `ConcatV2D` | 216 | 1783.138 |
| `Neg` | 216 | 1750.177 |
| `Data` | 3 | 17.600 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_118` | 3 | 1391.367 |
| `MatMul_82` | 3 | 1391.227 |
| `MatMul_130` | 3 | 1386.587 |
| `MatMul_122` | 3 | 1385.448 |
| `MatMul_94` | 3 | 1385.388 |
| `MatMul_90` | 3 | 1385.288 |
| `MatMul_10` | 3 | 1384.729 |
| `MatMul_38` | 3 | 1384.327 |
| `MatMul_110` | 3 | 1384.327 |
| `MatMul_58` | 3 | 1384.167 |
| `MatMul_126` | 3 | 1384.107 |
| `MatMul_70` | 3 | 1384.048 |
| `MatMul_46` | 3 | 1383.987 |
| `MatMul_66` | 3 | 1383.949 |
| `MatMul_62` | 3 | 1383.947 |
| `MatMul_138` | 3 | 1383.887 |
| `MatMul_6` | 3 | 1383.828 |
| `MatMul_142` | 3 | 1383.828 |
| `MatMul_106` | 3 | 1383.728 |
| `MatMul_86` | 3 | 1383.727 |
| `MatMul_34` | 3 | 1383.688 |
| `MatMul_2` | 3 | 1383.627 |
| `MatMul_102` | 3 | 1383.589 |
| `MatMul_54` | 3 | 1383.527 |
| `MatMul_18` | 3 | 1383.429 |
| `MatMul_14` | 3 | 1383.428 |
| `MatMul_42` | 3 | 1383.427 |
| `MatMul_74` | 3 | 1383.407 |
| `MatMul_114` | 3 | 1383.267 |
| `MatMul_78` | 3 | 1383.248 |
| `MatMul_98` | 3 | 1383.028 |
| `MatMul_30` | 3 | 1383.027 |
| `MatMul_134` | 3 | 1382.868 |
| `MatMul_50` | 3 | 1382.547 |
| `MatMul_22` | 3 | 1381.947 |
| `MatMul_26` | 3 | 1381.827 |
| `PromptFlashAttention_1` | 3 | 934.139 |
| `PromptFlashAttention_18` | 3 | 930.278 |
| `PromptFlashAttention_8` | 3 | 925.399 |
| `PromptFlashAttention_29` | 3 | 923.478 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_118` | 3 | 1391.367 |
| `MatMul_82` | 3 | 1391.227 |
| `MatMul_130` | 3 | 1386.587 |
| `MatMul_122` | 3 | 1385.448 |
| `MatMul_94` | 3 | 1385.388 |
| `MatMul_90` | 3 | 1385.288 |
| `MatMul_10` | 3 | 1384.729 |
| `MatMul_38` | 3 | 1384.327 |
| `MatMul_110` | 3 | 1384.327 |
| `MatMul_58` | 3 | 1384.167 |
| `MatMul_126` | 3 | 1384.107 |
| `MatMul_70` | 3 | 1384.048 |
| `MatMul_46` | 3 | 1383.987 |
| `MatMul_66` | 3 | 1383.949 |
| `MatMul_62` | 3 | 1383.947 |
| `MatMul_138` | 3 | 1383.887 |
| `MatMul_6` | 3 | 1383.828 |
| `MatMul_142` | 3 | 1383.828 |
| `MatMul_106` | 3 | 1383.728 |
| `MatMul_86` | 3 | 1383.727 |
| `MatMul_34` | 3 | 1383.688 |
| `MatMul_2` | 3 | 1383.627 |
| `MatMul_102` | 3 | 1383.589 |
| `MatMul_54` | 3 | 1383.527 |
| `MatMul_18` | 3 | 1383.429 |
| `MatMul_14` | 3 | 1383.428 |
| `MatMul_42` | 3 | 1383.427 |
| `MatMul_74` | 3 | 1383.407 |
| `MatMul_114` | 3 | 1383.267 |
| `MatMul_78` | 3 | 1383.248 |
| `MatMul_98` | 3 | 1383.028 |
| `MatMul_30` | 3 | 1383.027 |
| `MatMul_134` | 3 | 1382.868 |
| `MatMul_50` | 3 | 1382.547 |
| `MatMul_22` | 3 | 1381.947 |
| `MatMul_26` | 3 | 1381.827 |
| `MatMul_143_to_v3` | 3 | 790.156 |
| `MatMul_131_to_v3` | 3 | 789.315 |
| `MatMul_87_to_v3` | 3 | 786.775 |
| `MatMul_103_to_v3` | 3 | 786.535 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49831.772 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28156.240 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15826.697 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12593.336 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_118` | 1 | 465.589 |
| `MatMul_82` | 1 | 465.129 |
| `MatMul_130` | 1 | 463.449 |
| `MatMul_10` | 1 | 463.370 |
| `MatMul_38` | 1 | 463.069 |
| `MatMul_82` | 1 | 463.069 |
| `MatMul_82` | 1 | 463.029 |
| `MatMul_118` | 1 | 462.909 |
| `MatMul_118` | 1 | 462.869 |
| `MatMul_94` | 1 | 462.749 |
| `MatMul_110` | 1 | 462.429 |
| `MatMul_122` | 1 | 462.310 |
| `MatMul_74` | 1 | 462.209 |
| `MatMul_142` | 1 | 462.209 |
| `MatMul_34` | 1 | 462.149 |
| `MatMul_90` | 1 | 462.089 |
| `MatMul_114` | 1 | 462.029 |
| `MatMul_62` | 1 | 461.969 |
| `MatMul_122` | 1 | 461.969 |
| `MatMul_126` | 1 | 461.929 |
| `MatMul_98` | 1 | 461.909 |
| `MatMul_90` | 1 | 461.830 |
| `MatMul_70` | 1 | 461.789 |
| `MatMul_106` | 1 | 461.770 |
| `MatMul_46` | 1 | 461.749 |
| `MatMul_138` | 1 | 461.729 |
| `MatMul_86` | 1 | 461.689 |
| `MatMul_14` | 1 | 461.669 |
| `MatMul_58` | 1 | 461.669 |
| `MatMul_66` | 1 | 461.650 |
| `MatMul_42` | 1 | 461.649 |
| `MatMul_66` | 1 | 461.649 |
| `MatMul_54` | 1 | 461.629 |
| `MatMul_102` | 1 | 461.609 |
| `MatMul_130` | 1 | 461.589 |
| `MatMul_130` | 1 | 461.549 |
| `MatMul_30` | 1 | 461.529 |
| `MatMul_54` | 1 | 461.509 |
| `MatMul_58` | 1 | 461.509 |
| `MatMul_94` | 1 | 461.470 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 64309.696 |
| `cache_compiler inference` | 3 | 63998.536 |
| `colqwen.torchair.text_forward.step0` | 1 | 63281.030 |
| `TorchNpuGraphBase::Run` | 3 | 63277.856 |
| `colqwen.torchair.text_forward.step1` | 1 | 62846.960 |
| `colqwen.torchair.text_forward.step2` | 1 | 62696.250 |
| `AssembleInputs` | 3 | 62055.866 |
| `RefreshAtTensorFromGeTensor` | 3 | 902.200 |
| `ExecuteGraph` | 3 | 520.520 |
| `aten::empty` | 3 | 438.600 |
| `AssembleOutputs` | 3 | 257.260 |
| `aten::set_` | 3 | 220.290 |
| `empty_tensor` | 3 | 214.800 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 279709.500 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 183966.890 |
| `launch` | 1451 | 20522.580 |
| `InputCopy` | 3 | 201.690 |
| `ModelExecute` | 3 | 66.380 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 43.820 |
| `step_info` | 6 | 24.590 |
| `OutputCopy` | 3 | 1.680 |

