# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bnsd/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bnsd/torchair/output/profiles/pipe/raw/liteserver-c001-4_99022_20261007121904037_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `179813.150 us`
- `Free`: `3569.354 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4597.750 us`
- `Stage`: `183382.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56639.443 |
| `MatMul` | 108 | 49842.751 |
| `PromptFlashAttention` | 108 | 32573.851 |
| `Mul` | 870 | 9684.448 |
| `Square` | 435 | 6522.719 |
| `AutomaticBufferFusionOp` | 543 | 6049.126 |
| `SplitVD` | 216 | 5529.838 |
| `Transpose` | 432 | 5082.688 |
| `Add` | 225 | 3021.357 |
| `ApplyRotaryPosEmb` | 108 | 2456.486 |
| `Cast` | 219 | 2394.788 |
| `Data` | 3 | 16.700 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1390.427 |
| `MatMul_110` | 3 | 1389.488 |
| `MatMul_122` | 3 | 1387.548 |
| `MatMul_134` | 3 | 1386.848 |
| `MatMul_26` | 3 | 1386.528 |
| `MatMul_90` | 3 | 1385.887 |
| `MatMul_130` | 3 | 1385.647 |
| `MatMul_66` | 3 | 1385.447 |
| `MatMul_62` | 3 | 1385.047 |
| `MatMul_58` | 3 | 1384.967 |
| `MatMul_78` | 3 | 1384.687 |
| `MatMul_42` | 3 | 1384.608 |
| `MatMul_2` | 3 | 1384.588 |
| `MatMul_70` | 3 | 1384.408 |
| `MatMul_22` | 3 | 1384.387 |
| `MatMul_10` | 3 | 1384.288 |
| `MatMul_74` | 3 | 1384.209 |
| `MatMul_118` | 3 | 1384.187 |
| `MatMul_114` | 3 | 1384.148 |
| `MatMul_94` | 3 | 1384.127 |
| `MatMul_82` | 3 | 1384.108 |
| `MatMul_38` | 3 | 1384.007 |
| `MatMul_126` | 3 | 1383.948 |
| `MatMul_86` | 3 | 1383.688 |
| `MatMul_138` | 3 | 1383.687 |
| `MatMul_18` | 3 | 1383.648 |
| `MatMul_46` | 3 | 1383.587 |
| `MatMul_98` | 3 | 1383.327 |
| `MatMul_34` | 3 | 1383.287 |
| `MatMul_142` | 3 | 1383.128 |
| `MatMul_30` | 3 | 1382.928 |
| `MatMul_102` | 3 | 1382.827 |
| `MatMul_54` | 3 | 1382.509 |
| `MatMul_6` | 3 | 1382.407 |
| `MatMul_14` | 3 | 1382.167 |
| `MatMul_50` | 3 | 1382.027 |
| `PromptFlashAttention_21` | 3 | 930.978 |
| `PromptFlashAttention_17` | 3 | 924.599 |
| `PromptFlashAttention_11` | 3 | 923.718 |
| `PromptFlashAttention_20` | 3 | 923.200 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1390.427 |
| `MatMul_110` | 3 | 1389.488 |
| `MatMul_122` | 3 | 1387.548 |
| `MatMul_134` | 3 | 1386.848 |
| `MatMul_26` | 3 | 1386.528 |
| `MatMul_90` | 3 | 1385.887 |
| `MatMul_130` | 3 | 1385.647 |
| `MatMul_66` | 3 | 1385.447 |
| `MatMul_62` | 3 | 1385.047 |
| `MatMul_58` | 3 | 1384.967 |
| `MatMul_78` | 3 | 1384.687 |
| `MatMul_42` | 3 | 1384.608 |
| `MatMul_2` | 3 | 1384.588 |
| `MatMul_70` | 3 | 1384.408 |
| `MatMul_22` | 3 | 1384.387 |
| `MatMul_10` | 3 | 1384.288 |
| `MatMul_74` | 3 | 1384.209 |
| `MatMul_118` | 3 | 1384.187 |
| `MatMul_114` | 3 | 1384.148 |
| `MatMul_94` | 3 | 1384.127 |
| `MatMul_82` | 3 | 1384.108 |
| `MatMul_38` | 3 | 1384.007 |
| `MatMul_126` | 3 | 1383.948 |
| `MatMul_86` | 3 | 1383.688 |
| `MatMul_138` | 3 | 1383.687 |
| `MatMul_18` | 3 | 1383.648 |
| `MatMul_46` | 3 | 1383.587 |
| `MatMul_98` | 3 | 1383.327 |
| `MatMul_34` | 3 | 1383.287 |
| `MatMul_142` | 3 | 1383.128 |
| `MatMul_30` | 3 | 1382.928 |
| `MatMul_102` | 3 | 1382.827 |
| `MatMul_54` | 3 | 1382.509 |
| `MatMul_6` | 3 | 1382.407 |
| `MatMul_14` | 3 | 1382.167 |
| `MatMul_50` | 3 | 1382.027 |
| `MatMul_135_to_v3` | 3 | 789.295 |
| `MatMul_131_to_v3` | 3 | 786.855 |
| `MatMul_103_to_v3` | 3 | 786.235 |
| `MatMul_143_to_v3` | 3 | 785.676 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49842.751 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28147.898 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15854.117 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12637.428 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 1 | 464.589 |
| `MatMul_134` | 1 | 464.169 |
| `MatMul_122` | 1 | 463.849 |
| `MatMul_110` | 1 | 463.749 |
| `MatMul_130` | 1 | 463.269 |
| `MatMul_66` | 1 | 463.209 |
| `MatMul_106` | 1 | 463.089 |
| `MatMul_110` | 1 | 462.910 |
| `MatMul_110` | 1 | 462.829 |
| `MatMul_114` | 1 | 462.749 |
| `MatMul_106` | 1 | 462.749 |
| `MatMul_26` | 1 | 462.670 |
| `MatMul_42` | 1 | 462.669 |
| `MatMul_70` | 1 | 462.669 |
| `MatMul_74` | 1 | 462.590 |
| `MatMul_78` | 1 | 462.529 |
| `MatMul_82` | 1 | 462.490 |
| `MatMul_138` | 1 | 462.489 |
| `MatMul_90` | 1 | 462.429 |
| `MatMul_46` | 1 | 462.249 |
| `MatMul_38` | 1 | 462.189 |
| `MatMul_34` | 1 | 462.169 |
| `MatMul_62` | 1 | 462.149 |
| `MatMul_98` | 1 | 462.089 |
| `MatMul_26` | 1 | 462.089 |
| `MatMul_122` | 1 | 462.030 |
| `MatMul_10` | 1 | 462.009 |
| `MatMul_62` | 1 | 461.989 |
| `MatMul_118` | 1 | 461.969 |
| `MatMul_58` | 1 | 461.929 |
| `MatMul_18` | 1 | 461.909 |
| `MatMul_94` | 1 | 461.889 |
| `MatMul_58` | 1 | 461.869 |
| `MatMul_90` | 1 | 461.849 |
| `MatMul_2` | 1 | 461.810 |
| `MatMul_26` | 1 | 461.769 |
| `MatMul_2` | 1 | 461.769 |
| `MatMul_126` | 1 | 461.730 |
| `MatMul_122` | 1 | 461.669 |
| `MatMul_86` | 1 | 461.649 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 63279.259 |
| `cache_compiler inference` | 3 | 62980.119 |
| `colqwen.torchair.text_forward.step0` | 1 | 62052.880 |
| `colqwen.torchair.text_forward.step1` | 1 | 61277.390 |
| `colqwen.torchair.text_forward.step2` | 1 | 61142.200 |
| `TorchDynamo Cache Lookup` | 3 | 60213.949 |
| `Torch-Compiled Region: 0/0` | 3 | 3874.350 |
| `TorchNpuGraphBase::Run` | 3 | 2752.030 |
| `RefreshAtTensorFromGeTensor` | 3 | 1093.830 |
| `ExecuteGraph` | 3 | 561.250 |
| `aten::empty` | 3 | 519.790 |
| `AssembleInputs` | 3 | 345.640 |
| `AssembleOutputs` | 3 | 331.140 |
| `aten::set_` | 3 | 287.120 |
| `empty_tensor` | 3 | 260.420 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 234387.040 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 178072.130 |
| `launch` | 1199 | 18574.420 |
| `InputCopy` | 3 | 216.550 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 59.070 |
| `ModelExecute` | 3 | 58.130 |
| `step_info` | 6 | 29.920 |
| `OutputCopy` | 3 | 2.280 |

