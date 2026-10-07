# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/torchair/output/profiles/memory/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/torchair/output/profiles/memory/raw/liteserver-c001-4_11764_20261007110902959_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `185770.842 us`
- `Free`: `3135.872 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4075.250 us`
- `Stage`: `188906.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56367.047 |
| `MatMul` | 108 | 49815.036 |
| `PromptFlashAttention` | 108 | 33371.649 |
| `Mul` | 870 | 9816.991 |
| `AutomaticBufferFusionOp` | 759 | 9029.653 |
| `SplitVD` | 432 | 7295.696 |
| `Square` | 435 | 6184.031 |
| `Transpose` | 432 | 5054.647 |
| `Add` | 225 | 2975.615 |
| `Cast` | 219 | 2267.406 |
| `Neg` | 216 | 1834.265 |
| `ConcatV2D` | 216 | 1743.917 |
| `Data` | 3 | 16.020 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1390.488 |
| `MatMul_70` | 3 | 1389.548 |
| `MatMul_90` | 3 | 1386.668 |
| `MatMul_114` | 3 | 1386.068 |
| `MatMul_82` | 3 | 1386.047 |
| `MatMul_10` | 3 | 1385.188 |
| `MatMul_118` | 3 | 1385.008 |
| `MatMul_18` | 3 | 1384.907 |
| `MatMul_42` | 3 | 1384.568 |
| `MatMul_46` | 3 | 1384.527 |
| `MatMul_50` | 3 | 1384.408 |
| `MatMul_86` | 3 | 1384.407 |
| `MatMul_138` | 3 | 1384.367 |
| `MatMul_58` | 3 | 1383.907 |
| `MatMul_94` | 3 | 1383.828 |
| `MatMul_130` | 3 | 1383.769 |
| `MatMul_14` | 3 | 1383.747 |
| `MatMul_74` | 3 | 1383.508 |
| `MatMul_142` | 3 | 1383.209 |
| `MatMul_22` | 3 | 1383.168 |
| `MatMul_38` | 3 | 1383.127 |
| `MatMul_34` | 3 | 1383.007 |
| `MatMul_102` | 3 | 1382.747 |
| `MatMul_134` | 3 | 1382.547 |
| `MatMul_98` | 3 | 1382.468 |
| `MatMul_62` | 3 | 1382.308 |
| `MatMul_122` | 3 | 1382.227 |
| `MatMul_6` | 3 | 1382.128 |
| `MatMul_26` | 3 | 1382.088 |
| `MatMul_30` | 3 | 1381.947 |
| `MatMul_2` | 3 | 1381.747 |
| `MatMul_110` | 3 | 1381.728 |
| `MatMul_126` | 3 | 1381.569 |
| `MatMul_54` | 3 | 1381.427 |
| `MatMul_66` | 3 | 1381.349 |
| `MatMul_78` | 3 | 1381.287 |
| `PromptFlashAttention_4` | 3 | 957.359 |
| `PromptFlashAttention_3` | 3 | 949.499 |
| `PromptFlashAttention_8` | 3 | 949.419 |
| `PromptFlashAttention_18` | 3 | 944.859 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1390.488 |
| `MatMul_70` | 3 | 1389.548 |
| `MatMul_90` | 3 | 1386.668 |
| `MatMul_114` | 3 | 1386.068 |
| `MatMul_82` | 3 | 1386.047 |
| `MatMul_10` | 3 | 1385.188 |
| `MatMul_118` | 3 | 1385.008 |
| `MatMul_18` | 3 | 1384.907 |
| `MatMul_42` | 3 | 1384.568 |
| `MatMul_46` | 3 | 1384.527 |
| `MatMul_50` | 3 | 1384.408 |
| `MatMul_86` | 3 | 1384.407 |
| `MatMul_138` | 3 | 1384.367 |
| `MatMul_58` | 3 | 1383.907 |
| `MatMul_94` | 3 | 1383.828 |
| `MatMul_130` | 3 | 1383.769 |
| `MatMul_14` | 3 | 1383.747 |
| `MatMul_74` | 3 | 1383.508 |
| `MatMul_142` | 3 | 1383.209 |
| `MatMul_22` | 3 | 1383.168 |
| `MatMul_38` | 3 | 1383.127 |
| `MatMul_34` | 3 | 1383.007 |
| `MatMul_102` | 3 | 1382.747 |
| `MatMul_134` | 3 | 1382.547 |
| `MatMul_98` | 3 | 1382.468 |
| `MatMul_62` | 3 | 1382.308 |
| `MatMul_122` | 3 | 1382.227 |
| `MatMul_6` | 3 | 1382.128 |
| `MatMul_26` | 3 | 1382.088 |
| `MatMul_30` | 3 | 1381.947 |
| `MatMul_2` | 3 | 1381.747 |
| `MatMul_110` | 3 | 1381.728 |
| `MatMul_126` | 3 | 1381.569 |
| `MatMul_54` | 3 | 1381.427 |
| `MatMul_66` | 3 | 1381.349 |
| `MatMul_78` | 3 | 1381.287 |
| `MatMul_131_to_v3` | 3 | 797.497 |
| `MatMul_63_to_v3` | 3 | 791.015 |
| `MatMul_71_to_v3` | 3 | 790.956 |
| `MatMul_15_to_v3` | 3 | 789.136 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49815.036 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28134.186 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15728.096 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12504.765 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_70` | 1 | 463.750 |
| `MatMul_106` | 1 | 463.709 |
| `MatMul_106` | 1 | 463.549 |
| `MatMul_106` | 1 | 463.230 |
| `MatMul_70` | 1 | 463.069 |
| `MatMul_114` | 1 | 462.769 |
| `MatMul_70` | 1 | 462.729 |
| `MatMul_90` | 1 | 462.290 |
| `MatMul_82` | 1 | 462.269 |
| `MatMul_90` | 1 | 462.229 |
| `MatMul_46` | 1 | 462.189 |
| `MatMul_90` | 1 | 462.149 |
| `MatMul_138` | 1 | 462.109 |
| `MatMul_82` | 1 | 462.089 |
| `MatMul_14` | 1 | 462.089 |
| `MatMul_18` | 1 | 462.009 |
| `MatMul_58` | 1 | 462.009 |
| `MatMul_50` | 1 | 461.950 |
| `MatMul_130` | 1 | 461.890 |
| `MatMul_42` | 1 | 461.889 |
| `MatMul_10` | 1 | 461.849 |
| `MatMul_86` | 1 | 461.789 |
| `MatMul_118` | 1 | 461.709 |
| `MatMul_114` | 1 | 461.690 |
| `MatMul_10` | 1 | 461.689 |
| `MatMul_82` | 1 | 461.689 |
| `MatMul_26` | 1 | 461.689 |
| `MatMul_118` | 1 | 461.669 |
| `MatMul_94` | 1 | 461.669 |
| `MatMul_10` | 1 | 461.650 |
| `MatMul_118` | 1 | 461.630 |
| `MatMul_42` | 1 | 461.610 |
| `MatMul_114` | 1 | 461.609 |
| `MatMul_86` | 1 | 461.509 |
| `MatMul_18` | 1 | 461.509 |
| `MatMul_110` | 1 | 461.469 |
| `MatMul_38` | 1 | 461.449 |
| `MatMul_74` | 1 | 461.429 |
| `MatMul_46` | 1 | 461.409 |
| `MatMul_18` | 1 | 461.389 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 64580.838 |
| `cache_compiler inference` | 3 | 64272.478 |
| `colqwen.torchair.text_forward.step0` | 1 | 63764.660 |
| `TorchNpuGraphBase::Run` | 3 | 63492.628 |
| `colqwen.torchair.text_forward.step2` | 1 | 63036.950 |
| `colqwen.torchair.text_forward.step1` | 1 | 63018.310 |
| `AssembleInputs` | 3 | 62173.728 |
| `RefreshAtTensorFromGeTensor` | 3 | 950.180 |
| `ExecuteGraph` | 3 | 564.520 |
| `aten::empty` | 3 | 473.920 |
| `AssembleOutputs` | 3 | 263.650 |
| `empty_tensor` | 3 | 240.610 |
| `aten::set_` | 3 | 230.820 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 253561.420 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 184318.330 |
| `launch` | 1451 | 19382.430 |
| `InputCopy` | 3 | 224.750 |
| `ModelExecute` | 3 | 57.420 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 55.450 |
| `step_info` | 6 | 25.640 |
| `OutputCopy` | 3 | 2.050 |

