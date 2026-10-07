# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/torchair/output/profiles/pipe/raw/liteserver-c001-4_11764_20261007110855075_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `185801.142 us`
- `Free`: `3116.472 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3556.250 us`
- `Stage`: `188917.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56459.247 |
| `MatMul` | 108 | 49830.837 |
| `PromptFlashAttention` | 108 | 33259.928 |
| `Mul` | 870 | 9819.074 |
| `AutomaticBufferFusionOp` | 759 | 9031.642 |
| `SplitVD` | 432 | 7326.411 |
| `Square` | 435 | 6181.319 |
| `Transpose` | 432 | 5058.490 |
| `Add` | 225 | 2976.142 |
| `Cast` | 219 | 2263.913 |
| `Neg` | 216 | 1836.053 |
| `ConcatV2D` | 216 | 1740.817 |
| `Data` | 3 | 18.401 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1391.348 |
| `MatMul_70` | 3 | 1389.767 |
| `MatMul_82` | 3 | 1387.848 |
| `MatMul_90` | 3 | 1387.647 |
| `MatMul_42` | 3 | 1386.869 |
| `MatMul_46` | 3 | 1386.848 |
| `MatMul_138` | 3 | 1386.168 |
| `MatMul_114` | 3 | 1385.507 |
| `MatMul_10` | 3 | 1385.469 |
| `MatMul_38` | 3 | 1385.447 |
| `MatMul_86` | 3 | 1385.047 |
| `MatMul_94` | 3 | 1384.308 |
| `MatMul_74` | 3 | 1384.188 |
| `MatMul_58` | 3 | 1384.108 |
| `MatMul_118` | 3 | 1384.108 |
| `MatMul_134` | 3 | 1384.008 |
| `MatMul_50` | 3 | 1383.887 |
| `MatMul_130` | 3 | 1383.707 |
| `MatMul_142` | 3 | 1383.627 |
| `MatMul_2` | 3 | 1383.307 |
| `MatMul_110` | 3 | 1383.128 |
| `MatMul_26` | 3 | 1383.047 |
| `MatMul_18` | 3 | 1382.908 |
| `MatMul_102` | 3 | 1382.729 |
| `MatMul_34` | 3 | 1382.647 |
| `MatMul_126` | 3 | 1382.608 |
| `MatMul_62` | 3 | 1382.407 |
| `MatMul_98` | 3 | 1382.347 |
| `MatMul_54` | 3 | 1382.287 |
| `MatMul_78` | 3 | 1382.247 |
| `MatMul_22` | 3 | 1382.227 |
| `MatMul_66` | 3 | 1382.169 |
| `MatMul_14` | 3 | 1382.007 |
| `MatMul_6` | 3 | 1381.988 |
| `MatMul_30` | 3 | 1381.809 |
| `MatMul_122` | 3 | 1381.069 |
| `PromptFlashAttention_30` | 3 | 949.959 |
| `PromptFlashAttention_8` | 3 | 947.398 |
| `PromptFlashAttention_7` | 3 | 946.700 |
| `PromptFlashAttention_4` | 3 | 943.758 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_106` | 3 | 1391.348 |
| `MatMul_70` | 3 | 1389.767 |
| `MatMul_82` | 3 | 1387.848 |
| `MatMul_90` | 3 | 1387.647 |
| `MatMul_42` | 3 | 1386.869 |
| `MatMul_46` | 3 | 1386.848 |
| `MatMul_138` | 3 | 1386.168 |
| `MatMul_114` | 3 | 1385.507 |
| `MatMul_10` | 3 | 1385.469 |
| `MatMul_38` | 3 | 1385.447 |
| `MatMul_86` | 3 | 1385.047 |
| `MatMul_94` | 3 | 1384.308 |
| `MatMul_74` | 3 | 1384.188 |
| `MatMul_58` | 3 | 1384.108 |
| `MatMul_118` | 3 | 1384.108 |
| `MatMul_134` | 3 | 1384.008 |
| `MatMul_50` | 3 | 1383.887 |
| `MatMul_130` | 3 | 1383.707 |
| `MatMul_142` | 3 | 1383.627 |
| `MatMul_2` | 3 | 1383.307 |
| `MatMul_110` | 3 | 1383.128 |
| `MatMul_26` | 3 | 1383.047 |
| `MatMul_18` | 3 | 1382.908 |
| `MatMul_102` | 3 | 1382.729 |
| `MatMul_34` | 3 | 1382.647 |
| `MatMul_126` | 3 | 1382.608 |
| `MatMul_62` | 3 | 1382.407 |
| `MatMul_98` | 3 | 1382.347 |
| `MatMul_54` | 3 | 1382.287 |
| `MatMul_78` | 3 | 1382.247 |
| `MatMul_22` | 3 | 1382.227 |
| `MatMul_66` | 3 | 1382.169 |
| `MatMul_14` | 3 | 1382.007 |
| `MatMul_6` | 3 | 1381.988 |
| `MatMul_30` | 3 | 1381.809 |
| `MatMul_122` | 3 | 1381.069 |
| `MatMul_131_to_v3` | 3 | 798.095 |
| `MatMul_135_to_v3` | 3 | 791.316 |
| `MatMul_83_to_v3` | 3 | 790.535 |
| `MatMul_63_to_v3` | 3 | 789.676 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49830.837 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28192.680 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15735.495 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12531.072 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_70` | 1 | 464.589 |
| `MatMul_106` | 1 | 464.469 |
| `MatMul_106` | 1 | 463.909 |
| `MatMul_46` | 1 | 463.229 |
| `MatMul_10` | 1 | 463.189 |
| `MatMul_106` | 1 | 462.970 |
| `MatMul_90` | 1 | 462.909 |
| `MatMul_82` | 1 | 462.849 |
| `MatMul_90` | 1 | 462.829 |
| `MatMul_82` | 1 | 462.809 |
| `MatMul_42` | 1 | 462.750 |
| `MatMul_70` | 1 | 462.649 |
| `MatMul_70` | 1 | 462.529 |
| `MatMul_138` | 1 | 462.409 |
| `MatMul_42` | 1 | 462.329 |
| `MatMul_138` | 1 | 462.310 |
| `MatMul_94` | 1 | 462.249 |
| `MatMul_86` | 1 | 462.229 |
| `MatMul_82` | 1 | 462.190 |
| `MatMul_38` | 1 | 462.149 |
| `MatMul_38` | 1 | 462.149 |
| `MatMul_114` | 1 | 462.109 |
| `MatMul_142` | 1 | 462.089 |
| `MatMul_46` | 1 | 462.049 |
| `MatMul_74` | 1 | 461.969 |
| `MatMul_90` | 1 | 461.909 |
| `MatMul_58` | 1 | 461.889 |
| `MatMul_34` | 1 | 461.809 |
| `MatMul_42` | 1 | 461.790 |
| `MatMul_130` | 1 | 461.769 |
| `MatMul_114` | 1 | 461.749 |
| `MatMul_110` | 1 | 461.710 |
| `MatMul_134` | 1 | 461.709 |
| `MatMul_66` | 1 | 461.690 |
| `MatMul_50` | 1 | 461.689 |
| `MatMul_62` | 1 | 461.689 |
| `MatMul_102` | 1 | 461.670 |
| `MatMul_114` | 1 | 461.649 |
| `MatMul_86` | 1 | 461.609 |
| `MatMul_46` | 1 | 461.570 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 64551.279 |
| `cache_compiler inference` | 3 | 64245.349 |
| `TorchNpuGraphBase::Run` | 3 | 63529.929 |
| `colqwen.torchair.text_forward.step0` | 1 | 63403.480 |
| `colqwen.torchair.text_forward.step1` | 1 | 63104.270 |
| `colqwen.torchair.text_forward.step2` | 1 | 62883.720 |
| `AssembleInputs` | 3 | 62193.399 |
| `RefreshAtTensorFromGeTensor` | 3 | 890.740 |
| `ExecuteGraph` | 3 | 491.480 |
| `aten::empty` | 3 | 432.140 |
| `AssembleOutputs` | 3 | 395.990 |
| `aten::set_` | 3 | 215.800 |
| `empty_tensor` | 3 | 215.630 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 253561.420 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 184448.320 |
| `launch` | 1451 | 19372.030 |
| `InputCopy` | 3 | 196.210 |
| `ModelExecute` | 3 | 56.210 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 43.310 |
| `step_info` | 6 | 17.450 |
| `OutputCopy` | 3 | 2.030 |

