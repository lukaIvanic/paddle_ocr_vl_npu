# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline_end/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/baseline_end/torchair/output/profiles/pipe/raw/liteserver-c001-4_104856_20261007122614406_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `187729.641 us`
- `Free`: `3020.911 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `3510.750 us`
- `Stage`: `190751.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56620.992 |
| `MatMul` | 108 | 49889.713 |
| `PromptFlashAttention` | 108 | 33332.849 |
| `Mul` | 870 | 10404.114 |
| `AutomaticBufferFusionOp` | 759 | 9048.480 |
| `SplitVD` | 432 | 7233.806 |
| `Square` | 435 | 6684.264 |
| `Transpose` | 432 | 5454.543 |
| `Add` | 225 | 2922.197 |
| `Cast` | 219 | 2408.573 |
| `Neg` | 216 | 1935.328 |
| `ConcatV2D` | 216 | 1778.770 |
| `Data` | 3 | 17.180 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_82` | 3 | 1394.968 |
| `MatMul_118` | 3 | 1393.508 |
| `MatMul_122` | 3 | 1388.208 |
| `MatMul_110` | 3 | 1386.908 |
| `MatMul_34` | 3 | 1386.807 |
| `MatMul_114` | 3 | 1386.767 |
| `MatMul_126` | 3 | 1386.408 |
| `MatMul_138` | 3 | 1386.268 |
| `MatMul_6` | 3 | 1386.088 |
| `MatMul_66` | 3 | 1386.027 |
| `MatMul_134` | 3 | 1386.008 |
| `MatMul_74` | 3 | 1385.828 |
| `MatMul_98` | 3 | 1385.807 |
| `MatMul_142` | 3 | 1385.667 |
| `MatMul_90` | 3 | 1385.587 |
| `MatMul_70` | 3 | 1385.507 |
| `MatMul_62` | 3 | 1385.468 |
| `MatMul_130` | 3 | 1385.347 |
| `MatMul_58` | 3 | 1385.307 |
| `MatMul_42` | 3 | 1385.248 |
| `MatMul_106` | 3 | 1385.207 |
| `MatMul_46` | 3 | 1385.128 |
| `MatMul_18` | 3 | 1384.967 |
| `MatMul_78` | 3 | 1384.688 |
| `MatMul_38` | 3 | 1384.667 |
| `MatMul_30` | 3 | 1384.587 |
| `MatMul_50` | 3 | 1384.587 |
| `MatMul_94` | 3 | 1384.508 |
| `MatMul_86` | 3 | 1384.427 |
| `MatMul_14` | 3 | 1384.408 |
| `MatMul_10` | 3 | 1384.327 |
| `MatMul_26` | 3 | 1384.308 |
| `MatMul_54` | 3 | 1384.288 |
| `MatMul_102` | 3 | 1384.148 |
| `MatMul_22` | 3 | 1384.109 |
| `MatMul_2` | 3 | 1383.628 |
| `PromptFlashAttention_1` | 3 | 970.039 |
| `PromptFlashAttention_9` | 3 | 962.679 |
| `PromptFlashAttention_18` | 3 | 957.860 |
| `PromptFlashAttention_8` | 3 | 957.418 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_82` | 3 | 1394.968 |
| `MatMul_118` | 3 | 1393.508 |
| `MatMul_122` | 3 | 1388.208 |
| `MatMul_110` | 3 | 1386.908 |
| `MatMul_34` | 3 | 1386.807 |
| `MatMul_114` | 3 | 1386.767 |
| `MatMul_126` | 3 | 1386.408 |
| `MatMul_138` | 3 | 1386.268 |
| `MatMul_6` | 3 | 1386.088 |
| `MatMul_66` | 3 | 1386.027 |
| `MatMul_134` | 3 | 1386.008 |
| `MatMul_74` | 3 | 1385.828 |
| `MatMul_98` | 3 | 1385.807 |
| `MatMul_142` | 3 | 1385.667 |
| `MatMul_90` | 3 | 1385.587 |
| `MatMul_70` | 3 | 1385.507 |
| `MatMul_62` | 3 | 1385.468 |
| `MatMul_130` | 3 | 1385.347 |
| `MatMul_58` | 3 | 1385.307 |
| `MatMul_42` | 3 | 1385.248 |
| `MatMul_106` | 3 | 1385.207 |
| `MatMul_46` | 3 | 1385.128 |
| `MatMul_18` | 3 | 1384.967 |
| `MatMul_78` | 3 | 1384.688 |
| `MatMul_38` | 3 | 1384.667 |
| `MatMul_30` | 3 | 1384.587 |
| `MatMul_50` | 3 | 1384.587 |
| `MatMul_94` | 3 | 1384.508 |
| `MatMul_86` | 3 | 1384.427 |
| `MatMul_14` | 3 | 1384.408 |
| `MatMul_10` | 3 | 1384.327 |
| `MatMul_26` | 3 | 1384.308 |
| `MatMul_54` | 3 | 1384.288 |
| `MatMul_102` | 3 | 1384.148 |
| `MatMul_22` | 3 | 1384.109 |
| `MatMul_2` | 3 | 1383.628 |
| `MatMul_123_to_v3` | 3 | 789.816 |
| `MatMul_19_to_v3` | 3 | 789.595 |
| `MatMul_87_to_v3` | 3 | 788.556 |
| `MatMul_143_to_v3` | 3 | 788.175 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49889.713 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28157.444 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15874.177 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12589.371 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_82` | 1 | 466.669 |
| `MatMul_118` | 1 | 465.409 |
| `MatMul_82` | 1 | 464.370 |
| `MatMul_118` | 1 | 464.249 |
| `MatMul_82` | 1 | 463.929 |
| `MatMul_118` | 1 | 463.850 |
| `MatMul_62` | 1 | 463.229 |
| `MatMul_122` | 1 | 463.129 |
| `MatMul_6` | 1 | 463.109 |
| `MatMul_110` | 1 | 463.069 |
| `MatMul_122` | 1 | 462.949 |
| `MatMul_42` | 1 | 462.850 |
| `MatMul_74` | 1 | 462.669 |
| `MatMul_138` | 1 | 462.610 |
| `MatMul_70` | 1 | 462.609 |
| `MatMul_34` | 1 | 462.569 |
| `MatMul_114` | 1 | 462.569 |
| `MatMul_46` | 1 | 462.550 |
| `MatMul_130` | 1 | 462.549 |
| `MatMul_134` | 1 | 462.549 |
| `MatMul_106` | 1 | 462.529 |
| `MatMul_34` | 1 | 462.529 |
| `MatMul_126` | 1 | 462.429 |
| `MatMul_30` | 1 | 462.309 |
| `MatMul_78` | 1 | 462.309 |
| `MatMul_98` | 1 | 462.309 |
| `MatMul_66` | 1 | 462.289 |
| `MatMul_142` | 1 | 462.229 |
| `MatMul_50` | 1 | 462.209 |
| `MatMul_126` | 1 | 462.189 |
| `MatMul_90` | 1 | 462.169 |
| `MatMul_94` | 1 | 462.149 |
| `MatMul_122` | 1 | 462.130 |
| `MatMul_110` | 1 | 462.129 |
| `MatMul_114` | 1 | 462.129 |
| `MatMul_86` | 1 | 462.109 |
| `MatMul_90` | 1 | 462.109 |
| `MatMul_114` | 1 | 462.069 |
| `MatMul_102` | 1 | 462.030 |
| `MatMul_134` | 1 | 462.030 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 65153.302 |
| `cache_compiler inference` | 3 | 64855.062 |
| `colqwen.torchair.text_forward.step0` | 1 | 64160.700 |
| `TorchNpuGraphBase::Run` | 3 | 64145.322 |
| `colqwen.torchair.text_forward.step1` | 1 | 63672.060 |
| `colqwen.torchair.text_forward.step2` | 1 | 63469.020 |
| `AssembleInputs` | 3 | 62865.652 |
| `RefreshAtTensorFromGeTensor` | 3 | 927.480 |
| `ExecuteGraph` | 3 | 566.970 |
| `aten::empty` | 3 | 449.560 |
| `AssembleOutputs` | 3 | 258.550 |
| `aten::set_` | 3 | 233.210 |
| `empty_tensor` | 3 | 225.120 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 255958.380 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 186342.170 |
| `launch` | 1451 | 19573.230 |
| `InputCopy` | 3 | 239.440 |
| `ModelExecute` | 3 | 71.960 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 59.570 |
| `step_info` | 6 | 39.400 |
| `OutputCopy` | 3 | 2.130 |

