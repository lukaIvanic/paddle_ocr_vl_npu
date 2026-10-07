# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/rotary_bnsd/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/rotary_bnsd/torchair/output/profiles/pipe/raw/liteserver-c001-4_97537_20261007121654596_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `182136.161 us`
- `Free`: `5139.821 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `7286.500 us`
- `Stage`: `187276.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56594.099 |
| `MatMul` | 108 | 49830.171 |
| `PromptFlashAttention` | 108 | 32427.190 |
| `Mul` | 870 | 10181.521 |
| `Square` | 435 | 6823.494 |
| `AutomaticBufferFusionOp` | 543 | 6176.600 |
| `Transpose` | 432 | 5568.138 |
| `SplitVD` | 216 | 5545.898 |
| `RotaryMul` | 216 | 3637.291 |
| `Add` | 225 | 2871.120 |
| `Cast` | 219 | 2463.752 |
| `Data` | 3 | 17.980 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_110` | 3 | 1392.949 |
| `MatMul_142` | 3 | 1390.067 |
| `MatMul_90` | 3 | 1386.649 |
| `MatMul_122` | 3 | 1385.607 |
| `MatMul_94` | 3 | 1384.967 |
| `MatMul_86` | 3 | 1384.747 |
| `MatMul_58` | 3 | 1384.709 |
| `MatMul_46` | 3 | 1384.647 |
| `MatMul_118` | 3 | 1384.507 |
| `MatMul_66` | 3 | 1384.447 |
| `MatMul_130` | 3 | 1384.427 |
| `MatMul_126` | 3 | 1384.289 |
| `MatMul_42` | 3 | 1384.288 |
| `MatMul_114` | 3 | 1384.188 |
| `MatMul_70` | 3 | 1384.168 |
| `MatMul_106` | 3 | 1384.127 |
| `MatMul_134` | 3 | 1384.109 |
| `MatMul_22` | 3 | 1383.828 |
| `MatMul_62` | 3 | 1383.807 |
| `MatMul_38` | 3 | 1383.587 |
| `MatMul_10` | 3 | 1383.387 |
| `MatMul_78` | 3 | 1383.228 |
| `MatMul_30` | 3 | 1383.207 |
| `MatMul_18` | 3 | 1383.147 |
| `MatMul_54` | 3 | 1383.147 |
| `MatMul_26` | 3 | 1383.087 |
| `MatMul_102` | 3 | 1383.048 |
| `MatMul_82` | 3 | 1382.967 |
| `MatMul_50` | 3 | 1382.908 |
| `MatMul_14` | 3 | 1382.868 |
| `MatMul_74` | 3 | 1382.848 |
| `MatMul_98` | 3 | 1382.747 |
| `MatMul_34` | 3 | 1382.727 |
| `MatMul_138` | 3 | 1382.667 |
| `MatMul_6` | 3 | 1382.067 |
| `MatMul_2` | 3 | 1382.007 |
| `PromptFlashAttention_18` | 3 | 927.119 |
| `PromptFlashAttention_19` | 3 | 926.179 |
| `PromptFlashAttention_1` | 3 | 925.418 |
| `PromptFlashAttention_9` | 3 | 919.818 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_110` | 3 | 1392.949 |
| `MatMul_142` | 3 | 1390.067 |
| `MatMul_90` | 3 | 1386.649 |
| `MatMul_122` | 3 | 1385.607 |
| `MatMul_94` | 3 | 1384.967 |
| `MatMul_86` | 3 | 1384.747 |
| `MatMul_58` | 3 | 1384.709 |
| `MatMul_46` | 3 | 1384.647 |
| `MatMul_118` | 3 | 1384.507 |
| `MatMul_66` | 3 | 1384.447 |
| `MatMul_130` | 3 | 1384.427 |
| `MatMul_126` | 3 | 1384.289 |
| `MatMul_42` | 3 | 1384.288 |
| `MatMul_114` | 3 | 1384.188 |
| `MatMul_70` | 3 | 1384.168 |
| `MatMul_106` | 3 | 1384.127 |
| `MatMul_134` | 3 | 1384.109 |
| `MatMul_22` | 3 | 1383.828 |
| `MatMul_62` | 3 | 1383.807 |
| `MatMul_38` | 3 | 1383.587 |
| `MatMul_10` | 3 | 1383.387 |
| `MatMul_78` | 3 | 1383.228 |
| `MatMul_30` | 3 | 1383.207 |
| `MatMul_18` | 3 | 1383.147 |
| `MatMul_54` | 3 | 1383.147 |
| `MatMul_26` | 3 | 1383.087 |
| `MatMul_102` | 3 | 1383.048 |
| `MatMul_82` | 3 | 1382.967 |
| `MatMul_50` | 3 | 1382.908 |
| `MatMul_14` | 3 | 1382.868 |
| `MatMul_74` | 3 | 1382.848 |
| `MatMul_98` | 3 | 1382.747 |
| `MatMul_34` | 3 | 1382.727 |
| `MatMul_138` | 3 | 1382.667 |
| `MatMul_6` | 3 | 1382.067 |
| `MatMul_2` | 3 | 1382.007 |
| `MatMul_19_to_v3` | 3 | 792.675 |
| `MatMul_143_to_v3` | 3 | 786.497 |
| `MatMul_59_to_v3` | 3 | 786.256 |
| `MatMul_47_to_v3` | 3 | 785.776 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49830.171 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28127.982 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15909.655 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12556.462 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_110` | 1 | 466.369 |
| `MatMul_142` | 1 | 464.029 |
| `MatMul_142` | 1 | 463.669 |
| `MatMul_110` | 1 | 463.290 |
| `MatMul_110` | 1 | 463.290 |
| `MatMul_90` | 1 | 462.770 |
| `MatMul_42` | 1 | 462.710 |
| `MatMul_46` | 1 | 462.649 |
| `MatMul_66` | 1 | 462.649 |
| `MatMul_94` | 1 | 462.649 |
| `MatMul_70` | 1 | 462.369 |
| `MatMul_142` | 1 | 462.369 |
| `MatMul_114` | 1 | 462.350 |
| `MatMul_118` | 1 | 462.349 |
| `MatMul_122` | 1 | 462.329 |
| `MatMul_86` | 1 | 462.289 |
| `MatMul_38` | 1 | 462.229 |
| `MatMul_90` | 1 | 462.229 |
| `MatMul_58` | 1 | 462.189 |
| `MatMul_134` | 1 | 462.150 |
| `MatMul_130` | 1 | 462.069 |
| `MatMul_54` | 1 | 461.929 |
| `MatMul_106` | 1 | 461.889 |
| `MatMul_34` | 1 | 461.829 |
| `MatMul_122` | 1 | 461.769 |
| `MatMul_126` | 1 | 461.769 |
| `MatMul_30` | 1 | 461.749 |
| `MatMul_50` | 1 | 461.669 |
| `MatMul_98` | 1 | 461.669 |
| `MatMul_26` | 1 | 461.669 |
| `MatMul_90` | 1 | 461.650 |
| `MatMul_62` | 1 | 461.649 |
| `MatMul_58` | 1 | 461.510 |
| `MatMul_122` | 1 | 461.509 |
| `MatMul_94` | 1 | 461.489 |
| `MatMul_102` | 1 | 461.470 |
| `MatMul_22` | 1 | 461.450 |
| `MatMul_74` | 1 | 461.449 |
| `MatMul_78` | 1 | 461.449 |
| `MatMul_2` | 1 | 461.349 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 65259.684 |
| `cache_compiler inference` | 3 | 64863.724 |
| `colqwen.torchair.text_forward.step0` | 1 | 63903.620 |
| `colqwen.torchair.text_forward.step1` | 1 | 63332.010 |
| `colqwen.torchair.text_forward.step2` | 1 | 62075.630 |
| `TorchDynamo Cache Lookup` | 3 | 61118.644 |
| `Torch-Compiled Region: 0/0` | 3 | 5313.140 |
| `TorchNpuGraphBase::Run` | 3 | 3335.230 |
| `RefreshAtTensorFromGeTensor` | 3 | 1193.660 |
| `ExecuteGraph` | 3 | 754.270 |
| `aten::empty` | 3 | 564.780 |
| `AssembleInputs` | 3 | 522.760 |
| `AssembleOutputs` | 3 | 343.690 |
| `aten::set_` | 3 | 303.560 |
| `empty_tensor` | 3 | 276.740 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 230343.820 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 179922.490 |
| `launch` | 1235 | 18228.160 |
| `InputCopy` | 3 | 353.020 |
| `ModelExecute` | 3 | 94.000 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 74.850 |
| `step_info` | 6 | 24.890 |
| `OutputCopy` | 3 | 2.460 |

