# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/swiglu/torchair/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/swiglu/torchair/output/profiles/pipe/raw/liteserver-c001-4_102200_20261007122333482_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `181517.226 us`
- `Free`: `3573.272 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `4513.750 us`
- `Stage`: `185090.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 56119.179 |
| `MatMul` | 108 | 49805.333 |
| `PromptFlashAttention` | 108 | 33371.471 |
| `Mul` | 870 | 10308.835 |
| `Square` | 435 | 6014.692 |
| `Transpose` | 432 | 4792.558 |
| `SwiGlu` | 108 | 4618.437 |
| `AutomaticBufferFusionOp` | 651 | 4174.587 |
| `SplitVD` | 324 | 3391.130 |
| `Add` | 225 | 3171.370 |
| `Cast` | 219 | 2210.666 |
| `Neg` | 216 | 1805.784 |
| `ConcatV2D` | 216 | 1717.287 |
| `Data` | 3 | 16.940 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_2` | 3 | 1391.887 |
| `MatMul_34` | 3 | 1388.107 |
| `MatMul_66` | 3 | 1384.687 |
| `MatMul_42` | 3 | 1384.428 |
| `MatMul_118` | 3 | 1384.347 |
| `MatMul_102` | 3 | 1383.888 |
| `MatMul_58` | 3 | 1383.868 |
| `MatMul_122` | 3 | 1383.728 |
| `MatMul_142` | 3 | 1383.547 |
| `MatMul_82` | 3 | 1383.528 |
| `MatMul_90` | 3 | 1383.447 |
| `MatMul_110` | 3 | 1383.447 |
| `MatMul_10` | 3 | 1383.368 |
| `MatMul_30` | 3 | 1383.349 |
| `MatMul_6` | 3 | 1383.287 |
| `MatMul_86` | 3 | 1383.267 |
| `MatMul_94` | 3 | 1383.267 |
| `MatMul_70` | 3 | 1383.168 |
| `MatMul_38` | 3 | 1383.068 |
| `MatMul_62` | 3 | 1382.928 |
| `MatMul_46` | 3 | 1382.927 |
| `MatMul_134` | 3 | 1382.927 |
| `MatMul_14` | 3 | 1382.889 |
| `MatMul_126` | 3 | 1382.888 |
| `MatMul_50` | 3 | 1382.788 |
| `MatMul_138` | 3 | 1382.767 |
| `MatMul_130` | 3 | 1382.729 |
| `MatMul_98` | 3 | 1382.688 |
| `MatMul_74` | 3 | 1382.667 |
| `MatMul_114` | 3 | 1382.528 |
| `MatMul_54` | 3 | 1382.487 |
| `MatMul_78` | 3 | 1382.487 |
| `MatMul_106` | 3 | 1382.228 |
| `MatMul_26` | 3 | 1382.108 |
| `MatMul_18` | 3 | 1381.987 |
| `MatMul_22` | 3 | 1381.627 |
| `PromptFlashAttention_8` | 3 | 956.960 |
| `PromptFlashAttention_2` | 3 | 952.700 |
| `PromptFlashAttention_18` | 3 | 950.858 |
| `PromptFlashAttention_4` | 3 | 949.220 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `MatMul_2` | 3 | 1391.887 |
| `MatMul_34` | 3 | 1388.107 |
| `MatMul_66` | 3 | 1384.687 |
| `MatMul_42` | 3 | 1384.428 |
| `MatMul_118` | 3 | 1384.347 |
| `MatMul_102` | 3 | 1383.888 |
| `MatMul_58` | 3 | 1383.868 |
| `MatMul_122` | 3 | 1383.728 |
| `MatMul_142` | 3 | 1383.547 |
| `MatMul_82` | 3 | 1383.528 |
| `MatMul_90` | 3 | 1383.447 |
| `MatMul_110` | 3 | 1383.447 |
| `MatMul_10` | 3 | 1383.368 |
| `MatMul_30` | 3 | 1383.349 |
| `MatMul_6` | 3 | 1383.287 |
| `MatMul_86` | 3 | 1383.267 |
| `MatMul_94` | 3 | 1383.267 |
| `MatMul_70` | 3 | 1383.168 |
| `MatMul_38` | 3 | 1383.068 |
| `MatMul_62` | 3 | 1382.928 |
| `MatMul_46` | 3 | 1382.927 |
| `MatMul_134` | 3 | 1382.927 |
| `MatMul_14` | 3 | 1382.889 |
| `MatMul_126` | 3 | 1382.888 |
| `MatMul_50` | 3 | 1382.788 |
| `MatMul_138` | 3 | 1382.767 |
| `MatMul_130` | 3 | 1382.729 |
| `MatMul_98` | 3 | 1382.688 |
| `MatMul_74` | 3 | 1382.667 |
| `MatMul_114` | 3 | 1382.528 |
| `MatMul_54` | 3 | 1382.487 |
| `MatMul_78` | 3 | 1382.487 |
| `MatMul_106` | 3 | 1382.228 |
| `MatMul_26` | 3 | 1382.108 |
| `MatMul_18` | 3 | 1381.987 |
| `MatMul_22` | 3 | 1381.627 |
| `MatMul_79_to_v3` | 3 | 791.076 |
| `MatMul_123_to_v3` | 3 | 790.015 |
| `MatMul_143_to_v3` | 3 | 788.436 |
| `MatMul_51_to_v3` | 3 | 787.795 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMul | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 49805.333 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28193.582 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 15721.296 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12204.301 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `MatMul_2` | 1 | 464.949 |
| `MatMul_2` | 1 | 464.089 |
| `MatMul_66` | 1 | 463.049 |
| `MatMul_2` | 1 | 462.849 |
| `MatMul_34` | 1 | 462.829 |
| `MatMul_34` | 1 | 462.669 |
| `MatMul_42` | 1 | 462.609 |
| `MatMul_34` | 1 | 462.609 |
| `MatMul_110` | 1 | 462.349 |
| `MatMul_30` | 1 | 462.070 |
| `MatMul_102` | 1 | 461.890 |
| `MatMul_118` | 1 | 461.889 |
| `MatMul_114` | 1 | 461.770 |
| `MatMul_46` | 1 | 461.729 |
| `MatMul_98` | 1 | 461.729 |
| `MatMul_122` | 1 | 461.729 |
| `MatMul_50` | 1 | 461.689 |
| `MatMul_86` | 1 | 461.689 |
| `MatMul_134` | 1 | 461.629 |
| `MatMul_58` | 1 | 461.569 |
| `MatMul_142` | 1 | 461.549 |
| `MatMul_6` | 1 | 461.549 |
| `MatMul_106` | 1 | 461.529 |
| `MatMul_74` | 1 | 461.449 |
| `MatMul_126` | 1 | 461.410 |
| `MatMul_122` | 1 | 461.389 |
| `MatMul_38` | 1 | 461.370 |
| `MatMul_90` | 1 | 461.369 |
| `MatMul_82` | 1 | 461.349 |
| `MatMul_78` | 1 | 461.329 |
| `MatMul_118` | 1 | 461.329 |
| `MatMul_70` | 1 | 461.310 |
| `MatMul_94` | 1 | 461.309 |
| `MatMul_10` | 1 | 461.250 |
| `MatMul_10` | 1 | 461.229 |
| `MatMul_70` | 1 | 461.229 |
| `MatMul_18` | 1 | 461.209 |
| `MatMul_90` | 1 | 461.209 |
| `MatMul_58` | 1 | 461.170 |
| `MatMul_6` | 1 | 461.169 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 63938.512 |
| `cache_compiler inference` | 3 | 63666.042 |
| `colqwen.torchair.text_forward.step0` | 1 | 62579.750 |
| `colqwen.torchair.text_forward.step1` | 1 | 61827.780 |
| `colqwen.torchair.text_forward.step2` | 1 | 61710.150 |
| `TorchDynamo Cache Lookup` | 3 | 60848.182 |
| `Torch-Compiled Region: 0/0` | 3 | 3830.450 |
| `TorchNpuGraphBase::Run` | 3 | 2723.230 |
| `RefreshAtTensorFromGeTensor` | 3 | 1109.620 |
| `ExecuteGraph` | 3 | 566.960 |
| `aten::empty` | 3 | 509.520 |
| `AssembleInputs` | 3 | 352.980 |
| `AssembleOutputs` | 3 | 296.110 |
| `aten::set_` | 3 | 269.500 |
| `empty_tensor` | 3 | 252.380 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `ModelLoad` | 1 | 260646.100 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 179756.400 |
| `launch` | 1415 | 21013.740 |
| `InputCopy` | 3 | 200.780 |
| `ModelExecute` | 3 | 72.440 |
| `aclrtLaunchKernelWithHostArgs` | 3 | 48.560 |
| `step_info` | 6 | 42.220 |
| `OutputCopy` | 3 | 1.760 |

