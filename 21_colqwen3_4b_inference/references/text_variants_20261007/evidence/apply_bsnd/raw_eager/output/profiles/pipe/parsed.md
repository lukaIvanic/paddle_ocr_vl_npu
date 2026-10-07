# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_100161_20261007121945351_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `210355.722 us`
- `Free`: `962715.456 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2553.500 us`
- `Stage`: `1173071.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 58727.905 |
| `MatMulV2` | 108 | 50330.745 |
| `PromptFlashAttention` | 108 | 36106.104 |
| `Mul` | 978 | 17535.539 |
| `Pows` | 435 | 10136.855 |
| `Cast` | 870 | 9626.394 |
| `ReduceMean` | 435 | 8927.624 |
| `Slice` | 540 | 6495.379 |
| `Add` | 660 | 4399.473 |
| `Swish` | 108 | 3871.575 |
| `ApplyRotaryPosEmb` | 108 | 2801.057 |
| `Rsqrt` | 435 | 1398.567 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58727.905 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50330.745 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 36106.104 |
| `aclnnMul_MulAiCore_Mul` | 978 | 17535.539 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10136.855 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9626.394 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8927.624 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3871.575 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 324 | 3063.220 |
| `aclnnAdd_AddAiCore_Add` | 225 | 2968.802 |
| `ApplyRotaryPosEmb` | 108 | 2801.057 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 1976.121 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1456.038 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1430.671 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1398.567 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58727.905 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50330.745 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50330.745 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28968.732 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17157.586 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12601.587 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.990 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.369 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.929 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.490 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.089 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.269 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.090 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.009 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.750 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.430 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.410 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.130 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.029 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.010 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.549 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.410 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.309 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.290 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.150 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.149 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.110 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.109 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.049 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.990 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.889 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.769 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.709 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.610 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.609 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.589 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1172513.470 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 393088.440 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 391012.000 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 389543.690 |
| `aten::to` | 870 | 213811.540 |
| `aten::_to_copy` | 870 | 179691.500 |
| `empty_tensor` | 4677 | 179653.750 |
| `aten::linear` | 432 | 146243.556 |
| `aten::mul` | 978 | 124468.940 |
| `aten::add` | 660 | 115188.220 |
| `aten::matmul` | 432 | 113172.374 |
| `aten::reshape` | 1296 | 111824.560 |
| `aclnnMatmul` | 432 | 109058.659 |
| `aten::copy_` | 978 | 88149.560 |
| `aten::pow` | 435 | 74262.350 |
| `aten::empty` | 870 | 65909.260 |
| `aten::mean` | 435 | 58532.740 |
| `aten::rsqrt` | 435 | 53762.340 |
| `aten::as_strided` | 1404 | 49303.250 |
| `aten::view` | 1296 | 48702.980 |
| `aten::t` | 432 | 48311.500 |
| `npu::npu_prompt_flash_attention` | 108 | 46637.750 |
| `aclnnInplaceCopy` | 978 | 41197.830 |
| `aclnnMul` | 978 | 41108.010 |
| `aclnnPromptFlashAttentionV3` | 108 | 36106.102 |
| `aten::transpose` | 432 | 33021.200 |
| `aten::item` | 435 | 29948.030 |
| `aten::unsqueeze` | 432 | 28890.920 |
| `aten::split_with_sizes` | 216 | 27411.850 |
| `aten::_reshape_alias` | 324 | 21362.320 |
| `aten::contiguous` | 108 | 20638.190 |
| `aclnnPowTensorScalar` | 435 | 18770.130 |
| `aclnnMean` | 435 | 18580.410 |
| `aclnnAdds` | 435 | 18294.370 |
| `aclnnRsqrt` | 435 | 18087.500 |
| `aten::clone` | 108 | 16989.210 |
| `aten::result_type` | 435 | 15040.890 |
| `aten::silu` | 108 | 14642.400 |
| `aten::_local_scalar_dense` | 435 | 14612.530 |
| `npu::npu_apply_rotary_pos_emb` | 108 | 13759.880 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 5109 | 34799.900 |
| `aclrtLaunchKernelWithHostArgs` | 5109 | 27350.350 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 22750.840 |
| `PromptFlashAttention_Tiling` | 216 | 16062.210 |
| `aclnnInplaceCopy` | 978 | 11788.730 |
| `aclnnMul` | 978 | 10752.810 |
| `InnerPromptFlashAttention` | 108 | 6779.040 |
| `aclnnMean` | 435 | 5876.510 |
| `aclnnMatmul` | 432 | 5278.210 |
| `aclnnAdds` | 435 | 4656.770 |
| `aclnnPowTensorScalar` | 435 | 4614.200 |
| `aclnnRsqrt` | 435 | 4534.410 |
| `aclnnInnerApplyRotaryPosEmb` | 108 | 2557.570 |
| `aclnnAdd` | 225 | 2436.680 |
| `aclnnInnerApplyRotaryPosEmbGetWorkspaceSize` | 108 | 1953.960 |
| `aclrtGetStreamAttribute` | 4677 | 1878.230 |
| `aclnnSilu` | 108 | 1788.790 |
| `aclrtGetHardwareSyncAddr` | 867 | 591.550 |
| `aclrtGetResInCurrentThread` | 432 | 391.560 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 169.120 |

