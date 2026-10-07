# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd_swiglu/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bsnd_swiglu/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_103399_20261007122412691_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `206169.501 us`
- `Free`: `929206.123 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2476.750 us`
- `Stage`: `1135376.000 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 59603.807 |
| `MatMulV2` | 108 | 50401.152 |
| `PromptFlashAttention` | 108 | 35862.713 |
| `Mul` | 870 | 15681.307 |
| `Pows` | 435 | 10042.442 |
| `Cast` | 870 | 9110.097 |
| `ReduceMean` | 435 | 8923.555 |
| `SwiGlu` | 108 | 4926.394 |
| `Add` | 660 | 4423.726 |
| `Slice` | 324 | 3006.804 |
| `ApplyRotaryPosEmb` | 108 | 2836.063 |
| `Rsqrt` | 435 | 1352.821 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59603.807 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50401.152 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 35862.713 |
| `aclnnMul_MulAiCore_Mul` | 870 | 15681.307 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10042.442 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9110.097 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8923.555 |
| `SwiGlu` | 108 | 4926.394 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 324 | 3006.804 |
| `aclnnAdd_AddAiCore_Add` | 225 | 2986.624 |
| `ApplyRotaryPosEmb` | 108 | 2836.063 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1437.102 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1352.821 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59603.807 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50401.152 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50401.152 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 30298.520 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 16724.418 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12580.869 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.089 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.009 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.249 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.229 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.749 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.849 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.649 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.070 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.030 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.650 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.330 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.310 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.149 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.029 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.909 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.890 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.849 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.449 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.269 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.169 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.150 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.009 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.869 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.729 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.670 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.569 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.550 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.550 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.530 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.470 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.390 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.330 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.229 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1134685.270 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 382209.150 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 376807.510 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 376701.120 |
| `aten::to` | 870 | 211206.260 |
| `aten::_to_copy` | 870 | 177402.990 |
| `empty_tensor` | 4569 | 174356.360 |
| `aten::linear` | 432 | 145853.697 |
| `aten::add` | 660 | 114216.760 |
| `aten::matmul` | 432 | 114109.243 |
| `aten::reshape` | 1296 | 112020.210 |
| `aclnnMatmul` | 432 | 110096.795 |
| `aten::mul` | 870 | 106855.880 |
| `aten::copy_` | 978 | 86378.110 |
| `aten::pow` | 435 | 73017.090 |
| `aten::empty` | 870 | 65366.680 |
| `aten::mean` | 435 | 57036.700 |
| `aten::rsqrt` | 435 | 52678.290 |
| `aten::view` | 1296 | 48399.890 |
| `aten::t` | 432 | 48206.650 |
| `npu::npu_prompt_flash_attention` | 108 | 46161.260 |
| `aten::as_strided` | 1188 | 41353.150 |
| `aclnnInplaceCopy` | 978 | 40162.170 |
| `aclnnPromptFlashAttentionV3` | 108 | 35862.716 |
| `aclnnMul` | 870 | 34418.900 |
| `aten::transpose` | 432 | 32837.650 |
| `aten::item` | 435 | 30183.770 |
| `aten::unsqueeze` | 432 | 28636.870 |
| `aten::_reshape_alias` | 324 | 21465.100 |
| `aten::contiguous` | 108 | 20465.520 |
| `aclnnPowTensorScalar` | 435 | 18196.670 |
| `aclnnAdds` | 435 | 17689.600 |
| `aclnnMean` | 435 | 17633.290 |
| `npu::npu_swiglu` | 108 | 17397.110 |
| `aclnnRsqrt` | 435 | 17277.650 |
| `aten::clone` | 108 | 16795.570 |
| `aten::result_type` | 435 | 15058.920 |
| `aten::split_with_sizes` | 108 | 14996.890 |
| `aten::_local_scalar_dense` | 435 | 14787.650 |
| `npu::npu_apply_rotary_pos_emb` | 108 | 13117.040 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 4785 | 31028.450 |
| `aclrtLaunchKernelWithHostArgs` | 4785 | 23969.140 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21698.790 |
| `PromptFlashAttention_Tiling` | 216 | 15232.180 |
| `aclnnInplaceCopy` | 978 | 9677.080 |
| `aclnnMul` | 870 | 7386.580 |
| `InnerPromptFlashAttention` | 108 | 6173.180 |
| `aclnnMean` | 435 | 4992.450 |
| `aclnnMatmul` | 432 | 4433.010 |
| `aclnnAdds` | 435 | 3805.380 |
| `aclnnPowTensorScalar` | 435 | 3779.270 |
| `aclnnRsqrt` | 435 | 3725.590 |
| `aclnnAdd` | 225 | 2056.560 |
| `aclnnInnerApplyRotaryPosEmb` | 108 | 2028.250 |
| `aclnnSwiGlu` | 108 | 1843.800 |
| `aclnnInnerApplyRotaryPosEmbGetWorkspaceSize` | 108 | 1838.270 |
| `aclrtGetStreamAttribute` | 4569 | 1779.080 |
| `aclnnSwiGluGetWorkspaceSize` | 108 | 1581.380 |
| `aclrtGetResInCurrentThread` | 648 | 701.820 |
| `aclrtGetHardwareSyncAddr` | 867 | 618.080 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 136.260 |

