# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bnsd/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/apply_bnsd/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_98668_20261007121733687_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `218065.899 us`
- `Free`: `1008815.694 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2461.250 us`
- `Stage`: `1226881.500 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 58915.633 |
| `MatMulV2` | 108 | 50183.487 |
| `PromptFlashAttention` | 108 | 33440.670 |
| `Mul` | 978 | 17586.532 |
| `Pows` | 435 | 10544.982 |
| `Transpose` | 324 | 10419.793 |
| `Cast` | 870 | 9117.172 |
| `ReduceMean` | 435 | 8664.976 |
| `Slice` | 432 | 6011.846 |
| `Add` | 660 | 4465.318 |
| `Swish` | 108 | 3594.460 |
| `ApplyRotaryPosEmb` | 108 | 2827.294 |
| `Rsqrt` | 435 | 1255.262 |
| `AsStrided` | 108 | 1040.141 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58915.633 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50183.487 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33440.670 |
| `aclnnMul_MulAiCore_Mul` | 978 | 17586.532 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10544.982 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 324 | 10419.793 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 9117.172 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8664.976 |
| `aclnnSilu_SiluAiCore_Swish` | 108 | 3594.460 |
| `aclnnAdd_AddAiCore_Add` | 225 | 3020.990 |
| `ApplyRotaryPosEmb` | 108 | 2827.294 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2451.613 |
| `aclnnSilu_SliceAiCore_Slice` | 108 | 2109.335 |
| `aclnnMul_SliceAiCore_Slice` | 108 | 1450.898 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1444.328 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1255.262 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 1040.141 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 58915.633 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50183.487 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50183.487 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 28962.366 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 17279.460 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12673.807 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.770 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.330 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.210 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.289 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.270 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.890 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.509 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.490 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.270 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.250 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.210 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.150 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.149 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.050 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.029 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.709 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.570 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.569 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.530 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.370 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.070 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 466.029 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.989 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.969 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.809 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.730 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.709 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.589 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.549 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.350 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.170 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 465.129 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1226290.910 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 411403.000 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 407958.390 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 407903.470 |
| `aten::to` | 870 | 205139.430 |
| `empty_tensor` | 5001 | 184321.850 |
| `aten::_to_copy` | 870 | 172280.220 |
| `aten::linear` | 432 | 142212.153 |
| `aten::mul` | 978 | 117880.670 |
| `aten::matmul` | 432 | 111979.963 |
| `aten::copy_` | 1302 | 111153.730 |
| `aten::add` | 660 | 110476.090 |
| `aclnnMatmul` | 432 | 109099.120 |
| `aten::reshape` | 1296 | 107251.510 |
| `aten::contiguous` | 432 | 80585.850 |
| `aten::pow` | 435 | 70169.740 |
| `aten::clone` | 432 | 65871.500 |
| `aten::empty` | 870 | 63311.990 |
| `aten::as_strided` | 1836 | 61782.020 |
| `aten::transpose` | 864 | 61581.150 |
| `aten::mean` | 435 | 55337.860 |
| `aten::rsqrt` | 435 | 51127.540 |
| `aclnnInplaceCopy` | 1302 | 50959.837 |
| `aten::t` | 432 | 47007.150 |
| `aten::view` | 1296 | 46755.640 |
| `npu::npu_prompt_flash_attention` | 108 | 43793.870 |
| `aclnnMul` | 978 | 37458.550 |
| `aclnnPromptFlashAttentionV3` | 108 | 33440.668 |
| `aten::item` | 435 | 29150.130 |
| `aten::unsqueeze` | 432 | 27166.920 |
| `aten::split_with_sizes` | 216 | 26573.900 |
| `aten::_reshape_alias` | 324 | 20676.680 |
| `aclnnPowTensorScalar` | 435 | 17203.900 |
| `aclnnMean` | 435 | 17103.320 |
| `aclnnRsqrt` | 435 | 16764.020 |
| `aclnnAdds` | 435 | 16746.620 |
| `aten::result_type` | 435 | 14620.540 |
| `aten::_local_scalar_dense` | 435 | 14190.970 |
| `aten::silu` | 108 | 14023.510 |
| `npu::npu_apply_rotary_pos_emb` | 108 | 12713.930 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 5433 | 34922.400 |
| `aclrtLaunchKernelWithHostArgs` | 5433 | 27327.530 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21221.200 |
| `PromptFlashAttention_Tiling` | 216 | 15247.070 |
| `aclnnInplaceCopy` | 1302 | 13221.730 |
| `aclnnMul` | 978 | 8942.520 |
| `InnerPromptFlashAttention` | 108 | 5449.890 |
| `aclnnMean` | 435 | 5022.110 |
| `aclnnMatmul` | 432 | 4240.110 |
| `aclnnAdds` | 435 | 3943.140 |
| `aclnnPowTensorScalar` | 435 | 3759.870 |
| `aclnnRsqrt` | 435 | 3717.540 |
| `aclnnInnerApplyRotaryPosEmb` | 108 | 2247.630 |
| `aclnnAdd` | 225 | 2018.520 |
| `aclrtGetStreamAttribute` | 5001 | 1999.340 |
| `aclnnInnerApplyRotaryPosEmbGetWorkspaceSize` | 108 | 1528.210 |
| `aclnnSilu` | 108 | 1463.590 |
| `aclrtGetHardwareSyncAddr` | 867 | 681.430 |
| `aclrtGetResInCurrentThread` | 432 | 364.200 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 130.230 |

