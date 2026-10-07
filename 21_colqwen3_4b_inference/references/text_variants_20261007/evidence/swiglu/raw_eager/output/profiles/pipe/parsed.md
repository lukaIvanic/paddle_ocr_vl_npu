# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/swiglu/raw_eager/output/profiles/pipe/raw`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_variants_npu0_20261007_3e026452/swiglu/raw_eager/output/profiles/pipe/raw/liteserver-c001-4_101846_20261007122154499_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `226382.343 us`
- `Free`: `1168196.000 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `2625.250 us`
- `Stage`: `1394578.250 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `MatMulV3` | 324 | 59651.026 |
| `MatMulV2` | 108 | 50436.231 |
| `PromptFlashAttention` | 108 | 33552.191 |
| `Mul` | 1302 | 20221.390 |
| `Transpose` | 324 | 10641.223 |
| `Pows` | 435 | 10220.925 |
| `Cast` | 870 | 8575.492 |
| `ReduceMean` | 435 | 8424.165 |
| `Add` | 876 | 6439.984 |
| `AsStrided` | 540 | 6000.574 |
| `SwiGlu` | 108 | 4864.657 |
| `Slice` | 216 | 2312.749 |
| `ConcatD` | 216 | 2065.158 |
| `Neg` | 216 | 1707.903 |
| `Rsqrt` | 435 | 1270.526 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59651.026 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50436.231 |
| `InnerPromptFlashAttention_PromptFlashAttention_PromptFlashAttention` | 108 | 33552.191 |
| `aclnnMul_MulAiCore_Mul` | 1302 | 20221.390 |
| `aclnnPowTensorScalar_PowsAiCore_Pows` | 435 | 10220.925 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 870 | 8575.492 |
| `aclnnMean_ReduceMeanAiCore_ReduceMean` | 435 | 8424.165 |
| `aclnnMul_TransposeAiCore_Transpose` | 216 | 6807.847 |
| `aclnnAdd_AddAiCore_Add` | 441 | 5016.602 |
| `SwiGlu` | 108 | 4864.657 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 108 | 3833.376 |
| `aclnnNeg_AsStridedAiCore_AsStrided` | 216 | 2603.421 |
| `aclnnCat_AsStridedAiCore_AsStrided` | 216 | 2499.428 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 216 | 2312.749 |
| `aclnnCat_ConcatD_ConcatD` | 216 | 2065.158 |
| `aclnnNeg_NegAiCore_Neg` | 216 | 1707.903 |
| `aclnnAdds_AddAiCore_Add` | 435 | 1423.382 |
| `aclnnRsqrt_RsqrtAiCore_Rsqrt` | 435 | 1270.526 |
| `aclnnInplaceCopy_AsStridedAiCore_AsStrided` | 108 | 897.725 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 324 | 59651.026 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 108 | 50436.231 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "1274,2560;19456,2560" -> "1274,19456" | ND;ND -> ND` | 108 | 50436.231 |
| `MatMulV3 | "1274,9728;2560,9728" -> "1274,2560" | ND;ND -> ND` | 108 | 30238.315 |
| `MatMulV3 | "1274,2560;6144,2560" -> "1274,6144" | ND;ND -> ND` | 108 | 16717.431 |
| `MatMulV3 | "1274,4096;2560,4096" -> "1274,2560" | ND;ND -> ND` | 108 | 12695.280 |

### TransData Names
_No rows._

### TransData Shape And Format Signatures
_No rows._

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 476.510 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.810 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 473.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 472.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 471.609 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.850 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.489 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 470.270 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.949 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.630 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.629 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.609 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.309 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.129 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.109 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.050 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.049 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 469.030 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.970 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.929 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.929 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.809 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.710 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.710 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.689 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.569 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.470 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.329 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.189 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.170 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.170 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.130 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.109 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 468.069 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.950 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.949 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.869 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.829 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 467.789 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `colqwen.text.transformer` | 3 | 1393944.680 |
| `colqwen.raw_eager.text_forward.step2` | 1 | 476302.590 |
| `colqwen.raw_eager.text_forward.step0` | 1 | 461614.700 |
| `colqwen.raw_eager.text_forward.step1` | 1 | 457194.570 |
| `empty_tensor` | 5757 | 220739.600 |
| `aten::to` | 870 | 217439.380 |
| `aten::_to_copy` | 870 | 182415.170 |
| `aten::mul` | 1302 | 163162.160 |
| `aten::linear` | 432 | 146892.298 |
| `aten::add` | 876 | 141413.320 |
| `aten::reshape` | 1296 | 114170.340 |
| `aten::matmul` | 432 | 113792.499 |
| `aclnnMatmul` | 432 | 110087.260 |
| `aten::copy_` | 1086 | 98444.310 |
| `aten::pow` | 435 | 74063.720 |
| `aten::empty` | 870 | 67445.460 |
| `aten::as_strided` | 1836 | 65326.110 |
| `aten::transpose` | 864 | 64173.800 |
| `aten::chunk` | 216 | 61418.880 |
| `aten::mean` | 435 | 58618.790 |
| `aten::rsqrt` | 435 | 54452.030 |
| `aclnnMul` | 1302 | 53946.797 |
| `aten::split` | 216 | 53035.590 |
| `aten::view` | 1296 | 49574.190 |
| `aten::t` | 432 | 49166.280 |
| `npu::npu_prompt_flash_attention` | 108 | 46026.690 |
| `aclnnInplaceCopy` | 1086 | 45593.055 |
| `aten::narrow` | 432 | 45435.420 |
| `aten::contiguous` | 216 | 42211.960 |
| `aten::clone` | 216 | 34244.050 |
| `aclnnPromptFlashAttentionV3` | 108 | 33552.190 |
| `aten::item` | 435 | 30630.610 |
| `aten::slice` | 432 | 30478.080 |
| `aten::cat` | 216 | 27692.420 |
| `aten::neg` | 216 | 26951.040 |
| `aten::_reshape_alias` | 324 | 21782.440 |
| `aclnnPowTensorScalar` | 435 | 18367.280 |
| `npu::npu_swiglu` | 108 | 18186.300 |
| `aclnnMean` | 435 | 18175.230 |
| `aclnnAdds` | 435 | 18116.160 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `launch` | 6513 | 43112.140 |
| `aclrtLaunchKernelWithHostArgs` | 6513 | 34030.530 |
| `InnerPromptFlashAttentionGetWorkspaceSize` | 108 | 21560.430 |
| `PromptFlashAttention_Tiling` | 216 | 15279.580 |
| `aclnnMul` | 1302 | 14196.840 |
| `aclnnInplaceCopy` | 1086 | 12542.890 |
| `InnerPromptFlashAttention` | 108 | 7860.710 |
| `aclnnMean` | 435 | 5605.560 |
| `aclnnMatmul` | 432 | 5097.780 |
| `aclnnAdds` | 435 | 4427.530 |
| `aclnnAdd` | 441 | 4370.400 |
| `aclnnPowTensorScalar` | 435 | 4368.860 |
| `aclnnRsqrt` | 435 | 4257.060 |
| `aclnnNeg` | 216 | 3424.990 |
| `aclnnCat` | 216 | 3235.040 |
| `aclnnSwiGlu` | 108 | 2556.120 |
| `aclrtGetStreamAttribute` | 5649 | 2309.460 |
| `aclnnSwiGluGetWorkspaceSize` | 108 | 1726.620 |
| `aclrtGetHardwareSyncAddr` | 867 | 636.070 |
| `aclrtGetResInCurrentThread` | 432 | 377.510 |
| `aclrtSynchronizeDeviceWithTimeout` | 7 | 160.480 |

