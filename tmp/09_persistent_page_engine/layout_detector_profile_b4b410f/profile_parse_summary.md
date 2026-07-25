# NPU Profile Summary

profile_dir: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/layout_detector_profile_b4b410f`
runs: `1`

## Run 1
run_root: `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/layout_detector_profile_b4b410f/liteserver-c001-4_448345_20260725215022241_ascend_pt`

### Step Trace Totals
- `Bubble`: `0.000 us`
- `Communication`: `0.000 us`
- `Communication(Not Overlapped and Exclude Receive)`: `0.000 us`
- `Communication(Not Overlapped)`: `0.000 us`
- `Computing`: `23402.000 us`
- `Free`: `121.820 us`
- `Overlapped`: `0.000 us`
- `Preparing`: `341.250 us`
- `Stage`: `23523.750 us`

### Kernel Types
| name | count | total_us |
|---|---:|---:|
| `TransData` | 447 | 6277.520 |
| `Conv2D` | 130 | 2905.020 |
| `Transpose` | 112 | 1974.040 |
| `Add` | 234 | 1893.040 |
| `Mul` | 268 | 1860.420 |
| `MatMulV2` | 109 | 1080.940 |
| `GridSample` | 18 | 1008.020 |
| `BNInfer` | 49 | 873.580 |
| `Relu` | 83 | 680.520 |
| `MemSet` | 29 | 473.080 |
| `ReduceSum` | 6 | 467.860 |
| `Index` | 1 | 380.000 |
| `LayerNormV3` | 23 | 351.540 |
| `FlashAttentionScore` | 7 | 302.960 |
| `Swish` | 31 | 279.020 |
| `ConcatD` | 13 | 275.500 |
| `Sub` | 95 | 178.900 |
| `ResizeBilinearV2` | 6 | 152.440 |
| `Slice` | 40 | 150.240 |
| `BatchMatMulV2` | 3 | 141.060 |
| `SoftmaxV2` | 6 | 128.700 |
| `UpsampleNearest3d` | 2 | 124.780 |
| `MaxPool3DWithArgmaxV2` | 1 | 122.620 |
| `Pack` | 9 | 116.260 |
| `PadV3` | 2 | 115.360 |
| `Cast` | 26 | 115.200 |
| `Rsqrt` | 80 | 102.700 |
| `SelectV2` | 2 | 101.000 |
| `ArgMaxWithValue` | 3 | 99.040 |
| `ArgMinWithValue` | 2 | 82.960 |

### Kernel Names
| name | count | total_us |
|---|---:|---:|
| `aclnnConvolution_TransData_TransData` | 363 | 4076.780 |
| `aclnnConvolution_Conv2dWithFlag_Conv2D` | 130 | 2905.020 |
| `aclnnMul_MulAiCore_Mul` | 256 | 1782.520 |
| `aclnnAdd_AddAiCore_Add` | 150 | 1761.120 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 27 | 1572.660 |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 108 | 1070.900 |
| `aclnnGridSampler2D_TransposeAiCore_Transpose` | 54 | 1063.380 |
| `aclnnGridSampler2D_GridSample_GridSample` | 18 | 1008.020 |
| `aclnnBatchNorm_BNInfer_BNInfer` | 49 | 873.580 |
| `aclnnRelu_Relu_Relu` | 83 | 680.520 |
| `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum` | 6 | 467.860 |
| `aclnnConvolution_TransDataFzToDst_MemSet` | 27 | 441.740 |
| `Index` | 1 | 380.000 |
| `aclnnConvolution_TransDataToFzWithoutGroup_TransData` | 27 | 362.020 |
| `aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3` | 23 | 351.540 |
| `aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore` | 7 | 302.960 |
| `aclnnFlashAttentionScore_TransposeAiCore_Transpose` | 21 | 299.180 |
| `aclnnSilu_SiluAiCore_Swish` | 31 | 279.020 |
| `aclnnCat_ConcatD_ConcatD` | 13 | 275.500 |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 15 | 219.940 |
| `aclnnGather_TransposeAiCore_Transpose` | 9 | 167.940 |
| `aclnnUpsampleBilinear2d_ResizeBilinearV2AICORE_ResizeBilinearV2` | 6 | 152.440 |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | 2 | 132.300 |
| `aclnnAdds_AddAiCore_Add` | 84 | 131.920 |
| `aclnnSoftmax_SoftmaxAiCore_SoftmaxV2` | 6 | 128.700 |
| `aclnnUpsampleNearest2dV2_UpsampleNearest3dNcdhw_UpsampleNearest3d` | 2 | 124.780 |
| `aclnnSub_SubAiCore_Sub` | 82 | 122.720 |
| `aclnnMaxPool2dWithMask_MaxPool3DWithArgmaxV2NcdhwAiCore_MaxPool3DWithArgmaxV2` | 1 | 122.620 |
| `aclnnStack_PackAiCore_Pack` | 9 | 116.260 |
| `aclnnConstantPadNd_PadV3AiCore_PadV3` | 2 | 115.360 |

### MatMul Names
| name | count | total_us |
|---|---:|---:|
| `aclnnAddmm_MatMulCommon_MatMulV2` | 108 | 1070.900 |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | 2 | 132.300 |
| `aclnnMatmul_TransposeAiCore_Transpose` | 2 | 26.200 |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 1 | 25.500 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 1 | 10.040 |
| `aclnnMatmul_BatchMatMulNd_BatchMatMulV2` | 1 | 8.760 |
| `aclnnMatmul_SliceAiCore_Slice` | 2 | 6.680 |

### MatMul Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `MatMulV2 | "300,256;256,256;256" -> "300,256" | ND;ND;ND -> ND` | 47 | 290.100 |
| `MatMulV2 | "13125,256;256,256;256" -> "13125,256" | ND;ND;ND -> ND` | 9 | 270.180 |
| `BatchMatMulV2 | "1,300,32;1,32,40000" -> "1,300,40000" | ND;ND -> ND` | 2 | 132.300 |
| `MatMulV2 | "300,4;512,4;512" -> "300,512" | ND;ND;ND -> ND` | 6 | 122.560 |
| `MatMulV2 | "300,1024;256,1024;256" -> "300,256" | ND;ND;ND -> ND` | 6 | 66.520 |
| `MatMulV2 | "300,256;1024,256;1024" -> "300,1024" | ND;ND;ND -> ND` | 6 | 52.180 |
| `MatMulV2 | "300,256;96,256;96" -> "300,96" | ND;ND;ND -> ND` | 6 | 49.840 |
| `MatMulV2 | "300,512;256,512;256" -> "300,256" | ND;ND;ND -> ND` | 6 | 46.860 |
| `MatMulV2 | "300,256;192,256;192" -> "300,192" | ND;ND;ND -> ND` | 6 | 41.240 |
| `MatMulV2 | "300,256;4,256;4" -> "300,4" | ND;ND;ND -> ND` | 6 | 35.040 |
| `MatMulV2 | "625,256;256,256;256" -> "625,256" | ND;ND;ND -> ND` | 3 | 29.340 |
| `MatMulV3 | "13125,256;25,256;25" -> "13125,25" | ND;ND;ND -> ND` | 1 | 25.500 |
| `MatMulV2 | "625,256;1024,256;1024" -> "625,1024" | ND;ND;ND -> ND` | 1 | 15.400 |
| `Transpose | "256,625;2" -> "625,256" | NCL;ND -> NCL` | 1 | 14.400 |
| `MatMulV2 | "13125,256;4,256;4" -> "13125,4" | ND;ND;ND -> ND` | 1 | 14.360 |
| `MatMulV2 | "300,256;32,256;32" -> "300,32" | ND;ND;ND -> ND` | 2 | 14.040 |
| `MatMulV2 | "625,1024;256,1024;256" -> "625,256" | ND;ND;ND -> ND` | 1 | 12.860 |
| `Transpose | "300,64;2" -> "64,300" | ND;ND -> ND` | 1 | 11.800 |
| `MatMulV2 | "625,256;256,256" -> "625,256" | ND;ND -> ND` | 1 | 10.040 |
| `BatchMatMulV2 | "1,300,64;1,64,300" -> "1,300,300" | ND;ND -> ND` | 1 | 8.760 |
| `Slice | "300,128;2;2" -> "300,64" | ND;ND;ND -> ND` | 2 | 6.680 |
| `MatMulV2 | "300,256;128,256;128" -> "300,128" | ND;ND;ND -> ND` | 1 | 5.800 |
| `MatMulV2 | "300,256;25,256;25" -> "300,25" | ND;ND;ND -> ND` | 1 | 4.580 |

### TransData Names
| name | count | total_us |
|---|---:|---:|
| `aclnnConvolution_TransData_TransData` | 363 | 4076.780 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 27 | 1572.660 |
| `aclnnConvolution_TransDataFzToDst_MemSet` | 27 | 441.740 |
| `aclnnConvolution_TransDataToFzWithoutGroup_TransData` | 27 | 362.020 |
| `aclnnUpsampleNearest2dV2_TransDataSpecial_TransData` | 8 | 109.180 |
| `aclnnUpsampleBilinear2d_TransDataSpecial_TransData` | 12 | 80.740 |
| `aclnnUpsampleBilinear2d_TransData_TransData` | 6 | 55.720 |
| `aclnnUpsampleNearest2dV2_TransData_TransData` | 4 | 20.420 |

### TransData Shape And Format Signatures
| name | count | total_us |
|---|---:|---:|
| `TransData | "25,12,16,8" -> "600,1,16,8" | FRACTAL_Z:1 -> FRACTAL_Z:192` | 18 | 913.240 |
| `TransData | "25,24,16,8" -> "1200,1,16,8" | FRACTAL_Z:1 -> FRACTAL_Z:384` | 6 | 506.500 |
| `TransData | "256,256,3,3" -> "288,16,16,8" | NCHW -> FRACTAL_Z:1` | 14 | 475.660 |
| `MemSet | N/A -> N/A | N/A -> N/A` | 27 | 441.740 |
| `TransData | "192,1,5,5" -> "25,12,16,8" | NCHW -> FRACTAL_Z:1` | 18 | 206.200 |
| `TransData | "1,32,50,50,8" -> "1,256,50,50" | NC1HWC0:1 -> NCHW` | 20 | 149.260 |
| `TransData | "1,256,100,100" -> "1,32,100,100,8" | NCHW -> NC1HWC0:1` | 10 | 145.380 |
| `TransData | "1,32,100,100,8" -> "1,256,100,100" | NC1HWC0:1 -> NCHW` | 11 | 140.120 |
| `TransData | "1,192,50,50" -> "1,24,50,50,8" | NCHW -> NC1HWC0:1` | 15 | 138.020 |
| `TransData | "1,24,50,50,8" -> "1,192,50,50" | NC1HWC0:192 -> NCHW` | 18 | 131.880 |
| `TransData | "1,256,50,50" -> "1,32,50,50,8" | NCHW -> NC1HWC0:1` | 16 | 131.740 |
| `TransData | "1,192,50,50" -> "1,24,50,50,8" | NCHW -> NC1HWC0:192` | 18 | 131.480 |
| `TransData | "1,24,50,50,8" -> "1,192,50,50" | NC1HWC0:1 -> NCHW` | 18 | 125.540 |
| `TransData | "384,1,5,5" -> "25,24,16,8" | NCHW -> FRACTAL_Z:1` | 6 | 108.540 |
| `TransData | "256,256,1,1" -> "32,16,16,8" | NCHW -> FRACTAL_Z:1` | 17 | 104.700 |
| `TransData | "1,1,800,800" -> "1,1,800,800,16" | NCHW -> NC1HWC0` | 4 | 91.560 |
| `TransData | "1,512,100,100" -> "1,64,100,100,8" | NCHW -> NC1HWC0:1` | 3 | 81.660 |
| `TransData | "192,192,1,1" -> "24,12,16,8" | NCHW -> FRACTAL_Z:1` | 15 | 80.140 |
| `TransData | "1,512,50,50" -> "1,64,50,50,8" | NCHW -> NC1HWC0:1` | 8 | 79.360 |
| `TransData | "9,64,16,8" -> "1152,1,16,8" | FRACTAL_Z:1 -> FRACTAL_Z:1024` | 1 | 78.980 |
| `TransData | "1,32,25,25,8" -> "1,256,25,25" | NC1HWC0:1 -> NCHW` | 12 | 78.880 |
| `TransData | "1,384,25,25" -> "1,48,25,25,8" | NCHW -> NC1HWC0:1` | 5 | 76.980 |
| `TransData | "1,48,200,200" -> "1,6,200,200,8" | NCHW -> NC1HWC0:1` | 6 | 76.360 |
| `TransData | "96,96,3,3" -> "108,6,16,8" | NCHW -> FRACTAL_Z:1` | 5 | 76.300 |
| `TransData | "1,6,200,200,8" -> "1,48,200,200" | NC1HWC0:1 -> NCHW` | 7 | 73.300 |
| `TransData | "1,48,25,25,8" -> "1,384,25,25" | NC1HWC0:384 -> NCHW` | 6 | 73.160 |
| `TransData | "1,96,100,100" -> "1,12,100,100,8" | NCHW -> NC1HWC0:1` | 5 | 70.660 |
| `TransData | "1,256,25,25" -> "1,32,25,25,8" | NCHW -> NC1HWC0:1` | 9 | 68.280 |
| `TransData | "1,48,25,25,8" -> "1,384,25,25" | NC1HWC0:1 -> NCHW` | 6 | 63.940 |
| `TransData | "256,512,1,1" -> "64,16,16,8" | NCHW -> FRACTAL_Z:1` | 9 | 62.360 |

### Suspect Kernels
| name | count | total_us |
|---|---:|---:|
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 85.940 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 85.020 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 84.940 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 84.080 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 83.380 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 83.140 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 78.980 |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | 1 | 69.300 |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | 1 | 63.000 |
| `aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore` | 1 | 60.420 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 52.180 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 51.400 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 51.380 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 51.320 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 51.080 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 51.060 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.960 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.860 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.740 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.660 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.640 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.460 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.380 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.200 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.060 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.040 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 50.020 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 49.800 |
| `aclnnConvolution_TransDataFzToDst_TransData` | 1 | 49.020 |
| `aclnnBatchNorm_BNInfer_BNInfer` | 1 | 46.840 |


### Operators
| name | count | total_us |
|---|---:|---:|
| `paddle_ocr_vl.layout_detector_graph_replay` | 1 | 32.900 |
| `aten::copy_` | 1 | 32.900 |
| `aclnnInplaceCopy` | 1 | 32.900 |


### APIs
| name | count | total_us |
|---|---:|---:|
| `aclrtSynchronizeDeviceWithTimeout` | 2 | 23341.020 |
| `aclnnInplaceCopy` | 1 | 65.220 |
| `launch` | 1 | 41.280 |
| `aclrtLaunchKernelWithHostArgs` | 1 | 32.960 |
| `aclmdlRIExecuteAsync` | 1 | 27.390 |
| `aclmdlRICaptureGetInfo` | 1 | 2.660 |
| `aclrtGetStreamAttribute` | 1 | 2.030 |

