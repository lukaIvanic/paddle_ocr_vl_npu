# Build only the GE host registration for this deliberately separate operator.
# The upstream ONNX, graph-plugin, tests, and ACLNN surfaces still describe the
# stock IncreFlashAttention name and are intentionally not part of this package.
add_subdirectory(op_host)
