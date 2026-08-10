#include <torch/extension.h>
#include <torch/library.h>

#include "npu_cpp_extension.h"

namespace paddleocr_vl_npu {
at::Tensor paddle_decode_kv_scatter_query_eager(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &cache_position,
    const at::Tensor &key_state,
    const at::Tensor &value_state)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    at::Tensor output = at_npu::native::OpPreparation::apply_tensor_without_format(query);
    output.zero_();
    EXEC_NPU_CMD_EXT(
        aclnnPaddleDecodeKvScatterQueryV1,
        query, key_cache, value_cache, cache_position, key_state, value_state,
        output);
    return output;
}

at::Tensor paddle_decode_kv_scatter_query_meta(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &cache_position,
    const at::Tensor &key_state,
    const at::Tensor &value_state)
{
    (void)key_cache;
    (void)value_cache;
    (void)cache_position;
    (void)key_state;
    (void)value_state;
    return at::empty_like(query);
}
} // namespace paddleocr_vl_npu

TORCH_LIBRARY_FRAGMENT(paddleocr_vl_npu, m)
{
    m.def(
        "paddle_decode_kv_scatter_query_eager("
        "Tensor query, Tensor(a!) key_cache, Tensor(b!) value_cache, "
        "Tensor cache_position, Tensor key_state, Tensor value_state) -> Tensor");
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, PrivateUse1, m)
{
    m.impl(
        "paddle_decode_kv_scatter_query_eager",
        &paddleocr_vl_npu::paddle_decode_kv_scatter_query_eager);
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, Meta, m)
{
    m.impl(
        "paddle_decode_kv_scatter_query_eager",
        &paddleocr_vl_npu::paddle_decode_kv_scatter_query_meta);
}
