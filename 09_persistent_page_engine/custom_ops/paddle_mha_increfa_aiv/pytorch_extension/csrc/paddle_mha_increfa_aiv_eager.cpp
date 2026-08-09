/**
 * Direct-eager PyTorch bridge for the separately packaged Paddle MHA AIV op.
 *
 * The aclnn enqueue pattern follows Ascend/op-plugin's official
 * npu_incre_flash_attention adapter.  This bridge intentionally calls the
 * separately generated aclnnInnerPaddleMhaIncreFlashAttentionAiv symbol.
 */

#include <cstring>

#include <torch/extension.h>
#include <torch/library.h>

#include "npu_cpp_extension.h"

namespace paddleocr_vl_npu {
namespace {

constexpr int64_t kExpectedHeads = 16;
constexpr int64_t kExpectedHeadDim = 128;
constexpr size_t kLayoutCapacity = 20;

void check_contract(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &atten_mask,
    int64_t num_heads,
    int64_t inner_precise)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "Paddle MHA IncreFA AIV eager requires NPU tensors");
    TORCH_CHECK(key.device() == query.device() && value.device() == query.device() &&
                    atten_mask.device() == query.device(),
                "Paddle MHA IncreFA AIV eager requires one NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf && key.scalar_type() == at::kHalf &&
                    value.scalar_type() == at::kHalf,
                "Paddle MHA IncreFA AIV eager requires FP16 Q/K/V");
    TORCH_CHECK(atten_mask.scalar_type() == at::kBool,
                "Paddle MHA IncreFA AIV eager requires a bool mask");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
                    atten_mask.dim() == 4,
                "Paddle MHA IncreFA AIV eager requires rank-4 BNSD tensors");
    TORCH_CHECK(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1,
                "Paddle MHA IncreFA AIV eager currently supports B1 only");
    TORCH_CHECK(query.size(2) == 1,
                "Paddle MHA IncreFA AIV eager requires one query token");
    TORCH_CHECK(key.sizes() == value.sizes(),
                "Paddle MHA IncreFA AIV eager requires matching K/V shapes");
    TORCH_CHECK(num_heads == kExpectedHeads && query.size(1) == num_heads &&
                    key.size(1) == num_heads,
                "Paddle MHA IncreFA AIV eager requires 16 equal Q/KV heads");
    TORCH_CHECK(query.size(3) == kExpectedHeadDim && key.size(3) == kExpectedHeadDim,
                "Paddle MHA IncreFA AIV eager requires head_dim=128");
    TORCH_CHECK(inner_precise == 1,
                "Paddle MHA IncreFA AIV eager fixes inner_precise=1");
}

} // namespace

at::Tensor paddle_mha_incre_flash_attention_aiv_eager(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &atten_mask,
    int64_t num_heads,
    double scale_value,
    int64_t inner_precise)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    check_contract(query, key, value, atten_mask, num_heads, inner_precise);

    at::Tensor output =
        at_npu::native::OpPreparation::apply_tensor_without_format(query);
    at::TensorList key_tensors = key;
    at::TensorList value_tensors = value;
    const c10::optional<at::Tensor> no_tensor = c10::nullopt;
    const c10::OptionalArrayRef<c10::SymInt> no_sequence_lengths = c10::nullopt;
    int64_t num_key_value_heads = 0;
    int64_t block_size = 0;
    char input_layout[kLayoutCapacity] = {};
    std::strncpy(input_layout, "BNSD", kLayoutCapacity - 1);

    EXEC_NPU_CMD_EXT(
        aclnnInnerPaddleMhaIncreFlashAttentionAiv,
        query,
        key_tensors,
        value_tensors,
        no_tensor,
        atten_mask,
        no_sequence_lengths,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        num_heads,
        scale_value,
        input_layout,
        num_key_value_heads,
        block_size,
        inner_precise,
        output);
    return output;
}

at::Tensor paddle_mha_incre_flash_attention_aiv_meta(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &atten_mask,
    int64_t num_heads,
    double scale_value,
    int64_t inner_precise)
{
    (void)key;
    (void)value;
    (void)atten_mask;
    (void)num_heads;
    (void)scale_value;
    (void)inner_precise;
    return at::empty_like(query);
}

} // namespace paddleocr_vl_npu

TORCH_LIBRARY_FRAGMENT(paddleocr_vl_npu, m)
{
    m.def(
        "paddle_mha_incre_flash_attention_aiv_eager("
        "Tensor query, Tensor key, Tensor value, Tensor atten_mask, "
        "int num_heads, float scale_value, int inner_precise=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, PrivateUse1, m)
{
    m.impl(
        "paddle_mha_incre_flash_attention_aiv_eager",
        &paddleocr_vl_npu::paddle_mha_incre_flash_attention_aiv_eager);
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, Meta, m)
{
    m.impl(
        "paddle_mha_incre_flash_attention_aiv_eager",
        &paddleocr_vl_npu::paddle_mha_incre_flash_attention_aiv_meta);
}
