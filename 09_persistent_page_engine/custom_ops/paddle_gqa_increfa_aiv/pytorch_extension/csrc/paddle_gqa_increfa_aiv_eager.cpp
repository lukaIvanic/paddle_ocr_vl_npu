#include <cstring>

#include <torch/extension.h>
#include <torch/library.h>

#include "npu_cpp_extension.h"

namespace paddleocr_vl_npu {
namespace {
constexpr int64_t kExpectedQueryHeads = 16;
constexpr int64_t kExpectedKvHeads = 2;
constexpr int64_t kExpectedHeadDim = 128;
constexpr size_t kLayoutCapacity = 20;

void check_contract(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &atten_mask, int64_t num_heads,
    int64_t num_key_value_heads, int64_t inner_precise,
    int64_t vector_core_count)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "Paddle GQA IncreFA AIV eager requires NPU tensors");
    TORCH_CHECK(key.device() == query.device() && value.device() == query.device() &&
                    atten_mask.device() == query.device(),
                "Paddle GQA IncreFA AIV eager requires one NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf && key.scalar_type() == at::kHalf &&
                    value.scalar_type() == at::kHalf,
                "Paddle GQA IncreFA AIV eager requires FP16 Q/K/V");
    TORCH_CHECK(atten_mask.scalar_type() == at::kBool,
                "Paddle GQA IncreFA AIV eager requires a bool mask");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 && atten_mask.dim() == 4,
                "Paddle GQA IncreFA AIV eager requires rank-4 BNSD tensors");
    TORCH_CHECK(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1,
                "Paddle GQA IncreFA AIV eager supports B1 only");
    TORCH_CHECK(query.size(2) == 1, "Paddle GQA IncreFA AIV eager requires one query token");
    TORCH_CHECK(key.sizes() == value.sizes(), "Paddle GQA IncreFA AIV eager requires matching K/V");
    TORCH_CHECK(num_heads == kExpectedQueryHeads && query.size(1) == kExpectedQueryHeads,
                "Paddle GQA IncreFA AIV eager requires 16 query heads");
    TORCH_CHECK(num_key_value_heads == kExpectedKvHeads && key.size(1) == kExpectedKvHeads,
                "Paddle GQA IncreFA AIV eager requires 2 KV heads");
    TORCH_CHECK(query.size(3) == kExpectedHeadDim && key.size(3) == kExpectedHeadDim,
                "Paddle GQA IncreFA AIV eager requires head_dim=128");
    TORCH_CHECK(inner_precise == 1, "Paddle GQA IncreFA AIV eager fixes inner_precise=1");
    TORCH_CHECK(vector_core_count >= 1 && vector_core_count <= 48,
                "vector_core_count must be in [1, 48]");
}
} // namespace

at::Tensor paddle_gqa_incre_flash_attention_aiv_eager(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &atten_mask, int64_t num_heads, int64_t num_key_value_heads,
    double scale_value, int64_t inner_precise, int64_t vector_core_count)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    check_contract(query, key, value, atten_mask, num_heads, num_key_value_heads,
                   inner_precise, vector_core_count);
    at::Tensor output = at_npu::native::OpPreparation::apply_tensor_without_format(query);
    // Poison-protect the validation lane: the upstream all-vector GQA bug left
    // most query heads unwritten, and allocator reuse could otherwise retain a
    // preceding stock result and create a false exact match.
    output.zero_();
    at::TensorList key_tensors = key;
    at::TensorList value_tensors = value;
    const c10::optional<at::Tensor> no_tensor = c10::nullopt;
    const c10::OptionalArrayRef<c10::SymInt> no_sequence_lengths = c10::nullopt;
    int64_t block_size = 0;
    char input_layout[kLayoutCapacity] = {};
    std::strncpy(input_layout, "BNSD", kLayoutCapacity - 1);

    EXEC_NPU_CMD_EXT(
        aclnnPaddleGqaIncreFlashAttentionAiv,
        query, key_tensors, value_tensors, no_tensor, atten_mask,
        no_sequence_lengths, no_tensor, no_tensor, no_tensor, no_tensor,
        no_tensor, no_tensor, no_tensor, no_tensor, no_tensor,
        num_heads, scale_value, input_layout, num_key_value_heads,
        block_size, inner_precise, vector_core_count, output);
    return output;
}

at::Tensor paddle_gqa_incre_flash_attention_aiv_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &atten_mask, int64_t num_heads, int64_t num_key_value_heads,
    double scale_value, int64_t inner_precise, int64_t vector_core_count)
{
    (void)key; (void)value; (void)atten_mask; (void)num_heads;
    (void)num_key_value_heads; (void)scale_value; (void)inner_precise;
    (void)vector_core_count;
    return at::empty_like(query);
}
} // namespace paddleocr_vl_npu

TORCH_LIBRARY_FRAGMENT(paddleocr_vl_npu, m)
{
    m.def(
        "paddle_gqa_incre_flash_attention_aiv_eager("
        "Tensor query, Tensor key, Tensor value, Tensor atten_mask, "
        "int num_heads, int num_key_value_heads, float scale_value, "
        "int inner_precise=1, int vector_core_count=48) -> Tensor");
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, PrivateUse1, m)
{
    m.impl("paddle_gqa_incre_flash_attention_aiv_eager",
           &paddleocr_vl_npu::paddle_gqa_incre_flash_attention_aiv_eager);
}

TORCH_LIBRARY_IMPL(paddleocr_vl_npu, Meta, m)
{
    m.impl("paddle_gqa_incre_flash_attention_aiv_eager",
           &paddleocr_vl_npu::paddle_gqa_incre_flash_attention_aiv_meta);
}
