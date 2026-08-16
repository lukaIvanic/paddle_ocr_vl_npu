#include <torch/extension.h>
#include <torch/library.h>

#include "npu_cpp_extension.h"

namespace unirec_layout {
namespace {

void check_contract(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_locations,
    const at::Tensor &attention_weights)
{
    TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1,
                "layout MSDA ACLNN requires NPU tensors");
    TORCH_CHECK(spatial_shapes.device() == value.device() &&
                    level_start_index.device() == value.device() &&
                    sampling_locations.device() == value.device() &&
                    attention_weights.device() == value.device(),
                "layout MSDA ACLNN requires all inputs on one NPU");
    TORCH_CHECK(value.dim() == 4, "value must have shape [B,S,H,D]");
    TORCH_CHECK(spatial_shapes.dim() == 2 && spatial_shapes.size(1) == 2,
                "spatial_shapes must have shape [L,2]");
    TORCH_CHECK(level_start_index.dim() == 1,
                "level_start_index must have shape [L]");
    TORCH_CHECK(sampling_locations.dim() == 6 &&
                    sampling_locations.size(5) == 2,
                "sampling_locations must have shape [B,Q,H,L,P,2]");
    TORCH_CHECK(attention_weights.dim() == 5,
                "attention_weights must have shape [B,Q,H,L,P]");
    TORCH_CHECK(value.scalar_type() == at::kHalf ||
                    value.scalar_type() == at::kFloat ||
                    value.scalar_type() == at::kBFloat16,
                "value must be FP16, FP32, or BF16");
    TORCH_CHECK(sampling_locations.scalar_type() == value.scalar_type() &&
                    attention_weights.scalar_type() == value.scalar_type(),
                "value, sampling_locations, and attention_weights must share dtype");
    TORCH_CHECK(spatial_shapes.scalar_type() == at::kInt ||
                    spatial_shapes.scalar_type() == at::kLong,
                "spatial_shapes must be INT32 or INT64");
    TORCH_CHECK(level_start_index.scalar_type() == spatial_shapes.scalar_type(),
                "level_start_index must share spatial_shapes dtype");
    TORCH_CHECK(spatial_shapes.size(0) == level_start_index.size(0),
                "spatial metadata level counts must match");
    TORCH_CHECK(value.size(0) == sampling_locations.size(0) &&
                    value.size(0) == attention_weights.size(0),
                "batch dimensions must match");
    TORCH_CHECK(value.size(2) == sampling_locations.size(2) &&
                    value.size(2) == attention_weights.size(2),
                "head dimensions must match");
    TORCH_CHECK(sampling_locations.size(1) == attention_weights.size(1) &&
                    sampling_locations.size(3) == attention_weights.size(3) &&
                    sampling_locations.size(4) == attention_weights.size(4),
                "sampling and attention dimensions must match");
    TORCH_CHECK(sampling_locations.size(3) == spatial_shapes.size(0),
                "sampling level count must match spatial_shapes");
}

} // namespace

at::Tensor layout_msda_aclnn(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_locations,
    const at::Tensor &attention_weights)
{
    const c10::OptionalDeviceGuard device_guard(device_of(value));
    check_contract(value, spatial_shapes, level_start_index,
                   sampling_locations, attention_weights);
    const int64_t batch = value.size(0);
    const int64_t queries = sampling_locations.size(1);
    const int64_t channels = value.size(2) * value.size(3);
    at::Tensor output =
        at_npu::native::OpPreparation::apply_tensor_without_format(
            {batch, queries, channels}, value.options());

    EXEC_NPU_CMD_EXT(
        aclnnMultiScaleDeformableAttnFunction,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output);
    return output;
}

at::Tensor layout_msda_aclnn_meta(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_locations,
    const at::Tensor &attention_weights)
{
    (void)spatial_shapes;
    (void)level_start_index;
    (void)attention_weights;
    return at::empty(
        {value.size(0), sampling_locations.size(1),
         value.size(2) * value.size(3)},
        value.options().device(at::kMeta));
}

} // namespace unirec_layout

TORCH_LIBRARY_FRAGMENT(unirec_layout, m)
{
    m.def(
        "msda_aclnn(Tensor value, Tensor spatial_shapes, "
        "Tensor level_start_index, Tensor sampling_locations, "
        "Tensor attention_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(unirec_layout, PrivateUse1, m)
{
    m.impl("msda_aclnn", &unirec_layout::layout_msda_aclnn);
}

TORCH_LIBRARY_IMPL(unirec_layout, Meta, m)
{
    m.impl("msda_aclnn", &unirec_layout::layout_msda_aclnn_meta);
}
