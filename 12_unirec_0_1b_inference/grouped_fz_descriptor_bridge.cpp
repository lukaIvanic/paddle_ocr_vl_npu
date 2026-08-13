#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace {

constexpr int64_t kFractalZPrimaryFormat = 4;
constexpr int64_t kFormatSubformatShift = 8;

torch_npu::NPUStorageImpl* get_npu_storage(const at::Tensor& tensor) {
  // The installed torch-npu package hides NPUBridge's symbols, but an NPU
  // tensor's StorageImpl is concretely NPUStorageImpl. The device check at each
  // call site makes this downcast equivalent to NPUBridge::GetNpuStorageImpl.
  return static_cast<torch_npu::NPUStorageImpl*>(
      tensor.storage().unsafeGetStorageImpl());
}

at::Tensor wrap_grouped_fz(
    at::Tensor packed,
    const std::vector<int64_t>& logical_shape,
    int64_t groups) {
  TORCH_CHECK(
      packed.device().type() == c10::DeviceType::PrivateUse1,
      "packed grouped-FZ storage must be an NPU tensor");
  TORCH_CHECK(packed.scalar_type() == at::ScalarType::Half,
              "grouped-FZ bridge currently requires float16 storage");
  TORCH_CHECK(packed.dim() == 4, "packed grouped-FZ storage must be 4D");
  TORCH_CHECK(logical_shape.size() == 4, "logical weight shape must be 4D");
  TORCH_CHECK(groups > 1 && groups <= 65535, "invalid grouped-FZ group count");
  TORCH_CHECK(packed.storage_offset() == 0, "packed storage must have zero offset");
  TORCH_CHECK(packed.is_contiguous(), "packed storage must be contiguous");

  const std::vector<int64_t> storage_shape = packed.sizes().vec();
  const int64_t encoded_format =
      kFractalZPrimaryFormat | (groups << kFormatSubformatShift);

  // Match torch-npu's own create_tensor_with_format_and_shape sequence, but
  // retain the caller-provided physical storage shape. FormatHelper cannot do
  // this itself: its public FRACTAL_Z shape inference has no group parameter.
  packed.unsafeGetTensorImpl()->set_sizes_contiguous(logical_shape);
  packed.unsafeGetTensorImpl()->empty_tensor_restride(
      c10::MemoryFormat::Contiguous);

  auto* storage = get_npu_storage(packed);
  TORCH_CHECK(storage != nullptr, "failed to obtain NPU storage descriptor");
  auto& descriptor = storage->npu_desc_;
  descriptor.base_sizes_ = logical_shape;
  descriptor.base_strides_ = packed.strides().vec();
  descriptor.storage_sizes_ = storage_shape;
  descriptor.origin_format_ = ACL_FORMAT_NCHW;
  descriptor.npu_format_ = static_cast<aclFormat>(encoded_format);

  return packed;
}

std::tuple<int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>>
describe_npu_storage(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      "descriptor inspection requires an NPU tensor");
  const auto* storage = get_npu_storage(tensor);
  TORCH_CHECK(storage != nullptr, "failed to obtain NPU storage descriptor");
  const auto& descriptor = storage->npu_desc_;
  return {
      static_cast<int64_t>(descriptor.origin_format_),
      static_cast<int64_t>(descriptor.npu_format_),
      std::vector<int64_t>(
          descriptor.base_sizes_.begin(), descriptor.base_sizes_.end()),
      std::vector<int64_t>(
          descriptor.storage_sizes_.begin(), descriptor.storage_sizes_.end()),
  };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "wrap_grouped_fz",
      &wrap_grouped_fz,
      "Attach an exact FRACTAL_Z:<groups> descriptor to prepacked NPU storage");
  module.def(
      "describe_npu_storage",
      &describe_npu_storage,
      "Return origin format, storage format, logical shape, and storage shape");
}
