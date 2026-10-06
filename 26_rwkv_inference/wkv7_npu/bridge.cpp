// Eager ACLNN bridge follows the parent repo's validated npu_cpp_extension pattern.
#include <torch/extension.h>
#include <torch/library.h>
#include "npu_cpp_extension.h"

using Pair = std::tuple<at::Tensor, at::Tensor>;

Pair wkv7(const at::Tensor& k, const at::Tensor& v, const at::Tensor& w,
          const at::Tensor& r, const at::Tensor& a, const at::Tensor& b,
          const at::Tensor& hi) {
    const c10::OptionalDeviceGuard guard(device_of(k));
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1,
                "WKV7 requires NPU inputs");
    TORCH_CHECK(k.dim() == 4 && k.size(0) > 0 && k.size(1) > 0 &&
                k.size(2) > 0 && k.size(2) <= 2048 && k.size(3) == 64,
                "Expected nonempty [B,H,T<=2048,64]");
    for (const auto& tensor : {k, v, w, r, a, b, hi}) {
        TORCH_CHECK(tensor.device() == k.device() && tensor.scalar_type() == at::kFloat &&
                    tensor.is_contiguous(), "Expected contiguous FP32 inputs on one NPU");
    }
    for (const auto& tensor : {v, w, r, a, b}) {
        TORCH_CHECK(tensor.sizes() == k.sizes(), "Sequence shapes must match");
    }
    TORCH_CHECK(hi.dim() == 4 && hi.size(0) == k.size(0) && hi.size(1) == k.size(1) &&
                hi.size(2) == 64 && hi.size(3) == 64, "Expected state [B,H,64,64]");
    auto out = at_npu::native::OpPreparation::apply_tensor_without_format(k);
    auto ht = at_npu::native::OpPreparation::apply_tensor_without_format(hi);
    EXEC_NPU_CMD_EXT(aclnnRwkvReferenceWkv7, k, v, w, r, a, b, hi, out, ht);
    return {out, ht};
}

Pair wkv7_meta(const at::Tensor& k, const at::Tensor& v, const at::Tensor& w,
               const at::Tensor& r, const at::Tensor& a, const at::Tensor& b,
               const at::Tensor& hi) {
    return {at::empty_like(k), at::empty_like(hi)};
}

TORCH_LIBRARY(rwkv_reference, m) {
    m.def("wkv7(Tensor k, Tensor v, Tensor w, Tensor r, Tensor a, Tensor b, Tensor hi) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(rwkv_reference, PrivateUse1, m) { m.impl("wkv7", &wkv7); }
TORCH_LIBRARY_IMPL(rwkv_reference, Meta, m) { m.impl("wkv7", &wkv7_meta); }
