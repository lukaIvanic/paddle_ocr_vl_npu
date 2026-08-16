#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "register/op_impl_registry.h"
#include "op_impl_registry_base_compat.h"

#ifndef UNIREC_MSDA_LIBRARY_ROLE
#define UNIREC_MSDA_LIBRARY_ROLE "unknown"
#endif

namespace {

ge::graphStatus infer_layout_msda_output(gert::InferShapeContext *context)
{
    const gert::Shape *value_shape = context->GetInputShape(0);
    const gert::Shape *location_shape = context->GetInputShape(3);
    gert::Shape *output_shape = context->GetOutputShape(0);
    if (value_shape == nullptr || location_shape == nullptr ||
        output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    output_shape->SetDimNum(3);
    output_shape->SetDim(0, value_shape->GetDim(0));
    if (location_shape->GetDim(1) < 32) {
        // 310P internal locations are [L,B,H,Q,P,2], and the internal
        // kernel output is [B,H*D,Q]. The stock CANN host function uses
        // location dimensions 5 and 1 here, producing [B,2,B*D].
        output_shape->SetDim(
            1,
            location_shape->GetDim(2) * value_shape->GetDim(3));
        output_shape->SetDim(2, location_shape->GetDim(3));
    } else {
        // Public/logical layout: [B,Q,H,L,P,2] -> [B,Q,H*D].
        output_shape->SetDim(1, location_shape->GetDim(1));
        output_shape->SetDim(
            2,
            location_shape->GetDim(2) * value_shape->GetDim(3));
    }

    static std::atomic<bool> reported{false};
    if (!reported.exchange(true)) {
        std::fprintf(
            stderr,
            "UNIREC_LAYOUT_MSDA_HOST_OPP_ACTIVE "
            "location_dim1=%ld output=[%ld,%ld,%ld]\n",
            static_cast<long>(location_shape->GetDim(1)),
            static_cast<long>(output_shape->GetDim(0)),
            static_cast<long>(output_shape->GetDim(1)),
            static_cast<long>(output_shape->GetDim(2)));
        const char *marker = std::getenv("UNIREC_LAYOUT_MSDA_HOST_INFER_MARKER");
        if (marker != nullptr && marker[0] != '\0') {
            if (std::FILE *file = std::fopen(marker, "a")) {
                std::fprintf(
                    file,
                    "location_dim1=%ld output=[%ld,%ld,%ld]\n",
                    static_cast<long>(location_shape->GetDim(1)),
                    static_cast<long>(output_shape->GetDim(0)),
                    static_cast<long>(output_shape->GetDim(1)),
                    static_cast<long>(output_shape->GetDim(2)));
                std::fclose(file);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus infer_layout_msda_dtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

#if defined(OP_TILING_LIB)
const bool layout_msda_host_opp_installed = []() {
    // This library is loaded by GE's compiler process through the user OPP.
    // A duplicate OpImplRegisterV2 does not replace CANN's built-in callback.
    // Mutate only the two broken infer pointers in that process and preserve
    // the installed tiler, workspace, and kernel callbacks.
    auto &functions = gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(
        "MultiScaleDeformableAttnFunction");
    functions.infer_shape = infer_layout_msda_output;
    functions.infer_datatype = infer_layout_msda_dtype;

    std::fprintf(
        stderr,
        "UNIREC_LAYOUT_MSDA_HOST_OPP_OVERRIDE_INSTALLED role=%s\n",
        UNIREC_MSDA_LIBRARY_ROLE);
    const char *marker = std::getenv("UNIREC_LAYOUT_MSDA_HOST_INFER_MARKER");
    if (marker != nullptr && marker[0] != '\0') {
        if (std::FILE *file = std::fopen(marker, "a")) {
            std::fprintf(
                file,
                "override_installed role=%s\n",
                UNIREC_MSDA_LIBRARY_ROLE);
            std::fclose(file);
        }
    }
    return true;
}();
#else
// GE loads and unloads the proto library repeatedly while constructing a
// graph. Never publish a callback pointer into that short-lived DSO. The
// op-tiling library above remains resident for the compiler lifetime.
const bool layout_msda_host_proto_observed = []() {
    std::fprintf(
        stderr,
        "UNIREC_LAYOUT_MSDA_HOST_OPP_LOADED_NO_OVERRIDE role=%s\n",
        UNIREC_MSDA_LIBRARY_ROLE);
    return true;
}();
#endif

} // namespace
