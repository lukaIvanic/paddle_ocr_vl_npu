#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "register/op_impl_registry.h"

#ifndef UNIREC_MSDA_LIBRARY_ROLE
#define UNIREC_MSDA_LIBRARY_ROLE "unknown"
#endif

namespace {

__attribute__((constructor)) void report_layout_msda_host_library_load()
{
    std::fprintf(
        stderr,
        "UNIREC_LAYOUT_MSDA_HOST_OPP_LOADED role=%s\n",
        UNIREC_MSDA_LIBRARY_ROLE);
    const char *marker = std::getenv("UNIREC_LAYOUT_MSDA_HOST_INFER_MARKER");
    if (marker != nullptr && marker[0] != '\0') {
        if (std::FILE *file = std::fopen(marker, "a")) {
            std::fprintf(file, "library_loaded role=%s\n", UNIREC_MSDA_LIBRARY_ROLE);
            std::fclose(file);
        }
    }
}

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

const bool layout_msda_host_opp_registered = []() {
    static gert::OpImplRegisterV2 registration(
        "MultiScaleDeformableAttnFunction");
    registration.InferShape(infer_layout_msda_output)
        .InferDataType(infer_layout_msda_dtype);
    return true;
}();

} // namespace
