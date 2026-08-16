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

} // namespace

#if defined(OP_TILING_LIB)
extern "C" __attribute__((visibility("default")))
int unirec_layout_msda_install_resident_host_infer(const char *trigger)
{
    // Change only the two broken infer pointers. Preserve CANN's stock tiler,
    // workspace, and kernel callbacks.
    auto &functions = gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(
        "MultiScaleDeformableAttnFunction");
    functions.infer_shape = infer_layout_msda_output;
    functions.infer_datatype = infer_layout_msda_dtype;

    static std::atomic<size_t> refresh_count{0};
    const size_t count = refresh_count.fetch_add(1) + 1;
    const char *safe_trigger = trigger == nullptr ? "unknown" : trigger;
    std::fprintf(
        stderr,
        "UNIREC_LAYOUT_MSDA_HOST_OPP_OVERRIDE_INSTALLED "
        "trigger=%s count=%zu\n",
        safe_trigger,
        count);
    const char *marker = std::getenv("UNIREC_LAYOUT_MSDA_HOST_INFER_MARKER");
    if (marker != nullptr && marker[0] != '\0') {
        if (std::FILE *file = std::fopen(marker, "a")) {
            std::fprintf(
                file,
                "override_installed trigger=%s count=%zu\n",
                safe_trigger,
                count);
            std::fclose(file);
        }
    }
    return 0;
}

namespace {
const bool layout_msda_host_opp_installed = []() {
    return unirec_layout_msda_install_resident_host_infer(
               "tiling_initial") == 0;
}();
} // namespace
#else
extern "C" int unirec_layout_msda_install_resident_host_infer(
    const char *trigger);

namespace {
// The proto library is loaded later and repeatedly on 310P. Refresh the
// registry from the NODELETE tiling DSO every time, so CANN's later built-in
// registration cannot win. The published callback still points into the
// resident tiling DSO, never into this short-lived proto DSO.
const bool layout_msda_host_proto_refreshed = []() {
    const int status = unirec_layout_msda_install_resident_host_infer(
        "proto_refresh");
    std::fprintf(
        stderr,
        "UNIREC_LAYOUT_MSDA_HOST_OPP_REFRESH_FROM_PROTO status=%d\n",
        status);
    return status == 0;
}();
} // namespace
#endif
