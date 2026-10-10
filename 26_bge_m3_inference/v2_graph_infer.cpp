// Graph metadata for our static, single-scale V2 test on CANN 9.0.1.
// The upstream V2 kernel/tiler/API are unchanged. This is not a general dynamic-mode adapter.
#include <cstring>
#include "register/op_impl_registry.h"

static ge::graphStatus Shape(gert::InferShapeContext* c) {
    const auto* attrs = c->GetAttrs();
    const auto* mode = attrs ? attrs->GetAttrPointer<char>(0) : nullptr;
    if (!mode || std::strcmp(mode, "static") || !c->GetInputShape(0) ||
        !c->GetOptionalInputShape(5) || c->GetOptionalInputShape(6)) return ge::GRAPH_FAILED;
    for (size_t i : {0, 2, 3}) *c->GetOutputShape(i) = *c->GetInputShape(0);
    for (size_t i : {1, 4, 5}) *c->GetOutputShape(i) = gert::Shape({1});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext* c) {
    c->SetOutputDataType(0, ge::DT_INT8);
    c->SetOutputDataType(1, ge::DT_INT8);
    c->SetOutputDataType(2, c->GetInputDataType(0));
    c->SetOutputDataType(3, c->GetInputDataType(0));
    c->SetOutputDataType(4, ge::DT_FLOAT);
    c->SetOutputDataType(5, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AddLayerNormQuantV2).InferShape(Shape).InferDataType(Dtype);
