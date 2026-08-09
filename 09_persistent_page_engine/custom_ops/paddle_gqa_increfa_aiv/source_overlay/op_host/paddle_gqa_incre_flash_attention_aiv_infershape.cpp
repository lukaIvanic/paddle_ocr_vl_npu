#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShapePaddleGqaIncreFlashAttentionAiv(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePaddleGqaIncreFlashAttentionAiv(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PaddleGqaIncreFlashAttentionAiv)
    .InferShape(InferShapePaddleGqaIncreFlashAttentionAiv)
    .InferDataType(InferDataTypePaddleGqaIncreFlashAttentionAiv);
} // namespace ops

