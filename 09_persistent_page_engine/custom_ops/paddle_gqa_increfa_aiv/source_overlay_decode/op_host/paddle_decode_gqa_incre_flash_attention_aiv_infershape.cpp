#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShapePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr || context->GetOutputShape(3) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    *context->GetOutputShape(1) = *context->GetInputShape(1);
    *context->GetOutputShape(2) = *context->GetInputShape(2);
    // Optional inputs that are not supplied are omitted from the runtime
    // InferShapeContext.  In the Paddle decoder specialization the mask is
    // therefore compact input 3, immediately after query/key/value.
    *context->GetOutputShape(3) = *context->GetInputShape(3);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    context->SetOutputDataType(2, context->GetInputDataType(2));
    context->SetOutputDataType(3, context->GetInputDataType(3));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PaddleDecodeGqaIncreFlashAttentionAiv)
    .InferShape(InferShapePaddleDecodeGqaIncreFlashAttentionAiv)
    .InferDataType(InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv);
} // namespace ops
