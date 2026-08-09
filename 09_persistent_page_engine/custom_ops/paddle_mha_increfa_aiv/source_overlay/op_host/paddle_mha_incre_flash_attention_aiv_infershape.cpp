/**
 * Shape inference for the explicit Paddle MHA AIV attention operator.
 */

#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShapePaddleMhaIncreFlashAttentionAiv(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePaddleMhaIncreFlashAttentionAiv(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PaddleMhaIncreFlashAttentionAiv)
    .InferShape(InferShapePaddleMhaIncreFlashAttentionAiv)
    .InferDataType(InferDataTypePaddleMhaIncreFlashAttentionAiv);
} // namespace ops
