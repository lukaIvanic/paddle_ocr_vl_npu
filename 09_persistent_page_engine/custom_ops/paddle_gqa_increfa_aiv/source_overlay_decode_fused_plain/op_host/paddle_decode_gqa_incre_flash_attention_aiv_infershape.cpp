#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShapePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        const size_t input_index = index == 0 ? 0 : index;
        if (context->GetInputShape(input_index) == nullptr ||
            context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(index) = *context->GetInputShape(input_index);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        const size_t input_index = index == 0 ? 0 : index;
        context->SetOutputDataType(index, context->GetInputDataType(input_index));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PaddleDecodeGqaIncreFlashAttentionAiv)
    .InferShape(InferShapePaddleDecodeGqaIncreFlashAttentionAiv)
    .InferDataType(InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv);
} // namespace ops
