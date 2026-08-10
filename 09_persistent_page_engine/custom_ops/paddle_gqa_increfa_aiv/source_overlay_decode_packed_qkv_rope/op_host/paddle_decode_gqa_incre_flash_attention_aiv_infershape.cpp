#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShapePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape({1, 16, 1, 128});
    constexpr size_t kInputForOutput[] = {1, 2, 3, 0};
    for (size_t output = 1; output < 5; ++output) {
        const size_t input = kInputForOutput[output - 1];
        if (context->GetInputShape(input) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) = *context->GetInputShape(input);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    constexpr size_t kInputForOutput[] = {1, 2, 3, 0};
    for (size_t output = 1; output < 5; ++output) {
        context->SetOutputDataType(
            output, context->GetInputDataType(kInputForOutput[output - 1]));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PaddleDecodeGqaIncreFlashAttentionAiv)
    .InferShape(InferShapePaddleDecodeGqaIncreFlashAttentionAiv)
    .InferDataType(InferDataTypePaddleDecodeGqaIncreFlashAttentionAiv);
} // namespace ops
