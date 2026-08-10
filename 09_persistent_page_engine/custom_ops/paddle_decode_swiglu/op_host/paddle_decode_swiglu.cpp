#include "../op_kernel/paddle_decode_swiglu_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    constexpr int64_t expected[] = {1, 1, 3072};
    for (size_t inputIndex = 0; inputIndex < 2; ++inputIndex) {
        const auto* shape = context->GetInputShape(inputIndex);
        if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 3) {
            return ge::GRAPH_FAILED;
        }
        for (size_t dim = 0; dim < 3; ++dim) {
            if (shape->GetStorageShape().GetDim(dim) != expected[dim]) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    auto* tiling = context->GetTilingData<PaddleDecodeSwiGluTilingData>();
    tiling->elements = 3072;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeSwiGluV1 : public OpDef {
public:
    explicit PaddleDecodeSwiGluV1(const char* name) : OpDef(name)
    {
        this->Input("gate").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("up").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeSwiGluV1);
}
