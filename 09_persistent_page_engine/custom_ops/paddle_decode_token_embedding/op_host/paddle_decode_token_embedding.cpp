#include "../op_kernel/paddle_decode_token_embedding_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* weight = context->GetInputShape(0);
    const gert::StorageShape* inputIds = context->GetInputShape(1);
    if (weight == nullptr || inputIds == nullptr ||
        weight->GetStorageShape().GetDimNum() != 2 ||
        weight->GetStorageShape().GetDim(0) != 103424 ||
        weight->GetStorageShape().GetDim(1) != 1024 ||
        inputIds->GetStorageShape().GetDimNum() != 2 ||
        inputIds->GetStorageShape().GetDim(0) != 1 ||
        inputIds->GetStorageShape().GetDim(1) != 1) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<PaddleDecodeTokenEmbeddingTilingData>();
    tiling->hiddenSize = 1024;
    tiling->vocabSize = 103424;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* weight = context->GetInputShape(0);
    const gert::Shape* inputIds = context->GetInputShape(1);
    gert::Shape* output = context->GetOutputShape(0);
    if (weight == nullptr || inputIds == nullptr || output == nullptr ||
        weight->GetDimNum() != 2 || inputIds->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    *output = *inputIds;
    output->AppendDim(weight->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeTokenEmbedding : public OpDef {
public:
    explicit PaddleDecodeTokenEmbedding(const char* name) : OpDef(name)
    {
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("input_ids").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("embedding").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeTokenEmbedding);
}
