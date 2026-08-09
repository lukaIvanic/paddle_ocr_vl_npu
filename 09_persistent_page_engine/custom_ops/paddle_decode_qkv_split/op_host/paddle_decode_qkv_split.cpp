#include "../op_kernel/paddle_decode_qkv_split_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* qkv = context->GetInputShape(0);
    if (qkv == nullptr || qkv->GetStorageShape().GetDimNum() != 3 ||
        qkv->GetStorageShape().GetDim(0) != 1 ||
        qkv->GetStorageShape().GetDim(1) != 1 ||
        qkv->GetStorageShape().GetDim(2) != 2560) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<PaddleDecodeQkvSplitTilingData>();
    tiling->queryElements = 2048;
    tiling->keyValueElements = 256;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* query = context->GetOutputShape(0);
    gert::Shape* key = context->GetOutputShape(1);
    gert::Shape* value = context->GetOutputShape(2);
    if (query == nullptr || key == nullptr || value == nullptr) {
        return GRAPH_FAILED;
    }
    query->SetDimNum(4);
    query->SetDim(0, 1);
    query->SetDim(1, 16);
    query->SetDim(2, 1);
    query->SetDim(3, 128);
    key->SetDimNum(4);
    key->SetDim(0, 1);
    key->SetDim(1, 2);
    key->SetDim(2, 1);
    key->SetDim(3, 128);
    value->SetDimNum(4);
    value->SetDim(0, 1);
    value->SetDim(1, 2);
    value->SetDim(2, 1);
    value->SetDim(3, 128);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
    context->SetOutputDataType(1, inputType);
    context->SetOutputDataType(2, inputType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeQkvSplitV2 : public OpDef {
public:
    explicit PaddleDecodeQkvSplitV2(const char* name) : OpDef(name)
    {
        this->Input("qkv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeQkvSplitV2);
}
