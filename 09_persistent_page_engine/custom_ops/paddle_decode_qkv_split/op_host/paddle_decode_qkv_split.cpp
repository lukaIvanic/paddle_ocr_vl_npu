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
static ge::graphStatus SetOutputShape(
    gert::InferShapeContext* context,
    int64_t heads)
{
    gert::Shape* output = context->GetOutputShape(0);
    if (output == nullptr) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(4);
    output->SetDim(0, 1);
    output->SetDim(1, heads);
    output->SetDim(2, 1);
    output->SetDim(3, 128);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferQueryShape(gert::InferShapeContext* context)
{
    return SetOutputShape(context, 16);
}

static ge::graphStatus InferKvShape(gert::InferShapeContext* context)
{
    return SetOutputShape(context, 2);
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeQuerySliceV2 : public OpDef {
public:
    explicit PaddleDecodeQuerySliceV2(const char* name) : OpDef(name)
    {
        this->Input("qkv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferQueryShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

class PaddleDecodeKeySliceV2 : public OpDef {
public:
    explicit PaddleDecodeKeySliceV2(const char* name) : OpDef(name)
    {
        this->Input("qkv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferKvShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

class PaddleDecodeValueSliceV2 : public OpDef {
public:
    explicit PaddleDecodeValueSliceV2(const char* name) : OpDef(name)
    {
        this->Input("qkv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferKvShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeQuerySliceV2);
OP_ADD(PaddleDecodeKeySliceV2);
OP_ADD(PaddleDecodeValueSliceV2);
}
