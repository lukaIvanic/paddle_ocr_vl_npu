#include "../op_kernel/paddle_decode_position_add_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* cachePosition = context->GetInputShape(0);
    const gert::StorageShape* ropeDelta = context->GetInputShape(1);
    if (cachePosition == nullptr || ropeDelta == nullptr ||
        cachePosition->GetStorageShape().GetDimNum() != 2 ||
        cachePosition->GetStorageShape().GetDim(0) != 1 ||
        cachePosition->GetStorageShape().GetDim(1) != 1 ||
        ropeDelta->GetStorageShape().GetDimNum() != 2 ||
        ropeDelta->GetStorageShape().GetDim(0) != 1 ||
        ropeDelta->GetStorageShape().GetDim(1) != 1) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<PaddleDecodePositionAddTilingData>();
    tiling->elements = 1;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* cachePosition = context->GetInputShape(0);
    const gert::Shape* ropeDelta = context->GetInputShape(1);
    gert::Shape* decodePosition = context->GetOutputShape(0);
    if (cachePosition == nullptr || ropeDelta == nullptr ||
        decodePosition == nullptr) {
        return GRAPH_FAILED;
    }
    *decodePosition = *cachePosition;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodePositionAddV1 : public OpDef {
public:
    explicit PaddleDecodePositionAddV1(const char* name) : OpDef(name)
    {
        this->Input("cache_position").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rope_delta").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("decode_position").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodePositionAddV1);
}
