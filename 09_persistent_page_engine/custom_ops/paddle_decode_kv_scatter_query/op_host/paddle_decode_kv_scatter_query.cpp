#include "../op_kernel/paddle_decode_kv_scatter_query_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static bool HasShape(const gert::StorageShape* shape, const int64_t* dims, size_t count)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != count) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (shape->GetStorageShape().GetDim(index) != dims[index]) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    constexpr int64_t queryShape[] = {1, 16, 1, 128};
    constexpr int64_t cacheShape[] = {1, 2, 1024, 128};
    constexpr int64_t positionShape[] = {1};
    constexpr int64_t stateShape[] = {1, 2, 1, 128};
    if (!HasShape(context->GetInputShape(0), queryShape, 4) ||
        !HasShape(context->GetInputShape(1), cacheShape, 4) ||
        !HasShape(context->GetInputShape(2), cacheShape, 4) ||
        !HasShape(context->GetInputShape(3), positionShape, 1) ||
        !HasShape(context->GetInputShape(4), stateShape, 4) ||
        !HasShape(context->GetInputShape(5), stateShape, 4)) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<PaddleDecodeKvScatterQueryTilingData>();
    tiling->cacheLength = 1024;
    tiling->queryElements = 2048;
    tiling->stateElements = 256;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    *context->GetOutputShape(1) = *context->GetInputShape(1);
    *context->GetOutputShape(2) = *context->GetInputShape(2);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    context->SetOutputDataType(2, context->GetInputDataType(2));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeKvScatterQueryV2 : public OpDef {
public:
    explicit PaddleDecodeKvScatterQueryV2(const char* name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cache_position").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("ordered_query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("key_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeKvScatterQueryV2);
}
