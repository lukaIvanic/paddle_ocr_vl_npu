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
        !HasShape(context->GetDynamicInputShape(1, 0), cacheShape, 4) ||
        !HasShape(context->GetDynamicInputShape(2, 0), cacheShape, 4) ||
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

static ge::graphStatus TilingFuncMixed24(gert::TilingContext* context)
{
    const ge::graphStatus status = TilingFunc(context);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    context->SetBlockDim(24);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr ||
        context->GetInputShape(6) == nullptr || context->GetInputShape(7) == nullptr ||
        context->GetOutputShape(2) == nullptr || context->GetOutputShape(3) == nullptr) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    auto* attentionMask = context->GetOutputShape(1);
    attentionMask->SetDimNum(4);
    attentionMask->SetDim(0, 1);
    attentionMask->SetDim(1, 1);
    attentionMask->SetDim(2, 1);
    attentionMask->SetDim(3, 1024);
    *context->GetOutputShape(2) = *context->GetInputShape(6);
    *context->GetOutputShape(3) = *context->GetInputShape(7);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_BOOL);
    context->SetOutputDataType(2, context->GetInputDataType(6));
    context->SetOutputDataType(3, context->GetInputDataType(7));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeKvScatterQueryV4 : public OpDef {
public:
    explicit PaddleDecodeKvScatterQueryV4(const char* name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(DYNAMIC).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(DYNAMIC).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cache_position").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_cache_ref").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value_cache_ref").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("ordered_query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_mask").ParamType(REQUIRED).DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("key_cache_ref").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value_cache_ref").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

class PaddleDecodeKvScatterQueryMixed24 : public PaddleDecodeKvScatterQueryV4 {
public:
    explicit PaddleDecodeKvScatterQueryMixed24(const char* name)
        : PaddleDecodeKvScatterQueryV4(name)
    {
        this->AICore().SetTiling(optiling::TilingFuncMixed24);
    }
};

OP_ADD(PaddleDecodeKvScatterQueryV4);
OP_ADD(PaddleDecodeKvScatterQueryMixed24);
}
