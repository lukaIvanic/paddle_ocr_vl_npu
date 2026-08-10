#include "../op_kernel/paddle_decode_rope_lookup_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* factorLut = context->GetInputShape(0);
    const gert::StorageShape* cachePosition = context->GetInputShape(1);
    const gert::StorageShape* ropeDelta = context->GetInputShape(2);
    if (factorLut == nullptr || cachePosition == nullptr || ropeDelta == nullptr ||
        factorLut->GetStorageShape().GetDimNum() != 3 ||
        factorLut->GetStorageShape().GetDim(0) != 2 ||
        factorLut->GetStorageShape().GetDim(1) != 1024 ||
        factorLut->GetStorageShape().GetDim(2) != 128 ||
        cachePosition->GetStorageShape().GetDimNum() != 2 ||
        cachePosition->GetStorageShape().GetDim(0) != 1 ||
        cachePosition->GetStorageShape().GetDim(1) != 1 ||
        ropeDelta->GetStorageShape().GetDimNum() != 2 ||
        ropeDelta->GetStorageShape().GetDim(0) != 1 ||
        ropeDelta->GetStorageShape().GetDim(1) != 1) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<PaddleDecodeRopeLookupTilingData>();
    tiling->cacheLength = 1024;
    tiling->headDim = 128;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* cos = context->GetOutputShape(0);
    gert::Shape* sin = context->GetOutputShape(1);
    if (cos == nullptr || sin == nullptr) {
        return GRAPH_FAILED;
    }
    cos->SetDimNum(4);
    sin->SetDimNum(4);
    for (int outputIndex = 0; outputIndex < 2; ++outputIndex) {
        gert::Shape* output = context->GetOutputShape(outputIndex);
        output->SetDim(0, 1);
        output->SetDim(1, 1);
        output->SetDim(2, 1);
        output->SetDim(3, 128);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
    context->SetOutputDataType(1, inputType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class PaddleDecodeRopeLookupV1 : public OpDef {
public:
    explicit PaddleDecodeRopeLookupV1(const char* name) : OpDef(name)
    {
        this->Input("factor_lut").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cache_position").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rope_delta").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cos").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sin").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PaddleDecodeRopeLookupV1);
}
