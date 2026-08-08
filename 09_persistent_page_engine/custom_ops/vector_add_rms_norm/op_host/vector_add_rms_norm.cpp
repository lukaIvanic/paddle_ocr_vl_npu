#include "../op_kernel/vector_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* input = context->GetInputShape(0);
    uint32_t size = 1;
    for (size_t index = 0; index < input->GetStorageShape().GetDimNum(); ++index) {
        size *= static_cast<uint32_t>(input->GetStorageShape().GetDim(index));
    }
    if (size != 1024) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<VectorAddRmsNormTilingData>();
    tiling->size = size;
    context->SetBlockDim(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    *context->GetOutputShape(0) = *input;
    *context->GetOutputShape(2) = *input;
    gert::Shape* rstd = context->GetOutputShape(1);
    *rstd = *input;
    rstd->SetDim(rstd->GetDimNum() - 1, 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, input_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class VectorAddRmsNorm : public OpDef {
public:
    explicit VectorAddRmsNorm(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rstd").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(VectorAddRmsNorm);
}
