
#include "rwkv_reference_wkv7_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    RwkvReferenceWkv7TilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNum(); //获取当前硬件平台的核数

    auto x_shape = context->GetInputShape(0)->GetStorageShape();//获取运行时Shape

    if (x_shape.GetDimNum() != 4 || x_shape.GetDim(3) != 64) return ge::GRAPH_FAILED;
    uint32_t B = x_shape.GetDim(0);
    uint32_t T = x_shape.GetDim(2);
    uint32_t HEAD_NUMS = x_shape.GetDim(1);
    uint32_t HEAD_SIZE = x_shape.GetDim(3);


    if (!B || !T || !HEAD_NUMS) return ge::GRAPH_FAILED;
    uint32_t tileLength = 48;
    uint32_t tileNum = T / tileLength; // 余数还需要考虑。
    uint32_t tileNumRemainLength = T % tileLength; // 余数

    // B * H 能被核数整除的情况，不能整除时将remainer按照每个core一个head进行处理
    uint32_t totalHeads = B * HEAD_NUMS;

    bool hasRemainer = false;
    if (tileNumRemainLength > 0)
        {
            hasRemainer = true;
        } else {
            hasRemainer = false;
        }

    tiling.set_tileNum(tileNum);
    tiling.set_tileNumRemainLength(tileNumRemainLength);
    tiling.set_totalHeads(totalHeads);
    tiling.set_T(T);

    tiling.set_tileLength(tileLength);
    tiling.set_HEAD_SIZE(HEAD_SIZE);
    tiling.set_HEAD_NUMS(HEAD_NUMS);
    tiling.set_hasRemainer(hasRemainer);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    *context->GetOutputShape(1) = *context->GetInputShape(6);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class RwkvReferenceWkv7 : public OpDef {
public:
    explicit RwkvReferenceWkv7(const char* name) : OpDef(name)
    {
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hi")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("ho")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(RwkvReferenceWkv7);
}
