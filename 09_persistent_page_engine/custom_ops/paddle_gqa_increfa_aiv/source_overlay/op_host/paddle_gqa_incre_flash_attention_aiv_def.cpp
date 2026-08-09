/** Independent B1 FP16/BNSD/16Q:2KV GQA AIV operator definition. */

#include "register/op_def_registry.h"

namespace ops {
class PaddleGqaIncreFlashAttentionAiv : public OpDef {
public:
    explicit PaddleGqaIncreFlashAttentionAiv(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(DYNAMIC).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(DYNAMIC).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("pse_shift").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("atten_mask").ParamType(OPTIONAL).DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dequant_scale1").ParamType(OPTIONAL).DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quant_scale1").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dequant_scale2").ParamType(OPTIONAL).DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quant_scale2").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quant_offset2").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("antiquant_scale").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("antiquant_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table").ParamType(OPTIONAL).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kv_padding_size").ParamType(OPTIONAL).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_heads").AttrType(REQUIRED).Int(16);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.08838834764831845);
        this->Attr("input_layout").AttrType(OPTIONAL).String("BNSD");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(2);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
        this->Attr("vector_core_count").AttrType(OPTIONAL).Int(48);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "paddle_gqa_incre_flash_attention_aiv");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(PaddleGqaIncreFlashAttentionAiv);
} // namespace ops
