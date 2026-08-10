/** SuperKernel-safe B1/KV1024 16Q:2KV attention-only AIV definition. */

#include "register/op_def_registry.h"

namespace ops {
class PaddleDecodeGqaAttentionAiv : public OpDef {
public:
    explicit PaddleDecodeGqaAttentionAiv(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        // This decoder-only entry patches the pinned all-vector kernel to bind
        // the fixed B1 K/V tensors directly. Dynamic list descriptors are not
        // stable when this function is extracted into a SuperKernel.
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("pse_shift").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("atten_mask").ParamType(REQUIRED).DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();

        this->Output("attention_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_heads").AttrType(REQUIRED).Int(16);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0883883461F);
        this->Attr("input_layout").AttrType(OPTIONAL).String("BNSD");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(2);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
        this->Attr("vector_core_count").AttrType(OPTIONAL).Int(16);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "paddle_decode_gqa_attention_aiv")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(PaddleDecodeGqaAttentionAiv);
} // namespace ops
