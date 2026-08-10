#include "incre_flash_attention_tiling_impl.h"
#include "register/op_def_registry.h"

namespace optiling {
ge::graphStatus TilingPrepareForPaddleDecodeGqaAttentionAiv(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PaddleDecodeGqaAttentionAiv)
    .Tiling(TilingIncreFlashAttention)
    .TilingParse<IncreFlashAttentionCompileInfo>(
        TilingPrepareForPaddleDecodeGqaAttentionAiv);
} // namespace optiling
