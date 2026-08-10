#include "incre_flash_attention_tiling_impl.h"
#include "register/op_def_registry.h"

namespace optiling {
ge::graphStatus TilingPrepareForPaddleDecodeGqaIncreFlashAttentionAiv(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PaddleDecodeGqaIncreFlashAttentionAiv)
    .Tiling(TilingIncreFlashAttention)
    .TilingParse<IncreFlashAttentionCompileInfo>(
        TilingPrepareForPaddleDecodeGqaIncreFlashAttentionAiv)
    .TilingInputsDataDependency(
        {5},
        {gert::TilingPlacement::TILING_ON_HOST,
         gert::TilingPlacement::TILING_ON_AICPU});
} // namespace optiling
