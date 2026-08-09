#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "incre_flash_attention_arch32.h"

extern "C" __global__ __aicore__ void paddle_gqa_incre_flash_attention_aiv(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
    __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale,
    __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    incre_flash_attention_FIAS_arch32(
        query, key, value, pseShift, attenMask, nullptr, actualSeqLengths,
        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
        antiquantScale, antiquantOffset, blocktable, nullptr, kvPaddingSize,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, attentionOut, nullptr, workspace, tiling);
}
