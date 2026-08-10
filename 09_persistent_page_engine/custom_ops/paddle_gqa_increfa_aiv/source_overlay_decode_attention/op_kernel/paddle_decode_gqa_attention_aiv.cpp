#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#define PADDLE_DECODE_GQA_PLAIN_KV 1
#include "incre_flash_attention_arch32.h"

extern "C" __global__ __aicore__ void paddle_decode_gqa_attention_aiv(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    incre_flash_attention_FIAS_arch32(
        query,
        key,
        value,
        nullptr,
        attenMask,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        attentionOut,
        nullptr,
        workspace,
        tiling);
}
