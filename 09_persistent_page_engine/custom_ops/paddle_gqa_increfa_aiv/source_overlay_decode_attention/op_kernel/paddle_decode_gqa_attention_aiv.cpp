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
    // The enclosing decoder has a 1:1 mixed AIC/AIV launch. This subfunction
    // is vector-only: AIC workers must not enter the inherited attention body
    // or the AIV-only completion barrier below.
    if (g_coreType == AIC) {
        return;
    }

    // The enclosing decoder launches 24 AIV workers, while this attention
    // tiling owns workers 0..15. Only those workers may allocate the inherited
    // attention pipe. Workers 16..23 remain in the subfunction so they can
    // join the completion barrier before the following output projection.
    if (GetBlockIdx() < 16U) {
        TPipe attentionPipe;
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
            tiling,
            &attentionPipe);
        PipeBarrier<PIPE_ALL>();
        attentionPipe.Destroy();
    }
    SyncAll<true>();
}
