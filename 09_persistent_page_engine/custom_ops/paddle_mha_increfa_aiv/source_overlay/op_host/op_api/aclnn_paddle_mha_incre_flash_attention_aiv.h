/**
 * Public aclnn surface for the separately named Paddle MHA AIV operator.
 */

#ifndef ACLNN_PADDLE_MHA_INCRE_FLASH_ATTENTION_AIV_H_
#define ACLNN_PADDLE_MHA_INCRE_FLASH_ATTENTION_AIV_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnPaddleMhaIncreFlashAttentionAivGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *blocktable,
    const aclTensor *kvPaddingSize,
    int64_t numHeads,
    double scaleValue,
    char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t blockSize,
    int64_t innerPrecise,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnPaddleMhaIncreFlashAttentionAiv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
