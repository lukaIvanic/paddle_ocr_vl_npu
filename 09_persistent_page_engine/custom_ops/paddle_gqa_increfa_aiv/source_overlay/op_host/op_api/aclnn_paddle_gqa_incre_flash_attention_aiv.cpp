#include "aclnn_paddle_gqa_incre_flash_attention_aiv.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerPaddleGqaIncreFlashAttentionAivGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShift, const aclTensor *attenMask, const aclIntArray *actualSeqLengths,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2,
    const aclTensor *quantScale2, const aclTensor *quantOffset2,
    const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    int64_t vectorCoreCount,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerPaddleGqaIncreFlashAttentionAiv(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnPaddleGqaIncreFlashAttentionAivGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShift, const aclTensor *attenMask, const aclIntArray *actualSeqLengths,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2,
    const aclTensor *quantScale2, const aclTensor *quantOffset2,
    const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    int64_t vectorCoreCount,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerPaddleGqaIncreFlashAttentionAivGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths,
        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
        antiquantScale, antiquantOffset, blocktable, kvPaddingSize,
        numHeads, scaleValue, inputLayout, numKeyValueHeads, blockSize,
        innerPrecise, vectorCoreCount, attentionOut, workspaceSize, executor);
}

aclnnStatus aclnnPaddleGqaIncreFlashAttentionAiv(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerPaddleGqaIncreFlashAttentionAiv(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
