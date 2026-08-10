#include "kernel_operator.h"
#include "paddle_decode_token_embedding_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kVocabSize = 103424;

class PaddleDecodeTokenEmbeddingKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR weight,
        GM_ADDR inputIds,
        GM_ADDR embedding,
        TPipe* pipe)
    {
        weightGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(weight),
            kVocabSize * kHiddenSize);
        inputIdsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(inputIds), 1);
        embeddingGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(embedding), kHiddenSize);
        pipe->InitBuffer(rowQueue, 1, kHiddenSize * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const int64_t tokenId = inputIdsGm.GetValue(0);
        if (tokenId < 0 || tokenId >= static_cast<int64_t>(kVocabSize)) {
            return;
        }
        LocalTensor<half> row = rowQueue.AllocTensor<half>();
        DataCopy(row, weightGm[tokenId * kHiddenSize], kHiddenSize);
        rowQueue.EnQue(row);
        row = rowQueue.DeQue<half>();
        DataCopy(embeddingGm, row, kHiddenSize);
        rowQueue.FreeTensor(row);
    }

private:
    GlobalTensor<half> weightGm;
    GlobalTensor<int64_t> inputIdsGm;
    GlobalTensor<half> embeddingGm;
    TQue<QuePosition::VECIN, 1> rowQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_token_embedding(
    GM_ADDR weight,
    GM_ADDR inputIds,
    GM_ADDR embedding,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeTokenEmbeddingTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.hiddenSize != kHiddenSize ||
        tilingData.vocabSize != kVocabSize) {
        return;
    }
    TPipe pipe;
    PaddleDecodeTokenEmbeddingKernel kernel;
    kernel.Init(weight, inputIds, embedding, &pipe);
    kernel.Process();
    // SuperKernel compilation removes the final PIPE_ALL barrier normally
    // supplied by TPipe::Destroy().  Finish the UB-to-GM embedding write before
    // the next fused subkernel reads it, then release the pipe resources.
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}
