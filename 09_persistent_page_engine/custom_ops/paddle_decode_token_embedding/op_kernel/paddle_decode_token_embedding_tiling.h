#ifndef PADDLE_DECODE_TOKEN_EMBEDDING_TILING_H
#define PADDLE_DECODE_TOKEN_EMBEDDING_TILING_H

#include <cstdint>

struct PaddleDecodeTokenEmbeddingTilingData {
    uint32_t hiddenSize;
    uint32_t vocabSize;
};

#endif
