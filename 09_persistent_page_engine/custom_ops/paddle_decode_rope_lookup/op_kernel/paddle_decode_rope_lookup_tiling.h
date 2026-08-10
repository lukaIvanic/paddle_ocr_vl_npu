#ifndef PADDLE_DECODE_ROPE_LOOKUP_TILING_H
#define PADDLE_DECODE_ROPE_LOOKUP_TILING_H

#include <cstdint>

struct PaddleDecodeRopeLookupTilingData {
    uint32_t cacheLength;
    uint32_t headDim;
};

#endif
