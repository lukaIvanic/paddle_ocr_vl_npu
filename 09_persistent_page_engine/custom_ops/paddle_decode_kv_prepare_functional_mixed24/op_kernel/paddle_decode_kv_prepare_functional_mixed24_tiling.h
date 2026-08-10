#ifndef PADDLE_DECODE_KV_PREPARE_FUNCTIONAL_MIXED24_TILING_H
#define PADDLE_DECODE_KV_PREPARE_FUNCTIONAL_MIXED24_TILING_H

#include <cstdint>

struct PaddleDecodeKvPrepareFunctionalMixed24TilingData {
    uint32_t cacheElements;
    uint32_t copyElementsPerCore;
};

#endif
