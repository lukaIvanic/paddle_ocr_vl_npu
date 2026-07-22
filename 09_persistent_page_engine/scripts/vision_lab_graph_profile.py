"""Version-pinned cached vision-graph costs used by routing experiments."""

from __future__ import annotations


# Cache-only device-event profile measured on 2026-07-22. This is deliberately
# version-pinned: routing must not silently reuse 910B2 timings after the model,
# graph source, compiler stack, or hardware changes. ``median_ms`` is the
# intended routing cost; raw throughput is included for audit/readability.
PINNED_910B2_PROFILE = {
    "measured_commit": "bbefb38e9217bfdd614ee72614cd8568bff8c324",
    "device_name": "Ascend910B2",
    "model_config_hash": "6d2211febbe9",
    "torch": "2.10.0+cpu",
    "torch_npu": "2.10.0",
    "vision_source_hash": "a2cd9cd7ae53",
    "attention": "prompt_flash_attention",
    "layout": "bnsd",
    "sparse_mode": 1,
    "execution_mode": "inference",
    "timing_basis": "median of 10 NPU device-event samples after 2 warmups",
    "graphs": {
        (1, 32): {"median_ms": 8.620610, "raw_physical_tokens_per_s": 3712.0},
        (1, 64): {"median_ms": 9.979780, "raw_physical_tokens_per_s": 6413.0},
        (1, 96): {"median_ms": 10.892880, "raw_physical_tokens_per_s": 8813.1},
        (1, 128): {"median_ms": 11.215550, "raw_physical_tokens_per_s": 11412.7},
        (1, 160): {"median_ms": 11.640450, "raw_physical_tokens_per_s": 13745.2},
        (1, 192): {"median_ms": 12.035350, "raw_physical_tokens_per_s": 15953.0},
        (1, 224): {"median_ms": 12.469300, "raw_physical_tokens_per_s": 17964.1},
        (1, 256): {"median_ms": 12.718330, "raw_physical_tokens_per_s": 20128.4},
        (1, 288): {"median_ms": 14.218220, "raw_physical_tokens_per_s": 20255.7},
        (1, 320): {"median_ms": 14.706450, "raw_physical_tokens_per_s": 21759.2},
        (1, 352): {"median_ms": 15.144390, "raw_physical_tokens_per_s": 23242.9},
        (1, 384): {"median_ms": 14.996730, "raw_physical_tokens_per_s": 25605.6},
        (1, 416): {"median_ms": 15.976960, "raw_physical_tokens_per_s": 26037.5},
        (1, 448): {"median_ms": 16.098060, "raw_physical_tokens_per_s": 27829.4},
        (1, 480): {"median_ms": 16.096730, "raw_physical_tokens_per_s": 29819.7},
        (1, 512): {"median_ms": 16.255239, "raw_physical_tokens_per_s": 31497.5},
        (1, 576): {"median_ms": 17.389590, "raw_physical_tokens_per_s": 33123.3},
        (1, 640): {"median_ms": 18.319080, "raw_physical_tokens_per_s": 34936.3},
        (1, 704): {"median_ms": 19.980740, "raw_physical_tokens_per_s": 35233.9},
        (1, 768): {"median_ms": 19.846950, "raw_physical_tokens_per_s": 38696.1},
        (1, 832): {"median_ms": 20.338870, "raw_physical_tokens_per_s": 40906.9},
        (1, 896): {"median_ms": 20.862309, "raw_physical_tokens_per_s": 42948.3},
        (1, 960): {"median_ms": 21.347060, "raw_physical_tokens_per_s": 44971.1},
        (1, 1024): {"median_ms": 22.167390, "raw_physical_tokens_per_s": 46194.0},
        (1, 1152): {"median_ms": 25.000620, "raw_physical_tokens_per_s": 46078.9},
        (1, 1280): {"median_ms": 26.380490, "raw_physical_tokens_per_s": 48520.7},
        (1, 1408): {"median_ms": 27.151820, "raw_physical_tokens_per_s": 51856.6},
        (1, 1536): {"median_ms": 29.161800, "raw_physical_tokens_per_s": 52671.6},
        (1, 1664): {"median_ms": 30.104520, "raw_physical_tokens_per_s": 55274.1},
        (1, 1792): {"median_ms": 28.499870, "raw_physical_tokens_per_s": 62877.5},
        (1, 1920): {"median_ms": 29.164190, "raw_physical_tokens_per_s": 65834.2},
        (1, 2048): {"median_ms": 31.565830, "raw_physical_tokens_per_s": 64880.3},
        (2, 3072): {"median_ms": 80.734600, "raw_physical_tokens_per_s": 76101.2},
        (4, 1024): {"median_ms": 46.402910, "raw_physical_tokens_per_s": 88270.3},
    },
}
