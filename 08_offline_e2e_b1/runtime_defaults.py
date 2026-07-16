"""Validated default runtime profile for the Experiment 08 pipeline."""

from __future__ import annotations


DECODE_BACKEND_CHOICES = ("raw_eager", "torchair")
DEFAULT_DECODE_BACKEND = "torchair"
DEFAULT_DECODE_BATCH_SIZE = 4
DEFAULT_VISION_BACKEND = "torchair"
READY_BUFFER_BATCH_MULTIPLIER = 4

# Measured dense policy: <=512 by 32, <=1024 by 64, <=2048 by 128.
# Larger vision sequences use the faithful eager overflow path.
OPTIMIZED_VISION_BUCKETS = (
    *range(32, 512 + 1, 32),
    *range(512 + 64, 1024 + 1, 64),
    *range(1024 + 128, 2048 + 1, 128),
)
