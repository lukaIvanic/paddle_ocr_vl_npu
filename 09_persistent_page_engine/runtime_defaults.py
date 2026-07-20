"""Validated default runtime profile for the Experiment 09 pipeline."""

from __future__ import annotations


DECODE_BACKEND_CHOICES = ("raw_eager", "torchair")
DEFAULT_DECODE_BACKEND = "torchair"
DEFAULT_DECODE_BATCH_SIZE = 4
DEFAULT_CACHE_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_VISION_BACKEND = "torchair"
DEFAULT_TEXT_BACKEND = "torchair"
READY_BUFFER_BATCH_MULTIPLIER = 4

# The official OmniDocBench lane is a throughput/quality run rather than the
# smaller interactive full-page profile above.  Keep its larger static shapes
# named here so the CLI, tests, and documentation cannot drift independently.
OMNIDOCBENCH_PAGE_COUNT = 1651
OMNIDOCBENCH_DECODE_BATCH_SIZE = 16
OMNIDOCBENCH_CACHE_LENGTH = 8192
OMNIDOCBENCH_MAX_NEW_TOKENS = 4096
PADDLEOCR_DEFAULT_MIN_PIXELS = 112896

# Measured dense policy: <=512 by 32, <=1024 by 64, <=2048 by 128.
# Larger vision sequences use the faithful eager overflow path.
OPTIMIZED_VISION_BUCKETS = (
    *range(32, 512 + 1, 32),
    *range(512 + 64, 1024 + 1, 64),
    *range(1024 + 128, 2048 + 1, 128),
)

# Text prompts are projected-image tokens plus a short task prompt.  The
# measured five-page distributions cluster tightly at 32-224 tokens, while a
# small table tail reaches 1,273 tokens.  These buckets retain 95.2% useful
# tokens at default min_pixels and 83.8% at min_pixels/8 on that corpus while
# avoiding the setup cost of compiling every dense vision bucket.
OPTIMIZED_TEXT_BUCKETS = (
    32,
    64,
    96,
    128,
    160,
    176,
    192,
    208,
    224,
    256,
    320,
    384,
    448,
    576,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1312,
)
