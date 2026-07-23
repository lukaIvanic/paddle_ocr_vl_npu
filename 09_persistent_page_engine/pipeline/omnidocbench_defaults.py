"""Validated OmniDocBench v1.6 execution profile."""

from __future__ import annotations


OMNIDOCBENCH_PAGE_COUNT = 1651
OMNIDOCBENCH_DECODE_BATCH_SIZE = 32
OMNIDOCBENCH_CACHE_LENGTH = 4096
# The largest prompt in the retained 256-page trace is 1,289 tokens, so 2,808
# generated tokens exactly fit the static cache: 1289 + 2808 - 1 == 4096.
OMNIDOCBENCH_MAX_NEW_TOKENS = 2808
