"""Validated OmniDocBench v1.6 execution profile."""

from __future__ import annotations


OMNIDOCBENCH_PAGE_COUNT = 1651
OMNIDOCBENCH_DECODE_BATCH_SIZE = 32
OMNIDOCBENCH_CACHE_LENGTH = 4096
# This remains a secondary global safety ceiling.  Each request is admitted
# whenever its prompt fits, then stops at EOS or its own KV-cache boundary.
OMNIDOCBENCH_MAX_NEW_TOKENS = OMNIDOCBENCH_CACHE_LENGTH
