#!/usr/bin/env python3
"""Run Experiment 09 with the former packed final-token gather.

This is an A/B reference wrapper for the production OmniDocBench runner. It
changes only ``PackedTextPrefillRuntime.run_prepared`` after the compiled text
prefill has returned: the former implementation selected all
``last_token_indices`` with ``torch.index_select`` and let the unchanged caller
discard padded member rows. Model graphs, caches, scheduling, admission, and
decode slot swapping remain the production paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE.parent
sys.path.insert(0, str(SCRIPTS_ROOT))

import run_omnidocbench
from paddleocr_vl.model.text_packed_prefill import (
    PackedTextPrefillRuntime,
    PreparedPackedTextPrefill,
)


def _run_prepared_with_index_select(
    self: PackedTextPrefillRuntime,
    prepared: PreparedPackedTextPrefill,
) -> torch.Tensor:
    scratch_cache = self.scratch_caches[prepared.physical_seq_len]
    hidden_states = self.compiled[prepared.physical_seq_len](
        prepared.inputs_embeds,
        prepared.position_ids,
        prepared.segment_ids,
        prepared.local_positions,
        *scratch_cache.flat_tensors(),
    )
    return torch.index_select(
        hidden_states,
        1,
        prepared.last_token_indices,
    )


def main() -> None:
    PackedTextPrefillRuntime.run_prepared = _run_prepared_with_index_select
    print(
        "packed_final_token_selection=index_select_reference",
        flush=True,
    )
    run_omnidocbench.main()


if __name__ == "__main__":
    main()
