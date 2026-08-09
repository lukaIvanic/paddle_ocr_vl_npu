from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from table_row_split_lab import adaptive_cutfit_snapped_proposal  # noqa: E402


def ruled_table(rows: int, *, width: int = 640, row_height: int = 28) -> Image.Image:
    image = Image.new("RGB", (width, rows * row_height + 1), "white")
    draw = ImageDraw.Draw(image)
    for row in range(rows + 1):
        y = min(image.height - 1, row * row_height)
        draw.line((0, y, width - 1, y), fill="black", width=2)
    return image


def test_cutfit_search_checks_every_u_and_can_select_non_power_of_two() -> None:
    proposal = adaptive_cutfit_snapped_proposal(ruled_table(19), max_rows=32)

    assert len(proposal.diagnostics["u_trials"]) == 32
    assert [trial["rows"] for trial in proposal.diagnostics["u_trials"]] == list(
        range(1, 33)
    )
    assert len(proposal.boundaries) - 1 == 6
    assert proposal.diagnostics["selected_crossings"] == 0


def test_cutfit_reserves_more_than_one_body_band_for_header_context() -> None:
    proposal = adaptive_cutfit_snapped_proposal(ruled_table(25), max_rows=32)
    band_heights = [
        right - left
        for left, right in zip(proposal.boundaries, proposal.boundaries[1:])
    ]

    assert len(proposal.boundaries) - 1 == 8
    assert band_heights[0] > min(band_heights[1:])
    assert proposal.boundaries[1] >= proposal.diagnostics["header_guard_y"]
    assert proposal.diagnostics["selected_first_band_weight"] > 1.0


def test_dense_cutfit_keeps_header_context_but_uses_more_bands() -> None:
    image = ruled_table(25)
    regular = adaptive_cutfit_snapped_proposal(image, max_rows=32)
    dense = adaptive_cutfit_snapped_proposal(
        image,
        max_rows=32,
        body_context_rows=2,
    )

    assert len(dense.boundaries) > len(regular.boundaries)
    assert len(dense.boundaries) - 1 == 12
    assert dense.boundaries[1] >= dense.diagnostics["header_guard_y"]
    assert dense.diagnostics["selected_crossings"] == 0


def test_cutfit_uses_whole_table_when_no_row_evidence_exists() -> None:
    image = Image.new("RGB", (640, 120), "white")

    proposal = adaptive_cutfit_snapped_proposal(image, max_rows=32)

    assert proposal.boundaries == (0, image.height)
    assert proposal.diagnostics["context_cap"] == 1
