from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from table_row_split_lab import adaptive_max_snapped_proposal  # noqa: E402


def ruled_table(rows: int, *, width: int = 640, row_height: int = 28) -> Image.Image:
    image = Image.new("RGB", (width, rows * row_height + 1), "white")
    draw = ImageDraw.Draw(image)
    for row in range(rows + 1):
        y = min(image.height - 1, row * row_height)
        draw.line((0, y, width - 1, y), fill="black", width=2)
    for row in range(rows):
        y = row * row_height + 9
        draw.rectangle((20, y, 95, y + 7), fill="black")
        draw.rectangle((220, y, 300, y + 7), fill="black")
    return image


def test_adaptive_uses_non_power_of_two_natural_row_count() -> None:
    proposal = adaptive_max_snapped_proposal(ruled_table(13), max_rows=32)

    assert len(proposal.boundaries) - 1 == 13
    assert proposal.name == "adaptive_max_32_snapped"
    assert proposal.diagnostics["cap_mode"] == "all_selected_natural_boundaries"
    assert proposal.diagnostics["snap_ink_crossings_after"] == 0


def test_adaptive_caps_at_32_using_existing_safe_boundaries() -> None:
    proposal = adaptive_max_snapped_proposal(ruled_table(45), max_rows=32)

    assert len(proposal.boundaries) - 1 == 32
    assert proposal.diagnostics["detected_natural_rows"] == 45
    assert proposal.diagnostics["cap_mode"] == (
        "grouped_at_existing_natural_boundaries"
    )


def test_adaptive_keeps_at_least_two_snapped_bands() -> None:
    image = Image.new("RGB", (640, 120), "white")

    proposal = adaptive_max_snapped_proposal(image, max_rows=32, minimum_rows=2)

    assert len(proposal.boundaries) - 1 == 2
    assert proposal.boundaries[0] == 0
    assert proposal.boundaries[-1] == image.height
