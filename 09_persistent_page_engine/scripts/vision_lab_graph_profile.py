"""Compatibility import for the serving-owned pinned vision profile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paddleocr_vl.serving.vision_router import PINNED_910B2_PROFILE

__all__ = ["PINNED_910B2_PROFILE"]
