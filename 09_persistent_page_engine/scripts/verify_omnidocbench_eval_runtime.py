#!/usr/bin/env python3
"""Verify that the installed evaluator can render one representative CDM formula."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator_root = args.evaluator_root.resolve()
    sys.path.insert(0, str(evaluator_root))

    from src.metrics.cdm.cdm import gen_color_list
    from src.metrics.cdm.modules.latex2bbox_color import latex2bbox_color
    from src.metrics.cdm.modules.texlive_env import describe_tex_runtime

    latex = r"\mathrm{传动侧} + \text{效率} = \frac{1}{2}"
    with tempfile.TemporaryDirectory(prefix="omnidocbench_cdm_smoke_") as tmp:
        root = Path(tmp)
        output = root / "output"
        intermediate = root / "temp"
        (output / "bbox").mkdir(parents=True)
        (output / "vis").mkdir(parents=True)
        intermediate.mkdir()

        latex2bbox_color(
            (latex, "smoke_case", str(output), str(intermediate), gen_color_list(num=5800))
        )

        bbox = output / "bbox" / "smoke_case.jsonl"
        base_png = output / "vis" / "smoke_case_base.png"
        vis_png = output / "vis" / "smoke_case.png"
        missing = [str(path) for path in (bbox, base_png, vis_png) if not path.is_file()]
        if missing:
            raise SystemExit(f"CDM smoke did not produce: {missing}")

        lines = [line for line in bbox.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            raise SystemExit("CDM smoke produced an empty bbox file")

        print(
            json.dumps(
                {
                    "status": "pass",
                    "bbox_count": len(lines),
                    "tex_runtime": describe_tex_runtime(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
