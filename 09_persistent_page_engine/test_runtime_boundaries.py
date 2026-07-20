"""Keep Experiment 09 production modules independent of runners and probes."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRODUCTION_MODULES = (
    "compile_utils.py",
    "continuous_decode.py",
    "decode_compile.py",
    "device_runtime.py",
    "engine.py",
    "paddlex_adapter.py",
    "pipeline.py",
    "preprocessing.py",
    "text_compile.py",
    "vision_compile.py",
)


class RuntimeBoundaryTest(unittest.TestCase):
    def test_production_modules_do_not_import_runners_or_probes(self) -> None:
        violations: list[str] = []
        for filename in PRODUCTION_MODULES:
            tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(("run_", "probe_", "test_")):
                        violations.append(f"{filename}: from {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(("run_", "probe_", "test_")):
                            violations.append(f"{filename}: import {alias.name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
