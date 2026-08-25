#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_with_process_tree_memory import parse_npu_hbm_usage


NPU_SMI_OUTPUT = """\
+------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 6     910B2               | OK            | 102.3       49                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          3407 / 65536         |
+===========================+===============+====================================================+
| 7     910B2               | OK            | 102.5       49                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          13414 / 65536        |
+===========================+===============+====================================================+
"""


class NpuSmiParserTest(unittest.TestCase):
    def test_selects_requested_physical_npu(self) -> None:
        self.assertEqual(parse_npu_hbm_usage(NPU_SMI_OUTPUT, 7), (13414, 65536))

    def test_rejects_missing_physical_npu(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical NPU 4"):
            parse_npu_hbm_usage(NPU_SMI_OUTPUT, 4)


if __name__ == "__main__":
    unittest.main()
