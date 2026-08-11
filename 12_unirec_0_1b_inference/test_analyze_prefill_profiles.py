from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "analyze_prefill_profiles.py"
SPEC = importlib.util.spec_from_file_location("analyze_prefill_profiles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_graph_from_reference(reference: dict) -> dict:
    lanes = []
    for lane in reference["lanes"]:
        profile = lane["profile"]
        lanes.append(
            {
                "name": lane["name"],
                "first128_calls": lane["first128_calls"],
                "steady_device_event_mean_ms": lane[
                    "steady_device_event_mean_ms"
                ],
                "weighted_first128_device_s": lane[
                    "weighted_first128_device_s"
                ],
                "profile_steps": 1,
                "parsed_profile": {
                    "summary": {
                        "runs": [
                            {
                                "kernel_details": copy.deepcopy(profile["kernel"]),
                                "operator_details": copy.deepcopy(
                                    profile["operator"]
                                ),
                                "api_statistic": copy.deepcopy(profile["api"]),
                            }
                        ]
                    }
                },
            }
        )
    return {
        "format": "unirec_prefill_graph_profile_suite_v1",
        "environment": {
            **reference["environment"],
            "device_name": "Ascend310P3",
        },
        "config": {"profile_metric": "pipe", "profile_steps": 1},
        "first128_workload": copy.deepcopy(reference["first128_workload"]),
        "lanes": lanes,
    }


def _raw_layout_from_reference(reference: dict) -> dict:
    return {
        "config": copy.deepcopy(reference["config"]),
        "summary": copy.deepcopy(reference["summary"]),
        "pages": [copy.deepcopy(page) for page in reference["page_contracts"]],
    }


def _producer_reference() -> dict:
    return {
        "status": "ok",
        "offset": 0,
        "limit": 128,
        "workers": 1,
        "recognition_preprocess_threads": 16,
        "artifact_storage": "discard",
        "cross_cache_length": 512,
        "layout_execution": "torchair",
        "layout_batch_size": 1,
        "vision_full_batches": True,
        "recognition_input_contract": "compact_uint8_hwc",
        "worker_summary": {
            "stage_s": {
                "worker_detector_call_sum_s": MODULE.REFERENCE_PRODUCER[
                    "layout_s"
                ],
                "worker_recognition_prefill_sum_s": MODULE.REFERENCE_PRODUCER[
                    "recognition_prefill_s"
                ],
                "worker_recognition_prefill_cache_d2h_sum_s": (
                    MODULE.REFERENCE_PRODUCER["d2h_s"]
                ),
            }
        },
    }


class AnalyzePrefillProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph_reference = _read(MODULE.REFERENCE_GRAPH)
        cls.layout_reference = _read(MODULE.REFERENCE_LAYOUT)

    def test_identity_control_is_one(self) -> None:
        analysis = MODULE.analyze(
            _raw_graph_from_reference(self.graph_reference),
            _raw_layout_from_reference(self.layout_reference),
            _producer_reference(),
            graph_reference=self.graph_reference,
            layout_reference=self.layout_reference,
        )
        self.assertAlmostEqual(
            analysis["accounting"]["layout"]["graph_ratio"], 1.0
        )
        self.assertAlmostEqual(
            analysis["accounting"]["recognition_prefill"]["vision_graph_ratio"],
            1.0,
        )
        self.assertAlmostEqual(
            analysis["accounting"]["recognition_prefill"][
                "cross_kv_graph_ratio"
            ],
            1.0,
        )
        self.assertTrue(
            all(abs(target["gap_s"]) < 1e-9 for target in analysis[
                "ranked_optimization_targets"
            ])
        )

    def test_kernel_group_delta_is_weighted(self) -> None:
        rows = MODULE._compare_groups(
            [{"name": "SlowOp", "duration_us": 30.0}],
            [{"name": "SlowOp", "duration_us": 10.0}],
            calls=100,
        )
        self.assertEqual(rows[0]["name"], "SlowOp")
        self.assertAlmostEqual(rows[0]["ratio"], 3.0)
        self.assertAlmostEqual(rows[0]["weighted_first128_delta_s"], 0.002)

    def test_rejects_non_310_device(self) -> None:
        raw = _raw_graph_from_reference(self.graph_reference)
        raw["environment"]["device_name"] = "Ascend910B2"
        with self.assertRaisesRegex(ValueError, "expected a 310P"):
            MODULE._normalize_310_graph(raw)


if __name__ == "__main__":
    unittest.main()
