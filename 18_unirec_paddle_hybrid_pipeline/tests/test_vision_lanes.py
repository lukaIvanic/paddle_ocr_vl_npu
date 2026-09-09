from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_lanes import UniRecVisionLanes


class Owner:
    def __init__(self, runtime, **options):
        self.options = options
        self.closed = False

    def encode_inputs(self, inputs):
        self.inputs = inputs
        return inputs, {"pairs": [[
            dict(key="a", calls=2, real_rows=3, wall_s=.2),
            dict(key="b", calls=1, real_rows=1, wall_s=.1),
        ]]}

    def close(self):
        self.closed = True


class VisionLaneTests(unittest.TestCase):
    def test_reuses_inputs_owner_and_keeps_graphs(self):
        lane = UniRecVisionLanes(SimpleNamespace(summary=lambda: {"bucket_calls": {}}),
                                2, owner_factory=Owner)
        inputs = [object(), object()]
        self.assertIs(lane.encode(inputs), inputs)
        self.assertIs(lane.owner.inputs, inputs)
        self.assertEqual(lane.owner.options, dict(lanes=2, same_key_shards=1,
            sharded_key_count=0, deinitialize_tbe_after_first_group=False,
            keep_all_loaded_graphs=True))
        lane.encode(inputs)
        s = lane.summary()["hybrid_lanes"]
        self.assertEqual(s['turn_wall']['count'], 2)
        self.assertEqual(s['group_width_counts'], {2: 2})
        self.assertEqual(s['by_key']['a']['calls'], 4)
        self.assertEqual(s['by_key']['a']['real_rows'], 6)
        self.assertAlmostEqual(s['lane_host_spans']['total_s'], .6)
        lane.close()
        self.assertTrue(lane.owner.closed)

    def test_empty_summary(self):
        lane = UniRecVisionLanes(SimpleNamespace(summary=lambda: {}), 1, owner_factory=Owner)
        self.assertIsNone(lane.summary()['hybrid_lanes']['turn_wall'])
        lane.close()

    def test_failure_surfaces_without_retry(self):
        class FailedOwner(Owner):
            def encode_inputs(self, inputs):
                raise RuntimeError('lane failed')
        lane = UniRecVisionLanes(None, 4, owner_factory=FailedOwner)
        with self.assertRaisesRegex(RuntimeError, 'lane failed'):
            lane.encode([])
        self.assertEqual(lane.turns, [])
        lane.close()
