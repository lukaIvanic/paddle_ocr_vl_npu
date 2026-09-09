"""UniRec's existing multi-lane owner, bounded by one hybrid prefill turn."""
from collections import Counter
import time

from hybrid_timing import distribution


class UniRecVisionLanes:
    def __init__(self, runtime, lanes, *, owner_factory=None):
        if owner_factory is None:
            from bounded_vision_owner import BoundedVisionOwner
            owner_factory = BoundedVisionOwner
        self.runtime = runtime
        self.lanes = lanes
        self.owner = owner_factory(
            runtime, lanes=lanes, same_key_shards=1, sharded_key_count=0,
            deinitialize_tbe_after_first_group=False, keep_all_loaded_graphs=True,
        )
        self.turns = []
        self.lane_spans = []
        self.group_widths = Counter()
        self.by_key = {}

    def encode(self, inputs):
        start = time.perf_counter()
        outputs, report = self.owner.encode_inputs(inputs)
        self.turns.append(time.perf_counter() - start)
        # These are overlapping host submit-through-stream-sync intervals,
        # not additive critical-path time or kernel-active time.
        for group in report["pairs"]:
            self.group_widths[len(group)] += 1
            for lane in group:
                self.lane_spans.append(lane["wall_s"])
                stats = self.by_key.setdefault(lane["key"], dict(calls=0, real_rows=0, spans=[]))
                stats["calls"] += lane["calls"]
                stats["real_rows"] += lane["real_rows"]
                stats["spans"].append(lane["wall_s"])
        return outputs

    def summary(self):
        def dist(values):
            return distribution(values) if values else None
        return {**self.runtime.summary(), "hybrid_lanes": {
            "lanes": self.lanes,
            "same_key_shards": 1,
            "residency": "keep_all",
            "input_storage": "existing_cpu_arrays",
            "timing_basis": "Wall envelope joins all vision lanes before returning; lane spans overlap and must not be summed as critical-path time.",
            "turn_wall": dist(self.turns),
            "lane_host_spans": dist(self.lane_spans),
            "group_width_counts": dict(self.group_widths),
            "by_key": {key: dict(calls=row["calls"], real_rows=row["real_rows"],
                                 lane_host_spans=dist(row["spans"]))
                       for key, row in self.by_key.items()},
        }}

    def close(self):
        self.owner.close()
