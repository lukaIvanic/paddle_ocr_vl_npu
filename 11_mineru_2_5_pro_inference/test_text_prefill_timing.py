"""CPU control tests for per-call packed text-prefill timing; no NPU inference."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from fixed_batch_engine import ContinuousBatchDecodeEngine
from mineru_prefill_timing import PrefillDeviceTimeline
from vision_timing_report import summarize_text_prefill_samples, summarize_vision_samples

HIDDEN = 8
VOCAB = 10
# Two packs (bucket 256 and 512) plus one eager overflow member.
LENGTHS = [100, 150, 400, 1500]
PACKS = [[0, 1], [2]]
OVERFLOW = [3]
BUCKETS = (128, 256, 512, 1024)


class FakeRuntime:
    def pack_indices(self, lengths):
        assert list(lengths) == LENGTHS
        return [list(pack) for pack in PACKS], list(OVERFLOW)

    def prepare(self, members):
        real = sum(member.sequence_length for member in members)
        bucket = next(b for b in BUCKETS if real <= b)
        return SimpleNamespace(real_tokens=real, physical_tokens=bucket, count=len(members))

    def run_prepared(self, prepared):
        return torch.zeros(1, prepared.count, HIDDEN)

    def redistribute_cache(self, prepared, destinations):
        return 64 * len(destinations)


def make_engine(*, metrics, samples):
    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(VOCAB, HIDDEN)),
            forward_prefill_static=lambda *, inputs_embeds, **_: torch.zeros(
                1, int(inputs_embeds.shape[1]), HIDDEN),
        ),
    )
    return ContinuousBatchDecodeEngine(
        model, None, batch_size=4, cache_length=2048, eos_token_id=0, pad_token_id=0,
        collect_prefill_metrics=metrics, packed_text_prefill_runtime=FakeRuntime(),
        text_prefill_timing_samples=samples,
    )


def make_entries():
    entries = []
    for slot, length in enumerate(LENGTHS):
        request = SimpleNamespace(
            input_ids=torch.zeros(1, length, dtype=torch.long),
            attention_mask=None, pixel_values=None, image_grid_thw=None,
            inputs_embeds=torch.zeros(1, length, HIDDEN),
            position_ids=torch.zeros(3, 1, length, dtype=torch.long),
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            max_new_tokens=4, host_staging=None,
        )
        entries.append((slot, 100 + slot, request))
    return entries


def make_arena():
    return SimpleNamespace(
        key_caches=(torch.zeros(4, 1, 2048, 2),),
        value_caches=(torch.zeros(4, 1, 2048, 2),),
        cache_length=2048,
    )


class FakeEvents:
    """Record-able NPU events with deterministic elapsed times."""

    def __init__(self):
        self.created = 0
        self.synchronize_calls = 0

    def __call__(self):
        self.created += 1
        event = Mock()
        event.elapsed_time.return_value = float(self.created)  # ms
        event.synchronize.side_effect = self._sync
        return event

    def _sync(self):
        self.synchronize_calls += 1


class TextPrefillTimingTests(unittest.TestCase):
    def test_tags_each_packed_and_overflow_call_with_one_sync(self):
        samples = []
        engine = make_engine(metrics=True, samples=samples)
        events = FakeEvents()
        with patch.object(PrefillDeviceTimeline, "_event", events):
            states, _elapsed, aggregate = engine._prefill_slots(make_arena(), make_entries())
        self.assertEqual(events.synchronize_calls, 1)
        self.assertEqual(sorted(states), [0, 1, 2, 3])
        self.assertIn("text_transformer_prefill", aggregate)
        self.assertEqual([row["stage"] for row in samples], ["text_transformer_prefill"] * 3)
        stripped = [{k: v for k, v in row.items() if k not in ("stage", "device_s")} for row in samples]
        self.assertEqual(stripped, [
            dict(route="bucket_256", real_tokens=250, physical_tokens=256, members=2,
                 member_lengths=[100, 150], request_ids=[100, 101]),
            dict(route="bucket_512", real_tokens=400, physical_tokens=512, members=1,
                 member_lengths=[400], request_ids=[102]),
            dict(route="eager_overflow", real_tokens=1500, physical_tokens=1500, members=1,
                 member_lengths=[1500], request_ids=[103]),
        ])
        self.assertTrue(all(row["device_s"] > 0 for row in samples))
        # Totals still include every text_transformer_prefill event.
        self.assertAlmostEqual(aggregate["text_transformer_prefill"],
                               sum(row["device_s"] for row in samples))

    def test_metrics_off_creates_no_events_tags_or_rows(self):
        samples = []
        engine = make_engine(metrics=False, samples=samples)
        seen_tags = []
        original = PrefillDeviceTimeline.measure

        def spy(timeline, name, fn, *, tags=None):
            seen_tags.append(tags)
            return original(timeline, name, fn, tags=tags)

        with patch.object(PrefillDeviceTimeline, "_event", side_effect=AssertionError("event created")), \
                patch.object(PrefillDeviceTimeline, "measure", spy):
            _states, _elapsed, aggregate = engine._prefill_slots(make_arena(), make_entries())
        self.assertEqual(samples, [])
        self.assertTrue(seen_tags and all(tags is None for tags in seen_tags))
        self.assertNotIn("text_transformer_prefill", aggregate)

    def test_metrics_on_without_samples_keeps_untagged_calls(self):
        engine = make_engine(metrics=True, samples=None)
        seen_tags = []
        original = PrefillDeviceTimeline.measure

        def spy(timeline, name, fn, *, tags=None):
            seen_tags.append(tags)
            return original(timeline, name, fn, tags=tags)

        events = FakeEvents()
        with patch.object(PrefillDeviceTimeline, "_event", events), \
                patch.object(PrefillDeviceTimeline, "measure", spy):
            _states, _elapsed, aggregate = engine._prefill_slots(make_arena(), make_entries())
        self.assertTrue(all(tags is None for tags in seen_tags))
        self.assertEqual(events.synchronize_calls, 1)
        self.assertIn("text_transformer_prefill", aggregate)

    def test_summary_matches_vision_shape(self):
        rows = [
            dict(route="bucket_256", real_tokens=250, physical_tokens=256, members=2, device_s=.004),
            dict(route="bucket_256", real_tokens=200, physical_tokens=256, members=1, device_s=.002),
            dict(route="bucket_1024", real_tokens=900, physical_tokens=1024, members=3, device_s=.010),
            dict(route="eager_overflow", real_tokens=1500, physical_tokens=1500, members=1, device_s=.020),
        ]
        text = summarize_text_prefill_samples(rows)
        vision = summarize_vision_samples(rows)
        self.assertEqual(set(text), set(vision))
        self.assertEqual(text["scope"], "text_transformer_prefill_device_event_region")
        self.assertEqual(set(text["by_route"]), {"bucket_256", "bucket_1024", "eager_overflow"})
        self.assertEqual(text["all"]["calls"], 4)
        bucket = text["by_route"]["bucket_256"]
        self.assertEqual((bucket["calls"], bucket["members"]), (2, 3))
        self.assertAlmostEqual(bucket["physical_tok_s"], 512 / .006)
        self.assertAlmostEqual(bucket["real_tok_s"], 450 / .006)
        self.assertEqual(text["by_exact_shape"]["eager_overflow:S1500"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
