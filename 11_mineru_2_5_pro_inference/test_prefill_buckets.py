"""CPU contracts for opt-in resolution, packing and window KV scheduling."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from fixed_batch_engine import ContinuousBatchDecodeEngine
from local_modeling_mineru import LocalMinerUStaticCache
from prefill_buckets_config import PRODUCTION_PREFILL_OPTIONS
from run_page_pipeline import pipeline_args
from run_official_transformers_omnidocbench import apply_processor_pixel_limits
from text_crop_processor import make_text_crop_processor, select_crop_processor
from text_prefill_compile import MinerUPackedTextPrefillRuntime, select_text_prefill_bucket
from test_streaming_decode import FakeEngine, FakeSource
from streaming_decode import run_decode_stream
from test_text_prefill_timing import make_engine, make_entries, FakeEvents, LENGTHS
from mineru_prefill_timing import PrefillDeviceTimeline


class CropTests(unittest.TestCase):
    def test_label_selection_and_exact_global_cap_processor_output(self):
        processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor())
        apply_processor_pixel_limits(processor.image_processor, min_pixels=25088, max_pixels=602112)
        capped = make_text_crop_processor(processor, 401408)
        reference = copy.deepcopy(processor)
        apply_processor_pixel_limits(reference.image_processor, max_pixels=401408)
        for label in ('title', 'header', 'table', 'equation', 'Text', None):
            self.assertIs(select_crop_processor(processor, capped, label), processor)
        self.assertIs(select_crop_processor(processor, capped, 'text'), capped)
        self.assertIsNone(make_text_crop_processor(processor, None))
        self.assertEqual(processor.image_processor.size['longest_edge'], 602112)
        for size in ((1100, 700), (700, 1100), (56, 56), (2100, 210)):
            image = Image.new('RGB', size, (80, 110, 240))
            got = capped.image_processor(images=[image], return_tensors='pt')
            expected = reference.image_processor(images=[image], return_tensors='pt')
            self.assertEqual(got.pixel_values.shape[0], expected.pixel_values.shape[0])
            self.assertTrue(torch.equal(got.image_grid_thw, expected.image_grid_thw))
            self.assertTrue(torch.equal(got.pixel_values, expected.pixel_values))
        self.assertEqual(capped.image_processor.size['shortest_edge'], 25088)

    def test_streaming_passes_layout_label_not_prompt(self):
        from streaming_pipeline import MinerUPageSource
        observed = []
        adapter = SimpleNamespace(text_crop_processor=object(),
            processor=SimpleNamespace(apply_chat_template=lambda *a, **k: 'same prompt'),
            build_messages=lambda *a, **k: [],
            _prepare_cpu_inputs=lambda image, chat, **kw: observed.append(kw) or (None, None, None, 0, 0))
        source = SimpleNamespace(adapter=adapter, trace=None, h2d=None)
        for label in ('text', 'title', None):
            MinerUPageSource._prepare_cpu(source, ({'block_type': label}, None, 'ocr', None))
        self.assertEqual(observed, [{'block_type': x} for x in ('text', 'title', None)])

    def test_defaults_and_explicit_set(self):
        default = pipeline_args(['--dataset-json', 'data.json', '--output-dir', 'output'])
        self.assertIsNone(default.processor_text_max_pixels)
        self.assertIsNone(default.local_text_pack_target)
        self.assertEqual(default.local_text_prefill_schedule, 'admission')
        self.assertEqual(default.processor_max_pixels, 1103872)
        candidate = pipeline_args(['--dataset-json', 'data.json', '--output-dir', 'output'] + PRODUCTION_PREFILL_OPTIONS)
        self.assertEqual(candidate.local_text_pack_target, 384)
        self.assertEqual(candidate.processor_text_max_pixels, 401408)
        self.assertEqual(candidate.local_vision_pack_target, 768)
        self.assertEqual(candidate.local_vision_lookahead, 32)


class PackingTests(unittest.TestCase):
    def runtime(self, target=None, max_members=32):
        runtime = object.__new__(MinerUPackedTextPrefillRuntime)
        runtime.buckets = (128, 256, 384, 512, 576, 832, 1024)
        runtime.max_members = max_members
        if target is not None:
            runtime.pack_target = target
        return runtime

    def test_target_best_fit_and_large_singletons_and_overflow(self):
        lengths = [100, 150, 234, 385, 513, 700, 1025, 50]
        runtime = self.runtime(384)
        packs, overflow = runtime.pack_indices(lengths)
        self.assertEqual(packs, [[5], [4], [3], [2, 1], [0, 7]])
        self.assertEqual(overflow, [6])
        self.assertEqual([select_text_prefill_bucket(sum(lengths[i] for i in pack), runtime.buckets)
                          for pack in packs], [832, 576, 512, 384, 256])

    def test_default_largest_bucket_and_member_bound(self):
        self.assertEqual(self.runtime().pack_indices([500, 400, 100]), ([[0, 1, 2]], []))
        self.assertEqual(self.runtime(384, 2).pack_indices([100] * 5), ([[0, 1], [2, 3], [4]], []))
        self.assertEqual(self.runtime(384).pack_indices([384, 1, 383]), ([[0], [2, 1]], []))


class WindowEngine(FakeEngine):
    text_prefill_schedule = 'window'
    vision_lookahead = 4

    def __init__(self):
        super().__init__()
        self.windows, self.admitted, self.released = [], [], []
        self.live = set()
        self.high_water = 0

    def _prefill_vision_window(self, window):
        assert not self.live, 'previous window still staged'
        self.windows.append([i for i, _ in window])
        self.live.update(i for i, _ in window)
        self.high_water = max(self.high_water, len(self.live))
        return window, 0, {'text_prefill_first_token_read_count': 1}

    def admit_prefilled_slots(self, arena, entries):
        states, elapsed, metrics = super()._prefill_slots(arena, entries)
        for _, i, _ in entries:
            self.admitted.append(i)
            self.live.remove(i)
            self.released.append(i)
        return states, elapsed, metrics

    def _prefill_slots(self, *args):
        raise AssertionError('window mode tried admission prefill')


class WindowTests(unittest.TestCase):
    def test_scheduler_order_once_eos_length_and_release(self):
        items = [(0, 99, 5), (1, 11, 1), (2, 21, 0), (3, 31, 8),
                 (4, 41, 8), (5, 51, 8), (6, 99, 8), (7, 61, 8), (8, 71, 8)]
        reference = FakeSource(items)
        run_decode_stream(FakeEngine(), reference)
        source, engine = FakeSource(items), WindowEngine()
        report = run_decode_stream(engine, source)
        self.assertEqual(source.results, reference.results)
        self.assertEqual(engine.admitted, list(range(9)))
        self.assertEqual(engine.released, list(range(9)))
        self.assertFalse(engine.live)
        self.assertLessEqual(engine.high_water, 4)
        self.assertEqual(engine.windows, [[0, 1, 2, 3], [4, 5, 6, 7], [8]])
        self.assertEqual(report['prefill_metrics']['text_prefill_first_token_read_count'], 3)

    def test_real_window_prefill_uses_exact_caches_single_read_and_no_release_sync(self):
        engine = make_engine(metrics=True, samples=[])
        allocations = []
        def allocate(**kw):
            length = kw['cache_length']
            allocations.append(length)
            return LocalMinerUStaticCache((torch.zeros(1, 1, length, 2),),
                                         (torch.zeros(1, 1, length, 2),), length)
        engine.model.allocate_static_cache = allocate
        requests = [(index, request) for _, index, request in make_entries()]
        events = FakeEvents()
        cpu_reads = []
        tensor_cpu = torch.Tensor.cpu
        def read_cpu(tensor, *args, **kwargs):
            cpu_reads.append(tensor.numel())
            return tensor_cpu(tensor, *args, **kwargs)
        with patch.object(PrefillDeviceTimeline, '_event', events), \
             patch.object(torch.Tensor, 'cpu', read_cpu), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('per-request token read')):
            leases, _, metrics = engine._prefill_vision_window(requests)
        self.assertEqual(allocations, LENGTHS)
        self.assertEqual(cpu_reads, [len(LENGTHS)])
        self.assertEqual(metrics['text_prefill_first_token_read_count'], 1)
        self.assertEqual(events.synchronize_calls, 0)  # The batched CPU read is the fence.
        self.assertEqual(len(engine.text_prefill_timing_samples), 3)
        destination = LocalMinerUStaticCache((torch.zeros(4, 1, 2048, 2),),
                                            (torch.zeros(4, 1, 2048, 2),), 2048)
        with torch.inference_mode():
            for row, (_, lease) in enumerate(leases):
                lease.cache.key_caches[0].fill_(row + 1)
                lease.cache.value_caches[0].fill_(-row - 1)
        entries = [(row, index, lease) for row, (index, lease) in enumerate(leases)]
        with patch('fixed_batch_engine.maybe_sync_device', side_effect=AssertionError('extra admission fence')):
            states, _, _ = engine.admit_prefilled_slots(destination, entries)
        self.assertEqual(len(states), 4)
        for row, length in enumerate(LENGTHS):
            self.assertTrue(torch.all(destination.key_caches[0][row, :, :length] == row + 1))
            self.assertTrue(torch.all(destination.value_caches[0][row, :, :length] == -row - 1))
            self.assertTrue(torch.all(destination.key_caches[0][row, :, length:] == 0))
            self.assertEqual(states[row]['cache_position'].item(), length)
        self.assertTrue(all(lease.cache is None for _, lease in leases))
        self.assertTrue(all(not request.host_staging for _, request in requests))


if __name__ == '__main__':
    unittest.main()
