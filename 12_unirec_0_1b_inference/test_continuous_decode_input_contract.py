#!/usr/bin/env python3
"""CPU check for the production UniRec decode input guard contract."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Keep this contract test CPU-only and independent of the Transformers model
# package that is intentionally absent from the local authoring environment.
modeling_stub = types.ModuleType("modeling_optimized_unirec")
modeling_stub.LOCAL_UNIREC_STATIC_CACHE_LEN = 2048


@dataclass
class FakeStaticCache:
    key_cache: tuple[torch.Tensor, ...]
    value_cache: tuple[torch.Tensor, ...]
    cache_len: int
    cross_key_cache: tuple[torch.Tensor, ...]
    cross_value_cache: tuple[torch.Tensor, ...]
    cross_attention_mask: torch.Tensor
    actual_cross_attention_length: int | None
    packed_cross_kv: torch.Tensor | None


modeling_stub.LocalUniRecStaticCache = FakeStaticCache
modeling_stub.OptimizedUniRecRunner = type("OptimizedUniRecRunner", (), {})
modeling_stub.UniRecPrefilledItem = type("UniRecPrefilledItem", (), {})
sys.modules.setdefault("modeling_optimized_unirec", modeling_stub)

from continuous_unirec import (
    ContinuousReadyItem,
    ContinuousUniRecDecoder,
    ContinuousWorkerPrefilledItem,
    production_decode_cache_parent,
)
from persistent_ready_queue import PersistentReadyQueue


class FakeDecodeModel:
    def __init__(self) -> None:
        self.decoder = types.SimpleNamespace(layers=[object()])

    def forward_cached_logits(self, **kwargs: object) -> torch.Tensor:
        batch = int(kwargs["decoder_input_ids"].shape[0])
        return torch.zeros((batch, 1), dtype=torch.float32)

    @staticmethod
    def select_next_token(logits: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (logits.shape[0],),
            2,
            dtype=torch.long,
            device=logits.device,
        )


class DelayedFakeDecodeModel(FakeDecodeModel):
    def __init__(self) -> None:
        super().__init__()
        self.cache_positions = torch.zeros(1, dtype=torch.int64)
        self.decode_started = threading.Event()

    def forward_cached_logits(self, **kwargs: object) -> torch.Tensor:
        self.decode_started.set()
        time.sleep(0.002)
        self.cache_positions = kwargs["cache_position"].detach().cpu().clone()
        batch = int(kwargs["decoder_input_ids"].shape[0])
        return torch.zeros((batch, 1), dtype=torch.float32)

    def select_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.where(
            self.cache_positions >= 5,
            torch.full_like(self.cache_positions, 2),
            torch.full_like(self.cache_positions, 3),
        ).to(device=logits.device)


class FakeDecodeRunner:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(
            decoder_attention_heads=1,
            d_model=1,
            decoder_start_token_id=1,
            eos_token_id=2,
        )
        self.model = FakeDecodeModel()
        self.device = "cpu"
        self.dtype = torch.float32
        self.dtype_name = "float32"

    @staticmethod
    def _get_static_cross_cache_len() -> int:
        return 4

    @staticmethod
    def _decode_text_batch(rows: list[list[int]]) -> list[str]:
        return [",".join(str(token) for token in row) for row in rows]


def ready_item(name: str) -> ContinuousReadyItem:
    packed = np.zeros((2, 1, 1, 2, 1), dtype=np.float32)
    return ContinuousReadyItem(
        request_id=name,
        payload=name,
        prefilled=ContinuousWorkerPrefilledItem(
            packed_cross_kv=packed,
            prep={"image": name, "prepare_total_s": 0.0},
            prefill_s=0.0,
            actual_cross_attention_length=2,
        ),
    )


class ContinuousDecodeInputContractTest(unittest.TestCase):
    def test_decode_cache_parent_is_stable_and_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = production_decode_cache_parent(root)
            second = production_decode_cache_parent(root)
            self.assertEqual(first, second)
            self.assertEqual(first.parent, root.resolve())
            self.assertRegex(
                first.name,
                r"^production_decode_graph_[0-9a-f]{16}$",
            )

    def test_decode_cache_parent_override_reuses_validated_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / "previous_complete_cache"
            previous.mkdir()
            with unittest.mock.patch.dict(
                "os.environ",
                {
                    "UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE": str(
                        previous
                    )
                },
            ):
                self.assertEqual(
                    production_decode_cache_parent(root),
                    previous.resolve(),
                )

    def test_decode_device_inputs_are_static_inference_tensors(self) -> None:
        next_token, cache_position = (
            ContinuousUniRecDecoder._allocate_decode_device_inputs(7, "cpu")
        )
        self.assertEqual(next_token.shape, (7, 1))
        self.assertEqual(cache_position.shape, (7,))
        self.assertEqual(next_token.dtype, torch.long)
        self.assertEqual(cache_position.dtype, torch.int64)
        self.assertTrue(next_token.is_contiguous())
        self.assertTrue(cache_position.is_contiguous())
        self.assertTrue(next_token.is_inference())
        self.assertTrue(cache_position.is_inference())

    def test_persistent_decoder_survives_idle_gap_and_later_request(self) -> None:
        source: PersistentReadyQueue[ContinuousReadyItem] = (
            PersistentReadyQueue(maxsize=4)
        )
        completed: list[str] = []
        completed_at: list[float] = []
        first_completed = threading.Event()
        second_completed = threading.Event()
        decoder = ContinuousUniRecDecoder(
            runner=FakeDecodeRunner(),
            batch_size=4,
            max_length=8,
            decode_mode="eager",
            compile_backend="eager",
            self_cache_length=8,
            cross_cache_length=4,
        )

        def publish() -> None:
            source.put(ready_item("first"))
            if not first_completed.wait(timeout=2.0):
                source.close()
                return
            time.sleep(0.03)
            source.put(ready_item("second"))
            second_completed.wait(timeout=2.0)
            source.close()

        def complete(item: object) -> None:
            completed.append(item.request_id)
            completed_at.append(time.monotonic())
            if len(completed) == 1:
                first_completed.set()
            elif len(completed) == 2:
                second_completed.set()

        thread = threading.Thread(target=publish)
        thread.start()
        real_empty = torch.empty

        def cpu_safe_empty(*args: object, **kwargs: object) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with unittest.mock.patch.object(
            torch,
            "empty",
            side_effect=cpu_safe_empty,
        ):
            summary = decoder.run(
                source,
                on_complete=complete,
                partial_batch_wait_s=0.005,
            )
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(completed, ["first", "second"])
        self.assertGreaterEqual(completed_at[1] - completed_at[0], 0.02)
        self.assertEqual(summary["submitted"], 2)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["source_mode"], "persistent_queue")

    def test_persistent_decoder_fills_inactive_slots_while_decoding(self) -> None:
        source: PersistentReadyQueue[ContinuousReadyItem] = (
            PersistentReadyQueue(maxsize=4)
        )
        runner = FakeDecodeRunner()
        runner.model = DelayedFakeDecodeModel()
        decoder = ContinuousUniRecDecoder(
            runner=runner,
            batch_size=4,
            max_length=8,
            decode_mode="eager",
            compile_backend="eager",
            self_cache_length=8,
            cross_cache_length=4,
        )
        completed: list[str] = []

        def publish() -> None:
            source.put(ready_item("first"))
            if not runner.model.decode_started.wait(timeout=1.0):
                source.close()
                return
            source.put(ready_item("second"))
            source.close()

        thread = threading.Thread(target=publish)
        thread.start()
        real_empty = torch.empty

        def cpu_safe_empty(*args: object, **kwargs: object) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with unittest.mock.patch.object(
            torch,
            "empty",
            side_effect=cpu_safe_empty,
        ):
            summary = decoder.run(
                source,
                on_complete=lambda item: completed.append(item.request_id),
                partial_batch_wait_s=0.0,
            )
        thread.join(timeout=1.0)

        self.assertEqual(sorted(completed), ["first", "second"])
        self.assertGreaterEqual(summary["opportunistic_slot_refills"], 1)


if __name__ == "__main__":
    unittest.main()
