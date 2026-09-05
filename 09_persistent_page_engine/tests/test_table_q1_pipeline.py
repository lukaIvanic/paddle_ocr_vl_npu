"""CPU control tests only; real NPU overlap/parity needs the serving run."""
import ast
import contextlib
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class Event:
    def synchronize(self):
        pass


class Stream:
    def synchronize(self):
        pass

    def record_event(self):
        return Event()

    def wait_event(self, event):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Load the actual control classes without importing model/TorchAir code.
        source = Path(__file__).resolve().parents[1] / "paddleocr_vl/serving/table_interleaved_runtime.py"
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name in ("_Call", "_Q1Pipeline")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        empty = torch.empty
        def unpinned(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(torch, "npu", SimpleNamespace(
            current_stream=lambda *_: Stream(), Stream=lambda **_: Stream(),
            stream=lambda _: contextlib.nullcontext(),
        ), create=True))
        self.stack.enter_context(patch.object(torch, "empty", unpinned))
        self.executed = []
        def forward(ids, positions, rope):
            self.executed.append(positions.tolist())
            self.assertLess(int(positions.max()), 6)
            return ids + positions[:, None] + 1
        self.call = namespace["_Q1Pipeline"](
            SimpleNamespace(fn=forward), 2, 1, torch.device("cpu"), record_device_timing=False,
        )
        self.arena = SimpleNamespace(
            cache=SimpleNamespace(cache_length=6), rope=torch.zeros((2, 1)),
            tensors=lambda *_: (), recognizer=SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(eos_token_id=99)),
            ),
        )

    def test_independent_tokens_and_reused_lookahead(self):
        rows = [SimpleNamespace(slot=0, position=1, tokens=[10]),
                SimpleNamespace(slot=1, position=2, tokens=[20])]
        for _ in range(3):
            expected = [row.tokens[-1] + row.position + 1 for row in rows]
            result, _ = self.call.run_pipelined(self.arena, 0, rows)
            self.assertEqual([row[0] for row in result], expected)
            for row, token in zip(rows, expected):
                row.tokens.append(token)
                row.position += 1
        self.assertEqual(self.call.reused_lookaheads, 2)
        self.call.drain()

    def test_finished_slot_and_replacement_discard_old_pending(self):
        first = SimpleNamespace(slot=0, position=1, tokens=[10])
        result, _ = self.call.run_pipelined(self.arena, 0, [first])
        self.assertEqual(result[0], [12])
        replacement = SimpleNamespace(slot=0, position=2, tokens=[40])
        result, _ = self.call.run_pipelined(self.arena, 0, [replacement])
        self.assertEqual(result[0], [43])
        self.assertEqual(self.call.discarded_lookaheads, 1)
        self.call.drain()

    def test_cache_boundary_has_no_out_of_bounds_lookahead(self):
        row = SimpleNamespace(slot=1, position=5, tokens=[4])
        result, _ = self.call.run_pipelined(self.arena, 0, [row])
        self.assertEqual(result[1], [10])
        self.assertEqual(len(self.executed), 1)
        self.assertIsNone(self.call.pending)


if __name__ == "__main__":
    unittest.main()
