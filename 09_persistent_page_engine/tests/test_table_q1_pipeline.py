"""CPU control tests only; real NPU overlap/parity needs the serving run."""
import ast
import contextlib
import __future__
from dataclasses import dataclass, replace
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import time

import torch
from PIL import Image


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
                 if isinstance(node, ast.ClassDef) and node.name in ("_Arena", "_Call", "_Q1Pipeline")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        self.classes = namespace
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

    def guarded_arena(self):
        tensors = tuple(torch.arange(4*2*6*4, dtype=torch.float32).reshape(4,2,6,4).clone() for _ in range(2))
        cache = SimpleNamespace(cache_length=6, flat_tensors=lambda: tensors)
        recognizer = SimpleNamespace(cache_length=6, device=torch.device('cpu'), dtype=torch.float32,
            model=SimpleNamespace(allocate_static_cache=lambda **_:cache,
                                  config=SimpleNamespace(eos_token_id=99)))
        arena = self.classes['_Arena'](recognizer,4,prefix_capacity=4)
        def forward(ids, positions, rope, *kv):
            q=ids.shape[1]
            for row, position in enumerate(positions.tolist()):
                for t in kv:
                    t[row,:,position:position+q].fill_(int(ids[row,0])+1)
            return ids+1
        return arena, tensors, forward

    def test_nonadjacent_verifier_restores_all_unselected_kv(self):
        arena,tensors,forward=self.guarded_arena()
        original=[t.clone() for t in tensors]
        call=self.classes['_Call'](SimpleNamespace(fn=forward),4,4,torch.device('cpu'),record_device_timing=False)
        call.protected_slots=(1,2)
        rows=[SimpleNamespace(slot=0,position=2,tokens=[10]),SimpleNamespace(slot=3,position=1,tokens=[20])]
        out,_=call.run(arena,0,rows,{})
        for a,b in zip(tensors,original):
            self.assertTrue(torch.equal(a[1:3],b[1:3]))
            self.assertTrue(torch.equal(a[0,:,:2],b[0,:,:2]))
            self.assertTrue(torch.all(a[0,:,2:6]==11))
            self.assertTrue(torch.all(a[3,:,1:5]==21))
        self.assertEqual(out[0][0],11)
        self.assertEqual(out[3][0],21)
        self.assertEqual(arena.prefix_calls,1)

    def test_nonadjacent_q1_lookahead_restores_dummy_prefix_each_launch(self):
        arena,tensors,forward=self.guarded_arena()
        original=[t.clone() for t in tensors]
        call=self.classes['_Q1Pipeline'](SimpleNamespace(fn=forward),4,1,torch.device('cpu'),record_device_timing=False)
        call.protected_slots=(1,2)
        rows=[SimpleNamespace(slot=0,position=1,tokens=[10]),SimpleNamespace(slot=3,position=2,tokens=[20])]
        for _ in range(2):
            out,_=call.run_pipelined(arena,0,rows)
            for a,b in zip(tensors,original):
                self.assertTrue(torch.equal(a[1:3],b[1:3]))
            for row in rows:
                row.tokens.append(out[row.slot][0]); row.position+=1
        self.assertEqual(call.reused_lookaheads,1)
        self.assertEqual(arena.prefix_calls,3)
        call.drain()

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

    def test_pending_ownership_is_per_arena_and_slot(self):
        row = SimpleNamespace(slot=0, position=1, tokens=[10])
        self.call.run_pipelined(self.arena, 0, [row])
        self.assertTrue(self.call.conflicts(self.arena, {0}))
        self.assertFalse(self.call.conflicts(self.arena, {1}))
        self.assertFalse(self.call.conflicts(SimpleNamespace(), {0}))
        # Unrelated work did not consume/discard this result.
        row.tokens.append(12)
        row.position += 1
        result, _ = self.call.run_pipelined(self.arena, 0, [row])
        self.assertEqual(result[0], [15])
        self.assertEqual(self.call.reused_lookaheads, 1)
        self.call.drain()


class RequestIdentityTests(unittest.TestCase):
    def test_source_identity_survives_preprocessing_but_runtime_ids_are_unique(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts/serve_table_speculative_api.py"
        module = ast.parse(source_path.read_text())
        route = next(node for node in module.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_table_route")
        worker = next(node for node in module.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_interleaved_worker_loop")
        mode = next(node for node in worker.body if isinstance(node, ast.With))
        prepare = next(node for node in mode.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
        @dataclass
        class Request:
            request_id: str
        sources = {"rotated-source": {"request_id": "rotated-source"}}
        observed = []
        def rows(source, image, args):
            observed.append(source["request_id"])
            return [Request("row0"), Request("row1")], [], {"row_draft_rotation_cw": 90}
        live = SimpleNamespace(_exact_target_crop_from_raw=lambda source, image: image,
                               _prepare_rows=rows, _b1_args=lambda args: args)
        b1, draft, runtime = Mock(), Mock(), Mock()
        runtime.jobs = {}
        metadata = {}
        namespace = dict(Image=Image, io=io, time=time, replace=replace,
                         targets_by_id=sources, config={"height_threshold_px": 0}, counts=None,
                         live_lab=live, args=None, b1=b1, draft=draft, runtime=runtime,
                         fixed_lab=SimpleNamespace(request_for=lambda source, *_: Request(source["request_id"])),
                         metadata=metadata)
        exec(compile(ast.Module(body=[route, prepare], type_ignores=[]), str(source_path), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        blob = io.BytesIO()
        Image.new("RGB", (8, 8)).save(blob, format="PNG")
        for key in ("http-a", "http-b"):
            namespace["prepare"]({"request_id": key, "source_request_id": "rotated-source",
                                  "crop_type": "table", "image_bytes": blob.getvalue()})
        self.assertEqual(observed, ["rotated-source", "rotated-source"])
        self.assertEqual(sources["rotated-source"]["request_id"], "rotated-source")
        self.assertEqual([call.args[0].request_id for call in b1._prepare_cpu.call_args_list], ["http-a", "http-b"])
        self.assertEqual([[r.request_id for r in call.args[0]] for call in draft._iter_packed_prefill_groups.call_args_list],
                         [["http-a:row_0000", "http-a:row_0001"], ["http-b:row_0000", "http-b:row_0001"]])


if __name__ == "__main__":
    unittest.main()
