"""Frozen source-only receipt for the scoped reading-order pass at a191a2a2.

No torch, NPU execution or compilation. Compare with the accepted Poisson100
baseline, permitting only the listed method/local renames, definition
order, comments and recognizer docstrings. The incoming-crop preparation class
also permits its agreed class/attribute renames and class description. The
scheduler permits only the explicitly listed identifier renames. Later structural
integration is deliberately not claimed to be a naming-only change.
"""
import ast
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATH = '19_table_ocr_serving/p02_serving_runtime.py'
BASELINE = '364a0c2b'
RENAMES = {
    '_stage_crop': '_copy_inputs_to_npu',
    '_enqueue_crop': '_submit_vision_and_text_prefill',
    '_finalize_crop': '_wait_for_prefill_result',
    '_result_from_completion': '_build_recognition_result',
    '_run_decode': '_run_continuous_decoding_loop_until_pipeline_shutdown',
}
LOCAL_RENAMES = {
    'serve': {'ready_source': 'request_preparation'},
    '_run_continuous_decoding_loop_until_pipeline_shutdown': {
        'ready_source': 'request_preparation', 'decoded': 'decode_summary',
        'handle_completion': 'send_finished_crop_result', 'completion': 'completed_crop',
        'fraction': 'fraction_of_decode_token_slots', 'decode_wall_s': 'decoding_loop_elapsed_s',
    },
    '_prefill_for_decode': {
        'prepared': 'prepared_crop', 'staged': 'input_transfer',
        'inflight': 'prefill_submission', 'prefilled': 'prefill_result',
    },
    '_copy_inputs_to_npu': {'prepared': 'prepared_crop', 'move_inputs': 'copy_input_tensors'},
    '_submit_vision_and_text_prefill': {'staged': 'input_transfer', 'prepared': 'prepared_crop'},
    '_wait_for_prefill_result': {
        'inflight': 'prefill_submission', 'staged': 'input_transfer', 'prepared': 'prepared_crop',
        'cpu': 'cpu_timing', 'spans': 'device_stage_spans', 'seconds': 'stage_seconds',
    },
    '_build_recognition_result': {
        'prefilled': 'prefill_result', 'completion': 'completed_crop',
        'cpu': 'cpu_timing', 'prefill': 'prefill_timing', 'timing': 'request_timing',
    },
}


def recognizer(source):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == 'ContinuousRecognizer')


class RecognizerReadingOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{BASELINE}:{PATH}'], text=True)
        cls.after = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'a191a2a2:{PATH}'], text=True)

    def test_method_bodies_signatures_and_decorators_are_unchanged(self):
        class AllowedEdits(ast.NodeTransformer):
            def __init__(self, local_names):
                self.local_names = local_names

            def visit_FunctionDef(self, node):
                node.name = RENAMES.get(node.name, self.local_names.get(node.name, node.name))
                self.generic_visit(node)
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
                return node

            def visit_Attribute(self, node):
                node.attr = RENAMES.get(node.attr, node.attr)
                return self.generic_visit(node)

            def visit_Name(self, node):
                node.id = self.local_names.get(node.id, node.id)
                if node.id == '_OpenPrefillSource':
                    node.id = '_IncomingCropPreparation'
                return node

            def visit_arg(self, node):
                node.arg = self.local_names.get(node.arg, node.arg)
                return self.generic_visit(node)

        def methods(source):
            return {RENAMES.get(node.name, node.name): ast.dump(AllowedEdits(
                        LOCAL_RENAMES.get(RENAMES.get(node.name, node.name), {})).visit(node))
                    for node in recognizer(source).body if isinstance(node, ast.FunctionDef)}

        self.assertEqual(methods(self.before), methods(self.after))
        # Do not allow extra class-level state or changed inheritance either.
        headers = []
        for source in (self.before, self.after):
            node = recognizer(source)
            if ast.get_docstring(node) is not None:
                node.body.pop(0)
            node.body = [child for child in node.body if not isinstance(child, ast.FunctionDef)]
            headers.append(ast.dump(node))
        self.assertEqual(*headers)

    def test_other_runtime_code_is_unchanged(self):
        def outside_class(source):
            lines = source.splitlines(keepends=True)
            classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                       and node.name in ('ContinuousRecognizer', '_OpenPrefillSource',
                                         '_IncomingCropPreparation', 'ContinuousDecodeScheduler')]
            for node in reversed(classes):
                del lines[node.lineno - 1:node.end_lineno]
            return ''.join(lines)

        # One external documentation reference follows the renamed method.
        expected = outside_class(self.before).replace(
            'ContinuousRecognizer._result_from_completion',
            'ContinuousRecognizer._build_recognition_result')
        self.assertEqual(expected, outside_class(self.after))

    def test_incoming_crop_preparation_only_renames(self):
        class RenameAttributes(ast.NodeTransformer):
            def visit_Attribute(self, node):
                node.attr = {'pending': 'crops_awaiting_prefill',
                             'executor': 'cpu_preparation_worker'}.get(node.attr, node.attr)
                return self.generic_visit(node)

        def signature(source, name):
            node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.ClassDef) and node.name == name)
            node.name = '_IncomingCropPreparation'
            self.assertIsNotNone(ast.get_docstring(node))
            node.body.pop(0)
            return ast.dump(RenameAttributes().visit(node))

        self.assertEqual(signature(self.before, '_OpenPrefillSource'),
                         signature(self.after, '_IncomingCropPreparation'))

    def test_reading_order(self):
        self.assertEqual(
            [node.name for node in recognizer(self.after).body if isinstance(node, ast.FunctionDef)],
            ['__init__', 'serve', '_run_continuous_decoding_loop_until_pipeline_shutdown',
             '_prepare_cpu', '_prefill_for_decode',
             '_copy_inputs_to_npu', '_submit_vision_and_text_prefill',
             '_wait_for_prefill_result', '_build_recognition_result',
             '_setup_stage', '_prepare_decode_lm_head', 'configuration'])

    def test_scheduler_only_approved_identifier_renames(self):
        renames = {
            'pending': 'previous_token_copy', 'current': 'new_token_copy',
            '_schedule_token_copy': '_start_copying_tokens_to_cpu',
            '_wait_tokens': '_wait_for_copied_tokens',
            'retire_pending': 'process_copied_tokens_and_refill_slots',
            'record_completion': 'record_and_report_finished_crop',
        }

        class RenameIdentifiers(ast.NodeTransformer):
            def visit_Name(self, node):
                node.id = renames.get(node.id, node.id)
                return node

            def visit_arg(self, node):
                node.arg = renames.get(node.arg, node.arg)
                return self.generic_visit(node)

            def visit_Attribute(self, node):
                node.attr = renames.get(node.attr, node.attr)
                return self.generic_visit(node)

            def visit_FunctionDef(self, node):
                node.name = renames.get(node.name, node.name)
                return self.generic_visit(node)

        def signature(source):
            node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.ClassDef) and node.name == 'ContinuousDecodeScheduler')
            return ast.dump(RenameIdentifiers().visit(node))

        self.assertEqual(signature(self.before), signature(self.after))

    def test_model_sources_and_serving_entrypoint_are_unchanged(self):
        for path in sorted((ROOT / '19_table_ocr_serving').glob('p0*.py')):
            if path.name == 'p02_serving_runtime.py':
                continue
            with self.subTest(file=path.name):
                previous = subprocess.check_output(
                    ['git', '-C', str(ROOT), 'show', f'{BASELINE}:{path.relative_to(ROOT)}'])
                self.assertEqual(previous, path.read_bytes())


class RuntimeIntegrationStructureTests(unittest.TestCase):
    def test_removed_owners_adapters_and_intermediate_records(self):
        tree=ast.parse((ROOT/PATH).read_text())
        names={n.name for n in tree.body if isinstance(n,ast.ClassDef)}
        self.assertTrue({'ContinuousRecognizer','DecodeArena','PreparedCrop','DecodeRequest','ServingSummary'} <= names)
        self.assertFalse(names & {'ContinuousDecodeScheduler','_IncomingCropPreparation',
            'OpenReadyDecodeSource','_IterableReadyDecodeSource','StagedCrop','InFlightCrop',
            'PrefilledCrop','ReadyDecodeRequest','ContinuousDecodeRun','ContinuousDecodeResult'})
        methods={n.name for n in recognizer((ROOT/PATH).read_text()).body if isinstance(n,ast.FunctionDef)}
        self.assertFalse(methods & {'run_stream','run','_copy_inputs_to_npu',
            '_submit_vision_and_text_prefill','_wait_for_prefill_result'})
        self.assertIn('_run_continuous_decoding_loop_until_pipeline_shutdown',methods)

    def test_decode_arena_and_token_copy_operations_preserved(self):
        before=subprocess.check_output(['git','-C',str(ROOT),'show',f'a191a2a2:{PATH}'],text=True)
        after=(ROOT/PATH).read_text()
        def classes(source):
            return {n.name:n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
        class Rename(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='ReadyDecodeRequest': node.id='DecodeRequest'
                return node
            def visit_Attribute(self,node):
                if node.attr=='arena': node.attr='decode_arena'
                return self.generic_visit(node)
        old,new=classes(before),classes(after)
        for name in ('__init__','serve'):
            a=next(n for n in old['ContinuousRecognizer'].body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new['ContinuousRecognizer'].body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual([ast.dump(d) for d in a.decorator_list],
                             [ast.dump(d) for d in b.decorator_list],name)
        self.assertEqual(ast.dump(Rename().visit(old['DecodeArena'])),ast.dump(new['DecodeArena']))
        for method in old['ContinuousDecodeScheduler'].body:
            if not isinstance(method,ast.FunctionDef) or method.name in ('__init__','run_stream','run'):
                continue
            moved=next(n for n in new['ContinuousRecognizer'].body
                if isinstance(n,ast.FunctionDef) and n.name==method.name)
            self.assertEqual(ast.dump(Rename().visit(method)),ast.dump(moved),method.name)


if __name__ == '__main__':
    unittest.main()
