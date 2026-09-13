"""Dataclass relocation preserves image decoding and dictionary serialization."""
from __future__ import annotations

import ast
from dataclasses import asdict, fields, MISSING
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

from PIL import Image
from test_text_simplification import ROOT, EXPERIMENT
import p02_serving_runtime as current


class RuntimeRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = subprocess.check_output(['git','-C',str(ROOT),'show',
            '564da03f:19_table_ocr_serving/_support/serving/types.py'],text=True)
        tree = ast.parse(source)
        # Repetition metadata was removed in the previous, separate cleanup.
        result = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RecognitionResult')
        result.body = [n for n in result.body if not (isinstance(n,ast.AnnAssign) and n.target.id=='repetition')]
        cls.old = types.ModuleType('historical_serving_records')
        sys.modules[cls.old.__name__] = cls.old
        exec(compile(tree,'historical_serving_records','exec'),cls.old.__dict__)
        cls.old_tree = tree

    def test_definitions_unchanged_by_relocation(self):
        tree = ast.parse(Path(current.__file__).read_text())
        for old in self.old_tree.body:
            if isinstance(old,ast.ClassDef):
                name = 'ServingSummary' if old.name == 'ContinuousDecodeResult' else old.name
                moved = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
                moved.name = old.name
                self.assertEqual(ast.dump(old),ast.dump(moved),old.name)
        self.assertFalse((EXPERIMENT/'_support/serving/types.py').exists())

    def test_request_image_decode_unchanged(self):
        image = Image.new('RGBA',(3,2),(10,20,30,40))
        encoded = io.BytesIO(); image.save(encoded,format='PNG')
        for crop in (encoded.getvalue(),image):
            before = self.old.RecognitionRequest('crop',crop,'Table Recognition:',submitted_at=1.)
            after = current.RecognitionRequest('crop',crop,'Table Recognition:',submitted_at=1.)
            a,b = before.resolve_image(),after.resolve_image()
            self.assertEqual(a.crop.tobytes(),b.crop.tobytes())
            self.assertEqual(a.crop.mode,b.crop.mode)
            self.assertEqual(a.source_crop_size,b.source_crop_size)
            self.assertEqual(a.skip_special_tokens,b.skip_special_tokens)
            self.assertEqual(a.submitted_at,b.submitted_at)
            if crop is image:
                self.assertIs(a,before); self.assertIs(b,after)

    def test_result_and_summary_serialization_unchanged(self):
        def sample(module,name):
            cls = getattr(module,'ServingSummary' if module is current and name=='ContinuousDecodeResult' else name)
            kwargs = {f.name:0 for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
            if name=='RecognitionResult':
                kwargs.update(request_id='crop',text='<fcel>汉字<nl>',token_ids=[10,2],stop_reason='eos',
                    timing_s=module.RequestTiming(**{f.name:.25 for f in fields(module.RequestTiming)}),
                    device_stage_s=module.PrefillDeviceTiming(**{f.name:.5 for f in fields(module.PrefillDeviceTiming)}))
            return json.dumps(asdict(cls(**kwargs)),sort_keys=True)
        for name in ('RecognitionResult','ContinuousDecodeResult'):
            self.assertEqual(sample(self.old,name),sample(current,name))

    def test_http_import_does_not_load_runtime_or_torch(self):
        code = "import sys; sys.path.insert(0,sys.argv[1]); import p01_serve as serve; assert 'p02_serving_runtime' not in sys.modules; assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
        subprocess.run([sys.executable,'-c',code,str(EXPERIMENT)],check=True)


if __name__ == '__main__':
    unittest.main()
