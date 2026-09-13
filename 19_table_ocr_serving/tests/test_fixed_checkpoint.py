"""Fixed-checkpoint and preprocessing parity, without an NPU or compilation."""
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
import torch

from fixed_architecture_reference import PaddleOCRVLConfig

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / '19_table_ocr_serving'
sys.path.insert(0, str(PRODUCT))
sys.modules.setdefault('torch_npu', types.ModuleType('torch_npu'))
import crop_processing as crops
import paddle_ocr_vl_1_6_modeling as modeling


def historical(name):
    path = f'19_table_ocr_serving/{name}.py'
    source = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'1c2b891d:{path}'], text=True)
    module = types.ModuleType('fixed_reference_' + name)
    module.__file__ = str(ROOT / path)
    sys.modules[module.__name__] = module
    helpers = types.ModuleType('_support.model.compile_utils')
    helper_source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '564da03f:19_table_ocr_serving/_support/model/compile_utils.py'], text=True)
    exec(compile(helper_source, 'historical_compile_helpers', 'exec'), helpers.__dict__)
    with patch.dict(sys.modules, {helpers.__name__: helpers}):
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


class FixedCheckpointTests(unittest.TestCase):
    def test_resize_dimensions_preserve_rounding_and_rejections(self):
        old = historical('crop_processing')
        import random
        rng = random.Random(19)
        boundaries = (1, 13, 14, 27, 28, 29, 41, 42, 43, 167, 168, 169,
                      383, 384, 385, 895, 896, 897, 5600, 5601)
        sizes = [(h, w) for h in boundaries for w in boundaries]
        sizes += [(rng.randint(1,10000),rng.randint(1,10000)) for _ in range(10000)]
        for height, width in sizes:
            try:
                expected = old.smart_resize(height, width, factor=28,
                                            min_pixels=28224, max_pixels=802816)
            except ValueError as error:
                with self.assertRaises(ValueError) as actual:
                    crops.calculate_resized_image_shape(height, width)
                self.assertEqual(str(error), str(actual.exception))
            else:
                self.assertEqual(expected, crops.calculate_resized_image_shape(height, width),
                                 (height, width))
        for original, resized in [((500,800),(504,812)),
                                  ((100,100),(168,168)),
                                  ((24,417),(56,700)),
                                  ((417,24),(700,56)),
                                  ((1000,1000),(896,896))]:
            self.assertEqual(crops.calculate_resized_image_shape(*original), resized)

    def test_narrow_image_rounding_matches_reference(self):
        old = historical('crop_processing')
        # All short-side sizes and both orientations. This catches half-pixel
        # ties missed by the broader random sweep (notably 24 x 417).
        for short_side in range(1,28):
            for other_side in range(1,5601):
                for height, width in ((short_side,other_side),(other_side,short_side)):
                    try:
                        expected = old.smart_resize(height,width,factor=28,
                                                    min_pixels=28224,max_pixels=802816)
                    except ValueError as error:
                        with self.assertRaises(ValueError) as actual:
                            crops.calculate_resized_image_shape(height,width)
                        self.assertEqual(str(error),str(actual.exception))
                    else:
                        self.assertEqual(crops.calculate_resized_image_shape(height,width),
                                         expected,(height,width))

    def test_prompt_tokens_use_fixed_merge_size(self):
        old = historical('crop_processing')
        for count in (0, 1, 4, 1024):
            prompt = 'Table Recognition:'
            expected = (crops.BOS + 'User: ' + crops.IMAGE_START
                        + crops.IMAGE_TOKEN * count + crops.IMAGE_END
                        + prompt + '\nAssistant:\n')
            self.assertEqual(crops.build_paddleocr_vl_prompt(prompt, image_token_count=count), expected)
            self.assertEqual(old.build_paddleocr_vl_prompt(prompt, image_token_count=count), expected)
        encoded = []
        def encode(text):
            encoded.append(text)
            return types.SimpleNamespace(ids=list(text.encode('utf-8')))
        tokenizer = types.SimpleNamespace(encode=encode)
        for height, width in ((2, 2), (12, 12), (22, 36), (64, 64)):
            grid = torch.tensor([[1, height, width]])
            for prompt in ('Table Recognition:', 'OCR:', 'Formula Recognition:'):
                before = old.build_inputs(tokenizer, grid, prompt, merge_size=2)
                after = crops.prepare_prompt_tokens(tokenizer, grid, prompt)
                self.assertEqual(encoded[-2], encoded[-1])
                self.assertEqual(encoded[-1].count(crops.IMAGE_TOKEN), height * width // 4)
                for x, y in zip(before, after):
                    self.assertEqual(x.dtype, y.dtype)
                    self.assertTrue(torch.equal(x, y))

    def test_full_architecture_shapes_and_buffers(self):
        # These are the verified checkpoint values, not dimensions inferred
        # from the benchmark. In particular head_dim is 128, NOT 1024 / 16.
        cfg = PaddleOCRVLConfig.from_dict({
            'text_config': {'vocab_size': 103424, 'hidden_size': 1024,
                'intermediate_size': 3072, 'num_hidden_layers': 18,
                'num_attention_heads': 16, 'num_key_value_heads': 2,
                'head_dim': 128, 'rms_norm_eps': 1e-5, 'use_bias': False,
                'rope_parameters': {'rope_theta': 500000.,
                                    'mrope_section': [16, 24, 24]}},
            'vision_config': {'hidden_size': 1152, 'intermediate_size': 4304,
                'num_hidden_layers': 27, 'num_attention_heads': 16,
                'image_size': 384, 'patch_size': 14, 'num_channels': 3,
                'spatial_merge_size': 2, 'temporal_patch_size': 2,
                'layer_norm_eps': 1e-6}})
        old_text = historical('text_prefill_and_decode')
        old_vision = historical('vision_prefill')
        with patch.dict(sys.modules, {'text_prefill_and_decode': old_text,
                                      'vision_prefill': old_vision}):
            old_model = historical('paddle_ocr_vl_1_6_modeling')
        # Meta allocates no model storage and runs no inference.
        with torch.device('meta'):
            before = old_model.LocalPaddleOCRVLForConditionalGeneration(cfg)
            after = modeling.LocalPaddleOCRVLForConditionalGeneration()
        def shapes(model):
            return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
        self.assertEqual(shapes(before), shapes(after))
        self.assertEqual(len(before.model.layers), 18)
        self.assertEqual(len(after.visual.vision_model.encoder.layers), 27)
        self.assertFalse(hasattr(after, 'config'))
        # Rotary buffers are non-persistent; check their real CPU values too.
        import text_prefill_and_decode as text
        import vision_prefill as vision
        self.assertTrue(torch.equal(old_text.PaddleOCRRotaryEmbedding(cfg.text_config).inv_freq,
                                    text.PaddleOCRRotaryEmbedding().inv_freq))
        self.assertEqual(cfg.text_config.rope_parameters['mrope_section'], list(text.TEXT_MROPE_SECTION))
        self.assertEqual(vision.VISION_NORM_EPS, cfg.vision_config.layer_norm_eps)
        self.assertEqual(text.TEXT_RMS_EPS, cfg.text_config.rms_norm_eps)

    def test_weight_loader_needs_no_config_files(self):
        with tempfile.TemporaryDirectory() as directory, torch.device('meta'):
            expected = modeling.LocalPaddleOCRVLForConditionalGeneration().state_dict()
            fake = types.ModuleType('safetensors.torch')
            fake.load_file = load = Mock(return_value=expected)
            with patch.dict(sys.modules, {'safetensors.torch': fake}):
                result = modeling.LocalPaddleOCRVLForConditionalGeneration.from_pretrained(directory)
            load.assert_called_once_with(Path(directory) / 'model.safetensors', device='cpu')
            self.assertFalse(result.training)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_resize_and_uint8_patch_parity(self):
        old = historical('crop_processing')
        # Exact previously effective serving values. Model preprocessor defaults
        # alone were different; serving overrode both pixel limits.
        cfg = dict(do_convert_rgb=True, do_resize=True, temporal_patch_size=1,
            patch_size=14, merge_size=2, min_pixels=28224, max_pixels=802816,
            do_rescale=True, rescale_factor=1/255, do_normalize=True,
            image_mean=[.5]*3, image_std=[.5]*3)
        for height in (1, 13, 28, 167, 168, 169, 383, 896, 897, 2048):
            for width in (1, 14, 29, 168, 384, 896, 2048):
                with self.subTest(size=(width, height)):
                    try:
                        expected = old.image_grid_thw_from_size(width, height,
                            patch_size=14, merge_size=2, temporal_patch_size=1,
                            min_pixels=28224, max_pixels=802816, do_resize=True)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            crops.image_grid_hw_from_size(width, height)
                    else:
                        self.assertEqual(expected[0], 1)
                        self.assertEqual(expected[1:], crops.image_grid_hw_from_size(width, height))
        calls = []
        class FakeKorniaImage:
            @classmethod
            def fromarray(cls, array):
                obj = cls()
                obj.array = array
                return obj
            def resize(self, width, height, interpolation):
                calls.append((width, height, interpolation, self.array.shape, self.array.dtype))
                # Both paths invoke the identical resize operator. A deterministic
                # stand-in isolates RGB conversion and patch layout from its kernel.
                return types.SimpleNamespace(data=np.asarray(Image.fromarray(self.array).resize((width, height))))
        fake = types.ModuleType('kornia_rs.image')
        fake.Image = FakeKorniaImage
        with patch.dict(sys.modules, {'kornia_rs.image': fake}), \
             patch.object(crops, 'KorniaImage', FakeKorniaImage):
            for mode in ('RGB', 'RGBA', 'L'):
                for size in ((168,168), (503,301), (1300,900)):
                    image = Image.fromarray(np.random.default_rng(4).integers(
                        0,256,(size[1],size[0],3),dtype=np.uint8)).convert(mode)
                    before = old.preprocess_pil_image(image, cfg)
                    after = crops.preprocess_pil_image(image)
                    self.assertEqual(calls[-2], calls[-1])
                    for x, y in zip(before, after):
                        self.assertEqual(x.dtype, y.dtype)
                        self.assertEqual(x.stride(), y.stride())
                        self.assertTrue(torch.equal(x, y))


if __name__ == '__main__':
    unittest.main()
