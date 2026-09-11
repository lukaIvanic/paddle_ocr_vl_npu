"""Test-only historical fixtures and small-dimension arithmetic controls.

Production has no configurable architecture. Tests patch named constants only
while constructing tiny models/caches; full-size architecture is checked on meta.
"""
import inspect
import ast
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
    '1c2b891d:19_table_ocr_serving/_support/model/config.py'], text=True)
legacy = types.ModuleType('_support.model.config')
sys.modules[legacy.__name__] = legacy
exec(compile(source, '<historical-test-config>', 'exec'), legacy.__dict__)
PaddleOCRTextConfig = legacy.PaddleOCRTextConfig
PaddleOCRVisionConfig = legacy.PaddleOCRVisionConfig
PaddleOCRVLConfig = legacy.PaddleOCRVLConfig


def text_values(cfg):
    return dict(TEXT_HIDDEN_SIZE=cfg.hidden_size, TEXT_INTERMEDIATE_SIZE=cfg.intermediate_size,
        TEXT_LAYERS=cfg.num_hidden_layers, TEXT_HEADS=cfg.num_attention_heads,
        TEXT_KV_HEADS=cfg.num_key_value_heads, TEXT_HEAD_DIM=cfg.head_dim,
        TEXT_VOCAB_SIZE=cfg.vocab_size, TEXT_RMS_EPS=cfg.rms_norm_eps,
        TEXT_PAD_TOKEN_ID=cfg.pad_token_id, TEXT_EOS_TOKEN_ID=cfg.eos_token_id,
        TEXT_ROPE_THETA=(cfg.rope_parameters or {}).get('rope_theta',500000.),
        TEXT_MROPE_SECTION=tuple((cfg.rope_parameters or {}).get('mrope_section',[16,24,24])))


def text_call(module, cfg, fn, *args, **kwargs):
    values = {k:v for k,v in text_values(cfg).items() if hasattr(module,k)}
    with patch.dict(module.__dict__, values):
        return fn(*args, **kwargs)


def text_model(module, cfg):
    args = (cfg,) if 'config' in inspect.signature(module.PaddleOCRTextModel).parameters else ()
    return text_call(module,cfg,module.PaddleOCRTextModel,*args)


def cache(module, cfg, **kwargs):
    fn=module.LocalPaddleOCRVLStaticCache.allocate
    args=(cfg,) if 'config' in inspect.signature(fn).parameters else ()
    return text_call(module,cfg,fn,*args,**kwargs)


def vision_model(module,cfg):
    values=dict(VISION_HIDDEN_SIZE=cfg.hidden_size,VISION_INTERMEDIATE_SIZE=cfg.intermediate_size,
        VISION_LAYERS=cfg.num_hidden_layers,VISION_HEADS=cfg.num_attention_heads,
        VISION_CHANNELS=cfg.num_channels,VISION_IMAGE_SIZE=cfg.image_size,
        VISION_PATCH_SIZE=cfg.patch_size,VISION_MERGE_SIZE=cfg.spatial_merge_size,
        VISION_NORM_EPS=cfg.layer_norm_eps)
    args=(cfg,) if 'config' in inspect.signature(module.PaddleOCRVisionModel).parameters else ()
    with patch.dict(module.__dict__,{k:v for k,v in values.items() if hasattr(module,k)}):
        return module.PaddleOCRVisionModel(*args)


class FreezeArchitecture(ast.NodeTransformer):
    """Specialize reference config reads only; preserve all tensor operations."""
    text = dict(hidden_size='TEXT_HIDDEN_SIZE', intermediate_size='TEXT_INTERMEDIATE_SIZE',
        num_hidden_layers='TEXT_LAYERS', num_attention_heads='TEXT_HEADS',
        num_key_value_heads='TEXT_KV_HEADS',head_dim='TEXT_HEAD_DIM',
        vocab_size='TEXT_VOCAB_SIZE',rms_norm_eps='TEXT_RMS_EPS',pad_token_id='TEXT_PAD_TOKEN_ID')
    vision = dict(hidden_size='VISION_HIDDEN_SIZE',intermediate_size='VISION_INTERMEDIATE_SIZE',
        num_hidden_layers='VISION_LAYERS',num_attention_heads='VISION_HEADS',
        num_channels='VISION_CHANNELS',image_size='VISION_IMAGE_SIZE',patch_size='VISION_PATCH_SIZE',
        spatial_merge_size='VISION_MERGE_SIZE',layer_norm_eps='VISION_NORM_EPS')

    def __init__(self,kind='text'): self.kind=kind

    def visit_Attribute(self,n):
        path=ast.unparse(n)
        special = {'self.config.image_token_id': 'IMAGE_TOKEN_ID',
            'self.config.vision_start_token_id': 'VISION_START_TOKEN_ID',
            'self.config.vision_config.spatial_merge_size': 'VISION_MERGE_SIZE'}
        if path in special:
            return ast.copy_location(ast.Name(id=special[path], ctx=ast.Load()), n)
        for prefix,mapping in (('model.config.text_config.',self.text),('model.config.vision_config.',self.vision),
            ('config.text_config.',self.text),('config.vision_config.',self.vision),
            ('config.',self.text if self.kind=='text' else self.vision)):
            if path.startswith(prefix) and path[len(prefix):] in mapping:
                return ast.copy_location(ast.Name(id=mapping[path[len(prefix):]],ctx=ast.Load()),n)
        if path=='config.use_bias': return ast.copy_location(ast.Constant(False),n)
        return self.generic_visit(n)

    def visit_Assign(self,n):
        if any(ast.unparse(t)=='self.config' for t in n.targets): return None
        if ast.unparse(n.value)=='config.rope_parameters or {}': return None
        return self.generic_visit(n)

    def visit_arguments(self,n):
        n.args=[a for a in n.args if a.arg!='config']
        pairs = [(a, d) for a, d in zip(n.kwonlyargs, n.kw_defaults) if a.arg != 'model_dir']
        n.kwonlyargs = [a for a, _ in pairs]
        n.kw_defaults = [d for _, d in pairs]
        return self.generic_visit(n)

    def visit_Subscript(self,n):
        if ast.unparse(n)=="(config.rope_parameters or {})['mrope_section']":
            return ast.copy_location(ast.Name(id='TEXT_MROPE_SECTION',ctx=ast.Load()),n)
        return self.generic_visit(n)

    def visit_Call(self,n):
        if ast.unparse(n)=="float(rope.get('rope_theta', 500000.0))":
            return ast.copy_location(ast.Name(id='TEXT_ROPE_THETA',ctx=ast.Load()),n)
        n.args=[a for a in n.args if ast.unparse(a) not in ('config', 'config.text_config', 'config.vision_config')]
        n.keywords = [kw for kw in n.keywords if kw.arg != 'model_dir']
        return self.generic_visit(n)

    def visit_Dict(self,n):
        pairs=[(k,v) for k,v in zip(n.keys,n.values) if not isinstance(k,ast.Constant) or k.value!='model_config_hash']
        n.keys=[k for k,v in pairs]; n.values=[v for k,v in pairs]
        return self.generic_visit(n)
