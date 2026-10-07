"""Explicit, complete vision replay configuration; stdlib only."""
import hashlib
import json

DEFAULT_CONFIG = dict(attention_impl='prompt_flash_attention', projection_impl='linear',
    layer_norm_impl='manual_fp32', promptfa_pad_head_dim_to=0, approximate_precision=False,
    allow_internal_format=True, buckets=[384,768,3072], min_pixels=25088, max_pixels=602112)


def load_config(path=None, inherited=None):
    cfg = json.loads(path.read_text()) if path else dict(inherited or DEFAULT_CONFIG)
    if set(cfg) != set(DEFAULT_CONFIG):
        raise ValueError(f'configuration must specify exactly {sorted(DEFAULT_CONFIG)}; got {sorted(cfg)}')
    if cfg['attention_impl'] not in ['manual','prompt_flash_attention']:
        raise ValueError('unsupported attention implementation; do not substitute for production silently')
    if cfg['projection_impl'] not in ['linear','grouped_qkv','grouped_qkv_mlp_fc1']:
        raise ValueError('unsupported projection mode')
    if cfg['layer_norm_impl'] not in ['manual_fp32','module']:
        raise ValueError('unsupported LayerNorm mode')
    for key in ['approximate_precision','allow_internal_format']:
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be a JSON boolean')
    if not cfg['buckets'] or any(type(v) is not int or v <= 0 for v in cfg['buckets']):
        raise ValueError('buckets must be positive integers')
    if cfg['min_pixels'] <= 0 or cfg['max_pixels'] < cfg['min_pixels']:
        raise ValueError('invalid processor pixel limits')
    if cfg['promptfa_pad_head_dim_to'] not in [0,80,96,128]:
        raise ValueError('unsupported head padding size')
    if cfg['attention_impl'] != 'prompt_flash_attention' and (cfg['approximate_precision'] or cfg['promptfa_pad_head_dim_to']):
        raise ValueError('precision/padding overrides require PromptFA')
    return cfg


def config_key(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]


def runtime_options(cfg):
    return {k:cfg[k] for k in ['attention_impl','projection_impl','layer_norm_impl','promptfa_pad_head_dim_to']}
