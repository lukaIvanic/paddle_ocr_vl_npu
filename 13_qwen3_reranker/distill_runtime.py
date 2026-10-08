"""Deterministic NPU scoring with an explicit teacher/student input format."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import array

from run_query_first_training_curve import plans, save
from training_smoke_data import body
from transformers_rerank import PREFIX, SUFFIX


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


class Runtime:
    def __init__(self, model_dir, max_length=8192):
        import torch
        import torch_npu
        from transformers import AutoTokenizer
        self.torch = torch
        torch.set_num_threads(8)
        torch.manual_seed(1047)
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)
        self.device = torch.device('npu:0')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
        self.encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        self.prefix, self.suffix = self.encode(PREFIX), self.encode(SUFFIX)
        self.limit = max_length - len(self.prefix) - len(self.suffix)
        no, yes = [self.tokenizer.convert_tokens_to_ids(x) for x in ('no', 'yes')]
        assert self.encode('no') == [no] and self.encode('yes') == [yes]
        self.answers = torch.tensor([no, yes], device=self.device)
        self.lengths = {}

    def records(self, groups, section, order='query_first'):
        rows = []
        truncated = 0
        token_hash = hashlib.sha256()
        # Batch CPU tokenization without changing pair order, padding or truncation.
        # The resulting token stream is still checked against saved teacher hashes.
        for start in range(0, len(groups), 32):
            pairs = [(g, j, document) for g in groups[start:start + 32]
                     for j, document in enumerate(g['documents'])]
            encoded = self.tokenizer(
                [body(g['instruction'], g['query'], document, order) for g, _, document in pairs],
                add_special_tokens=False, padding=False, truncation=False)['input_ids']
            for (g, j, _), raw in zip(pairs, encoded):
                truncated += len(raw) > self.limit
                rows.append({'ids': self.prefix + raw[:self.limit] + self.suffix,
                             'index': len(rows), 'group_id': g['id'], 'candidate': j})
                token_hash.update(array.array('I', [len(rows[-1]['ids'])] + rows[-1]['ids']).tobytes())
        self.lengths[section] = {'pairs': len(rows), 'truncated': truncated,
                                'tokens': sum(len(r['ids']) for r in rows),
                                'token_ids_sha256': token_hash.hexdigest()}
        return rows

    def load(self, model_dir, attention='fusion_attention'):
        from safetensors.torch import load_file
        from local_modeling_qwen3_reranker import LocalQwen3RerankerConfig, LocalQwen3RerankerForCausalLM
        config = LocalQwen3RerankerConfig.from_model_dir(Path(model_dir))
        model = LocalQwen3RerankerForCausalLM(config, attention_impl=attention)
        state = {}
        for p in sorted(Path(model_dir).glob('*.safetensors')):
            state.update({k.removeprefix('model.'): v for k, v in load_file(str(p)).items()})
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and (not missing or (config.tie_word_embeddings and missing == ['lm_head.weight']))
        del state
        model.to(device=self.device, dtype=self.torch.float32)
        if config.tie_word_embeddings:
            assert model.lm_head.weight is model.embed_tokens.weight
        assert not any(isinstance(m, self.torch.nn.Dropout) and m.p for m in model.modules())
        return model

    def logits(self, model, rows, hf=False, padded_length=None):
        from local_modeling_qwen3_reranker import build_left_padded_causal_bool_mask, build_left_padded_causal_mask
        import torch.nn.functional as F
        minimum = math.ceil(max(len(r['ids']) for r in rows) / 128) * 128
        length = minimum if padded_length is None else padded_length
        assert length >= minimum and length % 128 == 0
        x = self.tokenizer.pad({'input_ids': [r['ids'] for r in rows]}, padding='max_length',
                               max_length=length, return_tensors='pt')
        x = {k: v.to(self.device) for k, v in x.items()}
        pos = (x['attention_mask'].cumsum(-1) - 1).clamp_min(0)
        with self.torch.autocast('npu', dtype=self.torch.bfloat16):
            if hf:
                z = model(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, self.answers]
            else:
                mask = (build_left_padded_causal_mask(x['attention_mask'], model.embed_tokens.weight.dtype)
                        if model.attention_impl == 'eager'
                        else build_left_padded_causal_bool_mask(x['attention_mask']))
                h = model.forward_hidden_states_prepared(x['input_ids'], pos, mask)
                z = F.linear(h[:, -1], model.lm_head.weight[self.answers])
        return z.float()[:, 1] - z.float()[:, 0]

    def score(self, model, rows, section, batch_size=16, token_budget=16384, batch_plan=None):
        model.eval()
        values = collections.defaultdict(dict)
        started = time.monotonic()
        done = 0
        with self.torch.no_grad():
            schedule = plans(rows, batch_size, token_budget) if batch_plan is None else batch_plan
            for i, micro in enumerate(schedule, 1):
                scores = self.logits(model, micro, padded_length=micro[0].get('paired_padding_length')).cpu().tolist()
                assert all(math.isfinite(s) for s in scores)
                for r, s in zip(micro, scores):
                    values[r['group_id']][r['candidate']] = s
                done += len(micro)
                if i % 80 == 0:
                    print('SCORE_PROGRESS', json.dumps({'section': section, 'pairs': done,
                          'total': len(rows), 'seconds': time.monotonic() - started}), flush=True)
        return {g: [v[i] for i in range(len(v))] for g, v in values.items()}, time.monotonic() - started


def model_manifest(model_dir):
    return {p.name: digest(p) for p in sorted(Path(model_dir).glob('*.safetensors'))}
