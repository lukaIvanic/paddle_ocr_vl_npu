"""Separate state-reading RWKV-7 reranker, following upstream package/model.py."""
import torch
from torch import nn
from local_modeling_rwkv_embedding import Block, Weights
from run_cpu_reference import sha256

CHECKPOINT_SHA256 = '6b6f36543ad71aa48bca3cca97863c529e9153515fb777184c36475b3badb6be'

class Reranker(Weights):
    def __init__(self, checkpoint, device, dense_dtype, expected_sha256=CHECKPOINT_SHA256):
        if sha256(checkpoint) != expected_sha256:
            raise ValueError('Reranker checkpoint SHA256 mismatch')
        source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        values = {k.removeprefix('reranker.'): v for k, v in source.items()
                  if k.startswith('reranker.')}
        if 'token.weight' in values:
            values['emb.weight'] = values.pop('token.weight')
        indices = {int(k.split('.')[1]) for k in values if k.startswith('blocks.')}
        depth = max(indices) + 1
        self.depth = depth
        self.heads, head_size = values['blocks.0.att.r_k'].shape
        self.width = self.heads * head_size
        if indices != set(range(depth)) or head_size != 64 or values['emb.weight'].numel() != self.width:
            raise ValueError('Reranker layer indices or head dimensions mismatch')
        self.layer_indices = tuple(int(i) for i in source.get('reranker_layer_idx', range(depth)))
        if len(self.layer_indices) != depth:
            raise ValueError('Layer selection does not match reranker depth')
        root = {k:v for k,v in values.items() if not k.startswith('blocks.')}
        root.update({k:v for k,v in values.items() if k.startswith('blocks.0.ln0.')})
        expected = {'emb.weight','ln_out.weight','ln_out.bias','head.0.weight','head.0.bias',
                    'head.2.weight','blocks.0.ln0.weight','blocks.0.ln0.bias'}
        if set(root) != expected:
            raise ValueError(f'Reranker root keys mismatch: missing={expected-set(root)}, unexpected={set(root)-expected}')
        super().__init__(root, device, dense_dtype)
        self.blocks = nn.ModuleList([Block({k.removeprefix(f'blocks.{i}.'):v
            for k,v in values.items() if k.startswith(f'blocks.{i}.')}, device, dense_dtype)
            for i in range(depth)])
        self.eval()

    def run(self, states, trace=False):
        # One learned readout token, with a separate weight set from the embedder.
        x = self.get('emb.weight')[None].expand(states.shape[1], 1, -1)
        x = self.norm(x, 'blocks.0.ln0')
        first, layers = None, []
        for block, index in zip(self.blocks, self.layer_indices):
            x, first, _ = block.run(x, first, matrix=states[index])
            if trace:
                layers.append(x)
        x = self.norm(x, 'ln_out')
        y = self.linear(x, 'head.0.weight') + self.get('head.0.bias')
        logits = self.linear(torch.tanh(y), 'head.2.weight').reshape(-1)
        return logits, layers

    def forward(self, states):
        return self.run(states)[0]
