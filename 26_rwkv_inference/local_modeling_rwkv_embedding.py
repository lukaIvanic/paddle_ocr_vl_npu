"""Owned RWKV-7 embedder and state backbone; no training or Transformers.

State orientation: [value,key]. Token-shift state: [layers,2,batch,width].
Math/provenance follows the pinned C/CUDA references in PLAN.md.
"""
import torch
from torch import nn
from torch.nn import functional as F
from run_cpu_reference import CHECKPOINT_SHA256, sha256

class Weights(nn.Module):
    def __init__(self, values, device, dense_dtype):
        super().__init__()
        self.names = {}
        for i, (name, value) in enumerate(values.items()):
            key = f'p{i}'
            # Low-rank matrices use [input,output]; *.weight uses [output,input].
            matrix = value.ndim == 2 and name != 'att.r_k' and 'emb.weight' not in name
            self.register_buffer(key, value.to(device=device,
                dtype=dense_dtype if matrix else torch.float32).contiguous())
            self.names[name] = key
        self.dense_dtype = dense_dtype

    def get(self, name):
        return getattr(self, self.names[name])

    def linear(self, x, name):
        y = F.linear(x.reshape(-1, x.shape[-1]).to(self.dense_dtype), self.get(name)).float()
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def rank(self, x, name):
        y = (x.reshape(-1, x.shape[-1]).to(self.dense_dtype) @ self.get(name)).float()
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def norm(self, x, name):
        return F.layer_norm(x, (self.get(name + '.weight').numel(),), self.get(name + '.weight'), self.get(name + '.bias'), 1e-5)


class Block(Weights):
    def __init__(self, values, device, dense_dtype):
        expected = {'ln1.weight','ln1.bias','ln2.weight','ln2.bias',
                    'att.receptance.weight','att.key.weight','att.value.weight','att.output.weight',
                    'att.ln_x.weight','att.ln_x.bias','att.w0','att.w1','att.w2',
                    'att.a0','att.a1','att.a2','att.g1','att.g2','att.k_k','att.k_a','att.r_k',
                    'ffn.x_k','ffn.key.weight','ffn.value.weight'}
        expected.update('att.x_'+name for name in ['r','w','k','v','a','g'])
        expected.update(['ln0.weight','ln0.bias'] if 'ln0.weight' in values else ['att.v0','att.v1','att.v2'])
        if set(values) != expected:
            raise ValueError(f'Block keys mismatch: missing={expected-set(values)}, unexpected={set(values)-expected}')
        super().__init__(values, device, dense_dtype)
        self.matrix_recurrence = None  # Explicit benchmark opt-in; default unchanged.

    def run(self, x, first, previous=None, matrix=None, valid_lengths=None):
        B, T, C = x.shape
        H = C // 64
        if previous is None:
            previous = torch.zeros((2, B, C), device=x.device, dtype=torch.float32)
        z = self.norm(x, 'ln1')
        delta = torch.cat((previous[0][:, None], z[:, :-1]), dim=1) - z
        att_previous = (z[:, -1] if valid_lengths is None else
            z.gather(1, (valid_lengths.long()-1)[:, None, None].expand(B, 1, C)).squeeze(1))
        def mix(name):
            return z + delta * self.get('att.x_' + name)
        r = self.linear(mix('r'), 'att.receptance.weight')
        w = -.6065306597126334 * torch.sigmoid(self.get('att.w0') +
            self.rank(torch.tanh(self.rank(mix('w'), 'att.w1')), 'att.w2'))
        k = self.linear(mix('k'), 'att.key.weight')
        xv = mix('v')
        v = self.linear(xv, 'att.value.weight')
        a = torch.sigmoid(self.get('att.a0') + self.rank(self.rank(mix('a'), 'att.a1'), 'att.a2'))
        g = self.rank(torch.sigmoid(self.rank(mix('g'), 'att.g1')), 'att.g2')
        kk = F.normalize((k * self.get('att.k_k')).reshape(B, T, H, 64), dim=-1).reshape(B, T, C)
        k = k * (1 + (a - 1) * self.get('att.k_a'))
        if first is None:
            first = v
        else:
            v = v + (first - v) * torch.sigmoid(self.get('att.v0') +
                self.rank(self.rank(xv, 'att.v1'), 'att.v2'))
        def layout(y):
            return y.reshape(B, T, H, 64).permute(0, 2, 1, 3).contiguous()
        state = (torch.zeros((B, H, 64, 64), device=x.device, dtype=torch.float32)
                 if matrix is None else matrix.contiguous())
        inputs = None
        if self.matrix_recurrence is not None:
            y, state = self.matrix_recurrence(k, v, w, r, kk, a, state, valid_lengths)
        elif valid_lengths is not None:
            inputs = [layout(t) for t in (k, v, w, r, -kk, kk * a)]
            if T > 2048:
                raise ValueError('Endpoint state capture supports T<=2048')
            y, state = torch.ops.rwkv_endpoint.wkv7.default(*inputs, state, valid_lengths)
        elif T <= 2048:
            inputs = [layout(t) for t in (k, v, w, r, -kk, kk * a)]
            y, state = torch.ops.rwkv_reference.wkv7.default(*inputs, state)
        else:
            # Preserve the complete prepared sequence; only split the recurrence
            # at the bridge's validated call limit, carrying its FP32 state.
            inputs = [layout(t) for t in (k, v, w, r, -kk, kk * a)]
            chunks = []
            for offset in range(0, T, 2048):
                part, state = torch.ops.rwkv_reference.wkv7.default(
                    *(t[:, :, offset:offset+2048].contiguous() for t in inputs), state)
                chunks.append(part)
            y = torch.cat(chunks, dim=2)
            del chunks, part
        del inputs
        y = y.permute(0, 2, 1, 3).contiguous().reshape(B*T, C)
        y = F.group_norm(y, H, self.get('att.ln_x.weight'), self.get('att.ln_x.bias'), .00064).reshape(B, T, C)
        extra = ((r * k * self.get('att.r_k').reshape(C)).reshape(B, T, H, 64)
                 .sum(-1, keepdim=True) * v.reshape(B, T, H, 64)).reshape(B, T, C)
        x = x + self.linear((y + extra) * g, 'att.output.weight')
        # Dynamo cannot trace DELETE_DEREF for the mix closure variables.
        if not torch.compiler.is_compiling():
            del z, delta, r, w, k, xv, v, a, g, kk, y, extra
        z = self.norm(x, 'ln2')
        delta = torch.cat((previous[1][:, None], z[:, :-1]), dim=1) - z
        ffn_previous = (z[:, -1] if valid_lengths is None else
            z.gather(1, (valid_lengths.long()-1)[:, None, None].expand(B, 1, C)).squeeze(1))
        mixed = z + delta * self.get('ffn.x_k')
        x = x + self.linear(F.relu(self.linear(mixed, 'ffn.key.weight')).square(), 'ffn.value.weight')
        return x, first, (torch.stack((att_previous, ffn_previous)), state)

    def forward(self, x, first):
        x, first, _ = self.run(x, first)
        return x, first


class Embedding(Weights):
    def __init__(self, checkpoint, device, dense_dtype, expected_sha256=CHECKPOINT_SHA256):
        if sha256(checkpoint) != expected_sha256:
            raise ValueError('Checkpoint SHA256 mismatch')
        source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        indices = {int(k.split('.')[2]) for k in source if k.startswith('rwkv.blocks.')}
        self.depth = max(indices) + 1
        self.heads, head_size = source['rwkv.blocks.0.att.r_k'].shape
        self.width = self.heads * head_size
        if indices != set(range(self.depth)) or head_size != 64 or source['rwkv.emb.weight'].shape[1] != self.width:
            raise ValueError('Backbone layer indices or head dimensions mismatch')
        root = {k.removeprefix('rwkv.'): v for k, v in source.items()
                if k.startswith(('rwkv.emb.', 'rwkv.ln_out.', 'rwkv.blocks.0.ln0.'))}
        root.update({k: v for k, v in source.items() if k.startswith('head.retr_head.')})
        super().__init__(root, device, dense_dtype)
        self.blocks = nn.ModuleList([Block({k.removeprefix(f'rwkv.blocks.{i}.'): v
            for k, v in source.items() if k.startswith(f'rwkv.blocks.{i}.')}, device, dense_dtype)
            for i in range(self.depth)])
        self.eval()

    def run(self, ids, mask, trace):
        x = F.embedding(ids, self.get('emb.weight'))
        layers = [x] if trace else []
        x = self.norm(x, 'blocks.0.ln0')
        first = None
        for block in self.blocks:
            x, first = block(x, first)
            if trace:
                layers.append(x)
        x = self.norm(x, 'ln_out')
        if trace:
            layers.append(x)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        y = self.linear(F.relu(self.linear(pooled, 'head.retr_head.fc1.weight')), 'head.retr_head.fc2.weight')
        embedding = F.normalize(self.norm(y + pooled, 'head.retr_head.norm'), dim=-1)
        return embedding, layers

    def encode_states(self, ids, state=None, trace=False, valid_lengths=None):
        """Return final hidden outputs and [token-shift, TimeMix] state; never mutate inputs."""
        x = self.norm(F.embedding(ids, self.get('emb.weight')), 'blocks.0.ln0')
        previous, matrices = state if state is not None else (None, None)
        first, shifts, states, layers = None, [], [], []
        for i, block in enumerate(self.blocks):
            x, first, final = block.run(x, first,
                None if previous is None else previous[i],
                None if matrices is None else matrices[i], valid_lengths)
            shifts.append(final[0]); states.append(final[1])
            if trace:
                layers.append(x)
        x = self.norm(x, 'ln_out')
        return x, (torch.stack(shifts), torch.stack(states)), layers

    def forward(self, ids, mask):
        return self.run(ids, mask, False)[0]
