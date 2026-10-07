"""Opt-in full-backbone Ascend matrix recurrence adapter; preserves EOS states."""
import torch
from torch.nn import functional as F
from wkv7_matrix.rwkv7_chunk_scan import rwkv7_chunk_scan

class MatrixRecurrence(torch.nn.Module):
    def __init__(self, chunk_size=64, compute_dtype=torch.float32):
        super().__init__()
        self.chunk_size, self.compute_dtype = chunk_size, compute_dtype
        self.capture_diagnostics = False
        self.diagnostic_tensors = []

    def forward(self, k, v, w, r, kk, a, state, valid_lengths=None):
        B, T, C = k.shape
        padded = self.chunk_size * (1 << ((T - 1) // self.chunk_size).bit_length())
        lengths = (torch.full((B,), T, device=k.device, dtype=torch.int32)
                   if valid_lengths is None else valid_lengths)
        # Log-decay 0 and zero rank-one vectors make extra positions identity
        # state transitions. r=0 also matches the endpoint kernel's zero output.
        active = torch.arange(padded, device=k.device)[None, :, None] < lengths[:, None, None]
        vectors = [torch.where(active, F.pad(x, (0, 0, 0, padded-T)), 0.)
                   for x in (k, v, w, r, kk, a)]
        k, v, w, r, kk, a = [x.reshape(B, padded, C//64, 64) for x in vectors]
        out, final = rwkv7_chunk_scan(state, w, k, v, kk, a, r,
            chunk_size=self.chunk_size, compute_dtype=self.compute_dtype,
            w_is_log_decay=True, dense_chunk_prefix=True,
            dense_prefix_algorithm='tree_root',
            diagnostic_tensors=self.diagnostic_tensors if self.capture_diagnostics else None)
        return out[:, :T].reshape(B, T, C//64, 64).permute(0, 2, 1, 3).contiguous(), final
