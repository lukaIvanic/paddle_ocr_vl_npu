"""CPU-resident image_grid_thw must reproduce the original vision positions exactly."""
import unittest

import torch
import torch.nn.functional as F

from local_modeling_mineru import MinerUVisionRotaryEmbedding, MinerUVisionTransformer
from native_custom_backend import make_local_fixed_batch_vlm_client

MERGE = 2
HEAD_DIM = 80  # MinerU vision: embed_dim 1280 / 16 heads
MAX_TOKENS = 3072  # 602112-pixel recognition cap


def original_rot_pos_emb(rotary, grid_thw):
    """The pre-change implementation, kept verbatim as the reference."""
    pos_ids = []
    for t, h, w in grid_thw:
        t_int, h_int, w_int = int(t.item()), int(h.item()), int(w.item())
        hpos_ids = torch.arange(h_int, device=grid_thw.device).unsqueeze(1).expand(-1, w_int)
        hpos_ids = hpos_ids.reshape(h_int // MERGE, MERGE, w_int // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
        wpos_ids = torch.arange(w_int, device=grid_thw.device).unsqueeze(0).expand(h_int, -1)
        wpos_ids = wpos_ids.reshape(h_int // MERGE, MERGE, w_int // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t_int, 1))
    pos_ids_tensor = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()
    return rotary(max_grid_size)[pos_ids_tensor].flatten(1)


def cu_seqlens(grid_thw):
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def vision():
    model = MinerUVisionTransformer.__new__(MinerUVisionTransformer)
    torch.nn.Module.__init__(model)
    model.spatial_merge_size = MERGE
    model.rotary_pos_emb = MinerUVisionRotaryEmbedding(HEAD_DIM // 2)
    return model


class VisionGridDeviceTests(unittest.TestCase):
    def test_every_capped_single_crop_grid_matches_original(self):
        model = vision()
        grids = [(h, w) for h in range(MERGE, MAX_TOKENS + 1, MERGE)
                 for w in range(MERGE, MAX_TOKENS // h + 1, MERGE) if h * w <= MAX_TOKENS]
        self.assertGreater(len(grids), 1000)
        for h, w in grids:
            grid = torch.tensor([[1, h, w]], dtype=torch.int64)
            new = model.rot_pos_emb(grid)
            self.assertTrue(torch.equal(new, original_rot_pos_emb(model.rotary_pos_emb, grid)), (h, w))
            self.assertEqual(new.shape, (h * w, HEAD_DIM // 2))
            self.assertEqual(cu_seqlens(grid).tolist(), [0, h * w])

    def test_multi_image_grid_matches_original(self):
        model = vision()
        generator = torch.Generator().manual_seed(0)
        for _ in range(200):
            count = int(torch.randint(2, 6, (1,), generator=generator))
            grid = torch.stack([torch.ones(count, dtype=torch.int64),
                                torch.randint(1, 40, (count,), generator=generator) * MERGE,
                                torch.randint(1, 40, (count,), generator=generator) * MERGE], dim=1)
            self.assertTrue(torch.equal(model.rot_pos_emb(grid), original_rot_pos_emb(model.rotary_pos_emb, grid)))

    def test_invalid_grid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported vision grid device"):
            make_local_fixed_batch_vlm_client(None, None, None, batch_size=1, vision_grid_device="gpu",
                                              system_prompt="", allow_truncated_content=False)


if __name__ == "__main__":
    unittest.main()
