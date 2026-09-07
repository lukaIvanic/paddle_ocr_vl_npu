import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from crop_replay_source import CropReplayPageSource
from generation_trace import image_fingerprint
from prepare_crop_cap_replay import selected
from saved_paddle_crops import sha256
from test_streaming_pipeline import client, Block


class ReplayTests(unittest.TestCase):
    def test_current_live_preset_replay_command(self):
        from run_crop_cap_replay import replay_command
        reference = json.loads((Path(__file__).parent / "references/live_paddle_full1651_910b/run_summary_shard_00.json").read_text())
        command = replay_command(reference, Path("/tmp/out"), Path("/tmp/replay.json"), "/tmp/dataset.json", 2, 602112)
        self.assertIn("--crop-replay-manifest", command)
        self.assertNotIn("--global-request-stream", command)

    def test_cap_boundary(self):
        row = dict(phase="recognition", prompt_token_ids=[99]*768)
        self.assertFalse(selected(row, 99, 602112))
        row["prompt_token_ids"].append(99)
        self.assertTrue(selected(row, 99, 602112))
        with self.assertRaises(ValueError):
            selected(dict(row, phase="layout"), 99, 602112)

    def test_reuse_and_original_block_index(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (8, 8), "white")
            image.save(root / "crop.png")
            base = dict(type="text", bbox=[0, 0, 1, 1], recognize=True,
                        rerun=False, baseline_raw_text="KEEP")
            target = dict(base, index=1, rerun=True, crop="crop.png", crop_size=[8, 8],
                          crop_sha256=sha256(root / "crop.png"), crop_pixel_sha256=image_fingerprint(image),
                          recognition_image_sha256=image_fingerprint(image))
            manifest = dict(schema="mineru_selective_crop_replay_v1", reference_run="baseline",
                            pages=[dict(image_name="p.png", blocks=[dict(base, index=0), target])])
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            stub = ModuleType("mineru_vl_utils.structs")
            stub.ContentBlock = Block
            c = client()
            c.helper.resize_by_need = lambda x: x
            c.prompts["[default]"] = "OCR:"
            completed = []
            with patch.dict(sys.modules, {"mineru_vl_utils.structs": stub}):
                source = CropReplayPageSource(c, [("p.png", lambda: self.fail("no layout"))],
                    manifest_path=path, on_page=lambda name, blocks: completed.append(blocks))
                try:
                    index, _ = source.pull(block=True)
                    self.assertEqual(source.inflight[index]["block_index"], 1)
                    source.complete(index, [123])
                    self.assertTrue(source.closed)
                    self.assertEqual(completed[0][0]["content"], "KEEP")
                    self.assertEqual(completed[0][1]["content"], "123")
                    self.assertEqual(source.metadata()["requests_completed_by_phase"], {"recognition": 1})
                finally:
                    source.close()


if __name__ == "__main__":
    unittest.main()
