import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from saved_paddle_crops import geometry_record, LABEL_MAP, sha256
from saved_layout_source import SavedLayoutPageSource
from test_streaming_pipeline import client, Block


class SavedCropTests(unittest.TestCase):
    def block(self, label="text", bbox=None):
        return {"block_label": label, "block_bbox": bbox or [1, 2, 9, 8],
                "block_polygon_points": [[1, 2], [9, 2], [9, 8], [1, 8]],
                "block_id": 7, "block_order": 3, "group_id": 1,
                "block_content": "FORBIDDEN_PADDLE_OCR"}

    def test_explicit_geometry_allowlist_and_mapping(self):
        for label in LABEL_MAP:
            record = geometry_record(self.block(label), 10, 10, 0)
            self.assertEqual(record["bbox"], [0.1, 0.2, 0.9, 0.8])
            self.assertNotIn("FORBIDDEN", json.dumps(record))
            self.assertNotIn("block_content", record)
        with self.assertRaises(KeyError):
            geometry_record(self.block("new_unknown_label"), 10, 10, 0)
        for bbox in ([-1, 0, 3, 4], [0, 0, 11, 10], [3, 3, 3, 5]):
            with self.assertRaises(ValueError):
                geometry_record(self.block(bbox=bbox), 10, 10, 0)

    def test_stream_routes_only_recognition_and_checks_hashes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "crop.png"
            Image.new("RGB", (8, 6), "white").save(image)
            record = geometry_record(self.block(), 10, 10, 0)
            record.update(crop="crop.png", crop_sha256=sha256(image), crop_size=[8, 6])
            visual = geometry_record(self.block("chart"), 10, 10, 1)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "saved_ocr_content_used": False,
                "pages": [{"image_name": "page.png", "blocks": [record, visual]}]}))
            stub = ModuleType("mineru_vl_utils.structs")
            stub.ContentBlock = Block
            c = client()
            c.prompts["[default]"] = "Text Recognition:"
            c.helper.resize_by_need = lambda im: im
            completed = []
            with patch.dict(sys.modules, {"mineru_vl_utils.structs": stub}):
                source = SavedLayoutPageSource(c, [("page.png", lambda: self.fail("page loader invoked"))],
                    manifest_path=manifest, on_page=lambda name, blocks: completed.append(blocks))
                try:
                    item = source.pull(block=True)
                    self.assertEqual(source.inflight[item[0]]["phase"], "recognition")
                    source.complete(item[0], [123])
                    self.assertTrue(source.closed)
                    self.assertEqual(source.metadata()["requests_admitted_by_phase"], {"recognition": 1})
                    self.assertEqual(len(completed[0]), 2)
                    self.assertEqual(completed[0][0]["content"], "123")
                    self.assertIsNone(completed[0][1]["content"])
                    image.write_bytes(b"corrupt")
                    with self.assertRaises(ValueError):
                        source._saved_job("page.png")
                finally:
                    source.close()


if __name__ == "__main__":
    unittest.main()
