import json
from pathlib import Path
import tempfile
import unittest

from generation_trace import GenerationTrace, request_identity


class Image:
    mode = "RGB"
    size = (1, 1)

    def tobytes(self):
        return b"abc"


class TraceTests(unittest.TestCase):
    def test_unfiltered_ids_and_prompt_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = GenerationTrace(path, eos_token_id=9)
            trace.contexts = [request_identity("page.png", "layout")]
            trace.begin_batch([Image()], ["prompt"])
            trace.prepared([1, 2, 3], 10)
            trace.finish_batch([[4, 8, 9]], ["text"])
            trace.close()
            result = json.loads(path.read_text())
            self.assertEqual(result["generated_token_ids"], [4, 8, 9])
            self.assertEqual(result["prompt_token_ids"], [1, 2, 3])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(result["request_id"], "page.png:layout")
            with self.assertRaises(FileExistsError):
                GenerationTrace(path, eos_token_id=9)

    def test_duplicates_and_invalid_stops_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = GenerationTrace(Path(directory) / "trace.jsonl", eos_token_id=9)
            record = dict(request_identity("p", "recognition", 2), max_new_tokens=3)
            with self.assertRaisesRegex(ValueError, "unexplained"):
                trace.write(record, [1], "bad")
            trace.close()

    def test_geometry_is_copied(self):
        block = {"type": "text", "bbox": [0, 0, 1, 1]}
        result = request_identity("p", "recognition", 3, block)
        block["bbox"][0] = 0.5
        self.assertEqual(result["bbox"][0], 0)


if __name__ == "__main__":
    unittest.main()
