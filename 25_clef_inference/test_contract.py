"""Host-only fixture/provenance checks; not inference validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from run_local_smoke import encode_record, question_options, systemone_answer

from download_model import verify

ROOT = Path(__file__).parent


class ContractTests(unittest.TestCase):
    def test_fixture_contract(self):
        cases = json.loads((ROOT / "smoke_cases.json").read_text())
        self.assertEqual(len(cases), len({c["id"] for c in cases}))
        types = set()
        for case in cases:
            self.assertNotIn("images", case)
            self.assertNotIn("videos", case)
            for question in case["questions"].values():
                types.add(question["type"])
                self.assertTrue(question["instructions"])
                if question["type"] != "noul":
                    self.assertGreater(len(question["criteria"]), 1)
        self.assertEqual(types, {"choice", "score", "noul"})
        for language in ["english", "chinese"]:
            pair = [c for c in cases if c.get("ranking_group") == language]
            self.assertEqual(len(pair), 2)
            self.assertEqual({c["relevant"] for c in pair}, {True, False})
            self.assertEqual(pair[0]["questions"], pair[1]["questions"])

    def test_release_inventory(self):
        release = json.loads((ROOT / "release.json").read_text())
        self.assertEqual(len(release["revision"]), 40)
        self.assertEqual(len(release["files"]), 16)
        self.assertEqual(sum(name.startswith("model-") for name in release["files"]), 4)
        for name, entry in release["files"].items():
            self.assertEqual(Path(name).name, name)
            self.assertGreater(entry["size"], 0)
            digest = entry.get("sha256", entry.get("blob_id"))
            self.assertEqual(len(digest), 64 if "sha256" in entry else 40)

    def test_digest_verification(self):
        payload = b"example\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(payload)
            sha = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            blob = dict(size=len(payload), blob_id=hashlib.sha1(
                f"blob {len(payload)}\0".encode()+payload).hexdigest())
            self.assertTrue(verify(path, sha))
            self.assertTrue(verify(path, blob))
            path.write_bytes(b"exampleX")
            self.assertFalse(verify(path, sha))
            self.assertFalse(verify(path, blob))
            self.assertFalse(verify(path.with_name("missing"), sha))


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        return SimpleNamespace(ids=list(map(ord, text)))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(state="中文", questions={"q": dict(type="choice", instructions="Choose",
                                                          criteria={"b": "Beta", "a": "Alpha"})})

    def test_option_order_and_spans(self):
        encoded = encode_record(CharacterTokenizer(), self.case)
        q = encoded.questions[0]
        text = "".join(map(chr, encoded.input_ids))
        self.assertEqual(q.option_ids, ("a", "b"))
        self.assertEqual(text[slice(*q.question_span)], "Choose")
        self.assertEqual(text[slice(*q.option_spans[0])], '{"description":"Alpha","option_id":"a"}')
        self.assertIn("STATE:\n中文\n\nSCHEMA", text)

    def test_no_silent_truncation_or_media(self):
        with self.assertRaisesRegex(ValueError, "truncation is disabled"):
            encode_record(CharacterTokenizer(), self.case, max_length=10)
        with self.assertRaisesRegex(ValueError, "text only"):
            encode_record(CharacterTokenizer(), dict(self.case, images=["image.png"]))
        with self.assertRaisesRegex(ValueError, "At least one question"):
            encode_record(CharacterTokenizer(), dict(self.case, questions={}))

    def test_decision_types(self):
        self.assertEqual([key for key, _ in question_options(dict(type="noul"))], ["true", "false"])
        self.assertEqual(systemone_answer(dict(type="noul"), {"true": 0.123456, "false": 0.876544}),
                         {"type": "noul", "noul": 0.1235})
        answer = systemone_answer(dict(type="score", criteria=["bad", "good"]), {"0": 0.25, "1": 0.75})
        self.assertEqual(answer["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
