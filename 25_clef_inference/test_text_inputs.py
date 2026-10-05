"""Small input-contract checks; checkpoint token parity is in the NPU smoke."""
from types import SimpleNamespace
import unittest

from text_inputs import encode_record, question_options, systemone_answer


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
