import unittest
from protocol import TASKS, format_text


class ProtocolTests(unittest.TestCase):
    def test_all_queries_use_exact_english_instruction(self):
        for task, (_, _, instruction, _) in TASKS.items():
            self.assertEqual(format_text("中国", task, "query"), f"Instruct: {instruction}\nQuery:中国")

    def test_documents_are_unmodified(self):
        for task in TASKS:
            self.assertEqual(format_text(" 中国\n文档 ", task, "passage"), " 中国\n文档 ")

    def test_roles_fail_closed(self):
        with self.assertRaises(ValueError):
            format_text("x", "EcomRetrieval", None)

    def test_reference_mean(self):
        self.assertAlmostEqual(sum(v[3] for v in TASKS.values()) / len(TASKS), .7102525)


if __name__ == "__main__":
    unittest.main()
