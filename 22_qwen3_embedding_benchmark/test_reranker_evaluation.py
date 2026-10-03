import unittest
from run_reranker_evaluation import candidate_rows, batches, command


class EvaluationTests(unittest.TestCase):
    def test_stream_preserves_candidates_and_tail(self):
        candidates = {'q2': {str(i): float(i) for i in range(100)},
                      'q1': {str(i): float(i) for i in range(100)}}
        queries = {'q1': 'query1', 'q2': 'query2'}
        corpus = {str(i): {'text': str(i)} for i in range(100)}
        rows = list(candidate_rows(candidates, queries, corpus, lambda docs: [d['text'] for d in docs]))
        self.assertEqual([(r['qid'], r['did']) for r in rows[:2]], [('q1', '99'), ('q1', '98')])
        self.assertEqual(len({(r['qid'], r['did']) for r in rows}), 200)
        self.assertEqual([len(b) for b in batches(iter(rows), 128)], [128, 72])
        self.assertEqual(list(batches(iter([]), 128)), [])

    def test_reject_incomplete_candidates(self):
        with self.assertRaises(ValueError):
            list(candidate_rows({'q': {'d': 1}}, {'q': 'query'}, {'d': {}}, lambda ds: ds))

    def test_tested_eager_server_configuration(self):
        cmd = command(18325)
        self.assertIn('--enforce-eager', cmd)
        self.assertIn('--no-enable-prefix-caching', cmd)
        self.assertIn('--no-async-scheduling', cmd)
        for flag, value in [('--dtype', 'float16'), ('--max-num-seqs','32'),
                            ('--max-num-batched-tokens','16384'), ('--max-model-len','8192')]:
            self.assertEqual(cmd[cmd.index(flag)+1], value)


if __name__ == '__main__':
    unittest.main()
