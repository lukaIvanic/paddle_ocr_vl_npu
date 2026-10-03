import unittest
import importlib.util
from run_reranker_evaluation import candidate_rows, batches, command, metric_summary


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

    @unittest.skipUnless(importlib.util.find_spec('pytrec_eval'), 'Pinned evaluator dependency is remote')
    def test_metric_coverage_and_recall_invariance(self):
        before = {'q': {'relevant': .1, 'other': .9}}
        after = {'q': {'relevant': .9, 'other': .1}}
        metrics, per_query = metric_summary(after, before, {'q': {'relevant': 1}}, False)
        self.assertEqual(metrics['reranker']['ndcg_cut_10'], 1.)
        self.assertLess(metrics['embedding']['ndcg_cut_10'], 1.)
        self.assertEqual(metrics['reranker']['recall_100'], metrics['embedding']['recall_100'])
        self.assertEqual(set(per_query['reranker']), {'q'})
        with self.assertRaises(ValueError):
            metric_summary({'q': {'relevant': 1.}}, before, {'q': {'relevant': 1}}, False)
        with self.assertRaises(ValueError):
            metric_summary(after, before, {'missing': {'relevant': 1}}, False)


if __name__ == '__main__':
    unittest.main()
