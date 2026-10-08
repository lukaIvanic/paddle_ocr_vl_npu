import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from run_reranker_evaluation import candidate_rows, batches, command, metric_summary, suite_summary, restore_journal, unscored_rows, select_query_shard


class EvaluationTests(unittest.TestCase):
    def test_query_shards_are_disjoint_complete_and_keep_candidate_groups(self):
        baseline={f'q{i}':{str(j):float(j) for j in range(100)} for i in range(11)}
        qrels={q:{'0':1} for q in baseline}
        seen=set()
        sizes=[]
        for index in range(4):
            selected,rels=select_query_shard(baseline,qrels,index,4)
            self.assertFalse(seen.intersection(selected))
            self.assertEqual(set(selected),set(rels))
            self.assertTrue(all(selected[q] is baseline[q] for q in selected))
            seen.update(selected);sizes.append(len(selected))
        self.assertEqual(seen,set(baseline))
        self.assertEqual(sizes,[3,3,3,2])

    def test_resume_preserves_scores_and_partial_queries(self):
        rows = [{'qid':'q1','did':'d1'}, {'qid':'q1','did':'d2'}, {'qid':'q2','did':'d1'}]
        entry = {'pairs':2, 'tokens':30, 'lengths':[10,20], 'truncated':0,
                 'pairs_scored':[['q2','d1',.25], ['q1','d2',.75]]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'journal.jsonl'
            path.write_text(json.dumps(entry)+'\n')
            scores, records = restore_journal(path)
            self.assertEqual(scores, {'q1':{'d2':.75}, 'q2':{'d1':.25}})
            self.assertEqual(list(unscored_rows(iter(rows), scores)), [rows[0]])
            self.assertEqual(sum(r['tokens'] for r in records), 30)
            path.write_text((json.dumps(entry)+'\n')*2)
            with self.assertRaises(ValueError):
                restore_journal(path)
            path.write_text(json.dumps({**entry,'tokens':31})+'\n')
            with self.assertRaises(ValueError):
                restore_journal(path)

    def test_macro_summary_is_task_weighted_and_partial_is_labeled(self):
        rows = [{'task': name, 'queries': n, 'pairs': n*100, 'tokens': n*1000,
                 'metrics': {'reranker': {'ndcg_cut_10': score}, 'embedding': {'ndcg_cut_10': .5}}}
                for name,n,score in [('EcomRetrieval',1000,1.), ('T2Retrieval',22812,0.)]]
        result = suite_summary(rows)
        self.assertEqual(result['macro_ndcg_at_10_percent']['reranker'], 50.)
        self.assertEqual(result['scope'], 'partial_CMTEB-R')
        self.assertIsNone(result['delta_vs_published_pp'])
        with self.assertRaises(ValueError):
            suite_summary(rows + rows)

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
