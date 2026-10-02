"""CPU-only scoring/metrics bookkeeping tests, not model inference."""
import unittest
import tempfile
import json
from pathlib import Path
from contextlib import redirect_stdout
import io
import time
from unittest.mock import Mock, patch
import numpy as np
import torch
from run_hr_evaluation import aggregate, distribution, maxsim_column, Journal, select_workload


class HrTests(unittest.TestCase):
    def test_maxsim_ragged_queries_and_zero_document_rows(self):
        q=[torch.tensor([[-1.,0.],[0.,1.]]),torch.tensor([[1.,1.]])]
        d=torch.tensor([[1.,0.],[0.,0.]])
        actual=maxsim_column(torch.cat(q),d,[0,2])
        expected=np.array([float((x@d.T).max(-1).values.sum()) for x in q])
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(actual.tolist(),[0.,1.])

    def test_weighted_throughput(self):
        rows=[dict(kind='page',wall_s=w,sections={'vision_transformer':
            dict(host_s=w,device_interval_ms=w*900,tokens=t,route='eager')}) for w,t in [(1,10),(3,30)]]
        result=aggregate(rows)
        self.assertAlmostEqual(result['by_section'][0]['device_tok_s'],40/3.6)
        self.assertEqual(len(result['by_shape']),2)
        self.assertEqual(result['item_wall_s']['page']['count'],2)

    def test_distribution(self):
        self.assertEqual(distribution([]),{'count':0})
        self.assertEqual(distribution([1,2,3])['p50'],2)

    def test_journal_phase_and_run_elapsed_are_distinct(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            journal=Journal(Path(root))
            journal.emit('scoring_progress',elapsed_s=1.25)
            journal.close()
            row=json.loads((Path(root)/'events.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['elapsed_s'],1.25)
            self.assertIn('run_elapsed_s',row)

    def test_events_are_deferred_without_synchronization(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), patch('torch.npu',create=True) as npu:
            a,b=Mock(),Mock(); npu.Event.side_effect=[a,b]
            b.query.return_value=False; a.elapsed_time.return_value=2.0
            journal=Journal(Path(root))
            row=dict(kind='page',id='x',sections={},wall_s=.1)
            with journal.section(row,'vision',tokens=100,device=True):
                pass
            npu.synchronize.assert_not_called(); b.synchronize.assert_not_called()
            journal.resolve(); self.assertEqual(row['sections']['vision']['device_status'],'pending')
            b.query.return_value=True
            journal.complete(row,1,1,time.perf_counter()-.1)
            self.assertEqual(row['sections']['vision']['device_tok_s'],50000)
            self.assertEqual(len(journal.pending),0)
            # Completion is flushed immediately, without waiting for heartbeat.
            self.assertIn('item_finish',(Path(root)/'events.jsonl').read_text())
            self.assertIn('vision',(Path(root)/'items.jsonl').read_text())
            journal.close()
            b.synchronize.assert_not_called()

    def test_dev_selection_is_fixed_and_full_is_explicit(self):
        corpus=[dict(id=str(i)) for i in range(1110)]
        queries=[dict(id='q'+str(i)) for i in range(318)]
        qrels={q['id']:{'5':1,'6':1} for q in queries}
        c,q,r=select_workload(corpus,queries,qrels,'dev')
        self.assertEqual((len(c),len(q)),(111,32))
        self.assertEqual(c[0]['id'],'5'); self.assertEqual(c[-1]['id'],'1105')
        self.assertTrue(all(v=={'5':1} for v in r.values()))
        self.assertEqual((c,q,r),select_workload(corpus,queries,qrels,'dev'))
        self.assertEqual(select_workload(corpus,queries,qrels,'full'),(corpus,queries,qrels))

if __name__=='__main__':
    unittest.main()
