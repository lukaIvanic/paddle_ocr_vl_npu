"""CPU-only scoring/metrics bookkeeping tests, not model inference."""
import unittest
import numpy as np
import torch
from run_hr_evaluation import aggregate, distribution, maxsim_column


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
            dict(wall_s=w,device_ms=w*900,tokens=t,route='eager')}) for w,t in [(1,10),(3,30)]]
        result=aggregate(rows)
        self.assertEqual(result['by_section'][0]['wall_tok_s'],10)
        self.assertEqual(len(result['by_shape']),2)
        self.assertEqual(result['item_wall_s']['page']['count'],2)

    def test_distribution(self):
        self.assertEqual(distribution([]),{'count':0})
        self.assertEqual(distribution([1,2,3])['p50'],2)

if __name__=='__main__':
    unittest.main()
