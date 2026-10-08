import csv
import tempfile
import unittest
from pathlib import Path

from analyze_portable_profile import analyze_csv, projection_roles, category


class ProfileAnalysis(unittest.TestCase):
    def write(self,root,rows):
        path=Path(root)/'kernels.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
        return path

    def test_normalization_overlaps_and_zero_counters(self):
        rows=[{'Type':'MatMulV2','Name':'MatMul_1','Duration(us)':'1000','Start Time(us)':'0',
               'Input Shapes':'1280,2560;2560,6144','Output Shapes':'1280,6144',
               'aic_mac_ratio':'0','aic_mte2_ratio':'N/A'},
              {'Type':'MatMulV2','Name':'MatMul_2','Duration(us)':'1000','Start Time(us)':'500',
               'Input Shapes':'1280,2560;2560,6144','Output Shapes':'1280,6144',
               'aic_mac_ratio':'20','aic_mte2_ratio':''}]
        with tempfile.TemporaryDirectory() as root:
            r=analyze_csv(self.write(root,rows),2,[dict(role='text.layers.*.qkv',in_features=2560,out_features=6144)])
        self.assertEqual(r['kernel_sum_ms_per_forward'],1)
        self.assertEqual(r['interval_union_ms'],1.5)
        self.assertEqual(r['groups'][0]['calls_per_forward'],1)
        self.assertEqual(r['groups'][0]['counters']['aic_mac_ratio']['duration_weighted_mean'],10)
        self.assertEqual(r['counter_columns_without_values'],['aic_mte2_ratio'])
        self.assertEqual(r['groups'][0]['projection_role_candidates'],['text.layers.*.qkv'])

    def test_alternate_columns_and_missing_counters(self):
        with tempfile.TemporaryDirectory() as root:
            r=analyze_csv(self.write(root,[{'Op Type':'Transpose','Task Duration(us)':'25'}]),1,[])
        self.assertEqual(r['kernel_sum_ms_per_forward'],.025)
        self.assertEqual(r['counter_columns_with_values'],[])
        self.assertIsNone(r['interval_union_ms'])

    def test_malformed_duration_is_not_silently_zero(self):
        for invalid in ('','N/A','nan','-1'):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    analyze_csv(self.write(root,[{'Type':'MatMul','Duration(us)':invalid}]),1,[])

    def test_ambiguous_projection_roles_remain_candidates(self):
        m=[dict(role='vision.proj',in_features=1024,out_features=1024),
           dict(role='other.proj',in_features=1024,out_features=1024)]
        self.assertEqual(projection_roles('5040,1024;1024,1024','5040,1024',m),['other.proj','vision.proj'])
        self.assertEqual(projection_roles('5040,1024;64,64,16,16','5040,1024',m),[])

    def test_unknown_fusions_not_falsely_attributed_to_norm(self):
        self.assertEqual(category('Add','Add_12'),'other_elementwise_or_unclassified')
        self.assertEqual(category('StridedSliceD','StridedSliceD_1'),'layout_slice_repeat_padding')


if __name__=='__main__':
    unittest.main()
