"""Portable accounting and complete-config controls, without NPU imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from analyze_vision_diagnostics import analyze, bucket, profile
from vision_diagnostic_config import DEFAULT_CONFIG, config_key, load_config


class Accounting(unittest.TestCase):
    def test_alternate_types_wait_and_format_conversion(self):
        rows=[]
        for kind,fmt,out,wait in [('BatchMatMulV2','ND;FRACTAL_NZ','ND','12'),
                ('GroupedMatmul','ND;ND','ND','3'),('OddAttention310','ND','ND','6'),
                ('TransData','ND','FRACTAL_NZ','0'),('Cast','ND','ND','N/A'),
                ('UnknownKernel310','ND','ND','0')]:
            rows.append({'Type':kind,'Duration(us)':'300','Wait Time(us)':wait,
                         'Block Num':'8','Mix Block Num':'0','Input Formats':fmt,'Output Formats':out,
                         'Accelerator Core':'AI_CORE','Start Time(us)':str(len(rows)*350),
                         'Device_id':'0','Stream ID':'1','Step Id':'0'})
        result=profile(rows,3)
        self.assertAlmostEqual(result['buckets']['matmul']['duration_ms_per_forward'],.2)
        self.assertAlmostEqual(result['total']['wait_ms_per_forward'],.007)
        self.assertEqual(result['total']['missing_wait'],1)
        self.assertIn('UnknownKernel310',result['unclassified_types'])
        self.assertEqual(result['conversion_directions'],{'ND -> FRACTAL_NZ':1})
        self.assertEqual(bucket(rows[4]),'remaining') # FP16/FP32 Cast is not a format conversion.
        self.assertEqual(len(result['kernel_calls']),3)

    def test_config_is_complete_and_cache_identity_distinguishes_internal_format(self):
        cfg=copy.deepcopy(DEFAULT_CONFIG);cfg.pop('allow_internal_format')
        with self.assertRaises(ValueError):load_config(inherited=cfg)
        off=copy.deepcopy(DEFAULT_CONFIG);off['allow_internal_format']=False
        self.assertNotEqual(config_key(DEFAULT_CONFIG),config_key(off))
        self.assertFalse(load_config(inherited=off)['allow_internal_format'])

    def test_missing_measurement_is_not_zero_and_receipt_failure_wins(self):
        p=profile([{'Type':'OtherAttention','Duration(us)':'NaN','Wait Time(us)':'N/A'}],3)
        self.assertIsNone(p['total']['duration_ms_per_forward'])
        self.assertIsNone(p['total']['wait_ms_per_forward'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'lane').mkdir(); (root/'lane.receipt').mkdir()
            (root/'lane'/'result.json').write_text(json.dumps({'variant':'baseline','status':'completed'}))
            (root/'lane.receipt'/'exit.json').write_text(json.dumps({'status':'failed','exit_code':1}))
            (root/'early_failure.receipt').mkdir()
            (root/'early_failure.receipt'/'exit.json').write_text(json.dumps({'status':'timeout'}))
            result=analyze(root,3)
            self.assertEqual({r['status'] for r in result['lanes']},{'failed','timeout'})


if __name__=='__main__':unittest.main()
