import csv
from pathlib import Path
import tempfile
import unittest

from analyze_attention_pipes import analyze_csv, number, stats


class CounterTests(unittest.TestCase):
    def test_missing_is_not_zero(self):
        self.assertIsNone(number('N/A'))
        self.assertIsNone(number('nan'))
        self.assertIsNone(number(''))
        self.assertEqual(number('0'),0)
        self.assertEqual(stats([None,0,2])['mean'],1)

    def test_preserve_zero_in_weighted_mean_and_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'kernel_details.csv'
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['Type','Duration(us)','aic_mte2_time(us)','aic_mac_ratio'])
                writer.writeheader()
                for duration,time,ratio in [(10,0,0),(20,6,1),(30,'N/A','')]:
                    writer.writerow({'Type':'PromptFlashAttention','Duration(us)':duration,
                        'aic_mte2_time(us)':time,'aic_mac_ratio':ratio})
            result=analyze_csv(path,3)[0]
            self.assertEqual(result['elapsed_ms_per_forward'],.02)
            self.assertEqual(result['calls_per_forward'],1)
            metric=result['pmu']['aic_mte2_time(us)']
            self.assertEqual(metric['duration_weighted_mean'],4)
            self.assertEqual(metric['mean'],3)
            self.assertEqual(metric['missing_count'],1)
            self.assertEqual(metric['engine_time_sum_us_per_forward'],2)
            self.assertNotIn('engine_time_sum_us_per_forward',result['pmu']['aic_mac_ratio'])

    def test_negative_traffic_is_flagged_not_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'kernel_details.csv'
            path.write_text('Type,Duration(us),aic_GM_to_L1_datas(KB)\nPromptFlashAttention,10,-99\n')
            result=analyze_csv(path,1)[0]
            metric=result['pmu']['aic_GM_to_L1_datas(KB)']
            self.assertEqual(metric['invalid_negative_count'],1)
            self.assertIsNone(metric['mean'])
            self.assertEqual(result['calls'][0]['pmu']['aic_GM_to_L1_datas(KB)'],-99)


if __name__ == '__main__':
    unittest.main()
