import tempfile
import unittest
from pathlib import Path

from forward_profile_analysis import distribution, interval_union_us, summarize_kernel_csv


class ProfileAccountingTests(unittest.TestCase):
    def test_overlapping_streams_do_not_double_count_elapsed_time(self):
        self.assertEqual(interval_union_us([(0, 10), (3, 6), (8, 15), (20, 25)]), 20)
        self.assertEqual(interval_union_us([]), 0)
        with self.assertRaises(ValueError):
            interval_union_us([(4, 3)])

    def test_invalid_timing_cannot_be_reported_as_a_valid_benchmark(self):
        for values in ([], [float('nan')], [-1], [float('inf')]):
            with self.assertRaises(ValueError):
                distribution(values)
        self.assertEqual(distribution([1, 3])['p50'], 2)

    def test_kernel_sums_and_stream_coverage_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'kernel_details.csv'
            path.write_text('Type,Start Time(us),Duration(us)\nMatMul,0,100\nCast,50,100\n')
            result = summarize_kernel_csv(path, 2)
        self.assertEqual(result['kernel_duration_sum_ms_per_forward'], .1)
        self.assertEqual(result['interval_union_ms'], .15)
        self.assertEqual(result['uncovered_envelope_ms'], 0)
        self.assertEqual(result['top_types'][0]['count_per_forward'], .5)


if __name__ == '__main__':
    unittest.main()
