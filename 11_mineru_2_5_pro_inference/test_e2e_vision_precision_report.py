import unittest
from report_e2e_vision_precision import length_rows


class ProductionLengthAccounting(unittest.TestCase):
    def test_counts_useful_tokens_once_and_weights_by_time(self):
        rows=[dict(real_tokens=800,physical_tokens=1024,members=1,member_lengths=[800],device_s=.1),
              dict(real_tokens=1000,physical_tokens=1024,members=1,member_lengths=[1000],device_s=.2),
              dict(real_tokens=672,physical_tokens=768,members=2,member_lengths=[480,192],device_s=.1)]
        groups=length_rows(rows)
        self.assertAlmostEqual(groups['1024:single']['real_tok_s'],1800/.3)
        self.assertEqual(groups['768:packed']['real_tok_s'],6720)
        self.assertEqual(groups['768:packed']['members'],2)
        self.assertEqual(groups['1024:single']['actual_min'],800)
        with self.assertRaises(ValueError):
            length_rows([dict(rows[0],real_tokens=3076)])


if __name__=='__main__':unittest.main()
