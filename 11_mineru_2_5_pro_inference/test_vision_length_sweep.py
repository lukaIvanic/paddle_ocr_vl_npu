"""Coverage and failed-pair accounting; no accelerator or model imports."""
import json
from pathlib import Path
import tempfile
import unittest
from vision_length_sweep import choose, report


class LengthSweep(unittest.TestCase):
    def test_actual_crop_selection_covers_each_bucket_without_reuse(self):
        inventory = [dict(id=str(n), real_tokens=n) for n in [128,256,384,400,480,512,600,720,768]]
        picked = choose(inventory, [384,512,768], 2)
        self.assertEqual([r['bucket'] for r in picked], [384,384,512,512,768,768])
        self.assertEqual(len({r['id'] for r in picked}), 6)
        with self.assertRaises(ValueError):
            choose(inventory, [384,512,768,1024], 2)

    def test_failed_receipt_prevents_speedup_and_drift_does_not_hide_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lanes = []
            for repeat in [0,1]:
                for variant in ['baseline','pfa_approx']:
                    name = f'{variant}_{repeat}'
                    lanes.append(dict(name=name,route='real_crop',repeat=repeat,variant=variant))
                    (root/name).mkdir(); (root/(name+'.receipt')).mkdir()
                    ms = 100 if variant == 'baseline' else 80
                    result = dict(status='completed',device='TEST_ACCOUNTING',tags=dict(real_tokens=1000,physical_tokens=1024),
                        timing=dict(wall_ms=dict(mean=ms),device_ms=dict(mean=ms)),
                        full_encoder_parity=dict(relative_l2=.1, max_abs=1., exact=False),
                        wall_real_tok_s=1000000/ms, real_tok_s=1000000/ms)
                    (root/name/'result.json').write_text(json.dumps(result))
                    status = 'failed' if repeat == 1 and variant == 'pfa_approx' else 'completed'
                    (root/(name+'.receipt')/'exit.json').write_text(json.dumps(dict(status=status)))
            (root/'sweep.json').write_text(json.dumps(dict(lanes=lanes,routes=['real_crop'],repeats=2,scope='accounting test')))
            summary = report(root)
            self.assertEqual(len(summary['pairs']),1)
            self.assertEqual(summary['pairs'][0]['wall_tok_s_gain_pct'],25)
            self.assertIsNone(summary['production_weighted_gain'])
            self.assertEqual(summary['rows'][-1]['status'],'failed')


if __name__ == '__main__': unittest.main()
