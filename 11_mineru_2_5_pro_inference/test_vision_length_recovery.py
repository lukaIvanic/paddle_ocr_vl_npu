"""Exercise interruption/recovery bookkeeping without claiming model validation."""
from argparse import Namespace
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vision_diagnostic_config import DEFAULT_CONFIG
from vision_length_sweep import run


class Recovery(unittest.TestCase):
    def fixture(self, base):
        model=base/'model';model.mkdir()
        for name in ['config.json','model.safetensors']:
            (model/name).write_bytes(b'fixture, not model weights')
        image=base/'crop.png';image.write_bytes(b'fixture, not a benchmark image')
        selection=dict(config=dict(DEFAULT_CONFIG),model=str(model),processor_fast=False,scope='TEST_ACCOUNTING',
            selected=[dict(id='fixture',image=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),real_tokens=640,bucket=768)])
        path=base/'selection.json';path.write_text(json.dumps(selection))
        args=Namespace(selection=path,output_dir=base/'output',cache_root=base/'shared_cache',resume=False,
            steps=30,repeats=2,capture_timeout_s=3600,lane_timeout_s=1800)
        self.calls=[];self.fail_at=None;self.selection=selection
        self.commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        return args

    def child(self, command, receipt, timeout):
        def value(flag):return command[command.index(flag)+1]
        mode=command[3];self.calls.append((mode,command))
        root=Path(value('--output-dir'));root.mkdir();receipt.mkdir()
        (receipt/'command.json').write_text(json.dumps(dict(argv=command)))
        (receipt/'before.json').write_text(json.dumps(dict(utc='2026-10-08T00:00:00Z')))
        if mode=='capture-crops':
            cfg=json.loads(Path(value('--config-json')).read_text())
            tensor=root/'crop_0_bucket_768.pt';tensor.write_bytes(b'captured fixture')
            entry=dict(tags=dict(real_tokens=640,physical_tokens=768,members=1,member_lengths=[640]),bucket=768,
                image_sha256=self.selection['selected'][0]['image_sha256'],file=tensor.name,sha256=hashlib.sha256(tensor.read_bytes()).hexdigest())
            model=Path(self.selection['model'])
            self.manifest=dict(commit=self.commit,model=str(model),vision_config=cfg,
                model_hashes={name:hashlib.sha256((model/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors']},
                routes={'crop_0_bucket_768':entry})
            (root/'manifest.json').write_text(json.dumps(self.manifest))
        else:
            if self.fail_at==len(self.calls):
                (receipt/'exit.json').write_text(json.dumps(dict(status='failed')))
                raise RuntimeError('simulated interruption')
            variant=value('--variant');entry=self.manifest['routes'][value('--route')]
            cfg=self.manifest['vision_config'];ms=100 if variant=='baseline' else 80
            result=dict(status='completed',route=value('--route'),variant=variant,device='TEST_ACCOUNTING',soc_version=200,
                torch='fixture',torch_npu='fixture',commit=self.commit,source_capture_sha256=entry['sha256'],
                model_hashes=self.manifest['model_hashes'],capture_config=cfg,vision_config=dict(cfg,approximate_precision=variant=='pfa_approx'),
                tags=entry['tags'],timing_gate=dict(new_graphs=0,recompile_warnings=0),repeat_parity=dict(exact=True),
                full_encoder_parity=dict(exact=variant=='baseline',relative_l2=0.01,max_abs=0.1),
                timing=dict(wall_ms=dict(mean=ms),device_ms=dict(mean=ms)),wall_real_tok_s=640000/ms,real_tok_s=640000/ms)
            (root/'result.json').write_text(json.dumps(result))
        (receipt/'exit.json').write_text(json.dumps(dict(status='completed')))

    def invoke(self,args):
        with patch('vision_length_sweep.run_lane',side_effect=self.child),redirect_stdout(io.StringIO()):
            run(args)

    def test_resume_preserves_finished_pair_and_repeats_only_incomplete_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.fail_at=5
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):self.invoke(args)
            first=args.output_dir/'crop_0_bucket_768_r1_baseline.receipt/command.json'
            original=first.read_bytes()
            args.resume=True;self.fail_at=None;self.calls=[];self.invoke(args)
            self.assertEqual([c[0] for c in self.calls],['replay','replay'])
            self.assertEqual([c[1][c[1].index('--variant')+1] for c in self.calls],['baseline','pfa_approx'])
            self.assertTrue(all('r2_' in c[1][c[1].index('--output-dir')+1] for c in self.calls))
            self.assertEqual(original,first.read_bytes())
            self.assertTrue(list((args.output_dir/'attempts').glob('*/crop_0_bucket_768_r2_baseline/result.json')))
            result=json.loads((args.output_dir/'length_results.json').read_text())
            self.assertEqual(len(result['pairs']),2)
            self.assertTrue(all(c[1][c[1].index('--cache-root')+1]==str(args.cache_root) for c in self.calls))
            self.calls=[];self.invoke(args);self.assertEqual(self.calls,[])

    def test_changed_steps_or_model_are_rejected_without_running_children(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.invoke(args);args.resume=True;self.calls=[]
            args.steps=10
            with self.assertRaisesRegex(ValueError,'steps'):self.invoke(args)
            args.steps=30
            (Path(self.selection['model'])/'model.safetensors').write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError,'model changed'):self.invoke(args)
            self.assertEqual(self.calls,[])

    def test_legacy_run_discloses_missing_old_environment_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.invoke(args)
            (args.output_dir/'run_state.json').unlink();args.resume=True;self.calls=[];self.invoke(args)
            self.assertEqual(self.calls,[])
            self.assertIn('Full old CANN/processor environment equality cannot be established',
                (args.output_dir/'length_results.md').read_text())

    def test_new_output_directory_can_share_graph_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.invoke(args)
            old_cache=args.cache_root;args.output_dir=Path(directory)/'second_output';self.calls=[];self.invoke(args)
            self.assertEqual(len(self.calls),5)
            self.assertTrue(all(c[1][c[1].index('--cache-root')+1]==str(old_cache) for c in self.calls))

    def test_interrupted_capture_is_preserved_and_recaptured_with_same_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            def interrupted(command,receipt,timeout):
                self.child(command,receipt,timeout)
                (receipt/'exit.json').unlink()
                raise RuntimeError('capture interrupted before receipt completed')
            with patch('vision_length_sweep.run_lane',side_effect=interrupted),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'capture interrupted'):run(args)
            args.resume=True;self.calls=[];self.invoke(args)
            self.assertEqual([c[0] for c in self.calls],['capture-crops']+['replay']*4)
            self.assertTrue(list((args.output_dir/'attempts').glob('*/capture/manifest.json')))

    def test_corrupt_capture_is_rejected_instead_of_silently_recaptured(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.invoke(args)
            (args.output_dir/'capture/crop_0_bucket_768.pt').write_bytes(b'corruption')
            args.resume=True;self.calls=[]
            with self.assertRaisesRegex(ValueError,'tensor hash mismatch'):self.invoke(args)
            self.assertEqual(self.calls,[])

    def test_live_old_process_group_blocks_retry_without_moving_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));self.fail_at=5
            with self.assertRaises(RuntimeError):self.invoke(args)
            receipt=args.output_dir/'crop_0_bucket_768_r2_pfa_approx.receipt'
            (receipt/'process.json').write_text(json.dumps(dict(process_group=os.getpgrp())))
            args.resume=True;self.fail_at=None;self.calls=[]
            with self.assertRaisesRegex(RuntimeError,'still alive'):self.invoke(args)
            self.assertTrue(receipt.exists());self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
