from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'18_unirec_paddle_hybrid_pipeline'), str(ROOT/'09_persistent_page_engine')]
from hybrid_timing import PipelineTiming
from coordinator import Coordinator
from run_pipeline import engine_report


class TimingTests(unittest.TestCase):
    def test_nested_owner_partition_is_exact_and_does_not_add_cpu(self):
        t = PipelineTiming()
        # ns origin does not matter for duration accounting.
        origin = t.trace.origin_ns
        with patch('hybrid_timing.time.perf_counter_ns', side_effect=[origin,origin+10,origin+30,origin+100]):
            with t.scope('pipeline'):
                with t.scope('child'):
                    pass
        t.trace.record_span('CPU service','background',origin,origin+100)
        r=t.summary()
        self.assertAlmostEqual(r['owner_exclusive']['pipeline']['total_s'],80/1e9)
        self.assertAlmostEqual(r['owner_partition_sum_s'],100/1e9)
        self.assertAlmostEqual(r['owner_partition_error_s'],0)

    def test_joint_wait_counted_once(self):
        t=PipelineTiming()
        with t.scope('pipeline'):
            with t.scope('shared.wait',args={'blockers':['page.input','unirec.cpu_preparation']}):
                pass
        r=t.summary()
        self.assertEqual(len(r['wait_blocker_sets']),1)
        self.assertEqual(r['exposed_cpu_dependency_wait_s'],r['owner_inclusive']['shared.wait']['total_s'])

    def test_worker_queue_service_and_ready_stay_off_owner_partition(self):
        t=PipelineTiming()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with t.scope('pipeline'):
                f=t.submit(pool,lambda:42,stage='cpu.crop',flow_id='page_000001_block_000002')
                self.assertEqual(f.result(timeout=2),42)
                t.consume(f)
        r=t.summary()
        for k in ('cpu_service','cpu_queue_residence','cpu_ready_residence'):
            self.assertEqual(r[k]['cpu.crop']['count'],1)
        self.assertNotIn('cpu.crop',r['owner_exclusive'])

    def test_disabled_timing_does_not_wrap(self):
        t=PipelineTiming(False)
        obj=SimpleNamespace(call=lambda:42)
        original=obj.call
        t.instrument(obj,'call','test')
        self.assertIs(obj.call,original)
        with t.scope('pipeline'):
            self.assertEqual(obj.call(),42)
        self.assertEqual(t.summary(),{'enabled':False})

    def test_legacy_pause_timers_are_removed_without_mutating_engine(self):
        summary={'timing_detail':{'run_wall_s':100,'scheduler_bookkeeping_residual_s':90,'decode_s':10}}
        a=SimpleNamespace(summary=summary,graph_calls=1,capacity=2)
        r=engine_report(a)
        self.assertEqual(r['summary']['timing_detail'],{'decode_s':10})
        self.assertIn('run_wall_s',summary['timing_detail'])
        self.assertEqual(r['cooperative_pause_inclusive_legacy_timing']['values']['legacy_run_wall_s'],100)

    def test_wait_snapshot_distinguishes_joint_dependencies_and_ready_race(self):
        f=Future()
        request=SimpleNamespace(request_id='crop')
        adapter=SimpleNamespace(done=False,free=1,active=1,ready_count=0,capacity=2,occupied=1,
                                pending=[1],cpu=SimpleNamespace(futures={'crop':f}),
                                planned_prefill_requests=lambda:[request])
        page_future=Future()
        pages=SimpleNamespace(preparation=SimpleNamespace(input_future=page_future,crop_future=None),
                              exhausted=False,set_wakeup=lambda n:None)
        c=Coordinator({'unirec':adapter},pages)
        self.assertEqual(c.wait_snapshot()['blockers'],['page.input','unirec.cpu_preparation'])
        f.set_result(42)
        self.assertEqual(c.wait_snapshot()['blockers'],['ready_before_wait'])
