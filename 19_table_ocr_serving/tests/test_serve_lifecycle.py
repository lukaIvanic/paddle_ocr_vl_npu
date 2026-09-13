"""CPU-only checks of HTTP routing and HTTP/inference communication; no model is loaded."""

import ast
import asyncio
import html
import io
import json
import queue
import re
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p01_serve as serve


def make_config(**overrides):
    values = dict(model_path=Path('/unused/model'),
                  graph_cache_directory=Path('/unused/graphs'),
                  log_folder=Path('/unused/logs'), request_timeout_s=2.0,
                  shutdown_timeout_s=2.0)
    values.update(overrides)
    return serve.ServeConfig(**values)


def fake_inference_process(jobs, results, config):
    """Spawnable test worker: exercise real IPC without importing inference code."""
    import os

    results.put(dict(kind='ready', worker_pid=os.getpid(),
                     configuration={'model_path': str(config.model_path),
                                    'torch_imported': 'torch' in sys.modules}))
    completed = 0
    while True:
        job = jobs.get()
        if job is None:
            break
        if job['image_bytes'] == b'slow':
            __import__('time').sleep(0.2)
        results.put(dict(kind='result', request_id=job['request_id'], ok=True,
                         payload={'text': job['image_bytes'].decode('utf-8')}))
        completed += 1
    results.put(dict(kind='service_summary', payload={'requests': completed}))


class FakeProcess:
    """Only process lifecycle is mocked; result delivery uses real Python threads."""

    def __init__(self, *, target, args, name):
        self.target, self.args, self.name = target, args, name
        self.pid = None
        self.alive = False
        self.terminated = False
        self.startup_message = dict(kind='ready', configuration={'loaded': True}, worker_pid=123)
        self.join_timeouts = []

    def start(self):
        self.pid = 123
        self.alive = True
        if self.startup_message is not None:
            self.args[1].put(self.startup_message)

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.join_timeouts.append(timeout)
        self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False


class ServeLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_folder = Path(directory.name) / 'logs'

    def test_real_spawned_process_communication(self):
        import os

        with patch.object(serve, 'run_inference_process', fake_inference_process):
            connection = serve.InferenceServer(make_config(request_timeout_s=5, log_folder=self.log_folder))
        try:
            self.start(connection)
            self.assertNotEqual(connection.worker_pid, os.getpid())
            self.assertFalse(connection.worker_runtime_info['torch_imported'])
            self.assertEqual(connection.worker_runtime_info['model_path'], '/unused/model')
            result = connection.recognize('real-ipc', 'table', b'test OCR')
            self.assertEqual(result['payload']['text'], 'test OCR')
            self.assertEqual(connection.close(), {'requests': 1})
            self.assertEqual(connection.inference_process.exitcode, 0)
        finally:
            if connection.inference_process.is_alive():
                connection.inference_process.terminate()
                connection.inference_process.join(timeout=2)
            self.stop_reader(connection)
            connection.jobs.close()
            connection.results.close()
            connection.jobs.join_thread()
            connection.results.join_thread()

    def make_connection(self, **config_overrides):
        config_overrides.setdefault('log_folder', self.log_folder)
        context = SimpleNamespace(Queue=queue.Queue, Process=FakeProcess)
        with patch.object(serve.mp, 'get_context', return_value=context) as get_context:
            connection = serve.InferenceServer(make_config(**config_overrides))
        get_context.assert_called_once_with('spawn')
        self.assertIs(connection.inference_process.target, serve.run_inference_process)
        self.assertIs(connection.inference_process.args[0], connection.jobs)
        self.assertIs(connection.inference_process.args[1], connection.results)
        self.assertFalse(connection.result_reader.is_alive())
        self.addCleanup(self.stop_reader, connection)
        return connection

    @staticmethod
    def stop_reader(connection):
        connection.stop_reading_results.set()
        if connection.result_reader.ident is not None:
            connection.result_reader.join(timeout=1)
        connection._close_logging()

    def start(self, connection):
        with patch('sys.stdout', new=io.StringIO()):
            connection.start()

    def test_startup_success_and_out_of_order_request_results(self):
        connection = self.make_connection()
        self.start(connection)
        self.assertEqual(connection.worker_runtime_info, {'loaded': True})
        self.assertEqual(connection.worker_pid, 123)
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = {
                str(i): callers.submit(connection.recognize, str(i), 'table', b'image')
                for i in range(4)
            }
            jobs = [connection.jobs.get(timeout=1) for _ in range(4)]
            self.assertEqual(len(connection.pending_requests), 4)
            for job in reversed(jobs):
                self.assertIn('submitted_monotonic_s', job)
                connection.results.put(dict(kind='result', request_id=job['request_id'],
                                            ok=True, payload={'text': job['request_id']}))
            for request_id, future in futures.items():
                self.assertEqual(future.result(timeout=1)['payload']['text'], request_id)
        self.assertEqual(connection.pending_requests, {})

    def test_startup_error(self):
        connection = self.make_connection()
        message = dict(kind='startup_error', error='model failed', traceback='failure details')
        connection.inference_process.startup_message = message
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(RuntimeError):
            self.start(connection)
        self.assertEqual(connection.startup_error, message)

    def test_startup_wait_has_no_deadline(self):
        connection = self.make_connection(request_timeout_s=.001)
        with patch.object(connection.startup_finished, 'wait', wraps=connection.startup_finished.wait) as wait:
            self.start(connection)
        wait.assert_called_once_with()

    def test_worker_exit_during_startup_ends_the_wait(self):
        connection = self.make_connection()
        connection.inference_process.startup_message = None
        def exit_immediately():
            connection.inference_process.pid = 123
            connection.inference_process.alive = False
        connection.inference_process.start = exit_immediately
        with patch('sys.stderr', new=io.StringIO()), self.assertRaisesRegex(RuntimeError, 'worker exited'):
            self.start(connection)

    def test_request_timeout_does_not_cancel_job_and_late_result_is_ignored(self):
        connection = self.make_connection(request_timeout_s=0.05)
        self.start(connection)
        with self.assertRaises(serve.InferenceTimeout):
            connection.recognize('expired', 'table', b'image')
        self.assertEqual(connection.jobs.get_nowait()['request_id'], 'expired')
        self.assertTrue(connection.pending_requests['expired'].cancelled())
        connection.results.put(dict(kind='result', request_id='expired', ok=True, payload={}))
        # A later valid request still receives its own result, not the expired one.
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(connection.recognize, 'next', 'table', b'image')
            self.assertEqual(connection.jobs.get(timeout=1)['request_id'], 'next')
            connection.results.put(dict(kind='result', request_id='next', ok=True, payload={'text': 'next'}))
            self.assertEqual(future.result(timeout=1)['payload']['text'], 'next')

    def test_failed_queue_submission_releases_capacity(self):
        connection = self.make_connection()
        self.start(connection)
        with patch.object(connection.jobs, 'put_nowait', side_effect=OSError('closed')), self.assertRaises(OSError):
            connection.recognize('full', 'table', b'image')
        self.assertEqual(connection.pending_requests, {})

    def test_capacity_counts_requests_removed_from_the_job_queue(self):
        connection = self.make_connection(max_in_flight_requests=1)
        self.start(connection)
        with ThreadPoolExecutor(max_workers=1) as callers:
            first = callers.submit(connection.recognize, 'first', 'table', b'image')
            connection.jobs.get(timeout=1)  # The worker has taken it, but OCR is not done.
            self.assertTrue(connection.jobs.empty())
            with self.assertRaises(serve.InferenceCapacityFull):
                connection.recognize('second', 'text', b'image')
            connection.results.put(dict(kind='result',request_id='first',ok=True,payload={}))
            first.result(timeout=1)
        self.assertEqual(connection.pending_requests,{})

    def test_shutdown_finishes_pending_request_and_receives_summary(self):
        connection = self.make_connection()
        self.start(connection)
        with ThreadPoolExecutor(max_workers=2) as callers:
            request = callers.submit(connection.recognize, 'last', 'table', b'image')
            self.assertEqual(connection.jobs.get(timeout=1)['request_id'], 'last')
            shutdown = callers.submit(connection.close)
            self.assertIsNone(connection.jobs.get(timeout=1))
            self.assertFalse(shutdown.done())
            connection.results.put(dict(kind='result', request_id='last', ok=True, payload={'text': 'done'}))
            connection.results.put(dict(kind='service_summary', payload={'requests': 1}))
            self.assertEqual(request.result(timeout=1)['payload']['text'], 'done')
            self.assertEqual(shutdown.result(timeout=1), {'requests': 1})
        self.assertFalse(connection.inference_process.terminated)
        self.assertEqual(connection.inference_process.join_timeouts, [10.0])
        self.assertTrue(connection.stop_reading_results.is_set())
        summary_path = connection.serve_config.log_folder / 'service_summary.json'
        self.assertEqual(json.loads(summary_path.read_text())['summary'], {'requests': 1})
        self.assertFalse(summary_path.with_name('.service_summary.json.tmp').exists())

    def test_shutdown_timeout_terminates_worker(self):
        connection = self.make_connection(shutdown_timeout_s=0.02)
        self.start(connection)
        self.assertIsNone(connection.close())
        self.assertTrue(connection.inference_process.terminated)

    def make_handler(self, path, body=b'image'):
        handler = serve.HttpRequestHandler.__new__(serve.HttpRequestHandler)
        handler.path = path
        handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.server = SimpleNamespace(serve_config=make_config(), inference_server=Mock())
        handler._json = Mock()
        return handler

    def test_http_result_errors_and_removed_drain_endpoint(self):
        handler = self.make_handler('/v1/ocr?crop_type=table&request_id=public')
        handler.server.inference_server.recognize.return_value = dict(ok=True, payload={'text': 'table'})
        handler.do_POST()
        status, payload = handler._json.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual(payload['request_id'], 'public')
        self.assertIn('http_wall_s', payload)
        request_id, crop_type, image_bytes = handler.server.inference_server.recognize.call_args.args
        self.assertEqual((crop_type, image_bytes), ('table', b'image'))
        self.assertNotEqual(request_id, 'public')
        for error, expected in ((serve.InferenceCapacityFull(), 503), (serve.InferenceTimeout(), 504), (ValueError('bad'), 500)):
            handler = self.make_handler('/v1/ocr?crop_type=table')
            handler.server.inference_server.recognize.side_effect = error
            handler.do_POST()
            self.assertEqual(handler._json.call_args.args[0], expected)
        for path, body, expected in (('/v1/drain', b'', 404),
                                     ('/v1/ocr?crop_type=unknown', b'image', 400),
                                     ('/v1/ocr?crop_type=table', b'', 413)):
            handler = self.make_handler(path, body)
            handler.do_POST()
            self.assertEqual(handler._json.call_args.args[0], expected)
            handler.server.inference_server.recognize.assert_not_called()

    def test_health_and_ready_responses(self):
        connection = self.make_connection()
        for path in ('/health', '/ready'):
            handler = self.make_handler(path)
            handler.server.inference_server = connection
            handler.do_GET()
            status, payload = handler._json.call_args.args
            self.assertFalse(payload['ok' if path == '/health' else 'ready'])
            self.assertEqual(status, 200 if path == '/health' else 503)
        self.start(connection)
        handler = self.make_handler('/ready')
        handler.server.inference_server = connection
        handler.do_GET()
        self.assertEqual(handler._json.call_args.args[0], 200)
        self.assertEqual(handler._json.call_args.args[1]['configuration'], {'loaded': True})

    def test_main_lifecycle(self):
        for http_error in (None, RuntimeError('http failure')):
            with self.subTest(http_error=http_error), tempfile.TemporaryDirectory() as directory:
                config = make_config(log_folder=Path(directory) / 'logs')
                connection = Mock(worker_runtime_info={'loaded': True}, worker_pid=123)
                connection.close.return_value = {'requests': 4}
                order = []
                connection.start.side_effect = lambda: order.append('start inference')
                def stop():
                    order.append('stop inference')
                    return {'requests': 4}
                connection.close.side_effect = stop
                server = serve.HttpServer.__new__(serve.HttpServer)
                server.inference_server = connection
                server.server_close = Mock(side_effect=lambda: order.append('close HTTP'))
                def run_http():
                    order.append('serve HTTP')
                    if http_error is not None:
                        raise http_error
                server.run = Mock(side_effect=run_http)
                with patch.object(serve, 'parse_args', return_value=config), \
                     patch.object(serve, 'InferenceServer', return_value=connection), \
                     patch.object(serve, 'HttpServer', return_value=server) as constructor, \
                     patch('sys.stdout', new=io.StringIO()):
                    if http_error is None:
                        serve.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'http failure'):
                            serve.main()
                constructor.assert_called_once_with(config, connection)
                self.assertEqual(order, ['start inference', 'serve HTTP', 'close HTTP', 'stop inference'])

    def test_http_close_does_not_stop_inference(self):
        server = serve.HttpServer.__new__(serve.HttpServer)
        server.inference_server = Mock()
        server.server_close = Mock()
        server.close()
        server.server_close.assert_called_once_with()
        server.inference_server.close.assert_not_called()

    def test_main_closes_inference_even_when_http_close_fails(self):
        inference = Mock()
        http = Mock()
        http.close.side_effect = OSError('socket cleanup failed')
        with patch.object(serve, 'parse_args', return_value=make_config()), \
             patch.object(serve, 'InferenceServer', return_value=inference), \
             patch.object(serve, 'HttpServer', return_value=http), \
             self.assertRaisesRegex(OSError, 'socket cleanup failed'):
            serve.main()
        inference.close.assert_called_once_with()

    def test_close_before_start_is_safe(self):
        inference = self.make_connection()
        self.assertIsNone(inference.close())
        self.assertFalse(inference.inference_process.terminated)

    def test_real_http_with_independent_inference_service(self):
        import http.client

        config = make_config(host='127.0.0.1', port=0, log_folder=self.log_folder, request_timeout_s=5)
        with patch.object(serve, 'run_inference_process', fake_inference_process):
            inference = serve.InferenceServer(config)
        http_server = serve.HttpServer(config, inference)
        listening, request_started = threading.Event(), threading.Event()
        original_activate = http_server.server_activate
        original_recognize = inference.recognize
        def activate():
            original_activate()
            listening.set()
        def recognize(*args):
            request_started.set()
            return original_recognize(*args)
        def client_request():
            self.assertTrue(listening.wait(timeout=3))
            client = http.client.HTTPConnection(*http_server.server_address, timeout=5)
            try:
                client.request('POST', '/v1/ocr?crop_type=table&request_id=one', body=b'slow')
                self.assertTrue(request_started.wait(timeout=3))
                # Stop the HTTP loop while an accepted request still awaits OCR.
                http_server._stop_accepting_connections(serve.signal.SIGTERM, None)
                response = client.getresponse()
                return response.status, json.loads(response.read())
            finally:
                client.close()
                http_server._stop_accepting_connections(serve.signal.SIGTERM, None)
        try:
            self.start(inference)
            with ThreadPoolExecutor(max_workers=1) as clients, \
                 patch.object(http_server, 'server_activate', side_effect=activate), \
                 patch.object(inference, 'recognize', side_effect=recognize), \
                 patch.object(serve.signal, 'signal'), patch('sys.stdout', new=io.StringIO()):
                future = clients.submit(client_request)
                http_server.run()
                http_server.close()
                self.assertTrue(inference.inference_process.is_alive())
                self.assertEqual(inference.close(), {'requests': 1})
                status, payload = future.result(timeout=3)
            self.assertEqual((status, payload['text'], payload['request_id']), (200, 'slow', 'one'))
            self.assertEqual(inference.inference_process.exitcode, 0)
        finally:
            http_server.close()
            if inference.inference_process.is_alive():
                inference.inference_process.terminate()
                inference.inference_process.join(timeout=2)
            self.stop_reader(inference)
            for channel in (inference.jobs, inference.results):
                channel.close()
                channel.join_thread()

    def test_http_run_and_signal_handler(self):
        server = serve.HttpServer.__new__(serve.HttpServer)
        server.serve_config = make_config()
        server.inference_server = SimpleNamespace(worker_pid=123, _log=Mock())
        server.stop_requested = threading.Event()
        server.serve_forever = Mock()
        server.server_bind = Mock()
        server.server_activate = Mock()
        shutdown_called = threading.Event()
        server.shutdown = Mock(side_effect=shutdown_called.set)
        with patch.object(serve.signal, 'signal') as register, patch('sys.stdout', new=io.StringIO()):
            server.run()
        self.assertEqual([call.args[0] for call in register.call_args_list],
                         [serve.signal.SIGTERM, serve.signal.SIGINT])
        server.serve_forever.assert_called_once_with(poll_interval=0.25)
        for call in register.call_args_list:
            call.args[1](call.args[0], None)
        self.assertTrue(shutdown_called.wait(timeout=1))
        server.shutdown.assert_called_once_with()

    def test_worker_empty_queue_completion_and_request_error(self):
        jobs, results = queue.Queue(), queue.Queue()
        worker = serve.InferenceWorker(jobs, results, make_config())
        worker.report_status = Mock()  # This test exercises input, not the loaded model's heartbeat.
        with patch.dict(sys.modules, {'p02_serving_runtime':SimpleNamespace(RecognitionRequest=SimpleNamespace)}):
            self.assertIsNone(worker.pull(block=False))
        self.assertFalse(worker.closed)
        jobs.put(dict(request_id='one', crop_type='table', image_bytes=b'image', submitted_monotonic_s=12.0))
        # The worker only constructs this record; keep this test independent of Torch.
        with patch.dict(sys.modules, {'p02_serving_runtime':SimpleNamespace(RecognitionRequest=SimpleNamespace)}):
            request = worker.pull(block=False)
        self.assertEqual(request.crop, b'image')
        self.assertEqual(request.submitted_at, 12.0)
        self.assertEqual(request.prompt, 'Table Recognition:')
        worker.emit_error('one', ValueError('bad image'))
        message = results.get_nowait()
        self.assertEqual((message['kind'], message['request_id'], message['ok']), ('result', 'one', False))
        self.assertEqual(message['error'], 'ValueError: bad image')
        self.assertEqual(worker.jobs_in_progress, {})
        jobs.put(None)
        with patch.dict(sys.modules, {'p02_serving_runtime':SimpleNamespace(RecognitionRequest=SimpleNamespace)}):
            self.assertIsNone(worker.pull(block=True))
            self.assertTrue(worker.closed)
            self.assertIsNone(worker.pull(block=False))

    def test_inference_entrypoint_reports_worker_failure(self):
        jobs, results = queue.Queue(), queue.Queue()
        config = make_config()
        with patch.object(serve, 'InferenceWorker') as worker_class:
            worker_class.return_value.run.side_effect = RuntimeError('load failed')
            serve.run_inference_process(jobs, results, config)
        worker_class.assert_called_once_with(jobs, results, config)
        message = results.get_nowait()
        self.assertEqual(message['kind'], 'startup_error')
        self.assertEqual(message['error'], 'RuntimeError: load failed')

    def test_defaults_and_independent_timeouts(self):
        config = serve.ServeConfig(Path('/model'), Path('/cache'), Path('/logs'))
        self.assertEqual(config.request_timeout_s,60)
        self.assertEqual(config.max_in_flight_requests,64)
        self.assertFalse(hasattr(config,'startup_timeout_s'))
        self.assertEqual(config.shutdown_timeout_s,900)
        self.assertNotIn('__post_init__', serve.ServeConfig.__dict__)
        self.assertFalse(hasattr(serve, 'CropOCR'))


    def test_crop_types_prompts_and_formatting(self):
        # Load the actual string-only formatting functions without importing Torch.
        path=Path(serve.__file__).with_name('p03_crop_processing.py')
        tree=ast.parse(path.read_text())
        names={'normalize_math_delimiters','convert_otsl_to_html','_parse_otsl_rows'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names
               or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_OTSL_TOKEN' for t in n.targets)]
        ns={'html':html,'re':re}
        exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),ns)
        formatting=SimpleNamespace(**{name:ns[name] for name in names})
        for crop_type,prompt in {'table':'Table Recognition:','text':'OCR:',
                                 'formula':'Formula Recognition:'}.items():
            with self.subTest(crop_type=crop_type):
                handler=self.make_handler(f'/v1/ocr?crop_type={crop_type}')
                handler.server.inference_server.recognize.return_value=dict(ok=True,payload={})
                handler.do_POST()
                self.assertEqual(handler._json.call_args.args[0],200)
                jobs,results=queue.Queue(),queue.Queue()
                worker=serve.InferenceWorker(jobs,results,make_config())
                worker.report_status=Mock()
                jobs.put(dict(request_id='crop',crop_type=crop_type,image_bytes=b'image',submitted_monotonic_s=0))
                with patch.dict(sys.modules,{'p02_serving_runtime':SimpleNamespace(RecognitionRequest=SimpleNamespace)}):
                    self.assertEqual(worker.pull(block=False).prompt,prompt)
                raw=r'<fcel>\(x\)<nl>'
                recognition=SimpleNamespace(request_id='crop',text=raw,token_ids=[10,2])
                with patch.dict(sys.modules,{'p03_crop_processing':formatting}), \
                     patch.object(serve,'asdict',side_effect=vars):
                    worker.emit_result(recognition)
                payload=results.get_nowait()['payload']
                normalized=ns['normalize_math_delimiters'](raw)
                expected=ns['convert_otsl_to_html'](normalized) if crop_type=='table' else normalized
                self.assertEqual(payload['text'],expected)
                self.assertEqual(payload['raw_text'],raw)
                self.assertEqual(payload['token_ids'],[10,2])
                self.assertEqual(payload['crop_type'],crop_type)


class AsyncInferenceServerTests(unittest.IsolatedAsyncioTestCase):
    setUp = ServeLifecycleTests.setUp
    make_connection = ServeLifecycleTests.make_connection
    start = ServeLifecycleTests.start
    stop_reader = staticmethod(ServeLifecycleTests.stop_reader)

    def start_connection(self, **kwargs):
        connection=self.make_connection(**kwargs)
        self.start(connection)
        return connection

    async def wait_for_release(self, connection):
        async def wait():
            while connection.pending_requests:
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(),1)

    async def test_concurrent_individual_calls_return_their_own_results(self):
        connection=self.start_connection()
        tasks=[asyncio.create_task(connection.recognize_async(b'image',crop_type=kind,request_id='same-public-id'))
               for kind in ('table','text','formula')]
        await asyncio.sleep(0)
        jobs=[connection.jobs.get_nowait() for _ in tasks]
        self.assertEqual(len({job['request_id'] for job in jobs}),3)
        for job in reversed(jobs):
            connection.results.put(dict(kind='result',request_id=job['request_id'],ok=True,
                                        payload={'text':job['crop_type'],'raw_text':job['crop_type']}))
        replies=await asyncio.gather(*tasks)
        self.assertEqual([r['text'] for r in replies],['table','text','formula'])
        self.assertTrue(all(r['request_id']=='same-public-id' for r in replies))
        self.assertTrue(all('http_wall_s' not in r for r in replies))
        self.assertEqual(connection.pending_requests,{})

    async def test_timeout_retains_capacity_until_late_result(self):
        connection=self.start_connection(request_timeout_s=.02,max_in_flight_requests=1)
        with self.assertRaises(serve.InferenceTimeout):
            await connection.recognize_async(b'image',crop_type='table')
        job=connection.jobs.get_nowait()
        self.assertTrue(connection.pending_requests[job['request_id']].cancelled())
        with self.assertRaises(serve.InferenceCapacityFull):
            await connection.recognize_async(b'next',crop_type='text')
        connection.results.put(dict(kind='result',request_id=job['request_id'],ok=True,payload={'text':'late'}))
        await self.wait_for_release(connection)
        task=asyncio.create_task(connection.recognize_async(b'next',crop_type='text'))
        await asyncio.sleep(0)
        job=connection.jobs.get_nowait()
        connection.results.put(dict(kind='result',request_id=job['request_id'],ok=True,payload={'text':'next'}))
        self.assertEqual((await task)['text'],'next')

    async def test_cancellation_also_retains_capacity(self):
        connection=self.start_connection(max_in_flight_requests=1)
        task=asyncio.create_task(connection.recognize_async(b'image',crop_type='formula'))
        await asyncio.sleep(0)
        job=connection.jobs.get_nowait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        with self.assertRaises(serve.InferenceCapacityFull):
            connection.recognize('http-request','table',b'image')
        connection.results.put(dict(kind='result',request_id=job['request_id'],ok=True,payload={}))
        await self.wait_for_release(connection)

    async def test_invalid_inputs_do_not_consume_capacity(self):
        connection=self.start_connection(max_image_bytes=4)
        for content,kind,error in ((b'a','unknown',ValueError),(b'','table',ValueError),
                                  (b'12345','text',ValueError),('path.png','table',TypeError)):
            with self.subTest(content=content,kind=kind), self.assertRaises(error):
                await connection.recognize_async(content,crop_type=kind)
        self.assertEqual(connection.pending_requests,{})

    async def test_worker_failure_unblocks_async_requests(self):
        connection=self.start_connection()
        task=asyncio.create_task(connection.recognize_async(b'image',crop_type='text'))
        await asyncio.sleep(0)
        connection.results.put(dict(kind='startup_error',error='worker failed',traceback='details'))
        with self.assertRaisesRegex(RuntimeError,'worker failed'): await task
        self.assertEqual(connection.pending_requests,{})
        with self.assertRaisesRegex(RuntimeError,'not accepting'):
            await connection.recognize_async(b'image',crop_type='text')

    async def test_worker_exit_without_error_message_unblocks_requests(self):
        connection=self.start_connection()
        task=asyncio.create_task(connection.recognize_async(b'image',crop_type='text'))
        await asyncio.sleep(0)
        connection.inference_process.alive=False
        with self.assertRaisesRegex(RuntimeError,'worker exited'):
            await asyncio.wait_for(task,1)
        self.assertEqual(connection.pending_requests,{})

    async def test_close_drains_requests_and_rejects_new_ones(self):
        connection=self.start_connection()
        result=asyncio.create_task(connection.recognize_async(b'image',crop_type='formula'))
        await asyncio.sleep(0)
        job=connection.jobs.get_nowait()
        closing=asyncio.create_task(asyncio.to_thread(connection.close))
        self.assertIsNone(await asyncio.to_thread(connection.jobs.get,True,1))
        with self.assertRaisesRegex(RuntimeError,'not accepting'):
            await connection.recognize_async(b'image',crop_type='table')
        self.assertFalse(closing.done())
        connection.results.put(dict(kind='result',request_id=job['request_id'],ok=True,payload={'text':'done'}))
        connection.results.put(dict(kind='service_summary',payload={'requests':1}))
        self.assertEqual((await result)['text'],'done')
        await closing
        self.assertEqual(connection.pending_requests,{})

    async def test_explicit_start_and_close_use_one_real_child_without_http(self):
        with patch.object(serve,'run_inference_process',fake_inference_process):
            connection=serve.InferenceServer(make_config(log_folder=self.log_folder,request_timeout_s=5))
        try:
            with patch.object(serve,'HttpServer',side_effect=AssertionError('offline must not open HTTP')):
                try:
                    connection.start()
                    result=await connection.recognize_async(b'text',crop_type='text')
                    self.assertEqual(result['text'],'text')
                finally:
                    connection.close()
            self.assertEqual(connection.service_summary,{'requests':1})
            self.assertFalse(connection.is_alive)
        finally:
            if connection.is_alive:
                connection.inference_process.terminate(); connection.inference_process.join(timeout=2)
            self.stop_reader(connection)
            for channel in (connection.jobs,connection.results):
                channel.close(); channel.join_thread()


    # Reuse the fake process/temporary-directory setup above; no NPU is loaded.
    def test_existing_result_logged_without_content_and_flushed(self):
        connection=self.make_connection(metrics_level='detailed')
        payload=dict(request_id='client-can-change-this', crop_type='text', worker_wall_s=.5,
                     generated_tokens_including_eos=4, stop_reason='eos', text='PRIVATE OCR',
                     raw_text='PRIVATE OCR', token_ids=[8,9,10,2], timing_s={'detokenize':.001})
        connection._log('request_accepted',('internal-id','text',1))
        connection._log('request_finished',dict(request_id='internal-id',ok=True,payload=payload))
        payload['request_id']='changed-by-caller'
        with patch('sys.stdout',new=io.StringIO()) as console:
            connection.log_writer.start()
            connection._close_logging()
        text=(self.log_folder/'events.jsonl').read_text()
        self.assertEqual(text,console.getvalue())
        self.assertNotIn('PRIVATE OCR',text)
        self.assertNotIn('token_ids',text)
        records=[json.loads(line) for line in text.splitlines()]
        self.assertEqual(records[-1]['request_id'],'internal-id')
        self.assertEqual(records[-1]['generated_tokens_including_eos'],4)
        self.assertEqual(records[-1]['timing_s'],{'detokenize':.001})

    def test_heartbeat_uses_elapsed_time_and_counts_eos(self):
        connection=self.make_connection(metrics_level='basic')
        for observed,tokens in ((100,0),(118,36)):
            connection._log('heartbeat',dict(kind='heartbeat',observed_monotonic_s=observed,
                output_tokens_including_eos=tokens,unfinished_requests=0))
        with patch('sys.stdout',new=io.StringIO()):
            connection.log_writer.start(); connection._close_logging()
        records=[json.loads(line) for line in (self.log_folder/'events.jsonl').read_text().splitlines()]
        self.assertIsNone(records[0]['output_tokens_per_s'])
        self.assertEqual(records[1]['interval_s'],18)
        self.assertEqual(records[1]['output_tokens_per_s'],2)
        self.assertNotIn('p95_latency_s',records[1])

    def test_full_logging_queue_warns_without_blocking(self):
        connection=self.make_connection()
        connection.log_queue=queue.Queue(maxsize=1)
        connection._log('test',{'number':1})
        connection._log('test',{'number':2})
        self.assertTrue(connection.logs_dropped.is_set())
        with patch('sys.stdout',new=io.StringIO()), patch('sys.stderr',new=io.StringIO()) as warning:
            connection.log_writer.start(); connection._close_logging()
        self.assertIn('records were dropped',warning.getvalue())

    def test_file_failure_does_not_stop_console(self):
        self.log_folder.write_text('not a directory')
        connection=self.make_connection()
        connection._log('test',{'number':1})
        with patch('sys.stdout',new=io.StringIO()) as console, patch('sys.stderr',new=io.StringIO()) as warning:
            connection.log_writer.start(); connection._close_logging()
        self.assertIn('"event":"test"',console.getvalue())
        self.assertIn('file logging failed',warning.getvalue())

    def test_idle_worker_keeps_reporting_until_shutdown(self):
        jobs,results=queue.Queue(),queue.Queue()
        worker=serve.InferenceWorker(jobs,results,make_config())
        worker.recognizer=SimpleNamespace(output_tokens=0,batch_size=8,
            decode_arena=SimpleNamespace(num_active=0),crops_awaiting_prefill=[],ready_queue=[])
        timer=threading.Timer(.08,lambda:jobs.put(None))
        with patch.object(serve,'HEARTBEAT_SECONDS',.02), \
             patch.dict(sys.modules,{'p02_serving_runtime':SimpleNamespace(RecognitionRequest=SimpleNamespace)}):
            timer.start()
            self.assertIsNone(worker.pull(block=True))
        timer.join()
        self.assertTrue(worker.closed)
        self.assertGreaterEqual(results.qsize(),2)
        self.assertEqual(results.get()['output_tokens_including_eos'],0)

    def test_shutdown_without_pending_requests_is_not_a_failure(self):
        connection=self.make_connection()
        connection._fail_pending_requests('service stopped')
        self.assertTrue(connection.log_queue.empty())

    def test_failed_pending_requests_are_logged_individually(self):
        connection=self.make_connection()
        connection.pending_requests={name:serve.Future() for name in ('one','two')}
        replies=list(connection.pending_requests.values())
        connection._fail_pending_requests('worker exited')
        events=[connection.log_queue.get_nowait() for _ in range(2)]
        self.assertEqual({data['request_id'] for _,event,data in events if event=='request_failed'}, {'one','two'})
        self.assertTrue(all(isinstance(reply.exception(),RuntimeError) for reply in replies))


if __name__ == '__main__':
    unittest.main()
