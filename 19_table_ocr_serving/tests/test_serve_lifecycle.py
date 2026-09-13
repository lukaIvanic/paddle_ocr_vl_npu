"""CPU-only checks of HTTP routing and HTTP/inference communication; no model is loaded."""

import io
import json
import queue
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
                  log_folder=Path('/unused/logs'), request_timeout_s=2.0)
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
            self.assertEqual(len(connection.result_queues_by_request_id), 4)
            for job in reversed(jobs):
                self.assertIn('submitted_monotonic_s', job)
                connection.results.put(dict(kind='result', request_id=job['request_id'],
                                            ok=True, payload={'text': job['request_id']}))
            for request_id, future in futures.items():
                self.assertEqual(future.result(timeout=1)['payload']['text'], request_id)
        self.assertEqual(connection.result_queues_by_request_id, {})

    def test_startup_error_and_timeout(self):
        for message in (dict(kind='startup_error', error='model failed', traceback='failure details'), None):
            with self.subTest(message=message):
                connection = self.make_connection(request_timeout_s=0.02)
                connection.inference_process.startup_message = message
                with patch('sys.stderr', new=io.StringIO()), self.assertRaises(RuntimeError):
                    self.start(connection)
                if message is not None:
                    self.assertEqual(connection.startup_error, message)

    def test_request_timeout_does_not_cancel_job_and_late_result_is_ignored(self):
        connection = self.make_connection(request_timeout_s=0.05)
        self.start(connection)
        with self.assertRaises(serve.InferenceTimeout):
            connection.recognize('expired', 'table', b'image')
        self.assertEqual(connection.jobs.get_nowait()['request_id'], 'expired')
        self.assertEqual(connection.result_queues_by_request_id, {})
        connection.results.put(dict(kind='result', request_id='expired', ok=True, payload={}))
        # A later valid request still receives its own result, not the expired one.
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(connection.recognize, 'next', 'table', b'image')
            self.assertEqual(connection.jobs.get(timeout=1)['request_id'], 'next')
            connection.results.put(dict(kind='result', request_id='next', ok=True, payload={'text': 'next'}))
            self.assertEqual(future.result(timeout=1)['payload']['text'], 'next')

    def test_full_queue_removes_request_registration(self):
        connection = self.make_connection()
        with patch.object(connection.jobs, 'put', side_effect=queue.Full), self.assertRaises(serve.InferenceQueueFull):
            connection.recognize('full', 'table', b'image')
        self.assertEqual(connection.result_queues_by_request_id, {})

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
        connection = self.make_connection(request_timeout_s=0.02)
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
        for error, expected in ((serve.InferenceQueueFull(), 503), (serve.InferenceTimeout(), 504), (ValueError('bad'), 500)):
            handler = self.make_handler('/v1/ocr?crop_type=table')
            handler.server.inference_server.recognize.side_effect = error
            handler.do_POST()
            self.assertEqual(handler._json.call_args.args[0], expected)
        for path, body, expected in (('/v1/drain', b'', 404),
                                     ('/v1/ocr?crop_type=text', b'image', 400),
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
        server.inference_server = SimpleNamespace(worker_pid=123)
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
        self.assertIsNone(worker.pull(block=False))
        self.assertFalse(worker.closed)
        jobs.put(dict(request_id='one', crop_type='table', image_bytes=b'image', submitted_monotonic_s=12.0))
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


if __name__ == '__main__':
    unittest.main()
