"""Benchmark-only selection of the engine's existing preprocessing options.

The HTTP API, spawned worker, scheduler and model graphs remain unchanged.
Do not add these research switches to the product entrypoint.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '19_table_ocr_serving'))
import serve as product

original_worker = product._worker_main


def worker(jobs, results, config):
    import serving_runtime

    class BenchmarkRecognizer(serving_runtime.ContinuousRecognizer):
        def __init__(self, **kwargs):
            super().__init__(
                **kwargs,
                image_resize_backend=os.environ['PADDLE_BENCH_RESIZE'],
                compact_uint8_preprocess=os.environ['PADDLE_BENCH_UINT8'] == '1',
            )

        def configuration(self):
            result = super().configuration()
            result['preprocessing_benchmark'] = {
                'resize_backend': self.image_resize_backend,
                'compact_uint8': self.compact_uint8_preprocess,
            }
            return result

    serving_runtime.ContinuousRecognizer = BenchmarkRecognizer
    original_worker(jobs, results, config)


product._worker_main = worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bench-resize', choices=('pillow', 'kornia_rs'), required=True)
    parser.add_argument('--bench-uint8', choices=('0', '1'), required=True)
    args, remaining = parser.parse_known_args()
    os.environ['PADDLE_BENCH_RESIZE'] = args.bench_resize
    os.environ['PADDLE_BENCH_UINT8'] = args.bench_uint8
    sys.argv = [sys.argv[0], *remaining]
    product.main()
