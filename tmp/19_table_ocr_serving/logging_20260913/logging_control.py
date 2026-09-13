"""Benchmark-only no-output control. Not an additional product serving option."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '19_table_ocr_serving'))
import p01_serve


if __name__ == '__main__':
    # Only suppress delivery to the writer. The same runtime, counters,
    # heartbeats, HTTP path, synthetic setup and real warmup still execute.
    p01_serve.InferenceServer._log = lambda self, *args, **kwargs: None
    p01_serve.main()
