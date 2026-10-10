"""Explicit candidate flags; run_page_pipeline defaults are unchanged."""
PRODUCTION_PREFILL_OPTIONS = [
    '--processor-text-max-pixels', '401408',
    '--local-vision-buckets', '384,512,768,896,1024,1280,1536,1792,1920,2048,2560,3072',
    '--local-text-buckets', '128,256,384,512,576,832,1024',
    '--local-text-pack-target', '384',
    '--local-text-prefill-schedule', 'window',
]
