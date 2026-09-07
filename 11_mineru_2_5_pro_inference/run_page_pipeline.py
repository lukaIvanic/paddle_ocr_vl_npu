"""Live PP-DocLayoutV3 + MinerU page-to-Markdown CLI with production defaults.

Accepts ordinary page images or an explicit dataset manifest. Native MinerU
layout remains available with --layout-backend mineru. No saved crops required.
"""
import sys

from run_official_transformers_omnidocbench import main, parse_args


PRODUCTION_OPTIONS = [
    "--backend", "local-continuous-client", "--layout-backend", "pp-doclayout-v3",
    "--batch-size", "32", "--page-batch-size", "32", "--streaming-pages",
    "--streaming-page-window", "32", "--warmup-pages", "0", "--no-resume", "--fail-fast",
    "--processor-min-pixels", "25088", "--processor-max-pixels", "1103872",
    "--local-compiled-cache-length", "4096", "--local-decode-attention", "increfa",
    "--local-decode-increfa-length-mode", "pse_sentinel_310p",
    "--local-decode-weight-format", "decode_nz", "--local-decode-rotary-impl", "npu_apply",
    "--local-prepare-prefetch-depth", "64", "--local-prefill-metrics",
    "--local-text-backend", "torchair-packed", "--local-text-max-members", "32",
    "--local-vision-attention", "prompt_flash_attention", "--local-vision-backend", "torchair",
    "--local-vision-pack-target", "768", "--local-vision-lookahead", "32",
    "--local-text-torchair-cache-dir", ".runtime_cache/11_mineru_2_5_pro_inference/text_prefill_packed_fp16",
    "--local-vision-torchair-cache-dir", ".runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16_9511b2e",
    "--local-torchair-cache-dir", ".runtime_cache/11_mineru_2_5_pro_inference/production_increfa_real_nz_compile",
    "--token-trace", "--hash-model-files",
]


def pipeline_args(argv=None):
    args = parse_args(PRODUCTION_OPTIONS + (sys.argv[1:] if argv is None else argv))
    if args.input_images is None and args.dataset_json is None:
        raise ValueError("provide --input-images or --dataset-json explicitly")
    return args


if __name__ == "__main__":
    main(pipeline_args())
