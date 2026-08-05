#!/usr/bin/env python3
"""Run original OpenOCR/Transformers UniRec inference on the six smoke crops."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast, __version__ as transformers_version


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_IMAGES = [
    PROJECT_ROOT / "crops/crop_01_text_block_en.png",
    PROJECT_ROOT / "crops/crop_02_equation_matrix.png",
    PROJECT_ROOT / "crops/crop_03_code_block.png",
    PROJECT_ROOT / "crops/crop_04_handwritten_title_zh.png",
    PROJECT_ROOT / "crops/crop_05_table_rwkv_dims.png",
    PROJECT_ROOT / "crops/crop_06_chart_cubic_spline.png",
]
EXPECTED_TRANSFORMERS_VERSION = "5.2.0"


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(inner) for inner in value]
    return value


def infer_openocr_device(device: str) -> str:
    if device.startswith("cuda"):
        return "gpu"
    if device.startswith("npu"):
        return "npu"
    return device


def parse_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.rsplit(":", 1)[1])
    except ValueError:
        return 0


def configure_npu_jit_compile(mode: str, device: str) -> None:
    if mode == "default" or not device.startswith("npu"):
        return
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_compile_mode(jit_compile=(mode == "on"))
    except Exception as exc:
        raise RuntimeError(f"failed to set NPU jit_compile={mode}: {exc}") from exc


def build_tokenizer(model_path: Path) -> PreTrainedTokenizerFast:
    tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token=tokenizer_config.get("bos_token", "<s>"),
        eos_token=tokenizer_config.get("eos_token", "</s>"),
        pad_token=tokenizer_config.get("pad_token", "<pad>"),
        unk_token=tokenizer_config.get("unk_token", "<unk>"),
        clean_up_tokenization_spaces=tokenizer_config.get("clean_up_tokenization_spaces", False),
        model_max_length=tokenizer_config.get("model_max_length", 2048),
    )


def maybe_make_transformers52_metadata_compat_dir(model_path: Path, enabled: bool) -> tuple[Path, dict[str, Any]]:
    meta: dict[str, Any] = {"enabled": enabled, "used": False, "reason": None, "compat_model_path": None}
    if not enabled:
        return model_path, meta
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    needs_config_patch = config.get("model_type") == "m2m_100"
    needs_tokenizer_patch = tokenizer_config.get("tokenizer_class") == "PreTrainedTokenizer"
    if not needs_config_patch and not needs_tokenizer_patch:
        meta["reason"] = "model metadata already compatible"
        return model_path, meta

    digest = hashlib.sha256(str(model_path).encode("utf-8")).hexdigest()[:12]
    compat_root = PROJECT_ROOT / ".runtime_cache/12_unirec_0_1b_inference/tf52_model_metadata_compat" / digest
    compat_root.mkdir(parents=True, exist_ok=True)
    patched_file_names = {"config.json", "tokenizer_config.json"}
    for child in model_path.iterdir():
        if child.name in patched_file_names:
            continue
        target = compat_root / child.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(child)
    for name in patched_file_names:
        target = compat_root / name
        if target.exists() or target.is_symlink():
            target.unlink()

    patched_config = dict(config)
    if needs_config_patch:
        patched_config["model_type"] = ""
    patched_tokenizer_config = dict(tokenizer_config)
    if needs_tokenizer_patch:
        patched_tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (compat_root / "config.json").write_text(json.dumps(patched_config, indent=2) + "\n", encoding="utf-8")
    (compat_root / "tokenizer_config.json").write_text(
        json.dumps(patched_tokenizer_config, indent=2) + "\n",
        encoding="utf-8",
    )
    meta.update(
        {
            "used": True,
            "reason": "UniRec HF metadata is not AutoTokenizer-compatible with transformers 5.2.0",
            "compat_model_path": str(compat_root),
            "config_model_type_original": config.get("model_type"),
            "tokenizer_class_original": tokenizer_config.get("tokenizer_class"),
        }
    )
    return compat_root.resolve(), meta


def apply_text_patch(path: Path, before: str, after: str, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return "already_patched"
    if before not in text:
        raise RuntimeError(f"expected OpenOCR compatibility patch target not found in {path}: {marker}")
    path.write_text(text.replace(before, after), encoding="utf-8")
    return "patched"


def patch_openocr_for_transformers52(openocr_root: Path, mode: str) -> dict[str, Any]:
    module_path = openocr_root / "openrec/modeling/unirec_modeling/modeling_unirec.py"
    meta: dict[str, Any] = {"mode": mode, "module_path": str(module_path), "actions": []}
    if mode == "skip":
        return meta
    text = module_path.read_text(encoding="utf-8")
    decoder_ok = "M2M100Decoder(config, self.shared)" not in text or "self.decoder = M2M100Decoder(config)" in text
    cache_ok = "get_seq_length" in text
    if mode == "check":
        if not decoder_ok or not cache_ok:
            raise RuntimeError("OpenOCR checkout needs --openocr-transformers52-compat patch")
        meta["actions"].append("already_compatible")
        return meta

    decoder_before = "        self.decoder = M2M100Decoder(config, self.shared)\n"
    decoder_after = (
        "        try:\n"
        "            self.decoder = M2M100Decoder(config, self.shared)\n"
        "        except TypeError:\n"
        "            self.decoder = M2M100Decoder(config)\n"
        "            self.decoder.embed_tokens = self.shared\n"
    )
    cache_before = (
        "        if past_key_values is not None:\n"
        "            past_length = past_key_values[0][0].shape[2]\n"
    )
    cache_after = (
        "        if past_key_values is not None:\n"
        "            if hasattr(past_key_values, \"get_seq_length\"):\n"
        "                past_length = past_key_values.get_seq_length()\n"
        "            else:\n"
        "                past_length = past_key_values[0][0].shape[2]\n"
    )
    meta["actions"].append(
        {"decoder_constructor": apply_text_patch(module_path, decoder_before, decoder_after, "M2M100Decoder(config)")}
    )
    meta["actions"].append(
        {"generation_cache_api": apply_text_patch(module_path, cache_before, cache_after, "get_seq_length")}
    )
    return meta


def patch_openocr_for_npu_device(openocr_root: Path, requested_device: str, mode: str) -> dict[str, Any]:
    module_path = openocr_root / "tools/infer_rec.py"
    meta: dict[str, Any] = {
        "mode": mode,
        "requested_device": requested_device,
        "module_path": str(module_path),
        "actions": [],
    }
    if not requested_device.startswith("npu"):
        meta["actions"].append("not_requested")
        return meta
    meta["actions"].append(
        {
            "source_patch": "not_used",
            "reason": (
                "The NPU path monkeypatches tools.infer_rec.set_device in this Python process before "
                "OpenRecognizer construction, then hard-validates recognizer.device and first_parameter_device."
            ),
        }
    )
    return meta


def synchronize_device_obj(device: Any) -> None:
    device_text = str(device)
    if device_text.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
    elif device_text.startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        torch.npu.synchronize()


def build_openocr_config(
    *,
    openocr_root: Path,
    config_path: Path,
    runtime_model_path: Path,
    device: str,
    dtype: str,
    max_length: int,
) -> dict[str, Any]:
    if str(openocr_root) not in sys.path:
        sys.path.insert(0, str(openocr_root))
    from tools.engine.config import Config

    cfg = Config(str(config_path)).cfg
    cfg["Global"]["pretrained_model"] = str(runtime_model_path / "model.pth")
    cfg["Global"]["vlm_ocr_config"] = str(runtime_model_path)
    cfg["Global"]["device"] = infer_openocr_device(device)
    cfg["Global"]["use_transformers"] = True
    cfg["Global"]["use_amp"] = dtype == "float16"
    cfg["Global"]["max_text_length"] = int(max_length)
    cfg["PostProcess"]["tokenizer_path"] = str(runtime_model_path)
    return cfg


def load_openocr_recognizer(
    *,
    openocr_root: Path,
    config_path: Path,
    runtime_model_path: Path,
    device: str,
    dtype: str,
    max_length: int,
) -> tuple[Any, dict[str, Any]]:
    cfg = build_openocr_config(
        openocr_root=openocr_root,
        config_path=config_path,
        runtime_model_path=runtime_model_path,
        device=device,
        dtype=dtype,
        max_length=max_length,
    )
    from tools import infer_rec as openocr_infer_rec

    if device.startswith("cuda"):
        use_gpu = "true"
    elif device.startswith("npu"):
        # Do not trust OpenOCR's CUDA/GPU-oriented device parser for NPU. The
        # NPU lane already observed a source patch reporting "already_patched"
        # while OpenOCR still logged CPU fallback. Force the runtime set_device
        # function in this process and validate the loaded model device below.
        import torch
        import torch_npu  # noqa: F401

        requested_device_index = parse_device_index(device)
        requested_torch_device = torch.device(f"npu:{requested_device_index}")

        def _codex_forced_npu_set_device(_device: str, numId: int = 0) -> torch.device:
            return requested_torch_device

        openocr_infer_rec.set_device = _codex_forced_npu_set_device
        use_gpu = "false"
    else:
        use_gpu = "false"
    load_start = time.perf_counter()
    with pushd(openocr_root):
        recognizer = openocr_infer_rec.OpenRecognizer(cfg, use_gpu=use_gpu, numId=parse_device_index(device))
    recognizer._codex_openocr_root = str(openocr_root)
    forced_runtime_device = None
    forced_runtime_device_after_load = None
    if device.startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        forced_runtime_device = torch.device(device)
        if str(recognizer.device) != str(forced_runtime_device):
            recognizer.device = forced_runtime_device
            recognizer.model.to(recognizer.device)
            recognizer.model.eval()
            forced_runtime_device_after_load = str(forced_runtime_device)
        parameter_device = next(recognizer.model.parameters()).device
        if parameter_device.type != "npu" or str(recognizer.device) != str(forced_runtime_device):
            raise RuntimeError(
                "OpenOCR original baseline failed to load on NPU. "
                f"recognizer.device={recognizer.device}, first_parameter_device={parameter_device}, "
                f"expected={forced_runtime_device}. Stop instead of reporting CPU fallback as NPU."
            )
        recognizer.cfg["Global"]["device"] = "npu"
    synchronize_device_obj(recognizer.device)
    model_load_s = time.perf_counter() - load_start
    return recognizer, {
        "model_load_s": model_load_s,
        "loaded_once_for_all_images": True,
        "openocr_constructor_use_gpu": use_gpu,
        "runtime_device": str(recognizer.device),
        "expected_runtime_device": str(forced_runtime_device) if forced_runtime_device is not None else str(recognizer.device),
        "forced_runtime_device_after_load": forced_runtime_device_after_load,
        "runtime_set_device_override": bool(device.startswith("npu")),
        "first_parameter_device": str(next(recognizer.model.parameters()).device),
    }


def run_one_image(
    *,
    image_path: Path,
    recognizer: Any,
    max_length: int,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, Any]:
    synchronize_device_obj(recognizer.device)
    start = time.perf_counter()
    with pushd(Path(recognizer._codex_openocr_root)):
        preds = recognizer(img_path=str(image_path), batch_num=1)
    synchronize_device_obj(recognizer.device)
    total_latency_s = time.perf_counter() - start
    pred = preds[0] if preds else {}
    text = str(pred.get("text", ""))
    openocr_inference_s = float(pred["elapse"]) if pred.get("elapse") is not None else None
    token_count = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
    return {
        "image": str(image_path),
        "status": "ok",
        "returncode": 0,
        "text": text,
        "score": pred.get("score"),
        "output_token_count_approx": int(token_count),
        "ttft_s": None,
        "ttft_note": "Original OpenOCR generate call does not expose first-token timing.",
        "openocr_reported_inference_s": openocr_inference_s,
        "openocr_reported_inference_note": "OpenOCR elapse is reported from inside its call; total_latency_s is externally device-synchronized.",
        "total_latency_s": total_latency_s,
        "tokens_per_s_openocr_reported": (
            float(token_count) / openocr_inference_s if openocr_inference_s and token_count > 0 else None
        ),
        "tokens_per_s_total_latency": float(token_count) / total_latency_s if total_latency_s > 0 and token_count > 0 else None,
        "max_length": int(max_length),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--npu-jit-compile", choices=("off", "on", "default"), default="off")
    parser.add_argument("--openocr-transformers52-compat", choices=("check", "patch", "skip"), default="check")
    parser.add_argument("--openocr-npu-device-compat", choices=("check", "patch", "skip"), default="patch")
    parser.add_argument("--no-transformers52-metadata-compat", action="store_true")
    parser.add_argument("--allow-transformers-version-mismatch", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "tmp/12_unirec_0_1b_inference/original_transformers_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if transformers_version != EXPECTED_TRANSFORMERS_VERSION and not args.allow_transformers_version_mismatch:
        raise RuntimeError(f"Expected transformers=={EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}")
    model_path = args.model_path.expanduser().resolve()
    openocr_root = args.openocr_root.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else openocr_root / "configs/rec/unirec/focalsvtr_ardecoder_unirec.yml"
    )
    images = [path.expanduser().resolve() for path in (args.image or DEFAULT_IMAGES)]
    configure_npu_jit_compile(args.npu_jit_compile, args.device)
    openocr_compat = patch_openocr_for_transformers52(openocr_root, args.openocr_transformers52_compat)
    openocr_npu_compat = patch_openocr_for_npu_device(openocr_root, args.device, args.openocr_npu_device_compat)
    runtime_model_path, metadata_compat = maybe_make_transformers52_metadata_compat_dir(
        model_path,
        enabled=not args.no_transformers52_metadata_compat,
    )
    tokenizer = build_tokenizer(runtime_model_path)
    recognizer, load_meta = load_openocr_recognizer(
        openocr_root=openocr_root,
        config_path=config_path,
        runtime_model_path=runtime_model_path,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )
    print(f"Loaded OpenOCR recognizer once in {load_meta['model_load_s']:.3f}s on {load_meta['runtime_device']}", flush=True)

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] original transformers: {image_path.name}", flush=True)
        result = run_one_image(
            image_path=image_path,
            recognizer=recognizer,
            max_length=args.max_length,
            tokenizer=tokenizer,
        )
        print(f"total_latency_s={result['total_latency_s']:.4f}", flush=True)
        print(f"openocr_reported_inference_s={result['openocr_reported_inference_s']}", flush=True)
        print(f"tokens_per_s_total_latency={result['tokens_per_s_total_latency']}", flush=True)
        print("generation:")
        print(result["text"])
        results.append(result)
        if result["returncode"] != 0:
            break

    payload = {
        "experiment": "12_unirec_original_transformers_per_image",
        "status": "ok" if all(item["returncode"] == 0 for item in results) else "error",
        "model_path": str(model_path),
        "runtime_model_path": str(runtime_model_path),
        "openocr_root": str(openocr_root),
        "device": args.device,
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "original_generation_note": (
            "This path intentionally uses OpenOCR's original UniRec generate call. "
            "In the current OpenOCR code it calls model.generate(**inputs) without forwarding Global.max_text_length, "
            "so output length may follow the Transformers generation default instead of --max-length."
        ),
        "transformers_version": transformers_version,
        "transformers52_metadata_compat": metadata_compat,
        "openocr_transformers52_compat": openocr_compat,
        "openocr_npu_device_compat": openocr_npu_compat,
        "openocr_model_load": load_meta,
        "images": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output_json}")
    print(json.dumps(clean_json(payload), indent=2, ensure_ascii=False))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
