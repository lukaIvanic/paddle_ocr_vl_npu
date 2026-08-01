"""Exact, device-independent hashes for recognition inputs.

The accuracy lab uses these fingerprints to distinguish frontend/input drift
from NPU model-execution drift.  Hashing is intentionally opt-in in the E2E
runner because pixel tensors dominate the byte volume.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch
from PIL import Image


def fingerprint_pil_image(image: Image.Image) -> dict[str, Any]:
    """Return the canonical hash used by both layout and E2E accuracy traces."""

    payload = image.tobytes()
    digest = hashlib.sha256(
        f"{image.mode}:{image.width}:{image.height}:".encode() + payload
    ).hexdigest()
    return {
        "mode": image.mode,
        "size": [int(image.width), int(image.height)],
        "nbytes": len(payload),
        "sha256": digest,
    }


def fingerprint_cpu_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Hash one contiguous CPU tensor including its dtype and shape contract."""

    value = tensor.detach()
    if value.device.type != "cpu":
        raise ValueError(
            "recognition-input fingerprints must be recorded before H2D: "
            f"device={value.device}"
        )
    value = value.contiguous()
    array = value.numpy()
    raw = array.tobytes(order="C")
    shape = [int(dimension) for dimension in value.shape]
    dtype = str(value.dtype)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": dtype, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b"\0")
    digest.update(raw)
    return {
        "dtype": dtype,
        "shape": shape,
        "nbytes": len(raw),
        "sha256": digest.hexdigest(),
    }


def fingerprint_recognition_inputs(
    *,
    crop: Image.Image,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Fingerprint the raw crop and every final CPU model input tensor."""

    crop_record = fingerprint_pil_image(crop)
    tensor_records = {
        str(name): fingerprint_cpu_tensor(tensor)
        for name, tensor in sorted(tensors.items())
    }
    combined_payload = {
        "crop_sha256": crop_record["sha256"],
        "tensors": {
            name: record["sha256"]
            for name, record in tensor_records.items()
        },
    }
    combined = hashlib.sha256(
        json.dumps(
            combined_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "crop": crop_record,
        "prepared_inputs_sha256": combined,
        "tensors": tensor_records,
    }
