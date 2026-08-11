"""Persistent CPU artifact for isolated UniRec prefill and decode replay."""

from __future__ import annotations

import json
import os
import time
import zlib
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORMAT_NAME = "unirec_cross_kv_v1"
ALIGNMENT = 64


class _SharedPageLease:
    """Attach, unlink, and retain a worker-owned shared page arena."""

    def __init__(self, name: str) -> None:
        self.storage = SharedMemory(name=name)
        self.storage.unlink()

    def array(self, descriptor: dict[str, Any]) -> np.ndarray:
        return np.ndarray(
            tuple(int(value) for value in descriptor["shape"]),
            dtype=np.dtype(descriptor["dtype"]),
            buffer=self.storage.buf,
            offset=int(descriptor["offset"]),
        )

    def close(self) -> None:
        self.storage.close()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(row), ensure_ascii=False))
            handle.write("\n")
    os.replace(partial, path)


class CrossKvArtifactWriter:
    """Stream page-scoped shared-memory cross K/V into one mmapable file."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.output_dir / "cross_kv.bin"
        self.partial_data_path = self.output_dir / "cross_kv.bin.partial"
        self.manifest_path = self.output_dir / "crops.jsonl"
        self.pages_path = self.output_dir / "pages.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        conflicts = [
            path
            for path in (
                self.data_path,
                self.partial_data_path,
                self.manifest_path,
                self.pages_path,
                self.summary_path,
            )
            if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                "prefill artifact output already exists: "
                + ", ".join(str(path) for path in conflicts)
            )
        self._data = self.partial_data_path.open("wb")
        self.crop_rows: list[dict[str, Any]] = []
        self.page_rows: list[dict[str, Any]] = []
        self.page_count = 0
        self.crop_count = 0
        self.rejected_crop_count = 0
        self.real_source_tokens = 0
        self.physical_source_tokens = 0
        self.cross_kv_bytes = 0
        self.shared_payload_bytes = 0
        self.attach_s = 0.0
        self.cross_kv_write_s = 0.0
        self.manifest_write_s = 0.0
        self.closed = False

    def _align(self) -> int:
        position = self._data.tell()
        padding = (-position) % ALIGNMENT
        if padding:
            self._data.write(b"\0" * padding)
        return position + padding

    def add_page(self, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("prefill artifact writer is closed")
        shared = payload.get("shared_memory")
        if not isinstance(shared, dict):
            raise RuntimeError("worker page has no shared-memory payload")
        attach_started = time.perf_counter()
        lease = _SharedPageLease(str(shared["name"]))
        self.attach_s += time.perf_counter() - attach_started
        page_crop_rows: list[dict[str, Any]] = []
        array: np.ndarray | None = None
        try:
            for crop in payload["crops"]:
                descriptor = crop.get("worker_cross_kv_descriptor")
                metadata = crop.get("worker_prefill_metadata")
                if not isinstance(descriptor, dict) or not isinstance(metadata, dict):
                    raise RuntimeError("worker crop has no cross-KV export payload")
                array = lease.array(descriptor)
                if array.ndim != 5 or array.dtype != np.float16:
                    raise RuntimeError(
                        f"unexpected cross-KV tensor: shape={array.shape} dtype={array.dtype}"
                    )
                if not array.flags.c_contiguous:
                    raise RuntimeError("worker cross-KV tensor is not contiguous")
                source_length = int(array.shape[-2])
                if source_length != int(metadata["actual_cross_attention_length"]):
                    raise RuntimeError(
                        "cross-KV source length mismatch: "
                        f"tensor={source_length} metadata="
                        f"{metadata['actual_cross_attention_length']}"
                    )
                write_started = time.perf_counter()
                offset = self._align()
                byte_view = memoryview(array).cast("B")
                checksum = zlib.crc32(byte_view) & 0xFFFFFFFF
                written = self._data.write(byte_view)
                del byte_view
                self.cross_kv_write_s += time.perf_counter() - write_started
                if written != int(array.nbytes):
                    raise OSError(
                        f"short cross-KV write: {written} != {array.nbytes}"
                    )
                page_index = int(payload["page_index"])
                crop_index = int(crop["crop_index"])
                row = {
                    "format": FORMAT_NAME,
                    "request_id": f"page_{page_index:06d}_crop_{crop_index:04d}",
                    "page_index": page_index,
                    "page_image": payload["image_path"],
                    "crop_index": crop_index,
                    "label": crop["label"],
                    "figure_token_map": crop.get("figure_token_map", {}),
                    "cross_kv": {
                        "file": self.data_path.name,
                        "offset": offset,
                        "nbytes": int(array.nbytes),
                        "shape": list(array.shape),
                        "dtype": array.dtype.str,
                        "crc32": f"{checksum:08x}",
                        "source_length": source_length,
                    },
                    "prefill": metadata,
                }
                self.crop_rows.append(row)
                page_crop_rows.append(
                    {
                        "request_id": row["request_id"],
                        "crop_index": crop_index,
                        "label": crop["label"],
                        "figure_token_map": crop.get("figure_token_map", {}),
                    }
                )
                self.crop_count += 1
                self.real_source_tokens += int(
                    metadata["text_prefill_real_source_tokens"]
                )
                self.physical_source_tokens += int(
                    metadata["text_prefill_physical_source_tokens"]
                )
                self.cross_kv_bytes += int(array.nbytes)
                array = None
        finally:
            array = None
            lease.close()

        self.page_rows.append(
            {
                "format": FORMAT_NAME,
                "page_index": int(payload["page_index"]),
                "image_path": payload["image_path"],
                "width": int(payload["width"]),
                "height": int(payload["height"]),
                "layout_results": payload["layout_results"],
                "blocks": payload["blocks"],
                "vlm_block_ids": payload["vlm_block_ids"],
                "drop_figures_set": payload["drop_figures_set"],
                "frontend_timing_s": payload["frontend_timing_s"],
                "worker_prefill_stats": payload.get("worker_prefill_stats"),
                "cross_capacity_rejected_crops": int(
                    payload.get("cross_capacity_rejected_crops", 0)
                ),
                "crops": page_crop_rows,
            }
        )
        self.page_count += 1
        self.rejected_crop_count += int(
            payload.get("cross_capacity_rejected_crops", 0)
        )
        self.shared_payload_bytes += int(shared["nbytes"])

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("prefill artifact writer is already closed")
        manifest_started = time.perf_counter()
        self._data.flush()
        self._data.close()
        os.replace(self.partial_data_path, self.data_path)
        _write_jsonl(self.manifest_path, self.crop_rows)
        _write_jsonl(self.pages_path, self.page_rows)
        self.manifest_write_s += time.perf_counter() - manifest_started
        file_bytes = self.data_path.stat().st_size
        result = {
            "format": FORMAT_NAME,
            **_json_value(summary),
            "artifact": {
                "directory": str(self.output_dir),
                "cross_kv_file": self.data_path.name,
                "crop_manifest": self.manifest_path.name,
                "page_manifest": self.pages_path.name,
                "page_count": self.page_count,
                "crop_count": self.crop_count,
                "rejected_crop_count": self.rejected_crop_count,
                "real_source_tokens": self.real_source_tokens,
                "physical_source_tokens": self.physical_source_tokens,
                "cross_kv_payload_bytes": self.cross_kv_bytes,
                "cross_kv_file_bytes": file_bytes,
                "shared_payload_bytes": self.shared_payload_bytes,
            },
            "coordinator_timing_s": {
                "shared_memory_attach": self.attach_s,
                "cross_kv_crc_and_write": self.cross_kv_write_s,
                "artifact_finalize": self.manifest_write_s,
            },
        }
        partial = self.summary_path.with_suffix(".json.partial")
        partial.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, self.summary_path)
        self.closed = True
        return result

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._data.close()


class CrossKvDiscardSink:
    """Validate and release page-scoped cross K/V without retaining its bytes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.json"
        if self.summary_path.exists():
            raise FileExistsError(
                f"prefill discard summary already exists: {self.summary_path}"
            )
        self.page_count = 0
        self.crop_count = 0
        self.rejected_crop_count = 0
        self.real_source_tokens = 0
        self.physical_source_tokens = 0
        self.cross_kv_bytes = 0
        self.shared_payload_bytes = 0
        self.attach_s = 0.0
        self.descriptor_validation_s = 0.0
        self.finalize_s = 0.0
        self.closed = False

    def add_page(self, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("prefill discard sink is closed")
        shared = payload.get("shared_memory")
        if not isinstance(shared, dict):
            raise RuntimeError("worker page has no shared-memory payload")
        shared_nbytes = int(shared["nbytes"])
        attach_started = time.perf_counter()
        lease = _SharedPageLease(str(shared["name"]))
        self.attach_s += time.perf_counter() - attach_started
        try:
            validation_started = time.perf_counter()
            for crop in payload["crops"]:
                descriptor = crop.get("worker_cross_kv_descriptor")
                metadata = crop.get("worker_prefill_metadata")
                if not isinstance(descriptor, dict) or not isinstance(metadata, dict):
                    raise RuntimeError("worker crop has no cross-KV discard payload")
                shape = tuple(int(value) for value in descriptor["shape"])
                dtype = np.dtype(descriptor["dtype"])
                if len(shape) != 5 or dtype != np.dtype(np.float16):
                    raise RuntimeError(
                        f"unexpected cross-KV descriptor: shape={shape} dtype={dtype}"
                    )
                offset = int(descriptor["offset"])
                nbytes = int(descriptor["nbytes"])
                expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if nbytes != expected_nbytes:
                    raise RuntimeError(
                        f"cross-KV descriptor byte mismatch: {nbytes} != {expected_nbytes}"
                    )
                if offset < 0 or offset + nbytes > shared_nbytes:
                    raise RuntimeError(
                        "cross-KV descriptor exceeds its shared page arena: "
                        f"offset={offset} nbytes={nbytes} arena={shared_nbytes}"
                    )
                source_length = int(shape[-2])
                if source_length != int(metadata["actual_cross_attention_length"]):
                    raise RuntimeError(
                        "cross-KV source length mismatch: "
                        f"descriptor={source_length} metadata="
                        f"{metadata['actual_cross_attention_length']}"
                    )
                self.crop_count += 1
                self.real_source_tokens += int(
                    metadata["text_prefill_real_source_tokens"]
                )
                self.physical_source_tokens += int(
                    metadata["text_prefill_physical_source_tokens"]
                )
                self.cross_kv_bytes += nbytes
            self.descriptor_validation_s += time.perf_counter() - validation_started
        finally:
            lease.close()
        self.page_count += 1
        self.rejected_crop_count += int(
            payload.get("cross_capacity_rejected_crops", 0)
        )
        self.shared_payload_bytes += shared_nbytes

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("prefill discard sink is already closed")
        finalize_started = time.perf_counter()
        result = {
            "format": FORMAT_NAME,
            **_json_value(summary),
            "artifact": {
                "storage_mode": "discard",
                "directory": str(self.output_dir),
                "cross_kv_file": None,
                "crop_manifest": None,
                "page_manifest": None,
                "page_count": self.page_count,
                "crop_count": self.crop_count,
                "rejected_crop_count": self.rejected_crop_count,
                "real_source_tokens": self.real_source_tokens,
                "physical_source_tokens": self.physical_source_tokens,
                "cross_kv_payload_bytes": self.cross_kv_bytes,
                "cross_kv_file_bytes": 0,
                "shared_payload_bytes": self.shared_payload_bytes,
            },
            "coordinator_timing_s": {
                "shared_memory_attach": self.attach_s,
                "cross_kv_descriptor_validation": self.descriptor_validation_s,
                "cross_kv_crc_and_write": 0.0,
                "artifact_finalize": 0.0,
            },
        }
        partial = self.summary_path.with_suffix(".json.partial")
        partial.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, self.summary_path)
        self.finalize_s = time.perf_counter() - finalize_started
        result["coordinator_timing_s"]["artifact_finalize"] = self.finalize_s
        self.closed = True
        return result

    def abort(self) -> None:
        self.closed = True


def read_crop_array(
    artifact_dir: Path,
    row: dict[str, Any],
    *,
    verify_crc: bool = True,
) -> np.memmap:
    spec = row["cross_kv"]
    array = np.memmap(
        artifact_dir / spec["file"],
        mode="r",
        dtype=np.dtype(spec["dtype"]),
        offset=int(spec["offset"]),
        shape=tuple(int(value) for value in spec["shape"]),
        order="C",
    )
    if int(array.nbytes) != int(spec["nbytes"]):
        raise RuntimeError("cross-KV manifest byte count mismatch")
    if verify_crc:
        checksum = zlib.crc32(memoryview(array).cast("B")) & 0xFFFFFFFF
        if f"{checksum:08x}" != str(spec["crc32"]):
            raise RuntimeError(
                f"cross-KV checksum mismatch for {row.get('request_id')}"
            )
    return array


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
