"""Lossless generation records, written after token materialization, not per step.

No torch dependency. The page frontend supplies identities before model calls;
the adapter supplies the original CPU prompt IDs and unfiltered output IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def image_fingerprint(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size}".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def request_identity(page: str, phase: str, block_index: int | None = None,
                     block: Any = None) -> dict[str, Any]:
    result = {"request_id": f"{page}:{phase}" + (
        f":{block_index}" if block_index is not None else ""
    ), "page": page, "phase": phase, "block_index": block_index}
    if block is not None:
        result.update(block_type=block["type"], bbox=list(block["bbox"]),
                      angle=block.get("angle", 0))
    return result


class GenerationTrace:
    def __init__(self, path: Path, *, eos_token_id: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8", buffering=1)
        self.eos_token_id = int(eos_token_id)
        self.contexts: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.prepared_count = 0
        self.written = 0
        self.seen: set[str] = set()
        self.pages: list[str] = []

    def begin_batch(self, images, prompts) -> None:
        if len(self.contexts) != len(images) or len(prompts) != len(images):
            raise ValueError("trace identity/image/prompt count mismatch")
        self.records = [dict(context, schema_version=1, chat_prompt=prompt,
                             image_sha256=image_fingerprint(image))
                        for context, image, prompt in zip(self.contexts, images, prompts)]
        self.prepared_count = 0

    def prepared(self, prompt_ids: list[int], max_new_tokens: int) -> None:
        self.records[self.prepared_count].update(
            prompt_token_ids=prompt_ids, max_new_tokens=int(max_new_tokens))
        self.prepared_count += 1

    def write(self, record: dict[str, Any], ids: list[int], text: str) -> None:
        identity = record["request_id"]
        if identity in self.seen:
            raise ValueError(f"duplicate generation trace identity: {identity}")
        self.seen.add(identity)
        if not ids:
            raise ValueError(f"empty generation: {identity}")
        stop_reason = "eos" if ids[-1] == self.eos_token_id else "length"
        if stop_reason == "length" and len(ids) != record["max_new_tokens"]:
            raise ValueError(f"unexplained generation stop: {identity}")
        payload = dict(record, generated_token_ids=ids, raw_text=text,
                       stop_reason=stop_reason)
        self.handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.written += 1

    def finish_batch(self, rows, texts) -> None:
        if not len(self.records) == self.prepared_count == len(rows) == len(texts):
            raise ValueError("incomplete generation trace batch")
        for record, ids, text in zip(self.records, rows, texts):
            self.write(record, ids, text)
        self.records = []
        self.contexts = []

    def close(self) -> None:
        self.handle.close()


def install_stepping_trace(client, trace: GenerationTrace) -> None:
    """Observe the existing helper calls without changing their work order."""
    helper = client.helper
    layout = helper.batch_prepare_for_layout
    extract = helper.batch_prepare_for_extract

    def prepare_layout(executor, images):
        result = layout(executor, images)
        if len(result) != len(trace.pages):
            raise ValueError("trace page count mismatch")
        trace.contexts = [request_identity(page, "layout") for page in trace.pages]
        return result

    def prepare_extract(executor, images, blocks_list, *args, **kwargs):
        result = extract(executor, images, blocks_list, *args, **kwargs)
        trace.contexts = [
            request_identity(page, "recognition", index, blocks[index])
            for page, blocks, prepared in zip(trace.pages, blocks_list, result)
            for index in prepared[3]
        ]
        return result

    helper.batch_prepare_for_layout = prepare_layout
    helper.batch_prepare_for_extract = prepare_extract
    client.client.generation_trace = trace
