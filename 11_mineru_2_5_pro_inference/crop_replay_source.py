"""Selective crop inference, with every other block seeded from baseline raw output.

Only experiment manifests produced by prepare_crop_cap_replay are accepted.
The ordinary page postprocessor still sees every block in its original order.
"""
import json
from pathlib import Path

from generation_trace import image_fingerprint
from saved_layout_source import SavedLayoutPageSource
from saved_paddle_crops import sha256, SKIP_TYPES
from streaming_pipeline import MinerUPageSource


class CropReplayPageSource(SavedLayoutPageSource):
    def __init__(self, client, pages, *, manifest_path, **kwargs):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_hash = sha256(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema") != "mineru_selective_crop_replay_v1":
            raise ValueError("unsupported crop replay manifest")
        self.saved = {p["image_name"]: p for p in manifest["pages"]}
        if len(self.saved) != len(manifest["pages"]):
            raise ValueError("duplicate replay pages")
        self.reference_run = manifest["reference_run"]
        MinerUPageSource.__init__(self, client, pages, **kwargs)

    def _saved_job(self, name):
        from PIL import Image
        from mineru_vl_utils.structs import ContentBlock
        blocks, images, prompts, params, indices = [], [], [], [], []
        for index, record in enumerate(self.saved[name]["blocks"]):
            kind = record["type"]
            if record["index"] != index or record["recognize"] != (kind not in SKIP_TYPES):
                raise ValueError("replay order or recognition policy mismatch")
            content = record["baseline_raw_text"] if record["recognize"] else None
            blocks.append(ContentBlock(type=kind, bbox=record["bbox"], angle=None,
                                       content=None if record["rerun"] else content))
            if record["recognize"] and not record["rerun"]:
                blocks[-1].scored = None
            if not record["rerun"]:
                continue
            if not record["recognize"]:
                raise ValueError("cannot rerun an excluded image/chart block")
            path = (self.manifest_path.parent / record["crop"]).resolve()
            if not path.is_relative_to(self.manifest_path.parent) or sha256(path) != record["crop_sha256"]:
                raise ValueError("replay crop file/hash mismatch")
            with Image.open(path) as image:
                crop = image.convert("RGB")
            if list(crop.size) != record["crop_size"] or image_fingerprint(crop) != record["crop_pixel_sha256"]:
                raise ValueError("replay raw crop pixels differ from live baseline")
            image = self.client.helper.resize_by_need(crop)
            if image_fingerprint(image) != record["recognition_image_sha256"]:
                raise ValueError("replay helper-resized crop differs from live baseline")
            images.append(image)
            prompts.append(self.client.prompts.get(kind) or self.client.prompts["[default]"])
            params.append(self.client.sampling_params.get(kind) or self.client.sampling_params.get("[default]"))
            indices.append(index)
        return blocks, (images, prompts, params, indices)

    def metadata(self):
        return {**MinerUPageSource.metadata(self), "layout_source": "frozen_live_layout_selective_crop_replay",
                "reference_run": self.reference_run, "replay_manifest_sha256": self.manifest_hash,
                "timing_scope": "selected crops plus page reconstruction; NOT full E2E throughput",
                "requests_admitted_by_phase": dict(self.phase_admitted),
                "requests_completed_by_phase": dict(self.phase_completed)}
