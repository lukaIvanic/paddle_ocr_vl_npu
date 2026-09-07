"""Recognition-only frontend for the unchanged production MinerU stream."""
import json
from pathlib import Path

from saved_paddle_crops import sha256, SKIP_TYPES
from streaming_pipeline import MinerUPageSource, PageState


class SavedLayoutPageSource(MinerUPageSource):
    def __init__(self, client, pages, *, manifest_path, **kwargs):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_hash = sha256(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest["schema_version"] != 1 or manifest["saved_ocr_content_used"] is not False:
            raise ValueError("unsupported saved crop manifest")
        self.saved = {p["image_name"]: p for p in manifest["pages"]}
        if len(self.saved) != len(manifest["pages"]):
            raise ValueError("duplicate saved pages")
        super().__init__(client, pages, **kwargs)

    def _fill_pages(self):
        while not self.input_exhausted and len(self.pages) < self.page_window:
            try:
                name, _loader = next(self.pages_iter)
            except StopIteration:
                self.input_exhausted = True
                break
            if name in self.seen_pages or name not in self.saved:
                raise ValueError(f"duplicate or missing saved page: {name}")
            self.seen_pages.add(name)
            self.pages[name] = PageState(name)
            self.front_jobs.append(("recognition", name, self.frontend.submit(self._saved_job, name)))
            self.max_pages = max(self.max_pages, len(self.pages))

    def _saved_job(self, name):
        from PIL import Image
        from mineru_vl_utils.structs import ContentBlock
        blocks, images, prompts, params, indices = [], [], [], [], []
        for index, record in enumerate(self.saved[name]["blocks"]):
            if index != record["index"]:
                raise ValueError("saved block order corrupted")
            kind = record["type"]
            blocks.append(ContentBlock(type=kind, bbox=record["bbox"], angle=None, content=None))
            if record["recognize"] != (kind not in SKIP_TYPES):
                raise ValueError("image-analysis policy mismatch")
            if not record["recognize"]:
                continue
            path = (self.manifest_path.parent / record["crop"]).resolve()
            if not path.is_relative_to(self.manifest_path.parent) or sha256(path) != record["crop_sha256"]:
                raise ValueError(f"crop path/hash mismatch: {path}")
            with Image.open(path) as image:
                if list(image.size) != record["crop_size"]:
                    raise ValueError(f"crop dimensions mismatch: {path}")
                crop = image.convert("RGB")
            images.append(self.client.helper.resize_by_need(crop))
            prompts.append(self.client.prompts.get(kind) or self.client.prompts["[default]"])
            params.append(self.client.sampling_params.get(kind) or self.client.sampling_params.get("[default]"))
            indices.append(index)
        return blocks, (images, prompts, params, indices)

    def metadata(self):
        return {**super().metadata(), "layout_source": "PP-DocLayoutV3_saved_final_regions",
                "crop_manifest_sha256": self.manifest_hash,
                "requests_admitted_by_phase": dict(self.phase_admitted),
                "requests_completed_by_phase": dict(self.phase_completed)}
