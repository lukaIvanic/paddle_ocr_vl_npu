"""UniRec's production vision, packed prefill, fixed-arena decode and converter."""
from types import SimpleNamespace
from .base import Adapter


class UniRecAdapter(Adapter):
    def __init__(self, runner, vision, decoder, emit, converter):
        from persistent_ready_queue import PersistentReadyQueue
        super().__init__(decoder.batch_size, emit)
        self.runner, self.vision, self.decoder = runner, vision, decoder
        self.converter = converter
        self.source = PersistentReadyQueue(maxsize=self.ready_capacity)
        self.source.register_upstream()
        self.steps = decoder.iter_run(self.source, on_complete=self.complete)
        next(self.steps)
        self.start_cpu_preparation(self.prepare_cpu, self.capacity, "hybrid-unirec-cpu")

    def prepare_cpu(self, request):
        import numpy as np
        from PIL import Image
        image = request.crop.convert("RGB")
        size = self.runner.processor.get_processed_size(*image.size)
        pixels = np.ascontiguousarray(np.asarray(image.resize(size, Image.Resampling.BICUBIC)))
        return pixels, image.size

    @property
    def ready_count(self):
        return self.source.qsize()

    def set_upstream(self, pending, *, closed=False):
        if pending and not self.source.upstream_pending:
            self.source.register_upstream()
        elif not pending and self.source.upstream_pending:
            self.source.complete_upstream()
        if closed and not pending:
            self.source.close()

    def prefill(self):
        import torch
        from continuous_unirec import ContinuousReadyItem, ContinuousWorkerPrefilledItem
        from run_opendoc_batched_unirec import iter_greedy_text_packs
        from vision_full_batch import PreprocessedVisionInput
        requests, prepared = self.take_prepared_requests()
        inputs = []
        crops = []
        for i, (request, (pixels, image_size)) in enumerate(zip(requests, prepared, strict=True)):
            inputs.append(PreprocessedVisionInput(i, pixels, image_size, request.request_id))
            crops.append(SimpleNamespace(image_size=image_size, source_index=i, request=request))
        with torch.inference_mode():
            spans = []
            def begin(stage):
                pair = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                spans.append((stage, pair))
                pair[0].record()
                return pair[1]
            end = begin("vision_encode_envelope")
            encoded = self.vision.encode(inputs)
            end.record()
            for packed, group in iter_greedy_text_packs(crops, runner=self.runner):
                if not packed:
                    raise RuntimeError("UniRec crop exceeds the production packed text-prefill contract")
                values = [(encoded[c.source_index].hidden_states, encoded[c.source_index].prep) for c in group]
                end = begin("text_prefill_envelope")
                items = self.runner.prefill_encoder_hidden_states_packed_for_cohort(values, decode_ready=False)
                end.record()
                for crop, item in zip(group, items, strict=True):
                    self.prefill_tokens.update({
                        "text_real_source": item.text_prefill_real_source_tokens,
                        "text_physical_source": item.text_prefill_physical_source_tokens,
                    })
                    cache = item.kv_cache
                    actual_length = cache.actual_cross_attention_length
                    kv = torch.stack(tuple(
                        tensor[:, :, :actual_length, :]
                        for tensor in (*cache.cross_key_cache, *cache.cross_value_cache)
                    ), dim=0).contiguous()
                    prefilled = ContinuousWorkerPrefilledItem(
                        packed_cross_kv=kv, prep=item.prep, prefill_s=item.prefill_s,
                        actual_cross_attention_length=cache.actual_cross_attention_length,
                        text_prefill_execution=item.text_prefill_execution,
                        text_prefill_real_source_tokens=item.text_prefill_real_source_tokens,
                        text_prefill_physical_source_tokens=item.text_prefill_physical_source_tokens,
                    )
                    def release(value=prefilled):
                        value.packed_cross_kv = None
                    self.source.put(ContinuousReadyItem(crop.request.request_id, crop.request, prefilled, release))
        torch.npu.synchronize()
        for stage, (start, end) in spans:
            self.prefill_device_s[stage] += start.elapsed_time(end) / 1000

    def complete(self, completed):
        from run_opendoc_batched_unirec import _postprocess_recognizer_text
        from routing import TASKS
        task = TASKS[completed.payload.prompt]
        value = completed.result
        content = _postprocess_recognizer_text(self.converter.markdown_converter, value["text"], task)
        content = self.converter.truncate_repetitive_content(content)
        if ("\\(" in content and "\\)" in content) or ("\\[" in content and "\\]" in content):
            content = content.replace("$", "").replace("\\(", " $ ").replace("\\)", " $ ").replace("\\[", " $$ ").replace("\\]", " $$ ")
        if task == "table":
            content = self.converter.convert_otsl_to_html(content)
        ids = value["generated_ids"]
        stop = "eos" if ids[-1] == self.runner.config.eos_token_id else "length"
        self.emit(completed.request_id, content, ids, stop, "unirec")

    def close(self):
        self.cpu.close()
        self.steps.close()
