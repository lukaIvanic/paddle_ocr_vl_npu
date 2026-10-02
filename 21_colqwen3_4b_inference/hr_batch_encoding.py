"""Observed page-batch loop for the explicit HR batching experiment."""
import hashlib
import io
import time

import torch
from PIL import Image

from batched_prefill import prepare_batched_images
from optimized_prefill import text_args_for_promptfa
from prepared_prefill import prepare_text, finish_embeddings


def encode_page_batches(model, processor, execution, corpus, args, journal, profiler):
    from run_hr_evaluation import image_resize_kwargs
    embeddings, records = [], []
    window = time.perf_counter()
    for start in range(0, len(corpus), args.page_batch_size):
        items = corpus[start:start+args.page_batch_size]
        row = dict(kind='page_batch', id=items[0]['id'], index=start, sections={},
                   ids=[x['id'] for x in items], batch_size=len(items))
        begin = time.perf_counter()
        journal.emit('item_start', kind=row['kind'], id=row['id'], index=start, total=len(corpus))
        with journal.section(row, 'preprocess'):
            images = []
            try:
                for item in items:
                    with Image.open(io.BytesIO(item['image']['bytes'])) as image:
                        images.append(image.convert('RGB'))
                row['image_sizes'] = [list(image.size) for image in images]
                row['image_sha256'] = [hashlib.sha256(x['image']['bytes']).hexdigest() for x in items]
                batch = processor.process_images(images, **image_resize_kwargs(processor, args.max_image_tokens))
            finally:
                for image in images:
                    image.close()
        grids = batch['image_grid_thw']
        vt = int(grids.prod(-1).sum())
        tt = int(batch['attention_mask'].sum())
        counts = (batch['input_ids'] == model.config.image_token_id).sum(-1).tolist()
        row.update(vision_tokens=vt, text_tokens=tt, image_grid_thw=grids.tolist(),
                   merged_image_tokens_per_page=counts,
                   resized_image_wh=[[g[2]*processor.image_processor.patch_size,
                                      g[1]*processor.image_processor.patch_size] for g in grids.tolist()])
        if args.max_image_tokens is not None and max(counts) > args.max_image_tokens:
            raise ValueError('Processor exceeded the requested image token budget')
        with journal.section(row, 'input_transfer', device=True):
            batch = {k:v.to(args.device) for k,v in batch.items()}
        with journal.section(row, 'vision_prepare', vt, device=True):
            prepared = prepare_batched_images(model, batch, execution.patch)
        with journal.section(row, 'vision_transformer', vt, device=True, route='batched_optimized_eager'):
            vision = execution.batch_vision(*prepared.vision_args)
        with journal.section(row, 'text_prepare', tt, device=True):
            text_args = text_args_for_promptfa(prepare_text(model, prepared, vision))
        with journal.section(row, 'text_transformer', tt, device=True, route='batched_optimized_eager'):
            hidden = execution.text(*text_args)
        with journal.section(row, 'retrieval_projection', tt, device=True):
            output = finish_embeddings(model, prepared, hidden)
        with journal.section(row, 'output_materialize_wait', tt):
            output = output.cpu()
        with journal.section(row, 'validate_retain'):
            if not bool(torch.isfinite(output).all()):
                raise ValueError('Nonfinite embeddings')
            norms = output.float().norm(dim=-1)
            active = norms > 0
            if not bool(active.any(-1).all()) or float((norms[active]-1).abs().max()) > .002:
                raise ValueError('Invalid embedding norms')
            row.update(embedding_rows_per_page=output.shape[1],
                       active_embedding_rows_per_page=active.sum(-1).tolist())
            embeddings.extend(output.unbind(0))
        row['usable_embedding_s'] = time.perf_counter()-begin
        row['wall_s'] = time.perf_counter()-begin
        records.append(row)
        journal.complete(row, start+len(items), len(corpus), window)
        profiler.step()
        # Drop transient NPU tensors before preparing the following batch.
        del batch, prepared, vision, text_args, hidden, output
    return embeddings, records, time.perf_counter()-window
