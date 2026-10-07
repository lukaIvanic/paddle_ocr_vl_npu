"""Same-shape real-page B1/B2/B4 complete-forward experiment.

Preprocessing and input/output transfers are outside timing. Existing eager
per-page preparation/mergers remain inside; both transformer stacks run true
batches. No image resizing override, mixed-image attention, or repeated rows.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import traceback

import torch
from torch import nn
import torch.nn.functional as F

from bench_prepared_prefill import compare, load_case
from local_modeling_colqwen3 import LocalColQwen3, rotate_half
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
    configure_compiler, prompt_attention, text_args_for_promptfa)
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import StageCompiler, prepare_text, finish_embeddings, gelu_tanh
from profile_warm_forward import Forward, measure, emit
from run_hr_evaluation import read_data


class BatchVision(nn.Module):
    """Same optimized vision arithmetic with an explicit independent page axis."""
    def __init__(self, source):
        super().__init__()
        self.blocks, self.taps = source.blocks, source.taps

    def forward(self, hidden, cos, sin):
        b, s, _ = hidden.shape
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        taps = []
        for index, block in enumerate(self.blocks):
            q, k, v = block.qkv(block.norm(block.norm1, hidden)).reshape(
                b, s, 3, block.heads, -1).permute(2, 0, 1, 3, 4).unbind(0)
            q = (q.float()*cos + rotate_half(q.float())*sin).to(q.dtype)
            k = (k.float()*cos + rotate_half(k.float())*sin).to(k.dtype)
            q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
            out = prompt_attention(q, k, v, block.scale).reshape(b, s, -1)
            hidden = hidden + block.proj(out)
            hidden = hidden + block.fc2(gelu_tanh(block.fc1(block.norm(block.norm2, hidden))))
            if index in self.taps:
                taps.append(hidden)
        return hidden, taps[0], taps[1], taps[2]


class BatchForward:
    def __init__(self, model, rows, patch, vision, text):
        self.model, self.rows, self.patch = model, rows, patch
        self.vision, self.text = vision, text

    def prepare(self):
        prepared = [prepare_linear_patch_inputs(self.model, row, self.patch) for row in self.rows]
        args = tuple(torch.stack([p.vision_args[i] for p in prepared]) for i in range(3))
        return prepared, args

    def text_inputs(self, prepared, visual):
        individual = [text_args_for_promptfa(prepare_text(self.model, p,
            tuple(v[i] for v in visual))) for i, p in enumerate(prepared)]
        return tuple(torch.cat([a[j] for a in individual]) for j in range(7))

    def __call__(self):
        prepared, args = self.prepare()
        visual = self.vision(*args)
        hidden = self.text(*self.text_inputs(prepared, visual))
        # Keep the existing complete retrieval projection and output masking.
        return torch.cat([finish_embeddings(self.model, p, hidden[i:i+1])
                          for i, p in enumerate(prepared)])


def diagnostics(actual, reference):
    a, b = actual.float().cpu(), reference.float().cpu()
    delta = (a-b).abs()
    cos = F.cosine_similarity(a, b, dim=-1).flatten()
    return dict(compare(actual, reference),
        fraction_coordinates_outside_old_tolerance=float((delta > .002+.002*b.abs()).float().mean()),
        abs_error_quantiles={str(q): float(delta.flatten().quantile(q)) for q in (.5,.9,.99,.999,1.)},
        token_cosine_quantiles={str(q): float(cos.quantile(q)) for q in (0.,.01,.5,.99,1.)})


def select_pages(args, processor, anchor):
    from PIL import Image
    corpus, _, _ = read_data(args.dataset_root)
    selected, manifest = [anchor], []
    target = {k: (tuple(v.shape), v.dtype) for k, v in anchor.items()}
    first = corpus[5]
    manifest.append(dict(id=first['id'], sha256=hashlib.sha256(first['image']['bytes']).hexdigest()))
    seen = {manifest[0]['sha256']}
    for item in corpus:
        payload = item['image']['bytes']
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            continue
        with Image.open(io.BytesIO(payload)) as image:
            batch = processor.process_images([image.convert('RGB')])
        signature = {k: (tuple(v.shape), v.dtype) for k, v in batch.items()}
        if signature != target or not torch.equal(batch['image_grid_thw'], anchor['image_grid_thw'].cpu()):
            continue
        selected.append({k: v.to(args.device) for k, v in batch.items()})
        manifest.append(dict(id=item['id'], sha256=digest))
        seen.add(digest)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError('Could not find four distinct pages matching the original page shape')
    # Verify the first manifest entry really identifies the supplied anchor.
    with Image.open(io.BytesIO(first['image']['bytes'])) as image:
        fresh = processor.process_images([image.convert('RGB')])
    if not all(torch.equal(fresh[k].cpu(), v.cpu()) for k, v in anchor.items()):
        raise RuntimeError('Corpus page 5 differs from the supplied anchor')
    return selected, manifest


@torch.inference_mode()
def run(args, result):
    import torch_npu
    from transformers import AutoProcessor
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    result.update(device=torch.npu.get_device_name(), torch=torch.__version__, torch_npu=torch_npu.__version__,
                  options=asdict(Options()), scope=__doc__)
    anchor, hf = load_case(args.anchor, 0, args.device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    rows, result['pages'] = select_pages(args, processor, anchor)
    result['input_shapes_per_page'] = {k: list(v.shape) for k, v in anchor.items()}
    model = LocalColQwen3.from_pretrained(args.model, device=args.device)
    reference = [model(**row).cpu() for row in rows]
    result['original_vs_hf_page5'] = compare(reference[0], hf)
    if not result['original_vs_hf_page5']['passed']:
        raise RuntimeError('Untouched reference differs from its HF anchor')
    patch = LinearPatchEmbed(model.visual.patch_embed).eval()
    vision = OptimizedVisionStage(model, Options()).eval()
    batched_vision = BatchVision(vision).eval()
    text = OptimizedTextStage(model, Options()).eval()
    singles = [Forward(model, row, patch, vision, text) for row in rows]
    optimized = [fn().cpu() for fn in singles]
    result['optimized_vs_original'] = [diagnostics(a,b) for a,b in zip(optimized, reference)]
    result['optimized_vs_hf_page5'] = diagnostics(optimized[0], hf)
    torch.save(dict(reference=reference, optimized=optimized), args.output_dir/'single_outputs.pt')
    result['measurements'] = []

    def save():
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def benchmark(name, functions, expected, batch_size):
        for _ in range(args.warmups):
            for fn in functions:
                fn()
        samples, checks = [], []
        # Alternate through all four real pages in every repetition.
        for _ in range(args.repeats):
            for fn in functions:
                timing, out = measure(fn, 1)
                samples.extend(timing['wall_samples_ms'])
        for fn, target in zip(functions, expected):
            checks.append(compare(fn(), target))
        if not all(c['passed'] for c in checks):
            result['failed_checks'] = checks
            save()
            raise RuntimeError(name+' batch/compile parity differs from same optimized single-page outputs')
        item = dict(name=name, batch_size=batch_size, calls=len(samples),
                    batch_mean_ms=sum(samples)/len(samples),
                    pages_per_second=1000*batch_size*len(samples)/sum(samples),
                    samples_ms=samples, output_checks=checks)
        result['measurements'].append(item)
        save()
        emit('batch_measurement', **item)

    if not args.skip_baselines:
        benchmark('original_manual_eager', [lambda row=row: model(**row) for row in rows], reference, 1)
        benchmark('existing_optimized_eager', singles, optimized, 1)
    compiler = StageCompiler(args.model, args.cache_root, emit)
    configure_compiler(compiler, Options())
    compiler.identity.update(internal_format=True,
        batch_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract='same_shape_real_pages_batched_vision_text_eager_per_page_preparation_v1')
    for size in args.batch_sizes:
        functions = [BatchForward(model, rows[i:i+size], patch, batched_vision, text)
                     for i in range(0, 4, size)]
        targets = [torch.cat(optimized[i:i+size]) for i in range(0, 4, size)]
        # Check independent page results before spending time compiling.
        batch_outputs = [fn().cpu() for fn in functions]
        checks = [diagnostics(out, target) for out,target in zip(batch_outputs,targets)]
        result.setdefault('batched_eager_vs_single', {})[str(size)] = checks
        # Localize batch-size arithmetic changes using identical captured inputs.
        prepared, vargs = functions[0].prepare()
        together = batched_vision(*vargs)
        separate = [vision(*(a[i] for a in vargs)) for i in range(size)]
        vision_checks = [compare(together[j],torch.stack([v[j] for v in separate])) for j in range(4)]
        ta = functions[0].text_inputs(prepared,together)
        same_inputs_single_text = torch.cat([text(*(a[i:i+1].contiguous() for a in ta)) for i in range(size)])
        text_check = diagnostics(text(*ta),same_inputs_single_text)
        independence = None
        if size > 1:
            # Change the other pages' pixels but hold page zero exactly fixed.
            changed_rows = [dict(row) for row in functions[0].rows]
            for row in changed_rows[1:]:
                row['pixel_values'] = torch.zeros_like(row['pixel_values'])
            changed = BatchForward(model,changed_rows,patch,batched_vision,text)().cpu()
            independence = compare(changed[:1],batch_outputs[0][:1])
            if not independence['exact']:
                raise RuntimeError('Changing batch partners changed page zero')
        result.setdefault('batch_diagnostics',{})[str(size)] = dict(
            vision_vs_individual=vision_checks, text_same_inputs_vs_individual=text_check,
            page_zero_independent_of_partner_pixels=independence)
        torch.save(batch_outputs,args.output_dir/f'batch{size}_eager_outputs.pt')
        save()
        emit('batch_diagnostics',batch_size=size,checks=result['batch_diagnostics'][str(size)],
             full_forward_vs_individual=checks)
        if args.diagnostic_only:
            continue
        if not all(c['passed'] for c in checks):
            raise RuntimeError('Batched eager versus single-page numerical check failed')
        benchmark('batched_optimized_eager', functions, targets, size)
        prepared, vargs = functions[0].prepare()
        vcall = compiler.get('batch_vision', batched_vision, vargs)
        visual = vcall(*vargs)
        targs = functions[0].text_inputs(prepared, visual)
        tcall = compiler.get('batch_text', text, targs)
        tcall(*targs)
        compiled = [BatchForward(model, fn.rows, patch, vcall, tcall) for fn in functions]
        benchmark('batched_optimized_torchair', compiled, targets, size)
    result['cache_records'] = compiler.records
    result['status'] = 'completed'
    save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--batch-sizes',type=int,nargs='+',choices=(1,2,4),default=[1,2,4])
    p.add_argument('--skip-baselines',action='store_true')
    p.add_argument('--diagnostic-only',action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(status='started', commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    try:
        run(args, result)
    except Exception:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
