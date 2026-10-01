#!/usr/bin/env python3
"""Download pinned public ViDoRe v3 snapshots, verify bytes and inspect Parquet."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from run_hf_baseline import sha256


def parquet_inventory(directory):
    import pyarrow.parquet as pq

    def rows(component, columns):
        files = sorted((directory / component).glob('*.parquet'))
        if not files:
            raise ValueError(f'Missing component {component}')
        return [row for f in files for row in pq.read_table(f, columns=columns).to_pylist()]

    corpus = rows('corpus', ['corpus_id'])
    queries = rows('queries', ['query_id', 'language'])
    qrels = rows('qrels', ['query_id', 'corpus_id', 'score'])
    corpus_ids = {str(x['corpus_id']) for x in corpus}
    query_ids = {str(x['query_id']) for x in queries}
    if len(corpus_ids) != len(corpus) or len(query_ids) != len(queries):
        raise ValueError('Duplicate corpus/query IDs')
    if not all(str(x['corpus_id']) in corpus_ids and str(x['query_id']) in query_ids for x in qrels):
        raise ValueError('Dangling qrels references')
    return {'pages': len(corpus), 'queries': len(queries), 'qrels': len(qrels),
            'query_languages': dict(Counter(x['language'] for x in queries)),
            'parquet': {str(f.relative_to(directory)): {
                'rows': pq.ParquetFile(f).metadata.num_rows,
                'columns': pq.ParquetFile(f).schema_arrow.names}
                for f in sorted(directory.glob('*/*.parquet'))}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--endpoint', default='https://huggingface.co')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--only', nargs='+', help='Optional dataset suffixes, e.g. hr physics')
    args = p.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    from huggingface_hub import HfApi, snapshot_download
    lock = json.loads(Path(__file__).with_name('vidore_v3_revisions.json').read_text())
    if args.only:
        wanted = {'vidore/vidore_v3_' + name for name in args.only}
        if not wanted <= lock.keys():
            p.error('Unknown dataset suffix')
        lock = {k: v for k, v in lock.items() if k in wanted}
    args.root.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint, token=False)
    begin = time.monotonic()
    for repo, revision in lock.items():
        target = args.root / repo.split('/')[-1]
        print(json.dumps({'phase': 'download_start', 'repo': repo, 'revision': revision}), flush=True)
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError(f'Revision mismatch: {repo}')
        snapshot_download(repo, repo_type='dataset', revision=revision,
                          local_dir=target, endpoint=args.endpoint,
                          token=False, max_workers=args.workers)
        manifest = {'repo': repo, 'revision': revision, 'endpoint': args.endpoint, 'files': {}}
        for entry in info.siblings:
            path = target / entry.rfilename
            if not path.is_file() or path.stat().st_size != entry.size:
                raise ValueError(f'Missing or wrong size: {path}')
            digest = sha256(path)
            if entry.lfs and digest != entry.lfs.sha256:
                raise ValueError(f'LFS SHA256 mismatch: {path}')
            if not entry.lfs:
                data = path.read_bytes()
                blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                if blob != entry.blob_id:
                    raise ValueError(f'Git blob mismatch: {path}')
            manifest['files'][entry.rfilename] = {'bytes': entry.size, 'sha256': digest}
        manifest['inventory'] = parquet_inventory(target)
        manifest['status'] = 'verified'
        (args.root / (repo.split('/')[-1] + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'phase': 'dataset_verified', 'repo': repo,
                          'inventory': manifest['inventory'], 'elapsed_s': time.monotonic() - begin}), flush=True)
    print('VIDORE_V3_DOWNLOAD_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
