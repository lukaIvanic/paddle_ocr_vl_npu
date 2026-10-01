#!/usr/bin/env python3
"""Export two unchanged ViDoRe page images and linked English queries for smoke."""
import argparse
import io
import json
from pathlib import Path

from run_hf_baseline import sha256


def main():
    import pyarrow.parquet as pq
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index, domain in enumerate(['hr', 'computer_science']):
        name = 'vidore_v3_' + domain
        directory = args.dataset_root / name
        manifest = json.loads((args.dataset_root / (name + '.manifest.json')).read_text())
        if manifest['status'] != 'verified':
            raise ValueError(f'Unverified dataset: {name}')
        queries = [r for f in sorted((directory / 'queries').glob('*.parquet'))
                   for r in pq.read_table(f).to_pylist() if r['language'] == 'english']
        query = sorted(queries, key=lambda r: str(r['query_id']))[0]
        qrels = [r for f in sorted((directory / 'qrels').glob('*.parquet'))
                 for r in pq.read_table(f, columns=['query_id', 'corpus_id', 'score']).to_pylist()
                 if str(r['query_id']) == str(query['query_id']) and int(r['score']) > 0]
        target = sorted(qrels, key=lambda r: (-int(r['score']), str(r['corpus_id'])))[0]
        found = None
        for f in sorted((directory / 'corpus').glob('*.parquet')):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=['corpus_id', 'image']):
                for row in batch.to_pylist():
                    if str(row['corpus_id']) == str(target['corpus_id']):
                        found = row['image']['bytes']
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f'Missing embedded image for {target}')
        with Image.open(io.BytesIO(found)) as image:
            size, image_format = image.size, image.format.lower()
            image.verify()
        path = args.output_dir / f'page_{index:02d}.{image_format}'
        path.write_bytes(found)
        samples.append({'dataset': manifest['repo'], 'revision': manifest['revision'],
                        'query_id': query['query_id'], 'query': query['query'],
                        'corpus_id': target['corpus_id'], 'relevance': target['score'],
                        'image': str(path.resolve()), 'size': list(size), 'sha256': sha256(path)})
    record = {'scope': 'two-page input smoke, NOT a benchmark score', 'samples': samples}
    (args.output_dir / 'samples.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
