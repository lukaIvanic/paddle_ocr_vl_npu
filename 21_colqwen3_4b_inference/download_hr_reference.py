"""Download the pinned MTEB HR snapshot, sharing corpus/qrels across languages."""
import argparse
import json
from pathlib import Path
import time
from run_hf_baseline import sha256

REPO = 'vidore/vidore_v3_hr_mteb_format'
REVISION = 'bc7d43d64815ed30f664168c8052106484aba7fd'
FILES = {
    'english-corpus/test-00000-of-00001.parquet': '93b89ac0665e8e1dc3a298bf8f5b4ec7809dec01e2564f40335a8ca1d9e098ad',
    'english-queries/test-00000-of-00001.parquet': '385874527338ca229f2a5634c45513378c8460d123ea2db4758e63c24a3b95be',
    'english-qrels/test-00000-of-00001.parquet': '0d164f3769db821b044ec13a2ed6d321016107f77af7c4f5bdcf5471c46e2c81',
}
# At REVISION every language's corpus and qrels has the same LFS SHA256 as
# English. Download those shared files once; query files are language-specific.
QUERY_SHA256 = {
    'english': FILES['english-queries/test-00000-of-00001.parquet'],
    'french': '275b1c427d44b0695850e1824bbc6eee3c47d1e715dbf531ea8ee079a96e5b3a',
    'german': 'ac5d9dd96f27500c10403236d3c180abe5cb0c1106a46974941298653872974e',
    'italian': 'aaa9112fd18f3deb856ac6d9cf3836e82067ac5951970fc9a7c81ad476e527db',
    'portuguese': '4ee02ddce3a5c4269c4b15e4fc174197cfe8ec03a80e35811cad5793c24818d5',
    'spanish': 'd09dc9171911a02e8d2c761b538fe63ae2d36a4ebf23392a3976d283e00a2b72',
}
LANGUAGES = tuple(QUERY_SHA256)


def files_for_languages(languages):
    files = {k: v for k, v in FILES.items() if '-queries/' not in k}
    files.update({f'{lang}-queries/test-00000-of-00001.parquet': QUERY_SHA256[lang]
                  for lang in languages})
    return files

def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--languages', nargs='+', choices=(*LANGUAGES, 'all'), default=['english'])
    args = p.parse_args()
    languages = LANGUAGES if args.languages == ['all'] else tuple(args.languages)
    if 'all' in languages or len(set(languages)) != len(languages):
        p.error('Use all alone, or distinct language names')
    files = files_for_languages(languages)
    start = time.monotonic()
    snapshot_download(REPO, repo_type='dataset', revision=REVISION,
                      local_dir=args.root, endpoint=args.endpoint, token=False,
                      allow_patterns=list(files), max_workers=3)
    for name, expected in files.items():
        actual = sha256(args.root / name)
        if actual != expected:
            raise ValueError(f'Hash mismatch: {name}')
        print(json.dumps({'verified': name, 'sha256': actual}), flush=True)
    record = dict(repo=REPO, revision=REVISION, files=files, languages=languages,
                  status='verified', elapsed_s=time.monotonic()-start, endpoint=args.endpoint)
    (args.root/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
