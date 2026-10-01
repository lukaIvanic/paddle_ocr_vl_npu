"""Download only the pinned English MTEB HR snapshot used by the public score."""
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

def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    args = p.parse_args()
    start = time.monotonic()
    snapshot_download(REPO, repo_type='dataset', revision=REVISION,
                      local_dir=args.root, endpoint=args.endpoint, token=False,
                      allow_patterns=list(FILES), max_workers=3)
    for name, expected in FILES.items():
        actual = sha256(args.root / name)
        if actual != expected:
            raise ValueError(f'Hash mismatch: {name}')
        print(json.dumps({'verified': name, 'sha256': actual}), flush=True)
    record = dict(repo=REPO, revision=REVISION, files=FILES,
                  status='verified', elapsed_s=time.monotonic()-start, endpoint=args.endpoint)
    (args.root/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
