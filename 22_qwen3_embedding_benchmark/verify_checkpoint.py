"""Verify cached inference bytes against the original published-result revision."""
import hashlib
import json
from pathlib import Path
import sys
from protocol import MODEL_REVISION

EXPECTED = {
    "model.safetensors": ("sha256", "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    "tokenizer.json": ("sha256", "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    "config.json": ("git_blob_sha1", "cef2749ee93607b8f9a58ec72f4f6bfaf874e71d"),
    "tokenizer_config.json": ("git_blob_sha1", "7345216a0785dc7086e8c245b2a9d3896ce2b756"),
}


def main():
    root = Path(sys.argv[1])
    files = {}
    for name, (kind, expected) in EXPECTED.items():
        path = root / name
        digest = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
        if kind == "git_blob_sha1":
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Checkpoint mismatch {name}: {actual} != {expected}")
        files[name] = {"hash_type": kind, "hash": actual, "bytes": path.stat().st_size}
    print(json.dumps({"verified_reference_revision": MODEL_REVISION, "files": files}, indent=2))


if __name__ == "__main__":
    main()
