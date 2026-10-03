"""Prepare a separate vLLM view; never alter the original verified Eos bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from download_eos import RELEASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.bundle.resolve()
    digest = hashlib.sha256((root / "MODEL_MANIFEST.json").read_bytes()).hexdigest()
    if digest not in {r[2] for r in RELEASES.values()}:
        raise ValueError("Unexpected Decision2 revision")
    sys.path.insert(0, str(root))
    from decision2 import verify_bundle
    verify_bundle(root)
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root / "backbone/config.json").read_text())
    config.update(architectures=["Decision2EosForPooling"], decision2_bundle=str(root))
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files = {f.name: f for f in (root / 'backbone').glob('*.safetensors')}
    index = root / 'backbone/model.safetensors.index.json'
    if index.exists():
        files[index.name] = index
    files.update({"tokenizer.json": root / "tokenizer.json",
                  "tokenizer_config.json": root / "tokenizer_config.json"})
    if not any(n.endswith('.safetensors') for n in files):
        raise FileNotFoundError('No backbone weights')
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        (a.output / name).symlink_to(source)
    print(json.dumps({"event": "serving_model_prepared", "bundle": str(root),
                      "view": str(a.output), "manifest_sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
