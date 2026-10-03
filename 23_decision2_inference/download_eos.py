"""Download the complete pinned release; preserve its exact manifest inventory."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

REPO = "vllm-sr/Decision-2.0-Eos-0.8B"
REVISION = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = f"{args.endpoint.rstrip('/')}/{REPO}/resolve/{REVISION}/"
    manifest_bytes = urllib.request.urlopen(base + "MODEL_MANIFEST.json", timeout=60).read()
    manifest = json.loads(manifest_bytes)
    (root / "MODEL_MANIFEST.json").write_bytes(manifest_bytes)
    state = {"bytes": 0, "files_done": 0}
    lock = threading.Lock()
    stop = threading.Event()
    start = time.monotonic()

    def heartbeat():
        while not stop.wait(5):
            elapsed = time.monotonic() - start
            print(json.dumps({"event": "download_progress", **state,
                              "elapsed_s": elapsed,
                              "average_MB_s": state["bytes"] / 1e6 / elapsed}), flush=True)

    def fetch(entry):
        name, expected = entry
        dest = root / name
        if not dest.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe manifest path: {name}")
        if dest.is_file() and digest(dest) == expected:
            with lock:
                state["files_done"] += 1
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(base + name, timeout=120) as src, partial.open("wb") as out:
                    while block := src.read(8 << 20):
                        out.write(block)
                        with lock:
                            state["bytes"] += len(block)
                if digest(partial) != expected:
                    raise ValueError(f"SHA256 mismatch: {name}")
                partial.replace(dest)
                with lock:
                    state["files_done"] += 1
                print(json.dumps({"event": "file_verified", "file": name,
                                  "bytes": dest.stat().st_size}), flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, manifest["files_sha256"].items()))
    finally:
        stop.set()
        t.join()
    print(json.dumps({"event": "download_complete", "repo": REPO,
                      "revision": REVISION, "manifest_sha256": digest(root / "MODEL_MANIFEST.json"),
                      "elapsed_s": time.monotonic() - start, **state}), flush=True)


if __name__ == "__main__":
    main()
