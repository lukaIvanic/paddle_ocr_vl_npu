"""Download the complete pinned release; preserve its exact manifest inventory."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

REPO = "vllm-sr/Decision-2.0-Eos-0.8B"
REVISION = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"
RELEASES = {
    "eos": (REPO, REVISION, "e8b1081be4a76deca5247792a4031c8e19e2b52b95d91775c13d9e257c407101"),
    "nox": ("vllm-sr/Decision-2.0-Nox-4B", "25e8f67d1b486c647222df3aac640d2d5d736bbe",
            "49771ea33a451274687ad2a246ce5fb78f8f42ec904fc91a467477179975bb16"),
}


def open_url(url, timeout, headers=None):
    # The mirror rejects urllib's default User-Agent on its resolve-cache route.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


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
    p.add_argument("--release", choices=list(RELEASES), default="eos")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo, revision, manifest_sha = RELEASES[args.release]
    base = f"{args.endpoint.rstrip('/')}/{repo}/resolve/{revision}/"
    manifest_bytes = open_url(base + "MODEL_MANIFEST.json?download=true", timeout=60).read()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
        raise ValueError("Unexpected release manifest")
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
                url = base + name + "?download=true"
                if name.startswith("backbone/model") and name.endswith(".safetensors"):
                    # Pin both the known release size and final SHA; validate each
                    # range so a proxy returning the full file cannot corrupt it.
                    with open_url(url, 120, {"Range": "bytes=0-0"}) as probe:
                        content_range = probe.headers.get("Content-Range", "")
                        if probe.status != 206 or not content_range.startswith("bytes 0-0/"):
                            raise ValueError("Missing valid range length")
                        total = int(content_range.split("/")[-1])
                    chunk = 32 << 20
                    with partial.open("wb") as out:
                        out.truncate(total)
                        def fetch_range(start):
                            end = min(start + chunk, total) - 1
                            for retry in range(5):
                                try:
                                    # Range in URL prevents intermediaries from
                                    # confusing responses for different ranges.
                                    with open_url(url + f'&range_start={start}', 45, {"Range": f"bytes={start}-{end}"}) as src:
                                        expected_range = f"bytes {start}-{end}/{total}"
                                        if src.status != 206 or src.headers.get("Content-Range") != expected_range:
                                            raise ValueError(f"Invalid range response: {src.headers.get('Content-Range')}")
                                        offset = start
                                        while block := src.read(1 << 20):
                                            if offset + len(block) > end + 1:
                                                raise ValueError("Range response too long")
                                            view = memoryview(block)
                                            while view:
                                                count = os.pwrite(out.fileno(), view, offset)
                                                offset += count
                                                view = view[count:]
                                            with lock:
                                                state["bytes"] += len(block)
                                        if offset != end + 1:
                                            raise ValueError("Incomplete range response")
                                    return
                                except Exception as exc:
                                    print(json.dumps({'event':'range_retry', 'file':name, 'start':start,
                                                      'attempt':retry+1, 'error':repr(exc)}), flush=True)
                                    if retry == 4:
                                        raise
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ranges:
                            list(ranges.map(fetch_range, range(0, total, chunk)))
                else:
                    with open_url(url, timeout=120) as src, partial.open("wb") as out:
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
            except Exception as exc:
                print(json.dumps({'event':'file_retry', 'file':name, 'attempt':attempt+1, 'error':repr(exc)}), flush=True)
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
    print(json.dumps({"event": "download_complete", "repo": repo,
                      "revision": revision, "manifest_sha256": digest(root / "MODEL_MANIFEST.json"),
                      "elapsed_s": time.monotonic() - start, **state}), flush=True)


if __name__ == "__main__":
    main()
